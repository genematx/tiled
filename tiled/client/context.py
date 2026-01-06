import asyncio
import dataclasses
import functools
import getpass
import os
import re
import sys
import time
import urllib.parse
import warnings
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractAsyncContextManager as ContextManager
from contextlib import nullcontext
from pathlib import Path
from typing import Any, List, Literal, TypeVar

from urllib.parse import parse_qs, urlparse

import httpx
import platformdirs
from pydantic import TypeAdapter

from tiled.schemas import About, AboutAuthenticationProvider

from .._version import __version__ as tiled_version
from ..utils import UNSET, DictView, parse_time_string
from .auth import CannotRefreshAuthentication, TiledAuth, build_refresh_request
from .decoders import SUPPORTED_DECODERS
from .transport import AsyncTransport, Transport
from .utils import (
    DEFAULT_TIMEOUT_PARAMS,
    MSGPACK_MIME_TYPE,
    handle_error,
    retry_context,
)

USER_AGENT = f"python-tiled/{tiled_version}"
API_KEY_AUTH_HEADER_PATTERN = re.compile(r"^Apikey (\w+)$")

DEFAULT_POOL_SIZE = 100
T = TypeVar("T")


def raise_if_cannot_prompt():
    if not _can_prompt() and not bool(int(os.environ.get("TILED_FORCE_PROMPT", "0"))):
        raise CannotPrompt(
            """
Tiled has detected that it is running in a 'headless' context where it cannot
prompt the user to provide credentials in the stdin. Options:

- Provide an API key in the environment variable TILED_API_KEY for Tiled to
  use.

- If Tiled has detected wrongly, set the environment variable
  TILED_FORCE_PROMPT=1 to override and force an interactive prompt.

- If you are developing an application that is wraping Tiled,
  obtain tokens using functions tiled.client.context.password_grant
  and/or device_code_grant, and pass them like Context.authenticate(tokens=token).
"""
        )


def identity_provider_input(
    providers: List[AboutAuthenticationProvider],
) -> AboutAuthenticationProvider:
    while True:
        print("Authentication providers:")
        for i, spec in enumerate(providers, start=1):
            print(f"{i} - {spec.provider}")
        raw_choice = input(
            "Choose an authentication provider (or press Enter to cancel): "
        )
        if not raw_choice:
            print("No authentication provider chosen. Failed.")
            break
        try:
            choice = int(raw_choice)
        except TypeError:
            print("Choice must be a number.")
            continue
        try:
            spec = providers[choice - 1]
        except IndexError:
            print(f"Choice must be a number 1 through {len(providers)}.")
            continue
        break
    return spec


def username_input():
    raise_if_cannot_prompt()
    return input("Username: ")


def password_input():
    raise_if_cannot_prompt()
    return getpass.getpass()


class PasswordRejected(RuntimeError):
    pass


def prompt_for_credentials(context, providers: List[AboutAuthenticationProvider]):
    """
    Prompt for credentials or third-party login at an interactive terminal.
    """
    if len(providers) == 1:
        # There is only one choice, so no need to prompt the user.
        spec = providers[0]
    else:
        spec = identity_provider_input(providers)
    auth_endpoint = spec.links["auth_endpoint"]
    provider = spec.provider
    mode = spec.mode
    # Note: "password" is included here for back-compat with older servers;
    # the new name for this mode is "internal".
    if mode == "internal" or mode == "password":
        # Prompt for username, password at terminal.
        username = username_input()
        PASSWORD_ATTEMPTS = 3
        for _attempt in range(PASSWORD_ATTEMPTS):
            password = password_input()
            grantor = password_grant.__wrapped__(
                context, auth_endpoint, provider, username, password
            )
            try:
                tokens = yield from grantor
            except httpx.HTTPStatusError as err:
                if err.response.status_code == httpx.codes.UNAUTHORIZED:
                    print(
                        "Username or password not recognized. Retry, or press Enter to cancel."
                    )
                    grantor.close()
                    continue
                raise
            else:
                if tokens is not None:
                    # Success! We have tokens.
                    break
        else:
            # All attempts failed.
            raise PasswordRejected
    elif mode == "external":
        # Display link and access code, and try to open web browser.
        # Block while polling the server awaiting confirmation of authorization.
        tokens = yield from device_code_grant.__wrapped__(context, auth_endpoint)
    else:
        raise ValueError(f"Server has unknown authentication mechanism {mode!r}")
    confirmation_message = spec.confirmation_message
    if confirmation_message:
        username = tokens["identity"]["id"]
        print(confirmation_message.format(id=username))
    return tokens


@dataclasses.dataclass(frozen=True)
class TiledRequest:
    client: httpx._client.BaseClient
    request: httpx.Request
    auth: httpx.Auth = httpx._client.UseClientDefault()
    raw: bool = False
    lock: ContextManager = nullcontext()


def requestor(generator: bool = False, asynchronous: bool | None = None) -> Callable:
    """A decorator that serves requests on behalf of the decorated generator function.

    The wrapped function is expected to yield tuples of `(client,
    request)`, such that ``client.send(request)`` will produce an
    ``httpx.Response``. The wrapped function can also yield other
    things; with ``generator=True`` the resulting function will be a
    generator that yields anything that does not match the above
    pattern.

    Parameters
    ==========

    """

    def decorator(func: T) -> T:
        @functools.wraps(func)
        def inner(*args, **kwargs):
            # match (asynchronous, generator, hasattr(args[0], "context")):
            if asynchronous is None:
                context = _find_context(args[0])
                if context is None:
                    raise ValueError(
                        "Either decorate a BaseClient method, or call "
                        "requestor() with the *asynchronous* parameters. "
                        f"Got {args[0]}."
                    )
            # Bound client methods
            if asynchronous is None and generator:
                sender = args[0].context.send_requests_generator
            elif asynchronous is None and not generator:
                sender = args[0].context.send_requests
            # Unbound functions
            elif not asynchronous and generator:
                sender = send_requests_generator
            elif asynchronous and generator:
                sender = send_requests_generator_async
            elif asynchronous:
                sender = send_requests_async
            else:
                sender = send_requests
            return sender(func(*args, **kwargs))

        return inner

    return decorator


class Context:
    """
    Wrap an httpx.Client with an optional cache and authentication functionality.
    """

    def __init__(
        self,
        uri,
        *,
        headers=None,
        api_key=None,
        cache=UNSET,
        timeout=None,
        verify=True,
        app=None,
        raise_server_exceptions=True,
        awaitable=False,
    ):
        self.context = self
        # The uri is expected to reach the root API route.
        uri = httpx.URL(uri)
        headers = headers or {}
        # Define this here instead of at module scope so that the SUPPORTED_DECODERS
        # may be modified before Context instantiation.
        ACCEPT_ENCODING = ", ".join(
            [key for key in SUPPORTED_DECODERS.keys() if key != "identity"]
        )
        # Resolve this here, not at module scope, because the test suite
        # injects TILED_CACHE_DIR env var to use a temporary directory.
        TILED_CACHE_DIR = Path(
            os.getenv("TILED_CACHE_DIR", platformdirs.user_cache_dir("tiled"))
        )
        headers.setdefault("accept-encoding", ACCEPT_ENCODING)
        # Set the User Agent to help the server fail informatively if the client
        # version is too old.
        headers.setdefault("user-agent", USER_AGENT)

        # If ?api_key=... is present, move it from the query into a header.
        # The server would accept it in the query parameter, but using
        # a header is a little more secure (e.g. not logged).
        parsed_params = urllib.parse.parse_qs(uri.query.decode())
        api_key_list = parsed_params.pop("api_key", None)
        if api_key_list is not None:
            if api_key is not None:
                raise ValueError(
                    "api_key was provided as query parameter in URI and as keyword argument. Pick one."
                )
            if len(api_key_list) != 1:
                raise ValueError("Cannot handle two api_key query parameters")
            (api_key,) = api_key_list
        if api_key is None:
            # Check for an API key from the environment.
            api_key = os.getenv("TILED_API_KEY")
        # We will set the API key via the `api_key` property below,
        # after constructing the Client object.

        # FastAPI redirects /api -> /api/ so add it here to save a request.
        path = uri.path
        if not path.endswith("/"):
            path = f"{path}/"
        # Construct the uri *without* api_key param.
        # Drop any params/fragments.
        self.api_uri = httpx.URL(
            urllib.parse.urlunsplit((uri.scheme, uri.netloc.decode(), path, {}, ""))
        )
        if timeout is None:
            timeout = httpx.Timeout(**DEFAULT_TIMEOUT_PARAMS)
        if cache is UNSET:
            cache = None
        if app is None:
            ATransport = AsyncTransport if awaitable else Transport
            Client = httpx.AsyncClient if awaitable else httpx.Client
            client = Client(
                transport=ATransport(cache=cache),
                verify=verify,
                timeout=timeout,
                follow_redirects=True,
            )
            self._connection_lock = (
                asyncio.Semaphore(DEFAULT_POOL_SIZE) if awaitable else nullcontext()
            )
            # Do this in the setter to avoid being overwritten.
            client.headers = headers
        elif awaitable:
            # https://tekmusings.com/why-starlette-test-client-breaks-your-pytest-asyncio-tests.html
            base_uri = f"{uri.scheme}://{uri.netloc}"
            client = httpx.AsyncClient(
                transport=AsyncTransport(transport=httpx.ASGITransport(app)),
                verify=False,
                follow_redirects=True,
                base_url=base_uri,
            )
            client.app = app
            self._connection_lock = nullcontext()
        else:
            # Set up an ASGI client.
            # Because we have been handed an app, we can infer that
            # starlette is available.
            from starlette.testclient import TestClient

            base_uri = f"{uri.scheme}://{uri.netloc}"
            # verify parameter is dropped, as there is no SSL in ASGI mode
            client = TestClient(
                app=app,
                raise_server_exceptions=raise_server_exceptions,
                base_url=base_uri,
            )
            self._connection_lock = nullcontext()
            client.timeout = timeout
            client.headers = headers
            # Do this in the setter to avoid being overwritten.
            client.follow_redirects = True
            client._transport = Transport(transport=client._transport, cache=cache)
            client.__enter__()
            # The TestClient is meant to be used only as a context manager,
            # where the context starts and stops and the wrapped ASGI app.
            # We are abusing it slightly to enable interactive use of the
            # TestClient.

            import threading

            # The threading module has its own (internal) atexit
            # mechanism that runs at thread shutdown, prior to the atexit
            # mechanism that runs at interpreter shutdown.
            # We need to intervene at that layer to close the portal, or else
            # we will wait forever for a thread run by the portal to join().
            threading._register_atexit(client.__exit__)

        self.http_client = client
        self._verify = verify
        self._cache = cache
        self._token_cache = Path(TILED_CACHE_DIR / "tokens")
        self._api_key = api_key  # Stash it for use during `connect()`

        # Decide how we will we resolve HTTP requests
        self.is_awaitable = awaitable
        self.send_requests = send_requests_async if awaitable else send_requests
        self.send_requests_generator = (
            send_requests_generator_async if awaitable else send_requests_generator
        )

    def connect(self) -> Generator[TiledRequest, httpx.Response, None]:
        # Make an initial "safe" request to:
        # (1) Get the server_info.
        # (2) Let the server set the CSRF cookie.
        # No authentication has been set up yet, so these requests will be unauthenticated.
        # https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html#double-submit-cookie
        server_info = yield self.build_request(
            "GET",
            self.api_uri,
            headers={
                "Accept": MSGPACK_MIME_TYPE,
                "Cache-Control": "no-cache, no-store",
            },
        )
        server_info = server_info.json()
        self.server_info: About = TypeAdapter(About).validate_python(server_info)
        if self._api_key is UNSET:
            if not self.server_info.authentication.providers:
                # This is a single-user server.
                # Extract the API key from the app and set it.
                from ..server.settings import get_settings

                settings = self.http_client.app.dependency_overrides[get_settings]()
                self._api_key = settings.single_user_api_key or None
            else:
                # This is a multi-user server but no API key was passed,
                # so we will leave it as None on the Context.
                self._api_key = None
        self.api_key = self._api_key  # property setter sets Authorization header
        self.admin = Admin(self)  # accessor for admin-related requests

    def build_request(
        self,
        method,
        url,
        *args,
        auth=httpx._client.UseClientDefault(),
        raw=False,
        **kwargs,
    ):
        """Wrapper around httpx.Client.build_request suitable for yielding to
        `send_requests()`.

        *raw* can be used to instruct `send_requests()` and cousins
        that no retries or error handling should be performed; it is
        the responsibility of the client code to perform these steps.

        """
        request = self.http_client.build_request(method, url, *args, **kwargs)
        return TiledRequest(
            client=self.http_client,
            request=request,
            auth=auth,
            raw=raw,
            lock=self._connection_lock,
        )

    def __repr__(self):
        auth_info = []
        if (self.api_key is None) and (self.http_client.auth is None):
            auth_info.append("(unauthenticated)")
        else:
            auth_info.append("authenticated")
            if self.server_info.authentication.links:
                whoami = self.whoami()
                auth_info.append("as")
                if whoami["type"] == "service":
                    auth_info.append(f"service '{whoami['uuid']}'")
                else:
                    auth_info.append(
                        ",".join(
                            f"'{identity['id']}'" for identity in whoami["identities"]
                        )
                    )
            if self.api_key is not None:
                auth_info.append(
                    f"with API key '{self.api_key[:min(len(self.api_key)//2, 8)]}...'"
                )
        auth_repr = " ".join(auth_info)
        return f"<{type(self).__name__} {auth_repr}>"

    @requestor(asynchronous=False)
    def __enter__(self):
        yield from self.connect()
        return self

    @requestor(asynchronous=True)
    def __aenter__(self):
        yield from self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    async def __aexit__(self, *args):
        await self.aclose()

    def close(self):
        self.http_client.__exit__()

    async def aclose(self):
        await self.http_client.__aexit__()

    def __getstate__(self):
        if getattr(self.http_client, "app", None):
            raise TypeError(
                "Cannot pickle a Tiled Context built around an ASGI. "
                "Only Tiled Context connected to remote servers can be pickled."
            )
        return (
            tiled_version,
            self.api_uri,
            self.http_client.headers,
            list(self.http_client.cookies.jar),
            self.http_client.timeout,
            self.http_client.auth,
            self._verify,
            self._token_cache,
            self.server_info,
            self.cache,
            self.is_awaitable,
        )

    def __setstate__(self, state):
        (
            state_tiled_version,
            api_uri,
            headers,
            cookies_list,
            timeout,
            auth,
            verify,
            token_cache,
            server_info,
            cache,
            awaitable,
        ) = state
        if state_tiled_version != tiled_version:
            raise TypeError(
                f"Cannot unpickle {type(self).__name__} from tiled version {state_tiled_version} "
                f"using tiled version {tiled_version}. Pickle should only be used to short-term "
                "transfer between identical versions of tiled."
            )
        self.api_uri = api_uri
        cookies = httpx.Cookies()
        for cookie in cookies_list:
            cookies.set(
                cookie.name, cookie.value, domain=cookie.domain, path=cookie.path
            )
        Client = httpx.AsyncClient if awaitable else httpx.Client
        Trnsprt = AsyncTransport if awaitable else Transport
        self.http_client = Client(
            verify=verify,
            transport=Trnsprt(cache=cache),
            cookies=cookies,
            timeout=timeout,
            headers=headers,
            follow_redirects=True,
            auth=auth,
        )
        self._connection_lock = (
            asyncio.Semaphore(DEFAULT_POOL_SIZE) if awaitable else nullcontext()
        )
        self._token_cache = token_cache
        self._cache = cache
        self._verify = verify
        self.server_info = server_info
        self.is_awaitable = awaitable

    @classmethod
    @requestor(asynchronous=False)
    def from_any_uri(
        cls,
        uri,
        *,
        headers=None,
        api_key=None,
        cache=UNSET,
        timeout=None,
        verify=True,
        app=None,
        awaitable: bool = False,
    ):
        """
        Accept a URI to a specific node.

        For example, given URI "https://example.com/api/v1//metadata/a/b/c"
        return a Context connected to "https://examples/api/v1" and the list
        ["a", "b", "c"].
        """
        uri = httpx.URL(uri)
        node_path_parts = []
        # Ensure that HTTPS is used if available
        # Logic will follow only one redirect, it is intended ONLY to toggle HTTPS.
        # The redirect will be followed only if the netloc host is identical to the original.
        if uri.scheme == "http":
            response_from_http = yield TiledRequest(None, httpx.Request("GET", uri))
            if response_from_http.is_redirect:
                redirect_uri = httpx.URL(response_from_http.headers["location"])
                if redirect_uri.scheme == "https" and redirect_uri.host == uri.host:
                    uri = redirect_uri
        if "/metadata" in uri.path:
            api_path, _, node_path = uri.path.partition("/metadata")
            api_uri = uri.copy_with(path=api_path)
            node_path_parts.extend(
                [segment for segment in node_path.split("/") if segment]
            )
        # Below we handle the case where we are given *less* than the full URI
        # to the root endpoint. Here we are taking some care to plan for the case
        # where tiled is served at a sub-path, even though that is not yet supported
        # on the server side.
        elif "/api" not in uri.path:
            # Looks like we were given the root path (to the HTML landing page).
            path = uri.path
            if path.endswith("/"):
                path = path[:-1]
            api_uri = uri.copy_with(path=f"{path}/api/v1")
        elif "/v1" not in uri.path:
            # Looks like we were given the /api but no version.
            path = uri.path
            if path.endswith("/"):
                path = path[:-1]
            api_uri = uri.copy_with(path=f"{path}/v1")
        else:
            api_uri = uri
        context = cls(
            api_uri,
            headers=headers,
            api_key=api_key,
            cache=cache,
            timeout=timeout,
            verify=verify,
            app=app,
            awaitable=awaitable,
        )
        return context, node_path_parts

    @classmethod
    def from_app(
        cls,
        app,
        *,
        cache=UNSET,
        headers=None,
        timeout=None,
        api_key=UNSET,
        raise_server_exceptions=True,
        uri=None,
        awaitable=False,
    ):
        """
        Construct a Context around a FastAPI app. Primarily for testing.
        """
        context = cls(
            uri="http://local-tiled-app/api/v1" if not uri else uri,
            headers=headers,
            api_key=None,
            cache=cache,
            timeout=timeout,
            app=app,
            raise_server_exceptions=raise_server_exceptions,
            awaitable=awaitable,
        )
        context._api_key = api_key
        return context

    @property
    def tokens(self) -> DictView[Literal["access_token", "refresh_token"], str]:
        "A view of the current access and refresh tokens."
        return DictView(self.http_client.auth.tokens)

    @property
    def api_key(self):
        # Extract from header to ensure that there is one "ground truth" here
        # and no possibility of state getting out of sync.
        header = self.http_client.headers.get("Authorization", "")
        match = API_KEY_AUTH_HEADER_PATTERN.match(header)
        if match is not None:
            return match.group(1)

    @api_key.setter
    def api_key(self, api_key):
        if api_key is None:
            if self.http_client.headers.get("Authorization", "").startswith("Apikey "):
                self.http_client.headers.pop("Authorization")
        else:
            self.http_client.headers["Authorization"] = f"Apikey {api_key}"

    @property
    def cache(self):
        return self.http_client._transport.cache

    @cache.setter
    def cache(self, cache):
        self.http_client._transport.cache = cache

    def which_api_key(self):
        """
        A 'who am I' for API keys
        """
        if not self.api_key:
            raise RuntimeError("Not API key is configured for the client.")
        for attempt in retry_context():
            with attempt:
                return handle_error(
                    self.http_client.get(
                        self.server_info.authentication.links.apikey,
                        headers={"Accept": MSGPACK_MIME_TYPE},
                    )
                ).json()

    @requestor()
    def create_api_key(self, scopes=None, expires_in=None, note=None, access_tags=None):
        """
        Generate a new API key.

        Users with administrative scopes may use ``Context.admin.revoke_api_key``
        to create API keys on behalf of other users or services.

        Parameters
        ----------
        scopes : Optional[List[str]]
            Restrict the access available to the API key by listing specific scopes.
            By default, this will have the same access as the user.
        expires_in : Optional[Union[int, str]]
            Number of seconds until API key expires, given as integer seconds
            or a string like: '3y' (years), '3d' (days), '5m' (minutes), '1h'
            (hours), '30s' (seconds). If None, it will never expire or it will
            have the maximum lifetime allowed by the server.
        note : Optional[str]
            Description (for humans).
        access_tags : Optional[List[str]]
            Restrict the access available to the API key by listing specific tags.
            By default, this will have no limits on access tags.
        """
        if isinstance(expires_in, str):
            expires_in = parse_time_string(expires_in)
        return (
            yield self.build_request(
                "POST",
                self.server_info.authentication.links.apikey,
                headers={"Accept": MSGPACK_MIME_TYPE},
                json={
                    "scopes": scopes,
                    "access_tags": access_tags,
                    "expires_in": expires_in,
                    "note": note,
                },
            )
        ).json()

    def revoke_api_key(self, first_eight):
        """
        Revoke an API key.

        The API key must belong to the currently-authenticated user or service.
        Users with administrative scopes may use ``Context.admin.revoke_api_key``
        to revoke API keys belonging to other users.

        Parameters
        ----------
        first_eight : str
            Identify the API key to be deleted by passing its first 8 characters.
            (Any additional characters passed will be truncated.)
        """
        url_path = self.server_info.authentication.links.apikey
        for attempt in retry_context():
            with attempt:
                handle_error(
                    self.http_client.delete(
                        url_path,
                        headers={"x-csrf": self.http_client.cookies["tiled_csrf"]},
                        params={
                            **parse_qs(urlparse(url_path).query),
                            "first_eight": first_eight[:8],
                        },
                    )
                )

    @property
    def app(self):
        warnings.warn(
            "The 'app' accessor on Context is deprecated. Use Context.http_client.app.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.http_client.app

    @property
    def base_url(self):
        warnings.warn(
            "The 'base_url' accessor on Context is deprecated. Use Context.http_client.base_url.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.http_client.base_url

    @property
    def event_hooks(self):
        "httpx.Client event hooks. This is exposed for testing."
        warnings.warn(
            "The 'event_hooks' accessor on Context is deprecated. Use Context.http_client.event_hooks.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.http_client.event_hooks

    @requestor()
    def authenticate(
        self,
        *,
        remember_me=True,
    ):
        """
        Log in to a Tiled server.

        Depending on the server's authentication method, this will prompt for username/password:

        >>> c.login()
        Username: USERNAME
        Password: <input is hidden>

        or prompt you to open a link in a web browser to login with a third party:

        >>> c.login()
        You have ... minutes to visit this URL

        https://...

        and enter the code: XXXX-XXXX

        Parameters
        ----------
        remember_me : bool
            Next time, try to automatically authenticate using this session.
        """
        # Obtain tokens via OAuth2 unless the caller has passed them.
        providers = self.server_info.authentication.providers
        tokens = yield from prompt_for_credentials(
            self,
            providers,
        )
        self.configure_auth(tokens, remember_me=remember_me)

    # These two methods are aliased for convenience.
    login = authenticate

    def configure_auth(self, tokens, remember_me=True):
        """
        Configure Tiled client with tokens for refresh flow.

        Parameters
        ----------
        tokens : dict, optional
            Must include keys 'access_token' and 'refresh_token'
        """
        self.http_client.auth = None
        if self.api_key is not None:
            raise RuntimeError(
                "An API key is set. Cannot use both API key and OAuth2 authentication."
            )
        # Configure an httpx.Auth instance on the http_client, which
        # will manage refreshing the tokens as needed.
        refresh_url = self.server_info.authentication.links.refresh_session
        csrf_token = self.http_client.cookies["tiled_csrf"]
        if remember_me:
            token_directory = self._token_directory()
        else:
            # Clear any existing tokens.
            temp_auth = TiledAuth(refresh_url, csrf_token, self._token_directory())
            temp_auth.sync_clear_token("access_token")
            temp_auth.sync_clear_token("refresh_token")
            # Store tokens in memory only, with no syncing to disk.
            token_directory = None
        auth = TiledAuth(refresh_url, csrf_token, token_directory)
        auth.sync_set_token("access_token", tokens["access_token"])
        auth.sync_set_token("refresh_token", tokens["refresh_token"])
        self.http_client.auth = auth

    @property
    def authenticated(self) -> bool:
        """
        Boolean indicated whether session is authenticated (true) or anonymous (false)

        Examples
        --------

        An anonymous session at first, after login, is authenticated.

        >>> client.context.authenticated
        False
        >>> client.login()
        Username: USERNAME
        Password: <input is hidden>
        >>> client.context.authenticated
        True

        """
        # Confirm the state of properties that authentication consists of
        return (self.api_key is not None) or (self.http_client.auth is not None)

    def _token_directory(self):
        # e.g. ~/.config/tiled/tokens/{host:port}
        # with the templated element URL-encoded so it is a valid filename.
        path = Path(
            self._token_cache,
            urllib.parse.quote_plus(str(self.api_uri)),
        )

        # If this directory already exists, it might contain subdirectories
        # left by older versions of tiled that supported caching tokens for
        # multiple users of one server. Clean them up.
        if path.is_dir():
            import shutil

            [shutil.rmtree(item) for item in path.iterdir() if item.is_dir()]

        return path

    def use_cached_tokens(self):
        """
        Attempt to reconnect using cached tokens.

        Returns
        -------
        success : bool
            Indicating whether valid cached tokens were found
        """
        refresh_url = self.server_info.authentication.links.refresh_session
        csrf_token = self.http_client.cookies["tiled_csrf"]

        # Try automatically authenticating using cached tokens, if any.
        token_directory = self._token_directory()
        # We have to make an HTTP request to let the server validate whether we
        # have a valid session.
        self.http_client.auth = TiledAuth(refresh_url, csrf_token, token_directory)
        # This will either:
        # * Use an access_token and succeed.
        # * Use a refresh_token to attempt refresh flow and succeed.
        # * Use a refresh_token to attempt refresh flow and fail, raise.
        # * Find no tokens and raise.
        try:
            self.whoami()
            return True
        except CannotRefreshAuthentication:
            self.http_client.auth = None
            return False

    def force_auth_refresh(self):
        """
        Execute refresh flow.

        This method is exposed for testing and debugging uses.

        It should never be necessary for the user to call. Refresh flow is
        automatically executed by tiled.client.auth.TiledAuth when the current
        access_token expires.
        """
        if self.http_client.auth is None:
            raise RuntimeError(
                "No authentication has been set up. Cannot reauthenticate."
            )
        refresh_token = self.http_client.auth.sync_get_token(
            "refresh_token", reload_from_disk=True
        )
        if refresh_token is None:
            raise CannotRefreshAuthentication("There is no refresh_token.")
        csrf_token = self.http_client.cookies["tiled_csrf"]
        refresh_request = build_refresh_request(
            self.http_client.auth.refresh_url,
            refresh_token,
            csrf_token,
        )
        for attempt in retry_context():
            with attempt:
                token_response = self.http_client.send(refresh_request, auth=None)
                if token_response.status_code == httpx.codes.UNAUTHORIZED:
                    raise CannotRefreshAuthentication(
                        "Session cannot be refreshed. Log in again."
                    )
                handle_error(token_response)
        tokens = token_response.json()
        self.http_client.auth.sync_set_token("access_token", tokens["access_token"])
        self.http_client.auth.sync_set_token("refresh_token", tokens["refresh_token"])
        return tokens

    def whoami(self):
        "Return information about the currently-authenticated user or service."
        for attempt in retry_context():
            with attempt:
                return handle_error(
                    self.http_client.get(
                        self.server_info.authentication.links.whoami,
                        headers={"Accept": MSGPACK_MIME_TYPE},
                    )
                ).json()

    def logout(self):
        """
        Log out of the current session (if any).

        This method is idempotent.
        """
        if self.http_client.auth is None:
            return

        # Revoke the current session.
        refresh_token = self.http_client.auth.sync_get_token("refresh_token")
        for attempt in retry_context():
            with attempt:
                handle_error(
                    self.http_client.post(
                        f"{self.api_uri}auth/session/revoke",
                        json={"refresh_token": refresh_token},
                        # Circumvent auth because this request is not authenticated.
                        # The refresh_token in the body is the relevant proof, not the
                        # 'Authentication' header.
                        auth=None,
                    )
                )

        # Clear on-disk state.
        self.http_client.auth.sync_clear_token("access_token")
        self.http_client.auth.sync_clear_token("refresh_token")

        # Clear in-memory state.
        self.http_client.headers.pop("Authorization", None)
        self.http_client.auth = None

    def revoke_session(self, session_id):
        """
        Revoke a Session so it cannot be refreshed.

        This may be done to ensure that a possibly-leaked refresh token cannot be used.
        """
        for attempt in retry_context():
            with attempt:
                handle_error(
                    self.http_client.delete(
                        self.server_info.authentication.links.revoke_session.format(
                            session_id=session_id
                        ),
                        headers={"x-csrf": self.http_client.cookies["tiled_csrf"]},
                    )
                )


class Admin:
    "Accessor for requests that require administrative privileges."

    def __init__(self, context: Context):
        self.context = context
        self.base_url = context.server_info.links["self"]

    def list_principals(self, offset=0, limit=100):
        "List Principals (users and services) in the authentication database."
        url_path = f"{self.base_url}/auth/principal"
        params = {
            **parse_qs(urlparse(url_path).query),
            "offset": offset,
            "limit": limit,
        }
        for attempt in retry_context():
            with attempt:
                return handle_error(
                    self.context.http_client.get(url_path, params=params)
                ).json()

    def show_principal(self, uuid):
        "Show one Principal (user or service) in the authentication database."
        for attempt in retry_context():
            with attempt:
                return handle_error(
                    self.context.http_client.get(
                        f"{self.base_url}/auth/principal/{uuid}"
                    )
                ).json()

    def create_api_key(
        self, uuid, scopes=None, expires_in=None, note=None, access_tags=None
    ):
        """
        Generate a new API key for another user or service.

        Parameters
        ----------
        uuid : str
            Identify the principal -- the user or service
        scopes : Optional[List[str]]
            Restrict the access available to the API key by listing specific scopes.
            By default, this will have the same access as the principal.
        expires_in : Optional[int]
            Number of seconds until API key expires. If None,
            it will never expire or it will have the maximum lifetime
            allowed by the server.
        note : Optional[str]
            Description (for humans).
        access_tags : Optional[List[str]]
            Restrict the access available to the API key by listing specific tags.
            By default, this will have no limits on access tags.
        """
        for attempt in retry_context():
            with attempt:
                return handle_error(
                    self.context.http_client.post(
                        f"{self.base_url}/auth/principal/{uuid}/apikey",
                        headers={"Accept": MSGPACK_MIME_TYPE},
                        json={
                            "scopes": scopes,
                            "access_tags": access_tags,
                            "expires_in": expires_in,
                            "note": note,
                        },
                    )
                ).json()

    def create_service_principal(
        self,
        role,
    ):
        """
        Generate a new service principal.

        Parameters
        ----------
        role : str
            Specify the role (e.g. user or admin)
        """
        url_path = f"{self.base_url}/auth/principal"
        for attempt in retry_context():
            with attempt:
                return handle_error(
                    self.context.http_client.post(
                        url_path,
                        headers={"Accept": MSGPACK_MIME_TYPE},
                        params={**parse_qs(urlparse(url_path).query), "role": role},
                    )
                ).json()

    def revoke_api_key(self, uuid, first_eight=None):
        """
        Revoke an API key belonging to any user or service.

        Parameters
        ----------
        uuid : str
            Identify the principal whose API key will be deleted. This is
            required in order to reduce the chance of accidentally revoking
            the wrong key.
        first_eight : str
            Identify the API key to be deleted by passing its first 8 characters.
            (Any additional characters passed will be truncated.)
        """
        url_path = f"{self.base_url}/auth/principal/{uuid}/apikey"
        for attempt in retry_context():
            with attempt:
                return handle_error(
                    self.context.http_client.delete(
                        url_path,
                        headers={"Accept": MSGPACK_MIME_TYPE},
                        params={
                            **parse_qs(urlparse(url_path).query),
                            "first_eight": first_eight[:8],
                        },
                    )
                )


class CannotPrompt(Exception):
    pass


def _can_prompt():
    "Infer whether the user can be prompted for a password or user code."

    if (not sys.__stdin__.closed) and sys.__stdin__.isatty():
        return True
    # In IPython (TerminalInteractiveShell) the above is true, but in
    # Jupyter (ZMQInteractiveShell) it is False.
    # Jupyter own mechanism for giving a prompt, so we always return
    # True if we detect IPython/Jupyter.
    if "IPython" in sys.modules:
        import IPython

        if IPython.get_ipython() is not None:
            return True
    return False


@requestor()
def password_grant(context, auth_endpoint, provider, username, password):
    form_data = {
        "grant_type": "password",
        "username": username,
        "password": password,
    }
    token_response = yield context.build_request(
        "POST", auth_endpoint, data=form_data, auth=None
    )
    return token_response.json() if token_response is not None else None


@requestor()
def device_code_grant(context, auth_endpoint):
    verification_response = yield context.build_request(
        "POST", auth_endpoint, json={}, auth=None
    )
    verification = verification_response.json()
    authorization_uri = verification["authorization_uri"]
    print(
        f"""
You have {int(verification['expires_in']) // 60} minutes to visit this URL

{authorization_uri}

and enter the code:

{verification['user_code']}

"""
    )
    import webbrowser

    webbrowser.open(authorization_uri)
    deadline = verification["expires_in"] + time.monotonic()
    print("Waiting...", end="", flush=True)
    while True:
        time.sleep(verification["interval"])
        if time.monotonic() > deadline:
            raise Exception("Deadline expired.")
        for attempt in retry_context():
            with attempt:
                # Intentionally do not wrap this in handle_error(...).
                # Check status codes manually below.
                access_response = yield context.build_request(
                    "POST",
                    verification["verification_uri"],
                    json={
                        "device_code": verification["device_code"],
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    },
                    auth=None,
                    raw=True,
                )
                if (access_response.status_code == httpx.codes.BAD_REQUEST) and (
                    access_response.json()["detail"]["error"] == "authorization_pending"
                ):
                    print(".", end="", flush=True)
                    continue
                handle_error(access_response)
        print("")
        break
    tokens = access_response.json()
    return tokens


def send_requests(requestor: Generator[TiledRequest, httpx.Response, T]) -> T:
    """Perform an HTTP request using *http_client* on behalf of *requestor*."""
    # Prime the generator
    response = None
    while True:
        try:
            request = requestor.send(response)
        except StopIteration as exc:
            return exc.value
        try:
            # Create a default client if one is not explicitly given
            client = httpx.Client() if request.client is None else request.client
            # Finally, we can actually execute the request
            if request.raw:
                response = client.send(request.request, auth=request.auth)
            else:
                for attempt in retry_context():
                    with attempt:
                        response = handle_error(
                            client.send(request.request, auth=request.auth)
                        )
            assert response is not None
        except AttributeError as exc:
            requestor.throw(exc)
        except Exception as exc:
            requestor.throw(exc)


def send_requests_generator(
    requestor: Generator[tuple[httpx.Client | None, httpx.Request], httpx.Response, T]
) -> Generator[T, Any, None]:
    """Perform an HTTP request using *http_client* on behalf of *requestor*."""
    # Prime the generator
    response = None
    while True:
        try:
            maybe_request = requestor.send(response)
        except StopIteration as exc:
            return exc.value
        # Check if yielded item should be passed through (not an HTTP request)
        if not isinstance(maybe_request, TiledRequest):
            yield maybe_request
            continue
        request = maybe_request
        try:
            # Create a default client if one is not explicitly given
            client = httpx.Client() if request.client is None else request.client
            # Finally, we can actually execute the request
            if request.raw:
                response = client.send(request.request, auth=request.auth)
            else:
                for attempt in retry_context():
                    with attempt:
                        response = handle_error(
                            client.send(request.request, auth=request.auth)
                        )
        except Exception as exc:
            requestor.throw(exc)


async def send_requests_async(
    requestor: Generator[
        # Yield type is wrong, need to come back and fix
        TiledRequest | Sequence[Generator],
        httpx.Response,
        T,
    ],
) -> T:
    """Perform an HTTP request using an asynchronous *http_client* on behalf of *requestor*."""
    # Prime the generator
    response = None
    while True:
        try:
            request = requestor.send(response)
        except StopIteration as exc:
            return exc.value
        # Sequences of request generators so we can do them concurrently
        if isinstance(request, Sequence):
            response = await asyncio.gather(
                *(send_requests_async(sub_requestor) for sub_requestor in request)
            )
            continue
        try:
            # Create a default client if one is not explicitly given
            client = httpx.AsyncClient() if request.client is None else request.client
            # Finally, we can actually execute the request
            if request.raw:
                response = client.send(request.request, auth=request.auth)
            else:
                for attempt in retry_context():
                    async with request.lock:  # Avoid overwhelming the connection pool
                        with attempt:
                            response = handle_error(
                                await client.send(request.request, auth=request.auth)
                            )
        except Exception as exc:
            requestor.throw(exc)


async def send_requests_generator_async(
    requestor: Generator[
        tuple[httpx.AsyncClient | None, httpx.Request], httpx.Response, T
    ]
) -> Generator[T, Any, None]:
    """Perform an HTTP request using an asynchronous *http_client* on behalf of *requestor*."""
    # Prime the generator
    response = None
    while True:
        try:
            maybe_request = requestor.send(response)
        except StopIteration:
            return  # Async generators can't return a value
        # Check if yielded item should be passed through (not an HTTP request)
        if not isinstance(maybe_request, TiledRequest):
            yield maybe_request
            continue
        request = maybe_request
        try:
            # Create a default client if one is not explicitly given
            client = httpx.AsyncClient() if request.client is None else request.client
            # Finally, we can actually execute the request
            if request.raw:
                response = client.send(request.request, auth=request.auth)
            else:
                for attempt in retry_context():
                    async with request.lock:  # Avoid overwhelming the connection pool
                        with attempt:
                            response = handle_error(
                                await client.send(request.request, auth=request.auth)
                            )
        except Exception as exc:
            requestor.throw(exc)


def _find_context(obj) -> Context | None:
    """Find the Context attached to an object.

    Either obj.context, or the object itself, otherwise None

    """
    if isinstance(obj, Context):
        return obj
    return getattr(obj, "context", None)
