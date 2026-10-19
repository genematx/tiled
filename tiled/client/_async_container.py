"""An asynchronous container client built on the sans-I/O protocol functions.

This is an experimental, minimal async counterpart to
`tiled.client.container.Container`. It supports creating nodes (`new`,
`create_container`, `write_array`), single-segment child lookup (`get`), and
listing keys.

The create flow, `new`, mirrors the synchronous `Container.new` exactly. The
pure protocol knowledge -- request building (including server-version
backcompat and the register-vs-metadata endpoint choice), the adopt-match
check, and response-document merging -- is shared with the sync client via
`tiled.client._container_protocol`. The retry orchestration is hand-written
here in async form because it needs attempt-level I/O state: the
`may_have_reached_server` flag must be updated per attempt, and a 409
Conflict must branch into the adopt path between attempts. For that reason
`new` drives its own `stamina.retry_context` loop around single-attempt
sends (`AsyncContext.send_once`) instead of using the retrying
`AsyncContext.send` choke point; the predicate and limits are the same ones
`send` (and the sync `retry_context`) use, so the retry semantics match.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Optional, Union

import httpx
import stamina

from ..structures.array import ArrayStructure
from ..structures.core import Spec, StructureFamily
from ..structures.data_source import DataSource
from ._async_array import AsyncArrayClient
from ._async_context import AsyncContext
from ._container_protocol import (
    CreateFailureAction,
    build_create_request,
    classify_create_failure,
    match_existing_node,
    merge_create_document,
)
from ._request_spec import RequestSpec
from .utils import (
    MSGPACK_MIME_TYPE,
    TILED_RETRY_ATTEMPTS,
    TILED_RETRY_TIMEOUT,
    ClientError,
    should_retry,
)

logger = logging.getLogger("tiled.client")

JSON = dict[str, Any]
AnyClient = Union["AsyncContainer", AsyncArrayClient]


class AsyncContainer:
    "Async client-side wrapper around a container node."

    def __init__(self, context: AsyncContext, item: JSON) -> None:
        self.context = context
        self._item = item

    @classmethod
    async def from_context(
        cls, context: AsyncContext, path: Union[str, Iterable[str]] = ()
    ) -> "AsyncContainer":
        "Fetch the item JSON for the container at `path` and wrap it."
        if isinstance(path, str):
            segments = [segment for segment in path.split("/") if segment]
        else:
            segments = list(path)
        url_path = str(context.api_uri) + "metadata/" + "/".join(segments)
        spec = RequestSpec("GET", url_path, headers={"Accept": MSGPACK_MIME_TYPE})
        response = await context.send(spec)
        return cls(context, response.json()["data"])

    @property
    def item(self) -> JSON:
        "JSON payload describing this item. Mostly for internal use."
        return self._item

    @property
    def uri(self) -> str:
        "Direct link to this entry"
        return self._item["links"]["self"]

    @property
    def path_parts(self) -> list[str]:
        "Location of node in tree, given as list of path segments."
        return self._item["attributes"]["ancestors"] + [self._item["id"]]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {'/'.join(self.path_parts)!r}>"

    def _client_for_item(self, item: JSON, structure: Any = None) -> AnyClient:
        "Dispatch a client for `item` by structure family (container or array)."
        family = StructureFamily(item["attributes"]["structure_family"])
        if family == StructureFamily.container:
            return AsyncContainer(self.context, item)
        if family == StructureFamily.array:
            return AsyncArrayClient(self.context, item, structure=structure)
        raise NotImplementedError(
            f"No async client implemented for structure family {family!r}"
        )

    async def keys(self) -> list[str]:
        "List the keys of this container's children. (First page only.)"
        link = self._item["links"]["search"]
        spec = RequestSpec(
            "GET", link, params={"fields": ""}, headers={"Accept": MSGPACK_MIME_TYPE}
        )
        response = await self.context.send(spec)
        return [result["id"] for result in response.json()["data"]]

    async def get(self, key: str) -> AnyClient:
        "Look up a direct child of this container by a single path segment."
        self_link = self._item["links"]["self"].rstrip("/")
        spec = RequestSpec(
            "GET", f"{self_link}/{key}", headers={"Accept": MSGPACK_MIME_TYPE}
        )
        try:
            response = await self.context.send(spec)
        except ClientError as err:
            if err.response.status_code == httpx.codes.NOT_FOUND:
                raise KeyError(key) from None
            raise
        return self._client_for_item(response.json()["data"])

    async def new(
        self,
        structure_family: Union[StructureFamily, str],
        data_sources: list[DataSource],
        *,
        key: Optional[str] = None,
        metadata: Optional[JSON] = None,
        specs: Optional[list[Union[str, Spec]]] = None,
        access_tags: Optional[list[str]] = None,
    ) -> AnyClient:
        """Create a new item within this container.

        This is the async counterpart of the synchronous `Container.new`; see
        that method for the full documentation. The orchestration below is
        intentionally a line-for-line mirror of the sync loop.
        """
        spec, item, key = build_create_request(
            self.uri,
            self.path_parts,
            structure_family,
            data_sources,
            key=key,
            metadata=metadata,
            specs=specs,
            access_tags=access_tags,
            library_version=self.context.server_info.library_version,
        )

        # All failure semantics live in `classify_create_failure`; this loop
        # is purely mechanical: send, classify, act on the decision, re-raise.
        may_have_reached_server = False
        async for attempt in stamina.retry_context(
            on=should_retry,
            attempts=TILED_RETRY_ATTEMPTS,
            timeout=TILED_RETRY_TIMEOUT,
        ):
            with attempt:
                try:
                    document = (await self.context.send_once(spec)).json()
                except Exception as exc:
                    decision = classify_create_failure(exc, may_have_reached_server)
                    may_have_reached_server = decision.may_have_reached_server
                    if decision.action is CreateFailureAction.TRY_ADOPT:
                        existing = await self._adopt_after_retried_create_conflict(
                            key, structure_family, specs, metadata
                        )
                        if existing is not None:
                            return existing
                    raise

        item, structure = merge_create_document(
            item, document, structure_family, data_sources
        )
        return self._client_for_item(item, structure)

    async def _adopt_after_retried_create_conflict(
        self,
        key: str,
        structure_family: Union[StructureFamily, str],
        specs: Optional[list[Union[str, Spec]]],
        metadata: Optional[JSON],
    ) -> Optional[AnyClient]:
        """Fetch and return an existing node after a retried create hit 409 Conflict.

        The async counterpart of the synchronous
        `Container._adopt_after_retried_create_conflict`; see that method for
        the full rationale. The match check is pure and shared:
        `match_existing_node`.
        """
        try:
            existing = await self.get(key)
        except Exception:
            logger.warning(
                "Create of %r returned 409 Conflict after a retried request, "
                "but the existing item could not be fetched; treating as a "
                "genuine conflict.",
                key,
            )
            return None

        mismatch = match_existing_node(
            existing.item["attributes"], structure_family, specs, metadata
        )
        if mismatch is not None:
            logger.warning(
                "Create of %r returned 409 Conflict after a retried request, "
                "but the existing item does not match the request (%s); "
                "treating as a genuine conflict.",
                key,
                mismatch,
            )
            return None

        logger.warning(
            "Create of %r returned 409 Conflict after a retried request that "
            "may have reached the server; adopting the existing (matching) "
            "item as the result of this create.",
            key,
        )
        return existing

    async def create_container(
        self,
        key: Optional[str] = None,
        *,
        metadata: Optional[JSON] = None,
        specs: Optional[list[Union[str, Spec]]] = None,
        access_tags: Optional[list[str]] = None,
    ) -> "AsyncContainer":
        "Create a new, empty container. See the synchronous `create_container`."
        return await self.new(
            StructureFamily.container,
            [],
            key=key,
            metadata=metadata,
            specs=specs,
            access_tags=access_tags,
        )

    async def write_array(
        self,
        array: Any,
        *,
        key: Optional[str] = None,
        metadata: Optional[JSON] = None,
        dims: Optional[list[str]] = None,
        specs: Optional[list[Union[str, Spec]]] = None,
        access_tags: Optional[list[str]] = None,
    ) -> AsyncArrayClient:
        """Write an array. See the synchronous `write_array`.

        Unlike the sync client, this writes the data in a single request
        (no chunk-wise upload fan-out).
        """
        import numpy

        if not (hasattr(array, "shape") and hasattr(array, "dtype")):
            # This does not implement enough of the array-like interface.
            # Coerce to numpy.
            array = numpy.asarray(array)
        structure = ArrayStructure.from_array(array, dims=dims)
        client = await self.new(
            StructureFamily.array,
            [
                DataSource(structure=structure, structure_family=StructureFamily.array),
            ],
            key=key,
            metadata=metadata,
            specs=specs,
            access_tags=access_tags,
        )
        await client.write(array)
        return client
