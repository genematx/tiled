"""Pure, I/O-free protocol logic for creating nodes within a container.

Each function here builds a `RequestSpec` describing a single HTTP exchange
(a request-builder), transforms request/response payloads (a parser/merger),
or encodes a control-flow decision (`classify_create_failure`). No I/O
happens in this module; the sync shell (`tiled.client.container.Container.new`)
and the async shell (`tiled.client._async_container.AsyncContainer.new`) both
delegate all protocol knowledge to these functions, keeping their retry loops
purely mechanical.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Union

import httpx
import orjson
from packaging.version import Version

from ..structures.core import STRUCTURE_TYPES, Spec, StructureFamily
from ..structures.data_source import DataSource
from ..utils import safe_json_dump
from ._request_spec import RequestSpec
from .utils import MSGPACK_MIME_TYPE, normalize_specs, request_may_have_reached_server

JSON = dict[str, Any]
Specs = Optional[Union[list[str], list[Spec], str]]


def _as_spec_set(spec_dicts: Optional[Iterable[JSON]]) -> set[tuple[str, Any]]:
    # Specs that have been passed through normalize_specs -- as well as the
    # specs on a fetched item -- are lists of {"name": ..., "version": ...}
    # dicts; represent them as a set for order-insensitive comparison.
    return {(spec["name"], spec.get("version")) for spec in (spec_dicts or [])}


def build_create_request(
    uri: str,
    path_parts: list[str],
    structure_family: Union[StructureFamily, str],
    data_sources: list[DataSource],
    *,
    key: Optional[str],
    metadata: Optional[JSON],
    specs: Specs,
    access_tags: Optional[list[str]],
    library_version: str,
) -> tuple[RequestSpec, JSON, str]:
    """Build the request creating a new node under the container at `uri`.

    Returns `(spec, item, key)`: the request to send, the client-side `item`
    skeleton to be completed by `merge_create_document` once the create
    succeeds, and the key -- generated here if not supplied -- which acts as
    an idempotency token across retries of this request.

    `library_version` is the server's reported library version; it selects
    backward-compatible body construction for older servers.
    """
    metadata = metadata or {}

    # Backompatibility: if the server is older than 0.2.4,
    # it can not accept the "properties" field in the data source.
    # This can be removed in later releases.
    if Version(library_version) < Version("0.2.4"):
        data_sources_as_dicts = [
            {k: v for k, v in asdict(ds).items() if k != "properties"}
            for ds in data_sources
        ]
    else:
        data_sources_as_dicts = [asdict(ds) for ds in data_sources]

    item = {
        "attributes": {
            "ancestors": path_parts,
            "metadata": metadata,
            "structure_family": StructureFamily(structure_family),
            "specs": normalize_specs(specs or []),
            "data_sources": data_sources_as_dicts,
            "access_tags": access_tags,
        }
    }
    if key is None:
        key = str(uuid.uuid4())
    body = dict(item["attributes"])
    body["id"] = key

    if any(data_source.assets for data_source in data_sources):
        endpoint = uri.replace("/metadata/", "/register/", 1)
    else:
        endpoint = uri

    spec = RequestSpec(
        "POST",
        endpoint,
        headers={
            "Accept": MSGPACK_MIME_TYPE,
            "Content-Type": "application/json",
        },
        content=safe_json_dump(body),
    )
    return spec, item, key


class CreateFailureAction(enum.Enum):
    "What the create retry loop should do about a failed attempt."

    RAISE = "raise"
    TRY_ADOPT = "try_adopt"


@dataclass(frozen=True)
class CreateFailureDecision:
    "The outcome of `classify_create_failure` for one failed create attempt."

    may_have_reached_server: bool
    action: CreateFailureAction


def classify_create_failure(
    exc: Exception, may_have_reached_server: bool
) -> CreateFailureDecision:
    """Decide how one failed create attempt affects the create retry loop.

    This function owns the cross-attempt state transition of the create flow;
    the shells' retry loops must contain no semantic decisions of their own,
    only the mechanical skeleton: send, classify, act on the decision,
    re-raise into the retry machinery.

    `may_have_reached_server` tracks whether any failed attempt so far may
    have been received and processed by the server: the request was sent but
    the response was lost (read/write transport error) or a proxy/server-side
    error (5xx) occurred after the fact. Once True it never goes back to
    False. In that case a 409 Conflict on a subsequent attempt most likely
    refers to the node created by our own earlier attempt (every create
    carries a key -- supplied or client-generated -- that acts as an
    idempotency token), so the caller should try to adopt that node
    (`TRY_ADOPT`) instead of failing.

    Ordering subtleties, encoded exactly as in the original inline ladder:
    a 5xx sets the flag but still raises (an attempt never adopts on its own
    failure); a 409 does not set the flag; adoption is offered only when the
    *incoming* flag is already True. Every decision carries `RAISE` or ends
    in a re-raise: whether another attempt happens is the retry machinery's
    call, not this function's.
    """
    updated_flag = may_have_reached_server or request_may_have_reached_server(exc)
    if (
        may_have_reached_server
        and isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code == httpx.codes.CONFLICT
    ):
        action = CreateFailureAction.TRY_ADOPT
    else:
        action = CreateFailureAction.RAISE
    return CreateFailureDecision(updated_flag, action)


def match_existing_node(
    existing_attrs: JSON,
    structure_family: Union[StructureFamily, str],
    specs: Specs,
    metadata: Optional[JSON],
) -> Optional[str]:
    """Check whether an existing node matches a create request it conflicted with.

    Returns None if the node's attributes match the request (same structure
    family, same specs, and the same metadata after the JSON round-trip
    normalization that a create applies), or a human-readable description of
    the first mismatch otherwise.
    """
    structure_family = StructureFamily(structure_family)
    specs = normalize_specs(specs or [])
    existing_family = StructureFamily(existing_attrs["structure_family"])
    # The elif chain evaluates lazily, so cheaper checks short-circuit the
    # more expensive ones (the metadata check serializes the submitted
    # metadata with the same JSON round-trip normalization that a create
    # applies: numpy scalars, datetimes, etc. become plain JSON types).
    if existing_family != structure_family:
        mismatch = f"structure_family {existing_family!r} != {structure_family!r}"
    elif _as_spec_set(existing_attrs.get("specs")) != _as_spec_set(specs):
        mismatch = f"specs {existing_attrs.get('specs')!r} != {specs!r}"
    elif existing_attrs.get("metadata") != orjson.loads(safe_json_dump(metadata or {})):
        mismatch = "metadata differs"
    else:
        mismatch = None
    return mismatch


def merge_create_document(
    item: JSON,
    document: JSON,
    structure_family: Union[StructureFamily, str],
    data_sources: list[DataSource],
) -> tuple[JSON, Any]:
    """Merge the server's create response `document` into the `item` skeleton.

    Returns the completed `(item, structure)` pair from which a client can be
    dispatched. Both `item` and `document` are modified in place.
    """
    if structure_family == StructureFamily.container:
        structure: Any = {"contents": None, "count": None}
    else:
        # Only containers can have multiple data_sources right now.
        (data_source,) = data_sources
        structure = data_source.structure
    item["attributes"]["structure"] = structure

    # If server returned modified metadata update the local copy.
    # Otherwise, round-trip it through a JSON serializer locally
    # to ensure that any special types (e.g. datetimes, numpy arrays)
    # are normalized to natively JSON serializable types.
    if "metadata" in document:
        item["attributes"]["metadata"] = document.pop("metadata")
    else:
        item["attributes"]["metadata"] = orjson.loads(
            safe_json_dump(item["attributes"]["metadata"])
        )

    # Ditto for structure
    if "structure" in document:
        item["attributes"]["structure"] = STRUCTURE_TYPES[structure_family](
            document.pop("structure")
        )

    # And for data sources
    if "data_sources" in document:
        item["attributes"]["data_sources"] = [ds for ds in document.pop("data_sources")]

    # And for access_tags
    if "access_tags" in document:
        item["attributes"]["access_tags"] = document.pop("access_tags")

    # Merge in "id" and "links" returned by the server.
    item.update(document)

    # Ensure this is a dataclass, not a dict.
    # When we apply type hints and mypy to the client it should be possible
    # to dispense with this.
    if (structure_family != StructureFamily.container) and isinstance(structure, dict):
        structure_type = STRUCTURE_TYPES[structure_family]
        structure = structure_type.from_json(structure)

    return item, structure
