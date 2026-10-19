"""Minimal asynchronous counterpart of `tiled.client.container.Container`.

Drives the same requestor generators as the sync `Container` -- notably
`_new_requestor`, which encapsulates the entire multi-exchange create
flow (server-version backcompat, retry-state tracking, adoption of our
own node after a retried create hit 409 Conflict, and response document
post-processing) -- via `send_requests_async`.

Experimental; the surface is intentionally minimal: `new`,
`create_container`, `get`, and `write_array`. See `AsyncArrayClient` for
the array counterpart and `AsyncContext` for what is deliberately out of
scope on the async side.
"""

import itertools
from typing import Any, Dict, List, Optional, Union

import httpx

from ..structures.core import Spec, StructureFamily
from ..structures.data_source import DataSource
from ._async_context import AsyncContext
from ._requestor import send_requests_async
from .array import AsyncArrayClient, _metadata_requestor
from .container import Container, _new_requestor
from .utils import ClientError


class AsyncContainer:
    """Asynchronous client for a container node.

    Use `from_context` to construct:

    >>> async with AsyncContext("https://tiled.example.com/api/v1",
    ...                         api_key="secret") as context:
    ...     root = await AsyncContainer.from_context(context)
    ...     child = await root.create_container("x")
    """

    # See the comment on `Container._SUGGESTED_MAX_UPLOAD_SIZE`.
    _SUGGESTED_MAX_UPLOAD_SIZE = Container._SUGGESTED_MAX_UPLOAD_SIZE

    def __init__(self, context: AsyncContext, item: Dict[str, Any]):
        self.context = context
        self.item = item

    @classmethod
    async def from_context(
        cls, context: AsyncContext, path: str = ""
    ) -> "AsyncContainer":
        "Construct from one async GET to the `/metadata/{path}` endpoint."
        content = await send_requests_async(
            _metadata_requestor(f"{context.api_uri}metadata/{path.lstrip('/')}"),
            context,
        )
        return cls(context, content["data"])

    def __repr__(self):
        path = "/".join(segment for segment in self.path_parts if segment)
        return f"<{type(self).__name__} /{path}>"

    @property
    def uri(self) -> str:
        "Direct link to this node."
        return self.item["links"]["self"]

    @property
    def path_parts(self) -> List[str]:
        "Location of this node in the tree, as a list of path segments."
        return list(self.item["attributes"]["ancestors"]) + [self.item["id"]]

    def _dispatch(
        self, item: Dict[str, Any], structure: Any = None
    ) -> Union["AsyncContainer", AsyncArrayClient]:
        "Wrap an item document in the appropriate async client class."
        structure_family = StructureFamily(item["attributes"]["structure_family"])
        if structure_family == StructureFamily.container:
            return AsyncContainer(self.context, item)
        if structure_family == StructureFamily.array:
            return AsyncArrayClient(self.context, item, structure=structure)
        raise NotImplementedError(
            f"The experimental async client does not support structure family "
            f"{structure_family!r}; only `container` and `array` so far."
        )

    async def get(self, key: str) -> Union["AsyncContainer", AsyncArrayClient]:
        "Look up a direct child of this node by a single-segment key."
        try:
            content = await send_requests_async(
                _metadata_requestor(f"{self.uri.rstrip('/')}/{key}"),
                self.context,
            )
        except ClientError as err:
            if err.response.status_code == httpx.codes.NOT_FOUND:
                raise KeyError(key)
            raise
        return self._dispatch(content["data"])

    async def new(
        self,
        structure_family: Union[str, StructureFamily],
        data_sources: List[DataSource],
        *,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        specs: Optional[List[Union[str, Spec]]] = None,
        access_tags: Optional[List[str]] = None,
    ) -> Union["AsyncContainer", AsyncArrayClient]:
        """Create a new item within this node.

        Async counterpart of `Container.new`, driving the same
        `_new_requestor` generator (and therefore the same retry/adoption
        semantics for creates whose response was lost in transit).
        """
        item, structure, _adopted = await send_requests_async(
            _new_requestor(
                self_link=self.uri,
                ancestors=self.path_parts,
                structure_family=structure_family,
                data_sources=data_sources,
                key=key,
                metadata=metadata,
                specs=specs,
                access_tags=access_tags,
                server_library_version=self.context.server_info.library_version,
            ),
            self.context,
        )
        # When an existing node was adopted (after a retried create hit 409),
        # `structure` is None and the client class parses the structure from
        # the fetched item's attributes.
        return self._dispatch(item, structure)

    async def create_container(
        self,
        key: Optional[str] = None,
        *,
        metadata: Optional[Dict[str, Any]] = None,
        specs: Optional[List[Union[str, Spec]]] = None,
        access_tags: Optional[List[str]] = None,
    ) -> "AsyncContainer":
        "Create a new, empty container. See `Container.create_container`."
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
        array,
        *,
        key: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        dims: Optional[List[str]] = None,
        specs: Optional[List[Union[str, Spec]]] = None,
        access_tags: Optional[List[str]] = None,
    ) -> AsyncArrayClient:
        """Write an array. Async counterpart of `Container.write_array`.

        Builds the structure the same way as the sync method; chunked
        uploads are written block by block (no dask fan-out here).
        """
        import numpy
        from dask.array.core import normalize_chunks

        from ..structures.array import ArrayStructure, BuiltinDtype

        if not (hasattr(array, "shape") and hasattr(array, "dtype")):
            # This does not implement enough of the array-like interface.
            # Coerce to numpy.
            array = numpy.asarray(array)

        # Determine chunks such that each chunk is not too large to upload.
        # Any existing chunking will be taken into account.
        # If the array is small, there will be only one chunk.
        if hasattr(array, "chunks"):
            chunks = normalize_chunks(
                array.chunks,
                limit=self._SUGGESTED_MAX_UPLOAD_SIZE,
                dtype=array.dtype,
                shape=array.shape,
            )
        else:
            chunks = normalize_chunks(
                tuple("auto" for _ in array.shape),
                limit=self._SUGGESTED_MAX_UPLOAD_SIZE,
                dtype=array.dtype,
                shape=array.shape,
            )

        structure = ArrayStructure(
            shape=array.shape,
            chunks=chunks,
            dims=dims,
            data_type=BuiltinDtype.from_numpy_dtype(array.dtype),
        )
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
        # Materialize (e.g. compute a dask array) before uploading.
        array = numpy.asarray(array)
        chunked = any(len(dim) > 1 for dim in chunks)
        if not chunked:
            await client.write(array)
        else:
            # Write each block; its offsets follow from the chunk sizes.
            starts = [tuple(itertools.accumulate(dim, initial=0)) for dim in chunks]
            for block in itertools.product(*(range(len(dim)) for dim in chunks)):
                block_slice = tuple(
                    slice(starts[axis][i], starts[axis][i] + chunks[axis][i])
                    for axis, i in enumerate(block)
                )
                await client.write_block(
                    numpy.ascontiguousarray(array[block_slice]), block=block
                )
        return client
