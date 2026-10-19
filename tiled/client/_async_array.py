"""An asynchronous array client built on the sans-I/O protocol functions.

This is an experimental, minimal async counterpart to
`tiled.client.array.ArrayClient`. It returns in-memory numpy arrays (no dask
integration) and fetches multi-request reads concurrently with
`asyncio.gather`, throttled by the `AsyncContext` semaphore.
"""

from __future__ import annotations

import asyncio
import builtins
import itertools
import math
from typing import Any, Iterable, Union

import numpy
from numpy.typing import NDArray

from ..ndslice import NDBlock, NDSlice
from ..structures.core import STRUCTURE_TYPES, StructureFamily
from ._array_protocol import (
    DEFAULT_RESPONSE_BYTESIZE_LIMIT,
    PATCH_CONFLICT_STATUS_CODES,
    RequestSpec,
    build_get_block_request,
    build_get_slice_request,
    build_patch_request,
    build_write_block_request,
    build_write_full_request,
    empty_array,
    parse_array_response,
    parse_patch_response,
    patch_conflict_message,
    plan_read_split,
)
from ._async_context import AsyncContext
from .utils import ClientError, slices_to_dask_chunks

JSON = dict[str, Any]


class AsyncArrayClient:
    "Async client-side wrapper around an array-like, returning in-memory arrays."

    # See `_DaskArrayClient.RESPONSE_BYTESIZE_LIMIT` for the meaning.
    RESPONSE_BYTESIZE_LIMIT = DEFAULT_RESPONSE_BYTESIZE_LIMIT

    def __init__(
        self, context: AsyncContext, item: JSON, structure: Any = None
    ) -> None:
        # A known structure may be passed in (e.g. by `AsyncContainer.new`,
        # where `item` may not carry a JSON-encoded structure); otherwise it
        # is parsed from the item.
        self.context = context
        self._item = item
        if structure is None:
            structure_type = STRUCTURE_TYPES[self.structure_family]
            structure = structure_type.from_json(item["attributes"]["structure"])
        self._structure = structure

    @classmethod
    async def from_path(
        cls, context: AsyncContext, path: Union[str, Iterable[str]]
    ) -> "AsyncArrayClient":
        "Fetch the item JSON for `path` and construct a client around it."
        if isinstance(path, str):
            segments = [segment for segment in path.split("/") if segment]
        else:
            segments = list(path)
        url_path = str(context.api_uri) + "metadata/" + "/".join(segments)
        spec = RequestSpec("GET", url_path, headers={"Accept": "application/json"})
        response = await context.send(spec)
        return cls(context, response.json()["data"])

    @property
    def item(self) -> JSON:
        "JSON payload describing this item. Mostly for internal use."
        return self._item

    def structure(self) -> Any:
        "Return a dataclass describing the structure of the array."
        return self._structure

    @property
    def structure_family(self) -> StructureFamily:
        return StructureFamily[self._item["attributes"]["structure_family"]]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._structure.shape

    @property
    def chunks(self) -> tuple[tuple[int, ...], ...]:
        return self._structure.chunks

    @property
    def dtype(self) -> numpy.dtype:
        return self._structure.data_type.to_numpy_dtype()

    @property
    def ndim(self) -> int:
        return len(self._structure.shape)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} shape={self.shape} dtype={self.dtype}>"

    async def read_block(
        self, block: tuple[int, ...], slice: Any = None
    ) -> NDArray[Any]:
        """Access the data for one block of this chunked array.

        Optionally, access only a slice *within* this block.
        """
        block_, block_slice = NDBlock(block), NDSlice(slice)
        try:
            block_.shape_from_chunks(self.chunks)
        except IndexError:
            raise IndexError(f"Block index {block} out of range")
        spec, exp_shape = build_get_block_request(
            self.item["links"]["block"], block_, block_slice, self.chunks
        )
        if spec is None:
            # An array with 0 as one of its dimensions never contains data,
            # so we can short-circuit here without contacting the service.
            return empty_array(self.dtype, exp_shape)
        response = await self.context.send(spec)
        return parse_array_response(response.content, self.dtype, exp_shape)

    async def _get_slice(self, slice: NDSlice) -> NDArray[Any]:
        "Fetch the data for one (sub-)slice of the full array."
        spec, exp_shape = build_get_slice_request(
            self.item["links"]["full"], slice, self.shape
        )
        if spec is None:
            return empty_array(self.dtype, exp_shape)
        response = await self.context.send(spec)
        return parse_array_response(response.content, self.dtype, exp_shape)

    async def read(self, slice: Any = None) -> NDArray[Any]:
        """Access the entire array or its slice.

        Reads expected to exceed `RESPONSE_BYTESIZE_LIMIT` are split along
        chunk boundaries (exactly as the sync client does) and fetched
        concurrently with `asyncio.gather`.
        """
        # Determine the expected shape of the resulting array after slicing
        if arr_slice := NDSlice(slice):
            arr_slice = arr_slice.expand_for_shape(self.shape)  # Remove "..."
        exp_shape = arr_slice.shape_after_slice(self.shape)

        # Check for special case of shape with 0 in it.
        if 0 in exp_shape:
            return empty_array(self.dtype, exp_shape)

        # If the expected response is small, fetch it in one go.
        total_bytes = math.prod(exp_shape) * self.dtype.itemsize
        if total_bytes < self.RESPONSE_BYTESIZE_LIMIT:
            return await self._get_slice(arr_slice)

        # The response is expected to be large; subslice it, fetch the pieces
        # concurrently, and reassemble them into one array.
        indexed_slices = plan_read_split(
            arr_slice,
            self.shape,
            self.chunks,
            self.dtype.itemsize,
            self.RESPONSE_BYTESIZE_LIMIT,
        )
        pieces = await asyncio.gather(
            *(self._get_slice(slc) for slc in indexed_slices.values())
        )
        # Compute the offset of each piece within the result from the sizes
        # of the pieces along each axis (the same grid dask would use).
        grid = slices_to_dask_chunks(indexed_slices, self.shape)
        offsets = tuple(tuple(itertools.accumulate(axis, initial=0)) for axis in grid)
        result = numpy.empty(exp_shape, dtype=self.dtype)
        for index, piece in zip(indexed_slices.keys(), pieces):
            destination = tuple(
                builtins.slice(offsets[axis][i], offsets[axis][i] + piece.shape[axis])
                for axis, i in enumerate(index)
            )
            result[destination] = piece
        return result

    async def write(self, array: Any, persist: bool = True) -> None:
        "Overwrite the full array."
        spec = build_write_full_request(self.item["links"]["full"], array, persist)
        await self.context.send(spec)

    async def write_block(
        self,
        array: Any,
        block: tuple[int, ...],
        slice: Any = ...,
        persist: bool = True,
    ) -> None:
        "Overwrite one block of the array, or a slice within it."
        spec = build_write_block_request(
            self.item["links"]["block"], array, block, slice, persist
        )
        await self.context.send(spec)

    async def patch(
        self,
        array: NDArray[Any],
        offset: Union[int, tuple[int, ...]],
        extend: bool = False,
        persist: bool = True,
    ) -> None:
        """Write data into a slice of an array, maybe extending the shape.

        See `tiled.client.array._DaskArrayClient.patch` for details.
        """
        spec = build_patch_request(
            self.item["links"]["full"], array, offset, self.dtype, extend, persist
        )
        try:
            response = await self.context.send(spec)
        except ClientError as exc:
            if exc.response.status_code in PATCH_CONFLICT_STATUS_CODES:
                raise ValueError(patch_conflict_message(exc.response.json())) from None
            raise
        # Update cached structure.
        self._structure = parse_patch_response(response.json(), self.structure_family)
