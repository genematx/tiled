"""Pure, I/O-free protocol logic for the array client.

Each function here either builds a `RequestSpec` describing a single HTTP
exchange (a request-builder) or converts response content into client-side
values (a response-parser). No I/O happens in this module; the sync shell
(`tiled.client.array`) and the async shell (`tiled.client._async_array`)
both delegate all protocol knowledge to these functions.
"""

from __future__ import annotations

import itertools
from typing import Any, Optional, Union
from urllib.parse import parse_qs, urlparse

import httpx
import numpy
from numpy.typing import DTypeLike, NDArray

from ..ndslice import NDBlock, NDSlice, split_slice
from ..structures.core import STRUCTURE_TYPES, StructureFamily
from ..type_aliases import Chunks
from ._request_spec import RequestSpec
from .utils import params_from_slice

OCTET_STREAM = "application/octet-stream"

# The limit on the expected size of the response body (before compression).
# This is used to determine how to combine multiple requests when fetching
# data in blocks.
DEFAULT_RESPONSE_BYTESIZE_LIMIT = 100 * 1024 * 1024  # 100 MiB

# Status codes with which the server reports that patch parameters conflict
# with the existing array (e.g. would overwrite without `extend=True`).
PATCH_CONFLICT_STATUS_CODES = (httpx.codes.BAD_REQUEST, httpx.codes.CONFLICT)


def build_get_block_request(
    url_path: str,
    block: NDBlock,
    block_slice: Optional[NDSlice],
    chunks: Chunks,
) -> tuple[Optional[RequestSpec], tuple[int, ...]]:
    """Build a request for one block (chunk) of an array, via `/array/block`.

    Returns a `(spec, expected_shape)` pair. The spec is None when the
    expected shape contains 0: an array with 0 as one of its dimensions never
    contains data (this is valid, and has come up in the wild), so the caller
    can short-circuit with `empty_array` and skip the request entirely.
    """
    exp_shape: tuple[int, ...] = ()
    # Expand the block to convert for URL
    block = block.expand_for_shape(tuple(len(dim) for dim in chunks))
    if shape := block.shape_from_chunks(chunks):
        exp_shape = block_slice.shape_after_slice(shape) if block_slice else shape
        # Check for special case of shape with 0 in it.
        if 0 in exp_shape:
            return None, exp_shape
    params = {
        **parse_qs(urlparse(url_path).query),
        "block": block.to_numpy_str(),
        "expected_shape": ",".join(map(str, exp_shape)) or "scalar",
    }
    params = params | ({"slice": block_slice.to_numpy_str()} if block_slice else {})
    spec = RequestSpec("GET", url_path, params=params, headers={"Accept": OCTET_STREAM})
    return spec, exp_shape


def build_get_slice_request(
    url_path: str,
    slice: NDSlice,
    shape: tuple[int, ...],
) -> tuple[Optional[RequestSpec], tuple[int, ...]]:
    """Build a request for a slice of the full array, via `/array/full`.

    Returns a `(spec, expected_shape)` pair; the spec is None when the
    expected shape contains 0 (see `build_get_block_request`).
    """
    # Determine the expected shape of the resulting array after slicing
    exp_shape = slice.shape_after_slice(shape) if slice else shape
    # Check for special case of shape with 0 in it.
    if 0 in exp_shape:
        return None, exp_shape
    params = {
        **parse_qs(urlparse(url_path).query),
        "expected_shape": ",".join(map(str, exp_shape)) or "scalar",
    }
    params = params | ({"slice": slice.to_numpy_str()} if slice else {})
    spec = RequestSpec("GET", url_path, params=params, headers={"Accept": OCTET_STREAM})
    return spec, exp_shape


def parse_array_response(
    content: bytes, dtype: DTypeLike, shape: tuple[int, ...]
) -> NDArray[Any]:
    "Convert a raw `octet-stream` response body into a numpy array."
    return numpy.frombuffer(content, dtype=dtype).reshape(shape)


def empty_array(dtype: DTypeLike, shape: tuple[int, ...]) -> NDArray[Any]:
    "Construct the (empty) result of a request short-circuited client-side."
    return numpy.array([], dtype=dtype).reshape(shape)


def plan_read_split(
    arr_slice: NDSlice,
    shape: tuple[int, ...],
    chunks: Chunks,
    itemsize: int,
    bytesize_limit: int,
) -> dict[tuple[int, ...], NDSlice]:
    """Split a large read into sub-slices, each below `bytesize_limit`.

    Chunk boundaries along each axis are used as the best candidate split
    points. Returns a mapping of chunk-grid index to sub-slice, suitable for
    fetching via `build_get_slice_request` and reassembling.
    """
    chunk_bounds = tuple(
        tuple(itertools.accumulate(axis_chunks, initial=0)) for axis_chunks in chunks
    )
    return split_slice(
        arr_slice.expand_for_shape(shape),
        max_size=bytesize_limit // itemsize,
        pref_splits=chunk_bounds,
    )


def build_write_full_request(
    url_path: str, array: Any, persist: bool = True
) -> RequestSpec:
    "Build a request overwriting the full array, via `PUT /array/full`."
    if not (hasattr(array, "shape") and hasattr(array, "dtype")):
        array = numpy.asarray(array)
    params: dict[str, Any] = {}
    if persist is False:
        # Extend the query only for non-default behavior.
        params["persist"] = persist
    return RequestSpec(
        "PUT",
        url_path,
        params=params,
        headers={"Content-Type": OCTET_STREAM},
        content=array.tobytes(),
    )


def build_write_block_request(
    block_url_template: str,
    array: Any,
    block: tuple[int, ...],
    slice: Any = ...,
    persist: bool = True,
) -> RequestSpec:
    "Build a request overwriting one block of the array, via `PUT /array/block`."
    if not (hasattr(array, "shape") and hasattr(array, "dtype")):
        array = numpy.asarray(array)
    url_path = block_url_template.format(*block)
    params = {
        **parse_qs(urlparse(url_path).query),
        **params_from_slice(slice),
    }
    if persist is False:
        # Extend the query only for non-default behavior.
        params["persist"] = persist
    return RequestSpec(
        "PUT",
        url_path,
        params=params,
        headers={"Content-Type": OCTET_STREAM},
        content=array.tobytes(),
    )


def build_patch_request(
    url_path: str,
    array: NDArray[Any],
    offset: Union[int, tuple[int, ...]],
    dtype: numpy.dtype,
    extend: bool = False,
    persist: bool = True,
) -> RequestSpec:
    "Build a request writing into a slice of the array, via `PATCH /array/full`."
    if array.dtype != dtype:
        raise ValueError(
            f"Data given to patch has dtype {array.dtype} which does not "
            f"match the dtype of this array {dtype}."
        )
    array_ = numpy.ascontiguousarray(array)
    if isinstance(offset, int):
        offset = (offset,)
    params: dict[str, Any] = {
        **parse_qs(urlparse(url_path).query),
        "offset": ",".join(map(str, offset)),
        "shape": ",".join(map(str, array_.shape)),
        "extend": bool(extend),
    }
    if persist is False:
        # Extend the query only for non-default behavior.
        params["persist"] = persist
    return RequestSpec(
        "PATCH",
        url_path,
        params=params,
        headers={"Content-Type": OCTET_STREAM},
        content=array_.tobytes(),
    )


def patch_conflict_message(body: dict[str, Any]) -> str:
    "Render the server's patch-conflict response body as a client-side message."
    return body.get("detail", "Array parameters conflict.").replace(
        "Use ?",  # URL query param
        "Pass keyword argument ",  # Python function argument
    )


def parse_patch_response(
    body: dict[str, Any], structure_family: StructureFamily
) -> Any:
    "Parse the updated structure returned by a successful patch request."
    structure_type = STRUCTURE_TYPES[structure_family]
    return structure_type.from_json(body)
