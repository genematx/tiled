import builtins
import concurrent.futures
import itertools
import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
from urllib.parse import parse_qs, urlparse

import dask
import dask.array
import httpx
import numpy
from numpy.typing import NDArray

from ..ndslice import NDBlock, NDSlice, split_slice
from ..structures.array import ArrayStructure
from ..structures.core import STRUCTURE_TYPES
from ._requestor import Requestor, TiledRequest, send_requests, send_requests_async
from .base import BaseClient
from .utils import (
    ClientError,
    chunks_repr,
    export_util,
    params_from_slice,
    slices_to_dask_chunks,
)

if TYPE_CHECKING:
    from ._async_context import AsyncContext
    from .stream import ArraySubscription

OCTET_STREAM = "application/octet-stream"


def _get_block_requestor(
    item: Dict[str, Any],
    structure: ArrayStructure,
    block: NDBlock,
    block_slice: Optional[NDSlice] = None,
) -> Requestor[NDArray]:
    """Requestor generator that fetches one block via the `/array/block` endpoint.

    Shared client logic, driven by either `send_requests` (sync) or
    `send_requests_async` (async); see `tiled.client._requestor`.
    """
    chunks = structure.chunks
    dtype = structure.data_type.to_numpy_dtype()

    # Determine the expected shape of the resulting array after slicing
    exp_shape = []
    # Expand the block to convert for URL
    block = block.expand_for_shape([len(dim) for dim in chunks])
    if shape := block.shape_from_chunks(chunks):
        exp_shape = block_slice.shape_after_slice(shape) if block_slice else shape
        # Check for special case of shape with 0 in it.
        if 0 in exp_shape:
            # This is valid, and it has come up in the wild.  An array with
            # 0 as one of the dimensions never contains data, so we can
            # short-circuit here without any further information from the
            # service.
            return numpy.array([], dtype=dtype).reshape(exp_shape)

    url_path = item["links"]["block"]
    params = {
        **parse_qs(urlparse(url_path).query),
        "block": block.to_numpy_str(),
        "expected_shape": ",".join(map(str, exp_shape)) or "scalar",
    }
    params = params | ({"slice": block_slice.to_numpy_str()} if block_slice else {})
    response = yield TiledRequest(
        "GET", url_path, params=params, headers={"Accept": OCTET_STREAM}
    )
    return numpy.frombuffer(response.read(), dtype=dtype).reshape(exp_shape)


def _get_slice_requestor(
    item: Dict[str, Any],
    structure: ArrayStructure,
    array_slice: NDSlice,
) -> Requestor[NDArray]:
    """Requestor generator that fetches a slice via the `/array/full` endpoint.

    Pass `NDSlice()` as `array_slice` to fetch the whole array.
    """
    shape = structure.shape
    dtype = structure.data_type.to_numpy_dtype()

    # Determine the expected shape of the resulting array after slicing
    exp_shape = array_slice.shape_after_slice(shape) if array_slice else shape

    # Check for special case of shape with 0 in it.
    if 0 in exp_shape:
        # This is valid, and it has come up in the wild.  An array with
        # 0 as one of the dimensions never contains data, so we can
        # short-circuit here without any further information from the service.
        return numpy.array([], dtype=dtype).reshape(exp_shape)

    url_path = item["links"]["full"]
    params = {
        **parse_qs(urlparse(url_path).query),
        "expected_shape": ",".join(map(str, exp_shape)) or "scalar",
    }
    params = params | ({"slice": array_slice.to_numpy_str()} if array_slice else {})
    response = yield TiledRequest(
        "GET", url_path, params=params, headers={"Accept": OCTET_STREAM}
    )
    return numpy.frombuffer(response.read(), dtype=dtype).reshape(exp_shape)


def _read_requestor(
    item: Dict[str, Any],
    structure: ArrayStructure,
    array_slice: NDSlice,
    bytesize_limit: int,
) -> Requestor[NDArray]:
    """Requestor generator that fetches the entire array or its slice.

    Splits reads larger than `bytesize_limit` into multiple sub-requests
    -- yielded as a list of sub-requestors so the driver may run them
    concurrently -- and reassembles the results. Mirrors the splitting
    logic in `_DaskArrayClient.read` (which instead recombines with dask).
    """
    shape = structure.shape
    dtype = structure.data_type.to_numpy_dtype()

    # Determine the expected shape of the resulting array after slicing
    if array_slice:
        array_slice = array_slice.expand_for_shape(shape)  # Remove "..."
    exp_shape = array_slice.shape_after_slice(shape)
    total_bytes = math.prod(exp_shape) * dtype.itemsize

    # Check for special case of shape with 0 in it.
    if 0 in exp_shape:
        return numpy.array([], dtype=dtype).reshape(exp_shape)

    # If the expected response is small, fetch it in one go.
    if total_bytes < bytesize_limit:
        return (yield from _get_slice_requestor(item, structure, array_slice))

    # The response is expected to be large; subslice it and recombine.
    # Build chunk boundaries along each axis to find best candidate split points
    chunk_bounds = tuple(
        tuple(itertools.accumulate(axis_chunks, initial=0))
        for axis_chunks in structure.chunks
    )
    indexed_slices = split_slice(
        array_slice.expand_for_shape(shape),
        max_size=bytesize_limit // dtype.itemsize,
        pref_splits=chunk_bounds,
    )
    indices = list(indexed_slices)
    blocks = yield [
        _get_slice_requestor(item, structure, indexed_slices[index])
        for index in indices
    ]
    # Reassemble the sub-slices into a single array. The sub-slices tile the
    # result in a regular grid, so each one's offset in the result follows
    # from the per-axis chunk sizes.
    offsets = tuple(
        tuple(itertools.accumulate(axis_chunks, initial=0))
        for axis_chunks in slices_to_dask_chunks(indexed_slices, shape)
    )
    result = numpy.empty(exp_shape, dtype=dtype)
    for index, block in zip(indices, blocks):
        position = tuple(
            builtins.slice(offsets[axis][i], offsets[axis][i] + block.shape[axis])
            for axis, i in enumerate(index)
        )
        result[position] = block
    return result


def _write_requestor(
    item: Dict[str, Any], array: NDArray, persist: bool
) -> Requestor[None]:
    "Requestor generator that overwrites the full array."
    params: Dict[str, Any] = {}
    if persist is False:
        # Extend the query only for non-default behavior.
        params["persist"] = persist
    yield TiledRequest(
        "PUT",
        item["links"]["full"],
        params=params,
        headers={"Content-Type": OCTET_STREAM},
        content=array.tobytes(),
    )


def _write_block_requestor(
    item: Dict[str, Any],
    array: NDArray,
    block: Tuple[int, ...],
    slice: Any,
    persist: bool,
) -> Requestor[None]:
    "Requestor generator that overwrites one block of the array."
    url_path = item["links"]["block"].format(*block)
    params = {
        **parse_qs(urlparse(url_path).query),
        **params_from_slice(slice),
    }
    if persist is False:
        # Extend the query only for non-default behavior.
        params["persist"] = persist
    yield TiledRequest(
        "PUT",
        url_path,
        params=params,
        headers={"Content-Type": OCTET_STREAM},
        content=array.tobytes(),
    )


def _patch_requestor(
    item: Dict[str, Any],
    array: NDArray,
    offset: Tuple[int, ...],
    extend: bool,
    persist: bool,
) -> Requestor[Dict[str, Any]]:
    """Requestor generator that writes into a slice, maybe extending the shape.

    Returns the updated structure, as JSON, reported by the server.
    """
    url_path = item["links"]["full"]
    params = {
        **parse_qs(urlparse(url_path).query),
        "offset": ",".join(map(str, offset)),
        "shape": ",".join(map(str, array.shape)),
        "extend": bool(extend),
    }
    if persist is False:
        # Extend the query only for non-default behavior.
        params["persist"] = persist
    try:
        response = yield TiledRequest(
            "PATCH",
            url_path,
            params=params,
            headers={"Content-Type": OCTET_STREAM},
            content=array.tobytes(),
        )
    except ClientError as err:
        if err.response.status_code in [
            httpx.codes.BAD_REQUEST,
            httpx.codes.CONFLICT,
        ]:
            raise ValueError(
                err.response.json()
                .get("detail", "Array parameters conflict.")
                .replace(
                    "Use ?",  # URL query param
                    "Pass keyword argument ",  # Python function argument
                )
            ) from None
        raise
    return response.json()


def _metadata_requestor(url: str) -> Requestor[Dict[str, Any]]:
    "Requestor generator that fetches an item's metadata document."
    response = yield TiledRequest("GET", url, headers={"Accept": "application/json"})
    return response.json()


class _DaskArrayClient(BaseClient):
    "Client-side wrapper around an array-like that returns dask arrays"

    # The limit on the expected size of the response body (before compression).
    # This will be used to determine how to combine multiple requests when fetching
    # data in blocks. If set to None, the client will not attempt to combine
    # requests and will fetch each chunk separately as determiied by the structure.
    RESPONSE_BYTESIZE_LIMIT = 100 * 1024 * 1024  # 100 MiB

    def __init__(self, *args, item, **kwargs):
        super().__init__(*args, item=item, **kwargs)

    @property
    def dims(self):
        return self.structure().dims

    @property
    def shape(self):
        return self.structure().shape

    @property
    def size(self):
        return numpy.prod(self.structure().shape)

    @property
    def dtype(self):
        return self.structure().data_type.to_numpy_dtype()

    @property
    def nbytes(self):
        structure = self.structure()
        itemsize = structure.data_type.to_numpy_dtype().itemsize
        return numpy.prod(structure.shape) * itemsize

    @property
    def chunks(self):
        return self.structure().chunks

    @property
    def ndim(self):
        return len(self.structure().shape)

    def __repr__(self):
        attrs = {
            "shape": self.shape,
            "chunks": chunks_repr(self.chunks),
            "dtype": self.dtype,
        }
        if dims := self.structure().dims:
            attrs["dims"] = dims
        return (
            f"<{type(self).__name__}"
            + "".join(f" {k}={v}" for k, v in attrs.items())
            + ">"
        )

    def __array__(self, *args, **kwargs):
        return self.read().__array__(*args, **kwargs)

    def _get_block(self, block: NDBlock, block_slice: Optional[NDSlice] = None):
        """Fetch the data for one chunk (block) in a chunked array.

        This private method is used internally by the client and requires the
        `block` and `block_slice` arguments to be pre-cast as NDBlock and NDSlice
        types, respectively.

        This method uses the `/array/block` endpoint to fetch one block of the array,
        at a time. The block boundaries are determined by the structure of the array,
        and usually correspond to the chunking of the array on the server side.

        See read_block() for a public version of this.

        Parameters
        ----------
        block : NDBlock
            The chunk index, e.g. (0, 0), (0, 1), (0, 2) .... for a 2D array
            chunked into 3 blocks.
        block_slice : NDSlice, optional
            A slice within this block to return.
        """
        result = send_requests(
            _get_block_requestor(self.item, self.structure(), block, block_slice),
            self.context,
        )
        if (ps := self.context.progress_state) is not None:
            ps.advance()
        return result

    def _get_slice(self, slice: NDSlice):
        """Fetch the data for a slice of the full array

        This private method is used internally by the client and requires the
        `slice` argument to be pre-cast as an NDSlice type.

        The request is made to the `/array/full` endpoint.

        See read() for a public version of this.

        Parameters
        ----------
        slice : NDSlice
            A slice of the array to return, pass NDSlice() to return the whole array.
        """
        result = send_requests(
            _get_slice_requestor(self.item, self.structure(), slice),
            self.context,
        )
        if (ps := self.context.progress_state) is not None:
            ps.advance()
        return result

    def read_block(self, block, slice=None):
        """Access the data for one block of this chunked (dask) array.

        This method uses the `/array/block` endpoint to fetch one block of the array,
        at a time. The block boundaries are determined by the structure of the array,
        and usually correspond to the chunking of the array on the server side.

        Optionally, access only a slice *within* this block.
        """

        block, block_slice = NDBlock(block), NDSlice(slice)

        try:
            shape = block.shape_from_chunks(self.chunks)
        except IndexError:
            raise IndexError(f"Block index {block} out of range")

        exp_shape = block_slice.shape_after_slice(shape) if block_slice else shape
        dask_array = dask.array.from_delayed(
            dask.delayed(self._get_block)(block, block_slice),
            dtype=self.dtype,
            shape=exp_shape,
        )
        return dask_array

    def fetch_count(self, slice=None) -> int:
        """Return the number of HTTP requests that read(slice) will make.

        Mirrors the split logic in read() so that callers (e.g. CompositeClient)
        can compute an accurate progress-bar total without triggering any fetches.
        """
        arr_slice = NDSlice(slice).expand_for_shape(self.shape)
        exp_shape = arr_slice.shape_after_slice(self.shape)
        if 0 in exp_shape:
            return 0
        total_bytes = math.prod(exp_shape) * self.dtype.itemsize
        if total_bytes < self.RESPONSE_BYTESIZE_LIMIT:
            return 1
        chunk_bounds = tuple(
            tuple(itertools.accumulate(axis_chunks, initial=0))
            for axis_chunks in self.chunks
        )
        indexed_slices = split_slice(
            arr_slice,
            max_size=self.RESPONSE_BYTESIZE_LIMIT // self.dtype.itemsize,
            pref_splits=chunk_bounds,
        )
        return len(indexed_slices)

    def read(self, slice=None):
        """Access the entire array or its slice

        The array will be internally chunked with dask.
        """

        # Determine the expected shape of the resulting array after slicing
        if arr_slice := NDSlice(slice):
            arr_slice = arr_slice.expand_for_shape(self.shape)  # Remove "..."
        exp_shape = arr_slice.shape_after_slice(self.shape)
        total_bytes = math.prod(exp_shape) * self.dtype.itemsize

        # Check for special case of shape with 0 in it.
        if 0 in exp_shape:
            # This is valid, and it has come up in the wild.  An array with
            # 0 as one of the dimensions never contains data, so we can
            # short-circuit here without any further information from the service.
            return dask.array.array([], dtype=self.dtype).reshape(exp_shape)

        # If the expected response is small, fetch it in one go.
        if total_bytes < self.RESPONSE_BYTESIZE_LIMIT:
            dask_array = dask.array.from_delayed(
                dask.delayed(self._get_slice)(arr_slice),
                dtype=self.dtype,
                shape=exp_shape,
            )
            return dask_array

        # The response is expected to be large, subslice it and recombine with dask
        # Build chunk boundaries along each axis to find best candidate split points
        chunk_bounds = tuple(
            tuple(itertools.accumulate(axis_chunks, initial=0))
            for axis_chunks in self.chunks
        )
        indexed_slices = split_slice(
            arr_slice.expand_for_shape(self.shape),
            max_size=self.RESPONSE_BYTESIZE_LIMIT // self.dtype.itemsize,
            pref_splits=chunk_bounds,
        )

        # Build a client-side dask array whose chunks correspond to subsplits of the slice
        name = "remote-dask-array-" f"{self.uri}"
        dask_tasks = {
            (name,) + indx: (self._get_slice, slc)
            for indx, (slc) in indexed_slices.items()
        }
        dask_array = dask.array.Array(
            name=name,
            dask=dask_tasks,
            dtype=self.dtype,
            chunks=slices_to_dask_chunks(indexed_slices, self.shape),
            shape=exp_shape,
        )
        return dask_array

    def write(self, array, persist=True):
        if not (hasattr(array, "shape") and hasattr(array, "dtype")):
            array = numpy.asarray(array)
        send_requests(_write_requestor(self.item, array, persist), self.context)

    def write_block(self, array, block, slice=..., persist=True):
        if not (hasattr(array, "shape") and hasattr(array, "dtype")):
            array = numpy.asarray(array)
        send_requests(
            _write_block_requestor(self.item, array, block, slice, persist),
            self.context,
        )

    def patch(
        self,
        array: NDArray,
        offset: Union[int, tuple[int, ...]],
        extend=False,
        persist=True,
    ):
        """
        Write data into a slice of an array, maybe extending the shape.

        Parameters
        ----------
        array : array-like
            The data to write
        offset : tuple[int, ...]
            Where to place this data in the array
        extend : bool
            Extend the array shape to fit the new slice, if necessary
        persist : bool | None
            Persist the changes on server storage if True. [default behavior]
            If False, the update is still streamed to subscribed listeners.

        Examples
        --------

        Create a (3, 2, 2) array of ones.

        >>> ac = c.write_array(numpy.ones((3, 2, 2)), key='y')
        >>> ac
        <ArrayClient shape=(3, 2, 2) chunks=((3,), (2,), (2,)) dtype=float64>

        Read it.

        >>> ac.read()
        array([[[1., 1.],
                [1., 1.]],

               [[1., 1.],
                [1., 1.]],

               [[1., 1.],
                [1., 1.]]])

        Extend the array by concatenating a (1, 2, 2) array of zeros.

        >>> ac.patch(numpy.zeros((1, 2, 2)), offset=(3,), extend=True)

        Read it.

        >>> array([[[1., 1.],
                    [1., 1.]],

                   [[1., 1.],
                    [1., 1.]],

                   [[1., 1.],
                    [1., 1.]],

                   [[0., 0.],
                    [0., 0.]]])
        """
        if array.dtype != self.dtype:
            raise ValueError(
                f"Data given to patch has dtype {array.dtype} which does not "
                f"match the dtype of this array {self.dtype}."
            )
        array_ = numpy.ascontiguousarray(array)
        if isinstance(offset, int):
            offset = (offset,)
        new_structure = send_requests(
            _patch_requestor(self.item, array_, offset, extend, persist),
            self.context,
        )
        # Update cached structure.
        structure_type = STRUCTURE_TYPES[self.structure_family]
        self._structure = structure_type.from_json(new_structure)

    def __getitem__(self, slice):
        return self.read(slice)

    # The default object.__iter__ works as expected here, no need to
    # implemented it specifically.

    def __len__(self):
        # As with numpy, len(arr) is the size of the zeroth axis.
        return self.structure().shape[0]

    def export(
        self, filepath, *, format=None, slice=None, link=None, template_vars=None
    ):
        """
        Download data in some format and write to a file.

        Parameters
        ----------
        file: str or buffer
            Filepath or writeable buffer.
        format : str, optional
            If format is None and `file` is a filepath, the format is inferred
            from the name, like 'table.csv' implies format="text/csv". The format
            may be given as a file extension ("csv") or a media type ("text/csv").
        slice : List[slice], optional
            List of slice objects. A convenient way to generate these is shown
            in the examples.
        link: str, optional
            Used internally. Refers to a key in the dictionary of links sent
            from the server.
        template_vars: dict, optional
            Used internally.

        Examples
        --------

        Export all.

        >>> a.export("numbers.csv")

        Export an N-dimensional slice.

        >>> import numpy
        >>> a.export("numbers.csv", slice=numpy.s_[:10, 50:100])
        """
        # For array, this is always 'full', but xarray clients set a custom link.
        if link is None:
            link = "full"
        template_vars = template_vars or {}
        params = params_from_slice(slice)
        return export_util(
            filepath,
            format,
            self.context.http_client.get,
            self.item["links"][link].format(**template_vars),
            params=params,
        )

    def subscribe(
        self,
        executor: Optional[concurrent.futures.Executor] = None,
    ) -> "ArraySubscription":
        """
        Subscribe to streaming updates about this array.

        Parameters
        ----------
        executor : concurrent.futures.Executor, optional
            Launches tasks asynchronously, in response to updates. By default,
            a concurrent.futures.ThreadPoolExecutor is used.

        Returns
        -------
        subscription : ArraySubscription
        """
        # Keep this import here to defer the websockets import until/unless needed.
        from .stream import ArraySubscription

        return ArraySubscription(self.context, self.path_parts, executor)


# Subclass with a public class that adds the dask-specific methods.


class DaskArrayClient(_DaskArrayClient):
    "Client-side wrapper around an array-like that returns dask arrays"

    def compute(self):
        "Alias to client.read().compute()"
        dask_arr = self.read()
        n = math.prod(len(c) for c in dask_arr.chunks)
        with self.context.tracking_progress(total=n):
            return dask_arr.compute()


class ArrayClient(DaskArrayClient):
    "Client-side wrapper around an array-like that returns in-memory arrays"

    def read(self, slice=None):
        """
        Access the entire array or a slice.
        """
        dask_arr = super().read(slice)
        n = self.fetch_count(slice)
        with self.context.tracking_progress(total=n):
            return dask_arr.compute()

    def read_block(self, block, slice=None):
        """
        Access the data for one block of this chunked array.

        Optionally, access only a slice *within* this block.
        """
        return super().read_block(block, slice).compute()


class AsyncArrayClient:
    """Asynchronous client for an array node, returning in-memory arrays.

    Drives the same requestor generators as the sync `ArrayClient`, via
    `send_requests_async`. Reads that span multiple requests are fetched
    concurrently with `asyncio.gather`, throttled by the `AsyncContext`
    semaphore. Experimental; the surface is intentionally minimal.
    """

    # See the comment on `_DaskArrayClient.RESPONSE_BYTESIZE_LIMIT`.
    RESPONSE_BYTESIZE_LIMIT = _DaskArrayClient.RESPONSE_BYTESIZE_LIMIT

    def __init__(
        self,
        context: "AsyncContext",
        item: Dict[str, Any],
        structure: Optional[ArrayStructure] = None,
    ):
        self.context = context
        self.item = item
        if structure is None:
            structure = ArrayStructure.from_json(item["attributes"]["structure"])
        self._structure = structure

    @classmethod
    async def from_context(
        cls, context: "AsyncContext", path: str
    ) -> "AsyncArrayClient":
        "Construct from one async GET to the `/metadata/{path}` endpoint."
        content = await send_requests_async(
            _metadata_requestor(f"{context.api_uri}metadata/{path.lstrip('/')}"),
            context,
        )
        return cls(context, content["data"])

    def __repr__(self):
        return (
            f"<{type(self).__name__} shape={self.shape} "
            f"chunks={chunks_repr(self.chunks)} dtype={self.dtype}>"
        )

    def structure(self) -> ArrayStructure:
        return self._structure

    @property
    def shape(self):
        return self._structure.shape

    @property
    def chunks(self):
        return self._structure.chunks

    @property
    def dtype(self):
        return self._structure.data_type.to_numpy_dtype()

    @property
    def ndim(self):
        return len(self._structure.shape)

    async def read(self, slice=None) -> NDArray:
        """
        Access the entire array or a slice.

        Large reads are split into multiple concurrent requests.
        """
        return await send_requests_async(
            _read_requestor(
                self.item, self._structure, NDSlice(slice), self.RESPONSE_BYTESIZE_LIMIT
            ),
            self.context,
        )

    async def read_block(self, block, slice=None) -> NDArray:
        """
        Access the data for one block of this chunked array.

        Optionally, access only a slice *within* this block.
        """
        block, block_slice = NDBlock(block), NDSlice(slice)
        try:
            block.shape_from_chunks(self.chunks)
        except IndexError:
            raise IndexError(f"Block index {block} out of range")
        return await send_requests_async(
            _get_block_requestor(self.item, self._structure, block, block_slice),
            self.context,
        )

    async def write(self, array, persist=True) -> None:
        if not (hasattr(array, "shape") and hasattr(array, "dtype")):
            array = numpy.asarray(array)
        await send_requests_async(
            _write_requestor(self.item, array, persist), self.context
        )

    async def write_block(self, array, block, slice=..., persist=True) -> None:
        if not (hasattr(array, "shape") and hasattr(array, "dtype")):
            array = numpy.asarray(array)
        await send_requests_async(
            _write_block_requestor(self.item, array, block, slice, persist),
            self.context,
        )

    async def patch(
        self,
        array: NDArray,
        offset: Union[int, tuple[int, ...]],
        extend=False,
        persist=True,
    ) -> None:
        """
        Write data into a slice of an array, maybe extending the shape.

        See `_DaskArrayClient.patch` for details and examples.
        """
        if array.dtype != self.dtype:
            raise ValueError(
                f"Data given to patch has dtype {array.dtype} which does not "
                f"match the dtype of this array {self.dtype}."
            )
        array_ = numpy.ascontiguousarray(array)
        if isinstance(offset, int):
            offset = (offset,)
        new_structure = await send_requests_async(
            _patch_requestor(self.item, array_, offset, extend, persist),
            self.context,
        )
        # Update cached structure.
        self._structure = ArrayStructure.from_json(new_structure)
