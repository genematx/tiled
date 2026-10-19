"""Smoke test for the experimental async array client.

Writes an array with the regular sync client against an in-process app,
then exercises the async path (`AsyncContext` + `AsyncArrayClient`,
driven by the sans-I/O `requestor` generators): block-wise and full
reads, concurrent multi-request reads, overwrites, and `patch`.
"""

import dask.array
import numpy
import pytest

from tiled.catalog import from_uri
from tiled.client import Context, from_context
from tiled.client._async_context import AsyncContext
from tiled.client.array import AsyncArrayClient
from tiled.server.app import build_app


@pytest.fixture
def app(tmp_path):
    # A file-backed catalog, so that data written during the sync phase
    # survives the app's lifespan restarting for the async phase.
    catalog = from_uri(
        f"sqlite:///{tmp_path / 'catalog.db'}",
        writable_storage=[f"file://localhost{tmp_path / 'data'}"],
        init_if_not_exists=True,
    )
    return build_app(catalog)


async def test_async_array_experiment(app):
    original = numpy.arange(24, dtype="float64").reshape((4, 6))

    # Phase 1: write with the SYNC client against the in-process app.
    with Context.from_app(app) as context:
        client = from_context(context)
        client.write_array(dask.array.from_array(original, chunks=(2, 6)), key="x")
        api_key = context.api_key

    # Phase 2: read back, overwrite, and patch with the ASYNC path.
    # `AsyncContext.__aenter__` restarts the app's lifespan, now on this
    # event loop, because `httpx.ASGITransport` does not run it.
    async with AsyncContext.from_app(app, api_key=api_key) as async_context:
        # Count requests so we can verify the concurrent split below.
        requests = []

        async def record(request):
            requests.append(request)

        async_context.http_client.event_hooks["request"].append(record)

        ac = await AsyncArrayClient.from_context(async_context, "x")
        assert ac.shape == (4, 6)
        assert ac.dtype == original.dtype
        assert ac.ndim == 2

        # Read the full array (small enough for a single request).
        requests.clear()
        full = await ac.read()
        assert numpy.array_equal(full, original)
        assert len(requests) == 1

        # Read block-wise, with and without an intra-block slice.
        assert numpy.array_equal(await ac.read_block((0, 0)), original[:2])
        assert numpy.array_equal(await ac.read_block((1, 0)), original[2:])
        assert numpy.array_equal(
            await ac.read_block((1, 0), slice=numpy.s_[0:1, 0:3]),
            original[2:3, 0:3],
        )
        with pytest.raises(IndexError):
            await ac.read_block((5, 0))

        # Read a slice.
        assert numpy.array_equal(await ac.read(numpy.s_[1:3, 2:5]), original[1:3, 2:5])

        # Force the read to split into multiple concurrent requests.
        ac.RESPONSE_BYTESIZE_LIMIT = original.nbytes // 4
        requests.clear()
        split = await ac.read()
        assert numpy.array_equal(split, original)
        assert len(requests) > 1
        ac.RESPONSE_BYTESIZE_LIMIT = AsyncArrayClient.RESPONSE_BYTESIZE_LIMIT

        # Overwrite the whole array.
        overwritten = original * 2
        await ac.write(overwritten)
        assert numpy.array_equal(await ac.read(), overwritten)

        # Overwrite one block.
        block_data = numpy.full((2, 6), -1.0)
        await ac.write_block(block_data, block=(1, 0))
        expected = numpy.concatenate([overwritten[:2], block_data])
        assert numpy.array_equal(await ac.read(), expected)

        # Patch: extend the array along the leading axis.
        extension = numpy.full((2, 6), 7.0)
        await ac.patch(extension, offset=(4,), extend=True)
        assert ac.shape == (6, 6)
        extended = numpy.concatenate([expected, extension])
        assert numpy.array_equal(await ac.read(), extended)

        # Patching with a mismatched dtype raises client-side.
        with pytest.raises(ValueError):
            await ac.patch(numpy.ones((1, 6), dtype="int32"), offset=(0,))

        # Patching out of bounds without `extend` surfaces the server's
        # explanation as a ValueError (translated inside the requestor).
        with pytest.raises(ValueError, match="extend"):
            await ac.patch(numpy.full((2, 6), 9.0), offset=(6,))
