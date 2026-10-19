"""Smoke test for the experimental sans-I/O async array client.

Writes an array with the synchronous client against an in-process app, then
exercises the asynchronous client (`AsyncContext` + `AsyncArrayClient`):
block-wise and full reads, overwriting (full and block-wise), and patching
(extending) the array.
"""

import dask.array
import numpy
import pytest

from tiled.catalog import from_uri
from tiled.client import Context, from_context
from tiled.client._async_array import AsyncArrayClient
from tiled.client._async_context import AsyncContext
from tiled.config import Authentication
from tiled.ndslice import NDBlock
from tiled.server.app import build_app


@pytest.fixture
def app(tmp_path):
    "An in-process app over a file-backed catalog (so data survives restarts)."
    catalog = from_uri(
        f"sqlite:///{tmp_path / 'catalog.db'}",
        writable_storage=str(tmp_path / "data"),
        init_if_not_exists=True,
    )
    return build_app(
        catalog, authentication=Authentication(single_user_api_key="secret")
    )


async def test_async_array_experiment(app):
    original = numpy.arange(24, dtype="float64").reshape((4, 6))

    # Write the array with the SYNC client. `Context.from_app` runs the app's
    # lifespan (initializing the catalog) for the duration of the `with` block.
    with Context.from_app(app) as context:
        client = from_context(context)
        client.write_array(dask.array.from_array(original, chunks=(2, 6)), key="x")

    # Now use the ASYNC client. The sync Context above shut the app's lifespan
    # down on exit, so `AsyncContext` runs the lifespan afresh, in this event
    # loop, via `app.router.lifespan_context`.
    async with AsyncContext.from_app(app) as async_context:
        client = await AsyncArrayClient.from_path(async_context, "x")
        assert client.shape == original.shape
        assert client.dtype == original.dtype
        assert client.chunks == ((2, 2), (6,))

        # Read the full array back, and a slice of it.
        numpy.testing.assert_array_equal(await client.read(), original)
        numpy.testing.assert_array_equal(
            await client.read(numpy.s_[1:3, ::2]), original[1:3, ::2]
        )

        # Read it back block by block.
        for block in NDBlock(...).chunk_indices(client.chunks):
            expected = original[NDBlock(block).slice_from_chunks(client.chunks)]
            numpy.testing.assert_array_equal(await client.read_block(block), expected)
        # Read a slice within one block.
        numpy.testing.assert_array_equal(
            await client.read_block((1, 0), numpy.s_[:1, 2:4]),
            original[2:3, 2:4],
        )

        # Force the multi-request concurrent read path (asyncio.gather).
        client.RESPONSE_BYTESIZE_LIMIT = original[0].nbytes
        numpy.testing.assert_array_equal(await client.read(), original)
        numpy.testing.assert_array_equal(
            await client.read(numpy.s_[::2, :]), original[::2, :]
        )
        client.RESPONSE_BYTESIZE_LIMIT = AsyncArrayClient.RESPONSE_BYTESIZE_LIMIT

        # Overwrite the full array and read it back.
        overwritten = original * -1
        await client.write(overwritten)
        numpy.testing.assert_array_equal(await client.read(), overwritten)

        # Overwrite one block and read it back.
        block_data = numpy.full((2, 6), 99.0)
        await client.write_block(block_data, block=(0, 0))
        numpy.testing.assert_array_equal(
            await client.read(),
            numpy.concatenate([block_data, overwritten[2:]]),
        )

        # Patch: extend the array along the leading axis.
        extension = numpy.ones((2, 6))
        await client.patch(extension, offset=(4,), extend=True)
        assert client.shape == (6, 6)  # structure updated from the response
        numpy.testing.assert_array_equal(
            await client.read(),
            numpy.concatenate([block_data, overwritten[2:], extension]),
        )

        # Patch without `extend` where extension is required raises ValueError.
        with pytest.raises(ValueError):
            await client.patch(extension, offset=(6,), extend=False)

        # Patch with a mismatched dtype is rejected client-side.
        with pytest.raises(ValueError):
            await client.patch(
                numpy.ones((2, 6), dtype="int32"), offset=(6,), extend=True
            )

    # Finally, confirm with the SYNC client that the async writes persisted.
    with Context.from_app(app) as context:
        result = from_context(context)["x"].read()
    numpy.testing.assert_array_equal(
        result, numpy.concatenate([block_data, overwritten[2:], extension])
    )
