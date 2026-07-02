"""Smoke test for the cpp-zarr Python wheel.

Writes small random volumes, reads them back, and checks the data survives the
round trip -- across dtypes, compressors, both storage orders (F and C), plus a
region (cropped) read. Run automatically by cibuildwheel against the freshly
built+installed wheel, so it also verifies the wheel imports and bundles its
native library. Exits non-zero on any failure.
"""
import os
import sys
import tempfile

import numpy as np
import cppzarr


def main():
    rng = np.random.default_rng(1234567)
    shape = (40, 24, 18)  # d0, d1, d2 (non-chunk-aligned -> exercises partial chunks)
    chunks = [16, 16, 16]
    dtypes = [np.uint8, np.uint16, np.float32, np.float64]
    # (compressor, order) combinations. zstd/F is our most-used combo; the others
    # add lz4, gzip, and a C-order case (which exercises the read transpose).
    combos = [("zstd", "F"), ("lz4", "F"), ("zstd", "C"), ("gzip", "F")]

    tmp = tempfile.mkdtemp()
    ok = True
    for dt in dtypes:
        name = np.dtype(dt).name
        if np.issubdtype(dt, np.integer):
            data = rng.integers(0, 200, size=shape, endpoint=True).astype(dt)
        else:
            data = (rng.standard_normal(shape) * 1000).astype(dt)

        for cname, order in combos:
            path = os.path.join(tmp, f"rt_{name}_{cname}_{order}.zarr")
            cppzarr.write_zarr(path, data, cname=cname, order=order, chunks=chunks)
            back = cppzarr.read_zarr(path)
            good = (back.dtype == data.dtype and back.shape == data.shape
                    and np.array_equal(back, data))
            print(f"{name:8s} {cname:8s} {order}  {'OK' if good else 'FAIL'}")
            ok = ok and good

        # Region (cropped) read: first half along d0.
        h = shape[0] // 2
        path = os.path.join(tmp, f"rt_{name}_lz4_F.zarr")
        sub = cppzarr.read_zarr(path, start_coords=[0, 0, 0], end_coords=[h, shape[1], shape[2]])
        rgood = sub.shape == (h, shape[1], shape[2]) and np.array_equal(sub, data[0:h])
        print(f"{name:8s} region     {'OK' if rgood else 'FAIL'}")
        ok = ok and rgood

    if not ok:
        sys.exit("cppzarr Python round-trip test FAILED")
    print("cppzarr Python round-trip tests PASSED")


if __name__ == "__main__":
    main()
