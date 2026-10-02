"""Smoke test for the cpp-zarr Python wheel.

Writes small random volumes, reads them back, and checks the data survives the
round trip -- across dtypes, compressors, both storage orders (F and C), plus a
region (cropped) read, arrays of 0 to 40 dimensions with region reads and crop
writes, and arrays in any memory layout (C order, F order, strided views) read
back in either order, and test arrays written by zarr-python 3 and TensorStore
(tests/test_arrays: Zarr v3, and v2 with other codecs and fill values). Run
automatically by cibuildwheel against the freshly
built+installed wheel, so it also verifies the wheel imports and bundles its
native library. Exits non-zero on any failure.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import cppzarr


def main():
    rng = np.random.default_rng(1234567)
    shape = (40, 24, 18)  # d0, d1, d2 (non-chunk-aligned -> exercises partial chunks)
    chunks = [16, 16, 16]
    dtypes = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32,
              np.uint64, np.int64, np.float32, np.float64]
    # (compressor, order) combinations. zstd/F is our most-used combo; the others
    # add lz4, gzip, and a C-order case (which exercises the read transpose).
    combos = [("zstd", "F"), ("lz4", "F"), ("zstd", "C"), ("gzip", "F")]

    tmp = tempfile.mkdtemp()
    ok = True
    for dt in dtypes:
        name = np.dtype(dt).name
        if np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
            lo, hi = max(info.min, -1000), min(info.max, 1000)  # negatives exercise signed types
            data = rng.integers(lo, hi, size=shape, endpoint=True).astype(dt)
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

    # Arrays of other numbers of dimensions (1D to 5D), both orders: region reads
    # and a crop write
    for shp, chks in [((1000,), [128]), ((300, 170), [128, 64]), ((5, 30, 40, 50), [2, 16, 16, 32]),
                      ((2, 3, 20, 30, 40), [1, 2, 16, 16, 32])]:
        data = rng.integers(0, 60000, size=shp).astype(np.uint16)
        for order in "FC":
            path = os.path.join(tmp, f"nd{len(shp)}_{order}.zarr")
            cppzarr.write_zarr(path, data, order=order, chunks=chks)
            good = np.array_equal(cppzarr.read_zarr(path), data)
            s = [n // 4 for n in shp]
            e = [max(si + 1, 3 * n // 4) for si, n in zip(s, shp)]
            region = tuple(slice(si, ei) for si, ei in zip(s, e))
            good = good and np.array_equal(cppzarr.read_zarr(path, s, e), data[region])
            patch = (data[region] // 2).astype(np.uint16)
            cppzarr.write_zarr(path, patch, start_coords=s, end_coords=e)
            exp = data.copy()
            exp[region] = patch
            good = good and np.array_equal(cppzarr.read_zarr(path), exp)
            print(f"{len(shp)}D       {order}         {'OK' if good else 'FAIL'}")
            ok = ok and good

    # A 0-dimensional array (scalar)
    s0 = np.array(3.25)
    p0 = os.path.join(tmp, "nd0.zarr")
    cppzarr.write_zarr(p0, s0)
    b0 = cppzarr.read_zarr(p0)
    b0c = cppzarr.read_zarr(p0, order='C')
    good = b0.shape == () and b0.dtype == s0.dtype and b0 == s0 and b0c.shape == () and b0c == s0
    print(f"0D               {'OK' if good else 'FAIL'}")
    ok = ok and good

    # Many dimensions (NumPy 1 allows up to 32, NumPy 2 up to 64), both orders
    nd = 40 if int(np.__version__.split('.')[0]) >= 2 else 30
    shp = [1] * nd
    shp[1], shp[nd // 2], shp[-1] = 5, 3, 4
    chks = [1] * nd
    chks[1], chks[nd // 2], chks[-1] = 2, 2, 3
    data = rng.integers(0, 60000, size=shp).astype(np.uint16)
    for order in "FC":
        path = os.path.join(tmp, f"nd{nd}_{order}.zarr")
        cppzarr.write_zarr(path, data, order=order, chunks=chks)
        s = [0] * nd
        s[1] = 1
        e = list(shp)
        e[-1] = 3
        region = tuple(slice(si, ei) for si, ei in zip(s, e))
        good = np.array_equal(cppzarr.read_zarr(path), data) and np.array_equal(cppzarr.read_zarr(path, s, e), data[region])
        print(f"{nd}D      {order}         {'OK' if good else 'FAIL'}")
        ok = ok and good

    # Memory layouts: write_zarr reads C-order and F-order arrays and strided views
    # in place (reversed axes are copied first); read_zarr returns the file's own
    # order by default, or the order asked for; full reads, region reads and crop writes
    for shp, chks in [((300, 170), [128, 64]), ((40, 24, 18), [16, 16, 16]), ((5, 30, 40, 50), [2, 16, 16, 32])]:
        base = rng.integers(0, 60000, size=shp).astype(np.uint16)
        perm = (1, 0) + tuple(range(2, len(shp)))
        layouts = {'C': base, 'F': np.asfortranarray(base),
                   'strided': np.repeat(base, 2, axis=0)[::2],
                   'permuted': np.ascontiguousarray(base.transpose(perm)).transpose(perm),
                   'reversed': base[::-1]}
        s = [n // 4 for n in shp]
        e = [max(si + 1, 3 * n // 4) for si, n in zip(s, shp)]
        region = tuple(slice(si, ei) for si, ei in zip(s, e))
        for zorder in "FC":
            for name, data in layouts.items():
                path = os.path.join(tmp, f"layout_{len(shp)}d_{zorder}_{name}.zarr")
                cppzarr.write_zarr(path, data, order=zorder, chunks=chks)
                back = cppzarr.read_zarr(path)
                f_back, c_back = cppzarr.read_zarr(path, order='F'), cppzarr.read_zarr(path, order='C')
                good = (np.array_equal(back, data) and back.flags[f'{zorder}_CONTIGUOUS'] and
                        np.array_equal(f_back, data) and f_back.flags['F_CONTIGUOUS'] and
                        np.array_equal(c_back, data) and c_back.flags['C_CONTIGUOUS'])
                for o in (None, 'F', 'C'):
                    sub = cppzarr.read_zarr(path, s, e, order=o)
                    good = good and np.array_equal(sub, data[region]) and sub.flags[f'{o or zorder}_CONTIGUOUS']
                patch = (data[region] // 2).astype(np.uint16)
                cppzarr.write_zarr(path, patch, start_coords=s, end_coords=e)
                exp = np.array(data)
                exp[region] = patch
                good = good and np.array_equal(cppzarr.read_zarr(path, order='C'), exp)
                print(f"{len(shp)}D {zorder}-order zarr, {name:8s} input  {'OK' if good else 'FAIL'}")
                ok = ok and good
    try:
        cppzarr.read_zarr(path, order='X')
        good = False
    except Exception:
        good = True
    print(f"invalid order rejected  {'OK' if good else 'FAIL'}")
    ok = ok and good

    # A chunk rewritten as all zeros reads back as zeros (its old file is removed)
    zr = os.path.join(tmp, "zero_rewrite.zarr")
    d = rng.integers(1, 1000, size=(40, 24, 18)).astype(np.uint16)
    cppzarr.write_zarr(zr, d, chunks=[16, 16, 16])
    cppzarr.write_zarr(zr, np.zeros_like(d), chunks=[16, 16, 16])
    good = not cppzarr.read_zarr(zr).any()
    print(f"zero rewrite     {'OK' if good else 'FAIL'}")
    ok = ok and good

    # 3-value coordinates and chunks keep working on a 2D array
    d2 = rng.integers(0, 60000, size=(70, 45)).astype(np.uint16)
    p2 = os.path.join(tmp, "nd2_3value_args.zarr")
    cppzarr.write_zarr(p2, d2, chunks=[32, 16, 16])
    good = np.array_equal(cppzarr.read_zarr(p2, [3, 5, 0], [40, 30, 1]), d2[3:40, 5:30])
    print(f"2D 3-value args  {'OK' if good else 'FAIL'}")
    ok = ok and good

    # Test arrays written by zarr-python 3 and TensorStore (tests/test_arrays): Zarr v3
    # arrays, and v2 arrays with codecs and fill values cpp-zarr does not write itself.
    # Reads in every order and a region read match what zarr-python reads; arrays
    # cpp-zarr cannot read are rejected; writing into a v3 array is refused and changes
    # nothing
    v3dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'tests', 'test_arrays')
    if not os.path.isfile(os.path.join(v3dir, 'arrays.json')):
        # made by tests/make_test_arrays.py; CI makes them and requires them
        required = bool(os.environ.get('CPPZARR_REQUIRE_TEST_ARRAYS'))
        print(f"test arrays not found, {'FAIL' if required else 'skipped'} (make them with tests/make_test_arrays.py)")
        ok = ok and not required
    else:
        with open(os.path.join(v3dir, 'arrays.json')) as f:
            fixtures = json.load(f)['arrays']
        for fx in fixtures:
            path = os.path.join(v3dir, fx['name'] + '.zarr')
            if 'error' in fx:
                try:
                    cppzarr.read_zarr(path)
                    good = False
                except Exception as e:
                    good = fx['error'] in str(e)
                print(f"test array {fx['name']:36s} rejected  {'OK' if good else 'FAIL'}")
                ok = ok and good
                continue
            exp = np.fromfile(os.path.join(v3dir, fx['name'] + '.bin'), dtype=fx['dtype']).reshape(fx['shape'])
            good = True
            for o in (None, 'F', 'C'):
                a = cppzarr.read_zarr(path, order=o)
                good = good and a.dtype == exp.dtype and np.array_equal(a, exp, equal_nan=exp.dtype.kind == 'f')
            if exp.ndim:
                s = [1 if n > 2 else 0 for n in exp.shape]
                e = [n - 1 if n > 2 else n for n in exp.shape]
                region = tuple(slice(a, b) for a, b in zip(s, e))
                good = good and np.array_equal(cppzarr.read_zarr(path, s, e), exp[region], equal_nan=exp.dtype.kind == 'f')
            print(f"test array {fx['name']:36s}           {'OK' if good else 'FAIL'}")
            ok = ok and good
        copy = os.path.join(tmp, 'v3_write.zarr')
        shutil.copytree(os.path.join(v3dir, 'v3_blosc_zstd_uint16.zarr'), copy)
        before = sorted(os.path.relpath(os.path.join(dp, f), copy) for dp, _, fs in os.walk(copy) for f in fs)
        good = True
        for crop in (False, True):
            try:
                if crop:
                    cppzarr.write_zarr(copy, np.ones((2, 2, 2), np.uint16), start_coords=[0, 0, 0], end_coords=[2, 2, 2])
                else:
                    cppzarr.write_zarr(copy, np.ones((10, 7, 5), np.uint16))
                good = False
            except Exception as e:
                good = good and 'Zarr v3' in str(e)
        after = sorted(os.path.relpath(os.path.join(dp, f), copy) for dp, _, fs in os.walk(copy) for f in fs)
        good = good and before == after
        print(f"writes into a Zarr v3 array refused  {'OK' if good else 'FAIL'}")
        ok = ok and good

    if not ok:
        sys.exit("cppzarr Python round-trip test FAILED")
    print("cppzarr Python round-trip tests PASSED")


if __name__ == "__main__":
    main()
