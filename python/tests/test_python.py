"""Smoke test for the cpp-zarr Python wheel.

Writes small random volumes, reads them back, and checks the data survives the
round trip -- across dtypes, compressors, both storage orders (F and C), plus a
region (cropped) read, arrays of 0 to 40 dimensions with region reads and crop
writes, and arrays in any memory layout (C order, F order, strided views) read
back in either order, and test arrays written by zarr-python 3 and TensorStore
(tests/test_arrays: Zarr v3, and v2 with other codecs and fill values), written into
and converted to Zarr v3. Zarr v3 writing: round trips, shards, crop writes, the
format of an existing array, conversion, and (when zarr-python and TensorStore are
installed) reading what cpp-zarr wrote with them. Run automatically by cibuildwheel
against the freshly built+installed wheel, so it also verifies the wheel imports and
bundles its native library. Exits non-zero on any failure.
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import cppzarr

# zarr-python and TensorStore read what cpp-zarr writes, when they are installed
try:
    import zarr
except ImportError:
    zarr = None
try:
    import tensorstore as ts
except ImportError:
    ts = None


def zarr_read(path):
    # The array as zarr-python reads it (None without zarr-python)
    return zarr.open_array(path, mode='r')[...] if zarr is not None else None


def ts_read(path):
    # The array as TensorStore reads it (None without TensorStore, and for Zarr v2)
    if ts is None or not os.path.isfile(os.path.join(path, 'zarr.json')):
        return None
    return ts.open({'driver': 'zarr3', 'kvstore': {'driver': 'file', 'path': path}}).result().read().result()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def same(a, b):
    # Equal arrays (NaN equal to NaN); None (a reader that is not installed) means
    # there is nothing to compare with
    if a is None or b is None:
        return True
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b, equal_nan=a.dtype.kind == 'f')


def values_like(rng, a):
    # Random values of a's dtype and shape (no NaNs)
    a = np.asarray(a)
    if a.dtype.kind == 'f':
        return np.asarray(rng.standard_normal(a.shape) * 1000, dtype=a.dtype)
    info = np.iinfo(a.dtype)
    return np.asarray(rng.integers(max(info.min, -10**6), min(info.max, 10**6), size=a.shape, endpoint=True), dtype=a.dtype)


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
    # cpp-zarr cannot read are rejected
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
        # Writes into copies of them: a region, then the whole array (as the array's
        # own type), read back by cpp-zarr and, when installed, zarr-python; arrays in
        # the opposite byte order are refused. Then the v2 ones converted to Zarr v3
        # (numcodecs zlib cannot be) read the same.
        for fx in fixtures:
            if 'error' in fx:
                continue
            exp = np.fromfile(os.path.join(v3dir, fx['name'] + '.bin'), dtype=fx['dtype']).reshape(fx['shape'])
            if exp.dtype.kind == 'b':      # (cpp-zarr does not write booleans)
                continue
            copy = os.path.join(tmp, 'ta_' + fx['name'] + '.zarr')
            shutil.copytree(os.path.join(v3dir, fx['name'] + '.zarr'), copy)
            meta_file = os.path.join(copy, 'zarr.json')
            big_endian = os.path.isfile(meta_file) and '"endian": "big"' in json.dumps(load_json(meta_file))
            good = True
            try:
                if exp.ndim:
                    s = [1 if n > 2 else 0 for n in exp.shape]
                    e = [n - 1 if n > 2 else n for n in exp.shape]
                    region = tuple(slice(a, b) for a, b in zip(s, e))
                    patch = values_like(rng, exp[region])
                    cppzarr.write_zarr(copy, patch, start_coords=s, end_coords=e)
                    exp[region] = patch
                    good = good and same(cppzarr.read_zarr(copy), exp) and same(zarr_read(copy), exp) and same(ts_read(copy), exp)
                whole = values_like(rng, exp)
                cppzarr.write_zarr(copy, whole, start_coords=[0] * exp.ndim, end_coords=list(exp.shape))
                good = good and same(cppzarr.read_zarr(copy), whole) and same(zarr_read(copy), whole) and \
                    same(ts_read(copy), whole) and not big_endian
                if fx['name'].startswith('v2_'):
                    try:
                        cppzarr.convert_to_v3(copy)
                        good = good and 'zlib' not in fx['name'] and os.path.isfile(os.path.join(copy, 'zarr.json')) and \
                            same(cppzarr.read_zarr(copy), whole) and same(zarr_read(copy), whole)
                    except Exception as e:
                        good = good and 'zlib' in fx['name'] and 'Zarr v3' in str(e)
            except Exception as e:
                good = big_endian and 'byte order' in str(e)
            print(f"writes into test array {fx['name']:36s} {'OK' if good else 'FAIL'}")
            ok = ok and good

    # Zarr v3 writing: every dtype and compressor; C order and c/0/0/0 chunk keys by
    # default (F order and other separators when asked); read back by cpp-zarr and,
    # when installed, by zarr-python and TensorStore
    for dt in dtypes:
        data = values_like(rng, np.zeros(shape, dt))
        for cname, order in [("zstd", None), ("lz4", "F"), ("gzip", None), ("none", "F"), ("zlib", None)]:
            path = os.path.join(tmp, f"v3_{np.dtype(dt).name}_{cname}_{order}.zarr")
            cppzarr.write_zarr(path, data, cname=cname, order=order, chunks=chunks, zarr_format=3)
            back = cppzarr.read_zarr(path)
            meta = load_json(os.path.join(path, 'zarr.json'))
            good = (np.array_equal(back, data) and back.flags[f'{order or "C"}_CONTIGUOUS'] and
                    not os.path.exists(os.path.join(path, '.zarray')) and os.path.isfile(os.path.join(path, 'c', '0', '0', '0')) and
                    meta['zarr_format'] == 3 and same(zarr_read(path), data) and same(ts_read(path), data))
            print(f"v3 {np.dtype(dt).name:8s} {cname:5s} {order or 'C'}  {'OK' if good else 'FAIL'}")
            ok = ok and good

    # Shards (the shard shape; chunks are the inner chunks), crop writes into them, and
    # into unsharded arrays; regions that are not chunk-aligned; '.' chunk keys
    for shards, sep, order in [([32, 16, 18], None, None), ([16, 32, 18], '.', 'F'), (None, '.', None)]:
        data = rng.integers(0, 60000, size=(70, 45, 33)).astype(np.uint16)
        path = os.path.join(tmp, f"v3_shards_{shards}_{order}.zarr")
        cppzarr.write_zarr(path, data, chunks=[16, 16, 9] if shards else [16, 8, 9], shards=shards, zarr_format=3,
                           dimension_separator=sep, order=order)
        good = np.array_equal(cppzarr.read_zarr(path), data)
        for s, e in [([5, 7, 3], [61, 40, 29]), ([64, 32, 30], [70, 45, 33]), ([0, 0, 0], [16, 16, 9])]:
            region = tuple(slice(a, b) for a, b in zip(s, e))
            patch = rng.integers(0, 60000, size=[b - a for a, b in zip(s, e)]).astype(np.uint16)
            cppzarr.write_zarr(path, patch, start_coords=s, end_coords=e)
            data[region] = patch
            good = good and np.array_equal(cppzarr.read_zarr(path), data) and np.array_equal(cppzarr.read_zarr(path, s, e), patch)
        good = good and same(zarr_read(path), data) and same(ts_read(path), data)
        print(f"v3 shards={shards} sep={sep} order={order}  {'OK' if good else 'FAIL'}")
        ok = ok and good

    # Shards of arrays with more than three dimensions keep every axis's shard and
    # inner chunk size
    d4 = rng.integers(0, 60000, size=(6, 8, 10, 12, 14)).astype(np.uint16)
    p4 = os.path.join(tmp, "v3_shards_5d.zarr")
    cppzarr.write_zarr(p4, d4, chunks=[2, 3, 4, 5, 6], shards=[4, 6, 8, 10, 12], zarr_format=3)
    meta = load_json(os.path.join(p4, 'zarr.json'))
    good = (meta['chunk_grid']['configuration']['chunk_shape'] == [4, 6, 8, 10, 12] and
            meta['codecs'][0]['configuration']['chunk_shape'] == [2, 3, 4, 5, 6] and
            np.array_equal(cppzarr.read_zarr(p4), d4) and same(zarr_read(p4), d4) and same(ts_read(p4), d4))
    print(f"v3 5D shards  {'OK' if good else 'FAIL'}")
    ok = ok and good

    # Writing a region into an existing array converts the data to the array's type,
    # as the MATLAB writer does; data in the other byte order is written correctly
    good = True
    for zf, atype, dtype in [(2, np.uint16, np.float32), (3, np.float32, np.int64), (3, np.uint8, np.int32),
                             (3, np.int16, np.dtype('>i2')), (2, np.float64, np.dtype('>f4'))]:
        # (no "<" or ">" in file names: Windows does not allow them)
        big = '_big_endian' if np.dtype(dtype).byteorder == '>' else ''
        path = os.path.join(tmp, f"convert_v{zf}_{np.dtype(atype).name}_{np.dtype(dtype).name}{big}.zarr")
        base = rng.integers(0, 200, size=(40, 24, 18)).astype(atype)
        cppzarr.write_zarr(path, base, chunks=[16, 16, 16], zarr_format=zf)
        patch = (rng.random((10, 9, 8)) * 200).astype(dtype)
        cppzarr.write_zarr(path, patch, start_coords=[3, 5, 7], end_coords=[13, 14, 15])
        exp = base.copy()
        exp[3:13, 5:14, 7:15] = patch.astype(atype)
        back = cppzarr.read_zarr(path)
        good = good and back.dtype == np.dtype(atype) and np.array_equal(back, exp) and same(zarr_read(path), exp)
    for dtype in ('>u2', '>i4', '>f8'):
        path = os.path.join(tmp, f"big_endian_{dtype[1:]}.zarr")
        d = np.arange(-50, 950).reshape(10, 100).astype(dtype) if dtype[1] != 'u' else np.arange(1000).reshape(10, 100).astype(dtype)
        cppzarr.write_zarr(path, d)
        back = cppzarr.read_zarr(path)
        good = good and back.dtype == d.dtype.newbyteorder('=') and np.array_equal(back, d)
    # (the module's own write function refuses data of another type than the array's)
    try:
        from cppzarr.cppzarr import pybind11_write_zarr
        pybind11_write_zarr(path, np.zeros((2, 2), np.float32), [0, 0], [2, 2], 'zstd', 1, 'F', [64, 64], '.', True, 2, [])
        good = False
    except Exception as e:
        good = good and 'does not match' in str(e)
    print(f"data converted to the array's type  {'OK' if good else 'FAIL'}")
    ok = ok and good

    # The format of an existing array is kept: writes default to it, a crop write
    # cannot change it, and a v2 write over a v3 array is refused. shards needs Zarr
    # v3 and a multiple of the inner chunk shape. A v2 array converted to v3 reads the
    # same, keeps its attributes, and can be written.
    d = rng.integers(0, 60000, size=(40, 24, 18)).astype(np.uint16)
    p3 = os.path.join(tmp, "v3_kept.zarr")
    cppzarr.write_zarr(p3, d, zarr_format=3)
    cppzarr.write_zarr(p3, d[::-1].copy())
    good = os.path.isfile(os.path.join(p3, 'zarr.json')) and np.array_equal(cppzarr.read_zarr(p3), d[::-1])
    cppzarr.write_zarr(p3, d[:4, :4, :4], start_coords=[1, 1, 1], end_coords=[5, 5, 5])
    exp = d[::-1].copy()
    exp[1:5, 1:5, 1:5] = d[:4, :4, :4]
    good = good and np.array_equal(cppzarr.read_zarr(p3), exp) and not os.path.exists(os.path.join(p3, '.zarray'))
    for kwargs in [dict(zarr_format=2, start_coords=[0, 0, 0], end_coords=[4, 4, 4]), dict(zarr_format=2),
                   dict(shards=[8, 8, 8], zarr_format=2), dict(shards=[24, 24, 24], chunks=[16, 16, 16], zarr_format=3)]:
        try:
            data = d[:4, :4, :4] if 'start_coords' in kwargs else d
            cppzarr.write_zarr(p3 if kwargs.get('zarr_format') != 3 else p3 + '_new', data, **kwargs)
            good = False
        except Exception as e:
            good = good and any(w in str(e) for w in ('Zarr v3', 'zarr_format=3', 'multiple'))
    good = good and np.array_equal(cppzarr.read_zarr(p3), exp)
    p2 = os.path.join(tmp, "v2_to_v3.zarr")
    cppzarr.write_zarr(p2, d, chunks=[16, 16, 8])
    with open(os.path.join(p2, '.zattrs'), 'w') as f:
        json.dump({'units': 'um'}, f)
    cppzarr.convert_to_v3(p2)
    meta = load_json(os.path.join(p2, 'zarr.json'))
    good = good and meta['attributes'] == {'units': 'um'} and not os.path.exists(os.path.join(p2, '.zarray')) and \
        np.array_equal(cppzarr.read_zarr(p2), d) and cppzarr.read_zarr(p2).flags['F_CONTIGUOUS'] and same(zarr_read(p2), d)
    cppzarr.write_zarr(p2, d[:4, :4, :4], start_coords=[1, 1, 1], end_coords=[5, 5, 5])
    exp = d.copy()
    exp[1:5, 1:5, 1:5] = d[:4, :4, :4]
    good = good and np.array_equal(cppzarr.read_zarr(p2), exp) and same(zarr_read(p2), exp)
    try:
        cppzarr.convert_to_v3(p2)
        good = False
    except Exception as e:
        good = good and 'not a Zarr v2 array' in str(e)
    print(f"v3 existing arrays and conversion  {'OK' if good else 'FAIL'}")
    ok = ok and good

    if not ok:
        sys.exit("cppzarr Python round-trip test FAILED")
    print("cppzarr Python round-trip tests PASSED")


if __name__ == "__main__":
    main()
