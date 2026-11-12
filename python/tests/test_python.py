"""Smoke test for the cpp-zarr Python wheel.

Writes small random volumes, reads them back, and checks the data survives the
round trip -- across dtypes, compressors, both storage orders (F and C), plus a
region (cropped) read, arrays of 0 to 40 dimensions with region reads and crop
writes, and arrays in any memory layout (C order, F order, strided views) read
back in either order, and test arrays written by zarr-python 3 and TensorStore
(tests/test_arrays: Zarr v3, and v2 with other codecs and fill values), written into
and converted to Zarr v3. Every data type: booleans, integers, float16, float32,
float64, complex64, complex128, and bfloat16 when ml_dtypes is installed. Zarr v3
writing: round trips, shards, crop writes, the
format of an existing array, conversion, and (when zarr-python and TensorStore are
installed) reading what cpp-zarr wrote with them. Run automatically by cibuildwheel
against the freshly built+installed wheel, so it also verifies the wheel imports and
bundles its native library. Exits non-zero on any failure.
"""
import json
import os
import shutil
import subprocess
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
# bfloat16 (ml_dtypes' type)
try:
    import ml_dtypes
except ImportError:
    ml_dtypes = None


def is_bfloat16(dt):
    return ml_dtypes is not None and np.dtype(dt) == np.dtype(ml_dtypes.bfloat16)


def zarr_read(path):
    # The array as zarr-python reads it (None without zarr-python, and for bfloat16,
    # which it does not read)
    if zarr is None:
        return None
    meta = [os.path.join(path, f) for f in ('zarr.json', '.zarray') if os.path.isfile(os.path.join(path, f))]
    if meta:
        m = load_json(meta[0])
        if str(m.get('data_type', m.get('dtype'))).lstrip('<>|') == 'bfloat16':
            return None
    return zarr.open_array(path, mode='r')[...]


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
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if is_bfloat16(a.dtype):
        a, b = a.astype(np.float32), b.astype(np.float32)
    return np.array_equal(a, b, equal_nan=a.dtype.kind in 'fc')


def values_like(rng, a):
    # Random values of a's dtype and shape (no NaNs)
    a = np.asarray(a)
    if a.dtype.kind == 'b':
        return rng.random(a.shape) < 0.5
    if a.dtype.kind == 'c':
        return np.asarray((rng.standard_normal(a.shape) + 1j * rng.standard_normal(a.shape)) * 1000, dtype=a.dtype)
    if a.dtype.kind == 'f' or is_bfloat16(a.dtype):
        return np.asarray(rng.standard_normal(a.shape) * 1000, dtype=np.float32).astype(a.dtype)
    info = np.iinfo(a.dtype)
    return np.asarray(rng.integers(max(info.min, -10**6), min(info.max, 10**6), size=a.shape, endpoint=True), dtype=a.dtype)


def matlab_cast(a, dt):
    # MATLAB's conversion between classes, the reference for write_zarr's into an array of another type (written
    # here with NumPy alone): into integers rounded to the nearest, halves away from zero, saturated at the type's
    # limits, NaN to 0; into bool any nonzero value (NaN too) true; into float16 directly from double and into
    # bfloat16 through float32; otherwise as NumPy converts
    dt, a = np.dtype(dt), np.asarray(a)
    if dt.kind in 'iu':
        info = np.iinfo(dt)
        if a.dtype.kind in 'biu':
            return np.array([min(max(int(v), info.min), info.max) for v in a.ravel()], dt).reshape(a.shape)
        x = a.astype(np.float64)
        r = np.where(np.abs(x - np.trunc(x)) == 0.5, np.trunc(x) + np.sign(x), np.round(x))
        out = np.zeros(a.shape, dt)
        hi, lo = r >= float(info.max), r <= float(info.min)
        mid = ~(hi | lo | np.isnan(r))
        out[mid] = r[mid].astype(dt)
        out[hi], out[lo] = info.max, info.min
        return out
    if dt.kind == 'b':
        return a != 0 if a.dtype.kind != 'f' else (a != 0) | np.isnan(a)
    if is_bfloat16(dt):
        return a.astype(np.float32).astype(dt)
    return a.astype(np.float64).astype(dt) if a.dtype.kind != 'c' else a.astype(dt)


def main():
    rng = np.random.default_rng(1234567)
    shape = (40, 24, 18)  # d0, d1, d2 (non-chunk-aligned -> exercises partial chunks)
    chunks = [16, 16, 16]
    dtypes = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32,
              np.uint64, np.int64, np.float32, np.float64, np.bool_, np.float16, np.complex64, np.complex128]
    if ml_dtypes is not None:
        dtypes.append(ml_dtypes.bfloat16)
    # (compressor, order) combinations. zstd/F is our most-used combo; the others
    # add lz4, gzip, and a C-order case (which exercises the read transpose).
    combos = [("zstd", "F"), ("lz4", "F"), ("zstd", "C"), ("gzip", "F")]

    tmp = tempfile.mkdtemp()
    ok = True
    for dt in dtypes:
        name = np.dtype(dt).name
        # (negatives exercise signed types)
        data = values_like(rng, np.zeros(shape, dt))

        for cname, order in combos:
            path = os.path.join(tmp, f"rt_{name}_{cname}_{order}.zarr")
            cppzarr.write_zarr(path, data, cname=cname, order=order, chunks=chunks)
            back = cppzarr.read_zarr(path)
            # (zarr-python reads it the same)
            good = (back.dtype == data.dtype and back.shape == data.shape
                    and same(back, data) and same(zarr_read(path), data))
            print(f"{name:8s} {cname:8s} {order}  {'OK' if good else 'FAIL'}")
            ok = ok and good

        # Region (cropped) read: first half along d0.
        h = shape[0] // 2
        path = os.path.join(tmp, f"rt_{name}_lz4_F.zarr")
        sub = cppzarr.read_zarr(path, start_coords=[0, 0, 0], end_coords=[h, shape[1], shape[2]])
        rgood = sub.shape == (h, shape[1], shape[2]) and same(sub, data[0:h])
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
        # (bfloat16 arrays need ml_dtypes)
        def fixture_dtype(fx):
            return np.dtype(ml_dtypes.bfloat16) if fx['dtype'] == 'bfloat16' else np.dtype(fx['dtype'])
        if ml_dtypes is None:
            skipped = [fx['name'] for fx in fixtures if fx.get('dtype') == 'bfloat16']
            fixtures = [fx for fx in fixtures if fx.get('dtype') != 'bfloat16']
            if skipped:
                print(f"bfloat16 test arrays skipped (no ml_dtypes): {', '.join(skipped)}")
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
            exp = np.fromfile(os.path.join(v3dir, fx['name'] + '.bin'), dtype=fixture_dtype(fx)).reshape(fx['shape'])
            good = True
            for o in (None, 'F', 'C'):
                a = cppzarr.read_zarr(path, order=o)
                good = good and same(a, exp)
            if exp.ndim:
                s = [1 if n > 2 else 0 for n in exp.shape]
                e = [n - 1 if n > 2 else n for n in exp.shape]
                region = tuple(slice(a, b) for a, b in zip(s, e))
                good = good and same(cppzarr.read_zarr(path, s, e), exp[region])
            print(f"test array {fx['name']:36s}           {'OK' if good else 'FAIL'}")
            ok = ok and good
        # Writes into copies of them: a region, then the whole array (as the array's
        # own type), read back by cpp-zarr and, when installed, zarr-python; arrays in
        # the opposite byte order are refused. Then the v2 ones converted to Zarr v3
        # (numcodecs zlib cannot be) read the same.
        for fx in fixtures:
            if 'error' in fx:
                continue
            exp = np.fromfile(os.path.join(v3dir, fx['name'] + '.bin'), dtype=fixture_dtype(fx)).reshape(fx['shape'])
            copy = os.path.join(tmp, 'ta_' + fx['name'] + '.zarr')
            shutil.copytree(os.path.join(v3dir, fx['name'] + '.zarr'), copy)
            meta_file = os.path.join(copy, 'zarr.json')
            # (Zarr v3: a bytes codec in big-endian order; v2: a ">" dtype)
            big_endian = (os.path.isfile(meta_file) and '"endian": "big"' in json.dumps(load_json(meta_file))) or \
                (os.path.isfile(os.path.join(copy, '.zarray')) and str(load_json(os.path.join(copy, '.zarray'))['dtype']).startswith('>'))
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
    # as the MATLAB writer does (matlab_cast); data in the other byte order is written correctly
    good = True
    for zf, atype, dtype in [(2, np.uint16, np.float32), (3, np.float32, np.int64), (3, np.uint8, np.int32),
                             (3, np.int16, np.dtype('>i2')), (2, np.float64, np.dtype('>f4')),
                             (3, np.float16, np.float64), (2, np.complex64, np.float32), (3, np.complex128, np.complex64),
                             (2, np.bool_, np.uint8), (3, np.float16, np.dtype('>f2'))]:
        # (no "<" or ">" in file names: Windows does not allow them)
        big = '_big_endian' if np.dtype(dtype).byteorder == '>' else ''
        path = os.path.join(tmp, f"convert_v{zf}_{np.dtype(atype).name}_{np.dtype(dtype).name}{big}.zarr")
        base = rng.integers(0, 200, size=(40, 24, 18)).astype(atype)
        cppzarr.write_zarr(path, base, chunks=[16, 16, 16], zarr_format=zf)
        patch = (rng.random((10, 9, 8)) * 200).astype(dtype)
        cppzarr.write_zarr(path, patch, start_coords=[3, 5, 7], end_coords=[13, 14, 15])
        exp = base.copy()
        exp[3:13, 5:14, 7:15] = matlab_cast(patch, atype)
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

    # The conversion's edge cases, through region writes, as the MATLAB writer converts them (rounding halves away
    # from zero, and 0.49999999999999994 to 0; saturating, also at the 64-bit limits; NaN to 0; any nonzero value
    # true), from strided views and F-order data too; complex data refused for real arrays
    good, bad = True, []
    third = 1 / 3
    cases = [(np.uint16, np.array([2.5, -1, 70000, np.nan, 65535.6, 0.5, -0.5, 1.5, -2.5, 0.49999999999999994, np.inf, -np.inf]),
              [3, 0, 65535, 0, 65535, 1, 0, 2, 0, 0, 65535, 0]),
             (np.int8, np.array([2.5, -1, 70000, np.nan, 65535.6, 0.5, -0.5, 1.5, -2.5, 0.49999999999999994, -129.5, 126.5]),
              [3, -1, 127, 0, 127, 1, -1, 2, -3, 0, -128, 127]),
             (np.int64, np.array([9.3e18, -9.3e18, 2.0 ** 62, np.nan, np.inf, -np.inf, 4611686018427387904.5, -2.5]),
              [2 ** 63 - 1, -2 ** 63, 2 ** 62, 0, 2 ** 63 - 1, -2 ** 63, 2 ** 62, -3]),
             (np.uint64, np.array([1.9e19, -1.0, 2.0 ** 63, 0.5, 1.8446744073709552e19]), [2 ** 64 - 1, 0, 2 ** 63, 1, 2 ** 64 - 1]),
             (np.int8, np.array([-2 ** 63, -1, 0, 2 ** 63 - 1], np.int64), [-128, -1, 0, 127]),
             (np.uint32, np.array([-2 ** 63, -1, 0, 2 ** 63 - 1], np.int64), [0, 0, 0, 2 ** 32 - 1]),
             (np.int64, np.array([2 ** 64 - 1, 2 ** 63, 5], np.uint64), [2 ** 63 - 1, 2 ** 63 - 1, 5]),
             (np.int16, np.array([2 ** 64 - 1, 2 ** 63, 5], np.uint64), [32767, 32767, 5]),
             (np.int32, np.array([1.5, 2.5, -1.5, 3.4e38, -3.4e38], np.float32), [2, 3, -2, 2 ** 31 - 1, -2 ** 31]),
             (np.uint8, np.array([1.5, 300, -7, 254.5], np.float16), [2, 255, 0, 255]),
             (np.bool_, np.array([0.0, -0.0, 0.1, np.nan, -3]), [False, False, True, True, True]),
             (np.float32, np.array([True, False]), [1.0, 0.0]),
             (np.float16, np.array([third, 65520.0, 1e-8, -2.5]), np.array([third, 65520.0, 1e-8, -2.5]).astype(np.float16)),
             (np.float32, np.array([third, 1e300]), np.array([third, 1e300]).astype(np.float32))]
    if ml_dtypes is not None:
        bf = np.dtype(ml_dtypes.bfloat16)
        cases.append((bf, np.array([third, 1.00390625, -3.0e38]), np.array([third, 1.00390625, -3.0e38]).astype(np.float32).astype(bf)))
    for k, (atype, vals, want) in enumerate(cases):
        want = np.asarray(want, dtype=atype)
        n = len(vals)
        path = os.path.join(tmp, f"cast_{k}.zarr")
        cppzarr.write_zarr(path, np.zeros((n, 3), atype), chunks=[4, 2], zarr_format=2 + k % 2)
        src = np.stack([vals, vals[::-1], vals], axis=1)
        if k % 3 == 1:
            src = np.asfortranarray(src)
        elif k % 3 == 2:
            src = np.stack([vals, vals * 0, vals[::-1], vals * 0, vals], axis=1)[:, ::2]   # (a strided view)
        cppzarr.write_zarr(path, src, start_coords=[0, 0], end_coords=[n, src.shape[1]])
        back = cppzarr.read_zarr(path)
        exp = np.stack([want, want[::-1], want], axis=1)
        good_k = back.dtype == np.dtype(atype) and back.tobytes(order='C') == exp.tobytes(order='C') \
            and np.array_equal(back, matlab_cast(src, atype), equal_nan=back.dtype.kind == 'f')
        if not good_k:
            bad.append(f'{np.dtype(atype).name} from {vals.dtype.name}: {back[:, 0].tolist()} (want {want.tolist()})')
        good = good and good_k
    try:
        path = os.path.join(tmp, "cast_complex.zarr")
        cppzarr.write_zarr(path, np.zeros((4, 4), np.uint16), chunks=[4, 4])
        cppzarr.write_zarr(path, np.ones((2, 2), np.complex64), start_coords=[0, 0], end_coords=[2, 2])
        good = False
        bad.append('complex data written into a uint16 array')
    except Exception as e:
        good = good and 'complex' in str(e)
    print(f"values converted as MATLAB converts them  {'OK' if good else 'FAIL'}" + ('' if good else f" ({'; '.join(bad)})"))
    ok = ok and good

    # sparse: by default all-zero chunks are not written (and an existing file of one is deleted); with
    # sparse=False they are, as the MATLAB writer's 'sparse', false writes them. uuid=False writes the same files
    # (in place, without temporary names)
    def chunk_count(p):
        return sum(len([f for f in fs if f not in ('.zarray', 'zarr.json', '.zattrs')]) for _, _, fs in os.walk(p))
    good = True
    for zf, sep in ((2, '.'), (3, '/')):
        z = np.zeros((32, 32, 10), np.uint16, order='F')
        ps, pf = os.path.join(tmp, f"sparse_v{zf}.zarr"), os.path.join(tmp, f"nonsparse_v{zf}.zarr")
        cppzarr.write_zarr(ps, z, order='F', chunks=[16, 16, 5], dimension_separator=sep, zarr_format=zf)
        cppzarr.write_zarr(pf, z, order='F', chunks=[16, 16, 5], dimension_separator=sep, zarr_format=zf, sparse=False)
        good = good and chunk_count(ps) == 0 and chunk_count(pf) == 8 and np.array_equal(cppzarr.read_zarr(pf), z) \
            and np.array_equal(cppzarr.read_zarr(ps), z) and same(zarr_read(pf), z)
        # a region of zeros over a chunk with values: its file deleted, or kept and written with zeros
        d = rng.integers(1, 100, size=(32, 32, 10)).astype(np.uint16)
        for p, sp, files in ((ps, True, 7), (pf, False, 8)):
            cppzarr.write_zarr(p, d, order='F', chunks=[16, 16, 5], dimension_separator=sep, zarr_format=zf, sparse=sp)
            cppzarr.write_zarr(p, np.zeros((16, 16, 5), np.uint16), start_coords=[0, 0, 0], end_coords=[16, 16, 5], sparse=sp)
            exp = d.copy()
            exp[:16, :16, :5] = 0
            good = good and chunk_count(p) == files and np.array_equal(cppzarr.read_zarr(p), exp)
    d = rng.integers(0, 1000, size=(40, 24, 18)).astype(np.uint16)
    pu, pn = os.path.join(tmp, "with_uuid.zarr"), os.path.join(tmp, "no_uuid.zarr")
    for p, u in ((pu, True), (pn, False)):
        cppzarr.write_zarr(p, d, chunks=[16, 16, 16], uuid=u, zarr_format=3, shards=[32, 32, 32])
        cppzarr.write_zarr(p, d[:5, :5, :5][::-1].copy(), start_coords=[2, 2, 2], end_coords=[7, 7, 7], uuid=u)
    files = lambda p: sorted((os.path.relpath(os.path.join(r, f), p), open(os.path.join(r, f), 'rb').read())
                             for r, _, fs in os.walk(p) for f in fs)
    good = good and files(pu) == files(pn) and np.array_equal(cppzarr.read_zarr(pn), cppzarr.read_zarr(pu))
    print(f"sparse=False writes all-zero chunks, uuid=False the same files  {'OK' if good else 'FAIL'}")
    ok = ok and good

    # create_zarr: the metadata write_zarr writes for data of that shape and type, and no chunk files; the array
    # reads as zeros and regions written into it fill it; shards need Zarr v3
    good, bad = True, []
    combos = [dict(), dict(zarr_format=3), dict(zarr_format=3, shards=[32, 32, 16], chunks=[16, 16, 8]),
              dict(order='C', dimension_separator='.', cname='lz4', clevel=5, chunks=[8, 16, 4]),
              dict(zarr_format=3, order='F', cname='zlib', chunks=[40, 24, 18])]
    for k, opts in enumerate(combos):
        for dt in [np.uint16, np.float32, np.bool_, np.int8, np.complex64] + ([ml_dtypes.bfloat16] if ml_dtypes is not None else []):
            name = np.dtype(dt).name
            pc, pw = os.path.join(tmp, f"created_{k}_{name}.zarr"), os.path.join(tmp, f"written_{k}_{name}.zarr")
            cppzarr.create_zarr(pc, (40, 24, 18), dt if k % 2 else name, **opts)
            cppzarr.write_zarr(pw, np.zeros((40, 24, 18), dt), **opts)
            meta = '.zarray' if os.path.isfile(os.path.join(pw, '.zarray')) else 'zarr.json'
            same_meta = load_json(os.path.join(pc, meta)) == load_json(os.path.join(pw, meta))
            empty = chunk_count(pc) == 0 and np.array_equal(cppzarr.read_zarr(pc), np.zeros((40, 24, 18), dt))
            part = values_like(rng, np.zeros((7, 9, 5), dt))
            cppzarr.write_zarr(pc, part, start_coords=[3, 4, 5], end_coords=[10, 13, 10])
            exp = np.zeros((40, 24, 18), dt)
            exp[3:10, 4:13, 5:10] = part
            filled = np.array_equal(cppzarr.read_zarr(pc), exp)
            if not (same_meta and empty and filled):
                bad.append(f'{k} {name}: metadata {same_meta}, empty {empty}, filled {filled}')
            good = good and same_meta and empty and filled
    for args, kw, words in [((os.path.join(tmp, 'cz_bad1.zarr'), (8, 8), 'uint16'), dict(shards=[8, 8], zarr_format=2), 'zarr_format=3'),
                            ((os.path.join(tmp, 'cz_bad2.zarr'), (8, 8), 'float128'), {}, 'data type'),
                            ((os.path.join(tmp, 'cz_bad3.zarr'), (8, 8), 'uint16'), dict(chunks=[4]), 'chunks')]:
        try:
            cppzarr.create_zarr(*args, **kw)
            good = False
            bad.append(f'{args[0]} not refused')
        except Exception as e:
            good = good and (words in str(e) or 'Unsupported' in str(e))
    print(f"create_zarr  {'OK' if good else 'FAIL'}" + ('' if good else f" ({'; '.join(bad)})"))
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

    # Reading into an array (out=): the region's values, and the fill value where chunks
    # (or a shard's inner chunks, or whole shards) are missing, whatever out held; out
    # of another shape, type or layout is refused. num_threads: the same values.
    good = True
    for zf, shards, dt in [(2, None, np.uint16), (3, None, np.float32), (3, [32, 32, 16], np.float32), (3, [32, 32, 16], np.complex64)]:
        p = os.path.join(tmp, f"out_{zf}_{shards}_{np.dtype(dt).name}.zarr")
        d = values_like(rng, np.zeros((70, 45, 33), dt))
        cppzarr.write_zarr(p, d, chunks=[16, 16, 8], zarr_format=zf, shards=shards)
        meta_file = os.path.join(p, 'zarr.json' if zf == 3 else '.zarray')
        meta = load_json(meta_file)
        meta['fill_value'] = 7 if dt == np.uint16 else ([1.5, 'NaN'] if dt == np.complex64 else 'NaN')
        with open(meta_file, 'w') as f:
            json.dump(meta, f)
        files = sorted(os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs if f not in ('zarr.json', '.zarray'))
        for f in files[::3]:
            os.remove(f)
        ref = cppzarr.read_zarr(p)
        for order in 'CF':
            for s, e in [([0, 0, 0], [70, 45, 33]), ([5, 7, 3], [61, 40, 29])]:
                region = tuple(slice(a, b) for a, b in zip(s, e))
                o = np.empty([b - a for a, b in zip(s, e)], dtype=dt, order=order)
                o[...] = 123
                got = cppzarr.read_zarr(p, s, e, out=o, num_threads=3)
                good = good and got is o and same(o, ref[region]) and same(cppzarr.read_zarr(p, s, e, num_threads=1), ref[region])
    p = os.path.join(tmp, "out_2_None_uint16.zarr")
    for bad, what in [(np.empty((70, 45, 33), np.float32), 'data type'), (np.empty((70, 45, 32), np.uint16), 'shape'),
                      (np.empty((70, 45, 66), np.uint16)[:, :, ::2], 'contiguous')]:
        try:
            cppzarr.read_zarr(p, out=bad)
            good = False
        except Exception as e:
            good = good and what in str(e)
    for bad in (0, -1, 1.5):
        try:
            cppzarr.read_zarr(p, num_threads=bad)
            good = False
        except Exception as e:
            good = good and 'num_threads' in str(e)
    print(f"reading into arrays, and num_threads  {'OK' if good else 'FAIL'}")
    ok = ok and good

    # Batches of reads: the same values as reading each region on its own, in the
    # requests' order, into new arrays or into given ones (one batch buffer), from
    # several arrays; and other Python threads run while cpp-zarr reads (the GIL is
    # released)
    import threading
    import time
    arrays = []
    for k, (zf, shards) in enumerate([(2, None), (3, [32, 32, 16]), (3, None)]):
        p = os.path.join(tmp, f"batch_{k}.zarr")
        d = values_like(rng, np.zeros((70, 45, 33), np.uint16))
        cppzarr.write_zarr(p, d, chunks=[16, 16, 8], zarr_format=zf, shards=shards)
        arrays.append((p, d))
    requests, expected = [], []
    for i in range(23):
        p, d = arrays[i % 3]
        s = [int(rng.integers(0, 40)), int(rng.integers(0, 20)), int(rng.integers(0, 10))]
        e = [s[0] + 30, s[1] + 25, s[2] + 20]
        requests.append((p, s, e))
        expected.append(d[tuple(slice(a, b) for a, b in zip(s, e))])
    good = True
    for threads in (None, 1, 4, 64):
        got = cppzarr.read_zarr_batch(requests, num_threads=threads)
        good = good and len(got) == len(expected) and all(np.array_equal(g, x) for g, x in zip(got, expected))
    buffer = np.zeros((len(requests), 30, 25, 20), np.uint16)
    got = cppzarr.read_zarr_batch([r + (buffer[i],) for i, r in enumerate(requests)], num_threads=4)
    good = good and all(g.base is buffer or g is buffer[i] for i, g in enumerate(got)) and np.array_equal(buffer, np.stack(expected))
    good = good and cppzarr.read_zarr_batch([]) == []
    # batches of other sizes than the reads run at once (10 reads 4 at a time, 5 at 4, 7 at
    # 3): the same values
    for k, threads in ((10, 4), (5, 4), (7, 3)):
        got = cppzarr.read_zarr_batch(requests[:k], num_threads=threads)
        good = good and len(got) == k and all(np.array_equal(g, x) for g, x in zip(got, expected[:k]))
    # batches of more threads than CPPZARR_READ_AHEAD_THREADS (split as 1.7.3's, a thread per 8M
    # elements: here 3 reads of 32 threads) and of as many: the same values; the setting read as
    # the library reads it
    from cppzarr import cppzarrwrappers as wrappers
    many = os.path.join(tmp, "many.zarr")
    dm = values_like(rng, np.zeros((256, 256, 256), np.uint8))
    cppzarr.write_zarr(many, dm, chunks=[64, 64, 64], zarr_format=3, shards=[128, 128, 128])
    for threads in (wrappers._READ_AHEAD_THREADS, wrappers._READ_AHEAD_THREADS + 32):
        got = cppzarr.read_zarr_batch([(many, None, None)] * 3, num_threads=threads)
        good = good and all(np.array_equal(g, dm) for g in got)
    saved = os.environ.get('CPPZARR_READ_AHEAD_THREADS')
    for v, n in (('', 64), ('12', 12), ('-3', 0), ('x', 0)):
        os.environ['CPPZARR_READ_AHEAD_THREADS'] = v
        good = good and wrappers._read_ahead_threads() == n
    if saved is None:
        del os.environ['CPPZARR_READ_AHEAD_THREADS']
    else:
        os.environ['CPPZARR_READ_AHEAD_THREADS'] = saved
    # a read that fails, in an earlier round or in the last one: the batch raises its error
    missing = (os.path.join(tmp, "missing.zarr"), [0, 0, 0], [30, 25, 20])
    for batch in (requests[:3] + [missing] + requests[3:10], requests[:9] + [missing]):
        try:
            cppzarr.read_zarr_batch(batch, num_threads=4)
            good = False
        except Exception as e:
            good = good and 'missing.zarr' in str(e)
    # a batch's threads by default are a call's: OMP_NUM_THREADS when it is set
    import subprocess
    child = subprocess.run([sys.executable, '-c', 'from cppzarr.cppzarrwrappers import pybind11_default_threads; print(pybind11_default_threads())'],
                           env=dict(os.environ, OMP_NUM_THREADS='3'), capture_output=True, text=True)
    good = good and child.stdout.strip() == '3'
    # whole arrays (no coordinates), and C order asked for
    whole = cppzarr.read_zarr_batch([(p, None, None) for p, _ in arrays], order='C', num_threads=3)
    good = good and all(np.array_equal(w, d) and w.flags['C_CONTIGUOUS'] for w, (_, d) in zip(whole, arrays))
    # reads of one thread each from network storage (here taken to be, for every setting) run up to
    # CPPZARR_BATCH_READS_PER_THREAD at once a thread as their I/O wait calls for (here taken to call for
    # all of them, readers added as reads end): the same values, into outs and new arrays; how many their
    # CPU time and I/O wait call for; and the setting as it is read
    many_small = [requests[i % len(requests)] for i in range(40)]
    want_small = [expected[i % len(requests)] for i in range(40)]
    saved_rpt, saved_net, saved_wanted = wrappers._READS_PER_THREAD, wrappers.pybind11_on_network, wrappers._readers_wanted
    saved_share, saved_fetched = wrappers._wait_share[0], wrappers._process_read_bytes
    saved_allowed, saved_learn, saved_trials = wrappers._extras_allowed, wrappers._extras_learn, dict(wrappers._trials)
    try:
        wrappers.pybind11_on_network = lambda folder: True
        wrappers._readers_wanted = lambda threads, most, cpu, wait: most
        # (and found to help: see the trials below)
        wrappers._extras_allowed, wrappers._extras_learn = lambda size: True, lambda size, used, rate: None
        # (and the batches taken to have data fetched from the storage: more bytes each time it is asked)
        fetched_bytes = [0]

        def more_fetched():
            fetched_bytes[0] += 4096
            return fetched_bytes[0]
        wrappers._process_read_bytes = more_fetched
        for rpt in (1, 2, 3):
            wrappers._READS_PER_THREAD = rpt
            for threads in (1, 3, 8):
                got = cppzarr.read_zarr_batch(many_small, num_threads=threads)
                good = good and all(np.array_equal(g, x) for g, x in zip(got, want_small))
            outs = np.zeros((40, 30, 25, 20), np.uint16)
            got = cppzarr.read_zarr_batch([r + (outs[i],) for i, r in enumerate(many_small)], num_threads=4)
            good = good and np.array_equal(outs, np.stack(want_small))
        # (a batch that has had no data fetched, served by the page cache, runs one reader a thread and has
        # the next start with one a thread; one that has data fetched runs more, as one does where the
        # bytes fetched cannot be read; where reads cannot be timed, one a thread always)
        timed = wrappers.pybind11_read_times_available()
        for fetching, more_readers in ((lambda: 12345, False), (more_fetched, True), (lambda: None, True)):
            wrappers._process_read_bytes = fetching
            wrappers._wait_share[0] = None
            got = cppzarr.read_zarr_batch(many_small, num_threads=4)
            good = good and all(np.array_equal(g, x) for g, x in zip(got, want_small))
            good = good and (wrappers._last_batch[0] > 4) == (more_readers and timed)
            good = good and (more_readers or not timed or wrappers._wait_share[0] == 0.0)
        wrappers._process_read_bytes = more_fetched
        # (whether more readers help is tried for each batch size: its first batch, which starts the
        # pools' threads, and the next run one reader a thread, the third more)
        wrappers._extras_allowed, wrappers._extras_learn = saved_allowed, saved_learn
        peaks = []
        for k in range(3):
            wrappers._wait_share[0] = None
            got = cppzarr.read_zarr_batch(many_small[:37], num_threads=4)
            good = good and all(np.array_equal(g, x) for g, x in zip(got, want_small[:37]))
            peaks.append(wrappers._last_batch[0])
        good = good and peaks[:2] == [4, 4] and (peaks[2] > 4) == timed
        wrappers._extras_allowed, wrappers._extras_learn = lambda size: True, lambda size, used, rate: None
        # (a read that fails among readers added on the way: the batch raises its error)
        try:
            cppzarr.read_zarr_batch(many_small[:20] + [(many_small[0][0], [0, 0, 0], [999, 1, 1])] + many_small[20:], num_threads=2)
            good = False
        except Exception:
            pass
    finally:
        good = good and (wrappers._wait_share[0] is None or wrappers._wait_share[0] >= 0)
        wrappers._READS_PER_THREAD, wrappers.pybind11_on_network, wrappers._readers_wanted = saved_rpt, saved_net, saved_wanted
        wrappers._wait_share[0], wrappers._process_read_bytes = saved_share, saved_fetched
        wrappers._extras_allowed, wrappers._extras_learn = saved_allowed, saved_learn
        wrappers._trials.clear()
        wrappers._trials.update(saved_trials)
    found = wrappers._process_read_bytes()
    # the trials' states: a size's first batch and the next without more readers, then a batch with them,
    # kept if it fetched at least 1.1 times as fast as the one before without (and tried against one
    # without every _EXTRA_ON batches), else left off for _EXTRA_OFF batches and tried again
    saved_trials = dict(wrappers._trials)
    wrappers._trials.clear()
    try:
        seen = []
        for used, rate in ((False, 1.0), (False, 100.0), (True, 105.0)):
            seen.append(wrappers._extras_allowed(7))
            wrappers._extras_learn(7, used, rate)
        good = good and seen == [False, False, True]
        seen = []
        for k in range(wrappers._EXTRA_OFF):
            seen.append(wrappers._extras_allowed(7))
            wrappers._extras_learn(7, False, 100.0)
        good = good and not any(seen) and wrappers._extras_allowed(7)
        wrappers._extras_learn(7, True, 115.0)
        seen = []
        for k in range(wrappers._EXTRA_ON):
            seen.append(wrappers._extras_allowed(7))
            wrappers._extras_learn(7, True, 115.0)
        good = good and all(seen) and not wrappers._extras_allowed(7)
        wrappers._extras_learn(7, False, 100.0)
        # (a batch whose reads called for no more readers gives a later baseline)
        good = good and wrappers._extras_allowed(7)
        wrappers._extras_learn(7, False, 200.0)
        wrappers._extras_learn(7, True, 210.0)
        good = good and not wrappers._extras_allowed(7) and wrappers._extras_allowed(8) is False
    finally:
        wrappers._trials.clear()
        wrappers._trials.update(saved_trials)
    good = good and (found is None or (isinstance(found, int) and found >= 0))
    good = good and [wrappers._readers_wanted(16, 48, 4.0, w) for w in (0.0, 1.0, 2.0, 3.4, 100.0)] == [16, 16, 32, 44, 48]
    good = good and wrappers._readers_wanted(16, 48, 0.0, 0.0) == 16 and wrappers._readers_wanted(4, 4, 1.0, 9.0) == 4
    saved = os.environ.get('CPPZARR_BATCH_READS_PER_THREAD')
    for v, n in (('', 3), ('2', 2), ('1', 1), ('0', 1), ('-2', 1), ('x', 3)):
        os.environ['CPPZARR_BATCH_READS_PER_THREAD'] = v
        good = good and wrappers._reads_per_thread() == n
    if saved is None:
        del os.environ['CPPZARR_BATCH_READS_PER_THREAD']
    else:
        os.environ['CPPZARR_BATCH_READS_PER_THREAD'] = saved
    good = good and isinstance(wrappers.pybind11_on_network(tmp), bool)
    # each array's metadata is read once a batch, not kept between batches: an array written again
    # (other values, another shape) reads as written in the next batch
    again = os.path.join(tmp, "batch_again.zarr")
    first = values_like(rng, np.zeros((40, 30, 20), np.uint16))
    cppzarr.write_zarr(again, first, chunks=[16, 16, 8], zarr_format=3, shards=[32, 32, 16])
    got = cppzarr.read_zarr_batch([(again, [0, 0, 0], [10, 10, 10])] * 5 + [(again, None, None)], num_threads=4)
    good = good and all(np.array_equal(g, first[:10, :10, :10]) for g in got[:5]) and np.array_equal(got[5], first)
    shutil.rmtree(again)
    second = values_like(rng, np.zeros((50, 30, 20), np.uint16))
    cppzarr.write_zarr(again, second, chunks=[8, 8, 8], zarr_format=3)
    got = cppzarr.read_zarr_batch([(again, [0, 0, 0], [10, 10, 10])] * 5 + [(again, None, None)], num_threads=4)
    good = good and all(np.array_equal(g, second[:10, :10, :10]) for g in got[:5]) and np.array_equal(got[5], second)
    # a batch leaves nothing for Python's garbage collector: its objects are freed as it returns (kept
    # until a collection, they made the collector run often, and in a process of many objects a collection
    # can take longer than a batch), here with readers added as reads end (the 2nd and 3rd batches: the
    # 1st starts the pools' threads)
    import gc
    saved_rpt, saved_net, saved_wanted = wrappers._READS_PER_THREAD, wrappers.pybind11_on_network, wrappers._readers_wanted
    saved_share, saved_fetched = wrappers._wait_share[0], wrappers._process_read_bytes
    collecting = gc.isenabled()
    try:
        wrappers.pybind11_on_network = lambda folder: True
        wrappers._readers_wanted = lambda threads, most, cpu, wait: most
        wrappers._READS_PER_THREAD = 3
        fetched_bytes = [0]

        def more_fetched():
            fetched_bytes[0] += 4096
            return fetched_bytes[0]
        wrappers._process_read_bytes = more_fetched
        saved_allowed, saved_learn = wrappers._extras_allowed, wrappers._extras_learn
        wrappers._extras_allowed, wrappers._extras_learn = lambda size: True, lambda size, used, rate: None
        gc.disable()
        found = []
        for k in range(3):
            gc.collect()
            got = cppzarr.read_zarr_batch(many_small, num_threads=4)
            found.append(gc.collect())
            good = good and all(np.array_equal(g, x) for g, x in zip(got, want_small))
            good = good and (wrappers._last_batch[0] > 4 or not wrappers.pybind11_read_times_available())
        good = good and found[1:] == [0, 0]
        if found[1:] != [0, 0]:
            print(f'  a batch left {found} objects for the garbage collector')
    finally:
        if collecting:
            gc.enable()
        wrappers._READS_PER_THREAD, wrappers.pybind11_on_network, wrappers._readers_wanted = saved_rpt, saved_net, saved_wanted
        wrappers._wait_share[0], wrappers._process_read_bytes = saved_share, saved_fetched
        wrappers._extras_allowed, wrappers._extras_learn = saved_allowed, saved_learn
    # a batch of one array starts no openers (its first read opens it), a batch of several arrays does
    opener_calls = [0]
    real_opener = wrappers._opener

    def counting_opener():
        opener_calls[0] += 1
        return real_opener()
    wrappers._opener = counting_opener
    try:
        got = cppzarr.read_zarr_batch([(again, [0, 0, 0], [10, 10, 10])] * 8, num_threads=2)
        good = good and opener_calls[0] == 0 and all(np.array_equal(g, second[:10, :10, :10]) for g in got)
        got = cppzarr.read_zarr_batch(requests[:12], num_threads=2)
        good = good and opener_calls[0] > 0 and all(np.array_equal(g, x) for g, x in zip(got, expected[:12]))
    finally:
        wrappers._opener = real_opener
    big = os.path.join(tmp, "gil.zarr")
    cppzarr.write_zarr(big, values_like(rng, np.zeros((512, 512, 128), np.uint16)), chunks=[128, 128, 64])
    count = [0]
    def spin(seconds):
        end = time.perf_counter() + seconds
        while time.perf_counter() < end:
            count[0] += 1
    spin(0.2)
    alone = count[0] / 0.2
    reader = threading.Thread(target=lambda: [cppzarr.read_zarr(big, num_threads=1) for _ in range(3)])
    count[0] = 0
    t0 = time.perf_counter()
    reader.start()
    while reader.is_alive():
        count[0] += 1
    rate = count[0] / (time.perf_counter() - t0)
    good = good and rate > 0.1 * alone
    print(f"batches of reads, and the GIL released while reading  {'OK' if good else 'FAIL'}")
    ok = ok and good

    # The buffers kept between calls: a write of two chunks of 8 MB keeps their buffers (up to
    # CPPZARR_KEEP_MB with the reads', by default 1 GiB or an eighth of the memory the process
    # may use), release_buffers() frees them, and reads and writes after it give the same data
    from cppzarr.cppzarr import pybind11_kept_bytes
    keep = os.path.join(tmp, "kept_release.zarr")
    kv = rng.integers(0, 60000, size=(8, 1024, 1024), dtype=np.uint16)
    cppzarr.write_zarr(keep, kv, chunks=(4, 1024, 1024), cname='zstd', zarr_format=3)
    kept, reads, cap = pybind11_kept_bytes()
    keeps = cap >= 64 << 20
    good = (not keeps or kept > 0) and reads <= kept <= cap and same(cppzarr.read_zarr(keep), kv)
    cppzarr.release_buffers()
    after, _, _ = pybind11_kept_bytes()
    good = good and after == 0 and same(cppzarr.read_zarr(keep), kv)
    kv = kv[::-1].copy()
    shutil.rmtree(keep, ignore_errors=True)
    cppzarr.write_zarr(keep, kv, chunks=(4, 1024, 1024), cname="lz4", zarr_format=3)
    again, _, _ = pybind11_kept_bytes()
    good = good and (not keeps or again > 0) and same(cppzarr.read_zarr(keep), kv)
    cppzarr.release_buffers()
    shutil.rmtree(keep, ignore_errors=True)
    print(f"buffers kept between calls, and release_buffers  {'OK' if good else 'FAIL'}"
          f"{'' if good else f' (kept {kept} of {cap}, after {after}, again {again})'}")
    ok = ok and good

    # Reading into GPU memory (with an NVIDIA GPU, nvCOMP, and CuPy or PyTorch): the same
    # values as reading into host memory, for every data type and codec, sharded or not,
    # regions that cut chunks, missing chunks, both orders, CuPy arrays and PyTorch tensors,
    # batches of several arrays; and the outputs refused
    gpu_ok, why = cppzarr.gpu_available()
    libs = []
    if gpu_ok:
        try:
            import cupy
            libs.append('cupy')
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                libs.append('torch')
        except Exception:
            pass
    if not gpu_ok or not libs:
        print(f"reading into GPU memory  skipped ({why or 'no CuPy or PyTorch'})")
        print(f"writing from GPU memory  skipped ({why or 'no CuPy or PyTorch'})")
    else:
        ok = ok and test_gpu(tmp, rng, libs)
        ok = ok and test_gpu_write(tmp, rng, libs)

    if not ok:
        sys.exit("cppzarr Python round-trip test FAILED")
    print("cppzarr Python round-trip tests PASSED")


def test_gpu(tmp, rng, libs):
    """GPU reads against host reads (see main)"""
    import cupy
    try:
        import torch
    except ImportError:
        torch = None

    def gpu_empty(lib, shape, dt, order):
        if lib == 'cupy':
            return cupy.empty(shape, dtype=dt, order=order)
        tdt = getattr(torch, 'bfloat16' if is_bfloat16(dt) else np.dtype(dt).name)
        if order == 'C':
            return torch.empty(shape, dtype=tdt, device='cuda')
        return torch.empty(shape[::-1], dtype=tdt, device='cuda').permute(*range(len(shape) - 1, -1, -1))

    def host_bytes(lib, a):
        # the output's bytes in its own order, to compare as they are (NaNs too)
        if lib == 'cupy':
            h = a.get(order='A')
        else:
            t = a.cpu()
            if t.dtype == torch.bfloat16:
                t = t.view(torch.int16)
            h = t.numpy()
        return np.asarray(h).tobytes(order='A')

    good = True
    dtypes = [np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.float16, np.float32, np.float64,
              np.complex64, np.complex128, np.bool_]
    if ml_dtypes is not None:
        dtypes.append(ml_dtypes.bfloat16)
    codecs = [('zstd', 1), ('lz4', 5), ('blosclz', 5), ('gzip', 5), ('none', 0)]
    shape = (70, 45, 33)
    arrays = []
    for dt in dtypes:
        d = values_like(rng, np.zeros(shape, dt))
        for cname, clevel in codecs:
            for zf, shards in [(2, None), (3, None), (3, [32, 32, 16])]:
                p = os.path.join(tmp, f"gpu_{np.dtype(dt).name}_{cname}_{zf}_{bool(shards)}.zarr")
                cppzarr.write_zarr(p, d, chunks=[16, 16, 8], cname=cname, clevel=clevel, zarr_format=zf, shards=shards)
                arrays.append((p, dt))
    # some chunks (or shards) missing
    for p, dt in arrays[::7]:
        files = sorted(os.path.join(r, f) for r, _, fs in os.walk(p) for f in fs if f not in ('zarr.json', '.zarray', '.zattrs'))
        for f in files[::4]:
            os.remove(f)
    regions = [([0, 0, 0], list(shape)), ([5, 7, 3], [61, 40, 29]), ([17, 17, 9], [18, 30, 10])]
    tried = 0
    for p, dt in arrays:
        for s, e in regions:
            for order in 'CF':
                ref = cppzarr.read_zarr(p, s, e, order=order)
                want = ref.tobytes(order='A')
                for lib in libs:
                    if lib == 'cupy' and is_bfloat16(dt):
                        continue      # (CuPy has no bfloat16)
                    out = gpu_empty(lib, [b - a for a, b in zip(s, e)], dt, order)
                    try:
                        got = cppzarr.read_zarr(p, s, e, out=out)
                    except Exception as ex:
                        # (a failed GPU read can leave the GPU unusable for the reads after it)
                        print('  GPU read failed:', lib, p, s, e, order, ex)
                        return False
                    tried += 1
                    if got is not out or host_bytes(lib, out) != want:
                        good = False
                        print('  GPU read differs:', lib, p, s, e, order)
                # a new CuPy array (gpu=True) in order; CuPy has no bfloat16
                try:
                    got = cppzarr.read_zarr(p, s, e, order=order, gpu=True)
                except Exception as ex:
                    if not (is_bfloat16(dt) and 'bfloat16' in str(ex)):
                        good = False
                        print('  gpu=True read failed:', p, s, e, order, ex)
                    continue
                tried += 1
                laid_out = got.flags.c_contiguous if order == 'C' else got.flags.f_contiguous
                if (is_bfloat16(dt) or not isinstance(got, cupy.ndarray) or got.dtype != ref.dtype or got.shape != ref.shape
                        or not laid_out or host_bytes('cupy', got) != want):
                    good = False
                    print('  gpu=True read differs:', p, s, e, order)
    # whole arrays into new CuPy arrays in their storage order, as read_zarr's NumPy arrays; and
    # a batch of new ones
    for p, dt in arrays[:60]:
        if is_bfloat16(dt):
            continue
        ref, got = cppzarr.read_zarr(p), cppzarr.read_zarr(p, gpu=True)
        tried += 1
        if (got.dtype != ref.dtype or got.shape != ref.shape or got.flags.c_contiguous != ref.flags.c_contiguous
                or got.flags.f_contiguous != ref.flags.f_contiguous or host_bytes('cupy', got) != ref.tobytes(order='A')):
            good = False
            print('  gpu=True whole-array read differs:', p)
    batch = [(p, [5, 7, 3], [61, 40, 29]) for p, dt in arrays[60:100] if not is_bfloat16(dt)]
    got = cppzarr.read_zarr_batch(batch, gpu=True, num_threads=4)
    good = good and all(isinstance(g, cupy.ndarray) and host_bytes('cupy', g) == cppzarr.read_zarr(p, s, e).tobytes(order='A')
                        for g, (p, s, e) in zip(got, batch))
    # a batch from several arrays (different data types and codecs) into one CuPy buffer each
    batch = [(p, [5, 7, 3], [61, 40, 29]) for p, dt in arrays[:40] if not is_bfloat16(dt)]
    outs = [cupy.empty((56, 33, 26), dtype=cppzarr.read_zarr(p, [0, 0, 0], [1, 1, 1]).dtype) for p, _, _ in batch]
    got = cppzarr.read_zarr_batch([r + (o,) for r, o in zip(batch, outs)], num_threads=4)
    good = good and all(g is o for g, o in zip(got, outs)) and all(
        np.array_equal(o.get(), cppzarr.read_zarr(p, s, e, order='C')) for o, (p, s, e) in zip(outs, batch))
    if torch is not None and 'torch' in libs:
        st = torch.cuda.Stream()
        o = torch.empty((56, 33, 26), dtype=torch.uint16, device='cuda')
        p = [q for q, dt in arrays if dt == np.uint16][0]
        cppzarr.read_zarr(p, [5, 7, 3], [61, 40, 29], out=o, stream=st)
        good = good and np.array_equal(o.cpu().numpy(), cppzarr.read_zarr(p, [5, 7, 3], [61, 40, 29], order='C'))
    # outputs from allocators whose memory belongs to no CUDA context (stream-ordered pools, CUDA's
    # virtual memory API): CuPy's async pool, and PyTorch's expandable segments and cudaMallocAsync
    # backend, each in a process of its own (PyTorch takes its allocator setting at its first
    # allocation)
    p = [q for q, dt in arrays if dt == np.uint16][0]
    want = cppzarr.read_zarr(p, [5, 7, 3], [61, 40, 29], order='C')
    try:
        pool = cupy.cuda.MemoryAsyncPool()
    except Exception:
        pool = None     # (no stream-ordered pools on this GPU or CUDA)
    if pool is not None:
        with cupy.cuda.using_allocator(pool.malloc):
            o = cupy.empty(want.shape, want.dtype)
        cppzarr.read_zarr(p, [5, 7, 3], [61, 40, 29], out=o)
        tried += 1
        if not np.array_equal(o.get(), want):
            good = False
            print('  GPU read into memory from a CuPy async pool differs')
    if torch is not None and 'torch' in libs:
        code = ("import sys, numpy as np, torch, cppzarr\n"
                "o = torch.empty((56, 33, 26), dtype=torch.uint16, device='cuda')\n"
                "cppzarr.read_zarr(sys.argv[1], [5, 7, 3], [61, 40, 29], out=o)\n"
                "w = cppzarr.read_zarr(sys.argv[1], [5, 7, 3], [61, 40, 29], order='C')\n"
                "print('SAME' if np.array_equal(o.cpu().numpy(), w) else 'DIFFERENT')\n")
        for conf in ('expandable_segments:True', 'backend:cudaMallocAsync'):
            r = subprocess.run([sys.executable, '-c', code, p], env=dict(os.environ, PYTORCH_CUDA_ALLOC_CONF=conf),
                               capture_output=True, text=True)
            tried += 1
            if 'SAME' not in r.stdout:
                good = False
                print('  GPU read into PyTorch memory with', conf, 'failed:', (r.stderr.strip().splitlines() or [r.stdout])[-1])
    # regions whose chunks' stored bytes run over several of the reads' pieces (1 MiB, 4 MiB in runs
    # of 8 MiB or more: such a run's file is opened once for all its pieces), many of one array (its
    # metadata read once for the batch), in one batch with regions of another array (of as many
    # dimensions: a GPU batch's arrays all have the same number)
    runs = os.path.join(tmp, 'gpu_runs.zarr')
    dr = (np.arange(3 * 128 * 192 * 160, dtype=np.uint32).reshape(3, 128, 192, 160) * 2654435761 >> 20).astype(np.uint16)
    cppzarr.write_zarr(runs, dr, chunks=[1, 32, 64, 64], shards=[3, 128, 192, 192], zarr_format=3, cname='zstd', clevel=1)
    runs2 = os.path.join(tmp, 'gpu_runs2.zarr')
    cppzarr.write_zarr(runs2, dr[:, :64, :96, :80] ^ 0x5a5a, chunks=[1, 16, 32, 32], shards=[3, 64, 96, 96], zarr_format=3,
                       cname='zstd', clevel=1)
    batch = [(runs, [t, 0, 0, 0], [t + 1, 128, 192, 160]) for t in range(3)] * 2 + [(runs, [0, 17, 5, 9], [3, 100, 190, 151])]
    batch += [(runs2, [0, 0, 0, 0], [3, 64, 96, 80]), (runs2, [1, 3, 5, 7], [2, 60, 90, 77])]
    outs = [cupy.empty(tuple(b - a for a, b in zip(s, e)), dtype=np.uint16) for q, s, e in batch]
    got = cppzarr.read_zarr_batch([r + (o,) for r, o in zip(batch, outs)], num_threads=4)
    tried += len(batch)
    if not all(np.array_equal(o.get(), cppzarr.read_zarr(q, s, e, order='C')) for o, (q, s, e) in zip(outs, batch)):
        good = False
        print('  GPU batch over runs of several pieces differs')
    # refused: another shape, another data type, a batch of GPU and host outputs, an output
    # that is not contiguous, a batch with an array that is not there, gpu=True with a host
    # output (alone, and in a batch)
    p = [q for q, dt in arrays if dt == np.uint16][0]
    for bad in (lambda: cppzarr.read_zarr(p, [0, 0, 0], [10, 10, 10], out=cupy.empty((10, 10, 9), np.uint16)),
                lambda: cppzarr.read_zarr(p, [0, 0, 0], [10, 10, 10], out=cupy.empty((10, 10, 10), np.int16)),
                lambda: cppzarr.read_zarr_batch([(p, [0, 0, 0], [10, 10, 10], cupy.empty((10, 10, 10), np.uint16)),
                                                 (p, [0, 0, 0], [10, 10, 10], np.empty((10, 10, 10), np.uint16))]),
                lambda: cppzarr.read_zarr(p, [0, 0, 0], [10, 10, 10], out=cupy.empty((10, 10, 20), np.uint16)[:, :, ::2]),
                lambda: cppzarr.read_zarr_batch([(p, [0, 0, 0], [10, 10, 10], cupy.empty((10, 10, 10), np.uint16)),
                                                 (os.path.join(tmp, 'missing.zarr'), [0, 0, 0], [10, 10, 10],
                                                  cupy.empty((10, 10, 10), np.uint16))]),
                lambda: cppzarr.read_zarr(p, [0, 0, 0], [10, 10, 10], out=np.empty((10, 10, 10), np.uint16), gpu=True),
                lambda: cppzarr.read_zarr_batch([(p, [0, 0, 0], [10, 10, 10]),
                                                 (p, [0, 0, 0], [10, 10, 10], np.empty((10, 10, 10), np.uint16))], gpu=True)):
        try:
            bad()
            good = False
            print('  a GPU output that should be refused was not')
        except Exception:
            pass
    print(f"reading into GPU memory ({' and '.join(libs)}, {tried} reads)  {'OK' if good else 'FAIL'}")
    return good


def test_gpu_write(tmp, rng, libs):
    """GPU writes against host writes: every data type (bfloat16 from PyTorch), compressors the GPU
    compresses (blosc's zstd and lz4, none) and ones written through host memory (blosclz, gzip),
    both formats, sharded or not, CuPy arrays and PyTorch tensors in C order, F order and strided,
    then regions of other data written into the arrays (converted to their data type on the GPU),
    and a write on a PyTorch stream: the arrays must read the same and hold the same chunk files"""
    try:
        import torch
    except ImportError:
        torch = None
    if 'torch' not in libs:
        torch = None
    cupy = None
    if 'cupy' in libs:
        import cupy

    def chunk_files(p):
        return sorted(os.path.relpath(os.path.join(d, f), p) for d, _, fs in os.walk(p) for f in fs
                      if f not in ('.zarray', 'zarr.json', '.zattrs'))

    def to_gpu(lib, a, layout):
        # a host array on the GPU, C-contiguous (layout 0), F-contiguous (1) or a strided view (2)
        if lib == 'cupy':
            if layout == 2:
                big = cupy.zeros(a.shape[:-1] + (2 * a.shape[-1],), dtype=a.dtype)
                big[..., ::2] = cupy.asarray(a)
                return big[..., ::2]
            return cupy.asarray(np.asarray(a, order='F' if layout == 1 else 'C'))
        tdt = getattr(torch, 'bfloat16' if is_bfloat16(a.dtype) else np.dtype(a.dtype).name)
        src = torch.from_numpy(np.ascontiguousarray(a).view(np.int16) if is_bfloat16(a.dtype) else np.ascontiguousarray(a))
        if is_bfloat16(a.dtype):
            src = src.view(torch.bfloat16)
        t = src.to('cuda')
        assert t.dtype == tdt
        if layout == 1:
            return t.permute(*range(t.ndim - 1, -1, -1)).contiguous().permute(*range(t.ndim - 1, -1, -1))
        if layout == 2:
            big = torch.zeros(t.shape[:-1] + (2 * t.shape[-1],), dtype=t.dtype, device='cuda')
            big[..., ::2] = t
            return big[..., ::2]
        return t

    good, n = True, 0
    shape = (45, 33, 26)
    dtypes = [np.uint8, np.int8, np.uint16, np.int16, np.uint32, np.int32, np.uint64, np.int64, np.float16, np.float32,
              np.float64, np.complex64, np.complex128, np.bool_]
    try:
        import ml_dtypes
        if torch is not None:
            dtypes.append(ml_dtypes.bfloat16)
    except ImportError:
        pass
    configs = [('zstd', 2, None), ('lz4', 2, None), ('zstd', 3, None), ('zstd', 3, [32, 32, 26]), ('lz4', 3, [32, 32, 26]),
               ('blosclz', 2, None), ('gzip', 3, None), ('none', 3, [32, 32, 26])]
    for k, dt in enumerate(dtypes):
        if np.dtype(dt) == np.bool_:
            data = rng.random(shape) > 0.5
        elif np.dtype(dt).kind == 'c':
            data = (rng.integers(0, 200, shape) + 1j * rng.integers(0, 50, shape)).astype(dt)
        else:
            data = rng.integers(0, 100, shape).astype(dt)
        data[:16, :16, :16] = 0
        part = data[3:40, 5:30, 2:20][::-1].copy()
        for c, (cname, zf, shards) in enumerate(configs):
            lib = libs[(k + c) % len(libs)]
            if is_bfloat16(dt) and lib != 'torch':
                continue
            layout = (k + c) % 3
            pc = os.path.join(tmp, f"gw_{np.dtype(dt).name}_{cname}_{zf}_{bool(shards)}_cpu.zarr")
            pg = os.path.join(tmp, f"gw_{np.dtype(dt).name}_{cname}_{zf}_{bool(shards)}_gpu.zarr")
            args = dict(cname=cname, zarr_format=zf, chunks=[16, 16, 13], shards=shards)
            try:
                cppzarr.write_zarr(pc, data, **args)
                cppzarr.write_zarr(pg, to_gpu(lib, data, layout), num_threads=4, **args)
                cppzarr.write_zarr(pc, part, [5, 3, 1], [42, 28, 19])
                cppzarr.write_zarr(pg, to_gpu(lib, part, (layout + 1) % 3), [5, 3, 1], [42, 28, 19])
            except Exception as ex:
                good = False
                print('  GPU write failed:', lib, pg, ex)
                continue
            n += 2
            a, b = cppzarr.read_zarr(pc), cppzarr.read_zarr(pg)
            if a.dtype != b.dtype or a.tobytes() != b.tobytes() or chunk_files(pc) != chunk_files(pg):
                good = False
                print('  GPU write differs from the host write:', lib, pg)
    # regions of other data converted to the array's data type (floats into uint16, with halves and values past
    # its limits), as host data is: as MATLAB converts (matlab_cast), in host memory, then written from there
    for lib in libs:
        pc, pg = os.path.join(tmp, f'gw_conv_{lib}_cpu.zarr'), os.path.join(tmp, f'gw_conv_{lib}_gpu.zarr')
        data = rng.integers(0, 1000, shape).astype(np.uint16)
        part = (rng.random((20, 20, 10)) * 999).astype(np.float32)
        part[0, 0, :4] = [2.5, -1.5, 70000, 0.5]
        cppzarr.write_zarr(pc, data, zarr_format=3)
        cppzarr.write_zarr(pg, to_gpu(lib, data, 0), zarr_format=3)
        cppzarr.write_zarr(pc, part, [5, 5, 5], [25, 25, 15])
        cppzarr.write_zarr(pg, to_gpu(lib, part, 0), [5, 5, 5], [25, 25, 15])
        n += 1
        exp = data.copy()
        exp[5:25, 5:25, 5:15] = matlab_cast(part, np.uint16)
        if cppzarr.read_zarr(pc).tobytes() != cppzarr.read_zarr(pg).tobytes() or not np.array_equal(cppzarr.read_zarr(pg), exp):
            good = False
            print('  GPU write of data converted to the array\'s data type differs:', lib)
    # on a PyTorch stream, data made on it
    if torch is not None:
        st = torch.cuda.Stream()
        with torch.cuda.stream(st):
            t = torch.arange(45 * 33 * 26, device='cuda', dtype=torch.int32).reshape(shape) % 1000
            p = os.path.join(tmp, 'gw_stream.zarr')
            cppzarr.write_zarr(p, t, zarr_format=3, stream=st)
        n += 1
        if not np.array_equal(cppzarr.read_zarr(p), (np.arange(45 * 33 * 26, dtype=np.int32).reshape(shape) % 1000)):
            good = False
            print('  GPU write on a PyTorch stream differs')
        # (no stream given: the current one, where the data was made, after the GPU spends a while
        # on it: a write on another stream would find zeros)
        with torch.cuda.stream(st):
            t = torch.zeros(shape, device='cuda', dtype=torch.int16)
            torch.cuda._sleep(100_000_000)
            t.fill_(3)
            p = os.path.join(tmp, 'gw_current_stream.zarr')
            cppzarr.write_zarr(p, t, zarr_format=3)
        n += 1
        if not np.array_equal(cppzarr.read_zarr(p), np.full(shape, 3, np.int16)):
            good = False
            print('  GPU write of data made on the current PyTorch stream differs')
    print(f"writing from GPU memory ({' and '.join(libs)}, {n} writes)  {'OK' if good else 'FAIL'}")
    return good


if __name__ == "__main__":
    main()
