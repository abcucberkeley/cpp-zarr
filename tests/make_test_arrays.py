"""Make the test arrays in tests/test_arrays (not committed: CI makes them before the tests,
which skip them elsewhere when they are missing): small Zarr v3 arrays written by zarr-python 3
(the reference implementation) and TensorStore, v2 arrays written by zarr-python with codecs
and fill values cpp-zarr does not write itself, the values zarr-python reads back from each
(C-order little-endian bytes in <name>.bin), and arrays cpp-zarr must reject. arrays.json
lists them; the C++, Python and MATLAB tests read every array and compare.

    python make_test_arrays.py        (needs zarr>=3, numcodecs, tensorstore and numpy)
"""
import json
import os
import shutil

import numcodecs
import numpy as np
import tensorstore as ts
import zarr
from zarr.codecs import (BloscCodec, BytesCodec, Crc32cCodec, GzipCodec, ShardingCodec, TransposeCodec,
                         ZstdCodec)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'test_arrays')
SHAPE, CHUNKS = (10, 7, 5), (4, 4, 5)
rng = np.random.default_rng(20261002)
fixtures = []


def values(dtype, shape=SHAPE):
    dtype = np.dtype(dtype)
    if dtype.kind == 'b':
        return rng.random(shape) < 0.5
    if dtype.kind == 'f':
        v = rng.standard_normal(shape) * 1000
        v.flat[::17] = np.nan
        return v.astype(dtype)
    info = np.iinfo(dtype)
    return rng.integers(max(info.min, -10**9), min(info.max, 10**12), size=shape, endpoint=True).astype(dtype)


def path(name):
    return os.path.join(OUT, name + '.zarr')


def record(name):
    # The values zarr-python reads, as C-order little-endian bytes
    a = zarr.open_array(path(name), mode='r')[...]
    a = np.array(a, order='C')      # (a 0-dimensional array stays 0-dimensional)
    if a.dtype.itemsize > 1:
        a = a.astype(a.dtype.newbyteorder('<'))
    a.tofile(os.path.join(OUT, name + '.bin'))
    fixtures.append({'name': name, 'dtype': a.dtype.str, 'shape': list(a.shape)})


def array(name, dtype, write='full', shape=SHAPE, chunks=CHUNKS, **kwargs):
    a = zarr.create_array(store=path(name), shape=shape, chunks=chunks, dtype=dtype, **kwargs)
    v = values(dtype, shape)
    if write == 'full':
        a[...] = v
    elif write == 'partial':           # leave the chunks along the last rows missing
        a[0:8] = v[0:8]
    elif write == 'scattered':         # two corners: whole shards and inner chunks missing
        a[0:4, 0:4] = v[0:4, 0:4]
        a[8:10, 4:7] = v[8:10, 4:7]
    record(name)


def edit(name, change):
    with open(os.path.join(path(name), 'zarr.json')) as f:
        meta = json.load(f)
    change(meta)
    with open(os.path.join(path(name), 'zarr.json'), 'w') as f:
        json.dump(meta, f, indent=2)


def rejected(name, error, make):
    make()
    for f in os.listdir(path(name)):     # only the metadata is needed
        if f not in ('zarr.json', '.zarray', '.zattrs'):
            p = os.path.join(path(name), f)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    fixtures.append({'name': name, 'error': error})


shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT)

# zarr-python's defaults (bytes, zstd; "c/0/0/0" chunk keys), with missing chunks
array('v3_default_uint16', 'uint16', 'partial')
# blosc, the main codec: zstd with byte shuffle, lz4 with bit shuffle, blosclz without
array('v3_blosc_zstd_uint16', 'uint16', compressors=[BloscCodec(cname='zstd', clevel=5, shuffle='shuffle')])
array('v3_blosc_lz4_bitshuffle_int32', 'int32', compressors=[BloscCodec(cname='lz4', clevel=1, shuffle='bitshuffle')])
array('v3_blosc_blosclz_noshuffle_float64', 'float64', 'partial', fill_value=1.5,
      compressors=[BloscCodec(cname='blosclz', clevel=3, shuffle='noshuffle')])
array('v3_gzip_int16', 'int16', 'partial', fill_value=-2, compressors=[GzipCodec(level=5)])
array('v3_crc32c_uncompressed_int8', 'int8', 'partial', fill_value=-3, compressors=[Crc32cCodec()])
array('v3_zstd_checksum_crc32c_uint64', 'uint64', compressors=[ZstdCodec(level=3, checksum=True), Crc32cCodec()])
array('v3_big_endian_uint32', 'uint32', 'partial', fill_value=7, serializer=BytesCodec(endian='big'),
      compressors=[ZstdCodec(level=1)])
# transpose: F order, and another axis order
array('v3_transpose_f_float32', 'float32', 'partial', fill_value=float('nan'), filters=[TransposeCodec(order=(2, 1, 0))],
      compressors=[BloscCodec(cname='zstd', clevel=1, shuffle='shuffle')])
array('v3_transpose_120_uint16', 'uint16', filters=[TransposeCodec(order=(1, 2, 0))], compressors=[ZstdCodec(level=1)])
# chunk key encodings
array('v3_v2_keys_dot_uint8', 'uint8', 'partial', chunk_key_encoding={'name': 'v2', 'separator': '.'},
      compressors=[BloscCodec(cname='lz4', clevel=1, shuffle='shuffle')])
array('v3_v2_keys_slash_int64', 'int64', chunk_key_encoding={'name': 'v2', 'separator': '/'}, compressors=[GzipCodec(level=1)])
array('v3_default_keys_dot_float64', 'float64', chunk_key_encoding={'name': 'default', 'separator': '.'},
      compressors=[ZstdCodec(level=1)])
# sharding: zarr-python's shards (index at the end with a checksum), and an index at the
# start without one around big-endian F-order inner chunks
array('v3_sharded_blosc_zstd_uint16', 'uint16', 'scattered', chunks=(4, 2, 5), shards=(8, 4, 5),
      compressors=[BloscCodec(cname='zstd', clevel=1, shuffle='shuffle')])
array('v3_sharded_index_start_float32', 'float32', 'scattered', chunks=(8, 4, 5), fill_value=float('nan'), compressors=None,
      serializer=ShardingCodec(chunk_shape=(4, 2, 5), index_codecs=[BytesCodec()], index_location='start',
                               codecs=[TransposeCodec(order=(2, 1, 0)), BytesCodec(endian='big'), ZstdCodec(level=3)]))
# other data types and numbers of dimensions
array('v3_bool_mask', 'bool', 'partial', fill_value=False)
array('v3_one_d_uint64', 'uint64', shape=(1000,), chunks=(128,))
array('v3_four_d_int16', 'int16', 'partial', shape=(9, 3, 6, 4), chunks=(4, 2, 4, 3),
      compressors=[BloscCodec(cname='lz4', clevel=1, shuffle='shuffle')])
array('v3_zero_d_float64', 'float64', shape=(), chunks=())
array('v3_zero_d_missing_int32', 'int32', 'none', shape=(), chunks=(), fill_value=9)
# a float fill value given as its bits (pi as float32), for missing chunks
array('v3_hex_fill_float32', 'float32', 'partial', compressors=[ZstdCodec(level=1)])
edit('v3_hex_fill_float32', lambda m: m.update(fill_value='0x40490fdb'))
os.remove(os.path.join(OUT, 'v3_hex_fill_float32.bin'))
fixtures.pop()
record('v3_hex_fill_float32')

# TensorStore: its default codecs, and F-order blosc shards with the index at the start
for name, codecs in [('v3_tensorstore_default_int32', None),
                     ('v3_tensorstore_sharded_f_uint16', [{'name': 'sharding_indexed', 'configuration': {
                         'chunk_shape': [4, 2, 5], 'index_location': 'start',
                         'codecs': [{'name': 'transpose', 'configuration': {'order': [2, 1, 0]}},
                                    {'name': 'bytes', 'configuration': {'endian': 'little'}},
                                    {'name': 'blosc', 'configuration': {'cname': 'zstd', 'clevel': 1, 'shuffle': 'shuffle',
                                                                        'typesize': 2, 'blocksize': 0}}]}}])]:
    dtype = 'int32' if 'int32' in name else 'uint16'
    meta = {'shape': list(SHAPE), 'data_type': dtype,
            'chunk_grid': {'name': 'regular', 'configuration': {'chunk_shape': [8, 4, 5] if codecs else list(CHUNKS)}}}
    if codecs:
        meta['codecs'] = codecs
    t = ts.open({'driver': 'zarr3', 'kvstore': {'driver': 'file', 'path': path(name)}, 'metadata': meta},
                create=True, delete_existing=True).result()
    t[0:8].write(values(dtype)[0:8]).result()
    record(name)

# Zarr v2 arrays from zarr-python: numcodecs compressors besides blosc and gzip (read only),
# F order, fill values that missing chunks read as (7, 1.5, NaN, True), and booleans
def v2array(name, dtype, write='full', **kwargs):
    array(name, dtype, write, zarr_format=2, **kwargs)


v2array('v2_numcodecs_zstd_uint16', 'uint16', 'partial', compressors=numcodecs.Zstd(level=1))
v2array('v2_numcodecs_zlib_int32_f', 'int32', order='F', compressors=numcodecs.Zlib(level=1))
v2array('v2_numcodecs_gzip_nan_float64', 'float64', 'partial', fill_value=float('nan'), compressors=numcodecs.GZip(level=1))
v2array('v2_blosc_fill7_uint16', 'uint16', 'partial', fill_value=7,
        compressors=numcodecs.Blosc(cname='zstd', clevel=1, shuffle=numcodecs.Blosc.SHUFFLE))
v2array('v2_blosc_fill1p5_float32', 'float32', 'partial', fill_value=1.5, compressors=numcodecs.Blosc(cname='lz4', clevel=1))
v2array('v2_bool_fill_true', 'bool', 'partial', fill_value=True, compressors=numcodecs.Blosc(cname='lz4', clevel=1))

# Arrays cpp-zarr must reject (with an error that names the reason, not a crash)
rejected('v3_group', 'group', lambda: zarr.create_group(store=path('v3_group')))
rejected('v3_float16', 'float16', lambda: zarr.create_array(store=path('v3_float16'), shape=SHAPE, chunks=CHUNKS, dtype='float16'))
rejected('v3_complex64', 'complex64', lambda: zarr.create_array(store=path('v3_complex64'), shape=SHAPE, chunks=CHUNKS, dtype='complex64'))
rejected('v3_sharded_then_compressed', 'after sharding_indexed', lambda: zarr.create_array(
    store=path('v3_sharded_then_compressed'), shape=SHAPE, chunks=(8, 4, 5), dtype='uint16',
    serializer=ShardingCodec(chunk_shape=(4, 2, 5))))


def unknown_codec():
    zarr.create_array(store=path('v3_unknown_codec'), shape=SHAPE, chunks=CHUNKS, dtype='uint16')
    edit('v3_unknown_codec', lambda m: m['codecs'].append({'name': 'fancy_codec'}))


def storage_transformer():
    zarr.create_array(store=path('v3_storage_transformer'), shape=SHAPE, chunks=CHUNKS, dtype='uint16')
    edit('v3_storage_transformer', lambda m: m.update(storage_transformers=[{'name': 'something'}]))


rejected('v3_unknown_codec', 'fancy_codec', unknown_codec)
rejected('v2_filters_delta', 'filters', lambda: zarr.create_array(
    store=path('v2_filters_delta'), shape=SHAPE, chunks=CHUNKS, dtype='int32', zarr_format=2,
    filters=[numcodecs.Delta(dtype='int32')], compressors=numcodecs.Blosc(cname='lz4')))
rejected('v3_storage_transformer', 'storage transformers', storage_transformer)

with open(os.path.join(OUT, 'arrays.json'), 'w') as f:
    json.dump({'made_with': f'zarr {zarr.__version__}, numcodecs {numcodecs.__version__}, tensorstore, numpy {np.__version__}',
               'arrays': fixtures}, f, indent=1)
print(f'{len(fixtures)} test arrays in {OUT}')
