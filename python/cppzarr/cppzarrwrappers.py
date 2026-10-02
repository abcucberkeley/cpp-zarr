import json
import numpy as np
import os
from .cppzarr import pybind11_convert_to_v3, pybind11_read_zarr, pybind11_write_zarr


def _default_chunks(ndim):
    # 256 along the last three axes (NumPy's usual spatial axes), 1 along any leading ones
    return [1] * max(ndim - 3, 0) + [256] * min(ndim, 3)


def _fit_coords(coords, ndim, is_end, name):
    # Extra trailing values are accepted when they describe singleton axes, so
    # 3-value coordinates keep working on 1D/2D arrays
    coords = [int(c) for c in coords]
    if len(coords) > ndim and all(c <= (1 if is_end else 0) for c in coords[ndim:]):
        coords = coords[:ndim]
    if len(coords) != ndim:
        raise Exception(f'{name} has {len(coords)} values but the array has {ndim} dimensions')
    return coords


def read_zarr(file_name, start_coords=None, end_coords=None, order=None):
    # order is the memory layout of the returned array: 'F' (first axis
    # contiguous), 'C' (last axis contiguous), or None (the default) for the
    # file's own storage order, like zarr-python
    if not (os.path.isfile(os.path.join(file_name, '.zarray')) or os.path.isfile(os.path.join(file_name, 'zarr.json'))):
        raise Exception(f'{file_name} does not exist. No .zarray or zarr.json metadata file was found')
    if order not in (None, 'F', 'C'):
        raise Exception(f"order must be 'F', 'C' or None, not {order!r}")
    # All-zero coordinates mean the whole array, for any number of dimensions
    if start_coords is None:
        start_coords = [0, 0, 0]
    if end_coords is None:
        end_coords = [0, 0, 0]
    im = pybind11_read_zarr(file_name, list(start_coords), list(end_coords), order or '')
    return im


def _metadata_dtype(meta):
    # The element type of an array's metadata (.zarray dtype, or zarr.json
    # data_type), in native byte order; None if NumPy cannot describe it as a
    # number type
    try:
        dt = np.dtype(meta['dtype'] if 'dtype' in meta else meta['data_type'])
    except Exception:
        return None
    return dt.newbyteorder('=') if dt.kind in 'uif' else None


def _existing_format(file_name):
    # The Zarr format of the array at file_name (2 or 3), or None if there is none
    if os.path.isfile(os.path.join(file_name, '.zarray')):
        return 2
    if os.path.isfile(os.path.join(file_name, 'zarr.json')):
        return 3
    return None


def write_zarr(file_name, data, start_coords=None, end_coords=None, cname='zstd', clevel=1, order=None, chunks=None,
               dimension_separator=None, zarr_format=None, shards=None):
    # zarr_format: 2 or 3. By default the format of the array already at file_name,
    # or 2 for a new one. Zarr v3 arrays default to C order and chunk files named
    # c/0/0/0 (order='C', dimension_separator='/'), v2 ones to F order and 0.0.0.
    # shards (Zarr v3 only) is the shard shape: chunks then sets the inner chunks.
    crop = start_coords is not None or end_coords is not None
    existing = _existing_format(file_name)
    if zarr_format not in (None, 2, 3):
        raise Exception(f'zarr_format must be 2 or 3, not {zarr_format!r}')
    if crop and existing is not None and zarr_format is not None and zarr_format != existing:
        raise Exception(f'{file_name} is a Zarr v{existing} array; writing a region keeps its format'
                        + (' (convert it with convert_to_v3 first)' if existing == 2 else ''))
    zarr_format = zarr_format or existing or 2
    if shards is not None and zarr_format != 3:
        raise Exception('shards needs zarr_format=3')
    if order is None:
        order = 'C' if zarr_format == 3 else 'F'
    if dimension_separator is None:
        dimension_separator = '/' if zarr_format == 3 else '.'

    # Writing into an existing array uses its number of dimensions (the data may
    # omit trailing singleton axes) and its data type: the data is converted to it,
    # as the MATLAB writer does
    ndim = data.ndim
    if crop and existing is not None:
        with open(os.path.join(file_name, '.zarray' if existing == 2 else 'zarr.json')) as f:
            meta = json.load(f)
        ndim = len(meta['shape'])
        dtype = _metadata_dtype(meta)
        if dtype is not None and data.dtype != dtype:
            data = np.asarray(data).astype(dtype)
    # (the writer takes data in this machine's byte order)
    if not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder('='))
    if data.ndim > ndim:
        raise Exception(f'The data has {data.ndim} dimensions but the array has {ndim}')
    data_shape = list(data.shape) + [1] * (ndim - data.ndim)

    if chunks is None:
        chunks = _default_chunks(ndim)
    chunks = [int(c) for c in chunks]
    if len(chunks) < ndim:
        raise Exception(f'chunks has {len(chunks)} values but the data has {ndim} dimensions')
    chunks = chunks[:ndim]
    if shards is not None:
        shards = [int(s) for s in shards]
        if len(shards) < ndim:
            raise Exception(f'shards has {len(shards)} values but the data has {ndim} dimensions')
        shards = shards[:ndim]
        if any(c <= 0 or s % c for s, c in zip(shards, chunks)):
            raise Exception(f'shards {shards} must be a multiple of chunks {chunks} along each axis')

    start_coords = [0] * ndim if start_coords is None else _fit_coords(start_coords, ndim, False, 'start_coords')
    if end_coords is None:
        end_coords = [s + n for s, n in zip(start_coords, data_shape)]
    else:
        end_coords = _fit_coords(end_coords, ndim, True, 'end_coords')
    if any(e - s <= 0 for s, e in zip(start_coords, end_coords)):
        raise Exception(f'Invalid start_coords or end_coords!')
    if [e - s for s, e in zip(start_coords, end_coords)] != data_shape:
        raise Exception(f'The region from start_coords to end_coords does not match the data shape {data.shape}')
    # The writer reads the array in place in any memory layout (C or F order, or a
    # strided view), so it is not converted. Only reversed or repeated axes
    # (negative or zero strides) and misaligned data need a copy first.
    if not data.flags['ALIGNED'] or any(n > 1 and (s <= 0 or s % data.itemsize)
                                        for s, n in zip(data.strides, data.shape)):
        data = np.array(data, order='K')
    pybind11_write_zarr(file_name, data, start_coords, end_coords, cname, clevel, order, chunks, dimension_separator, crop,
                        zarr_format, shards or [])
    return


def convert_to_v3(file_name):
    # Turn the Zarr v2 array at file_name into a Zarr v3 array in place: a zarr.json
    # replaces its .zarray (and takes in its .zattrs attributes), and its chunk files
    # are kept as they are, so no data is rewritten
    if _existing_format(file_name) != 2:
        raise Exception(f'{file_name} is not a Zarr v2 array (no .zarray metadata file was found)')
    pybind11_convert_to_v3(file_name)
