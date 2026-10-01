import json
import numpy as np
import os
from .cppzarr import pybind11_read_zarr, pybind11_write_zarr


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
    if not os.path.isfile(os.path.join(file_name, '.zarray')):
        raise Exception(f'{file_name} does not exist. The .zarray metadata file was not found')
    if order not in (None, 'F', 'C'):
        raise Exception(f"order must be 'F', 'C' or None, not {order!r}")
    # All-zero coordinates mean the whole array, for any number of dimensions
    if start_coords is None:
        start_coords = [0, 0, 0]
    if end_coords is None:
        end_coords = [0, 0, 0]
    im = pybind11_read_zarr(file_name, list(start_coords), list(end_coords), order or '')
    return im


def write_zarr(file_name, data, start_coords=None, end_coords=None, cname='zstd', clevel=1, order='F', chunks=None,
               dimension_separator='.'):
    crop = start_coords is not None or end_coords is not None

    # Writing into an existing array uses its number of dimensions; the data may
    # omit trailing singleton axes
    ndim = data.ndim
    zarray = os.path.join(file_name, '.zarray')
    if crop and os.path.isfile(zarray):
        with open(zarray) as f:
            ndim = len(json.load(f)['shape'])
    if data.ndim > ndim:
        raise Exception(f'The data has {data.ndim} dimensions but the array has {ndim}')
    data_shape = list(data.shape) + [1] * (ndim - data.ndim)

    if chunks is None:
        chunks = _default_chunks(ndim)
    chunks = [int(c) for c in chunks]
    if len(chunks) < ndim:
        raise Exception(f'chunks has {len(chunks)} values but the data has {ndim} dimensions')
    chunks = chunks[:ndim]

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
    pybind11_write_zarr(file_name, data, start_coords, end_coords, cname, clevel, order, chunks, dimension_separator, crop)
    return
