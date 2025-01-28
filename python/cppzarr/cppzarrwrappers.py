import numpy as np
import os
from .cppzarr import pybind11_read_zarr, pybind11_write_zarr


def read_zarr(file_name, start_coords=None, end_coords=None):
    if not os.path.isfile(os.path.join(file_name, '.zarray')):
        raise Exception(f'{file_name} does not exist. The .zarray metadata file was not found')
    if start_coords is None:
        start_coords = [0, 0, 0]
    if end_coords is None:
        end_coords = [0, 0, 0]
    im = pybind11_read_zarr(file_name, start_coords, end_coords)
    return im


def write_zarr(file_name, data, start_coords=None, end_coords=None, cname='zstd', clevel=1, order='F', chunks=None,
               dimension_separator='.'):
    crop = False
    if start_coords is not None or end_coords is not None:
        crop = True
    if chunks is None:
        chunks = [256, 256, 256]
    if start_coords is None:
        start_coords = [0, 0, 0]
    if end_coords is None:
        end_coords = [data.shape[0], data.shape[1], data.shape[2]]
    if end_coords[0]-start_coords[0] <= 0 or end_coords[1]-start_coords[1] <= 0 or end_coords[2]-start_coords[2] <= 0:
        raise Exception(f'Invalid start_coords or end_coords!')
    if data.flags['C_CONTIGUOUS'] or not data.flags['F_CONTIGUOUS']:
        data = np.asfortranarray(data)
    pybind11_write_zarr(file_name, data, start_coords, end_coords, cname, clevel, order, chunks, dimension_separator, crop)
    return
