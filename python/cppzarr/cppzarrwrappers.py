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


def write_zarr(file_name, data, cname='zstd', clevel=1, order='F', chunks=None, dimension_separator='.'):
    if chunks is None:
        chunks = [256, 256, 256]
    pybind11_write_zarr(file_name, data, cname, clevel, order, chunks, dimension_separator)
    return
