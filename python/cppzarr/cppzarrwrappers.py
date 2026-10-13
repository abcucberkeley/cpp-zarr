import itertools
import json
import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from .cppzarr import pybind11_convert_to_v3, pybind11_default_threads, pybind11_read_zarr, pybind11_write_zarr


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


def _threads(num_threads):
    # A call's number of threads: None (0 to the module) for the default, OpenMP's
    # (OMP_NUM_THREADS, or every CPU the process may run on)
    if num_threads is None:
        return 0
    if isinstance(num_threads, bool) or int(num_threads) != num_threads or num_threads < 1:
        raise Exception(f'num_threads must be a whole number of at least 1, not {num_threads!r}')
    return int(num_threads)


def read_zarr(file_name, start_coords=None, end_coords=None, order=None, num_threads=None, out=None):
    # order is the memory layout of the returned array: 'F' (first axis
    # contiguous), 'C' (last axis contiguous), or None (the default) for the
    # file's own storage order, like zarr-python (or out's).
    # num_threads: the threads this call uses (by default OMP_NUM_THREADS, or every
    # CPU the process may run on). The GIL is released while reading, so reads from
    # several Python threads run at once: give each a share of the CPUs.
    # out: a NumPy array to read into instead of a new one (of the region's shape and
    # the array's data type, writeable and C- or F-contiguous); it is returned.
    if not (os.path.isfile(os.path.join(file_name, '.zarray')) or os.path.isfile(os.path.join(file_name, 'zarr.json'))):
        raise Exception(f'{file_name} does not exist. No .zarray or zarr.json metadata file was found')
    if order not in (None, 'F', 'C'):
        raise Exception(f"order must be 'F', 'C' or None, not {order!r}")
    # All-zero coordinates mean the whole array, for any number of dimensions
    if start_coords is None:
        start_coords = [0, 0, 0]
    if end_coords is None:
        end_coords = [0, 0, 0]
    im = pybind11_read_zarr(file_name, list(start_coords), list(end_coords), order or '', _threads(num_threads), out)
    return im


def _cpus():
    # The CPUs this process may run on
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count() or 1


# The threads that run batches of reads, one pool per number of threads each read uses:
# a thread keeps its OpenMP team at one size (OpenMP runtimes end and recreate threads
# when a thread's team changes size, which made small reads several times slower)
_pools = {}
_pools_lock = threading.Lock()
_pools_pid = os.getpid()


def _pool(threads_per_read):
    global _pools_pid
    with _pools_lock:
        # (a forked child has the pools but none of their threads: it makes its own)
        if os.getpid() != _pools_pid:
            _pools.clear()
            _pools_pid = os.getpid()
        pool = _pools.get(threads_per_read)
        if pool is None:
            pool = _pools[threads_per_read] = ThreadPoolExecutor(max_workers=max(_cpus(), 1),
                                                                 thread_name_prefix=f'cppzarr{threads_per_read}')
        return pool


# Elements a read's thread works on: reads of up to this many use one thread each, and
# larger ones a thread per this many (an 8 MiB uint16 region with 64 chunks spent a third
# of its time on a call's fixed costs at 12 threads; a 128 MiB one scaled to 16 threads)
_ELEMENTS_PER_THREAD = 8 << 20


def _region_elements(file_name, start_coords, end_coords):
    # The number of elements a request reads (for the whole array, from its metadata)
    if start_coords is not None and end_coords is not None and any(end_coords):
        return int(np.prod([int(e) - int(s) for s, e in zip(start_coords, end_coords)], dtype=np.float64))
    for f in ('.zarray', 'zarr.json'):
        path = os.path.join(file_name, f)
        if os.path.isfile(path):
            with open(path) as h:
                shape = json.load(h)['shape']
            stop = [int(e) for e in end_coords] if end_coords is not None and any(end_coords) else shape
            begin = [int(s) for s in start_coords] if start_coords is not None else [0] * len(shape)
            return int(np.prod([e - s for s, e in zip(begin, stop)], dtype=np.float64))
    return 0


def read_zarr_batch(requests, order=None, num_threads=None):
    # Read many regions at once, as reading them one after another with read_zarr does,
    # but with the reads overlapping (the GIL is released while reading): the way to
    # read a batch of regions, from one array or many, when each is small or the
    # storage is slow. requests: a list of (file_name, start_coords, end_coords) or
    # (file_name, start_coords, end_coords, out); start_coords and end_coords may be
    # None (the whole array) and out None (a new array). Returns the arrays, in the
    # order of the requests.
    # num_threads: the threads the batch uses (by default, as for read_zarr,
    # OMP_NUM_THREADS, or every CPU this process may run on). Each read gets a thread per
    # 8M elements it reads (at least one, at most all of them), and as many reads as the
    # threads allow run at once.
    requests = [tuple(r) + (None,) * (4 - len(r)) for r in requests]
    if any(len(r) != 4 for r in requests):
        raise Exception('Each request is (file_name, start_coords, end_coords) or (file_name, start_coords, end_coords, out)')
    if not requests:
        return []
    total = _threads(num_threads) or pybind11_default_threads()
    sizes = sorted(_region_elements(r[0], r[1], r[2]) for r in requests)
    per_read = min(total, max(1, round(sizes[len(sizes) // 2] / _ELEMENTS_PER_THREAD)))
    at_once = max(1, min(len(requests), round(total / per_read)))
    # (the threads split as evenly as they go: some reads may get one more)
    threads = [total // at_once + (i < total % at_once) for i in range(at_once)]
    results = [None] * len(requests)
    following = itertools.count()
    lock = threading.Lock()
    failed = []

    def reader(each):
        # (each task takes the next request until none are left, or a read has failed)
        while not failed:
            with lock:
                i = next(following)
            if i >= len(requests):
                return
            file_name, start, end, out = requests[i]
            try:
                results[i] = read_zarr(file_name, start, end, order=order, num_threads=each, out=out)
            except BaseException:
                failed.append(True)
                raise

    futures = [_pool(each).submit(reader, each) for each in threads]
    # (a failed read's error is raised once the reads under way have ended)
    wait(futures)
    for f in futures:
        f.result()
    return results


def _metadata_dtype(meta):
    # The element type of an array's metadata (.zarray dtype, or zarr.json
    # data_type), in native byte order; None if NumPy cannot describe it as a
    # number type. bfloat16 is ml_dtypes' type.
    name = meta['dtype'] if 'dtype' in meta else meta['data_type']
    if isinstance(name, str) and name.lstrip('<>|') == 'bfloat16':
        try:
            import ml_dtypes
        except ImportError:
            raise Exception('Writing into a bfloat16 array needs the ml_dtypes package (pip install ml_dtypes)')
        return np.dtype(ml_dtypes.bfloat16)
    try:
        dt = np.dtype(name)
    except Exception:
        return None
    return dt.newbyteorder('=') if dt.kind in 'buifc' else None


def _existing_format(file_name):
    # The Zarr format of the array at file_name (2 or 3), or None if there is none
    if os.path.isfile(os.path.join(file_name, '.zarray')):
        return 2
    if os.path.isfile(os.path.join(file_name, 'zarr.json')):
        return 3
    return None


def write_zarr(file_name, data, start_coords=None, end_coords=None, cname='zstd', clevel=1, order=None, chunks=None,
               dimension_separator=None, zarr_format=None, shards=None, num_threads=None):
    # zarr_format: 2 or 3. By default the format of the array already at file_name,
    # or 2 for a new one. Zarr v3 arrays default to C order and chunk files named
    # c/0/0/0 (order='C', dimension_separator='/'), v2 ones to F order and chunk
    # files in nested folders, 0/0/0 (dimension_separator='/'; '.' names them 0.0.0,
    # all in one folder, which writes much slower on network file systems).
    # shards (Zarr v3 only) is the shard shape: chunks then sets the inner chunks.
    # num_threads: the threads this call uses, as for read_zarr (the GIL is released
    # while writing).
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
        dimension_separator = '/'

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
                        zarr_format, shards or [], _threads(num_threads))
    return


def convert_to_v3(file_name):
    # Turn the Zarr v2 array at file_name into a Zarr v3 array in place: a zarr.json
    # replaces its .zarray (and takes in its .zattrs attributes), and its chunk files
    # are kept as they are, so no data is rewritten
    if _existing_format(file_name) != 2:
        raise Exception(f'{file_name} is not a Zarr v2 array (no .zarray metadata file was found)')
    pybind11_convert_to_v3(file_name)
