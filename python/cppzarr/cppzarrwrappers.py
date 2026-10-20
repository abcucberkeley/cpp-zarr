import itertools
import json
import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from .cppzarr import (pybind11_convert_to_v3, pybind11_default_threads, pybind11_gpu_available, pybind11_gpu_set_nvcomp,
                      pybind11_read_zarr, pybind11_read_zarr_gpu, pybind11_region_info, pybind11_write_zarr)


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


def read_zarr(file_name, start_coords=None, end_coords=None, order=None, num_threads=None, out=None, stream=None,
              gpu=False):
    # order is the memory layout of the returned array: 'F' (first axis
    # contiguous), 'C' (last axis contiguous), or None (the default) for the
    # file's own storage order, like zarr-python (or out's).
    # num_threads: the threads this call uses (by default OMP_NUM_THREADS, or every
    # CPU the process may run on). The GIL is released while reading, so reads from
    # several Python threads run at once: give each a share of the CPUs.
    # out: a NumPy array to read into instead of a new one (of the region's shape and
    # the array's data type, writeable and C- or F-contiguous); it is returned. Or GPU
    # memory: a CuPy array, a PyTorch tensor on a CUDA device, or another array with
    # __cuda_array_interface__ (C- or F-contiguous): the chunks are then decompressed on
    # the GPU (see read_zarr_batch), with the GPU work on stream (a PyTorch or CuPy stream,
    # or a CUDA stream handle; by default the default stream), done when this returns.
    # gpu=True reads into a new CuPy array on the current CUDA device instead of a NumPy array
    # (needs CuPy; CuPy has no bfloat16: read bfloat16 arrays into a PyTorch tensor with out=).
    if order not in (None, 'F', 'C'):
        raise Exception(f"order must be 'F', 'C' or None, not {order!r}")
    if gpu and out is None:
        out = _new_gpu_arrays([(file_name, start_coords, end_coords)], order)[0]
    if out is not None and _cuda_array(out) is not None:
        # (a missing array is reported when its metadata is read)
        _read_gpu([(file_name, start_coords, end_coords, out)], order, num_threads, stream)
        return out
    if gpu:
        raise Exception('gpu=True reads into GPU memory: out must be GPU memory too (a CuPy array, or a PyTorch tensor on a '
                        'CUDA device), or None for a new CuPy array')
    if not (os.path.isfile(os.path.join(file_name, '.zarray')) or os.path.isfile(os.path.join(file_name, 'zarr.json'))):
        raise Exception(f'{file_name} does not exist. No .zarray or zarr.json metadata file was found')
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


# ---- GPU reads
# The data types of PyTorch tensors, as cpp-zarr names them (its unsigned types beyond uint8 are
# newer than some versions)
_TORCH_TYPES = {'uint8': '|u1', 'int8': '|i1', 'int16': '<i2', 'int32': '<i4', 'int64': '<i8', 'uint16': '<u2',
                'uint32': '<u4', 'uint64': '<u8', 'float16': '<f2', 'bfloat16': 'bfloat16', 'float32': '<f4',
                'float64': '<f8', 'complex64': '<c8', 'complex128': '<c16', 'bool': '|b1'}


def _contiguous(shape, strides):
    # (C-contiguous, F-contiguous) for element strides
    def ok(axes):
        expect = 1
        for d in axes:
            if shape[d] != 1 and strides[d] != expect:
                return False
            expect *= shape[d]
        return True
    n = len(shape)
    return ok(range(n - 1, -1, -1)), ok(range(n))


def _cuda_array(a):
    # (device pointer, data type, shape, C-contiguous, F-contiguous) of GPU memory a (a PyTorch
    # tensor on a CUDA device, or anything with __cuda_array_interface__, such as a CuPy
    # array), or None when a is not GPU memory
    if type(a).__module__.split('.')[0] == 'torch' and getattr(a, 'is_cuda', False):
        name = str(a.dtype).replace('torch.', '')
        if name not in _TORCH_TYPES:
            raise Exception(f'out has the data type {a.dtype}, which cpp-zarr does not read')
        shape = tuple(int(v) for v in a.shape)
        c, f = _contiguous(shape, tuple(a.stride()))
        return int(a.data_ptr()), _TORCH_TYPES[name], shape, c, f
    cai = getattr(a, '__cuda_array_interface__', None)
    if cai is None:
        return None
    shape = tuple(int(v) for v in cai['shape'])
    typestr = cai['typestr']
    itemsize = np.dtype(typestr).itemsize if typestr[1] != 'V' else int(typestr[2:])
    strides = cai.get('strides')
    if strides is None:
        c, f = True, len([v for v in shape if v != 1]) <= 1
    else:
        if any(st % itemsize for st in strides):
            raise Exception('out must be C- or F-contiguous')
        c, f = _contiguous(shape, tuple(st // itemsize for st in strides))
    if typestr[1] in 'iufcb':
        typestr = np.dtype(typestr).str if np.dtype(typestr).itemsize > 1 else '|' + typestr[1:]
    if cai.get('readonly', False) or (len(cai['data']) > 1 and cai['data'][1]):
        raise Exception('out must be writeable')
    return int(cai['data'][0]), typestr, shape, c, f


def _stream_handle(stream):
    # A CUDA stream's handle: of a PyTorch stream (cuda_stream), a CuPy stream (ptr), or an int
    if stream is None:
        return 0
    for attr in ('cuda_stream', 'ptr'):
        if hasattr(stream, attr):
            return int(getattr(stream, attr))
    if isinstance(stream, int):
        return stream
    raise Exception(f'stream must be a PyTorch or CuPy stream, or a CUDA stream handle, not {stream!r}')


_nvcomp_looked = False


def _find_nvcomp():
    # nvCOMP from its pip package (nvidia-libnvcomp-cu12, or the older nvidia-nvcomp-cu12/cu13
    # layout), for the library to load; CPPZARR_NVCOMP_LIBRARY, when set, comes first
    global _nvcomp_looked
    if _nvcomp_looked:
        return
    _nvcomp_looked = True
    if os.environ.get('CPPZARR_NVCOMP_LIBRARY'):
        return
    import glob, importlib
    for mod, subs in (('nvidia.libnvcomp', ('lib64', 'lib', 'bin')), ('nvidia.nvcomp', ('', 'lib64', 'lib', 'bin'))):
        try:
            m = importlib.import_module(mod)
        except ImportError:
            continue
        for base in getattr(m, '__path__', []):
            for sub in subs:
                found = sorted(f for pat in ('libnvcomp.so*', 'nvcomp*.dll') for f in glob.glob(os.path.join(base, sub, pat))
                               if 'cpu' not in os.path.basename(f))
                if found:
                    pybind11_gpu_set_nvcomp(found[-1])
                    return


def gpu_available():
    """(True, '') when reads into GPU memory can run here; (False, why) when not (no NVIDIA
    driver, no nvCOMP: pip install nvidia-libnvcomp-cu12)"""
    _find_nvcomp()
    return pybind11_gpu_available()


def _new_gpu_arrays(requests, order):
    # New CuPy arrays on the current CUDA device for (file_name, start_coords, end_coords)
    # requests: of each region's shape and its array's data type, in order (by default the
    # array's storage order), as read_zarr makes NumPy arrays
    try:
        import cupy
    except ImportError:
        raise Exception('gpu=True makes CuPy arrays, and CuPy is not installed (pip install cupy-cuda12x); or pass GPU '
                        'memory to read into as out= (a PyTorch tensor on a CUDA device, for one)')
    coords = lambda c: [int(v) for v in c] if c is not None else [0, 0, 0]
    info = pybind11_region_info([(f, coords(s), coords(e)) for f, s, e in requests])
    outs = []
    for (file_name, _, _), (shape, dtype, storage) in zip(requests, info):
        if dtype == 'bfloat16':
            raise Exception(f'{file_name} is a bfloat16 array, and CuPy has no bfloat16 type: read it into a PyTorch '
                            'bfloat16 tensor on a CUDA device with out= instead')
        outs.append(cupy.empty(tuple(shape), dtype=dtype, order=order or storage))
    return outs


def _read_gpu(requests, order, num_threads, stream):
    # requests: (file_name, start_coords, end_coords, out) with out GPU memory: one GPU batch for
    # each order the outputs have (C or F)
    _find_nvcomp()
    groups = {}
    for file_name, start, end, out in requests:
        ptr, typestr, shape, c, f = _cuda_array(out)
        if order == 'C' and not c or order == 'F' and not f:
            raise Exception(f"out must be {order}-contiguous for order '{order}'")
        if order is None and not (c or f):
            raise Exception('out must be C- or F-contiguous')
        o = order or ('C' if c else 'F')
        groups.setdefault(o, []).append((file_name, [int(v) for v in (start if start is not None else [0, 0, 0])],
                                         [int(v) for v in (end if end is not None else [0, 0, 0])], ptr, typestr, list(shape)))
    for o, reqs in groups.items():
        pybind11_read_zarr_gpu(reqs, o, _threads(num_threads), _stream_handle(stream))


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


def read_zarr_batch(requests, order=None, num_threads=None, stream=None, gpu=False):
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
    # Into GPU memory (every out a CuPy array, a PyTorch tensor on a CUDA device, or an array
    # with __cuda_array_interface__, all on one GPU): the chunks' stored bytes are read on the
    # host (the files with at least 32 threads at once, since reading mostly waits on the
    # storage; num_threads threads prepare them) and decompressed on the GPU (nvCOMP: pip install
    # nvidia-libnvcomp-cu12), all of the batch's at once, with the GPU work on stream (a
    # PyTorch or CuPy stream, or a CUDA stream handle; by default the default stream), done
    # when this returns. Bigger batches use the GPU better. gpu=True reads the requests without
    # an out into new CuPy arrays on the current CUDA device (see read_zarr).
    requests = [tuple(r) + (None,) * (4 - len(r)) for r in requests]
    if any(len(r) != 4 for r in requests):
        raise Exception('Each request is (file_name, start_coords, end_coords) or (file_name, start_coords, end_coords, out)')
    if not requests:
        return []
    if gpu:
        new = [i for i, r in enumerate(requests) if r[3] is None]
        for i, a in zip(new, _new_gpu_arrays([requests[i][:3] for i in new], order) if new else []):
            requests[i] = requests[i][:3] + (a,)
    on_gpu = [r[3] is not None and _cuda_array(r[3]) is not None for r in requests]
    if gpu or any(on_gpu):
        if not all(on_gpu):
            raise Exception('A batch reads into GPU memory or into host memory, not both: give every request a GPU out '
                            '(or none, with gpu=True)')
        # (a missing array is reported when its metadata is read, the batch's arrays' at once)
        _read_gpu(requests, order, num_threads, stream)
        return [r[3] for r in requests]
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
