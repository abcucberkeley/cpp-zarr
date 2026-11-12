import json
import math
import numpy as np
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from .cppzarr import (pybind11_convert_to_v3, pybind11_default_threads, pybind11_gpu_available, pybind11_gpu_set_cufile, pybind11_gpu_set_nvcomp,
                      pybind11_last_read_times, pybind11_on_network, pybind11_open_zarrs, pybind11_read_times_available,
                      pybind11_read_zarr, pybind11_read_zarr_gpu, pybind11_read_zarr_opened, pybind11_region_info,
                      pybind11_release_buffers, pybind11_write_zarr, pybind11_write_zarr_gpu, pybind11_create_zarr,
                      pybind11_convert_values)


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


def _cuda_data(a):
    # (device pointer, data type as cpp-zarr names it, shape, element strides) of GPU data to write (a
    # PyTorch tensor on a CUDA device, or anything with __cuda_array_interface__, such as a CuPy
    # array), or None when a is not GPU memory
    if type(a).__module__.split('.')[0] == 'torch' and getattr(a, 'is_cuda', False):
        name = str(a.dtype).replace('torch.', '')
        if name not in _TORCH_TYPES:
            raise Exception(f'The data has the data type {a.dtype}, which cpp-zarr does not write')
        return int(a.data_ptr()), _TORCH_TYPES[name], tuple(int(v) for v in a.shape), tuple(int(v) for v in a.stride())
    cai = getattr(a, '__cuda_array_interface__', None)
    if cai is None:
        return None
    shape = tuple(int(v) for v in cai['shape'])
    typestr = cai['typestr']
    if typestr[1] not in 'iufcb':
        raise Exception(f'The data has the data type {typestr}, which cpp-zarr does not write from GPU memory')
    dt = np.dtype(typestr)
    if not dt.isnative:
        raise Exception('The GPU data is not in this machine\'s byte order')
    strides = cai.get('strides')
    if strides is None:
        strides, acc = [0] * len(shape), 1
        for d in range(len(shape) - 1, -1, -1):
            strides[d] = acc
            acc *= shape[d]
    else:
        if any(st % dt.itemsize for st in strides):
            raise Exception('Unsupported memory layout: strides must be multiples of the element size')
        strides = [st // dt.itemsize for st in strides]
    return int(cai['data'][0]), dt.str if dt.itemsize > 1 else '|' + dt.str[1:], shape, tuple(strides)


def _gpu_dtype_name(a):
    # The NumPy name of GPU data's data type (bfloat16 for PyTorch's)
    if type(a).__module__.split('.')[0] == 'torch':
        return str(a.dtype).replace('torch.', '')
    return np.dtype(a.dtype).name


def _gpu_ready(a):
    # GPU data the writer takes: in this machine's byte order, with positive strides (others are
    # copied on the GPU first: reversed or repeated axes)
    if type(a).__module__.split('.')[0] != 'torch' and hasattr(a, 'dtype') and not np.dtype(a.dtype).isnative:
        a = a.astype(np.dtype(a.dtype).newbyteorder('='))
    _, _, shape, strides = _cuda_data(a)
    if any(n > 1 and s <= 0 for s, n in zip(strides, shape)):
        if type(a).__module__.split('.')[0] == 'torch':
            a = a.contiguous()
        elif type(a).__module__.split('.')[0] == 'cupy':
            import cupy
            a = cupy.ascontiguousarray(a)
        else:
            raise Exception('Unsupported memory layout: GPU data with reversed or repeated axes (strides of 0 or less)')
    return a


def _current_stream(a):
    # The current stream of GPU data's library (PyTorch's on the tensor's device, or CuPy's), which
    # the work that made the data most likely went on; None (the default stream) for other data
    module = type(a).__module__.split('.')[0]
    if module == 'torch':
        import torch
        return torch.cuda.current_stream(a.device)
    if module == 'cupy':
        import cupy
        return cupy.cuda.get_current_stream()
    return None


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
    # nvCOMP from its pip package, for the library to load, found from pip's record of the files
    # it installed: importing the package failed where another package named nvidia hid it (as in
    # NVIDIA's PyTorch containers), and the older one loads its own nvCOMP when imported. The
    # CUDA 12 build first, which the gpu extra installs and which runs on every GPU from Volta on
    # (CUDA 13 builds need a Turing or newer one), then the CUDA 13 one and the older
    # nvidia-nvcomp layout. CPPZARR_NVCOMP_LIBRARY, when set, comes first.
    global _nvcomp_looked
    if _nvcomp_looked:
        return
    _nvcomp_looked = True
    _find_cufile()
    if os.environ.get('CPPZARR_NVCOMP_LIBRARY'):
        return
    from importlib import metadata
    for name in ('nvidia-libnvcomp-cu12', 'nvidia-libnvcomp-cu13', 'nvidia-nvcomp-cu12', 'nvidia-nvcomp-cu13'):
        dist = None
        for spelling in (name, name.replace('-', '_')):
            try:
                dist = metadata.distribution(spelling)
                break
            except metadata.PackageNotFoundError:
                pass
        if dist is None:
            continue
        libs = sorted(str(dist.locate_file(f)) for f in (dist.files or [])
                      if (f.name.startswith('libnvcomp.so') or (f.name.startswith('nvcomp') and f.name.endswith('.dll')))
                      and 'cpu' not in f.name)
        libs = [p for p in libs if os.path.exists(p)]
        if libs:
            pybind11_gpu_set_nvcomp(libs[-1])
            return


def _find_cufile():
    # cuFile (GPUDirect Storage) from its pip package, found as nvCOMP is, for reads where GDS's
    # nvidia-fs driver is loaded: the CUDA 12 build first. CPPZARR_CUFILE_LIBRARY, when set, comes
    # first.
    if os.environ.get('CPPZARR_CUFILE_LIBRARY'):
        return
    from importlib import metadata
    for name in ('nvidia-cufile-cu12', 'nvidia-cufile-cu13', 'nvidia-cufile'):
        dist = None
        for spelling in (name, name.replace('-', '_')):
            try:
                dist = metadata.distribution(spelling)
                break
            except metadata.PackageNotFoundError:
                pass
        if dist is None:
            continue
        libs = sorted(str(dist.locate_file(f)) for f in (dist.files or []) if f.name.startswith('libcufile.so'))
        libs = [p for p in libs if os.path.exists(p)]
        if libs:
            pybind11_gpu_set_cufile(libs[0])
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


def _pool(threads_per_read, workers):
    # A pool for reads of threads_per_read threads each, with at least workers threads (and as many
    # as the process has CPUs)
    global _pools_pid
    with _pools_lock:
        # (a forked child has the pools but none of their threads: it makes its own)
        if os.getpid() != _pools_pid:
            _pools.clear()
            _pools_pid = os.getpid()
        pool = _pools.get(threads_per_read)
        if pool is None or pool._max_workers < workers:
            # (a bigger one replaces it: the old one's threads end once no call holds it, so a call
            # still handing reads to it is not cut short)
            pool = _pools[threads_per_read] = ThreadPoolExecutor(max_workers=max(_cpus(), workers, 1),
                                                                 thread_name_prefix=f'cppzarr{threads_per_read}')
        return pool


_openers = [None, None]

# A batch's openers at once (see read_zarr_batch), each opening one array at a time
_OPENERS = 4

# An array's metadata that one of a batch's reads or openers is reading (in the batch's opened)
_OPENING = object()


class _OpenFailed:
    # The error an array's metadata could not be read with, for each of the batch's reads of it to raise
    __slots__ = ('error',)

    def __init__(self, error):
        self.error = error


def _opener():
    # The pool that opens a batch's arrays ahead of their reads, one for each process (a forked child
    # makes its own): _OPENERS threads, which mostly wait for the storage (with 16, batches of single time
    # points of many arrays read from the page cache were up to 1.12 times as slow, its threads taking
    # turns on the CPUs with the reads)
    with _pools_lock:
        if _openers[0] is None or _openers[1] != os.getpid():
            _openers[0], _openers[1] = ThreadPoolExecutor(max_workers=_OPENERS, thread_name_prefix='cppzarr_open'), os.getpid()
        return _openers[0]


# The wait for the storage of the process's last batch of reads of one thread each from network storage,
# as a share of their CPU time (None before the first): a batch starts with as many readers as that calls
# for, not with one a thread, whose first round of reads would wait (a batch of 64 at 16 threads, all of
# it); a batch that waits otherwise than the one before adds or ends readers as it goes
_wait_share = [None]


# Elements a read's thread works on in a batch: reads of up to this many use one thread each,
# and larger ones a thread per this many, so that a batch of many reads runs more of them at
# once, each with fewer threads (a read's own fixed costs, its files' opening and the wait for
# its first bytes, then overlap with the other reads'): cold batches of 32 and 128 Cell
# Observatory hypercubes (64M uint16 elements) read from VAST on 16 cores 2.0-2.2 times as fast
# with 2 threads a read (8 reads at once) as with 8 (2 at once), as fast as with 64 threads, in
# less CPU time
_ELEMENTS_PER_THREAD = 32 << 20

# Reads of more than one thread each run at most half of a batch at once (but two), so that the
# batch takes at least two rounds of them: in one, it waits for its slowest read (8 hypercubes from
# the page cache took 1.13 times as long with all 8 at once). Smaller reads, whose time goes more to
# waiting for the storage, all run at once as the threads allow.
_LEAST_ROUNDS = 2


def _reads_per_thread():
    # CPPZARR_BATCH_READS_PER_THREAD (3 when unset or not a number; at least 1)
    v = os.environ.get('CPPZARR_BATCH_READS_PER_THREAD')
    try:
        return max(1, int(v)) if v else 3
    except ValueError:
        return 3


# Reads of one thread each (of up to _ELEMENTS_PER_THREAD elements, such as single time points) from
# network storage run up to this many at once for each of a batch's threads, as many as their waiting
# for the storage calls for: such a read waits for the server before its chunks arrive (from VAST about
# 1 ms for a shard's index, then for its first bytes), and meanwhile its thread has nothing to decode;
# from the page cache it hardly waits, and more reads than threads would only take turns on the CPUs.
# A batch starts with as many as the wait of the batch before called for (the first: one a thread) and,
# once as many reads as it has threads have ended, keeps threads x (CPU + 2 x wait) / CPU of them going
# (the reads' own time neither on a CPU nor waiting for one, as the library measures it: Linux), at most
# half the batch at once: cold batches of 512 single time points of the training data read from VAST
# with 16 threads of a B200 node, 11 of an array as a training loader's batches are or each of another,
# 1.3-1.5 times as fast as one at once a thread (each array's metadata read once in both). Only in a
# batch that has had data fetched from the storage, and where more at once have read faster (see
# _trials). Elsewhere, and in batches of bigger reads, one at once a thread.
_READS_PER_THREAD = _reads_per_thread()


# Reads of one thread each from network storage wait for the storage at least this share of their CPU time
# before a batch runs more of them at once than it has threads (see _readers_wanted)
_LEAST_WAIT_SHARE = 0.5


def _process_read_bytes():
    # The bytes this process has had fetched from its storage so far (Linux: /proc/self/io; the NFS client
    # counts what it fetches from the server, and nothing for reads the page cache serves), or None where
    # that cannot be read
    try:
        fd = os.open('/proc/self/io', os.O_RDONLY)
        try:
            text = os.read(fd, 1024)
        finally:
            os.close(fd)
        for line in text.split(b'\n'):
            if line.startswith(b'read_bytes:'):
                return int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


# Whether more readers than threads help the process's batches, for each batch size (number of requests):
# a batch that runs them is compared, in bytes fetched from the storage a second, with the last batch of
# that size that did not (not the process's first of it, which also starts the pools' threads), and they
# are kept while it reads at least _EXTRA_GAIN times as fast; else they are left off for _EXTRA_OFF
# batches of that size and tried again, and kept, tried against a batch without them every _EXTRA_ON
# (cold batches only; where the bytes fetched cannot be read, they are not tried, only called for). Cold
# batches of 64 single time points of one array read with every core of a 24-core node, 32 at once (24
# threads), took 1.08-1.12 times as long as 24 at once, where on a B200 node's 16 cores 32-48 at once
# read batches of single time points of the training data 1.2-1.5 times as fast as 16.
_EXTRA_GAIN, _EXTRA_OFF, _EXTRA_ON = 1.1, 32, 64
_trials = {}
_trials_lock = threading.Lock()


def _extras_allowed(size):
    # Whether a batch of size requests may run more readers than threads (see _trials)
    with _trials_lock:
        t = _trials.get(size)
        if t is None:
            if len(_trials) >= 64:
                _trials.clear()
            t = _trials[size] = ['first', 0.0, 0]
        return t[0] in ('try', 'on')


def _extras_learn(size, used, rate):
    # A cold batch of size requests has ended, having run more readers than threads (used) or not, at
    # rate bytes fetched a second
    with _trials_lock:
        t = _trials.get(size)
        if t is None:
            return
        state = t[0]
        if state == 'first':
            t[0] = 'measure'
        elif state == 'measure':
            t[0], t[1] = 'try', rate
        elif state == 'try':
            if not used:
                # (its reads did not call for more: a later baseline)
                t[1] = rate
            elif rate >= _EXTRA_GAIN * t[1]:
                t[0], t[2] = 'on', _EXTRA_ON
            else:
                t[0], t[2] = 'off', _EXTRA_OFF
        else:
            if state == 'off':
                t[1] = rate
            t[2] -= 1
            if t[2] <= 0:
                t[0] = 'try' if state == 'off' else 'measure'


def _readers_wanted(threads, most, cpu, wait):
    # How many reads of one thread each to keep going, from the CPU time and the wait for the storage of
    # those done: one a thread while they wait less than half their CPU time (from the page cache, where
    # the NFS client still asks the server about each file it opens: more readers then only took turns
    # on the CPUs, a warm batch of clustered single time points 1.12 times as slow with 20 readers on 16
    # threads), else threads x (CPU + 2 x wait) / CPU, at most most (twice the wait: reads that wait the
    # same on average do not all wait at once, and as many as would just cover the average read cold
    # time points from VAST 1.15 times as slow as at 3 a thread)
    if wait < _LEAST_WAIT_SHARE * cpu:
        return threads
    return max(threads, min(most, math.ceil(threads * (cpu + 2 * wait) / max(cpu, 1e-9))))


def _read_ahead_threads():
    # CPPZARR_READ_AHEAD_THREADS, as the library reads it (64 when unset)
    v = os.environ.get('CPPZARR_READ_AHEAD_THREADS')
    if not v:
        return 64
    try:
        return max(0, int(v))
    except ValueError:
        return 0


# Batches of more threads than this split as 1.7.3's did: a thread per 8M elements, as many reads at
# once as the threads allow. Their reads do not read ahead (the library's reads do not while the
# process's reads have more threads than CPPZARR_READ_AHEAD_THREADS), and without it hypercubes read
# 2 threads at a time with a B200 node's 128 cores took 1.4 times as long as 8 at a time.
_READ_AHEAD_THREADS = _read_ahead_threads()
_ELEMENTS_PER_THREAD_MANY = 8 << 20


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
    # 32M elements it reads (at least one, at most all of them; with more threads than
    # CPPZARR_READ_AHEAD_THREADS, 64, a thread per 8M), and as many reads as the threads
    # allow run at once (with fewer reads than that, the threads are shared out among them;
    # reads of one thread each from network storage, up to CPPZARR_BATCH_READS_PER_THREAD, 3,
    # at once a thread, as their wait for the storage calls for). Each array's metadata is read
    # once for the batch, however many of its regions the batch reads.
    # Into GPU memory (every out a CuPy array, a PyTorch tensor on a CUDA device, or an array
    # with __cuda_array_interface__, all on one GPU): the chunks' stored bytes are read on the
    # host (the files with at least 128 threads at once, since reading mostly waits on the
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
    many = total > _READ_AHEAD_THREADS
    per_read = min(total, max(1, round(sizes[len(sizes) // 2] / (_ELEMENTS_PER_THREAD_MANY if many else _ELEMENTS_PER_THREAD))))
    at_once = max(1, min(len(requests), round(total / per_read)))
    if per_read > 1 and _LEAST_ROUNDS > 1 and not many:
        at_once = min(at_once, max(2, -(-len(requests) // _LEAST_ROUNDS)))
    # (reads of more than one thread each share the threads evenly, or run one at a time: 4
    # hypercubes read 3 and 2 threads a read, two at a time, took 1.2 times as long with 5 threads as
    # one at a time, from the page cache, VAST and two other NFS servers; the read with fewer threads
    # ends the batch)
    while per_read > 1 and at_once > 1 and total % at_once and not many:
        at_once -= 1
    # (the threads split as evenly as they go: some reads may get one more)
    threads = [total // at_once + (i < total % at_once) for i in range(at_once)]
    if order not in (None, 'F', 'C'):
        raise Exception(f"order must be 'F', 'C' or None, not {order!r}")
    # (reads of one thread each from network storage: up to _READS_PER_THREAD at once a thread, as
    # their waiting for the storage calls for)
    most = len(threads)
    if (per_read == 1 and not many and _READS_PER_THREAD > 1 and len(requests) > total
            and pybind11_read_times_available() and pybind11_on_network(requests[0][0])):
        # (and at most half the batch at once, but the threads: all of it at once, the batch waits for
        # its slowest read, which no other read overlaps any more, where more reads at once only shared
        # out the same bandwidth: 64 single time points read cold with 24 threads, all 64 at once, were
        # up to 1.18 times as slow)
        most = min(len(requests), total * _READS_PER_THREAD, max(total, -(-len(requests) // _LEAST_ROUNDS)))
    results = [None] * len(requests)
    taken = [0]
    lock = threading.Lock()
    failed = []
    # (each array's metadata read once for the batch, by whichever comes to it first: its first read, or
    # one of the batch's openers, which open the arrays of the next requests in their order, up to
    # `window` requests past the reads; the batch's other reads of it wait for that, and it is dropped
    # after its last read. All opened before the reads, a batch of many arrays waited for the slowest;
    # opened only by the reads, a batch's reads of one array, which come together, all waited for it at
    # once. An array that is not there fails the reads that need it. Kept in one dict and waited for on
    # one condition, so that a batch makes few objects: with a Future an array and a task a request,
    # each batch of 64 single time points of 64 arrays made Python's garbage collector run twice, and
    # in a process holding many objects a collection of its older generation took twice as long as a
    # batch)
    opened = {}
    left = {}
    for r in requests:
        left[r[0]] = left.get(r[0], 0) + 1
    asked = [0]
    openers = [0]
    window = 4 * most
    opener_tasks = []
    arrived = threading.Condition(lock)

    def open_one(path):
        try:
            v = pybind11_open_zarrs([path])[0]
        except BaseException as e:
            v = _OpenFailed(e)
        with lock:
            opened[path] = v
            arrived.notify_all()
        return v

    def opener():
        # (opens the arrays of the next requests that no read or opener has come to, until `window`
        # requests past the reads)
        while True:
            with lock:
                end = min(taken[0] + window, len(requests))
                while asked[0] < end and (requests[asked[0]][0] in opened or not left[requests[asked[0]][0]]):
                    asked[0] += 1
                if failed or asked[0] >= end:
                    openers[0] -= 1
                    return
                path = requests[asked[0]][0]
                asked[0] += 1
                opened[path] = _OPENING
            open_one(path)

    def start_openers(n):
        # (without the lock: a pool starting threads waits for each to start)
        for _ in range(n):
            opener_tasks.append(_opener().submit(opener))

    def metadata(path):
        with lock:
            # (the openers end a window past the reads, and start again once the reads are half a
            # window closer; a batch of one array needs none)
            more = 0
            if (openers[0] < _OPENERS and len(left) > 1 and not failed and asked[0] < len(requests)
                    and asked[0] - taken[0] < window // 2):
                more = _OPENERS - openers[0]
                openers[0] = _OPENERS
            v = opened.get(path)
            if v is None:
                opened[path] = _OPENING
        if v is None:
            # (its own array first, the openers after: their threads may still have to start)
            v = open_one(path)
            start_openers(more)
        else:
            start_openers(more)
            if v is _OPENING:
                with lock:
                    v = opened[path]
                    while v is _OPENING:
                        arrived.wait()
                        v = opened[path]
        if isinstance(v, _OpenFailed):
            raise v.error
        return v

    def finished(path):
        with lock:
            left[path] -= 1
            if left[path] == 0:
                opened.pop(path, None)
    # (reads of one thread each from network storage, as many at once as their wait for the storage
    # calls for: from the batches before's to begin with, then the batch's own, once as many reads as
    # it has threads have ended; readers are added, or end after a read, to keep that many going)
    adaptive = most > len(threads)
    done = [0.0, 0.0, 0]
    futures = []
    # (and only once the batch has had data fetched from the storage: reads the page cache serves still
    # wait, the NFS client asking the server about each file they open, and on a B200 node busy with other
    # jobs warm batches of 512 single time points measured waits of half their CPU time, where more readers
    # made them up to 1.27 times as slow; where the bytes fetched cannot be read, as before)
    fetched_before = _process_read_bytes() if adaptive else None
    cold = [fetched_before is None]
    # (and where they have helped batches of this size, see _trials)
    allowed = adaptive and (fetched_before is None or _extras_allowed(len(requests)))
    first = len(threads)
    if allowed and _wait_share[0] is not None:
        first = min(_readers_wanted(total, most, 1.0, _wait_share[0]), len(requests))
    running, peak = [first], [first]

    def fetched():
        # (whether the process has had bytes fetched from its storage since the batch began)
        now = _process_read_bytes()
        return now is None or now > fetched_before

    def reader(each):
        # (each task takes the next request until none are left, a read has failed, or fewer readers are
        # called for)
        while not failed:
            with lock:
                i = taken[0]
                taken[0] += 1
            if i >= len(requests):
                return
            file_name, start, end, out = requests[i]
            try:
                # (as read_zarr reads, from the array's metadata read once for the batch)
                results[i] = pybind11_read_zarr_opened(metadata(file_name), list(start) if start is not None else [0, 0, 0],
                                                       list(end) if end is not None else [0, 0, 0], order or '', each, out, adaptive)
            except BaseException:
                failed.append(True)
                raise
            finished(file_name)
            if adaptive:
                cpu, waited = pybind11_last_read_times()
                # (asked without the lock, until the batch has had data fetched)
                now_cold = cold[0] or (allowed and done[2] + 1 >= total and fetched())
                more = 0
                with lock:
                    if waited >= 0:
                        done[0] += cpu
                        done[1] += waited
                    done[2] += 1
                    cold[0] = cold[0] or now_cold
                    if done[2] >= total and not failed:
                        want = min(_readers_wanted(total, most, done[0], done[1]), running[0] + len(requests) - taken[0])
                        if not cold[0] or not allowed:
                            want = min(want, total)
                        if want > running[0]:
                            more = want - running[0]
                            running[0] = want
                            peak[0] = max(peak[0], want)
                        elif want < running[0]:
                            running[0] -= 1
                            return
                if more:
                    # (started without the lock, which the other readers need meanwhile; the batch waits for
                    # this reader, so it sees them)
                    added = [_pool(1, most).submit(reader, 1) for _ in range(more)]
                    with lock:
                        futures.extend(added)

    try:
        # (the readers submitted without the lock: a pool starting threads waits for each to start,
        # and the readers already started need the lock; a process's first batch of 64 single time points
        # with 24 threads read its first region 10 ms late)
        began = time.perf_counter()
        started = [_pool(each, most).submit(reader, each) for each in (threads if first == len(threads) else [1] * first)]
        with lock:
            futures.extend(started)
        # (a failed read's error is raised once the reads under way have ended; readers started on the
        # way are waited for too)
        while True:
            with lock:
                pending = [f for f in futures if not f.done()]
            if not pending:
                break
            wait(pending)
        took = time.perf_counter() - began
        # (openers no read is left to need, after a failed one, are not started)
        with lock:
            for f in opener_tasks:
                f.cancel()
        for f in futures:
            f.result()
    finally:
        # (reader, which starts readers, holds itself: without this the batch's objects would stay until a
        # garbage collection)
        reader = None
    if adaptive:
        now = _process_read_bytes() if fetched_before is not None else None
        if done[0] > 0:
            # (a batch the page cache served starts the next with one reader a thread)
            _wait_share[0] = done[1] / done[0] if now is None or now > fetched_before else 0.0
        if now is not None and now > fetched_before:
            _extras_learn(len(requests), peak[0] > total, (now - fetched_before) / max(took, 1e-9))
    # (for tests and benchmarks: the last batch's most readers at once, and its measured reads' CPU time
    # and wait)
    global _last_batch
    _last_batch = (peak[0], done[0], done[1])
    return results


_last_batch = (0, 0.0, 0.0)


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


def _typestr(dt):
    # A NumPy data type as cpp-zarr names it ('|b1', '<u2', ..., and ml_dtypes' 'bfloat16')
    dt = np.dtype(dt)
    if dt.name == 'bfloat16':
        return 'bfloat16'
    if dt.kind == 'b':
        return '|b1'
    if dt.kind in 'iufc':
        return '<' + dt.kind + str(dt.itemsize)
    raise Exception(f'Unsupported data type: {dt}')


def _gpu_to_host(a):
    # GPU data copied into host memory as a NumPy array (PyTorch's bfloat16 as float32, which holds its values
    # exactly)
    if type(a).__module__.split('.')[0] == 'torch':
        import torch
        t = a.detach()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.cpu().numpy()
    if hasattr(a, 'get'):
        return a.get()
    import cupy
    return cupy.asarray(a).get()


def _convert_values(data, dtype, num_threads=None):
    # Host data converted to the data type dtype as MATLAB converts between classes, as the MATLAB writer converts
    # the data it writes into an array of another type: into integer types rounded to the nearest integer (halves
    # away from zero) and saturated at the type's limits, NaN to 0 (NumPy's astype truncates and wraps around);
    # into bool any nonzero value true; into floating point types rounded to the nearest value. Complex data goes
    # only into complex arrays. (In cpp-zarr, with the call's threads.)
    data = np.asarray(data)
    if data.dtype.kind == 'c':
        if dtype.kind != 'c':
            raise Exception(f'Complex data can only be written into complex64 or complex128 arrays, not {dtype.name}')
        return data.astype(dtype)
    if data.dtype.name in ('float16', 'bfloat16'):
        data = data.astype(np.float32)   # (exactly)
    elif data.dtype.kind not in 'biuf':
        raise Exception(f'Data of the type {data.dtype} cannot be written into an array of {dtype.name}')
    if not data.dtype.isnative:
        data = data.astype(data.dtype.newbyteorder('='))
    if not (data.flags['C_CONTIGUOUS'] or data.flags['F_CONTIGUOUS']):
        data = np.ascontiguousarray(data)
    out = pybind11_convert_values(data, _typestr(dtype), _threads(num_threads))
    return out.view(dtype) if dtype.name == 'bfloat16' else out


def create_zarr(file_name, shape, dtype, chunks=None, cname='zstd', clevel=1, order=None, dimension_separator=None,
                zarr_format=None, shards=None):
    # Writes only the metadata (.zarray or zarr.json) of a new array of this shape and data type (a NumPy data type
    # or its name; bfloat16 needs ml_dtypes), as write_zarr makes it for data of that shape and type, and no chunks:
    # regions written into it later (write_zarr with start_coords and end_coords) fill it, and what is never written
    # reads as zeros. The options, and their defaults, are write_zarr's. Chunk files already at file_name are left
    # as they are.
    if zarr_format not in (None, 2, 3):
        raise Exception(f'zarr_format must be 2 or 3, not {zarr_format!r}')
    zarr_format = zarr_format or _existing_format(file_name) or 2
    if shards is not None and zarr_format != 3:
        raise Exception('shards needs zarr_format=3')
    if order is None:
        order = 'C' if zarr_format == 3 else 'F'
    if dimension_separator is None:
        dimension_separator = '/'
    shape = [int(n) for n in shape]
    ndim = len(shape)
    if any(n < 0 for n in shape):
        raise Exception(f'Invalid shape {shape}')
    dt = _metadata_dtype({'dtype': dtype}) if isinstance(dtype, str) else np.dtype(dtype)
    if dt is None:
        raise Exception(f'Unsupported data type: {dtype}')
    if chunks is None:
        chunks = _default_chunks(ndim)
    chunks = [int(c) for c in chunks]
    if len(chunks) < ndim:
        raise Exception(f'chunks has {len(chunks)} values but the array has {ndim} dimensions')
    chunks = chunks[:ndim]
    if shards is not None:
        shards = [int(v) for v in shards]
        if len(shards) < ndim:
            raise Exception(f'shards has {len(shards)} values but the array has {ndim} dimensions')
        shards = shards[:ndim]
        if any(c <= 0 or v % c for v, c in zip(shards, chunks)):
            raise Exception(f'shards {shards} must be a multiple of chunks {chunks} along each axis')
    pybind11_create_zarr(file_name, shape, _typestr(dt), cname, clevel, order, chunks, dimension_separator, zarr_format,
                         shards or [])


def write_zarr(file_name, data, start_coords=None, end_coords=None, cname='zstd', clevel=1, order=None, chunks=None,
               dimension_separator=None, zarr_format=None, shards=None, num_threads=None, stream=None, uuid=True, sparse=True):
    # zarr_format: 2 or 3. By default the format of the array already at file_name,
    # or 2 for a new one. Zarr v3 arrays default to C order and chunk files named
    # c/0/0/0 (order='C', dimension_separator='/'), v2 ones to F order and chunk
    # files in nested folders, 0/0/0 (dimension_separator='/'; '.' names them 0.0.0,
    # all in one folder, which writes much slower on network file systems).
    # shards (Zarr v3 only) is the shard shape: chunks then sets the inner chunks.
    # num_threads: the threads this call uses, as for read_zarr (the GIL is released
    # while writing).
    # data in GPU memory (a CuPy array, a PyTorch tensor on a CUDA device, or another array with
    # __cuda_array_interface__): the chunks are put together and compressed on the GPU (nvCOMP:
    # pip install nvidia-libnvcomp-cu12) and their files written here, with the GPU work on stream
    # (a PyTorch or CuPy stream, or a CUDA stream handle; by default the data's library's current
    # stream, where the work that made the data most likely went), done when this returns. The GPU compresses blosc's zstd and lz4 and the zstd codec at nvCOMP's one level
    # of each, whatever clevel says (the chunks decode the same, and are about as small as at clevel
    # 1); arrays of other compressors are written from a copy of the data in host memory.
    # A region written into an existing array of another data type is converted to it as the MATLAB writer
    # converts data (see _convert_values: integers rounded, halves away from zero, and saturated, NaN to 0),
    # in host memory, and written from there (GPU data too).
    # uuid: each chunk file is written under a unique temporary name and then renamed into place, so that a
    # reader never sees a partly written file (False writes them in place: for a new array with one writer per
    # chunk). sparse: chunks whose values are all zero are left out (and an existing file of one deleted), as
    # missing chunks read as zeros; False writes them too.
    gpu = _cuda_data(data) is not None
    if gpu:
        _find_nvcomp()
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
        if gpu:
            if dtype is not None and _gpu_dtype_name(data) != dtype.name:
                # (converted in host memory and written from there, as the MATLAB writer converts GPU data of
                # another class; the data is waited for on its stream first)
                if stream is not None and hasattr(stream, 'synchronize'):
                    stream.synchronize()
                data, gpu = _convert_values(_gpu_to_host(data), dtype, num_threads), False
        elif dtype is not None and np.asarray(data).dtype != dtype:
            data = _convert_values(data, dtype, num_threads)
    # (the writer takes data in this machine's byte order)
    if gpu:
        data = _gpu_ready(data)
    elif not data.dtype.isnative:
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
        raise Exception(f'The region from start_coords to end_coords does not match the data shape {tuple(data.shape)}')
    if gpu:
        ptr, typestr, shape, strides = _cuda_data(data)
        pybind11_write_zarr_gpu(file_name, ptr, typestr, list(shape), list(strides), start_coords, end_coords, cname, clevel, order,
                                chunks, dimension_separator, crop, zarr_format, shards or [], _threads(num_threads),
                                _stream_handle(stream if stream is not None else _current_stream(data)), bool(uuid), bool(sparse))
        return
    # The writer reads the array in place in any memory layout (C or F order, or a
    # strided view), so it is not converted. Only reversed or repeated axes
    # (negative or zero strides) and misaligned data need a copy first.
    if not data.flags['ALIGNED'] or any(n > 1 and (s <= 0 or s % data.itemsize)
                                        for s, n in zip(data.strides, data.shape)):
        data = np.array(data, order='K')
    pybind11_write_zarr(file_name, data, start_coords, end_coords, cname, clevel, order, chunks, dimension_separator, crop,
                        zarr_format, shards or [], _threads(num_threads), bool(uuid), bool(sparse))
    return


def release_buffers():
    # Free the buffers reads and writes keep between calls: those of chunks over 1 MB are kept
    # for later calls up to CPPZARR_KEEP_MB MB in all (by default 1024, or an eighth of the
    # memory the process may use if that is less), and the C library keeps memory freed by the
    # calls' threads for their later use. For a long-lived process that is done reading and
    # writing for a while; the next call makes new buffers.
    pybind11_release_buffers()


def convert_to_v3(file_name):
    # Turn the Zarr v2 array at file_name into a Zarr v3 array in place: a zarr.json
    # replaces its .zarray (and takes in its .zattrs attributes), and its chunk files
    # are kept as they are, so no data is rewritten
    if _existing_format(file_name) != 2:
        raise Exception(f'{file_name} is not a Zarr v2 array (no .zarray metadata file was found)')
    pybind11_convert_to_v3(file_name)
