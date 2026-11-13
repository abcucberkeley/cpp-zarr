# cpp-zarr
An efficient parallel Zarr reader/writer that utilizes c-blosc/c-blosc2 and OpenMP.

## Arrays of any dimension
Arrays with any number of dimensions, including 0D (a single value), can be read and written in F or C order. In Python the only limit is NumPy's (32 dimensions in NumPy 1.x, 64 in NumPy 2.x).
1. Per-axis arguments take one value per axis: `bbox` is `[starts ends]`, and `chunks`, `start_coords`, and `end_coords` have one value for each axis
2. The 3D-style arguments still work on 1D and 2D arrays (e.g. a 6-value `bbox` or 3-value `chunks`)
3. Default chunks are 256 along three axes (the first three in MATLAB, the last three in Python) and 1 along the rest
4. MATLAB reads 1D arrays as column vectors and 0D arrays as scalars

## Python

A Python version of cpp-zarr is available through pip

### Prerequisites

#### Python
1. Python version 3.9-3.14

#### CPU
1. A CPU with AVX/SSE support is required for the reader (Almost all modern CPUs should be compatible)

#### OS
Linux: All Linux distros made within the past 10 years should work

Mac Apple Silicon (M1, M2, etc.): macOS 13 or newer is required

Mac Intel: Support has been discontinued as of v1.5.0 (macOS 12 or newer is required)

Windows: Windows 10 or newer is required

### Installation

#### Standard
````
pip install cpp-zarr
````

#### With GPU reads and writes
Reading straight into GPU memory and writing from it need Linux, an NVIDIA GPU (Volta or newer) and its driver. The gpu extra also installs nvCOMP, which decompresses and compresses on the GPU and which cppzarr finds by itself (see "Read straight into GPU memory" and "Write from GPU memory" below).
````
pip install "cpp-zarr[gpu]"

# gpu=True returns CuPy arrays, so it also needs CuPy (reading into PyTorch tensors does not)
pip install cupy-cuda12x
````

### Usage

The reader returns a numpy array for the given zarr file with optional arguments for reading a region

The writer takes an output filename and a numpy array with optional arguments for setting metadata

The following compressors are supported: blosclz, lz4, lz4hc, gzip, zlib, zstd, and none (uncompressed)

The following data types are supported: bool, uint8, int8, uint16, int16, uint32, int32, uint64, int64, float16, float32/single, float64/double, complex64, complex128, and bfloat16 (the Zarr extension data type; reading and writing it in Python needs the ml_dtypes package: `pip install ml_dtypes`)

zstd is recommended for when you want smaller file sizes or disk limited processing such as when using a cluster

lz4 is recommended for when file sizes are not a concern or cpu limited processing such as when only using a single machine

#### Read and Write a Zarr file
````
import cppzarr
im = cppzarr.read_zarr('filename.zarr')
# Do some processing here
cppzarr.write_zarr('outputFilename.zarr', im)

# Write a Zarr v3 file instead
cppzarr.write_zarr('outputFilename.zarr', im, zarr_format=3)

# On Linux and macOS a path that starts with ~ or ~name is in that home folder (as a shell expands it);
# nothing else in a path is expanded: spaces, parentheses, $ and a ~ further on are kept as they are
im = cppzarr.read_zarr('~/data/run (2).zarr')

# Paths are UTF-8 on every platform, so folders named in any script work on Windows too
# (the C++ functions take UTF-8 as well; there, a path whose bytes are not valid UTF-8 is read in
# Windows' ANSI code page, as before)
im = cppzarr.read_zarr('C:/data/café µm/a.zarr')
````

#### Read a region of a Zarr file
````
import cppzarr

# Optionally specify the start coordinates
im = cppzarr.read_zarr('filename.zarr', start_coords=[10, 10, 10])

# Optionally specify the end coordinates
im1 = cppzarr.read_zarr('filename.zarr', end_coords=[100, 100, 100])

# Optionally specify the start and end coordinates
im2 = cppzarr.read_zarr('filename.zarr', start_coords=[10, 10, 10], end_coords=[100, 100, 100])
````

#### Memory layout
read_zarr returns arrays in the file's own storage order by default, like zarr-python: C-order files as C-order arrays (the last axis is contiguous) and F-order files as F-order arrays (the first axis is contiguous). Pass order='C' or order='F' to choose the layout. write_zarr takes arrays in either order, and strided views, without copying them; its order argument sets the file's storage order (F by default, C for Zarr v3).
````
import cppzarr
im = cppzarr.read_zarr('filename.zarr', order='C')
````

#### Write a Zarr file with specific metadata
````
import numpy as np
import cppzarr
im = cppzarr.read_zarr('filename.zarr')

# Optionally specify specific metadata
cppzarr.write_zarr('outputFilename.zarr', im, cname='zstd', clevel=1, order='F', chunks=[256, 256, 256], dimension_separator='/')

# New Zarr v2 files put their chunk files in nested folders (dimension_separator='/', files named 0/0/0) by default.
# dimension_separator='.' names them 0.0.0 in one folder, which writes much slower on network file systems.

# Zarr v3 with 1024x1024x1024 shards made of 256x256x256 chunks (shards must be a multiple of chunks).
# Zarr v3 files default to order='C' and dimension_separator='/' (chunk files named c/0/0/0)
cppzarr.write_zarr('outputFilename.zarr', im, zarr_format=3, cname='zstd', clevel=1, chunks=[256, 256, 256], shards=[1024, 1024, 1024])

# Or create an array's metadata alone (with write_zarr's options), then write regions into it: what is
# never written reads as zeros
cppzarr.create_zarr('big.zarr', (4096, 4096, 2048), 'uint16', chunks=[256, 256, 256], zarr_format=3)
cppzarr.write_zarr('big.zarr', im, start_coords=[0, 0, 0], end_coords=list(im.shape))

# Data written into an array of another data type is converted as MATLAB converts between classes, as
# the MATLAB writer converts it: to integers rounded (halves away from zero) and saturated, NaN to 0
cppzarr.write_zarr('big.zarr', np.full((2, 2, 2), 70000.6), start_coords=[0, 0, 0], end_coords=[2, 2, 2])  # 65535

# Chunks whose values are all zero are not written (they read back as zeros); sparse=False writes them.
# uuid=False writes the chunk files in place instead of under temporary names (for a new array with one
# writer per chunk)
cppzarr.write_zarr('outputFilename.zarr', im, sparse=False, uuid=False)
````

#### Data types
````
import numpy as np
import cppzarr

# The array's data type is the numpy array's: float16, complex64 and bool arrays included
cppzarr.write_zarr('half.zarr', np.random.rand(256, 256, 64).astype(np.float16))

# bfloat16 arrays are ml_dtypes' bfloat16 type
import ml_dtypes
cppzarr.write_zarr('bf16.zarr', np.random.rand(256, 256, 64).astype(ml_dtypes.bfloat16), zarr_format=3)
````

#### Threads, reading into an array, and reading from several threads or processes
````
import numpy as np
import cppzarr

# A call uses every CPU by default (or OMP_NUM_THREADS); num_threads sets it for one call
im = cppzarr.read_zarr('filename.zarr', num_threads=4)

# Read a region into an existing array instead of a new one (of the region's shape and the
# array's data type), such as one slot of a batch buffer
batch = np.empty((8, 128, 128, 128), dtype=np.uint16)
cppzarr.read_zarr('filename.zarr', [0, 0, 0], [128, 128, 128], out=batch[0])

# out can be a view of any layout of positive strides too, such as a plane for each channel of
# (t, z, y, x, c) data, or a region of a bigger array: only its own elements are written
planes = np.empty((2, 2, 128, 1024, 1024), dtype=np.uint16)                        # (t, c, z, y, x)
cppzarr.read_zarr('tzyxc.zarr', [0] * 5, [2, 128, 1024, 1024, 2], out=np.moveaxis(planes, 1, -1))

# Reads and writes release the GIL, so several Python threads can read at once: give each
# a share of the CPUs with num_threads. The same goes for several processes reading at once
# (a multiprocessing pool, a data loader's workers): give each process a share with
# num_threads (a bigger one to those with more to read), or OMP_NUM_THREADS set before
# cppzarr is imported. Processes that each use every CPU make one another's reads several
# times slower. Processes made by fork() (multiprocessing's default on Linux, PyTorch's
# DataLoader workers) can read and write too, also after their parent did with several threads.

# Or read a batch of regions at once (from one array or several), new arrays or into given ones:
# the reads overlap and share num_threads threads (by default, as for a call, OMP_NUM_THREADS
# or every CPU), each read getting more of them the larger it is (a thread per 32M elements,
# so that a batch of many reads runs many of them at once; with more than
# CPPZARR_READ_AHEAD_THREADS threads, see below, a thread per 8M). Each array's metadata is read
# once for the batch, ahead of its reads. On network storage (Linux), reads of one thread each
# (single time points and other small regions) wait for the server before their data arrives,
# so up to CPPZARR_BATCH_READS_PER_THREAD (3) of them run at once for each thread, as many as
# their measured wait calls for (none more while they wait less than half their CPU time, nor
# in a batch that has had no data fetched from the storage, as from the page cache), where
# that has read faster: a process tries it for each batch size, from its third batch of it
regions = [('filename.zarr', [0, 0, 128*i], [128, 128, 128*(i+1)]) for i in range(8)]
arrays = cppzarr.read_zarr_batch(regions, num_threads=8)
cppzarr.read_zarr_batch([r + (batch[i],) for i, r in enumerate(regions)], num_threads=8)

# On network storage (NFS and the like), reads keep more reads of the storage in flight than
# they have threads (Linux): in calls of 2 to 5 threads, once a thread finds its shard's
# chunks come from the server, it asks for its next ones ahead, and in calls of up to 8 threads
# a pool of threads opens and reads the next chunk files of arrays without shards (while the
# process has no CPU to spare for the pool, a thread whose files come from the page cache reads
# them itself, the pool only opening them).
# CPPZARR_READ_AHEAD_MB (4) sets how far a thread reads ahead (0: not at all; in shards half as
# far while the process's reads have more than 16 threads), CPPZARR_READS the most threads the
# pool uses: by default 64 for a process that may use all of its machine's CPUs, and fewer for
# one that may use fewer (its share of 64, as its CPU binding and cgroup limit it, as a job of a
# few of a node's CPUs is; at least two for each of the call's threads), since processes that
# share a node share its storage too. Neither reads ahead while the process's reads have more than
# CPPZARR_READ_AHEAD_THREADS (64) threads in all, where reading ahead slowed reads down.
````

#### Read straight into GPU memory
GPU reads are supported on Linux for now: Windows builds include them, but they have not been
tested on a Windows GPU yet, and macOS has no NVIDIA GPUs (`gpu_available()` says why a machine
cannot read). `pip install "cpp-zarr[gpu]"` also installs nvCOMP, which cppzarr finds by itself; the
NVIDIA driver has to be installed already.
````
import cupy
import torch
import cppzarr

# gpu=True returns a CuPy array instead of a NumPy array: the chunks' compressed bytes are
# read, copied to the GPU and decompressed there with nvCOMP. For data needed on the GPU anyway,
# that is much less CPU work than decompressing on the CPU and copying the result. Needs an
# NVIDIA GPU (Volta or newer), CuPy (pip install cupy-cuda12x), and nvCOMP (pip install "cpp-zarr[gpu]")
im = cppzarr.read_zarr('filename.zarr', [0, 0, 0], [128, 128, 128], gpu=True)

# Or read into GPU memory you have (a CuPy array, a PyTorch CUDA tensor, or anything with
# __cuda_array_interface__); bfloat16 arrays need a PyTorch tensor, since CuPy has no bfloat16
out = cupy.empty((128, 128, 128), dtype=cupy.uint16)
cppzarr.read_zarr('filename.zarr', [0, 0, 0], [128, 128, 128], out=out)

# A batch goes to the GPU together (one decompression for all of its chunks, which is what GPUs
# do fast), here into the slots of a batch tensor, on a stream (the default stream otherwise);
# the call returns once the data is in place
batch = torch.empty((8, 128, 128, 128), dtype=torch.uint16, device='cuda')
regions = [('filename.zarr', [0, 0, 128*i], [128, 128, 128*(i+1)]) for i in range(8)]
cppzarr.read_zarr_batch([r + (batch[i],) for i, r in enumerate(regions)], stream=torch.cuda.current_stream())

# Whether GPU reads can run here, and if not, why
ok, why = cppzarr.gpu_available()
````
The GPU decompresses blosc's zstd, lz4 and zlib streams (byte shuffle or none) and Zarr's zstd and
gzip codecs; other chunks (blosclz, bitshuffle, ...) are decompressed on the CPU in the same call.
The values are the same as `read_zarr`'s, missing chunks the fill value. nvCOMP checks less than
the CPU's decoders do: the reader checks each zstd stream's structure first (one that fails is
decoded on the CPU, which reports damage), but damage inside a stream's entropy-coded data can
make nvCOMP fail with a GPU error or never finish (the call then returns an error after
CPPZARR_GPU_TIMEOUT seconds, 30 plus 1 per 50 MB by default). Arrays with checksums (Zarr v3's
crc32c codec) are checked on the CPU before anything goes to the GPU. For GPU reads, num_threads
sets the threads that prepare the chunks; the files are read with at least 128 threads at once
(CPPZARR_GPU_READS), since reading mostly waits on the storage.

#### Write from GPU memory
Data in GPU memory (a CuPy array, a PyTorch CUDA tensor, or anything with
`__cuda_array_interface__`, in any memory layout) is written from there, on Linux as GPU reads
are: the chunks are put together and compressed on the GPU with nvCOMP, and only the compressed
chunks come back to host memory, where the call's threads write their files as for data in host
memory (shards, writes into existing arrays, all-zero chunks left out). For data on the GPU anyway,
that is much faster than copying it to host memory and writing it from there, and takes less of
the CPU.
````
import cupy
import cppzarr

im = cupy.random.randint(0, 1000, (512, 512, 512), dtype=cupy.uint16)
cppzarr.write_zarr('filename.zarr', im, zarr_format=3, chunks=[64, 64, 64], shards=[256, 256, 256])

# The GPU work goes on a stream (a PyTorch or CuPy stream, or a CUDA stream handle; by default the
# data's library's current stream, where the work that made the data most likely went), after what
# is already on it; the call returns once the files are written
cppzarr.write_zarr('filename.zarr', im, stream=cupy.cuda.Stream.null)
````
The GPU compresses blosc's zstd and lz4 streams (byte shuffle or none) and Zarr's zstd codec, at
nvCOMP's one level of each whatever clevel says (blosc's block size still follows clevel, or a Zarr
v3 array's blocksize): the chunks decode to the same values as the CPU writer's, with other bytes.
Arrays of other compressors (blosclz, zlib, gzip, bitshuffle) are written from a copy of the data in
host memory. Several writes from as many threads at once overlap: the files of one are written while
the GPU compresses another's. CPPZARR_GPU_WRITE_PART_MB (256) sets the most of a write the GPU works
on at once (a bigger write goes in parts, the files of one written while the GPU works on the next).

#### Write a region to an existing Zarr file
````
import cppzarr
im = cppzarr.read_zarr('filename.zarr')

# Write the zarr file out normally
cppzarr.write_zarr('filename.zarr', im)

# Write to a specified region with different data (the file keeps its Zarr version)
cppzarr.write_zarr('filename.zarr', im[100:200,100:200,100:200], start_coords=[0,0,0], end_coords=[100,100,100])

# A write into an existing array compresses as its metadata says: a Zarr v3 array's blosc shuffle
# and blocksize included (v2 arrays get byte shuffle and blosc's own block size, as cpp-zarr has
# always written them)

# Writes keep the buffers of chunks over 1 MB for later calls, and so do reads for the chunk files
# over 1 MB they read ahead from network storage, up to CPPZARR_KEEP_MB MB in all (0: none; by
# default 1024, or an eighth of the memory the process may use if that is less: its job's memory
# limit, its address-space limit or the machine's memory; writes' first), so that a big array
# written or read a few chunks at a time does not allocate them again every call. On Linux the
# system may take them back while they are idle, when it or the job runs short of memory. With
# fewer chunks of 4 MB or more than threads, every thread helps put the chunks together. A
# long-lived process that is done reading and writing for a while can free them:
cppzarr.release_buffers()
````

#### Convert a Zarr v2 file to Zarr v3
````
import cppzarr

# Only the metadata is rewritten (.zarray and .zattrs become zarr.json); the chunk files are kept.
# Files with subfolders or numcodecs zlib compression cannot be converted
cppzarr.convert_to_v3('filename.zarr')
````

## CMake

The C++ library can be compiled using the CMakeLists.txt file

### Prerequisites
1. c-blosc, c-blosc2, nlohmann/json, and zlib are fetched automatically at configure time via CMake FetchContent, so an internet connection is required for the first configure. On Linux/Mac, libuuid is built from the tarball in the dependencies folder.
2. Currently the only officially supported compiler is gcc on Linux and Mac and MinGW on Windows but others may work

### Download and Install
````
git clone https://github.com/abcucberkeley/cpp-zarr
cd cpp-zarr
mkdir build
cd build
cmake ..
make -j
make install
````

### Reading into GPU memory
gpureadzarr.h reads regions of arrays straight into GPU memory, decompressing on the GPU (see the
Python section; Linux only for now). Building needs nothing more: the library loads the NVIDIA driver and nvCOMP (5 or
newer; set CPPZARR_NVCOMP_LIBRARY to its libnvcomp file, or put it on the library path) when it
first reads on a GPU.
````
#include "gpureadzarr.h"
zarr z("filename.zarr");
// out: device memory for 128x128x128 uint16s (cudaMalloc, or a framework's tensor)
std::vector<GpuRegion> regions{{&z, {0, 0, 0}, {128, 128, 128}, out}};
std::string err;
if(gpuReadZarr(regions, true /* C order */, stream /* or nullptr */, 0 /* threads: the default */, err))
    printf("%s\n", err.c_str());
````

### Writing from GPU memory
gpuwritezarr.h writes a region of an array from GPU memory, compressing on the GPU (see the Python
section; Linux only for now), as parallelWriteZarr writes one from host memory.
````
#include "gpuwritezarr.h"
// (the array's metadata written, or an existing one opened for a crop write, as for parallelWriteZarr)
zarr z("filename.zarr");
z.set_chunkInfo({0, 0, 0}, {128, 128, 128});
// data: device memory holding 128x128x128 uint16s in F order (element strides 1, 128, 128*128)
if(gpuWriteZarr(z, data, {0, 0, 0}, {128, 128, 128}, {128, 128, 128}, {1, 128, 128*128}, 16, true /* uuid */,
                false /* crop */, true /* sparse */, stream /* or nullptr */))
    printf("%s\n", z.get_errString().c_str());
````

## MATLAB

### Prerequisites
1. All necessary libraries are included for the Linux, Mac, and Windows versions.
2. For Linux and Windows, a CPU with AVX/SSE support is required for the reader (Almost all modern CPUs should be compatible)

### Download and Install
1. Download the latest release for your OS from here: https://github.com/abcucberkeley/cpp-zarr/releases
2. Unzip the folder
3. You can now put the folders wherever you'd like and add them to your path if needed. Keep the mex files with their associated library files so the mex function can always run.
4. Note for Mac Users: You may need to restart Matlab before using the Mex files if you have an open session

### Usage

#### createZarrFile - Create a custom .zarray metadata file
````
% Note the created .zarray file is probably hidden by default on your system
createZarrFile('path/to/file.zarr');

% Zarr v3 metadata (zarr.json) instead, with 1024x1024x1024 shards made of 256x256x256 chunks
createZarrFile('path/to/file.zarr','shape',[2048 2048 1024],'dtype','<u2','chunks',[1024 1024 1024],'chunk_shape',[256 256 256],'zarr_format',3);
````

#### parallelReadZarr - Read a Zarr image into an array
````
im = parallelReadZarr('path/to/file.zarr');

% Read a region: [starts ends] with one start and one end per axis (1-based, inclusive)
im = parallelReadZarr('path/to/file.zarr', 'bbox', [1 1 1 100 100 50]);

% Read with 4 threads. By default a call uses OpenMP's thread count for MATLAB, which on Linux
% follows maxNumCompThreads (OMP_NUM_THREADS has no effect there). parallelWriteZarr,
% parallelReadZarrGPU and parallelWriteZarrGPU take 'threads' too.
im = parallelReadZarr('path/to/file.zarr', 'threads', 4);
````

#### parallelWriteZarr - Write an array out as a Zarr image
````
im = rand(100,100,100);
parallelWriteZarr('path/to/file.zarr',im);

% Write with a chunk size (one value per axis)
parallelWriteZarr('path/to/file.zarr',im,'chunks',[64 64 64]);

% Write with a compressor and compression level (blosc's lz4 at level 5 by default; levels go from 0 to 9)
parallelWriteZarr('path/to/file.zarr',im,'cname','zstd','clevel',1);

% Write a new array of another data type than the data's (here float16; also 'bfloat16', 'uint16', 'single', '<u2', ...).
% MATLAB has no 16-bit float class: float16 and bfloat16 arrays read as single. Logical arrays are written as bool,
% complex single and double as complex64 and complex128, and data written into an existing array is converted to its
% data type as MATLAB converts between classes.
parallelWriteZarr('path/to/file.zarr',single(im),'dtype','float16');

% Write into a region of an existing Zarr file ([starts ends] as in parallelReadZarr); the file keeps its Zarr version
parallelWriteZarr('path/to/file.zarr',im(1:50,:,:),'bbox',[1 1 1 50 100 100]);

% Write a Zarr v3 file (C order and chunk files named c/0/0/0 by default; 'order','F' still works)
parallelWriteZarr('path/to/file.zarr',im,'zarr_format',3);

% New Zarr v2 files put their chunk files in nested folders (files named 0/0/0) by default. '.' names them
% 0.0.0 in one folder, which writes much slower on network file systems.
parallelWriteZarr('path/to/file.zarr',im,'dimension_separator','.');

% Write each chunk file directly. By default each one is written to a temporary file that is then
% renamed into place, so readers never see a partly written chunk and a write that fails keeps the
% old one. Skipping that writes faster on network file systems. It is safe for a new array that
% nothing reads until the write is done (for example one written under a temporary name and renamed
% after), as long as no two writers write the same chunk.
parallelWriteZarr('path/to/file.zarr',im,'uuid',false);

% Writes keep the buffers of chunks over 1 MB for later calls, and so do reads for the chunk files over
% 1 MB they read ahead from network storage (up to CPPZARR_KEEP_MB MB in all: by default 1024, or an
% eighth of the memory the job may use if that is less). clear mex frees them.
clear mex
````

#### parallelReadZarrGPU - Read a Zarr image into a gpuArray
````
% As parallelReadZarr, decompressing on the GPU (Linux; needs the Parallel Computing Toolbox, an
% NVIDIA GPU and nvCOMP: setenv('CPPZARR_NVCOMP_LIBRARY', 'path/to/libnvcomp.so.6') before the first
% read, or put it next to libcppZarr.so)
im = parallelReadZarrGPU('path/to/file.zarr', 'bbox', [1 1 1 100 100 50]);
````

#### parallelWriteZarrGPU - Write a gpuArray out as a Zarr image
````
% As parallelWriteZarr, with the same options, compressing on the GPU (Linux; needs the Parallel
% Computing Toolbox, an NVIDIA GPU and nvCOMP, as parallelReadZarrGPU). Data of another class than
% the array's data type is converted on the host and written from there: cast it on the GPU first,
% as uint16(im), to keep the write on the GPU.
im = gpuArray(uint16(randi(1000, 512, 512, 512)));
parallelWriteZarrGPU('path/to/file.zarr', im, 'cname', 'zstd', 'clevel', 1, 'zarr_format', 3);
````

#### convertZarrToV3 - Convert a Zarr v2 file to Zarr v3
````
% Only the metadata is rewritten (.zarray and .zattrs become zarr.json); the chunk files are kept.
% Files with subfolders or numcodecs zlib compression cannot be converted
convertZarrToV3('path/to/file.zarr');
````

## Reference

Please cite our software if you find it useful in your work:

Xiongtao Ruan, Matthew Mueller, Gaoxiang Liu, Frederik Görlitz, Tian-Ming Fu, Daniel E. Milkie, Joshua L. Lillvis, Alexander Kuhn, Chu Yi Aaron Herr, Wilmene Hercule, Marc Nienhaus, Alison N. Killilea, Eric Betzig, Srigokul Upadhyayula. Image processing tools for petabyte-scale light sheet microscopy data. Nature Methods (2024). https://doi.org/10.1038/s41592-024-02475-4
