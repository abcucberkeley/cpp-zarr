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
````
pip install cpp-zarr
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
import cppzarr
im = cppzarr.read_zarr('filename.zarr')

# Optionally specify specific metadata
cppzarr.write_zarr('outputFilename.zarr', im, cname='zstd', clevel=1, order='F', chunks=[256, 256, 256], dimension_separator='/')

# New Zarr v2 files put their chunk files in nested folders (dimension_separator='/', files named 0/0/0) by default.
# dimension_separator='.' names them 0.0.0 in one folder, which writes much slower on network file systems.

# Zarr v3 with 1024x1024x1024 shards made of 256x256x256 chunks (shards must be a multiple of chunks).
# Zarr v3 files default to order='C' and dimension_separator='/' (chunk files named c/0/0/0)
cppzarr.write_zarr('outputFilename.zarr', im, zarr_format=3, cname='zstd', clevel=1, chunks=[256, 256, 256], shards=[1024, 1024, 1024])
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
# array's data type, C- or F-contiguous), such as one slot of a batch buffer
batch = np.empty((8, 128, 128, 128), dtype=np.uint16)
cppzarr.read_zarr('filename.zarr', [0, 0, 0], [128, 128, 128], out=batch[0])

# Reads and writes release the GIL, so several Python threads can read at once: give each
# a share of the CPUs with num_threads. The same goes for several processes reading at once
# (a multiprocessing pool, a data loader's workers): give each process a share with
# num_threads (a bigger one to those with more to read), or OMP_NUM_THREADS set before
# cppzarr is imported. Processes that each use every CPU make one another's reads several
# times slower.

# Or read a batch of regions at once (from one array or several), new arrays or into given ones:
# the reads overlap and share num_threads threads (by default, as for a call, OMP_NUM_THREADS
# or every CPU), each read getting more of them the larger it is
regions = [('filename.zarr', [0, 0, 128*i], [128, 128, 128*(i+1)]) for i in range(8)]
arrays = cppzarr.read_zarr_batch(regions, num_threads=8)
cppzarr.read_zarr_batch([r + (batch[i],) for i, r in enumerate(regions)], num_threads=8)
````

#### Read straight into GPU memory
GPU reads are supported on Linux for now: Windows builds include them, but they have not been
tested on a Windows GPU yet, and macOS has no NVIDIA GPUs (`gpu_available()` says why a machine
cannot read). `pip install cpp-zarr[gpu]` also installs nvCOMP, which cppzarr finds by itself; the
NVIDIA driver has to be installed already.
````
import cupy
import torch
import cppzarr

# gpu=True returns a CuPy array instead of a NumPy array: the chunks' compressed bytes are
# read, copied to the GPU and decompressed there with nvCOMP. For data needed on the GPU anyway,
# that is much less CPU work than decompressing on the CPU and copying the result. Needs an
# NVIDIA GPU (Volta or newer), CuPy, and nvCOMP: pip install cpp-zarr[gpu]
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
sets the threads that prepare the chunks; the files are read with at least 32 threads at once
(CPPZARR_GPU_READS), since reading mostly waits on the storage.

#### Write a region to an existing Zarr file
````
import cppzarr
im = cppzarr.read_zarr('filename.zarr')

# Write the zarr file out normally
cppzarr.write_zarr('filename.zarr', im)

# Write to a specified region with different data (the file keeps its Zarr version)
cppzarr.write_zarr('filename.zarr', im[100:200,100:200,100:200], start_coords=[0,0,0], end_coords=[100,100,100])
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
````

#### parallelReadZarrGPU - Read a Zarr image into a gpuArray
````
% As parallelReadZarr, decompressing on the GPU (Linux; needs the Parallel Computing Toolbox, an
% NVIDIA GPU and nvCOMP: setenv('CPPZARR_NVCOMP_LIBRARY', 'path/to/libnvcomp.so.6') before the first
% read, or put it next to libcppZarr.so)
im = parallelReadZarrGPU('path/to/file.zarr', 'bbox', [1 1 1 100 100 50]);
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
