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

The following data types are supported: uint8, int8, uint16, int16, uint32, int32, uint64, int64, float32/single, and float64/double

zstd is recommended for when you want smaller file sizes or disk limited processing such as when using a cluster

lz4 is recommended for when file sizes are not a concern or cpu limited processing such as when only using a single machine

#### Read and Write a Zarr file
````
import cppzarr
im = cppzarr.read_zarr('filename.zarr')
# Do some processing here
cppzarr.write_zarr('outputFilename.zarr', im)
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
read_zarr returns arrays in the file's own storage order by default, like zarr-python: C-order files as C-order arrays (the last axis is contiguous) and F-order files as F-order arrays (the first axis is contiguous). Pass order='C' or order='F' to choose the layout. write_zarr takes arrays in either order, and strided views, without copying them; its order argument sets the file's storage order (F by default).
````
import cppzarr
im = cppzarr.read_zarr('filename.zarr', order='C')
````

#### Write a Zarr file with specific metadata
````
import cppzarr
im = cppzarr.read_zarr('filename.zarr')

# Optionally specify specific metadata
cppzarr.write_zarr('outputFilename.zarr', im, cname='zstd', clevel=1, order='F', chunks=[256, 256, 256], dimension_separator='.')
````

#### Write a region to an existing Zarr file
````
import cppzarr
im = cppzarr.read_zarr('filename.zarr')

# Write the zarr file out normally
cppzarr.write_zarr('filename.zarr', im)

# Write to a specified region with different data
cppzarr.write_zarr('filename.zarr', im[100:200,100:200,100:200], start_coords=[0,0,0], end_coords=[100,100,100])
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
% Write into a region of an existing Zarr file ([starts ends] as in parallelReadZarr)
parallelWriteZarr('path/to/file.zarr',im(1:50,:,:),'bbox',[1 1 1 50 100 100]);
````

## Reference

Please cite our software if you find it useful in your work:

Xiongtao Ruan, Matthew Mueller, Gaoxiang Liu, Frederik Görlitz, Tian-Ming Fu, Daniel E. Milkie, Joshua L. Lillvis, Alexander Kuhn, Chu Yi Aaron Herr, Wilmene Hercule, Marc Nienhaus, Alison N. Killilea, Eric Betzig, Srigokul Upadhyayula. Image processing tools for petabyte-scale light sheet microscopy data. Nature Methods (2024). https://doi.org/10.1038/s41592-024-02475-4
