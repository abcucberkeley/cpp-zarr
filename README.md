# cpp-zarr
An efficient parallel Zarr reader/writer that utilizes c-blosc/c-blosc2 and OpenMP.

## Limitations
1. Currently only 3D Zarr files are officially supported but better support may be added in the future
   1. 1D and 2D Zarr files are supported if the other axes are 1 like [10,1,1] or [10,10,1]
2. Currently the speed for reading and writing F order data is much faster than C order but better support may be added in the future

## Python

A Python version of cpp-zarr is available through pip

### Prerequisites

#### Python
1. Python version >=3.8

#### CPU
1. A CPU with AVX/SSE support is required for the reader (Almost all modern CPUs should be compatible)

#### OS
Linux: All Linux distros made within the past 10 years should work

Mac Apple Silicon (M1, M2, etc.): macOS 13 or newer is required

Mac Intel: macOS 12 or newer is required

Windows: Windows 10 or newer is required

### Installation
````
pip install cpp-zarr
````

### Usage

The reader returns a numpy array for the given zarr file with optional arguments for reading a region

The writer takes an output filename and a numpy array with optional arguments for setting metadata

The following compressors are supported: blosclz, lz4, lz4hc, gzip, zlib, zstd

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
1. Dependencies are included in the dependencies folder
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
````

#### parallelWriteTiff - Write an array out as a Zarr image
````
im = rand(100,100,100);
% The third input can always be 1 to use a uuid for the written blocks
% The fourth input is the size of the blocks
parallelWriteZarr('path/to/file.zarr',im);
````

## Reference

Please cite our software if you find it useful in your work:

Xiongtao Ruan, Matthew Mueller, Gaoxiang Liu, Frederik Görlitz, Tian-Ming Fu, Daniel E. Milkie, Joshua L. Lillvis, Alexander Kuhn, Chu Yi Aaron Herr, Wilmene Hercule, Marc Nienhaus, Alison N. Killilea, Eric Betzig, Srigokul Upadhyayula. Image processing tools for petabyte-scale light sheet microscopy data. Nature Methods (2024). https://doi.org/10.1038/s41592-024-02475-4
