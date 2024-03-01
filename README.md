# cpp-zarr
An efficient parallel Zarr reader/writer that utilizes c-blosc/c-blosc2 and OpenMP.

## Quick Start Guide (MATLAB)

### Prerequisites
1. All necessary libraries are included for the Linux, Mac, and Windows versions.

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

## Compiling with CMake

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

## Reference

Please cite our software if you find it useful in your work:

Xiongtao Ruan, Matthew Mueller, Gaoxiang Liu, Frederik Görlitz, Tian-Ming Fu, Daniel E. Milkie, Joshua L. Lillvis, Alexander Kuhn, Chu Yi Aaron Herr, Wilmene Hercule, Marc Nienhaus, Alison N. Killilea, Eric Betzig, Srigokul Upadhyayula. Image processing tools for petabyte-scale light sheet microscopy data. bioRxiv 2023.12.31.573734; doi: https://doi.org/10.1101/2023.12.31.573734
