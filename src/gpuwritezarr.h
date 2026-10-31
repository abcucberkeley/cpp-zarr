#ifndef GPUWRITEZARR_H
#define GPUWRITEZARR_H
#include <cstdint>
#include <string>
#include <vector>
#include "zarr.h"

// Writing regions of Zarr arrays from GPU memory (NVIDIA GPUs, CUDA): parallelWriteZarr's write,
// with the chunks put together and compressed on the GPU and stored on the host as
// parallelWriteZarr stores them (chunk files and shards, crop writes into existing arrays, sparse
// writes, checksums). The GPU compresses blosc's zstd and lz4 streams (with byte shuffle or none)
// and the zstd codec with nvCOMP, at its one level of each whatever the array's clevel (which
// still sets blosc's block size): the chunks decode as the CPU writer's do, with other bytes.
// Arrays of other codecs (blosclz, zlib, gzip, bitshuffle) are written from a copy of the region
// in host memory by parallelWriteZarr.
//
// Needs, at run time only, what GPU reads need (gpureadzarr.h): an NVIDIA driver and nvCOMP 5 or
// later. Linux for now. Environment variables: CPPZARR_GPU_WRITE_PART_MB (a write goes to the GPU
// in parts of at most this many MB of chunks, 256 by default, so that writing one part's files
// overlaps the GPU's work on the next and the GPU's buffers stay small); CPPZARR_GPU_PROFILE (each
// write prints where its time went to stderr).

// Whether GPU writes can run here; if not, why (no NVIDIA driver, no nvCOMP, ...)
bool gpuWriteAvailable(std::string &why);

// Writes the region [startCoords, endCoords) of Zarr from data, GPU memory holding the region
// (writeShape elements along each axis, inStrides elements apart along each axis), with the GPU
// work on stream (a CUstream / cudaStream_t; nullptr for the default stream), and returns once
// the files are written: 0, or 1 with the reason in Zarr's errString. As for parallelWriteZarr:
// Zarr's metadata is written (or read, for a crop write into an existing array) and
// set_chunkInfo(startCoords, endCoords) called before, bits is the element size in bits, and
// useUuid, crop and sparse mean what they mean there; the files are written with the call's
// threads (OpenMP's, as for parallelWriteZarr).
uint8_t gpuWriteZarr(zarr &Zarr, const void* data, const std::vector<uint64_t> &startCoords,
                     const std::vector<uint64_t> &endCoords, const std::vector<uint64_t> &writeShape,
                     const std::vector<uint64_t> &inStrides, const uint64_t bits, const bool useUuid,
                     const bool crop, const bool sparse = true, void* stream = nullptr);
#endif
