#ifndef GPUREADZARR_H
#define GPUREADZARR_H
#include <cstdint>
#include <string>
#include <vector>
#include "zarr.h"

// Reading regions of Zarr arrays straight into GPU memory (NVIDIA GPUs, CUDA). The chunks'
// stored bytes are read on the host, copied to the GPU as they are, decompressed there with
// nvCOMP (blosc's zstd, lz4, zlib and snappy streams with byte shuffle or none, and the zstd and
// gzip codecs; others are decompressed on the host) and put in place by the GPU: what the CPU reader
// gives, in GPU memory, for when the data is needed on the GPU anyway.
//
// Needs, at run time only, an NVIDIA driver and nvCOMP 5 or later (pip install
// nvidia-libnvcomp-cu12, or set CPPZARR_NVCOMP_LIBRARY to its libnvcomp file): cpp-zarr loads
// them on the first GPU read, and builds and runs without them.
//
// Environment variables: CPPZARR_NVCOMP_LIBRARY (nvCOMP's library file); CPPZARR_GPU_TIMEOUT
// (seconds a read waits for the GPU before it fails, 0 for no limit; by default 30 plus 1 per 50 MB
// decompressed: nvCOMP can hang on damaged data); CPPZARR_GPU_PROFILE (each read prints where its
// time went to stderr); CPPZARR_GPU_PART_MB (a read goes to the GPU in up to 4 parts of at least
// this many MB of stored bytes, 32 by default, so that reading the next part overlaps the GPU's
// work on the one before); CPPZARR_GPU_READS (the files are read with at least this many threads
// at once, 32 by default: reading mostly waits on the storage).

// Whether GPU reads can run here; if not, why (no NVIDIA driver, no nvCOMP, ...)
bool gpuReadAvailable(std::string &why);

// Where to load nvCOMP from (a libnvcomp file), before the first GPU read
void gpuSetNvcompLibrary(const std::string &path);

// One region to read: an opened array, the region [start, end) (whole array when both are
// empty) and the GPU memory it goes to (the region's shape, in the call's order), all regions'
// on the GPU of one CUDA context (the device's primary context for memory that CUDA runtime
// users such as PyTorch, CuPy and MATLAB allocate)
struct GpuRegion {
    zarr* array;
    std::vector<uint64_t> start, end;
    void* out;
};

// Reads the regions into their GPU memory, C order or F order (cOrder), with the GPU work on
// stream (a CUstream / cudaStream_t; nullptr for the default stream), and returns once it is
// done: 0, or 1 with the reason in errString. threads: the host threads that prepare the
// chunks for the GPU (0: the call's threads, as for parallelReadZarr); the files are read with at
// least CPPZARR_GPU_READS (32) at once, more for slow storage helping more than it costs, since
// a read waits more than it computes. The regions are read together: one decompression and one
// placement for all of their chunks, which a GPU does best in big batches.
// float16AsFloat32: float16 and bfloat16 arrays are read as float32 (their regions' memory
// holds float32s), for callers with no 16-bit floats (MATLAB), as its CPU reads do.
uint8_t gpuReadZarr(std::vector<GpuRegion> &regions, const bool cOrder, void* stream,
                    const int threads, std::string &errString, const bool float16AsFloat32 = false);

#endif
