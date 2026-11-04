#ifndef PARALLELWRITEZARR_H
#define PARALLELWRITEZARR_H
#include <cstdint>
#include "zarr.h"

uint8_t parallelWriteZarr(zarr &Zarr, void* zarrArr,
                          const std::vector<uint64_t> &startCoords,
                          const std::vector<uint64_t> &endCoords,
                          const std::vector<uint64_t> &writeShape,
                          const uint64_t bits, const bool useUuid,
                          const bool crop, const bool sparse=true);

// Same as above, for an input array in any memory layout: inStrides gives the
// distance in elements between neighboring elements along each axis (the version
// above takes an F-order array, where the first axis is contiguous)
uint8_t parallelWriteZarr(zarr &Zarr, void* zarrArr,
                          const std::vector<uint64_t> &startCoords,
                          const std::vector<uint64_t> &endCoords,
                          const std::vector<uint64_t> &writeShape,
                          const std::vector<uint64_t> &inStrides,
                          const uint64_t bits, const bool useUuid,
                          const bool crop, const bool sparse=true);

// Frees the buffers reads and writes keep between calls (the writer's for chunks over 1 MB,
// up to CPPZARR_WRITE_KEEP_MB in all, and the readers'), and hands memory the C library keeps
// back to the system (glibc's malloc_trim). For long-lived processes: MATLAB's clear mex calls
// it, Python's release_buffers().
void releaseBuffers();
// The bytes of the writer's buffers for chunks over 1 MB kept between calls, and the most it
// keeps (CPPZARR_WRITE_KEEP_MB: by default 1024 MB, or an eighth of the process's memory budget
// if less)
uint64_t keptWriteBytesNow();
uint64_t keptWriteBytesCap();
#endif
