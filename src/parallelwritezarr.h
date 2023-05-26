#ifndef PARALLELREADZARR_H
#define PARALLELREADZARR_H
#include <stdint.h>
#include "zarr.h"

void parallelWriteZarr(zarr &Zarr, void* zarrArr,
                          const std::vector<uint64_t> &startCoords,
                          const std::vector<uint64_t> &endCoords,
                          const std::vector<uint64_t> &writeShape,
                          const uint64_t bits, const bool useUuid, const bool crop);
#endif