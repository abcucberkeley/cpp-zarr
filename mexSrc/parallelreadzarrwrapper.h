#ifndef PARALLELWRITEZARRREAD_H
#define PARALLELWRITEZARRREAD_H
#include "zarr.h"
void parallelReadZarrMex(const zarr &Zarr, void* zarrArr,
                         const std::vector<uint64_t> &startCoords, 
                         const std::vector<uint64_t> &endCoords,
                         const std::vector<uint64_t> &readShape,
                         const uint64_t bits,
                         const bool useCtx=false);

void* parallelReadZarrWrapper(zarr Zarr, const bool &crop,
                              std::vector<uint64_t> startCoords, 
                              std::vector<uint64_t> endCoords);
#endif