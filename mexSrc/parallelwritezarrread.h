#ifndef PARALLELWRITEZARRREAD_H
#define PARALLELWRITEZARRREAD_H
#include "zarr.h"
void* parallelReadZarrWrapper(zarr &Zarr, const bool &crop,
                              std::vector<uint64_t> startCoords, 
                              std::vector<uint64_t> endCoords);
#endif