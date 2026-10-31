// The chunks a write stores, for writes whose chunks are compressed elsewhere (GPU writes): the
// CPU writer stores them in their chunk files and shards as it stores its own. Internal to the
// library (not installed).
#ifndef CPPZARR_STOREDCHUNKS_H
#define CPPZARR_STOREDCHUNKS_H
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>
#include "zarr.h"

// What one chunk becomes when it is stored: nothing (a sharded array's inner chunk
// past the array's edge, an all-zero chunk of a sparse write, or an inner chunk a
// crop write leaves alone that its shard does not hold), or its bytes, with the
// CRC32C to put after them when the array has one
struct StoredChunk {
    bool empty = true;
    bool zero = false;          // all zero in a sparse write (any file it had is removed)
    const uint8_t* data = nullptr;
    uint64_t n = 0;
    bool hasCrc = false;
    uint32_t crc = 0;
    // (or, for a chunk file, its n bytes in pieces of memory, in order)
    const std::vector<std::pair<const uint8_t*, uint64_t>>* pieces = nullptr;
};

// Chunks prepared elsewhere: the chunks [first, last) of the array's chunk list (as
// set_chunkInfo made it for the write's region; whole shards of a sharded array) and, for each
// one the write touches, get(f, out, why), which fills out as the CPU writer would (its stored
// bytes, or zero) or returns false with the reason. Called from several threads at once.
struct PreparedChunks {
    int64_t first = 0, last = 0;
    std::function<bool(int64_t, StoredChunk&, std::string&)> get;
};

// parallelWriteZarr's write of the region [startCoords, endCoords), for the chunks
// [prepared.first, prepared.last) only, storing prepared's chunks instead of making them from data
uint8_t parallelWriteZarrPrepared(zarr &Zarr, const std::vector<uint64_t> &startCoords,
                                  const std::vector<uint64_t> &endCoords, const std::vector<uint64_t> &writeShape,
                                  const uint64_t bits, const bool useUuid, const bool crop, const bool sparse,
                                  const PreparedChunks &prepared);
#endif
