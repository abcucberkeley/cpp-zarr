#include "gpuwritezarr.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <omp.h>
#ifndef _WIN32
#include <cerrno>
#include <climits>
#include <fcntl.h>
#include <sys/uio.h>
#include <unistd.h>
#endif
#include "blosc.h"
#include "gpu/cudaapi.h"
#include "gpu/gpucommon.h"
#include "helperfunctions.h"
#include "parallelreadzarr.h"
#include "parallelwork.h"
#include "pathutil.h"
#include "parallelwritezarr.h"
#include "storedchunks.h"

using namespace czcuda;

namespace {

uint32_t rd32(const uint8_t* p){ uint32_t v; memcpy(&v, p, 4); return v; }
void wr32(uint8_t* p, const uint64_t v){ const uint32_t w = (uint32_t)v; memcpy(p, &w, 4); }

// The biggest streams nvCOMP compresses, per codec
uint64_t maxStream(const int codec){
    return codec == Zstd || codec == LZ4 ? (uint64_t(1) << 24) : 0;
}

// blosc's header for a write's chunks: what blosc_compress_ctx writes for them (the block size,
// shuffle, whether blocks are split into their bytes' streams, the codec), found by compressing
// zeros as long as a chunk, with the chunk's size in it. Bigger chunks take 4 MiB, whose blocks
// blosc sizes the same (at most 1 MiB), or as long as the block size asked for (blocksize; 0 for
// blosc's own); a block bigger than nvCOMP compresses comes out at 32 MiB, still too big. Kept for
// later writes.
bool bloscHeader(const std::string &cname, const int clevel, const int shuffle, const uint64_t ts, const uint64_t sB,
                 const uint64_t blocksize, uint8_t header[16]){
    typedef std::tuple<std::string, int, int, uint64_t, uint64_t, uint64_t> Key;
    static std::mutex m;
    static std::map<Key, std::array<uint8_t, 16>> known;
    const Key key(cname, clevel, shuffle, ts, sB, blocksize);
    {
        std::lock_guard<std::mutex> g(m);
        auto it = known.find(key);
        if(it != known.end()){
            memcpy(header, it->second.data(), 16);
            return true;
        }
    }
    const uint64_t most = std::max<uint64_t>(uint64_t(4) << 20, std::min<uint64_t>(blocksize, 2*maxStream(Zstd)));
    const uint64_t probe = sB <= most ? sB : (most/ts)*ts;
    std::vector<uint8_t> zeros(probe, 0), dest(probe+BLOSC_MAX_OVERHEAD);
    const int n = blosc_compress_ctx(clevel, shuffle, (size_t)ts, (size_t)probe, zeros.data(), dest.data(), dest.size(),
                                     cname.c_str(), (size_t)blocksize, 1);
    if(n < BLOSC_MIN_HEADER_LENGTH) return false;
    memcpy(header, dest.data(), 16);
    wr32(header+4, sB);
    std::array<uint8_t, 16> h;
    memcpy(h.data(), header, 16);
    std::lock_guard<std::mutex> g(m);
    known[key] = h;
    return true;
}

// How a write's chunks are stored, and laid out in the GPU's buffer
struct Layout {
    int kind = 0;                   // 0: as they are (no compressor), 1: blosc, 2: one zstd frame (the zstd codec)
    int codec = -1;                 // the nvCOMP codec compressing them (-1: not compressed)
    bool shuffle = false, split = false;
    uint8_t header[16] = {0};       // blosc's chunk header (but each chunk's compressed size)
    uint64_t ts = 0, celems = 0, sB = 0;
    // a chunk's blocks (blosc's; one of the whole chunk otherwise) of be elements, bstride apart in
    // its buffer, and a split block's streams pstride apart
    uint64_t be = 0, nblocks = 1, bstride = 0, pstride = 0;
    uint64_t U = 0;                 // a chunk's bytes in the GPU's buffer (aligned)
    // a chunk's streams (every chunk has the same): their place in its buffer, bytes and block
    std::vector<uint64_t> sOff, sLen;
    std::vector<uint32_t> sBlock;
    uint64_t maxLen = 0;
};

// The layout of a write's chunks; false when the GPU does not compress them (why says what)
bool makeLayout(zarr &Z, const Api* a, const uint64_t ts, const uint64_t celems, const int bloscShuffle, const uint64_t align,
                Layout &L, std::string &why){
    L.ts = ts;
    L.celems = celems;
    L.sB = ts*celems;
    L.be = celems;
    L.nblocks = 1;
    auto alA = [&](const uint64_t n){ return (n+align-1)/align*align; };
    const std::string comp = Z.get_compressor();
    if(comp == "none"){
        L.U = alA(L.sB);
        return true;
    }
    if(comp == "zstd"){
        if(!a->compressAsync[Zstd]){
            why = "the nvCOMP loaded ("+a->nvcompPath+") does not compress zstd";
            return false;
        }
        if(L.sB > maxStream(Zstd)){
            why = "chunks of more than "+std::to_string(maxStream(Zstd) >> 20)+" MiB are more than nvCOMP compresses in one zstd frame";
            return false;
        }
        L.kind = 2;
        L.codec = Zstd;
        L.U = alA(L.sB);
        L.sOff.assign(1, 0);
        L.sLen.assign(1, L.sB);
        L.sBlock.assign(1, 0);
        L.maxLen = L.sB;
        return true;
    }
    if(comp != "blosc"){
        why = "the GPU does not compress "+comp;
        return false;
    }
    L.kind = 1;
    // (blosc's block size as the CPU writer takes it: a Zarr v3 array's, at most the chunk)
    const uint64_t asked = Z.get_zarr_format() == 3 ? std::min<uint64_t>(Z.get_blocksize(), L.sB) : 0;
    if(!bloscHeader(Z.get_cname(), (int)Z.get_clevel(), bloscShuffle, ts, L.sB, asked, L.header)){
        why = "blosc does not compress \""+Z.get_cname()+"\"";
        return false;
    }
    const uint8_t flags = L.header[2];
    const uint64_t blocksize = rd32(L.header+8);
    if(flags & 0x4){
        why = "the GPU does not bitshuffle";
        return false;
    }
    if(!blocksize || blocksize % ts || L.header[3] != ts){
        why = "blosc's blocks do not hold whole elements";
        return false;
    }
    // (blosc stores the chunks as they are: clevel 0, or chunks of under 128 bytes)
    if(flags & 0x2){
        L.U = alA(L.sB);
        return true;
    }
    const int format = flags >> 5;
    L.codec = format == 1 ? LZ4 : format == 4 ? Zstd : -1;
    if(L.codec < 0 || !a->compressAsync[L.codec]){
        why = "the GPU does not compress blosc's \""+Z.get_cname()+"\" streams";
        return false;
    }
    L.shuffle = (flags & 0x1) != 0;
    // (a block split into the streams of its ts bytes; one stream when ts is 1)
    L.split = !(flags & 0x10) && ts > 1;
    L.be = blocksize/ts;
    L.nblocks = (L.sB+blocksize-1)/blocksize;
    const uint64_t leftover = L.sB % blocksize;
    L.pstride = L.split ? alA(L.be) : 0;
    L.bstride = L.split ? ts*L.pstride : alA(blocksize);
    uint64_t end = 0;
    for(uint64_t b = 0; b < L.nblocks; b++){
        // (blosc never splits a last block shorter than the others)
        const uint64_t n = b+1 == L.nblocks && leftover ? leftover/ts : L.be;
        const uint64_t at = b*L.bstride;
        if(L.split && n == L.be){
            for(uint64_t j = 0; j < ts; j++){
                L.sOff.push_back(at+j*L.pstride);
                L.sLen.push_back(L.be);
                L.sBlock.push_back((uint32_t)b);
            }
            end = at+ts*L.pstride;
        }
        else{
            L.sOff.push_back(at);
            L.sLen.push_back(n*ts);
            L.sBlock.push_back((uint32_t)b);
            end = at+n*ts;
        }
    }
    L.U = alA(end);
    for(const uint64_t v : L.sLen) L.maxLen = std::max(L.maxLen, v);
    if(L.maxLen > maxStream(L.codec)){
        why = "blosc's blocks are bigger than nvCOMP compresses";
        return false;
    }
    return true;
}

// The index of a shard's current file, checked as parallelWriteZarr checks it (all entries empty
// for a shard with no file yet); false with the problem
bool readShardIndex(const std::string &path, const uint64_t perShard, const bool atStart, const bool checksum,
                    std::vector<uint64_t> &index, std::string &problem){
    const uint64_t emptyEntry = ~uint64_t(0);
    const uint64_t entryBytes = 2*perShard*sizeof(uint64_t), indexBytes = entryBytes+(checksum ? 4 : 0);
    index.assign(2*perShard, emptyEntry);
    std::ifstream f;
    openPath(f, path, std::ios::binary);
    if(!f.is_open()) return true;
    f.seekg(0, std::ios::end);
    const uint64_t len = (uint64_t)f.tellg();
    if(len < indexBytes){
        problem = "is smaller than its index";
        return false;
    }
    f.seekg(atStart ? 0 : (std::streamoff)(len-indexBytes), std::ios::beg);
    f.read(reinterpret_cast<char*>(index.data()), entryBytes);
    uint32_t crc = 0;
    if(checksum) f.read(reinterpret_cast<char*>(&crc), sizeof(crc));
    if(!f){
        problem = "could not be read";
        return false;
    }
    if(checksum && crc != crc32c(reinterpret_cast<const uint8_t*>(index.data()), entryBytes)){
        problem = "has an index that does not match its checksum";
        return false;
    }
    const uint64_t lo = atStart ? indexBytes : 0, hi = atStart ? len : len-indexBytes;
    for(uint64_t i = 0; i < perShard; i++){
        const uint64_t off = index[2*i], nb = index[2*i+1];
        if(off == emptyEntry && nb == emptyEntry) continue;
        if(off < lo || off > hi || nb > hi-off){
            problem = "has an index entry outside the file";
            return false;
        }
    }
    return true;
}

// A part of a write: the chunks [f0, f1) of the chunk list (whole shards of a sharded array), the
// ones it touches, and what the GPU made of them
struct Part {
    int64_t f0 = 0, f1 = 0;
    std::vector<int64_t> chunks;         // the touched chunks
    std::vector<int64_t> local;          // (each chunk f's place in chunks, at f-f0; -1 if untouched)
    std::vector<int64_t> cbase;          // (a chunk's current contents in the slot's input, or -1)
    WriteSlot* slot = nullptr;
    // its GPU work's ends (the compressed sizes back, the stored chunks back), and for
    // CPPZARR_GPU_PROFILE its start and the ends of putting the chunks together, compressing and
    // putting the stored chunks together
    CUevent ev[6] = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    uint64_t nst = 0;                    // streams compressed
    uint64_t resAt[3] = {0, 0, 0};       // (in the slot's work buffer: sizes, statuses, nonzero flags)
    uint64_t baseAt = 0;                 // (in the slot's input: the chunks' current contents)
    std::vector<uint8_t> zero;           // all zero, left out (sparse writes)
    std::vector<uint64_t> at, size;      // each chunk's place and bytes in the slot's stored chunks
    uint64_t stored = 0;
};

// A shard a GPU write stores whole (not a crop write, which keeps the inner chunks it does not
// touch: those go through parallelWriteZarr): its inner chunks appended to its file part by part,
// in order, each part's run of them at once, and its index written after its last one, as
// parallelWriteZarr writes a shard (the index at its start or end, with its CRC32C; a temporary file
// moved into place with useUuid)
struct ShardFile {
    std::string path, writePath;
    bool opened = false, done = false;
    uint64_t bytes = 0;
    std::vector<uint64_t> index;
    int64_t left = 0;              // its touched inner chunks not yet appended
#ifdef _WIN32
    std::ofstream f;
#else
    int fd = -1;
#endif
    // Opens its file (making its folder first if it is missing), with room for an index at the start
    bool open(const uint64_t room){
#ifdef _WIN32
        openPath(f, writePath, std::ios::binary | std::ios::trunc);
        if(!f.is_open()){
            const size_t slash = writePath.find_last_of("/\\");
            if(slash != std::string::npos) mkdirRecursive(writePath.substr(0, slash).c_str());
            f.clear();
            openPath(f, writePath, std::ios::binary | std::ios::trunc);
        }
        if(!f.is_open()) return false;
#else
        fd = ::open(writePath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
        if(fd < 0 && errno == ENOENT){
            const size_t slash = writePath.find_last_of('/');
            if(slash != std::string::npos){
                mkdirRecursive(writePath.substr(0, slash).c_str());
                fd = ::open(writePath.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0666);
            }
        }
        if(fd < 0) return false;
#endif
        opened = true;
        bytes = room;
        if(room){
            const std::vector<uint8_t> zeros(room, 0);
            return append({{zeros.data(), room}});
        }
        return true;
    }
    // Appends pieces of memory, in order (writev: no copy of them is made)
    bool append(const std::vector<std::pair<const uint8_t*, uint64_t>> &pieces){
#ifdef _WIN32
        for(const auto &q : pieces) f.write(reinterpret_cast<const char*>(q.first), (std::streamsize)q.second);
        return (bool)f;
#else
        std::vector<iovec> iov(pieces.size());
        for(size_t i = 0; i < pieces.size(); i++){
            iov[i].iov_base = const_cast<uint8_t*>(pieces[i].first);
            iov[i].iov_len = (size_t)pieces[i].second;
        }
        size_t i = 0;
        while(i < iov.size()){
            if(!iov[i].iov_len){
                i++;
                continue;
            }
            const ssize_t w = writev(fd, iov.data()+i, (int)std::min<size_t>(iov.size()-i, IOV_MAX));
            if(w < 0){
                if(errno == EINTR) continue;
                return false;
            }
            // (past what was written: whole pieces, and the start of the next)
            for(size_t left = (size_t)w; left && i < iov.size(); ){
                const size_t m = std::min(left, iov[i].iov_len);
                iov[i].iov_base = static_cast<uint8_t*>(iov[i].iov_base)+m;
                iov[i].iov_len -= m;
                left -= m;
                if(!iov[i].iov_len) i++;
            }
        }
        return true;
#endif
    }
    // Writes its index (and its CRC32C) after its inner chunks or in the room at its start, closes it
    bool finish(const bool atStart, const bool checksum){
        const uint32_t crc = crc32c(reinterpret_cast<const uint8_t*>(index.data()), index.size()*sizeof(uint64_t));
        std::vector<std::pair<const uint8_t*, uint64_t>> tail{{reinterpret_cast<const uint8_t*>(index.data()), index.size()*sizeof(uint64_t)}};
        if(checksum) tail.push_back({reinterpret_cast<const uint8_t*>(&crc), sizeof(crc)});
        bool ok = true;
#ifdef _WIN32
        if(atStart) f.seekp(0, std::ios::beg);
        ok = append(tail);
        f.close();
        ok = ok && (bool)f;
#else
        if(atStart){
            uint64_t at = 0;
            for(const auto &q : tail){
                for(uint64_t done = 0; done < q.second && ok; ){
                    const ssize_t w = pwrite(fd, q.first+done, (size_t)(q.second-done), (off_t)(at+done));
                    if(w < 0 && errno == EINTR) continue;
                    if(w <= 0) ok = false;
                    else done += (uint64_t)w;
                }
                at += q.second;
            }
        }
        else ok = append(tail);
        if(::close(fd) != 0) ok = false;
        fd = -1;
#endif
        opened = false;
        return ok;
    }
    // (after an error: closed, and a temporary file removed)
    void abandon(const bool tmp){
#ifdef _WIN32
        if(opened) f.close();
#else
        if(fd >= 0) ::close(fd);
        fd = -1;
#endif
        if(opened && tmp) removePath(writePath);
        opened = false;
    }
};

// What CPPZARR_GPU_PROFILE prints about a write
struct WriteStats {
    uint64_t chunks = 0, inBytes = 0, outBytes = 0, again = 0;
    size_t parts = 0;
    double plan = 0, prep = 0, layout = 0, files = 0, wait = 0;
    float gpu[4] = {0, 0, 0, 0};   // (put together, compress, stored chunks put together, copied to the host)
};

}  // namespace

bool gpuWriteAvailable(std::string &why){
    return api(&why) != nullptr;
}

uint8_t gpuWriteZarr(zarr &Zarr, const void* data, const std::vector<uint64_t> &startCoords,
                     const std::vector<uint64_t> &endCoords, const std::vector<uint64_t> &writeShape,
                     const std::vector<uint64_t> &inStrides, const uint64_t bits, const bool useUuid,
                     const bool crop, const bool sparse, void* stream){
    // (in a child made by fork(), on the thread that forked: see ForkSafety)
    if(ForkSafety::get().needRegionThread()){
        uint8_t r = 1;
        ForkSafety::get().onRegionThread([&]{
            r = gpuWriteZarr(Zarr, data, startCoords, endCoords, writeShape, inStrides, bits, useUuid, crop, sparse, stream);
        });
        return r;
    }
    std::string why;
    const Api* a = api(&why);
    if(!a){
        Zarr.set_errString("GPU writes are not available: "+why+"\n");
        return 1;
    }
    const uint64_t nDims = Zarr.get_ndims();
    if(writeShape.size() != nDims || inStrides.size() != nDims || startCoords.size() != nDims || endCoords.size() != nDims){
        Zarr.set_errString("The input has "+std::to_string(writeShape.size())+" dimensions and "+
                           std::to_string(inStrides.size())+" strides but the array has "+std::to_string(nDims)+" dimensions\n");
        return 1;
    }
    if(nDims == 0 && Zarr.get_shard()){
        Zarr.set_errString("Sharding is not supported for 0-dimensional arrays\n");
        return 1;
    }
    const std::string compressor = Zarr.get_compressor();
    if(compressor != "none" && compressor != "blosc" && compressor != "gzip" && compressor != "zstd"){
        Zarr.set_errString("Writing arrays with the \""+Zarr.get_cname()+"\" compressor is not supported\n");
        return 1;
    }
    if(oppositeEndianness(Zarr.get_dtype())){
        Zarr.set_errString("Writing arrays of data type \""+Zarr.get_dtype()+"\" (opposite byte order) is not supported\n");
        return 1;
    }
    const uint64_t ts = bits/8;
    if(ts < 1 || ts > 16 || dtypeInfo(Zarr.get_dtype()).bytes != ts){
        Zarr.set_errString("The data's elements are "+std::to_string(ts)+" bytes, not the array's \""+Zarr.get_dtype()+"\" elements\n");
        return 1;
    }
    for(uint64_t d = 0; d < nDims; d++){
        if(endCoords[d] > Zarr.get_shape(d) || startCoords[d] >= endCoords[d] || endCoords[d]-startCoords[d] != writeShape[d]){
            Zarr.set_errString("The region to write is outside the array, or not the data's shape, on axis "+std::to_string(d)+"\n");
            return 1;
        }
    }
    static const bool prof = std::getenv("CPPZARR_GPU_PROFILE") != nullptr;
    const auto t0 = std::chrono::steady_clock::now();
    auto now = [&](){ return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t0).count(); };
    WriteStats stats;

    const CUstream s = (CUstream)stream;
    bool pushed = false;
    Ctx* cx = nullptr;
    std::vector<WriteSlot*> slots;
    std::deque<Part> parts;
    // At the end, and after an error: the GPU work queued so far ends before the slots are given
    // back, and the context is popped
    auto release = [&](){
        if(pushed) a->cuStreamSynchronize(s);
        for(auto &p : parts) for(CUevent &e : p.ev) if(e){ a->cuEventDestroy(e); e = nullptr; }
        if(cx){
            std::lock_guard<std::mutex> g(cx->m);
            for(WriteSlot* sl : slots) sl->busy = false;
        }
        slots.clear();
        if(pushed){ CUcontext c; a->cuCtxPopCurrent(&c); }
        pushed = false;
    };
    try{
        if(!isGpuMemory(a, data)) throw Fail("The data to write to "+Zarr.get_fileName()+" is not GPU memory");
        ck(a, a->cuCtxPushCurrent(memoryContext(a, data)), "cuCtxPushCurrent");
        pushed = true;
        CUcontext ctx = nullptr;
        ck(a, a->cuCtxGetCurrent(&ctx), "cuCtxGetCurrent");
        cx = &context(a, ctx);
        slots.push_back(&takeWriteSlot(*cx));
        slots.push_back(&takeWriteSlot(*cx));

        // ---- the chunks: the (inner) chunk shape, its storage order, the fill value
        const bool sharded = Zarr.get_shard();
        std::vector<uint64_t> chunkDims(nDims), cps(nDims, 1);
        uint64_t celems = 1;
        for(uint64_t d = 0; d < nDims; d++){
            chunkDims[d] = sharded ? Zarr.get_chunk_shape(d) : Zarr.get_chunks(d);
            if(sharded) cps[d] = (Zarr.get_chunks(d)+chunkDims[d]-1)/chunkDims[d];
            celems *= chunkDims[d];
        }
        const uint64_t sB = celems*ts;
        if(compressor == "blosc" && sB > (uint64_t)BLOSC_MAX_BUFFERSIZE)
            throw Fail("Chunk is too large for the \""+Zarr.get_cname()+"\" compressor: "+std::to_string(sB)+" bytes exceeds the "
                       "blosc limit of "+std::to_string((uint64_t)BLOSC_MAX_BUFFERSIZE)+" bytes (~2 GB). Reduce the chunk size, or use "
                       "the gzip compressor.");
        const std::vector<uint64_t> axes = Zarr.get_chunkAxisOrder();
        uint8_t fillElem[16] = {0};
        fillValueElement(Zarr.get_fill_value(), Zarr.get_dtype(), fillElem);
        uint64_t fillLo = 0, fillHi = 0;
        {
            uint8_t f[16] = {0};
            memcpy(f, fillElem, ts);
            memcpy(&fillLo, f, 8);
            memcpy(&fillHi, f+8, 8);
        }
        // (sparse writes leave out all-zero chunks, when that is how a missing chunk reads back)
        const bool zeroRule = sparse && fillValueIsZero(Zarr.get_fill_value(), Zarr.get_dtype());
        const bool chunkChecksum = Zarr.get_chunkChecksum();
        const uint64_t shuffle = Zarr.get_shuffle();
        const int bloscShuffle = Zarr.get_zarr_format() != 3 || shuffle > 2 ? BLOSC_SHUFFLE : (int)shuffle;

        // ---- how the GPU stores them (or, for codecs it does not compress, that a copy of the
        // region in host memory is written by parallelWriteZarr)
        uint64_t align = 16;
        Layout L;
        std::string notOnGpu;
        bool onGpu = makeLayout(Zarr, a, ts, celems, bloscShuffle, align, L, notOnGpu);
        NvcompOpts opts{};
        if(onGpu && L.codec >= 0){
            NvcompAlignments al{1, 1, 1};
            if(!a->compAlignments[L.codec](opts, &al)){
                const uint64_t need = std::max<uint64_t>({16, al.input, al.output, al.temp});
                if(need > align){
                    align = need;
                    L = Layout();
                    onGpu = makeLayout(Zarr, a, ts, celems, bloscShuffle, align, L, notOnGpu);
                }
            }
        }
        // (CPPZARR_GPU_WRITE_HOST=1: always through host memory, for testing)
        if(const char* e = std::getenv("CPPZARR_GPU_WRITE_HOST")) if(*e && std::atoi(e)){
            onGpu = false;
            notOnGpu = "CPPZARR_GPU_WRITE_HOST is set";
        }
        uint64_t outSlot = 0;
        if(onGpu && L.codec >= 0){
            size_t maxOut = 0;
            const int mr = a->compMaxOutput[L.codec](L.maxLen, opts, &maxOut);
            if(mr) throw Fail("nvCOMP could not size its compressed streams (status "+std::to_string(mr)+")");
            outSlot = (maxOut+align-1)/align*align;
        }
        // the region's geometry, the same for every part (before each part's tables in its input):
        // the input's element strides, the region's start and shape, the array's shape, the
        // chunk's dimensions in storage order, slowest first, and their array axes
        const uint64_t nd1 = std::max<uint64_t>(nDims, 1);
        std::vector<int64_t> geo(5*nd1, 0);
        std::vector<int32_t> saxis(nd1, 0);
        for(uint64_t d = 0; d < nDims; d++){
            geo[d] = (int64_t)inStrides[d];
            geo[nd1+d] = (int64_t)startCoords[d];
            geo[2*nd1+d] = (int64_t)writeShape[d];
            geo[3*nd1+d] = (int64_t)Zarr.get_shape(d);
            geo[4*nd1+d] = (int64_t)chunkDims[axes[d]];
            saxis[d] = (int32_t)axes[d];
        }
        const uint64_t saxisAt = al16(8*geo.size()), geoBytes = saxisAt+al16(4*saxis.size());

        // Puts chunks together (cz_gather): nch chunks of chunkElems elements, their origins, places
        // in out and current contents in base at corg, cout and cbase, the geometry at geoAt
        auto gather = [&](const CUdeviceptr geoAt, const CUdeviceptr corg, const CUdeviceptr cout, const CUdeviceptr cbase,
                          const CUdeviceptr base, const CUdeviceptr out, const CUdeviceptr nonzero, const uint64_t nch,
                          const uint64_t chunkElems, const uint64_t be, const uint64_t nblocks, const uint64_t bstride,
                          const uint64_t pstride, int shuf, int split, int fillpast){
            if(!nch) return;
            CUdeviceptr pin = (CUdeviceptr)(uintptr_t)data, pistride = geoAt, pwstart = geoAt+8*nd1, pwshape = geoAt+16*nd1;
            CUdeviceptr pashape = geoAt+24*nd1, psdims = geoAt+32*nd1, psaxis = geoAt+saxisAt;
            CUdeviceptr pcorg = corg, pcout = cout, pcbase = cbase, pbase = base, pout = out, pnz = nonzero;
            unsigned long long ce = chunkElems, bel = be, nb = nblocks, bs = bstride, ps = pstride, lo = fillLo, hi = fillHi;
            // (about 8K elements to a thread block)
            int pieces = (int)std::min<uint64_t>({std::max<uint64_t>(1, (chunkElems+8191)/8192), 4096,
                                                  std::max<uint64_t>(1, 0x7fffffffULL/nch)});
            int ndv = (int)nDims, tsv = (int)ts;
            void* args[] = {&pin, &pistride, &pwstart, &pwshape, &pashape, &psdims, &psaxis, &pcorg, &pcout, &pcbase, &pbase, &pout,
                            &pnz, &ce, &bel, &nb, &bs, &ps, &pieces, &ndv, &tsv, &shuf, &split, &fillpast, &lo, &hi};
            const bool small = chunkElems+1024 < 0xffffffffULL;
            ck(a, a->cuLaunchKernel(small ? cx->gather32 : cx->gather64, (unsigned)(nch*(uint64_t)pieces), 1, 1, 256, 1, 1, 0, s,
                                    args, nullptr), "cuLaunchKernel (gather)");
        };

        if(!onGpu){
            // ---- the region copied to host memory in F order, in slabs along its last axis (each
            // put together on the GPU), then written as parallelWriteZarr writes it
            uint64_t elems = 1;
            for(uint64_t d = 0; d < nDims; d++) elems *= writeShape[d];
            const uint64_t depth = nDims ? writeShape[nDims-1] : 1, plane = elems/depth;
            std::unique_ptr<uint8_t[]> host(new (std::nothrow) uint8_t[std::max<uint64_t>(elems*ts, 1)]);
            if(!host) throw Fail("Not enough host memory to write "+Zarr.get_fileName()+" from the GPU ("+std::to_string(elems*ts >> 20)+
                                 " MiB, through host memory: "+notOnGpu+")");
            const uint64_t per = std::max<uint64_t>(1, std::min<uint64_t>(depth, (uint64_t(256) << 20)/std::max<uint64_t>(plane*ts, 1)));
            WriteSlot &sl = *slots[0];
            // (a slab is one chunk of the region's shape, k deep, with F order's storage order)
            std::vector<int64_t> g(geo);
            std::vector<int32_t> ax(saxis);
            for(uint64_t k = 0; k < nDims; k++){
                ax[k] = (int32_t)(nDims-1-k);
                g[4*nd1+k] = (int64_t)writeShape[nDims-1-k];
            }
            const uint64_t corgAt = geoBytes, coutAt = corgAt+al16(8*nd1), cbaseAt = coutAt+16, tabBytes = cbaseAt+16;
            growPinned(a, sl.hin, sl.hinBytes, tabBytes, 0);
            growDevice(a, sl.din, sl.dinBytes, tabBytes);
            growDevice(a, sl.unc, sl.uncBytes, per*plane*ts);
            growDevice(a, sl.work, sl.workBytes, 16);
            for(uint64_t z0 = 0; z0 < depth; z0 += per){
                const uint64_t k = std::min(per, depth-z0);
                if(nDims) g[4*nd1] = (int64_t)k;
                std::vector<int64_t> corg(nd1, 0);
                for(uint64_t d = 0; d < nDims; d++) corg[d] = (int64_t)startCoords[d];
                if(nDims) corg[nDims-1] += (int64_t)z0;
                const uint64_t zero = 0;
                const int64_t none = -1;
                memcpy(sl.hin, g.data(), 8*g.size());
                memcpy(sl.hin+saxisAt, ax.data(), 4*ax.size());
                memcpy(sl.hin+corgAt, corg.data(), 8*corg.size());
                memcpy(sl.hin+coutAt, &zero, 8);
                memcpy(sl.hin+cbaseAt, &none, 8);
                ck(a, a->cuMemcpyHtoDAsync(sl.din, sl.hin, tabBytes, s), "cuMemcpyHtoDAsync");
                gather(sl.din, sl.din+corgAt, sl.din+coutAt, sl.din+cbaseAt, sl.din, sl.unc, sl.work, 1, plane*k, plane*k, 1, 0, 0, 0, 0, 0);
                ck(a, a->cuMemcpyDtoHAsync(host.get()+z0*plane*ts, sl.unc, plane*k*ts, s), "cuMemcpyDtoHAsync");
                ck(a, a->cuStreamSynchronize(s), "cuStreamSynchronize");
            }
            release();
            std::vector<uint64_t> fStrides(nDims);
            uint64_t acc = 1;
            for(uint64_t d = 0; d < nDims; d++){ fStrides[d] = acc; acc *= writeShape[d]; }
            const double h1 = now();
            const uint8_t r = parallelWriteZarr(Zarr, host.get(), startCoords, endCoords, writeShape, fStrides, bits, useUuid, crop, sparse);
            if(prof) fprintf(stderr, "cppzarr GPU write: %s through host memory (%s): copied in %.1f ms, written in %.1f ms\n",
                             Zarr.get_fileName().c_str(), notOnGpu.c_str(), h1, now()-h1);
            return r;
        }

        // ---- the chunk list's chunks: their grid coordinates (as set_chunkInfo lists them: chunks,
        // or a sharded array's inner chunks shard by shard, last axis fastest), and the ones the
        // write touches (in a sharded array, as parallelWriteZarr finds them: inner chunks inside
        // the array that the region cuts)
        const int64_t numChunks = (int64_t)Zarr.get_numChunks();
        const int64_t perShard = sharded ? (int64_t)Zarr.get_numChunksPerShard() : 1;
        std::vector<uint64_t> grid((size_t)numChunks*nd1, 0);
        std::vector<uint8_t> touched((size_t)numChunks, 1);
        const int team = std::max(1, omp_get_max_threads());
        {
            std::vector<uint64_t> first(nDims), count(nDims);
            uint64_t nOuter = 1;
            for(uint64_t d = 0; d < nDims; d++){
                const uint64_t extent = cps[d]*chunkDims[d];
                first[d] = startCoords[d]/extent;
                const uint64_t last = endCoords[d]/extent+(endCoords[d]%extent ? 1 : 0);
                count[d] = last > first[d] ? last-first[d] : 0;
                nOuter *= count[d];
            }
            const bool regular = nOuter*(uint64_t)perShard == (uint64_t)numChunks;
            std::atomic<int> bad{0};
            // (in a region of all of the threads: see parallelWriteZarr)
            #pragma omp parallel num_threads(team)
            {
                std::vector<uint64_t> idx(nDims);
                const int w = omp_get_thread_num(), nt = omp_get_num_threads();
                for(int64_t f = w; f < numChunks && !bad; f += nt){
                    try{
                        if(regular){
                            uint64_t o = (uint64_t)f/(uint64_t)perShard, c = (uint64_t)f%(uint64_t)perShard;
                            for(int64_t d = (int64_t)nDims-1; d >= 0; d--){
                                idx[d] = (first[d]+o%count[d])*cps[d]+c%cps[d];
                                o /= count[d];
                                c /= cps[d];
                            }
                        }
                        else idx = Zarr.get_chunkAxisVals(Zarr.get_chunkNames(f));
                        uint8_t kind = 1;
                        for(uint64_t d = 0; d < nDims; d++){
                            grid[(size_t)f*nd1+d] = idx[d];
                            const uint64_t lo = idx[d]*chunkDims[d];
                            if(!sharded || kind == 2) continue;
                            if(lo >= Zarr.get_shape(d)) kind = 2;
                            else if(lo >= endCoords[d] || lo+chunkDims[d] <= startCoords[d]) kind = 0;
                        }
                        touched[(size_t)f] = kind;
                    }
                    catch(...){ bad = 1; }
                }
            }
            if(bad) throw Fail("The chunks of "+Zarr.get_fileName()+" could not be listed");
        }
        stats.plan = now();

        // ---- the parts: runs of chunks of at most CPPZARR_GPU_WRITE_PART_MB (256) MB put together,
        // as even as they go, so that the files of one are written while the GPU works on the next
        // (and the GPU's buffers stay small however much a write writes); no more than that: nvCOMP
        // compresses small batches much slower (128 MiB in parts of 32 MB took 2.7 times as long on a
        // B200 as in one). A shard's inner chunks can span parts (but in crop writes, whose shards
        // parallelWriteZarr stores, keeping the inner chunks they do not touch).
        double partMaxMB = 256;
        if(const char* e = std::getenv("CPPZARR_GPU_WRITE_PART_MB")) partMaxMB = std::max(1.0, std::atof(e));
        uint64_t nTouched = 0;
        for(const uint8_t t : touched) nTouched += t == 1;
        const uint64_t maxBytes = std::max<uint64_t>(1, (uint64_t)(partMaxMB*1e6));
        const uint64_t nParts = std::max<uint64_t>(1, (nTouched*L.U+maxBytes-1)/maxBytes);
        const uint64_t partBytes = std::max<uint64_t>(1, (nTouched*L.U+nParts-1)/nParts);
        const bool wholeShards = sharded && crop;
        {
            int64_t f0 = 0;
            uint64_t acc = 0;
            for(int64_t f = 0; f < numChunks; f++){
                acc += touched[(size_t)f] == 1 ? L.U : 0;
                if(acc >= partBytes && (!wholeShards || (f+1) % perShard == 0)){
                    parts.emplace_back();
                    parts.back().f0 = f0;
                    parts.back().f1 = f+1;
                    f0 = f+1;
                    acc = 0;
                }
            }
            if(f0 < numChunks){
                parts.emplace_back();
                parts.back().f0 = f0;
                parts.back().f1 = numChunks;
            }
        }
        stats.parts = parts.size();
        const std::string fileName = Zarr.get_fileName();
        const bool indexAtStart = Zarr.get_shardIndexAtStart(), indexChecksum = Zarr.get_shardIndexChecksum();
        const uint64_t emptyEntry = ~uint64_t(0);
        const uint64_t per = L.codec >= 0 ? L.sOff.size() : 0;

        // ---- A: a part's chunks put together on the GPU (with the current contents of the chunks a
        // crop write cuts, read and decoded here) and compressed; their compressed sizes, the
        // compression's statuses and whether each chunk has a byte that is not zero then copied back
        auto stageA = [&](Part &p, WriteSlot &sl){
            const double h0 = now();
            p.slot = &sl;
            for(int64_t f = p.f0; f < p.f1; f++) if(touched[(size_t)f] == 1) p.chunks.push_back(f);
            p.local.assign((size_t)(p.f1-p.f0), -1);
            for(size_t i = 0; i < p.chunks.size(); i++) p.local[(size_t)(p.chunks[i]-p.f0)] = (int64_t)i;
            const uint64_t nch = p.chunks.size(), nst = nch*per;
            p.nst = nst;
            stats.chunks += nch;
            stats.inBytes += nch*sB;
            // (in a crop write, the chunks the region cuts, whose other elements keep their values)
            p.cbase.assign(nch, -1);
            std::vector<size_t> partial;
            if(crop){
                for(size_t i = 0; i < nch; i++){
                    const uint64_t* idx = &grid[(size_t)p.chunks[i]*nd1];
                    bool cut = false;
                    for(uint64_t d = 0; d < nDims; d++){
                        cut = cut || idx[d]*chunkDims[d] < startCoords[d] ||
                              ((idx[d]+1)*chunkDims[d] > endCoords[d] && endCoords[d] < Zarr.get_shape(d));
                    }
                    if(cut) partial.push_back(i);
                }
            }
            // the input: the geometry; the chunks' origins, places in the buffer and current
            // contents; the streams' places in the buffer, places for their output and bytes; the
            // current contents
            const uint64_t corgAt = geoBytes, coutAt = corgAt+al16(8*nch*nd1), cbaseAt = coutAt+al16(8*nch);
            const uint64_t inoffAt = cbaseAt+al16(8*nch), outoffAt = inoffAt+al16(8*nst), inlenAt = outoffAt+al16(8*nst);
            p.baseAt = inlenAt+al16(8*nst);
            const uint64_t inBytes = p.baseAt+partial.size()*al16(sB);
            growPinned(a, sl.hin, sl.hinBytes, inBytes, 0);
            uint8_t* h = sl.hin;
            memcpy(h, geo.data(), 8*geo.size());
            memcpy(h+saxisAt, saxis.data(), 4*saxis.size());
            int64_t* corg = reinterpret_cast<int64_t*>(h+corgAt);
            uint64_t* cout = reinterpret_cast<uint64_t*>(h+coutAt);
            for(size_t i = 0; i < nch; i++){
                const uint64_t* idx = &grid[(size_t)p.chunks[i]*nd1];
                for(uint64_t d = 0; d < nDims; d++) corg[i*nDims+d] = (int64_t)(idx[d]*chunkDims[d]);
                cout[i] = i*L.U;
            }
            if(!partial.empty()){
                // (each one's current contents decoded, as parallelWriteZarr decodes them; none, and
                // the fill value, for one that is not stored; a sharded array's from its shard's index)
                std::vector<std::string> shardOf(partial.size());
                std::map<std::string, std::vector<uint64_t>> indexes;
                if(sharded){
                    for(size_t k = 0; k < partial.size(); k++){
                        const std::string &name = Zarr.get_chunkNames(p.chunks[partial[k]]);
                        const std::vector<uint64_t> sAV = Zarr.chunkToShard(Zarr.get_chunkAxisVals(name));
                        shardOf[k] = fileName+"/"+Zarr.get_subfoldersString(sAV)+"/"+Zarr.chunkKey(Zarr.chunkNameToShardName(name));
                        if(indexes.count(shardOf[k])) continue;
                        std::string problem;
                        if(!readShardIndex(shardOf[k], (uint64_t)perShard, indexAtStart, indexChecksum, indexes[shardOf[k]], problem))
                            throw Fail("The shard "+shardOf[k]+" "+problem+" (it may be damaged, or not match the array's metadata)");
                    }
                }
                std::vector<std::string> errs(partial.size());
                std::vector<uint8_t> present(partial.size(), 0);
                #pragma omp parallel num_threads(team)
                {
                    std::vector<uint8_t> stored;
                    const int w = omp_get_thread_num(), nt = omp_get_num_threads();
                    for(size_t k = (size_t)w; k < partial.size(); k += (size_t)nt){
                        const int64_t f = p.chunks[partial[k]];
                        const std::string &name = Zarr.get_chunkNames(f);
                        uint8_t* dst = h+p.baseAt+k*al16(sB);
                        std::string why2;
                        try{
                            std::string path = shardOf[k];
                            uint64_t off = 0, len = 0;
                            if(sharded){
                                const std::vector<uint64_t> &index = indexes.at(shardOf[k]);
                                const int64_t pos = f%perShard;
                                if(index[2*pos] == emptyEntry && index[2*pos+1] == emptyEntry) continue;
                                off = index[2*pos];
                                len = index[2*pos+1];
                            }
                            else path = fileName+"/"+Zarr.get_subfoldersString(Zarr.get_chunkAxisVals(name))+"/"+Zarr.chunkKey(name);
                            std::ifstream file;
                            openPath(file, path, std::ios::binary);
                            if(!file.is_open()){
                                if(sharded) errs[k] = "Could not read the shard "+path;
                                continue;
                            }
                            if(!sharded){
                                file.seekg(0, std::ios::end);
                                len = (uint64_t)file.tellg();
                            }
                            stored.resize((size_t)len);
                            file.seekg((std::streamoff)off, std::ios::beg);
                            file.read(reinterpret_cast<char*>(stored.data()), (std::streamsize)len);
                            if(!file){
                                errs[k] = sharded ? "Could not read the shard "+path : "Error in Writer Read. Chunk: "+name;
                                continue;
                            }
                            if(decodeChunk(compressor, chunkChecksum, stored.data(), stored.size(), dst, sB, why2)){
                                errs[k] = sharded ? "An inner chunk of the shard "+path+" "+why2 : "Error in Writer Read. Chunk: "+name+" ("+why2+")";
                                continue;
                            }
                            present[k] = 1;
                        }
                        catch(const std::exception &e){ errs[k] = std::string("Write error: ")+e.what(); }
                    }
                }
                for(const auto &e : errs) if(!e.empty()) throw Fail(e);
                for(size_t k = 0; k < partial.size(); k++) if(present[k]) p.cbase[partial[k]] = (int64_t)(k*al16(sB));
            }
            memcpy(h+cbaseAt, p.cbase.data(), 8*nch);
            if(nst){
                uint64_t* inoff = reinterpret_cast<uint64_t*>(h+inoffAt);
                uint64_t* outoff = reinterpret_cast<uint64_t*>(h+outoffAt);
                uint64_t* inlen = reinterpret_cast<uint64_t*>(h+inlenAt);
                for(size_t i = 0; i < nch; i++){
                    for(uint64_t t = 0; t < per; t++){
                        inoff[i*per+t] = i*L.U+L.sOff[t];
                        outoff[i*per+t] = (i*per+t)*outSlot;
                        inlen[i*per+t] = L.sLen[t];
                    }
                }
            }
            // the GPU's buffers (a write fails with what it needs where the GPU lacks the memory);
            // in work: the streams' input and output pointers, then their compressed sizes and
            // statuses and the chunks' nonzero flags, copied back together
            p.resAt[0] = 2*al16(8*nst);
            p.resAt[1] = p.resAt[0]+al16(8*nst);
            p.resAt[2] = p.resAt[1]+al16(4*nst);
            const uint64_t workEnd = p.resAt[2]+al16(4*nch);
            // (and the stored chunks' buffers, here and in host memory, for as much as they can come to,
            // so that they are not made bigger part by part: a bigger buffer of pinned host memory took
            // longer than the part's GPU work)
            const uint64_t storedMax = std::max<uint64_t>(16, nch*al16(std::max<uint64_t>(sB+16, L.kind == 2 ? outSlot : 0)));
            {
                size_t freeB = 0, totalB = 0;
                a->cuMemGetInfo(&freeB, &totalB);
                const uint64_t need = (inBytes > sl.dinBytes ? inBytes : 0)+(nch*L.U > sl.uncBytes ? nch*L.U : 0)+
                                      (nst*outSlot > sl.compBytes ? nst*outSlot : 0)+(storedMax > sl.packedBytes ? storedMax : 0);
                if(freeB && need > freeB)
                    throw Fail("Not enough GPU memory for this write ("+std::to_string(need >> 20)+" MiB more needed, "+
                               std::to_string(freeB >> 20)+" MiB free): set CPPZARR_GPU_WRITE_PART_MB lower, or use smaller shards");
            }
            growDevice(a, sl.packed, sl.packedBytes, storedMax);
            growPinned(a, sl.hout, sl.houtBytes, storedMax, 0);
            growDevice(a, sl.din, sl.dinBytes, inBytes);
            growDevice(a, sl.unc, sl.uncBytes, std::max<uint64_t>(nch*L.U, 16));
            growDevice(a, sl.work, sl.workBytes, workEnd);
            if(nst) growDevice(a, sl.comp, sl.compBytes, nst*outSlot);
            growPinned(a, sl.hres, sl.hresBytes, workEnd-p.resAt[0], 0);
            if(prof){
                for(int k = 0; k < 6; k++) if(!p.ev[k]) ck(a, a->cuEventCreate(&p.ev[k], k < 2 ? 0x1 /* CU_EVENT_BLOCKING_SYNC */ : 0), "cuEventCreate");
                a->cuEventRecord(p.ev[2], s);
            }
            ck(a, a->cuMemcpyHtoDAsync(sl.din, sl.hin, inBytes, s), "cuMemcpyHtoDAsync");
            ck(a, a->cuMemsetD8Async(sl.work+p.resAt[2], 0, 4*nch, s), "cuMemsetD8Async");
            gather(sl.din, sl.din+corgAt, sl.din+coutAt, sl.din+cbaseAt, sl.din+p.baseAt, sl.unc, sl.work+p.resAt[2], nch, celems, L.be,
                   L.nblocks, L.bstride, L.pstride, L.shuffle ? 1 : 0, L.split ? 1 : 0, sharded ? 0 : 1);
            if(prof) a->cuEventRecord(p.ev[3], s);
            if(nst){
                CUdeviceptr tin = sl.din+inoffAt, tout = sl.din+outoffAt, pin = sl.work, pout = sl.work+al16(8*nst);
                CUdeviceptr unc = sl.unc, comp = sl.comp;
                unsigned long long nn = nst;
                void* rargs[] = {&tin, &tout, &pin, &pout, &unc, &comp, &nn};
                ck(a, a->cuLaunchKernel(cx->resolve, (unsigned)std::min<uint64_t>((nst+255)/256, 1024), 1, 1, 256, 1, 1, 0, s, rargs, nullptr),
                   "cuLaunchKernel (resolve)");
                size_t tb = 0;
                const int tr = a->compTempSize6[L.codec] ? a->compTempSize6[L.codec](nst, L.maxLen, opts, &tb, nst*L.maxLen, s)
                                                         : a->compTempSize5[L.codec](nst, L.maxLen, opts, &tb, nst*L.maxLen);
                if(tr) throw Fail("nvCOMP could not size its work memory (status "+std::to_string(tr)+")");
                growDevice(a, sl.temp, sl.tempBytes, std::max<size_t>(tb, 16));
                const int cr = a->compressAsync[L.codec]((const void* const*)(uintptr_t)pin, (const size_t*)(uintptr_t)(sl.din+inlenAt),
                                                        L.maxLen, nst, (void*)(uintptr_t)sl.temp, sl.tempBytes,
                                                        (void* const*)(uintptr_t)pout, (size_t*)(uintptr_t)(sl.work+p.resAt[0]), opts,
                                                        (int*)(uintptr_t)(sl.work+p.resAt[1]), s);
                if(cr) throw Fail("nvCOMP could not start compressing (status "+std::to_string(cr)+", "+a->nvcompPath+")"+
                                  (cr == 10 ? ": that nvCOMP may not run on this GPU (CUDA 13 builds need a Turing or newer one; "
                                              "pip install nvidia-libnvcomp-cu12, or set CPPZARR_NVCOMP_LIBRARY)" : ""));
            }
            if(prof) a->cuEventRecord(p.ev[4], s);
            if(workEnd > p.resAt[0]) ck(a, a->cuMemcpyDtoHAsync(sl.hres, sl.work+p.resAt[0], workEnd-p.resAt[0], s), "cuMemcpyDtoHAsync");
            // (the host waits for these events without spinning: CU_EVENT_BLOCKING_SYNC)
            if(!p.ev[0]) ck(a, a->cuEventCreate(&p.ev[0], 0x1 | 0x2 /* CU_EVENT_BLOCKING_SYNC, CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
            ck(a, a->cuEventRecord(p.ev[0], s), "cuEventRecord");
            stats.prep += now()-h0;
        };

        // ---- B: once a part's compressed sizes are back, its stored chunks laid out (blosc's chunk
        // header and blocks' starts, each stream after its size, a stream that does not get
        // smaller as it is, a chunk that does not as it is: as blosc stores them), put together on
        // the GPU and copied to the host
        auto stageB = [&](Part &p){
            WriteSlot &sl = *p.slot;
            const double w0 = now();
            ck(a, a->cuEventSynchronize(p.ev[0]), "cuEventSynchronize");
            const double h0 = now();
            stats.wait += h0-w0;
            const uint64_t nch = p.chunks.size(), nst = p.nst;
            const uint8_t* res = sl.hres;
            const uint64_t* csize = reinterpret_cast<const uint64_t*>(res);
            const int32_t* status = reinterpret_cast<const int32_t*>(res+(p.resAt[1]-p.resAt[0]));
            const uint32_t* nonzero = reinterpret_cast<const uint32_t*>(res+(p.resAt[2]-p.resAt[0]));
            for(uint64_t i = 0; i < nst; i++){
                if(status[i]) throw Fail("nvCOMP could not compress a chunk of "+fileName+" (status "+std::to_string(status[i])+")");
            }
            // the copies (pieces of at most 64 KiB) that put the stored chunks together: from the
            // compressed streams (kind 0), the chunks put together (1) or the headers (2), after their
            // stream's size (kind | 4)
            std::vector<uint64_t> psrc, pdst;
            std::vector<uint32_t> pn, ppre;
            std::vector<uint8_t> pkind, tab;
            auto add = [&](uint64_t src, uint64_t dst, uint64_t n, const uint8_t kind, const uint32_t pre){
                bool first = true;
                do{
                    const uint64_t m = std::min<uint64_t>(n, 64 << 10);
                    psrc.push_back(src);
                    pdst.push_back(dst);
                    pn.push_back((uint32_t)m);
                    ppre.push_back(first ? pre : 0);
                    pkind.push_back(first ? kind : (uint8_t)(kind & 3));
                    src += m;
                    dst += m;
                    n -= m;
                    first = false;
                } while(n > 0);
            };
            // (blosc chunks that do not get smaller: put together again, in order, after their header)
            std::vector<size_t> again;
            p.zero.assign(nch, 0);
            p.at.assign(nch, 0);
            p.size.assign(nch, 0);
            std::vector<uint64_t> kept(per);
            uint64_t total = 0;
            for(size_t i = 0; i < nch; i++){
                if(zeroRule && !nonzero[i]){
                    p.zero[i] = 1;
                    continue;
                }
                p.at[i] = total;
                const uint64_t u = i*L.U;
                if(L.kind == 0){
                    add(u, total, sB, 1, 0);
                    p.size[i] = sB;
                }
                else if(L.kind == 2){
                    add(i*outSlot, total, csize[i], 0, 0);
                    p.size[i] = csize[i];
                }
                else{
                    uint8_t hd[16];
                    memcpy(hd, L.header, 16);
                    uint64_t n = sB+16;
                    if(L.codec >= 0){
                        n = 16+4*L.nblocks;
                        for(uint64_t t = 0; t < per; t++){
                            const uint64_t c = csize[i*per+t];
                            kept[t] = c < L.sLen[t] ? c : L.sLen[t];
                            n += 4+kept[t];
                        }
                    }
                    if(L.codec < 0 || n > sB+16){
                        // (as it is, after its header: blosc's memcpyed chunk)
                        hd[2] |= 0x2;
                        wr32(hd+12, sB+16);
                        add(tab.size(), total, 16, 2, 0);
                        tab.insert(tab.end(), hd, hd+16);
                        if(L.codec < 0) add(u, total+16, sB, 1, 0);
                        else again.push_back(i);
                        p.size[i] = sB+16;
                    }
                    else{
                        wr32(hd+12, n);
                        const size_t h = tab.size();
                        tab.insert(tab.end(), hd, hd+16);
                        tab.resize(h+16+4*L.nblocks, 0);
                        uint64_t o = 16+4*L.nblocks;
                        for(uint64_t t = 0; t < per; t++){
                            if(t == 0 || L.sBlock[t] != L.sBlock[t-1]) wr32(&tab[h+16+4*L.sBlock[t]], o);
                            if(kept[t] == L.sLen[t]) add(u+L.sOff[t], total+o+4, kept[t], 1 | 4, (uint32_t)kept[t]);
                            else add((i*per+t)*outSlot, total+o+4, kept[t], 0 | 4, (uint32_t)kept[t]);
                            o += 4+kept[t];
                        }
                        add(h, total, 16+4*L.nblocks, 2, 0);
                        p.size[i] = n;
                    }
                }
                total = al16(total+p.size[i]);
            }
            p.stored = total;
            stats.outBytes += total;
            stats.again += again.size();
            // the pieces' tables, the headers, and the tables of the chunks put together again
            const uint64_t np = psrc.size(), na = again.size();
            const uint64_t srcAt = 0, dstAt = srcAt+al16(8*np), nAt = dstAt+al16(8*np), preAt = nAt+al16(4*np), kindAt = preAt+al16(4*np);
            const uint64_t tabAt = kindAt+al16(np), corgAt = tabAt+al16(tab.size()), coutAt = corgAt+al16(8*na*nd1);
            const uint64_t cbaseAt = coutAt+al16(8*na), nzAt = cbaseAt+al16(8*na), opsBytes = nzAt+al16(4*na);
            growPinned(a, sl.hops, sl.hopsBytes, opsBytes, 0);
            uint8_t* h = sl.hops;
            memcpy(h+srcAt, psrc.data(), 8*np);
            memcpy(h+dstAt, pdst.data(), 8*np);
            memcpy(h+nAt, pn.data(), 4*np);
            memcpy(h+preAt, ppre.data(), 4*np);
            memcpy(h+kindAt, pkind.data(), np);
            if(!tab.empty()) memcpy(h+tabAt, tab.data(), tab.size());
            for(size_t k = 0; k < na; k++){
                const size_t i = again[k];
                const uint64_t* idx = &grid[(size_t)p.chunks[i]*nd1];
                for(uint64_t d = 0; d < nDims; d++) reinterpret_cast<int64_t*>(h+corgAt)[k*nDims+d] = (int64_t)(idx[d]*chunkDims[d]);
                reinterpret_cast<uint64_t*>(h+coutAt)[k] = p.at[i]+16;
                reinterpret_cast<int64_t*>(h+cbaseAt)[k] = p.cbase[i];
            }
            {
                size_t freeB = 0, totalB = 0;
                a->cuMemGetInfo(&freeB, &totalB);
                const uint64_t need = (opsBytes > sl.dopsBytes ? opsBytes : 0)+(total > sl.packedBytes ? total : 0);
                if(freeB && need > freeB)
                    throw Fail("Not enough GPU memory for this write ("+std::to_string(need >> 20)+" MiB more needed, "+
                               std::to_string(freeB >> 20)+" MiB free): set CPPZARR_GPU_WRITE_PART_MB lower, or use smaller shards");
            }
            growDevice(a, sl.dops, sl.dopsBytes, std::max<uint64_t>(opsBytes, 16));
            growDevice(a, sl.packed, sl.packedBytes, std::max<uint64_t>(total, 16));
            growPinned(a, sl.hout, sl.houtBytes, std::max<uint64_t>(total, 16), 0);
            ck(a, a->cuMemcpyHtoDAsync(sl.dops, sl.hops, opsBytes, s), "cuMemcpyHtoDAsync");
            if(np){
                CUdeviceptr ps = sl.dops+srcAt, pd = sl.dops+dstAt, pnn = sl.dops+nAt, pp = sl.dops+preAt, pk = sl.dops+kindAt;
                CUdeviceptr comp = sl.comp, unc = sl.unc, ptab = sl.dops+tabAt, out = sl.packed;
                void* args[] = {&ps, &pd, &pnn, &pp, &pk, &comp, &unc, &ptab, &out};
                if(np > 0x7fffffffULL) throw Fail("Too many chunks in one part of a GPU write: set CPPZARR_GPU_WRITE_PART_MB lower");
                ck(a, a->cuLaunchKernel(cx->pack, (unsigned)np, 1, 1, 256, 1, 1, 0, s, args, nullptr), "cuLaunchKernel (pack)");
            }
            if(na){
                // (blosc stores these unshuffled: put together again in order, into their place)
                const CUdeviceptr geoAt = sl.din;
                gather(geoAt, sl.dops+corgAt, sl.dops+coutAt, sl.dops+cbaseAt, sl.din+p.baseAt, sl.packed, sl.dops+nzAt, na, celems, celems,
                       1, 0, 0, 0, 0, sharded ? 0 : 1);
            }
            if(prof) a->cuEventRecord(p.ev[5], s);
            if(total) ck(a, a->cuMemcpyDtoHAsync(sl.hout, sl.packed, total, s), "cuMemcpyDtoHAsync");
            if(!p.ev[1]) ck(a, a->cuEventCreate(&p.ev[1], 0x1 | 0x2 /* CU_EVENT_BLOCKING_SYNC, CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
            ck(a, a->cuEventRecord(p.ev[1], s), "cuEventRecord");
            stats.layout += now()-h0;
        };

        // ---- the shards a write stores whole (sharded arrays, not crop writes): each one's touched
        // inner chunks, counted down as parts append them
        const std::string uuid = sharded && !crop ? generateUUID() : std::string();
        std::map<int64_t, ShardFile> shardFiles;
        if(sharded && !crop){
            for(int64_t f = 0; f < numChunks; f++){
                if(touched[(size_t)f] != 1) continue;
                ShardFile &sf = shardFiles[f/perShard];
                if(sf.left++ == 0){
                    const std::string &name = Zarr.get_chunkNames(f);
                    const std::vector<uint64_t> sAV = Zarr.chunkToShard(Zarr.get_chunkAxisVals(name));
                    sf.path = fileName+"/"+Zarr.get_subfoldersString(sAV)+"/"+Zarr.chunkKey(Zarr.chunkNameToShardName(name));
                    sf.writePath = useUuid ? sf.path+uuid : sf.path;
                    sf.index.assign(2*(size_t)perShard, emptyEntry);
                }
            }
        }
        // (after an error, the shards not finished are closed, and their temporary files removed)
        struct ShardCleanup {
            std::map<int64_t, ShardFile> &files;
            const bool tmp;
            ~ShardCleanup(){ for(auto &kv : files) if(!kv.second.done) kv.second.abandon(tmp); }
        } shardCleanup{shardFiles, useUuid};
        const uint64_t indexBytes = 16*(uint64_t)perShard+(indexChecksum ? 4 : 0);
        // A part's inner chunks appended to their shards (a shard's run of them at once, the shards on
        // the call's threads), and the shards whose last inner chunks they are finished
        auto appendShards = [&](Part &p){
            const uint8_t* out = p.slot->hout;
            std::vector<std::pair<size_t, size_t>> runs;
            for(size_t i = 0; i < p.chunks.size(); i++){
                if(runs.empty() || p.chunks[i]/perShard != p.chunks[runs.back().first]/perShard) runs.push_back({i, i+1});
                else runs.back().second = i+1;
            }
            std::vector<std::string> errs(runs.size());
            #pragma omp parallel num_threads(team)
            {
                std::vector<std::pair<const uint8_t*, uint64_t>> pieces;
                std::vector<uint32_t> crcs;
                const int w = omp_get_thread_num(), nt = omp_get_num_threads();
                for(size_t r = (size_t)w; r < runs.size(); r += (size_t)nt){
                    ShardFile &sf = shardFiles.at(p.chunks[runs[r].first]/perShard);
                    if(!sf.opened && !sf.open(indexAtStart ? indexBytes : 0)){
                        errs[r] = "Check permissions or filepath. Cannot write to path: "+sf.writePath;
                        continue;
                    }
                    pieces.clear();
                    crcs.assign(runs[r].second-runs[r].first, 0);
                    for(size_t i = runs[r].first; i < runs[r].second; i++){
                        sf.left--;
                        if(p.zero[i]) continue;
                        const uint64_t n = p.size[i]+(chunkChecksum ? 4 : 0);
                        const int64_t pos = p.chunks[i]%perShard;
                        sf.index[2*pos] = sf.bytes;
                        sf.index[2*pos+1] = n;
                        sf.bytes += n;
                        pieces.push_back({out+p.at[i], p.size[i]});
                        if(chunkChecksum){
                            uint32_t &c = crcs[i-runs[r].first];
                            c = crc32c(out+p.at[i], (size_t)p.size[i]);
                            pieces.push_back({reinterpret_cast<const uint8_t*>(&c), 4});
                        }
                    }
                    if(!sf.append(pieces)){
                        errs[r] = "Could not write all of "+sf.writePath+" (is the disk full?)";
                        continue;
                    }
                    if(sf.left == 0){
                        if(!sf.finish(indexAtStart, indexChecksum)){
                            errs[r] = "Could not write all of "+sf.writePath+" (is the disk full?)";
                            continue;
                        }
                        if(useUuid && !renameReplace(sf.writePath, sf.path)){
                            removePath(sf.writePath);
                            errs[r] = "Cannot move the temporary file into place: "+sf.path;
                            continue;
                        }
                        sf.done = true;
                    }
                }
            }
            for(const auto &e : errs) if(!e.empty()) throw Fail(e);
        };

        // ---- C: once a part's stored chunks are in host memory, its files written as
        // parallelWriteZarr writes them (with its chunks' checksums, when the array has them)
        auto stageC = [&](Part &p){
            const double w0 = now();
            ck(a, a->cuEventSynchronize(p.ev[1]), "cuEventSynchronize");
            const double c0 = now();
            stats.wait += c0-w0;
            if(prof){
                float g = 0;
                if(!a->cuEventElapsedTime(&g, p.ev[2], p.ev[3])) stats.gpu[0] += g;
                if(!a->cuEventElapsedTime(&g, p.ev[3], p.ev[4])) stats.gpu[1] += g;
                if(!a->cuEventElapsedTime(&g, p.ev[0], p.ev[5])) stats.gpu[2] += g;
                if(!a->cuEventElapsedTime(&g, p.ev[5], p.ev[1])) stats.gpu[3] += g;
            }
            if(sharded && !crop) appendShards(p);
            else{
                const uint8_t* out = p.slot->hout;
                PreparedChunks prepared;
                prepared.first = p.f0;
                prepared.last = p.f1;
                prepared.get = [&](const int64_t f, StoredChunk &c, std::string &why2) -> bool {
                    const int64_t i = f >= p.f0 && f < p.f1 ? p.local[(size_t)(f-p.f0)] : -1;
                    if(i < 0){
                        why2 = "Write error: the GPU did not prepare chunk "+Zarr.get_chunkNames(f);
                        return false;
                    }
                    if(p.zero[(size_t)i]){
                        c.zero = true;
                        return true;
                    }
                    c.empty = false;
                    c.data = out+p.at[(size_t)i];
                    c.n = p.size[(size_t)i];
                    if(chunkChecksum){
                        c.hasCrc = true;
                        c.crc = crc32c(c.data, (size_t)c.n);
                    }
                    return true;
                };
                if(parallelWriteZarrPrepared(Zarr, startCoords, endCoords, writeShape, bits, useUuid, crop, sparse, prepared)){
                    std::string e = Zarr.get_errString();
                    while(!e.empty() && e.back() == '\n') e.pop_back();
                    throw Fail(e);
                }
            }
            stats.files += now()-c0;
            // (its lists, no longer needed)
            std::vector<int64_t>().swap(p.chunks);
            std::vector<int64_t>().swap(p.local);
            std::vector<int64_t>().swap(p.cbase);
            std::vector<uint8_t>().swap(p.zero);
            std::vector<uint64_t>().swap(p.at);
            std::vector<uint64_t>().swap(p.size);
        };

        // ---- the parts, two at a time: the GPU puts the next one together while this one's files
        // are written
        if(!parts.empty()){
            stageA(parts[0], *slots[0]);
            stageB(parts[0]);
        }
        for(size_t k = 0; k < parts.size(); k++){
            if(k+1 < parts.size()) stageA(parts[k+1], *slots[(k+1)%2]);
            stageC(parts[k]);
            if(k+1 < parts.size()) stageB(parts[k+1]);
        }
        for(const auto &kv : shardFiles) if(!kv.second.done) throw Fail("Write error: the shard "+kv.second.path+" was not finished");
        if(prof){
            fprintf(stderr, "cppzarr GPU write: %s: %llu chunks (%llu stored as they are) in %zu parts, %.1f MB in, %.1f MB stored | "
                    "listing %.1f ms, setting up %.1f ms, laying out %.1f ms, writing files %.1f ms, waiting for the GPU %.1f ms | on "
                    "the GPU: put together %.1f, compress %.1f, stored chunks %.1f, copy to the host %.1f ms | done at %.1f ms\n",
                    fileName.c_str(), (unsigned long long)stats.chunks, (unsigned long long)stats.again, stats.parts, stats.inBytes/1e6,
                    stats.outBytes/1e6, stats.plan, stats.prep, stats.layout, stats.files, stats.wait, stats.gpu[0], stats.gpu[1],
                    stats.gpu[2], stats.gpu[3], now());
        }
    }
    catch(const std::exception &e){
        Zarr.set_errString(std::string(e.what())+"\n");
        release();
        return 1;
    }
    release();
    return 0;
}
