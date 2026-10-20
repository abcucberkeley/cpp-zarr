#include "gpureadzarr.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <thread>
#include <tuple>
#include "gpu/cudaapi.h"
#include "gpu/kernels_ptx.h"
#include "helperfunctions.h"
#include "parallelreadzarr.h"
#ifdef _WIN32
// (without windows.h's min and max macros, which the std::min and std::max calls would expand)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

using namespace czcuda;

namespace {

// ---------------------------------------------------------------- host threads
// A pool of threads that persist between calls (so that small reads do not start threads), on
// which a call runs fn(i) for i in [0, n) with up to `width` threads, the calling one among
// them. Several calls can use it at once.
class Pool {
    struct Job {
        const std::function<void(int64_t)>* fn = nullptr;
        int64_t n = 0;
        std::atomic<int64_t> next{0}, done{0};
        int want = 0, joined = 0;
        std::mutex dm;
        std::condition_variable dcv;
        std::string error;
    };
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::shared_ptr<Job>> jobs;
    std::vector<std::thread> threads;

    static void work(Job &j){
        while(true){
            const int64_t i = j.next.fetch_add(1);
            if(i >= j.n) return;
            try{
                (*j.fn)(i);
            }
            catch(const std::exception &e){
                std::lock_guard<std::mutex> g(j.dm);
                if(j.error.empty()) j.error = e.what();
            }
            catch(const std::string &e){
                std::lock_guard<std::mutex> g(j.dm);
                if(j.error.empty()) j.error = e;
            }
            if(j.done.fetch_add(1)+1 == j.n){
                std::lock_guard<std::mutex> g(j.dm);
                j.dcv.notify_all();
            }
        }
    }
    void worker(){
        while(true){
            std::shared_ptr<Job> j;
            {
                std::unique_lock<std::mutex> lk(m);
                cv.wait(lk, [&]{ return !jobs.empty(); });
                j = jobs.front();
                if(++j->joined >= j->want) jobs.pop_front();
            }
            work(*j);
        }
    }
public:
    ~Pool(){ for(auto &t : threads) t.detach(); }
    // Runs fn for every i; returns the first error an fn threw, or ""
    std::string run(const int64_t n, int width, const std::function<void(int64_t)> &fn){
        if(n <= 0) return "";
        width = (int)std::max<int64_t>(1, std::min<int64_t>(width, n));
        auto j = std::make_shared<Job>();
        j->fn = &fn;
        j->n = n;
        j->want = width-1;
        if(width > 1){
            std::lock_guard<std::mutex> g(m);
            while((int)threads.size() < std::min(width-1, 512)) threads.emplace_back([this]{ worker(); });
            jobs.push_back(j);
            // (as many threads as join it: woken all for every job, the threads of a wide pool
            // cost small reads more than they gave)
            for(int k = 0; k < j->want; k++) cv.notify_one();
        }
        work(*j);
        {
            std::unique_lock<std::mutex> lk(j->dm);
            j->dcv.wait(lk, [&]{ return j->done.load() >= j->n; });
        }
        if(width > 1){
            // (a job not every thread took yet leaves the queue: its items are all done)
            std::lock_guard<std::mutex> g(m);
            auto it = std::find(jobs.begin(), jobs.end(), j);
            if(it != jobs.end()) jobs.erase(it);
        }
        return j->error;
    }
};
Pool& pool(){
    static Pool* p = new Pool();   // (never destroyed: its threads may outlive static destruction)
    return *p;
}

// ---------------------------------------------------------------- files
// A file read at offsets (pread; positioned ReadFile on Windows)
class File {
#ifdef _WIN32
    HANDLE h = INVALID_HANDLE_VALUE;
#else
    int fd = -1;
#endif
public:
    File(){}
    File(const File&) = delete;
    File& operator=(const File&) = delete;
    ~File(){ close(); }
    bool open(const std::string &path){
    #ifdef _WIN32
        h = CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
        return h != INVALID_HANDLE_VALUE;
    #else
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        return fd >= 0;
    #endif
    }
    void close(){
    #ifdef _WIN32
        if(h != INVALID_HANDLE_VALUE) CloseHandle(h);
        h = INVALID_HANDLE_VALUE;
    #else
        if(fd >= 0) ::close(fd);
        fd = -1;
    #endif
    }
    int64_t size(){
    #ifdef _WIN32
        LARGE_INTEGER s;
        return GetFileSizeEx(h, &s) ? (int64_t)s.QuadPart : -1;
    #else
        struct stat st;
        return fstat(fd, &st) == 0 ? (int64_t)st.st_size : -1;
    #endif
    }
    bool readAt(uint64_t off, void* buf, uint64_t n){
        uint8_t* p = static_cast<uint8_t*>(buf);
        while(n > 0){
        #ifdef _WIN32
            OVERLAPPED o{};
            o.Offset = (DWORD)(off & 0xffffffffULL);
            o.OffsetHigh = (DWORD)(off >> 32);
            DWORD got = 0;
            const DWORD ask = (DWORD)std::min<uint64_t>(n, 1u << 30);
            if(!ReadFile(h, p, ask, &got, &o) || got == 0) return false;
            const uint64_t r = got;
        #else
            const ssize_t rr = pread(fd, p, n, (off_t)off);
            if(rr < 0 && errno == EINTR) continue;
            if(rr <= 0) return false;
            const uint64_t r = (uint64_t)rr;
        #endif
            p += r;
            off += r;
            n -= r;
        }
        return true;
    }
};

// ---------------------------------------------------------------- GPU state
// The kernels and reusable buffers of one CUDA context
struct Slot {
    bool busy = false;
    bool lost = false;               // (its GPU work never finished: its buffers are never reused)
    uint8_t* host = nullptr;         // pinned staging: stored bytes, chunks decoded on the host, tables
    size_t hostBytes = 0;
    uint8_t* small = nullptr;        // pinned: the decompression checks' results
    CUdeviceptr comp = 0, dec = 0, work = 0, temp = 0;
    size_t compBytes = 0, decBytes = 0, workBytes = 0, tempBytes = 0;
};
struct Ctx {
    CUcontext ctx = nullptr;
    CUmodule mod = nullptr;
    CUfunction scatter32 = nullptr, scatter64 = nullptr, resolve = nullptr, fill = nullptr, check = nullptr;
    int sms = 1;
    NvcompAlignments align[NCodecs];
    std::mutex m;
    std::vector<std::unique_ptr<Slot>> slots;
};
std::mutex ctxMutex;
std::vector<std::unique_ptr<Ctx>> ctxs;

struct Fail : std::runtime_error { using std::runtime_error::runtime_error; };

void ck(const Api* a, CUresult r, const char* what){
    if(r) throw Fail(std::string(what)+": "+errorText(a, r));
}

Ctx& context(const Api* a, CUcontext c){
    std::lock_guard<std::mutex> g(ctxMutex);
    for(auto &x : ctxs) if(x->ctx == c) return *x;
    auto x = std::unique_ptr<Ctx>(new Ctx());
    x->ctx = c;
    std::string ptx;
    for(const char* p : cppzarrKernelsPtx) ptx += p;
    char log[4096] = {0};
    int opts[2] = {5 /* CU_JIT_ERROR_LOG_BUFFER */, 6 /* CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES */};
    void* vals[2] = {log, (void*)(uintptr_t)sizeof(log)};
    const CUresult r = a->cuModuleLoadDataEx(&x->mod, ptx.c_str(), 2, opts, vals);
    if(r) throw Fail("The GPU kernels could not be loaded ("+errorText(a, r)+(log[0] ? ": "+std::string(log) : "")+
                     "): the NVIDIA driver may be too old");
    ck(a, a->cuModuleGetFunction(&x->scatter32, x->mod, "cz_scatter32"), "cuModuleGetFunction");
    ck(a, a->cuModuleGetFunction(&x->scatter64, x->mod, "cz_scatter64"), "cuModuleGetFunction");
    ck(a, a->cuModuleGetFunction(&x->resolve, x->mod, "cz_resolve"), "cuModuleGetFunction");
    ck(a, a->cuModuleGetFunction(&x->fill, x->mod, "cz_fill"), "cuModuleGetFunction");
    ck(a, a->cuModuleGetFunction(&x->check, x->mod, "cz_check"), "cuModuleGetFunction");
    CUdevice dev = 0;
    ck(a, a->cuCtxGetDevice(&dev), "cuCtxGetDevice");
    a->cuDeviceGetAttribute(&x->sms, 16 /* CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT */, dev);
    for(int k = 0; k < NCodecs; k++){
        x->align[k] = {1, 1, 1};
        if(a->alignments[k]){
            NvcompOpts o{};
            if(a->alignments[k](o, &x->align[k])) x->align[k] = {1, 1, 1};
        }
    }
    ctxs.push_back(std::move(x));
    return *ctxs.back();
}

Slot& takeSlot(Ctx &c){
    std::lock_guard<std::mutex> g(c.m);
    for(auto &s : c.slots) if(!s->busy && !s->lost){ s->busy = true; return *s; }
    c.slots.emplace_back(new Slot());
    c.slots.back()->busy = true;
    return *c.slots.back();
}

void growDevice(const Api* a, CUdeviceptr &p, size_t &have, const size_t need){
    if(need <= have) return;
    if(p) a->cuMemFree(p);
    p = 0;
    have = 0;
    const size_t n = std::max<size_t>(need+need/4, 1 << 20);
    ck(a, a->cuMemAlloc(&p, n), "Could not allocate GPU memory (cuMemAlloc)");
    have = n;
}

// Pinned host memory of at least need bytes, keeping the first keep bytes
void growHost(const Api* a, Slot &s, const size_t need, const size_t keep){
    if(need <= s.hostBytes) return;
    const size_t n = std::max<size_t>(need+need/4, 64 << 20);
    void* p = nullptr;
    ck(a, a->cuMemHostAlloc(&p, n, 0), "Could not allocate pinned host memory (cuMemHostAlloc)");
    if(keep && s.host) memcpy(p, s.host, keep);
    if(s.host) a->cuMemFreeHost(s.host);
    s.host = static_cast<uint8_t*>(p);
    s.hostBytes = n;
}

// ---------------------------------------------------------------- host phase
// What a region needs of its array
struct Region {
    zarr* z;
    std::vector<uint64_t> start, end, rshape, shape, inner, shard, cps;
    uint64_t ndim = 0, ts = 0, nInner = 1, chunkElems = 1, chunkBytes = 0, nChunks = 1;
    bool sharded = false, checksum = false, indexAtStart = false, indexCrc = true;
    std::string compressor;
    std::vector<uint64_t> axes;      // a stored chunk's axes, slowest first
    DtypeInfo dt;
    uint8_t fill[16];
    int swapUnit = 0;
    int widen = 0;                   // 16-bit floats written as float32s: 1 float16, 2 bfloat16
    uint64_t outTs = 0;              // an output element's bytes
    uint64_t present = 0;            // its chunks that were found
    uint64_t group = 0;
};

// A file to read: a shard (and the inner chunks of it the region needs: grid coordinates and
// their position in the shard) or a chunk
struct FileJob {
    uint32_t region;
    std::string path;
    std::vector<std::pair<std::vector<uint64_t>, uint64_t>> want;
    // found: (grid coordinates, offset, bytes)
    std::vector<std::tuple<std::vector<uint64_t>, uint64_t, uint64_t>> found;
    std::string error;
};

// A stored chunk, read
struct Chunk {
    uint32_t region;
    std::vector<uint64_t> grid;
    const std::string* path;
    uint64_t at, bytes;              // in its file, then (once its part is read) in the staging buffer
};

// A block of a chunk's decoded data, and the compressed streams it decodes from
struct Block {
    uint32_t chunk;
    uint64_t first, nelem, bytes;
    bool shuffled;
    int64_t storedAt;                // staging offset of a block that is stored as is, or -1
};
struct Stream {
    uint32_t block;
    int codec;                       // a czcuda::Codec, or -1: stored (copied)
    uint64_t at, bytes, outBytes, outOffset;
};

uint32_t rd32(const uint8_t* p){ uint32_t v; memcpy(&v, p, 4); return v; }
int32_t rdi32(const uint8_t* p){ int32_t v; memcpy(&v, p, 4); return v; }

// Bits of a byte range from its start, lowest first, as zstd writes its table descriptions; bits
// past the end read as 0
struct Bits {
    const uint8_t* p;
    uint64_t n, at;
    uint32_t peek(const int k) const {
        uint64_t w = 0;
        for(uint64_t i = 0; i < 5 && (at >> 3)+i < n; i++) w |= (uint64_t)p[(at >> 3)+i] << (8*i);
        return (uint32_t)((w >> (at & 7)) & ((uint64_t(1) << k)-1));
    }
    uint32_t get(const int k){ const uint32_t v = peek(k); at += (uint64_t)k; return v; }
};

// Whether bytes start a zstd FSE table description (RFC 8878 4.1.1) that zstd accepts: an
// accuracy log of at most maxLog, probabilities that add up, symbols up to maxSymbol; used, its
// bytes
bool fseTableOk(const uint8_t* p, const uint64_t n, const int maxLog, const int maxSymbol, uint64_t &used){
    Bits r{p, n, 0};
    const int log = (int)r.get(4)+5;
    if(log > maxLog) return false;
    int remaining = (1 << log)+1, threshold = 1 << log, nbBits = log+1, symbol = 0;
    bool zero = false;
    while(true){
        if(zero){
            // (symbols of probability 0 after one: 2-bit counts of more, 3 meaning more follow)
            uint32_t k;
            do{ k = r.get(2); symbol += (int)k; } while(k == 3 && symbol <= maxSymbol);
            if(symbol > maxSymbol) return false;
        }
        const int max = (2*threshold-1)-remaining;
        const uint32_t v = r.peek(nbBits);
        int count;
        if((int)(v & (uint32_t)(threshold-1)) < max){ count = (int)(v & (uint32_t)(threshold-1)); r.at += (uint64_t)(nbBits-1); }
        else{
            count = (int)(v & (uint32_t)(2*threshold-1));
            if(count >= threshold) count -= max;
            r.at += (uint64_t)nbBits;
        }
        count--;
        remaining -= count >= 0 ? count : -count;
        symbol++;
        zero = count == 0;
        if(remaining < threshold){
            if(remaining <= 1) break;
            int hb = 0;
            while((2 << hb) <= remaining) hb++;
            nbBits = hb+1;
            threshold = 1 << hb;
        }
        if(symbol > maxSymbol) break;
    }
    if(remaining != 1 || symbol > maxSymbol+1) return false;
    used = (r.at+7)/8;
    return used <= n;
}

// What a zstd frame's blocks have set up for the blocks after them
struct ZstdFrameState { bool huffman = false, seqTables = false; };

// Whether a compressed zstd block (RFC 8878 3.1.1.3) is laid out as zstd decodes it: its literals
// section (the header, a Huffman tree description that zstd accepts, the streams' sizes and end
// marks) and its sequences section (their number, a mode and table description for each kind of
// symbol, the bitstream's end mark), within the block, and no more output than a block holds.
bool zstdBlockOk(const uint8_t* b, const uint64_t size, const uint64_t blockMax, ZstdFrameState &st){
    const uint8_t* const end = b+size;
    // ---- literals
    const int ltype = b[0] & 3, sf = (b[0] >> 2) & 3;
    uint64_t regen = 0, comp = 0, hdr = 0;
    if(ltype <= 1){
        if(sf == 0 || sf == 2){ hdr = 1; regen = b[0] >> 3; }
        else if(sf == 1){ hdr = 2; if(size < 2) return false; regen = (b[0] >> 4) | ((uint64_t)b[1] << 4); }
        else{ hdr = 3; if(size < 3) return false; regen = (b[0] >> 4) | ((uint64_t)b[1] << 4) | ((uint64_t)b[2] << 12); }
        comp = ltype == 0 ? regen : 1;
    }
    else{
        hdr = sf <= 1 ? 3 : sf == 2 ? 4 : 5;
        if(size < hdr) return false;
        uint64_t v = 0;
        for(uint64_t k = 0; k < hdr; k++) v |= (uint64_t)b[k] << (8*k);
        v >>= 4;
        const int bits = sf <= 1 ? 10 : sf == 2 ? 14 : 18;
        regen = v & ((uint64_t(1) << bits)-1);
        comp = (v >> bits) & ((uint64_t(1) << bits)-1);
    }
    if(regen > blockMax || hdr+comp > size) return false;
    if(ltype >= 2){
        const uint8_t* l = b+hdr;
        uint64_t tree = 0;
        if(ltype == 2){
            // the Huffman tree description: FSE-compressed weights, or 4-bit weights
            if(comp < 1) return false;
            const uint8_t h = l[0];
            if(h < 128){
                uint64_t used = 0;
                if(h == 0 || 1+(uint64_t)h > comp || !fseTableOk(l+1, h, 6, 255, used)) return false;
                tree = 1+(uint64_t)h;
            }
            else{
                const uint64_t nw = (uint64_t)h-127, nb = (nw+1)/2;
                if(1+nb > comp) return false;
                uint32_t total = 0, ones = 0;
                for(uint64_t i = 0; i < nw; i++){
                    const uint32_t w = (i & 1) ? (l[1+i/2] & 15) : (l[1+i/2] >> 4);
                    if(w > 12) return false;
                    if(w) total += (1u << w) >> 1;
                    ones += w == 1;
                }
                if(!total) return false;
                int hb = 0;
                while((2u << hb) <= total) hb++;
                if(hb+1 > 12) return false;
                const uint32_t rest = (1u << (hb+1))-total;
                if(rest & (rest-1)) return false;           // (the last weight's share: a power of 2)
                if(rest == 1) ones++;
                if(ones < 2 || (ones & 1)) return false;
                tree = 1+nb;
            }
            st.huffman = true;
        }
        else if(!st.huffman) return false;                  // (treeless, with no tree before it)
        const uint8_t* q = l+tree;
        const uint64_t streams = comp-tree;
        if(sf == 0){
            if(streams < 1 || q[streams-1] == 0) return false;
        }
        else{
            if(streams < 10 || regen < 6) return false;
            const uint64_t s1 = q[0] | ((uint64_t)q[1] << 8), s2 = q[2] | ((uint64_t)q[3] << 8), s3 = q[4] | ((uint64_t)q[5] << 8);
            if(!s1 || !s2 || !s3 || 6+s1+s2+s3 >= streams) return false;
            const uint64_t ends[4] = {6+s1, 6+s1+s2, 6+s1+s2+s3, streams};
            for(const uint64_t e : ends) if(q[e-1] == 0) return false;
        }
    }
    // ---- sequences
    const uint8_t* s = b+hdr+comp;
    if(s >= end) return false;
    uint64_t nseq = *s++;
    if(nseq == 0xff){
        if(s+2 > end) return false;
        nseq = s[0]+((uint64_t)s[1] << 8)+0x7f00;
        s += 2;
    }
    else if(nseq > 0x7f){
        if(s >= end) return false;
        nseq = ((nseq-0x80) << 8)+*s++;
    }
    if(nseq == 0) return s == end;
    // (every sequence makes at least 3 bytes)
    if(regen+3*nseq > blockMax) return false;
    if(s >= end || (*s & 3)) return false;
    const uint8_t modes = *s++;
    static const int maxLog[3] = {9, 8, 9}, maxSym[3] = {35, 31, 52};
    for(int k = 0; k < 3; k++){
        const int mode = (modes >> (6-2*k)) & 3;
        if(mode == 1){
            if(s >= end || *s > maxSym[k]) return false;
            s++;
        }
        else if(mode == 2){
            uint64_t used = 0;
            if(!fseTableOk(s, (uint64_t)(end-s), maxLog[k], maxSym[k], used)) return false;
            s += used;
        }
        else if(mode == 3 && !st.seqTables) return false;   // (repeats a table no block had)
    }
    st.seqTables = true;
    return s < end && end[-1] != 0;
}

// Whether a zstd stream is one whole frame that decodes to `want` bytes, as far as its structure
// tells: nvCOMP decodes what it is given without checking it, and damaged frames made it read and
// write outside its buffers (which ends the process's CUDA context) or never finish. Checked: the
// magic number, the header (reserved bits, no dictionary, a window of at most 128 MiB, the content
// size when it is there), every block's type and size and a compressed block's layout
// (zstdBlockOk), and that the blocks end where the stream does. (Damage inside entropy-coded data
// can still get through: what only decoding finds.) A stream that fails is decoded on the host,
// which reports the damage.
bool zstdFrameOk(const uint8_t* p, const uint64_t n, const uint64_t want){
    if(n < 9 || rd32(p) != 0xFD2FB528u) return false;
    const uint8_t fhd = p[4];
    if(fhd & 0x08) return false;                               // (reserved)
    const int fcsFlag = fhd >> 6, single = (fhd >> 5) & 1, checksum = (fhd >> 2) & 1, did = fhd & 3;
    if(did) return false;                                      // (a dictionary)
    uint64_t q = 5;
    uint64_t window = 0;
    if(!single){
        const uint8_t wd = p[q++];
        const uint64_t e = wd >> 3, m = wd & 7;
        window = (uint64_t(1) << (10+e)) + ((uint64_t(1) << (10+e))/8)*m;
        if(10+e > 27) return false;
    }
    const int fcsBytes = fcsFlag == 0 ? (single ? 1 : 0) : fcsFlag == 1 ? 2 : fcsFlag == 2 ? 4 : 8;
    if(q+fcsBytes > n) return false;
    if(fcsBytes){
        uint64_t fcs = 0;
        for(int k = 0; k < fcsBytes; k++) fcs |= (uint64_t)p[q+k] << (8*k);
        if(fcsBytes == 2) fcs += 256;
        if(fcs != want) return false;
        if(single) window = fcs;
    }
    q += fcsBytes;
    const uint64_t blockMax = std::min<uint64_t>(window ? window : (uint64_t(1) << 17), uint64_t(1) << 17);
    uint64_t made = 0, upTo = 0;                               // (decoded bytes known, and at most)
    ZstdFrameState st;
    while(true){
        if(q+3 > n) return false;
        const uint32_t h = (uint32_t)p[q] | ((uint32_t)p[q+1] << 8) | ((uint32_t)p[q+2] << 16);
        q += 3;
        const int last = h & 1, type = (h >> 1) & 3;
        const uint64_t size = h >> 3;
        if(type == 3) return false;                            // (reserved)
        if(type == 0){                                         // raw
            if(size > blockMax || q+size > n) return false;
            q += size; made += size; upTo += size;
        }
        else if(type == 1){                                    // one byte, repeated
            if(size > blockMax || q+1 > n) return false;
            q += 1; made += size; upTo += size;
        }
        else{                                                  // compressed
            if(size == 0 || size > blockMax || q+size > n || !zstdBlockOk(p+q, size, blockMax, st)) return false;
            q += size;
            upTo += blockMax;
        }
        if(made > want) return false;
        if(last) break;
    }
    if(checksum) q += 4;
    return q == n && upTo >= want;
}

// Splits a blosc chunk into its blocks and streams for the GPU; false if the GPU cannot decode
// it (it is then decoded on the host): bitshuffle and other filters, blosclz, run-length and
// special-value streams, dictionaries, zstd frames that do not check out
bool bloscPlan(const uint8_t* c, const uint64_t at, const uint64_t size, const uint32_t chunk, const uint64_t chunkBytes,
               const uint64_t elemBytes, std::vector<Block> &blocks, std::vector<Stream> &streams, const std::vector<bool> &gpuCodec){
    if(size < 16) return false;
    const uint8_t ver = c[0], flags = c[2], ts = c[3];
    const uint64_t nbytes = rd32(c+4), blocksize = rd32(c+8), cbytes = rd32(c+12);
    if(nbytes != chunkBytes || cbytes > size || ts == 0) return false;
    // (blosc's shuffle unit is its typesize: the GPU unshuffles whole elements only)
    const bool extended = ver >= 3 && (flags & 0x5) == 0x5;
    const uint64_t hdr = extended ? 32 : 16;
    if(extended){
        if(size < 32) return false;
        // filters: none, or byte shuffle as the last one; no dictionary, no special values
        const uint8_t* filters = c+16;
        int shuffles = 0;
        for(int k = 0; k < 6; k++){
            if(filters[k] == 0) continue;
            if(filters[k] == 1 && k == 5) { shuffles++; continue; }
            return false;
        }
        // (any blosc2 flag: a dictionary, big-endian data, special values (zeros, NaNs, a
        // repeated value), lazy chunks; and user codecs)
        if(c[31] != 0 || ((flags >> 5) & 0x7) > 5) return false;
        (void)shuffles;
    }
    else if(flags & 0x4) return false;             // (bitshuffle)
    const bool shuffle = extended ? (c[16+5] == 1) : (flags & 0x1) != 0;
    if(shuffle && ts != elemBytes && ts > 1) return false;
    if(flags & 0x2){                               // memcpyed: the data after the header, as it is
        if(hdr+nbytes > size) return false;
        blocks.push_back({chunk, 0, nbytes/elemBytes, nbytes, false, (int64_t)(at+hdr)});
        return true;
    }
    if(blocksize == 0 || nbytes % ts || blocksize % elemBytes) return false;
    const int comp = (flags >> 5) & 0x7;
    const int codec = comp == 1 ? LZ4 : comp == 2 ? Snappy : comp == 4 ? Zstd : comp == 3 ? Deflate : -1;
    if(codec < 0 || !gpuCodec[codec]) return false;
    const uint64_t nblocks = (nbytes+blocksize-1)/blocksize;
    if(hdr+4*nblocks > size) return false;
    const bool dontSplit = (flags & 0x10) != 0;
    for(uint64_t b = 0; b < nblocks; b++){
        const uint64_t bs = std::min(blocksize, nbytes-b*blocksize);
        if(bs % ts) return false;
        const int32_t bst = rdi32(c+hdr+4*b);
        if(bst < 0 || (uint64_t)bst < hdr+4*nblocks || (uint64_t)bst >= size) return false;
        // (a block's streams end where the next block starts, or at the chunk's end)
        const uint64_t blockEnd = b+1 < nblocks ? (uint64_t)rdi32(c+hdr+4*(b+1)) : cbytes;
        // (blosc 1 splits a block into its bytes' streams unless told not to; blosc 2 decides
        // by the codec, and marks the chunk's choice the same way)
        const uint64_t nst = (!dontSplit && ts > 1 && (bs/ts)*ts == bs) ? ts : 1;
        const uint64_t sub = bs/nst;
        uint64_t p = (uint64_t)bst;
        const uint32_t blk = (uint32_t)blocks.size();
        uint64_t off = 0;
        bool stored = nst == 1;
        int64_t storedAt = -1;
        for(uint64_t k = 0; k < nst; k++){
            if(p+4 > size) return false;
            const int32_t csize = rdi32(c+p);
            p += 4;
            if(csize <= 0 || (uint64_t)csize > sub || p+(uint64_t)csize > std::min(size, blockEnd)) return false;
            if((uint64_t)csize == sub){
                if(nst == 1) storedAt = (int64_t)(at+p);
                else streams.push_back({blk, -1, at+p, sub, sub, off});
            }
            else{
                stored = false;
                if(codec == Deflate){
                    // (blosc's zlib streams: a zlib header, deflate data, an Adler-32)
                    if(csize < 6 || (c[p+1] & 0x20)) return false;
                    streams.push_back({blk, Deflate, at+p+2, (uint64_t)csize-6, sub, off});
                }
                else{
                    if(codec == Zstd && !zstdFrameOk(c+p, (uint64_t)csize, sub)) return false;
                    streams.push_back({blk, codec, at+p, (uint64_t)csize, sub, off});
                }
            }
            p += (uint64_t)csize;
            off += sub;
        }
        blocks.push_back({chunk, b*blocksize/elemBytes, bs/elemBytes, bs, shuffle && ts > 1, stored ? storedAt : -1});
    }
    return true;
}

struct Plan {
    std::vector<Region> regions;
    std::vector<std::unique_ptr<FileJob>> files;
    std::vector<Chunk> chunks;
};

// 16-byte aligned
uint64_t al16(uint64_t n){ return (n+15) & ~uint64_t(15); }

void planRegions(std::vector<GpuRegion> &in, const bool cOrder, const bool widen16, Plan &P){
    for(auto &g : in){
        Region r;
        r.z = g.array;
        zarr &z = *g.array;
        r.ndim = z.get_ndims();
        for(uint64_t d = 0; d < r.ndim; d++) r.shape.push_back(z.get_shape(d));
        r.start = g.start;
        r.end = g.end;
        if(r.start.empty() && r.end.empty()){
            r.start.assign(r.ndim, 0);
            r.end = r.shape;
        }
        if(r.start.size() != r.ndim || r.end.size() != r.ndim) throw Fail("The region of "+z.get_fileName()+" has the wrong number of coordinates");
        for(uint64_t d = 0; d < r.ndim; d++){
            if(r.end[d] > r.shape[d] || r.start[d] >= r.end[d]) throw Fail("The region of "+z.get_fileName()+" is outside the array on axis "+std::to_string(d));
            r.rshape.push_back(r.end[d]-r.start[d]);
        }
        if(!g.out) throw Fail("No GPU memory to read "+z.get_fileName()+" into");
        r.dt = dtypeInfo(z.get_dtype());
        if(!r.dt.kind) throw Fail(z.get_fileName()+": its data type is not supported");
        r.ts = r.dt.bytes;
        r.swapUnit = r.dt.bigEndian && r.dt.swapBytes() > 1 ? (int)r.dt.swapBytes() : 0;
        fillValueElement(z.get_fill_value(), z.get_dtype(), r.fill);
        // (the element as stored: big-endian data's in its byte order, which the GPU output,
        // in this machine's order, does not have)
        if(r.swapUnit)
            for(uint64_t u = 0; u+r.swapUnit <= r.ts; u += r.swapUnit) std::reverse(r.fill+u, r.fill+u+r.swapUnit);
        r.outTs = r.ts;
        if(widen16 && r.dt.kind == 'f' && r.ts == 2){
            r.widen = r.dt.bfloat16 ? 2 : 1;
            r.outTs = 4;
            uint16_t h;
            memcpy(&h, r.fill, 2);
            const float f = r.dt.bfloat16 ? bfloat16ToFloat(h) : halfToFloat(h);
            memcpy(r.fill, &f, 4);
        }
        r.sharded = z.get_shard();
        for(uint64_t d = 0; d < r.ndim; d++){
            r.inner.push_back(r.sharded ? z.get_chunk_shape(d) : z.get_chunks(d));
            if(r.sharded){
                r.shard.push_back(z.get_chunks(d));
                r.cps.push_back((r.shard[d]+r.inner[d]-1)/r.inner[d]);
                r.nInner *= r.cps[d];
            }
            r.chunkElems *= r.inner[d];
        }
        r.chunkBytes = r.chunkElems*r.ts;
        r.compressor = z.get_compressor();
        if(r.compressor != "none" && r.compressor != "blosc" && r.compressor != "gzip" && r.compressor != "zstd")
            throw Fail("The \""+r.compressor+"\" compressor is not supported");
        r.checksum = z.get_chunkChecksum();
        r.axes = z.get_chunkAxisOrder();
        if(r.sharded){
            r.indexAtStart = z.get_shardIndexAtStart();
            r.indexCrc = z.get_shardIndexChecksum();
        }
        (void)cOrder;
        P.regions.push_back(r);
    }
}

// The files each region needs, and the inner chunks of each shard
void planFiles(Plan &P){
    for(uint32_t ri = 0; ri < P.regions.size(); ri++){
        Region &r = P.regions[ri];
        zarr &z = *r.z;
        const uint64_t n = r.ndim;
        std::vector<uint64_t> lo(n), hi(n);
        r.nChunks = 1;
        for(uint64_t d = 0; d < n; d++){
            lo[d] = r.start[d]/r.inner[d];
            hi[d] = (r.end[d]+r.inner[d]-1)/r.inner[d];
            r.nChunks *= hi[d]-lo[d];
        }
        auto name = [&](const std::vector<uint64_t> &c){
            std::string s;
            for(uint64_t d = 0; d < n; d++){
                if(d) s += z.get_dimension_separator();
                s += std::to_string(c[d]);
            }
            return n ? s : std::string("0");
        };
        auto pathOf = [&](const std::vector<uint64_t> &c){
            const std::string sub = z.get_subfoldersString(c);
            return z.get_fileName()+"/"+(sub.empty() ? "" : sub+"/")+z.chunkKey(name(c));
        };
        if(!r.sharded){
            std::vector<uint64_t> c(lo);
            for(uint64_t i = 0; i < r.nChunks; i++){
                std::unique_ptr<FileJob> f(new FileJob());
                f->region = ri;
                f->path = pathOf(c);
                f->want.push_back({c, 0});
                P.files.push_back(std::move(f));
                for(int64_t d = (int64_t)n-1; d >= 0; d--){ if(++c[d] < hi[d]) break; c[d] = lo[d]; }
            }
            continue;
        }
        std::vector<uint64_t> slo(n), shi(n);
        uint64_t nShards = 1;
        for(uint64_t d = 0; d < n; d++){
            slo[d] = lo[d]/r.cps[d];
            shi[d] = (hi[d]-1)/r.cps[d]+1;
            nShards *= shi[d]-slo[d];
        }
        std::vector<uint64_t> s(slo);
        for(uint64_t i = 0; i < nShards; i++){
            std::unique_ptr<FileJob> f(new FileJob());
            f->region = ri;
            f->path = pathOf(s);
            // its inner chunks inside the region, last axis fastest
            std::vector<uint64_t> a(n), b(n), c(n);
            for(uint64_t d = 0; d < n; d++){
                a[d] = std::max(lo[d], s[d]*r.cps[d]);
                b[d] = std::min(hi[d], (s[d]+1)*r.cps[d]);
            }
            c = a;
            while(true){
                uint64_t pos = 0;
                for(uint64_t d = 0; d < n; d++) pos = pos*r.cps[d]+(c[d]-s[d]*r.cps[d]);
                f->want.push_back({c, pos});
                int64_t d = (int64_t)n-1;
                for(; d >= 0; d--){ if(++c[d] < b[d]) break; c[d] = a[d]; }
                if(d < 0) break;
            }
            P.files.push_back(std::move(f));
            for(int64_t d = (int64_t)n-1; d >= 0; d--){ if(++s[d] < shi[d]) break; s[d] = slo[d]; }
        }
    }
}

// Finds each file's stored chunks: a chunk file's size, or a shard's index (checked)
void findChunks(FileJob &f, const Region &r){
    File file;
    if(!file.open(f.path)) return;   // (a missing chunk or shard reads as the fill value)
    const int64_t sz = file.size();
    if(sz < 0){ f.error = "could not be read"; return; }
    if(!r.sharded){
        f.found.emplace_back(f.want[0].first, 0, (uint64_t)sz);
        return;
    }
    const uint64_t entries = 2*r.nInner*8, indexBytes = entries+(r.indexCrc ? 4 : 0);
    if((uint64_t)sz < indexBytes){ f.error = "is smaller than its index"; return; }
    std::vector<uint64_t> idx(2*r.nInner+1);
    if(!file.readAt(r.indexAtStart ? 0 : (uint64_t)sz-indexBytes, idx.data(), indexBytes)){ f.error = "could not be read"; return; }
    if(r.indexCrc){
        uint32_t crc;
        memcpy(&crc, reinterpret_cast<const uint8_t*>(idx.data())+entries, 4);
        if(crc != crc32c(reinterpret_cast<const uint8_t*>(idx.data()), entries)){ f.error = "has an index that does not match its checksum"; return; }
    }
    const uint64_t lo = r.indexAtStart ? indexBytes : 0, hi = r.indexAtStart ? (uint64_t)sz : (uint64_t)sz-indexBytes;
    const uint64_t empty = ~uint64_t(0);
    for(const auto &w : f.want){
        const uint64_t off = idx[2*w.second], n = idx[2*w.second+1];
        if(off == empty && n == empty) continue;
        if(off < lo || off > hi || n > hi-off){ f.error = "has an index entry outside the file"; return; }
        f.found.emplace_back(w.first, off, n);
    }
}

// The CUDA context GPU memory belongs to: the one it was allocated in, or for memory that belongs to
// none (allocated through CUDA's virtual memory API or a stream-ordered memory pool, as PyTorch's
// expandable segments and CuPy's async pool are) its device's primary context, the one the CUDA
// runtime uses
CUcontext memoryContext(const Api* a, const void* p){
    CUcontext c = nullptr;
    ck(a, a->cuPointerGetAttribute(&c, 1 /* CU_POINTER_ATTRIBUTE_CONTEXT */, (CUdeviceptr)(uintptr_t)p), "cuPointerGetAttribute");
    if(c) return c;
    int ordinal = -1;
    ck(a, a->cuPointerGetAttribute(&ordinal, 9 /* CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL */, (CUdeviceptr)(uintptr_t)p),
       "cuPointerGetAttribute (device)");
    // (each device's primary context retained once: the runtime keeps it anyway)
    static std::mutex m;
    static std::map<int, CUcontext> primary;
    std::lock_guard<std::mutex> g(m);
    auto it = primary.find(ordinal);
    if(it != primary.end()) return it->second;
    CUdevice dev = 0;
    ck(a, a->cuDeviceGet(&dev, ordinal), "cuDeviceGet");
    ck(a, a->cuDevicePrimaryCtxRetain(&c, dev), "cuDevicePrimaryCtxRetain");
    primary[ordinal] = c;
    return c;
}

// A part of a read whose GPU work is under way: its slot, the event its work ends with, and what
// the check of its decompression needs (the chunk of each codec's streams)
struct Pending {
    Slot* slot = nullptr;
    CUevent done = nullptr;
    CUevent ev[4] = {nullptr, nullptr, nullptr, nullptr};   // (for CPPZARR_GPU_PROFILE)
    uint64_t decBytes = 0;
    std::vector<uint32_t> streamChunk[NCodecs];
};

// What CPPZARR_GPU_PROFILE prints about a read
struct ReadStats {
    uint64_t stored = 0, onHost = 0, streams = 0;
    size_t parts = 0;
    double find = 0, host = 0;
    float gpu[3] = {0, 0, 0};
};

}  // namespace

bool gpuReadAvailable(std::string &why){
    return api(&why) != nullptr;
}

void gpuSetNvcompLibrary(const std::string &path){
    setNvcompLibrary(path);
}

uint8_t gpuReadZarr(std::vector<GpuRegion> &regionsIn, const bool cOrder, void* stream, const int threadsIn,
                    std::string &errString, const bool float16AsFloat32){
    std::string why;
    const Api* a = api(&why);
    if(!a){
        errString = "GPU reads are not available: "+why;
        return 1;
    }
    if(regionsIn.empty()) return 0;
    CUcontext ctx = nullptr;
    bool pushed = false;
    Ctx* cx = nullptr;
    const int threads = threadsIn > 0 ? threadsIn : std::max(1, callThreads());
    // The files are read with at least CPPZARR_GPU_READS (32) threads, however few the call has:
    // reading mostly waits on the storage, and cold reads from network storage need many at once
    // (batches of 4 hypercubes of 128 MiB from VAST, 5 cores: 1.3 times as fast at 32 as at 5)
    static const int minReaders = [](){
        const char* e = std::getenv("CPPZARR_GPU_READS");
        return e ? std::max(1, std::atoi(e)) : 32;
    }();
    const int readers = std::max(threads, minReaders);
    bool hung = false;
    // (with CPPZARR_GPU_PROFILE set, each read prints where its time went to stderr)
    static const bool prof = std::getenv("CPPZARR_GPU_PROFILE") != nullptr;
    Plan P;
    ReadStats stats;
    std::deque<Pending> pending;
    // A part's end: waits for its GPU work, with a time limit (CPPZARR_GPU_TIMEOUT seconds, 0 for
    // none; by default 30 s and 1 s per 50 MB decoded: nvCOMP never finished on some damaged data,
    // and a call that waited for it would never return), checks its decompression, frees its slot
    auto finishPart = [&](Pending &q){
        double limit = 30.0+(double)q.decBytes/50e6;
        if(const char* e = std::getenv("CPPZARR_GPU_TIMEOUT")) limit = std::atof(e);
        const auto w0 = std::chrono::steady_clock::now();
        CUresult r;
        int polls = 0;
        while((r = a->cuEventQuery(q.done)) == 600 /* CUDA_ERROR_NOT_READY */){
            if(limit > 0 && std::chrono::duration<double>(std::chrono::steady_clock::now()-w0).count() > limit){
                hung = true;
                throw Fail("The GPU did not finish decompressing within "+std::to_string((int)limit)+" s: a chunk of the batch "
                           "may be damaged (nvCOMP can hang on damaged data; the GPU stays busy until this process ends)");
            }
            if(++polls > 1000) std::this_thread::sleep_for(std::chrono::microseconds(50));
            else std::this_thread::yield();
        }
        if(r) throw Fail("The GPU work failed ("+errorText(a, r)+"): if this is an illegal memory access, a chunk of the "
                         "batch may be damaged (nvCOMP can fail that way on damaged data)");
        if(q.ev[0]){
            for(int k = 0; k < 3; k++){
                float g = 0;
                a->cuEventElapsedTime(&g, q.ev[k], q.ev[k+1]);
                stats.gpu[k] += g;
            }
        }
        for(CUevent &e : q.ev) if(e){ a->cuEventDestroy(e); e = nullptr; }
        a->cuEventDestroy(q.done);
        q.done = nullptr;
        if(!q.slot) return;
        for(int k = 0; k < NCodecs; k++){
            uint64_t bad, first;
            memcpy(&bad, q.slot->small+16*k, 8);
            memcpy(&first, q.slot->small+16*k+8, 8);
            if(!q.streamChunk[k].empty() && bad){
                const Chunk &c = P.chunks[q.streamChunk[k][first < q.streamChunk[k].size() ? first : 0]];
                static const char* names[NCodecs] = {"zstd", "lz4", "deflate", "gzip", "snappy"};
                throw Fail("The chunk "+*c.path+" could not be decompressed on the GPU ("+names[k]+", "+std::to_string(bad)+
                           " streams of the batch failed): it may be damaged");
            }
        }
        std::lock_guard<std::mutex> g(cx->m);
        q.slot->busy = false;
        q.slot = nullptr;
    };
    // After an error: the GPU work queued so far ends before its slots are reused (unless it never
    // finishes: their buffers are then never reused), and the context is popped
    auto fail = [&](){
        if(pushed && !hung) a->cuStreamSynchronize((CUstream)stream);
        if(cx){
            std::lock_guard<std::mutex> g(cx->m);
            for(auto &q : pending){
                if(!q.slot) continue;
                if(hung) q.slot->lost = true;
                else q.slot->busy = false;
            }
        }
        for(auto &q : pending){
            for(CUevent e : q.ev) if(e) a->cuEventDestroy(e);
            if(q.done) a->cuEventDestroy(q.done);
        }
        pending.clear();
        if(pushed){ CUcontext c; a->cuCtxPopCurrent(&c); }
    };
    try{
        // the outputs' context (the same for all), made current for the call
        for(const auto &g : regionsIn){
            unsigned int mt = 0;
            if(a->cuPointerGetAttribute(&mt, 2 /* CU_POINTER_ATTRIBUTE_MEMORY_TYPE */, (CUdeviceptr)(uintptr_t)g.out) ||
               (mt != 2 /* DEVICE */ && mt != 4 /* UNIFIED */)){
                throw Fail("The output for "+g.array->get_fileName()+" is not GPU memory");
            }
            const CUcontext c = memoryContext(a, g.out);
            if(ctx && c != ctx) throw Fail("All outputs of one GPU read must be on the same GPU");
            ctx = c;
        }
        ck(a, a->cuCtxPushCurrent(ctx), "cuCtxPushCurrent");
        pushed = true;
        cx = &context(a, ctx);
        std::vector<bool> gpuCodec(NCodecs);
        for(int k = 0; k < NCodecs; k++) gpuCodec[k] = a->decompressAsync[k] != nullptr;

        // ---- what to read: every region's chunks, from their shards' indexes or as files
        const auto t0 = std::chrono::steady_clock::now();
        auto now = [&](){ return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-t0).count(); };
        planRegions(regionsIn, cOrder, float16AsFloat32, P);
        planFiles(P);
        std::string err = pool().run((int64_t)P.files.size(), readers, [&](int64_t i){
            findChunks(*P.files[i], P.regions[P.files[i]->region]);
        });
        if(!err.empty()) throw Fail(err);
        for(const auto &f : P.files) if(!f->error.empty()) throw Fail("The shard "+f->path+" "+f->error+" (it may be damaged, or not match the array's metadata)");
        // (in file and offset order)
        for(auto &f : P.files){
            typedef std::tuple<std::vector<uint64_t>, uint64_t, uint64_t> Found;
            std::sort(f->found.begin(), f->found.end(), [](const Found &x, const Found &y){ return std::get<1>(x) < std::get<1>(y); });
            for(const auto &e : f->found){
                P.chunks.push_back({f->region, std::get<0>(e), &f->path, std::get<1>(e), std::get<2>(e)});
                P.regions[f->region].present++;
                stats.stored += std::get<2>(e);
            }
        }
        stats.find = now();
        // the regions' groups: how the GPU places their blocks (data type size, chunk shape and
        // axis order, byte swapping, widening)
        std::map<std::tuple<uint64_t, std::vector<uint64_t>, std::vector<uint64_t>, int, int>, uint64_t> groups;
        for(auto &r : P.regions){
            auto key = std::make_tuple(r.ts, r.inner, r.axes, r.swapUnit, r.widen);
            auto it = groups.find(key);
            if(it == groups.end()) it = groups.emplace(key, groups.size()).first;
            r.group = it->second;
        }
        const uint64_t ndim = P.regions[0].ndim, nr = P.regions.size();
        for(const auto &r : P.regions) if(r.ndim != ndim) throw Fail("All arrays of one GPU read must have the same number of dimensions");
        // the tables of the regions and groups, the same for every part
        std::vector<uint64_t> rout(nr);
        std::vector<int64_t> rshape(nr*ndim), rstride(nr*ndim);
        for(size_t i = 0; i < nr; i++){
            const Region &r = P.regions[i];
            rout[i] = (uint64_t)(uintptr_t)regionsIn[i].out;
            int64_t acc = 1;
            for(uint64_t k = 0; k < ndim; k++){
                const uint64_t d = cOrder ? ndim-1-k : k;
                rshape[i*ndim+d] = (int64_t)r.rshape[d];
                rstride[i*ndim+d] = acc;
                acc *= (int64_t)r.rshape[d];
            }
        }
        // (per group: the chunk's dimensions in storage order and the array axis of each)
        std::vector<int64_t> gdims(groups.size()*ndim);
        std::vector<int32_t> gaxes(groups.size()*ndim);
        for(const auto &kv : groups){
            const auto &inner = std::get<1>(kv.first);
            const auto &axes = std::get<2>(kv.first);
            for(uint64_t k = 0; k < ndim; k++){
                gdims[kv.second*ndim+k] = (int64_t)inner[axes[k]];
                gaxes[kv.second*ndim+k] = (int32_t)axes[k];
            }
        }
        CUstream s = (CUstream)stream;
        // regions with chunks that were not found start as the fill value
        for(size_t i = 0; i < nr; i++){
            const Region &r = P.regions[i];
            if(r.present >= r.nChunks) continue;
            uint64_t elems = 1;
            for(auto v : r.rshape) elems *= v;
            bool same = true;
            for(uint64_t j = 1; j < r.outTs; j++) same = same && r.fill[j] == r.fill[0];
            if(same){
                ck(a, a->cuMemsetD8Async((CUdeviceptr)rout[i], r.fill[0], elems*r.outTs, s), "cuMemsetD8Async");
            }
            else{
                uint64_t lo = 0, hi = 0;
                memcpy(&lo, r.fill, 8);
                memcpy(&hi, r.fill+8, 8);
                CUdeviceptr o = (CUdeviceptr)rout[i];
                int tsv = (int)r.outTs;
                void* args[] = {&o, &elems, &tsv, &lo, &hi};
                ck(a, a->cuLaunchKernel(cx->fill, (unsigned)std::max<uint64_t>(1, std::min<uint64_t>((elems+255)/256, 32*(uint64_t)cx->sms)), 1, 1,
                                        256, 1, 1, 0, s, args, nullptr), "cuLaunchKernel (fill)");
            }
        }

        // ---- the parts: up to 4 runs of consecutive chunks of at least CPPZARR_GPU_PART_MB (32) MB
        // of stored bytes each, each in a slot of its own (staging and GPU buffers). A part is read
        // and set up on the host while the GPU decompresses and places the one before it: a call's
        // reading overlaps its GPU work, and its GPU buffers hold a part of it, not all of it. (Each
        // part costs the GPU about 1.5 ms more on a TITAN V, nvCOMP's own: 10 parts of a 4x128 MiB
        // read took longer than 4, and a 10x8 MiB read was faster in one.)
        double partMB = 32;
        if(const char* e = std::getenv("CPPZARR_GPU_PART_MB")) partMB = std::atof(e);
        const uint64_t partBytes = std::max<uint64_t>({1, (uint64_t)(partMB*1e6), (stats.stored+3)/4});
        std::vector<std::pair<size_t, size_t>> parts;
        {
            size_t c0 = 0;
            uint64_t acc = 0;
            for(size_t i = 0; i < P.chunks.size(); i++){
                acc += P.chunks[i].bytes;
                if(acc >= partBytes){
                    parts.push_back({c0, i+1});
                    c0 = i+1;
                    acc = 0;
                }
            }
            if(c0 < P.chunks.size()) parts.push_back({c0, P.chunks.size()});
        }
        stats.parts = parts.size();
        for(const auto &part : parts){
            if(pending.size() == 2){
                finishPart(pending.front());
                pending.pop_front();
            }
            const double h0 = now();
            pending.emplace_back();
            Pending &q = pending.back();
            q.slot = &takeSlot(*cx);
            Slot &slot = *q.slot;
            const size_t c0 = part.first, nc = part.second-part.first;
            // runs of the part's chunks in each file (gaps up to 64 KiB read too), laid out in the
            // staging buffer; a chunk's at becomes its place there
            struct Run { const std::string* path; uint64_t off, bytes, at; };
            std::vector<Run> runs;
            std::vector<size_t> runOf(nc);
            for(size_t i = 0; i < nc; i++){
                const Chunk &c = P.chunks[c0+i];
                if(!runs.empty() && runs.back().path == c.path && c.at >= runs.back().off && c.at <= runs.back().off+runs.back().bytes+(64 << 10))
                    runs.back().bytes = std::max(runs.back().bytes, c.at+c.bytes-runs.back().off);
                else runs.push_back({c.path, c.at, c.bytes, 0});
                runOf[i] = runs.size()-1;
            }
            uint64_t stored = 0;
            for(auto &r : runs){ r.at = stored; stored += al16(r.bytes); }
            for(size_t i = 0; i < nc; i++){
                Chunk &c = P.chunks[c0+i];
                const Run &r = runs[runOf[i]];
                c.at = r.at+(c.at-r.off);
            }
            growHost(a, slot, stored+(1 << 20), 0);
            // the reads, in pieces of at most 4 MiB (so that every thread shares big runs), and of
            // 1 MiB (the NFS client's largest read) in runs under 8 MiB, so that small runs keep
            // more reads going: single time points of 128^3 x 2 hypercubes (5 MB runs) read cold
            // from VAST up to 1.14 times as fast in 1 MiB pieces, but 4 processes reading 42 MB
            // runs at once were 1.36 times as slow in them
            struct Piece { size_t run; uint64_t off, bytes; };
            std::vector<Piece> pieces;
            for(size_t i = 0; i < runs.size(); i++){
                const uint64_t pieceBytes = runs[i].bytes < (8 << 20) ? (1 << 20) : (4 << 20);
                for(uint64_t o = 0; o < runs[i].bytes; o += pieceBytes) pieces.push_back({i, o, std::min(pieceBytes, runs[i].bytes-o)});
            }
            err = pool().run((int64_t)pieces.size(), readers, [&](int64_t i){
                const Piece &p = pieces[i];
                const Run &r = runs[p.run];
                File f;
                if(!f.open(*r.path) || !f.readAt(r.off+p.off, slot.host+r.at+p.off, p.bytes))
                    throw Fail("The file "+*r.path+" could not be read");
            });
            if(!err.empty()) throw Fail(err);

            // the chunks' blocks and streams (their chunk: the chunk's index in the part); chunks
            // the GPU cannot decode are decoded here
            std::vector<std::vector<Block>> cb(nc);
            std::vector<std::vector<Stream>> cs(nc);
            std::vector<uint8_t> onHost(nc, 0);
            err = pool().run((int64_t)nc, threads, [&](int64_t i){
                Chunk &c = P.chunks[c0+i];
                const Region &r = P.regions[c.region];
                const uint8_t* p = slot.host+c.at;
                uint64_t n = c.bytes;
                if(r.checksum){
                    uint32_t want = 0;
                    if(n >= 4) memcpy(&want, p+n-4, 4);
                    if(n < 4 || want != crc32c(p, (size_t)n-4)) throw Fail("The chunk "+*c.path+" does not match its checksum (it may be damaged)");
                    n -= 4;
                    c.bytes = n;
                }
                if(r.compressor == "none"){
                    if(n != r.chunkBytes) throw Fail("Uncompressed chunk is "+std::to_string(n)+" bytes instead of "+std::to_string(r.chunkBytes)+": "+*c.path);
                    cb[i].push_back({(uint32_t)i, 0, r.chunkElems, r.chunkBytes, false, (int64_t)c.at});
                }
                else if(r.compressor == "zstd" && gpuCodec[Zstd] && zstdFrameOk(p, n, r.chunkBytes)){
                    cb[i].push_back({(uint32_t)i, 0, r.chunkElems, r.chunkBytes, false, -1});
                    cs[i].push_back({0, Zstd, c.at, n, r.chunkBytes, 0});
                }
                else if(r.compressor == "gzip" && n >= 18 && p[0] == 0x1f && p[1] == 0x8b && gpuCodec[Gzip]){
                    cb[i].push_back({(uint32_t)i, 0, r.chunkElems, r.chunkBytes, false, -1});
                    cs[i].push_back({0, Gzip, c.at, n, r.chunkBytes, 0});
                }
                else if(r.compressor == "gzip" && n >= 6 && (p[0] & 0x0f) == 8 && ((p[0] << 8) | p[1]) % 31 == 0 && !(p[1] & 0x20) && gpuCodec[Deflate]){
                    // (a zlib stream: its deflate data, between a 2-byte header and an Adler-32)
                    cb[i].push_back({(uint32_t)i, 0, r.chunkElems, r.chunkBytes, false, -1});
                    cs[i].push_back({0, Deflate, c.at+2, n-6, r.chunkBytes, 0});
                }
                else if(r.compressor == "blosc" && bloscPlan(p, c.at, n, (uint32_t)i, r.chunkBytes, r.ts, cb[i], cs[i], gpuCodec)){
                }
                else{
                    cb[i].clear();
                    cs[i].clear();
                    onHost[i] = 1;
                }
            });
            if(!err.empty()) throw Fail(err);
            // chunks decoded on the host go after the stored bytes
            uint64_t hostAt = stored;
            std::vector<uint64_t> hostPlace(nc, 0);
            for(size_t i = 0; i < nc; i++){
                if(!onHost[i]) continue;
                hostPlace[i] = hostAt;
                hostAt += al16(P.regions[P.chunks[c0+i].region].chunkBytes);
                stats.onHost++;
            }
            if(hostAt > stored){
                growHost(a, slot, hostAt+(1 << 20), stored);
                err = pool().run((int64_t)nc, threads, [&](int64_t i){
                    if(!onHost[i]) return;
                    const Chunk &c = P.chunks[c0+i];
                    const Region &r = P.regions[c.region];
                    std::string w;
                    if(decodeChunk(r.compressor, false, slot.host+c.at, c.bytes, slot.host+hostPlace[i], r.chunkBytes, w))
                        throw Fail("The chunk "+*c.path+" "+w);
                });
                if(!err.empty()) throw Fail(err);
                for(size_t i = 0; i < nc; i++){
                    if(!onHost[i]) continue;
                    const Region &r = P.regions[P.chunks[c0+i].region];
                    cb[i].push_back({(uint32_t)i, 0, r.chunkElems, r.chunkBytes, false, (int64_t)hostPlace[i]});
                }
            }
            // one list of the part's blocks, group by group, and of its streams
            std::vector<Block> blocks;
            std::vector<Stream> streams;
            std::vector<uint64_t> groupStart(groups.size()+1, 0);
            for(uint64_t gi = 0; gi < groups.size(); gi++){
                groupStart[gi] = blocks.size();
                for(size_t i = 0; i < nc; i++){
                    if(P.regions[P.chunks[c0+i].region].group != gi) continue;
                    for(auto &st : cs[i]){
                        Stream t = st;
                        t.block = (uint32_t)(blocks.size()+st.block);
                        streams.push_back(t);
                    }
                    for(auto &b : cb[i]) blocks.push_back(b);
                }
            }
            groupStart[groups.size()] = blocks.size();
            stats.streams += streams.size();
            // decoded blocks' places in the decode buffer
            std::vector<uint64_t> decAt(blocks.size(), 0);
            uint64_t decTotal = 0;
            for(size_t i = 0; i < blocks.size(); i++){
                if(blocks[i].storedAt >= 0) continue;
                decAt[i] = decTotal;
                decTotal += al16(blocks[i].bytes);
            }

            // ---- the tables the GPU reads, after the data in the staging buffer
            const uint64_t nb = blocks.size();
            std::vector<std::pair<const void*, uint64_t>> tables;
            std::vector<uint64_t> bsrc(nb), bfirst(nb);
            std::vector<uint32_t> bnel(nb), bchunk(nb);
            std::vector<uint8_t> bflags(nb);
            for(size_t i = 0; i < nb; i++){
                const Block &b = blocks[i];
                bsrc[i] = b.storedAt >= 0 ? ((uint64_t)b.storedAt | (uint64_t(1) << 63)) : decAt[i];
                bfirst[i] = b.first;
                if(b.nelem > 0xffffffffULL) throw Fail("A chunk block of more than 4G elements is not supported on the GPU");
                bnel[i] = (uint32_t)b.nelem;
                bchunk[i] = b.chunk;
                bflags[i] = b.shuffled ? 1 : 0;
            }
            std::vector<uint32_t> creg(nc);
            std::vector<int64_t> corg(nc*ndim);
            for(size_t i = 0; i < nc; i++){
                const Chunk &c = P.chunks[c0+i];
                const Region &r = P.regions[c.region];
                creg[i] = c.region;
                for(uint64_t d = 0; d < ndim; d++) corg[i*ndim+d] = (int64_t)(c.grid[d]*r.inner[d])-(int64_t)r.start[d];
            }
            // per codec: the streams' input offsets (in the copy of the staging buffer), sizes,
            // decoded sizes and output offsets (in the decode buffer)
            std::vector<std::vector<uint64_t>> sin(NCodecs), sbytes(NCodecs), sout(NCodecs), scap(NCodecs);
            // (a stream that does not start where nvCOMP needs it to, deflate's at a multiple of 4
            // bytes, is copied to a place that does, after the chunks decoded on the host)
            struct Copy { uint64_t from, to, bytes; };
            std::vector<Copy> copies, moves;
            uint64_t dataEnd = hostAt;
            for(size_t i = 0; i < streams.size(); i++){
                const Stream &st = streams[i];
                const uint64_t to = decAt[st.block]+st.outOffset;
                if(st.codec < 0){ copies.push_back({st.at, to, st.bytes}); continue; }
                uint64_t at = st.at;
                const uint64_t need = std::max<uint64_t>(cx->align[st.codec].input, 1);
                if(at % need){
                    const uint64_t al = std::max<uint64_t>(need, 16);
                    at = (dataEnd+al-1)/al*al;
                    moves.push_back({st.at, at, st.bytes});
                    dataEnd = at+st.bytes;
                }
                sin[st.codec].push_back(at);
                sbytes[st.codec].push_back(st.bytes);
                scap[st.codec].push_back(st.outBytes);
                sout[st.codec].push_back(to);
                q.streamChunk[st.codec].push_back((uint32_t)(c0+blocks[st.block].chunk));
            }
            if(!moves.empty()){
                growHost(a, slot, dataEnd, hostAt);
                for(const auto &m : moves) memcpy(slot.host+m.to, slot.host+m.from, m.bytes);
            }
            uint64_t tab = al16(dataEnd);
            std::map<std::string, uint64_t> where;
            auto put = [&](const std::string &k, const void* p, uint64_t n){ tables.push_back({p, n}); where[k] = tab; tab = al16(tab+n); };
            put("bsrc", bsrc.data(), 8*nb); put("bfirst", bfirst.data(), 8*nb); put("bnel", bnel.data(), 4*nb);
            put("bchunk", bchunk.data(), 4*nb); put("bflags", bflags.data(), nb); put("creg", creg.data(), 4*nc);
            put("corg", corg.data(), 8*nc*ndim); put("rout", rout.data(), 8*nr); put("rshape", rshape.data(), 8*nr*ndim);
            put("rstride", rstride.data(), 8*nr*ndim); put("gdims", gdims.data(), 8*gdims.size()); put("gaxes", gaxes.data(), 4*gaxes.size());
            for(int k = 0; k < NCodecs; k++){
                if(sin[k].empty()) continue;
                const std::string n = std::to_string(k);
                put("in"+n, sin[k].data(), 8*sin[k].size()); put("bytes"+n, sbytes[k].data(), 8*sbytes[k].size());
                put("cap"+n, scap[k].data(), 8*scap[k].size()); put("out"+n, sout[k].data(), 8*sout[k].size());
            }
            const uint64_t total = tab;
            growHost(a, slot, total, dataEnd);
            {
                uint64_t t = al16(dataEnd);
                for(const auto &p : tables){ if(p.second) memcpy(slot.host+t, p.first, p.second); t = al16(t+p.second); }
            }

            // ---- the part's GPU work
            {
                size_t freeB = 0, totalB = 0;
                a->cuMemGetInfo(&freeB, &totalB);
                const uint64_t need = (total > slot.compBytes ? total : 0)+(decTotal > slot.decBytes ? decTotal : 0);
                if(freeB && need > freeB)
                    throw Fail("Not enough GPU memory for this read ("+std::to_string(need >> 20)+" MiB more needed, "+
                               std::to_string(freeB >> 20)+" MiB free): read fewer regions at a time");
            }
            growDevice(a, slot.comp, slot.compBytes, total);
            growDevice(a, slot.dec, slot.decBytes, std::max<uint64_t>(decTotal, 16));
            uint64_t nmax = 1;
            for(int k = 0; k < NCodecs; k++) nmax = std::max<uint64_t>(nmax, sin[k].size());
            // per codec: input and output pointer arrays, decoded sizes, statuses; then the checks
            const uint64_t perCodec = al16(8*nmax)*3+al16(4*nmax);
            growDevice(a, slot.work, slot.workBytes, perCodec*NCodecs+16*NCodecs+64);
            if(!slot.small) ck(a, a->cuMemHostAlloc((void**)&slot.small, 4096, 0), "cuMemHostAlloc");
            const CUdeviceptr comp = slot.comp, dec = slot.dec, work = slot.work;
            const CUdeviceptr checks = work+perCodec*NCodecs;
            if(prof) for(CUevent &e : q.ev) ck(a, a->cuEventCreate(&e, 0), "cuEventCreate");
            if(prof) a->cuEventRecord(q.ev[0], s);
            ck(a, a->cuMemcpyHtoDAsync(comp, slot.host, total, s), "cuMemcpyHtoDAsync");
            for(const auto &cpy : copies) ck(a, a->cuMemcpyDtoDAsync(dec+cpy.to, comp+cpy.from, cpy.bytes, s), "cuMemcpyDtoDAsync");
            if(prof) a->cuEventRecord(q.ev[1], s);
            ck(a, a->cuMemsetD8Async(checks, 0, 16*NCodecs, s), "cuMemsetD8Async");
            for(int k = 0; k < NCodecs; k++){
                const uint64_t n = sin[k].size();
                if(!n) continue;
                const std::string kn = std::to_string(k);
                const CUdeviceptr inp = work+perCodec*k, outp = inp+al16(8*nmax), got = outp+al16(8*nmax), st = got+al16(8*nmax);
                const CUdeviceptr ck_ = checks+16*k;
                // (the first bad stream's index starts as the largest: atomicMin lowers it)
                ck(a, a->cuMemsetD8Async(ck_+8, 0xff, 8, s), "cuMemsetD8Async");
                CUdeviceptr tin = comp+where["in"+kn], tout = comp+where["out"+kn];
                CUdeviceptr pin = inp, pout = outp;
                uint64_t nn = n;
                void* rargs[] = {&tin, &tout, &pin, &pout, (void*)&comp, (void*)&dec, &nn};
                ck(a, a->cuLaunchKernel(cx->resolve, (unsigned)std::min<uint64_t>((n+255)/256, 1024), 1, 1, 256, 1, 1, 0, s, rargs, nullptr), "cuLaunchKernel (resolve)");
                uint64_t maxOut = *std::max_element(scap[k].begin(), scap[k].end()), totOut = 0;
                for(auto v : scap[k]) totOut += v;
                NvcompOpts o{};
                size_t tb = 0;
                const int tr = a->tempSize6[k] ? a->tempSize6[k](n, maxOut, o, &tb, totOut, s) : a->tempSize5[k](n, maxOut, o, &tb, totOut);
                if(tr) throw Fail("nvCOMP could not size its work memory (status "+std::to_string(tr)+")");
                growDevice(a, slot.temp, slot.tempBytes, std::max<size_t>(tb, 16));
                const int dr = a->decompressAsync[k]((const void* const*)(uintptr_t)inp, (const size_t*)(uintptr_t)(comp+where["bytes"+kn]),
                                                     (const size_t*)(uintptr_t)(comp+where["cap"+kn]), (size_t*)(uintptr_t)got, n,
                                                     (void*)(uintptr_t)slot.temp, slot.tempBytes, (void* const*)(uintptr_t)outp, o,
                                                     (int*)(uintptr_t)st, s);
                if(dr) throw Fail("nvCOMP could not start decompressing (status "+std::to_string(dr)+", "+a->nvcompPath+")"+
                                  (dr == 10 ? ": that nvCOMP may not run on this GPU (CUDA 13 builds need a Turing or newer one; "
                                              "pip install nvidia-libnvcomp-cu12, or set CPPZARR_NVCOMP_LIBRARY)" : ""));
                CUdeviceptr pst = st, pgot = got, pcap = comp+where["cap"+kn], pck = ck_;
                void* cargs[] = {&pst, &pgot, &pcap, &nn, &pck};
                ck(a, a->cuLaunchKernel(cx->check, (unsigned)std::min<uint64_t>((n+255)/256, 1024), 1, 1, 256, 1, 1, 0, s, cargs, nullptr), "cuLaunchKernel (check)");
            }
            if(prof) a->cuEventRecord(q.ev[2], s);
            ck(a, a->cuMemcpyDtoHAsync(slot.small, checks, 16*NCodecs, s), "cuMemcpyDtoHAsync");
            // place every block, group by group
            for(const auto &kv : groups){
                const uint64_t gi = kv.second, b0 = groupStart[gi], b1 = groupStart[gi+1];
                if(b1 == b0) continue;
                const uint64_t ts = std::get<0>(kv.first);
                const auto &inner = std::get<1>(kv.first);
                uint64_t ce = 1;
                for(auto v : inner) ce *= v;
                CUdeviceptr pbsrc = comp+where["bsrc"]+8*b0, pbnel = comp+where["bnel"]+4*b0, pbfirst = comp+where["bfirst"]+8*b0;
                CUdeviceptr pbchunk = comp+where["bchunk"]+4*b0, pbflags = comp+where["bflags"]+b0, pcreg = comp+where["creg"];
                CUdeviceptr pcorg = comp+where["corg"], prout = comp+where["rout"], prshape = comp+where["rshape"];
                CUdeviceptr prstride = comp+where["rstride"], pgdims = comp+where["gdims"]+8*gi*ndim, pgaxes = comp+where["gaxes"]+4*gi*ndim;
                CUdeviceptr phost = comp, pdec = dec;
                int nd = (int)ndim, tsv = (int)ts, sw = std::get<3>(kv.first), wd = std::get<4>(kv.first);
                void* args[] = {&pbsrc, &pbnel, &pbfirst, &pbchunk, &pbflags, &pcreg, &pcorg, &prout, &prshape, &prstride,
                                &pgdims, &pgaxes, &phost, &pdec, &nd, &tsv, &sw, &wd};
                const uint64_t nblk = b1-b0;
                if(nblk > 0x7fffffffULL) throw Fail("Too many chunk blocks for one GPU read: read fewer regions at a time");
                ck(a, a->cuLaunchKernel(ce < 0xffffffffULL ? cx->scatter32 : cx->scatter64, (unsigned)nblk, 1, 1, 256, 1, 1, 0, s, args, nullptr),
                   "cuLaunchKernel (scatter)");
            }
            if(prof) a->cuEventRecord(q.ev[3], s);
            ck(a, a->cuEventCreate(&q.done, 0x2 /* CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
            ck(a, a->cuEventRecord(q.done, s), "cuEventRecord");
            q.decBytes = decTotal;
            stats.host += now()-h0;
        }
        // (a read with no chunks at all: only the fills, which the call also waits for)
        if(parts.empty()){
            pending.emplace_back();
            ck(a, a->cuEventCreate(&pending.back().done, 0x2 /* CU_EVENT_DISABLE_TIMING */), "cuEventCreate");
            ck(a, a->cuEventRecord(pending.back().done, s), "cuEventRecord");
        }
        while(!pending.empty()){
            finishPart(pending.front());
            pending.pop_front();
        }
        if(prof){
            uint64_t outBytes = 0;
            for(const auto &r : P.regions){
                uint64_t n = r.outTs;
                for(auto v : r.rshape) n *= v;
                outBytes += n;
            }
            fprintf(stderr, "cppzarr GPU read: %zu regions, %zu chunks (%llu decoded on the host), %llu streams in %zu parts, %.1f MB "
                    "read, %.1f MB out | find %.1f ms, then reading and setting up %.1f ms | on the GPU: copy %.1f, decompress %.1f, "
                    "place %.1f ms | done at %.1f ms\n", P.regions.size(), P.chunks.size(), (unsigned long long)stats.onHost,
                    (unsigned long long)stats.streams, stats.parts, stats.stored/1e6, outBytes/1e6, stats.find, stats.host,
                    stats.gpu[0], stats.gpu[1], stats.gpu[2], now());
        }
    }
    catch(const std::exception &e){
        errString = e.what();
        fail();
        return 1;
    }
    catch(const std::string &e){
        errString = e;
        fail();
        return 1;
    }
    if(pushed){ CUcontext c; a->cuCtxPopCurrent(&c); }
    return 0;
}
