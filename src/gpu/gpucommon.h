// What GPU reads (gpureadzarr.cpp) and GPU writes (gpuwritezarr.cpp) share: host threads, each
// CUDA context's kernels and reusable buffers, and finding the context GPU memory belongs to.
// Internal to the library (not installed).
#ifndef CPPZARR_GPUCOMMON_H
#define CPPZARR_GPUCOMMON_H
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include "cudaapi.h"

namespace czcuda {

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

    static void work(Job &j);
    void worker();
public:
    ~Pool(){ for(auto &t : threads) t.detach(); }
    // Runs fn for every i; returns the first error an fn threw, or ""
    std::string run(const int64_t n, int width, const std::function<void(int64_t)> &fn);
};
Pool& pool();

// ---------------------------------------------------------------- GPU state
// A GPU read's part's reusable buffers
struct Slot {
    bool busy = false;
    bool lost = false;               // (its GPU work never finished: its buffers are never reused)
    uint8_t* host = nullptr;         // pinned staging: stored bytes, chunks decoded on the host, tables
    size_t hostBytes = 0;
    uint8_t* small = nullptr;        // pinned: the decompression checks' results
    CUdeviceptr comp = 0, dec = 0, work = 0, temp = 0;
    size_t compBytes = 0, decBytes = 0, workBytes = 0, tempBytes = 0;
    bool compRegistered = false;     // (comp registered with cuFile, which then reads into it directly)
};
// A GPU write's part's reusable buffers: pinned host memory for what goes to the GPU (tables and
// the current contents of chunks the write cuts) and what comes back (the compressed sizes, then
// the stored chunks); GPU memory for the chunks put together, compressed and stored
struct WriteSlot {
    bool busy = false;
    uint8_t* hin = nullptr;
    uint8_t* hres = nullptr;
    uint8_t* hops = nullptr;
    uint8_t* hout = nullptr;
    size_t hinBytes = 0, hresBytes = 0, hopsBytes = 0, houtBytes = 0;
    CUdeviceptr din = 0, unc = 0, comp = 0, work = 0, temp = 0, dops = 0, packed = 0;
    size_t dinBytes = 0, uncBytes = 0, compBytes = 0, workBytes = 0, tempBytes = 0, dopsBytes = 0, packedBytes = 0;
};
// The kernels and reusable buffers of one CUDA context
struct Ctx {
    CUcontext ctx = nullptr;
    CUmodule mod = nullptr;
    CUfunction scatter32 = nullptr, scatter64 = nullptr, resolve = nullptr, fill = nullptr, check = nullptr;
    CUfunction gather32 = nullptr, gather64 = nullptr, pack = nullptr;
    int sms = 1;
    NvcompAlignments align[NCodecs];
    CUstream copyStream = nullptr;   // (GDS reads: the stored bytes' copies to the host, on their own)
    std::mutex m;
    std::vector<std::unique_ptr<Slot>> slots;
    std::vector<std::unique_ptr<WriteSlot>> writeSlots;
};

struct Fail : std::runtime_error { using std::runtime_error::runtime_error; };

// Throws Fail (what: the driver's error) unless r is 0
void ck(const Api* a, CUresult r, const char* what);

// The context's kernels and buffers (loading its kernels the first time)
Ctx& context(const Api* a, CUcontext c);

// A read's or a write's slot of the context, free until it is given back (busy = false, under c.m)
Slot& takeSlot(Ctx &c);
WriteSlot& takeWriteSlot(Ctx &c);

// GPU memory of at least need bytes in p (have of them), its contents not kept
void growDevice(const Api* a, CUdeviceptr &p, size_t &have, const size_t need);

// Pinned host memory of at least need bytes in p (have of them), keeping its first keep bytes
void growPinned(const Api* a, uint8_t* &p, size_t &have, const size_t need, const size_t keep);

// The CUDA context GPU memory belongs to: the one it was allocated in, or for memory that belongs to
// none (allocated through CUDA's virtual memory API or a stream-ordered memory pool, as PyTorch's
// expandable segments and CuPy's async pool are) its device's primary context, the one the CUDA
// runtime uses
CUcontext memoryContext(const Api* a, const void* p);

// Whether p is GPU memory (device or managed)
bool isGpuMemory(const Api* a, const void* p);

// A thread with ctx current for a cuFile call, and its own context current again after it (cuFile's
// compatibility mode can leave another one, or none, current: the copies after the reads then
// failed with an invalid context)
struct CtxFor {
    const Api* a;
    CUcontext was = nullptr;
    CtxFor(const Api* api, CUcontext c) : a(api){ a->cuCtxGetCurrent(&was); a->cuCtxSetCurrent(c); }
    ~CtxFor(){ a->cuCtxSetCurrent(was); }
    CtxFor(const CtxFor&) = delete;
    CtxFor& operator=(const CtxFor&) = delete;
};

// 16-byte aligned
inline uint64_t al16(uint64_t n){ return (n+15) & ~uint64_t(15); }

}  // namespace czcuda
#endif
