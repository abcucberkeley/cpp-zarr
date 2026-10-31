#include "gpucommon.h"
#include <algorithm>
#include <cstring>
#include <map>
#include "kernels_ptx.h"

namespace czcuda {

// ---------------------------------------------------------------- host threads
void Pool::work(Job &j){
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

void Pool::worker(){
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

std::string Pool::run(const int64_t n, int width, const std::function<void(int64_t)> &fn){
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

Pool& pool(){
    static Pool* p = new Pool();   // (never destroyed: its threads may outlive static destruction)
    return *p;
}

// ---------------------------------------------------------------- GPU state
namespace {
std::mutex ctxMutex;
std::vector<std::unique_ptr<Ctx>> ctxs;
}  // namespace

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
    ck(a, a->cuModuleGetFunction(&x->gather32, x->mod, "cz_gather32"), "cuModuleGetFunction");
    ck(a, a->cuModuleGetFunction(&x->gather64, x->mod, "cz_gather64"), "cuModuleGetFunction");
    ck(a, a->cuModuleGetFunction(&x->pack, x->mod, "cz_pack"), "cuModuleGetFunction");
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

WriteSlot& takeWriteSlot(Ctx &c){
    std::lock_guard<std::mutex> g(c.m);
    for(auto &s : c.writeSlots) if(!s->busy){ s->busy = true; return *s; }
    c.writeSlots.emplace_back(new WriteSlot());
    c.writeSlots.back()->busy = true;
    return *c.writeSlots.back();
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

void growPinned(const Api* a, uint8_t* &p, size_t &have, const size_t need, const size_t keep){
    if(need <= have) return;
    const size_t n = std::max<size_t>(need+need/4, 1 << 20);
    void* q = nullptr;
    ck(a, a->cuMemHostAlloc(&q, n, 0), "Could not allocate pinned host memory (cuMemHostAlloc)");
    if(keep && p) memcpy(q, p, keep);
    if(p) a->cuMemFreeHost(p);
    p = static_cast<uint8_t*>(q);
    have = n;
}

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

bool isGpuMemory(const Api* a, const void* p){
    unsigned int mt = 0;
    return !a->cuPointerGetAttribute(&mt, 2 /* CU_POINTER_ATTRIBUTE_MEMORY_TYPE */, (CUdeviceptr)(uintptr_t)p) &&
           (mt == 2 /* DEVICE */ || mt == 4 /* UNIFIED */);
}

}  // namespace czcuda
