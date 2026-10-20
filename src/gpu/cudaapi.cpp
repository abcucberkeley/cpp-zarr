#include "cudaapi.h"
#include <cstdlib>
#include <mutex>
#include <vector>
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
#include <dlfcn.h>
#endif

namespace czcuda {
namespace {

void* openLibrary(const std::string &name){
#ifdef _WIN32
    return (void*)LoadLibraryA(name.c_str());
#else
    return dlopen(name.c_str(), RTLD_NOW | RTLD_LOCAL);
#endif
}

void* symbol(void* lib, const char* name){
#ifdef _WIN32
    return (void*)GetProcAddress((HMODULE)lib, name);
#else
    return dlsym(lib, name);
#endif
}

std::mutex loadMutex;
std::string nvcompPathSet;
bool tried = false;
Api loaded;
const Api* result = nullptr;
std::string failure;

template <typename F>
bool get(void* lib, F &f, const char* name){
    f = reinterpret_cast<F>(symbol(lib, name));
    return f != nullptr;
}

const char* const codecNames[NCodecs] = {"Zstd", "LZ4", "Deflate", "Gzip", "Snappy"};

bool loadDriver(Api &a, std::string &why){
#ifdef _WIN32
    void* lib = openLibrary("nvcuda.dll");
#else
    void* lib = openLibrary("libcuda.so.1");
#endif
    if(!lib){
        why = "no NVIDIA driver (libcuda) was found";
        return false;
    }
    bool ok = get(lib, a.cuInit, "cuInit") && get(lib, a.cuDriverGetVersion, "cuDriverGetVersion") &&
              get(lib, a.cuGetErrorString, "cuGetErrorString") && get(lib, a.cuPointerGetAttribute, "cuPointerGetAttribute") &&
              get(lib, a.cuCtxGetCurrent, "cuCtxGetCurrent") && get(lib, a.cuCtxSetCurrent, "cuCtxSetCurrent") &&
              get(lib, a.cuDeviceGet, "cuDeviceGet") && get(lib, a.cuDevicePrimaryCtxRetain, "cuDevicePrimaryCtxRetain") &&
              get(lib, a.cuCtxPushCurrent, "cuCtxPushCurrent_v2") &&
              get(lib, a.cuCtxPopCurrent, "cuCtxPopCurrent_v2") && get(lib, a.cuCtxGetDevice, "cuCtxGetDevice") &&
              get(lib, a.cuDeviceGetAttribute, "cuDeviceGetAttribute") && get(lib, a.cuModuleLoadDataEx, "cuModuleLoadDataEx") &&
              get(lib, a.cuModuleGetFunction, "cuModuleGetFunction") && get(lib, a.cuLaunchKernel, "cuLaunchKernel") &&
              get(lib, a.cuMemAlloc, "cuMemAlloc_v2") && get(lib, a.cuMemFree, "cuMemFree_v2") &&
              get(lib, a.cuMemGetInfo, "cuMemGetInfo_v2") && get(lib, a.cuMemHostAlloc, "cuMemHostAlloc") &&
              get(lib, a.cuMemFreeHost, "cuMemFreeHost") && get(lib, a.cuMemcpyHtoDAsync, "cuMemcpyHtoDAsync_v2") &&
              get(lib, a.cuMemcpyDtoHAsync, "cuMemcpyDtoHAsync_v2") && get(lib, a.cuMemcpyDtoDAsync, "cuMemcpyDtoDAsync_v2") &&
              get(lib, a.cuMemsetD8Async, "cuMemsetD8Async") && get(lib, a.cuStreamSynchronize, "cuStreamSynchronize") &&
              get(lib, a.cuStreamWaitEvent, "cuStreamWaitEvent") && get(lib, a.cuEventCreate, "cuEventCreate") &&
              get(lib, a.cuEventRecord, "cuEventRecord") && get(lib, a.cuEventSynchronize, "cuEventSynchronize") &&
              get(lib, a.cuEventQuery, "cuEventQuery") && get(lib, a.cuEventDestroy, "cuEventDestroy_v2") &&
              get(lib, a.cuEventElapsedTime, "cuEventElapsedTime");
    if(!ok){
        why = "the NVIDIA driver (libcuda) lacks functions cpp-zarr needs: it may be too old";
        return false;
    }
    const CUresult r = a.cuInit(0);
    if(r){
        const char* s = nullptr;
        a.cuGetErrorString(r, &s);
        why = std::string("the NVIDIA driver could not start (") + (s ? s : "error "+std::to_string(r)) + ")";
        return false;
    }
    return true;
}

bool loadNvcomp(Api &a, std::string &why){
    std::vector<std::string> names;
    if(!nvcompPathSet.empty()) names.push_back(nvcompPathSet);
    const char* env = std::getenv("CPPZARR_NVCOMP_LIBRARY");
    if(env && *env) names.push_back(env);
#ifdef _WIN32
    for(const char* n : {"nvcomp64_6.dll", "nvcomp_6.dll", "nvcomp64_5.dll", "nvcomp_5.dll", "nvcomp.dll"}) names.push_back(n);
#else
    for(const char* n : {"libnvcomp.so.6", "libnvcomp.so.5", "libnvcomp.so"}) names.push_back(n);
#endif
    void* lib = nullptr;
    for(const auto &n : names){
        lib = openLibrary(n);
        if(lib){
            a.nvcompPath = n;
            break;
        }
    }
    if(!lib){
        why = "nvCOMP was not found (pip install nvidia-libnvcomp-cu12, or set CPPZARR_NVCOMP_LIBRARY to its libnvcomp)";
        return false;
    }
    // nvCOMP 5 and later take options in their batched calls (nvCOMP 4 does not: refused)
    for(int c = 0; c < NCodecs; c++){
        const std::string n = std::string("nvcompBatched")+codecNames[c]+"Decompress";
        get(lib, a.alignments[c], (n+"GetRequiredAlignments").c_str());
        get(lib, a.decompressAsync[c], (n+"Async").c_str());
        if(!a.alignments[c]) a.decompressAsync[c] = nullptr;
        if(a.decompressAsync[c] && !get(lib, a.tempSize6[c], (n+"GetTempSize").c_str()) &&
           !get(lib, a.tempSize5[c], (n+"GetTempSizeAsync").c_str())){
            a.decompressAsync[c] = nullptr;
        }
        if(a.tempSize6[c]) a.nvcompMajor = 6;
        else if(a.tempSize5[c] && !a.nvcompMajor) a.nvcompMajor = 5;
    }
    if(!a.decompressAsync[Zstd]){
        why = "the nvCOMP found ("+a.nvcompPath+") is older than version 5, which cpp-zarr needs";
        return false;
    }
    return true;
}

}  // namespace

void setNvcompLibrary(const std::string &path){
    std::lock_guard<std::mutex> g(loadMutex);
    if(!tried) nvcompPathSet = path;
}

const Api* api(std::string* why){
    std::lock_guard<std::mutex> g(loadMutex);
    if(!tried){
        tried = true;
        Api a{};
        if(loadDriver(a, failure) && loadNvcomp(a, failure)){
            loaded = a;
            result = &loaded;
        }
    }
    if(!result && why) *why = failure;
    return result;
}

std::string errorText(const Api* a, CUresult r){
    const char* s = nullptr;
    if(a && a->cuGetErrorString) a->cuGetErrorString(r, &s);
    return s ? std::string(s) : "CUDA error "+std::to_string(r);
}

}  // namespace czcuda
