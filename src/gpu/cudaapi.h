// The CUDA driver API and nvCOMP, loaded when a GPU read first needs them (internal to the
// library, not installed). cpp-zarr builds and runs without either: only GPU reads need them.
// (The driver API works in whatever context the caller's GPU memory belongs to, the one CUDA
// runtime users such as PyTorch, CuPy and MATLAB share, whatever CUDA version they use.)
#ifndef CPPZARR_CUDAAPI_H
#define CPPZARR_CUDAAPI_H
#include <cstddef>
#include <cstdint>
#include <string>

namespace czcuda {

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef void* CUevent;
typedef unsigned long long CUdeviceptr;

// nvCOMP's decompression options for each of its codecs: 64 bytes, all zeros the defaults
struct NvcompOpts { int backend; int a; int b; char reserved[52]; };
static_assert(sizeof(NvcompOpts) == 64, "nvCOMP's options are 64 bytes");
struct NvcompAlignments { size_t input, output, temp; };

// The codecs nvCOMP decompresses for cpp-zarr
enum Codec { Zstd = 0, LZ4 = 1, Deflate = 2, Gzip = 3, Snappy = 4, NCodecs = 5 };

struct Api {
    // driver
    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDriverGetVersion)(int*);
    CUresult (*cuGetErrorString)(CUresult, const char**);
    CUresult (*cuPointerGetAttribute)(void*, int, CUdeviceptr);
    CUresult (*cuCtxGetCurrent)(CUcontext*);
    CUresult (*cuCtxSetCurrent)(CUcontext);
    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuDevicePrimaryCtxRetain)(CUcontext*, CUdevice);
    CUresult (*cuCtxPushCurrent)(CUcontext);
    CUresult (*cuCtxPopCurrent)(CUcontext*);
    CUresult (*cuCtxGetDevice)(CUdevice*);
    CUresult (*cuDeviceGetAttribute)(int*, int, CUdevice);
    CUresult (*cuModuleLoadDataEx)(CUmodule*, const void*, unsigned int, int*, void**);
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, CUstream, void**, void**);
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemGetInfo)(size_t*, size_t*);
    CUresult (*cuMemHostAlloc)(void**, size_t, unsigned int);
    CUresult (*cuMemFreeHost)(void*);
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
    CUresult (*cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
    CUresult (*cuMemcpyDtoDAsync)(CUdeviceptr, CUdeviceptr, size_t, CUstream);
    CUresult (*cuMemsetD8Async)(CUdeviceptr, unsigned char, size_t, CUstream);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuStreamWaitEvent)(CUstream, CUevent, unsigned int);
    CUresult (*cuEventCreate)(CUevent*, unsigned int);
    CUresult (*cuEventRecord)(CUevent, CUstream);
    CUresult (*cuEventSynchronize)(CUevent);
    CUresult (*cuEventQuery)(CUevent);
    CUresult (*cuEventDestroy)(CUevent);
    CUresult (*cuEventElapsedTime)(float*, CUevent, CUevent);
    // nvCOMP, per codec (null for one the loaded nvCOMP does not have)
    int (*decompressAsync[NCodecs])(const void* const*, const size_t*, const size_t*, size_t*, size_t, void*, size_t,
                                    void* const*, NvcompOpts, int*, void*);
    // (nvCOMP 6: ...GetTempSize, which takes a stream; nvCOMP 5: ...GetTempSizeAsync, which does not)
    int (*tempSize6[NCodecs])(size_t, size_t, NvcompOpts, size_t*, size_t, void*);
    int (*tempSize5[NCodecs])(size_t, size_t, NvcompOpts, size_t*, size_t);
    int (*alignments[NCodecs])(NvcompOpts, NvcompAlignments*);
    int nvcompMajor = 0;
    std::string nvcompPath;
};

// The loaded API, or nullptr with the reason in why (no NVIDIA driver, no nvCOMP, ...). Loads once.
const Api* api(std::string* why);

// Where to load nvCOMP from (a libnvcomp file), before the first GPU read: bindings that know
// where a pip-installed nvCOMP is (the nvidia-libnvcomp-cu12 package) set it. Otherwise
// CPPZARR_NVCOMP_LIBRARY, then the system's library path.
void setNvcompLibrary(const std::string &path);

// A driver error's text
std::string errorText(const Api* a, CUresult r);

}  // namespace czcuda
#endif
