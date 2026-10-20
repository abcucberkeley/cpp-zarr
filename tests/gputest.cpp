// GPU reads (gpureadzarr.h) against the CPU reader, byte for byte: every data type, codec
// (blosc's, with and without shuffle, bitshuffle and memcpyed chunks; gzip; none), order,
// format, sharding, regions that cut chunks, missing chunks and fill values, batches of regions
// from several arrays in one call, and the arrays other Zarr writers made (tests/test_arrays);
// float16 and bfloat16 also read as float32s; and a damaged chunk, which must be reported. Exits 0 with a message when this machine has no
// NVIDIA GPU, driver or nvCOMP. Usage: gpuTest <scratch folder>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "gpu/cudaapi.h"
#include "gpureadzarr.h"
#include "helperfunctions.h"
#include "parallelreadzarr.h"
#include "parallelwritezarr.h"
#include "zarr.h"

namespace fs = std::filesystem;
using namespace czcuda;

static const Api* A = nullptr;
static int total = 0, failures = 0;

static void ck(CUresult r, const char* what){
    if(r){ fprintf(stderr, "%s: %s\n", what, errorText(A, r).c_str()); exit(2); }
}

// The region read on the CPU (the reference) and on the GPU, compared; prints a FAIL line
static bool compare(zarr &Z, const std::vector<uint64_t> &start, const std::vector<uint64_t> &end, const bool cOrder,
                    const std::string &label){
    total++;
    const DtypeInfo t = dtypeInfo(Z.get_dtype());
    uint64_t n = 1;
    std::vector<uint64_t> shape;
    for(size_t d = 0; d < start.size(); d++){ shape.push_back(end[d]-start[d]); n *= end[d]-start[d]; }
    const uint64_t bytes = n*t.bytes;
    std::vector<uint8_t> cpu(bytes), gpu(bytes, 0);
    uint8_t fill[16];
    fillValueElement(Z.get_fill_value(), Z.get_dtype(), fill);
    Z.set_chunkInfo(start, end);
    if(parallelReadZarr(Z, cpu.data(), start, end, shape, t.bytes*8, true, false, cOrder, fill)){
        printf("FAIL  %s: the CPU read failed: %s\n", label.c_str(), Z.get_errString().c_str());
        failures++;
        return false;
    }
    CUdeviceptr d = 0;
    ck(A->cuMemAlloc(&d, std::max<uint64_t>(bytes, 1)), "cuMemAlloc");
    ck(A->cuMemsetD8Async(d, 0xab, bytes, nullptr), "cuMemsetD8Async");   // (not the fill value: missing chunks must be filled)
    std::vector<GpuRegion> regions{{&Z, start, end, (void*)(uintptr_t)d}};
    std::string err;
    const uint8_t r = gpuReadZarr(regions, cOrder, nullptr, 4, err);
    if(!r){
        ck(A->cuMemcpyDtoHAsync(gpu.data(), d, bytes, nullptr), "cuMemcpyDtoHAsync");
        ck(A->cuStreamSynchronize(nullptr), "cuStreamSynchronize");
    }
    A->cuMemFree(d);
    if(r){
        printf("FAIL  %s: %s\n", label.c_str(), err.c_str());
        failures++;
        return false;
    }
    if(memcmp(cpu.data(), gpu.data(), bytes)){
        uint64_t i = 0;
        while(cpu[i] == gpu[i]) i++;
        printf("FAIL  %s: differs from the CPU read (first at byte %llu: %u vs %u)\n", label.c_str(), (unsigned long long)i, cpu[i], gpu[i]);
        failures++;
        return false;
    }
    return true;
}

// A float16 or bfloat16 region read as float32s (float16AsFloat32), against the CPU read's values
// widened on the host
static bool compareWidened(zarr &Z, const std::vector<uint64_t> &start, const std::vector<uint64_t> &end, const bool cOrder,
                           const std::string &label){
    total++;
    const DtypeInfo t = dtypeInfo(Z.get_dtype());
    uint64_t n = 1;
    std::vector<uint64_t> shape;
    for(size_t d = 0; d < start.size(); d++){ shape.push_back(end[d]-start[d]); n *= end[d]-start[d]; }
    std::vector<uint16_t> cpu(n);
    uint8_t fill[16];
    fillValueElement(Z.get_fill_value(), Z.get_dtype(), fill);
    Z.set_chunkInfo(start, end);
    if(parallelReadZarr(Z, cpu.data(), start, end, shape, 16, true, false, cOrder, fill)){
        printf("FAIL  %s: the CPU read failed: %s\n", label.c_str(), Z.get_errString().c_str());
        failures++;
        return false;
    }
    std::vector<float> want(n), got(n);
    for(uint64_t i = 0; i < n; i++) want[i] = t.bfloat16 ? bfloat16ToFloat(cpu[i]) : halfToFloat(cpu[i]);
    CUdeviceptr d = 0;
    ck(A->cuMemAlloc(&d, std::max<uint64_t>(4*n, 1)), "cuMemAlloc");
    ck(A->cuMemsetD8Async(d, 0xab, 4*n, nullptr), "cuMemsetD8Async");
    std::vector<GpuRegion> regions{{&Z, start, end, (void*)(uintptr_t)d}};
    std::string err;
    const uint8_t r = gpuReadZarr(regions, cOrder, nullptr, 4, err, true);
    if(!r){
        ck(A->cuMemcpyDtoHAsync(got.data(), d, 4*n, nullptr), "cuMemcpyDtoHAsync");
        ck(A->cuStreamSynchronize(nullptr), "cuStreamSynchronize");
    }
    A->cuMemFree(d);
    if(r){
        printf("FAIL  %s (as float32): %s\n", label.c_str(), err.c_str());
        failures++;
        return false;
    }
    if(memcmp(want.data(), got.data(), 4*n)){
        uint64_t i = 0;
        while(!memcmp(&want[i], &got[i], 4)) i++;
        printf("FAIL  %s (as float32): differs from the CPU read (first at element %llu)\n", label.c_str(), (unsigned long long)i);
        failures++;
        return false;
    }
    return true;
}

// Regions of an array: all of it, one inside a chunk, ones that cut chunks, a random one
static std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> regionsOf(zarr &Z, std::mt19937 &rng){
    const uint64_t n = Z.get_ndims();
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>> out;
    std::vector<uint64_t> zero(n, 0), shape(n), a(n), b(n), c(n), e(n);
    for(uint64_t d = 0; d < n; d++) shape[d] = Z.get_shape(d);
    out.push_back({zero, shape});
    if(n == 0) return out;
    for(uint64_t d = 0; d < n; d++){
        a[d] = std::min<uint64_t>(1, shape[d]-1);
        b[d] = std::max<uint64_t>(a[d]+1, std::min<uint64_t>(shape[d], a[d]+2));
        c[d] = shape[d]/3;
        e[d] = std::max<uint64_t>(c[d]+1, shape[d]-shape[d]/4);
    }
    out.push_back({a, b});
    out.push_back({c, e});
    std::vector<uint64_t> s(n), t(n);
    for(uint64_t d = 0; d < n; d++){
        s[d] = std::uniform_int_distribution<uint64_t>(0, shape[d]-1)(rng);
        t[d] = std::uniform_int_distribution<uint64_t>(s[d]+1, shape[d])(rng);
    }
    out.push_back({s, t});
    return out;
}

static void testArray(const std::string &path, const std::string &label, std::mt19937 &rng){
    zarr Z;
    try{ Z = zarr(path); }
    catch(const std::string &){ return; }   // (arrays cpp-zarr does not read)
    const auto regions = regionsOf(Z, rng);
    const DtypeInfo t = dtypeInfo(Z.get_dtype());
    const bool half = t.kind == 'f' && t.bytes == 2;
    for(size_t i = 0; i < regions.size(); i++)
        for(const bool cOrder : {false, true}){
            const std::string l = label+(cOrder ? " C" : " F")+" region "+std::to_string(i);
            compare(Z, regions[i].first, regions[i].second, cOrder, l);
            if(half) compareWidened(Z, regions[i].first, regions[i].second, cOrder, l);
        }
}

int main(int argc, char** argv){
    const std::string dir = argc > 1 ? argv[1] : ".";
    std::string why;
    if(!gpuReadAvailable(why)){
        printf("GPU tests skipped: %s\n", why.c_str());
        return 0;
    }
    A = api(&why);
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    ck(A->cuDeviceGet(&dev, 0), "cuDeviceGet");
    ck(A->cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    ck(A->cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
    printf("GPU reads with %s\n", A->nvcompPath.c_str());
    std::mt19937 rng(77);
    std::uniform_int_distribution<int> byte(0, 255);

    struct Case { const char* name; const char* dtype; uint64_t bits; };
    const Case cases[] = {
        {"uint8", "<u1", 8}, {"int8", "<i1", 8}, {"uint16", "<u2", 16}, {"int16", "<i2", 16},
        {"uint32", "<u4", 32}, {"int32", "<i4", 32}, {"uint64", "<u8", 64}, {"int64", "<i8", 64},
        {"float", "<f4", 32}, {"double", "<f8", 64}, {"bool", "|b1", 8}, {"half", "<f2", 16},
        {"bf16", "bfloat16", 16}, {"cplx64", "<c8", 64}, {"cplx128", "<c16", 128},
    };
    // (shuffle 1: byte shuffle, the default; 0: none; 2: bitshuffle, decoded on the host)
    struct Codec { const char* cname; int shuffle; int clevel; };
    const Codec codecs[] = {{"zstd", 1, 1}, {"zstd", 0, 5}, {"lz4", 1, 5}, {"lz4hc", 1, 5}, {"blosclz", 1, 5},
                            {"zlib", 1, 5}, {"zstd", 2, 1}, {"zstd", 1, 0}, {"gzip", 0, 5}, {"none", 0, 0}};
    const std::vector<uint64_t> shape = {40, 24, 18}, chunks = {16, 16, 16};
    std::vector<std::string> written;
    for(const Case &c : cases){
        std::vector<uint8_t> data(shape[0]*shape[1]*shape[2]*(c.bits/8));
        // (smooth data that compresses, with noise in the low bytes)
        for(size_t i = 0; i < data.size(); i++) data[i] = (i % (c.bits/8)) ? (uint8_t)((i/97) & 0xff) : (uint8_t)byte(rng);
        for(const Codec &k : codecs){
            for(const char* order : {"F", "C"}){
                for(const int fmt : {2, 3}){
                    for(const bool sharded : {false, true}){
                        if(sharded && fmt == 2) continue;
                        const std::string path = dir+"/gpu_"+c.name+"_"+k.cname+std::to_string(k.shuffle)+"_"+std::to_string(k.clevel)+"_"+order+
                                                 (fmt == 3 ? "_v3" : "")+(sharded ? "_sh" : "")+".zarr";
                        std::error_code ec;
                        fs::remove_all(path, ec);
                        try{
                            // (the full constructor: it takes blosc's shuffle)
                            zarr Zw(path, sharded ? std::vector<uint64_t>{32, 16, 18} : chunks, 0, (uint64_t)k.clevel, k.cname, "blosc",
                                    (uint64_t)k.shuffle, ".", c.dtype, "0", {}, order, shape, 2, {0, 0, 0}, sharded,
                                    sharded ? std::vector<uint64_t>{16, 8, 9} : std::vector<uint64_t>{1, 1, 1});
                            if(fmt == 3) Zw.set_zarr_format(3);
                            Zw.set_order(order);
                            Zw.set_dimension_separator(fmt == 3 ? "/" : ".");
                            if(std::string(c.dtype) == "<u2") Zw.set_fill_value(7);
                            Zw.write_zarray();
                            Zw.set_chunkInfo({0, 0, 0}, shape);
                            if(parallelWriteZarr(Zw, data.data(), {0, 0, 0}, shape, shape, c.bits, false, false, false))
                                throw std::string("write: ")+Zw.get_errString();
                        }
                        catch(const std::string &e){
                            printf("FAIL  %s: %s\n", path.c_str(), e.c_str());
                            failures++;
                            continue;
                        }
                        written.push_back(path);
                    }
                }
            }
        }
    }
    printf("%zu arrays written\n", written.size());
    for(const auto &p : written){
        const std::string label = fs::path(p).filename().string();
        testArray(p, label, rng);
    }
    printf("%d of %d written-array regions read the same on the GPU\n", total-failures, total);
    // shards whose chunks of noise blosc stored as they were, between chunks it compressed: the
    // stored chunks (and the compressed streams) start at any byte of the shard
    int before = total, failedBefore = failures;
    for(const Case &c : cases){
        if(c.bits < 16) continue;
        const uint64_t ts = c.bits/8;
        std::vector<uint8_t> data(shape[0]*shape[1]*shape[2]*ts);
        for(uint64_t z = 0; z < shape[2]; z++)
            for(uint64_t y = 0; y < shape[1]; y++)
                for(uint64_t x = 0; x < shape[0]; x++){
                    const uint64_t i = (x+shape[0]*(y+shape[1]*z))*ts;
                    const bool noise = (x/16+y/8+z/9) % 2 == 0;
                    for(uint64_t j = 0; j < ts; j++) data[i+j] = (noise || !j) ? (uint8_t)byte(rng) : (uint8_t)((i/97) & 0xff);
                }
        for(const Codec &k : {Codec{"lz4", 1, 5}, Codec{"lz4", 0, 5}, Codec{"zstd", 0, 5}, Codec{"zlib", 1, 5}}){
            for(const char* order : {"F", "C"}){
                const std::string path = dir+"/gpu_mixed_"+c.name+"_"+k.cname+std::to_string(k.shuffle)+"_"+order+".zarr";
                std::error_code ec;
                fs::remove_all(path, ec);
                try{
                    zarr Zw(path, {32, 16, 18}, 0, (uint64_t)k.clevel, k.cname, "blosc", (uint64_t)k.shuffle, "/", c.dtype, "0", {},
                            order, shape, 3, {0, 0, 0}, true, {16, 8, 9});
                    Zw.write_zarray();
                    Zw.set_chunkInfo({0, 0, 0}, shape);
                    if(parallelWriteZarr(Zw, data.data(), {0, 0, 0}, shape, shape, c.bits, false, false, false))
                        throw std::string("write: ")+Zw.get_errString();
                }
                catch(const std::string &e){
                    printf("FAIL  %s: %s\n", path.c_str(), e.c_str());
                    failures++;
                    continue;
                }
                testArray(path, fs::path(path).filename().string(), rng);
            }
        }
    }
    printf("%d of %d regions of shards with stored chunks at any byte read the same on the GPU\n",
           (total-before)-(failures-failedBefore), total-before);
    // missing chunks (some chunk files removed), with fill values 7 (uint16) and 0
    before = total;
    failedBefore = failures;
    for(const auto &p : written){
        if(p.find("_uint16_zstd1_1") == std::string::npos && p.find("_float_lz41_5") == std::string::npos) continue;
        std::vector<fs::path> files;
        for(const auto &e : fs::recursive_directory_iterator(p)){
            const std::string f = e.path().filename().string();
            if(e.is_regular_file() && f != ".zarray" && f != "zarr.json" && f != ".zattrs") files.push_back(e.path());
        }
        std::sort(files.begin(), files.end());
        for(size_t i = 0; i < files.size(); i += 3) fs::remove(files[i]);
        testArray(p, fs::path(p).filename().string()+" (missing chunks)", rng);
    }
    printf("%d of %d regions with missing chunks read the same on the GPU\n", (total-before)-(failures-failedBefore), total-before);
    // a batch: regions of several arrays (different data types and codecs) in one call
    {
        before = total;
        std::vector<zarr> Zs;
        std::vector<std::string> picks;
        for(const auto &p : written) if(p.find("_C_v3") != std::string::npos && p.find("zstd1_1") != std::string::npos) picks.push_back(p);
        Zs.reserve(picks.size());
        for(const auto &p : picks) Zs.emplace_back(p);
        std::vector<GpuRegion> regions;
        std::vector<CUdeviceptr> outs;
        std::vector<std::vector<uint8_t>> refs;
        for(auto &Z : Zs){
            const std::vector<uint64_t> s = {3, 2, 1}, e = {37, 23, 17};
            const DtypeInfo t = dtypeInfo(Z.get_dtype());
            const uint64_t bytes = 34*21*16*t.bytes;
            refs.emplace_back(bytes);
            uint8_t fill[16];
            fillValueElement(Z.get_fill_value(), Z.get_dtype(), fill);
            Z.set_chunkInfo(s, e);
            parallelReadZarr(Z, refs.back().data(), s, e, {34, 21, 16}, t.bytes*8, true, false, true, fill);
            CUdeviceptr d;
            ck(A->cuMemAlloc(&d, bytes), "cuMemAlloc");
            outs.push_back(d);
            regions.push_back({&Z, s, e, (void*)(uintptr_t)d});
        }
        std::string err;
        total++;
        if(gpuReadZarr(regions, true, nullptr, 8, err)){
            printf("FAIL  batch of %zu arrays: %s\n", regions.size(), err.c_str());
            failures++;
        }
        else{
            for(size_t i = 0; i < outs.size(); i++){
                std::vector<uint8_t> got(refs[i].size());
                ck(A->cuMemcpyDtoHAsync(got.data(), outs[i], got.size(), nullptr), "cuMemcpyDtoHAsync");
                ck(A->cuStreamSynchronize(nullptr), "cuStreamSynchronize");
                if(got != refs[i]){ printf("FAIL  batch: %s differs\n", picks[i].c_str()); failures++; break; }
            }
        }
        for(auto d : outs) A->cuMemFree(d);
        printf("batch of %zu regions from %zu arrays in one call\n", regions.size(), picks.size());
    }
    // damaged chunks: the GPU read fails where the CPU read does (a compressed stream that
    // cannot be decoded), and reads what the CPU does where neither can tell (a chunk blosc
    // stored as it was, with no checksum to check)
    {
        int errors = 0;
        for(const char* pick : {"gpu_uint16_zstd1_1_C_v3.zarr", "gpu_float_lz41_5_F.zarr", "gpu_uint16_zstd0_5_C_v3.zarr", "gpu_int32_zlib1_5_C_v3_sh.zarr"}){
            total++;
            std::string p;
            for(const auto &w : written) if(fs::path(w).filename() == pick) p = w;
            if(p.empty()){ printf("FAIL  no array %s to damage\n", pick); failures++; continue; }
            std::vector<fs::path> files;
            for(const auto &e : fs::recursive_directory_iterator(p))
                if(e.is_regular_file() && e.path().filename() != "zarr.json" && e.path().filename() != ".zarray") files.push_back(e.path());
            std::sort(files.begin(), files.end());
            {
                // (the bytes after the blosc header and block starts overwritten)
                const uint64_t size = fs::file_size(files[0]);
                std::fstream f(files[0], std::ios::in | std::ios::out | std::ios::binary);
                f.seekp(48);
                const std::vector<char> junk(size > 64 ? size-64 : 0, (char)0xff);
                f.write(junk.data(), (std::streamsize)junk.size());
            }
            zarr Z(p);
            const DtypeInfo dt = dtypeInfo(Z.get_dtype());
            std::vector<uint64_t> s = {0, 0, 0}, e = {40, 24, 18};
            const uint64_t bytes = 40*24*18*dt.bytes;
            std::vector<uint8_t> cpu(bytes), gpu(bytes);
            Z.set_chunkInfo(s, e);
            const bool cpuFailed = parallelReadZarr(Z, cpu.data(), s, e, e, dt.bytes*8, true, false, true) != 0;
            CUdeviceptr d;
            ck(A->cuMemAlloc(&d, bytes), "cuMemAlloc");
            std::vector<GpuRegion> regions{{&Z, s, e, (void*)(uintptr_t)d}};
            std::string err;
            const bool gpuFailed = gpuReadZarr(regions, true, nullptr, 4, err) != 0;
            if(!gpuFailed){
                ck(A->cuMemcpyDtoHAsync(gpu.data(), d, bytes, nullptr), "cuMemcpyDtoHAsync");
                ck(A->cuStreamSynchronize(nullptr), "cuStreamSynchronize");
            }
            A->cuMemFree(d);
            if(cpuFailed != gpuFailed || (!cpuFailed && cpu != gpu)){
                printf("FAIL  damaged %s: the CPU read %s, the GPU read %s\n", pick, cpuFailed ? "failed" : "worked",
                       gpuFailed ? ("failed: "+err).c_str() : (cpu == gpu ? "worked, the same" : "worked, differently"));
                failures++;
            }
            if(gpuFailed){ errors++; printf("damaged %s: %s\n", pick, err.c_str()); }
        }
        total++;
        if(!errors){ printf("FAIL  no damaged chunk gave an error\n"); failures++; }
    }
    // arrays other Zarr writers made
#ifdef CPPZARR_TEST_ARRAYS
    before = total;
    failedBefore = failures;
    if(fs::exists(CPPZARR_TEST_ARRAYS)){
        for(const auto &e : fs::directory_iterator(CPPZARR_TEST_ARRAYS))
            if(e.is_directory() && e.path().extension() == ".zarr") testArray(e.path().string(), e.path().filename().string(), rng);
    }
    printf("%d of %d regions of test arrays from other writers read the same on the GPU\n", (total-before)-(failures-failedBefore), total-before);
#endif
    printf("\n%d/%d GPU reads matched\n", total-failures, total);
    return failures ? 1 : 0;
}
