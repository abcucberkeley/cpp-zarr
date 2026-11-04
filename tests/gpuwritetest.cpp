// GPU writes (gpuwritezarr.h) against CPU writes (parallelWriteZarr): every data type; blosc's
// zstd, lz4 and lz4hc with and without shuffle, blosc's chunks stored as they are (clevel 0,
// incompressible data), the zstd codec and no compressor on the GPU, and blosclz, zlib, bitshuffle
// and gzip through host memory; both orders, both formats, sharding; whole arrays, then crop
// writes into them that cut chunks and that leave a chunk all zero; inputs in F order, C order and
// strided; sparse all-zero chunks and a nonzero fill value; writes in many parts; blosc block sizes
// a Zarr v3 array asks for; and crop writes into arrays other Zarr writers made (tests/test_arrays:
// checksums, transposed chunks, shard indexes at the start). Each pair of arrays must hold the same
// files, each chunk decoding to the same bytes (padding included), and read the same. Exits 0 with
// a message without an NVIDIA GPU, driver or nvCOMP. Usage: gpuWriteTest <scratch folder>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>
#include "gpu/cudaapi.h"
#include "gpureadzarr.h"
#include "gpuwritezarr.h"
#include "helperfunctions.h"
#include "parallelreadzarr.h"
#include "parallelwritezarr.h"
#include "zarr.h"

namespace fs = std::filesystem;
using namespace czcuda;

// (an environment variable set, or removed with an empty value)
static void setEnv(const char* name, const char* value){
#ifdef _WIN32
    _putenv_s(name, value);
#else
    if(*value) setenv(name, value, 1);
    else unsetenv(name);
#endif
}

static const Api* A = nullptr;
static int total = 0, failures = 0;

static void ck(CUresult r, const char* what){
    if(r){ fprintf(stderr, "%s: %s\n", what, errorText(A, r).c_str()); exit(2); }
}

static void fail(const std::string &label, const std::string &what){
    printf("FAIL  %s: %s\n", label.c_str(), what.c_str());
    failures++;
}

// The files of an array but its metadata, relative to it, sorted
static std::vector<std::string> chunkFiles(const std::string &path){
    std::vector<std::string> files;
    for(const auto &e : fs::recursive_directory_iterator(path)){
        const std::string f = e.path().filename().string();
        if(e.is_regular_file() && f != ".zarray" && f != "zarr.json" && f != ".zattrs" && f != ".zgroup")
            files.push_back(fs::relative(e.path(), path).generic_string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

static std::vector<uint8_t> readFile(const std::string &path){
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

// The blosc header of an array's first stored chunk (in a sharded array, of its first stored inner
// chunk)
static bool firstHeader(const std::string &path, uint8_t h[16]){
    zarr Z(path);
    const uint64_t empty = ~uint64_t(0);
    for(const auto &f : chunkFiles(path)){
        const auto b = readFile(path+"/"+f);
        uint64_t at = 0;
        if(Z.get_shard()){
            uint64_t per = 1;
            for(uint64_t d = 0; d < Z.get_ndims(); d++) per *= (Z.get_chunks(d)+Z.get_chunk_shape(d)-1)/Z.get_chunk_shape(d);
            const uint64_t ib = 16*per+(Z.get_shardIndexChecksum() ? 4 : 0);
            if(b.size() < ib) continue;
            const uint64_t base = Z.get_shardIndexAtStart() ? 0 : b.size()-ib;
            at = empty;
            for(uint64_t i = 0; i < per && at == empty; i++) memcpy(&at, b.data()+base+16*i, 8);
            if(at == empty) continue;
        }
        if(b.size() < at+16) continue;
        memcpy(h, b.data()+at, 16);
        return true;
    }
    return false;
}

// Whether two stored chunks decode to the same bytes (both must decode)
static bool sameChunk(zarr &Z, const uint8_t* a, uint64_t na, const uint8_t* b, uint64_t nb, const uint64_t sB, std::string &why){
    std::vector<uint8_t> da(sB), db(sB);
    std::string wa, wb;
    if(decodeChunk(Z.get_compressor(), Z.get_chunkChecksum(), a, na, da.data(), sB, wa)){ why = "the CPU's does not decode: "+wa; return false; }
    if(decodeChunk(Z.get_compressor(), Z.get_chunkChecksum(), b, nb, db.data(), sB, wb)){ why = "the GPU's does not decode: "+wb; return false; }
    if(da != db){
        uint64_t i = 0;
        while(da[i] == db[i]) i++;
        why = "they decode differently (first at byte "+std::to_string(i)+")";
        return false;
    }
    return true;
}

// The CPU's array (a) and the GPU's (b): the same files, the same chunks, the same data
static bool sameArrays(const std::string &a, const std::string &b, const std::string &label){
    total++;
    const auto fa = chunkFiles(a), fb = chunkFiles(b);
    if(fa != fb){
        std::string d;
        for(const auto &f : fa) if(!std::count(fb.begin(), fb.end(), f)) d += " CPU only: "+f;
        for(const auto &f : fb) if(!std::count(fa.begin(), fa.end(), f)) d += " GPU only: "+f;
        fail(label, "different files:"+d.substr(0, 300));
        return false;
    }
    zarr Z(a);
    const uint64_t n = Z.get_ndims();
    const DtypeInfo t = dtypeInfo(Z.get_dtype());
    uint64_t celems = 1, perShard = 1;
    for(uint64_t d = 0; d < n; d++){
        celems *= Z.get_shard() ? Z.get_chunk_shape(d) : Z.get_chunks(d);
        if(Z.get_shard()) perShard *= (Z.get_chunks(d)+Z.get_chunk_shape(d)-1)/Z.get_chunk_shape(d);
    }
    const uint64_t sB = celems*t.bytes;
    for(const auto &f : fa){
        const auto ba = readFile(a+"/"+f), bb = readFile(b+"/"+f);
        std::string why;
        if(!Z.get_shard()){
            if(!sameChunk(Z, ba.data(), ba.size(), bb.data(), bb.size(), sB, why)){ fail(label, f+": "+why); return false; }
            continue;
        }
        const bool atStart = Z.get_shardIndexAtStart(), crc = Z.get_shardIndexChecksum();
        const uint64_t ib = 16*perShard+(crc ? 4 : 0);
        if(ba.size() < ib || bb.size() < ib){ fail(label, f+": a shard smaller than its index"); return false; }
        std::vector<uint64_t> xa(2*perShard), xb(2*perShard);
        memcpy(xa.data(), ba.data()+(atStart ? 0 : ba.size()-ib), 16*perShard);
        memcpy(xb.data(), bb.data()+(atStart ? 0 : bb.size()-ib), 16*perShard);
        if(crc){
            uint32_t ca, cb;
            memcpy(&ca, ba.data()+(atStart ? 0 : ba.size()-ib)+16*perShard, 4);
            memcpy(&cb, bb.data()+(atStart ? 0 : bb.size()-ib)+16*perShard, 4);
            if(cb != crc32c(reinterpret_cast<const uint8_t*>(xb.data()), 16*perShard)){ fail(label, f+": the GPU's shard index checksum is wrong"); return false; }
            (void)ca;
        }
        const uint64_t empty = ~uint64_t(0);
        for(uint64_t i = 0; i < perShard; i++){
            const bool ea = xa[2*i] == empty && xa[2*i+1] == empty, eb = xb[2*i] == empty && xb[2*i+1] == empty;
            if(ea != eb){ fail(label, f+": inner chunk "+std::to_string(i)+(ea ? " only in the GPU's shard" : " only in the CPU's shard")); return false; }
            if(ea) continue;
            if(xb[2*i]+xb[2*i+1] > bb.size()){ fail(label, f+": the GPU's index points outside its shard"); return false; }
            if(!sameChunk(Z, ba.data()+xa[2*i], xa[2*i+1], bb.data()+xb[2*i], xb[2*i+1], sB, why)){
                fail(label, f+" inner chunk "+std::to_string(i)+": "+why);
                return false;
            }
        }
    }
    // and the arrays read the same
    zarr Zb(b);
    std::vector<uint64_t> start(n, 0), end(n), shape(n);
    uint64_t elems = 1;
    for(uint64_t d = 0; d < n; d++){ end[d] = shape[d] = Z.get_shape(d); elems *= shape[d]; }
    std::vector<uint8_t> ra(elems*t.bytes), rb(elems*t.bytes);
    Z.set_chunkInfo(start, end);
    Zb.set_chunkInfo(start, end);
    if(parallelReadZarr(Z, ra.data(), start, end, shape, t.bytes*8, true, false, false) ||
       parallelReadZarr(Zb, rb.data(), start, end, shape, t.bytes*8, true, false, false) || ra != rb){
        fail(label, "the arrays read differently");
        return false;
    }
    return true;
}

// The data's input layouts on the GPU: 0 F order, 1 C order, 2 strided (F order inside a bigger box)
static std::vector<uint64_t> layoutStrides(const std::vector<uint64_t> &shape, const int layout, uint64_t &span){
    const size_t n = shape.size();
    std::vector<uint64_t> st(n);
    uint64_t acc = 1;
    for(size_t k = 0; k < n; k++){
        const size_t d = layout == 1 ? n-1-k : k;
        st[d] = acc;
        acc *= shape[d]+(layout == 2 ? (d == 0 ? 3 : 1) : 0);
    }
    span = acc;
    return st;
}

// The F-order host data copied to GPU memory in a layout; the memory is freed by the caller
static CUdeviceptr upload(const std::vector<uint8_t> &host, const std::vector<uint64_t> &shape, const uint64_t ts, const int layout,
                          std::vector<uint64_t> &strides){
    uint64_t span = 1;
    strides = layoutStrides(shape, layout, span);
    std::vector<uint8_t> laid(std::max<uint64_t>(span, 1)*ts, 0x5a);
    const size_t n = shape.size();
    uint64_t elems = 1;
    for(auto v : shape) elems *= v;
    std::vector<uint64_t> idx(n, 0);
    for(uint64_t i = 0; i < elems; i++){
        uint64_t off = 0;
        for(size_t d = 0; d < n; d++) off += idx[d]*strides[d];
        memcpy(&laid[off*ts], &host[i*ts], ts);
        for(size_t d = 0; d < n; d++){ if(++idx[d] < shape[d]) break; idx[d] = 0; }
    }
    CUdeviceptr p = 0;
    ck(A->cuMemAlloc(&p, laid.size()), "cuMemAlloc");
    ck(A->cuMemcpyHtoDAsync(p, laid.data(), laid.size(), nullptr), "cuMemcpyHtoDAsync");
    ck(A->cuStreamSynchronize(nullptr), "cuStreamSynchronize");
    return p;
}

// One write of the region [start, end) of the data (F order on the host): parallelWriteZarr into
// a, gpuWriteZarr into b; both must succeed, or both fail
static bool writeBoth(const std::string &a, const std::string &b, const std::vector<uint8_t> &data, const std::vector<uint64_t> &start,
                      const std::vector<uint64_t> &end, const bool crop, const int layout, CUstream stream, const std::string &label,
                      std::string *gpuErr = nullptr){
    zarr Za(a), Zb(b);
    const DtypeInfo t = dtypeInfo(Za.get_dtype());
    const size_t n = start.size();
    std::vector<uint64_t> shape(n), fst(n);
    uint64_t acc = 1;
    for(size_t d = 0; d < n; d++){ shape[d] = end[d]-start[d]; fst[d] = acc; acc *= shape[d]; }
    Za.set_chunkInfo(start, end);
    Zb.set_chunkInfo(start, end);
    const uint8_t ra = parallelWriteZarr(Za, (void*)data.data(), start, end, shape, fst, t.bytes*8, true, crop);
    std::vector<uint64_t> strides;
    const CUdeviceptr d = upload(data, shape, t.bytes, layout, strides);
    const uint8_t rb = gpuWriteZarr(Zb, (const void*)(uintptr_t)d, start, end, shape, strides, t.bytes*8, true, crop, true, stream);
    A->cuMemFree(d);
    if(gpuErr) *gpuErr = rb ? Zb.get_errString() : "";
    if(ra != rb){
        fail(label, std::string("the CPU write ")+(ra ? "failed ("+Za.get_errString()+")" : "worked")+", the GPU write "+
                    (rb ? "failed: "+Zb.get_errString() : "worked"));
        return false;
    }
    return true;
}

// A new array's metadata in both folders
static bool makeArrays(const std::string &a, const std::string &b, const std::vector<uint64_t> &shape, const std::vector<uint64_t> &chunks,
                       const std::vector<uint64_t> &inner, const bool sharded, const int clevel, const std::string &cname,
                       const std::string &id, const int shuffle, const std::string &dtype, const int fmt, const std::string &order,
                       const std::string &fill, const uint64_t blocksize = 0){
    for(const std::string &p : {a, b}){
        std::error_code ec;
        fs::remove_all(p, ec);
        try{
            zarr Z(p, chunks, blocksize, (uint64_t)clevel, cname, id, (uint64_t)shuffle, fmt == 3 ? "/" : ".", dtype, fill, {}, order, shape, 2,
                   std::vector<uint64_t>(shape.size(), 0), sharded, sharded ? inner : std::vector<uint64_t>(shape.size(), 1));
            if(fmt == 3) Z.set_zarr_format(3);
            Z.set_order(order);
            Z.set_dimension_separator(fmt == 3 ? "/" : ".");
            Z.write_zarray();
        }
        catch(const std::string &e){
            fail(p, "metadata: "+e);
            return false;
        }
    }
    return true;
}

// Data for an array: smooth with noise in the low bytes, one chunk's box of zeros, one of noise in
// every byte (which blosc stores as it is)
static std::vector<uint8_t> makeData(const std::vector<uint64_t> &shape, const std::vector<uint64_t> &chunk, const uint64_t ts, std::mt19937 &rng,
                                     const int seed){
    const size_t n = shape.size();
    uint64_t elems = 1;
    for(auto v : shape) elems *= v;
    std::vector<uint8_t> data(elems*ts);
    std::uniform_int_distribution<int> byte(0, 255);
    std::vector<uint64_t> idx(n, 0);
    for(uint64_t i = 0; i < elems; i++){
        bool zero = n > 0, noise = n > 0;
        for(size_t d = 0; d < n; d++){
            zero = zero && idx[d] < chunk[d];
            noise = noise && (d == 0 ? idx[d] >= chunk[d] && idx[d] < 2*chunk[d] : idx[d] < chunk[d]);
        }
        for(uint64_t j = 0; j < ts; j++){
            uint8_t v;
            if(zero) v = 0;
            else if(noise || j == 0) v = (uint8_t)byte(rng);
            else v = (uint8_t)(((i+(uint64_t)seed*131)/97+j) & 0xff);
            data[i*ts+j] = v;
        }
        for(size_t d = 0; d < n; d++){ if(++idx[d] < shape[d]) break; idx[d] = 0; }
    }
    return data;
}

int main(int argc, char** argv){
    const std::string dir = argc > 1 ? argv[1] : ".";
    std::string why;
    if(!gpuWriteAvailable(why)){
        printf("GPU write tests skipped: %s\n", why.c_str());
        return 0;
    }
    A = api(&why);
    CUdevice dev = 0;
    CUcontext ctx = nullptr;
    ck(A->cuDeviceGet(&dev, 0), "cuDeviceGet");
    ck(A->cuDevicePrimaryCtxRetain(&ctx, dev), "cuDevicePrimaryCtxRetain");
    ck(A->cuCtxSetCurrent(ctx), "cuCtxSetCurrent");
    CUstream stream = nullptr;
    ck(A->cuStreamCreate(&stream, 1 /* CU_STREAM_NON_BLOCKING */), "cuStreamCreate");
    printf("GPU writes with %s\n", A->nvcompPath.c_str());
    std::mt19937 rng(91);

    struct Case { const char* name; const char* dtype; uint64_t bits; };
    const Case cases[] = {
        {"uint8", "|u1", 8}, {"int8", "|i1", 8}, {"uint16", "<u2", 16}, {"int16", "<i2", 16},
        {"uint32", "<u4", 32}, {"int32", "<i4", 32}, {"uint64", "<u8", 64}, {"int64", "<i8", 64},
        {"float", "<f4", 32}, {"double", "<f8", 64}, {"bool", "|b1", 8}, {"half", "<f2", 16},
        {"bf16", "bfloat16", 16}, {"cplx64", "<c8", 64}, {"cplx128", "<c16", 128},
    };
    // (id "blosc": blosc's codecs, shuffle 1 byte shuffle, 0 none, 2 bitshuffle; "zstd": the zstd
    // codec; gzip and none)
    struct Codec { const char* cname; const char* id; int shuffle; int clevel; };
    const Codec codecs[] = {{"zstd", "blosc", 1, 1}, {"zstd", "blosc", 0, 5}, {"lz4", "blosc", 1, 5}, {"lz4", "blosc", 0, 1},
                            {"lz4hc", "blosc", 1, 3}, {"zstd", "blosc", 1, 0}, {"blosclz", "blosc", 1, 5}, {"zlib", "blosc", 1, 5},
                            {"zstd", "blosc", 2, 1}, {"zstd", "zstd", 0, 3}, {"gzip", "gzip", 0, 5}, {"none", "blosc", 0, 0}};
    const std::vector<uint64_t> shape = {40, 24, 18}, chunks = {16, 16, 16}, shard = {32, 16, 18}, inner = {16, 8, 9};
    int k = 0;
    for(const Case &c : cases){
        const uint64_t ts = c.bits/8;
        for(const Codec &cd : codecs){
            for(const char* order : {"F", "C"}){
                for(const int fmt : {2, 3}){
                    for(const bool sharded : {false, true}){
                        if(sharded && fmt == 2) continue;
                        if(std::string(cd.id) == "zstd" && fmt == 2) continue;
                        k++;
                        const std::string label = std::string(c.name)+"_"+cd.cname+"_"+cd.id+std::to_string(cd.shuffle)+"_"+
                                                  std::to_string(cd.clevel)+"_"+order+(fmt == 3 ? "_v3" : "_v2")+(sharded ? "_sh" : "");
                        const std::string a = dir+"/w_"+label+"_cpu.zarr", b = dir+"/w_"+label+"_gpu.zarr";
                        // (a nonzero fill value for uint16: its all-zero chunks are written)
                        const std::string fill = std::string(c.dtype) == "<u2" ? "7" : "0";
                        if(!makeArrays(a, b, shape, sharded ? shard : chunks, inner, sharded, cd.clevel, cd.cname, cd.id, cd.shuffle,
                                       c.dtype, fmt, order, fill)) continue;
                        const std::vector<uint64_t> cb = sharded ? inner : chunks;
                        // the whole array, then a crop write that cuts chunks, then one that leaves a
                        // chunk all zero (removed in sparse writes)
                        const auto d1 = makeData(shape, cb, ts, rng, 1);
                        if(!writeBoth(a, b, d1, {0, 0, 0}, shape, false, k % 3, (k % 2) ? stream : nullptr, label+" whole")) continue;
                        if(!sameArrays(a, b, label+" whole")) continue;
                        const std::vector<uint64_t> s = {5, 3, 2}, e = {29, 20, 15};
                        const auto d2 = makeData({24, 17, 13}, {8, 8, 8}, ts, rng, 2);
                        if(!writeBoth(a, b, d2, s, e, true, (k+1) % 3, nullptr, label+" crop")) continue;
                        if(!sameArrays(a, b, label+" crop")) continue;
                        const std::vector<uint8_t> d3(8*16*16*ts, 0);
                        if(!writeBoth(a, b, d3, {32, 0, 0}, {40, 16, 16}, true, (k+2) % 3, nullptr, label+" zero crop")) continue;
                        sameArrays(a, b, label+" zero crop");
                    }
                }
            }
        }
    }
    printf("%d of %d checks of written arrays matched\n", total-failures, total);

    // bigger arrays: chunks of 4 MiB (blosc blocks of up to 1 MiB) and 32 MiB, a write in many parts,
    // and 1-D and 5-D arrays
    {
        const int before = total, failedBefore = failures;
        struct Big { const char* name; std::vector<uint64_t> shape, chunks, inner; bool sharded; const char* cname; const char* id; int clevel; };
        const Big bigs[] = {
            {"big4m_zstd", {256, 128, 96}, {128, 128, 64}, {}, false, "zstd", "blosc", 5},
            {"big4m_lz4", {256, 128, 96}, {128, 128, 64}, {}, false, "lz4", "blosc", 9},
            {"big32m_zstdcodec", {256, 256, 300}, {256, 256, 256}, {}, false, "zstd", "zstd", 1},
            {"parts_sh", {200, 200, 120}, {64, 64, 64}, {32, 32, 32}, true, "zstd", "blosc", 1},
            {"parts", {200, 200, 120}, {32, 32, 32}, {}, false, "lz4", "blosc", 1},
            {"one_d", {100000}, {4096}, {}, false, "zstd", "blosc", 1},
            {"five_d_sh", {3, 8, 40, 36, 2}, {2, 8, 32, 32, 2}, {1, 8, 16, 16, 2}, true, "zstd", "blosc", 1},
        };
        for(const Big &g : bigs){
            const std::string a = dir+"/w_"+g.name+"_cpu.zarr", b = dir+"/w_"+g.name+"_gpu.zarr";
            if(!makeArrays(a, b, g.shape, g.chunks, g.inner, g.sharded, g.clevel, g.cname, g.id, 1, "<u2", 3, "C", "0")) continue;
            const bool parts = std::string(g.name).rfind("parts", 0) == 0;
            if(parts) setEnv("CPPZARR_GPU_WRITE_PART_MB", "1");
            const auto d1 = makeData(g.shape, g.sharded ? g.inner : g.chunks, 2, rng, 3);
            std::vector<uint64_t> zero(g.shape.size(), 0), s(g.shape.size()), e(g.shape.size()), ws(g.shape.size());
            for(size_t d = 0; d < g.shape.size(); d++){
                s[d] = g.shape[d] > 2 ? g.shape[d]/5 : 0;
                e[d] = g.shape[d] > 2 ? g.shape[d]-g.shape[d]/7 : g.shape[d];
                ws[d] = e[d]-s[d];
            }
            if(writeBoth(a, b, d1, zero, g.shape, false, 1, stream, std::string(g.name)+" whole") && sameArrays(a, b, std::string(g.name)+" whole")){
                const auto d2 = makeData(ws, g.sharded ? g.inner : g.chunks, 2, rng, 4);
                if(writeBoth(a, b, d2, s, e, true, 2, nullptr, std::string(g.name)+" crop")) sameArrays(a, b, std::string(g.name)+" crop");
            }
            if(parts) setEnv("CPPZARR_GPU_WRITE_PART_MB", "");
        }
        printf("%d of %d checks of bigger arrays matched\n", (total-before)-(failures-failedBefore), total-before);
    }

    // blosc block sizes a Zarr v3 array asks for, which both writers use: a small one, one of more
    // than 4 MiB (blosc's own blocks are at most 1 MiB), one bigger than nvCOMP compresses (written
    // through host memory), lz4's (blosc caps the blocks of codecs it splits), and a v2 array's (not
    // used)
    {
        const int before = total, failedBefore = failures;
        struct Bs { const char* name; std::vector<uint64_t> shape, chunks, inner; bool sharded; const char* cname; int clevel;
                    uint64_t blocksize; int fmt; uint32_t expect; };
        const Bs sizes[] = {
            {"bs16k", {128, 64, 64}, {64, 64, 32}, {}, false, "zstd", 1, 16384, 3, 16384},
            {"bs6m_sh", {256, 256, 128}, {256, 256, 128}, {128, 256, 128}, true, "zstd", 1, 6 << 20, 3, 6 << 20},
            {"bs24m", {256, 256, 300}, {256, 256, 256}, {}, false, "zstd", 1, 24 << 20, 3, 24 << 20},
            {"bs_lz4", {256, 128, 128}, {128, 128, 128}, {}, false, "lz4", 5, 2 << 20, 3, 512 << 10},
            {"bs_v2", {128, 64, 64}, {64, 64, 32}, {}, false, "zstd", 1, 16384, 2, 32768},
        };
        for(const Bs &g : sizes){
            const std::string a = dir+"/w_"+g.name+"_cpu.zarr", b = dir+"/w_"+g.name+"_gpu.zarr";
            if(!makeArrays(a, b, g.shape, g.chunks, g.inner, g.sharded, g.clevel, g.cname, "blosc", 1, "<u2", g.fmt, "C", "0", g.blocksize))
                continue;
            // (data that compresses in every chunk, so that no chunk is stored as it is)
            uint64_t elems = 1;
            for(auto v : g.shape) elems *= v;
            std::vector<uint8_t> data(elems*2);
            for(uint64_t i = 0; i < elems; i++){
                const uint16_t v = (uint16_t)(1+(i/3) % 2000);
                memcpy(&data[2*i], &v, 2);
            }
            const std::vector<uint64_t> zero(g.shape.size(), 0);
            if(!writeBoth(a, b, data, zero, g.shape, false, 1, stream, std::string(g.name)) || !sameArrays(a, b, std::string(g.name))) continue;
            total++;
            uint8_t ha[16], hb[16];
            if(!firstHeader(a, ha) || !firstHeader(b, hb)){
                fail(g.name, "no stored chunk");
                continue;
            }
            uint32_t bsa, bsb;
            memcpy(&bsa, ha+8, 4);
            memcpy(&bsb, hb+8, 4);
            if(bsa != g.expect || bsb != g.expect || ha[2] != hb[2] || ha[3] != hb[3])
                fail(g.name, "blosc blocks of "+std::to_string(bsa)+" bytes (CPU) and "+std::to_string(bsb)+" (GPU), flags "+
                             std::to_string(ha[2])+" and "+std::to_string(hb[2])+"; "+std::to_string(g.expect)+" expected");
        }
        printf("%d of %d checks of blosc block sizes matched\n", (total-before)-(failures-failedBefore), total-before);
    }

    // through host memory on purpose (CPPZARR_GPU_WRITE_HOST), for a codec the GPU compresses
    {
        const int before = total, failedBefore = failures;
        setEnv("CPPZARR_GPU_WRITE_HOST", "1");
        const std::string a = dir+"/w_host_cpu.zarr", b = dir+"/w_host_gpu.zarr";
        if(makeArrays(a, b, shape, shard, inner, true, 1, "zstd", "blosc", 1, "<f4", 3, "C", "0")){
            const auto d1 = makeData(shape, inner, 4, rng, 5);
            if(writeBoth(a, b, d1, {0, 0, 0}, shape, false, 2, nullptr, "through host memory")) sameArrays(a, b, "through host memory");
        }
        setEnv("CPPZARR_GPU_WRITE_HOST", "");
        printf("%d of %d checks through host memory matched\n", (total-before)-(failures-failedBefore), total-before);
    }

    // GPU writes read back on the GPU
    {
        const int before = total, failedBefore = failures;
        for(const char* label : {"uint16_zstd_blosc1_1_C_v3_sh", "float_lz4_blosc0_1_F_v2", "cplx128_zstd_zstd0_3_C_v3", "int8_none_blosc0_0_F_v3"}){
            total++;
            const std::string p = dir+"/w_"+std::string(label)+"_gpu.zarr";
            zarr Z(p);
            const DtypeInfo t = dtypeInfo(Z.get_dtype());
            std::vector<uint64_t> s = {0, 0, 0}, e = shape;
            const uint64_t bytes = 40*24*18*t.bytes;
            std::vector<uint8_t> cpu(bytes), gpu(bytes);
            Z.set_chunkInfo(s, e);
            parallelReadZarr(Z, cpu.data(), s, e, e, t.bytes*8, true, false, true);
            CUdeviceptr d;
            ck(A->cuMemAlloc(&d, bytes), "cuMemAlloc");
            std::vector<GpuRegion> regions{{&Z, s, e, (void*)(uintptr_t)d}};
            std::string err;
            if(gpuReadZarr(regions, true, nullptr, 4, err)) fail(label, "GPU read: "+err);
            else{
                ck(A->cuMemcpyDtoHAsync(gpu.data(), d, bytes, nullptr), "cuMemcpyDtoHAsync");
                ck(A->cuStreamSynchronize(nullptr), "cuStreamSynchronize");
                if(cpu != gpu) fail(label, "reads differently on the GPU");
            }
            A->cuMemFree(d);
        }
        printf("%d of %d GPU-written arrays read the same on the GPU\n", (total-before)-(failures-failedBefore), total-before);
    }

    // crop writes into arrays other Zarr writers made (copies of them)
#ifdef CPPZARR_TEST_ARRAYS
    {
        const int before = total, failedBefore = failures;
        if(fs::exists(CPPZARR_TEST_ARRAYS)){
            for(const auto &en : fs::directory_iterator(CPPZARR_TEST_ARRAYS)){
                if(!en.is_directory() || en.path().extension() != ".zarr") continue;
                const std::string name = en.path().stem().string();
                zarr Z;
                try{ Z = zarr(en.path().string()); }
                catch(const std::string &){ continue; }
                const DtypeInfo t = dtypeInfo(Z.get_dtype());
                if(!t.kind) continue;
                const std::string a = dir+"/o_"+name+"_cpu.zarr", b = dir+"/o_"+name+"_gpu.zarr";
                std::error_code ec;
                fs::remove_all(a, ec);
                fs::remove_all(b, ec);
                fs::copy(en.path(), a, fs::copy_options::recursive);
                fs::copy(en.path(), b, fs::copy_options::recursive);
                const uint64_t n = Z.get_ndims();
                std::vector<uint64_t> s(n), e(n), ws(n);
                for(uint64_t d = 0; d < n; d++){
                    s[d] = Z.get_shape(d) > 2 ? 1 : 0;
                    e[d] = Z.get_shape(d) > 3 ? Z.get_shape(d)-1 : Z.get_shape(d);
                    ws[d] = e[d]-s[d];
                }
                std::vector<uint64_t> cd(n);
                for(uint64_t d = 0; d < n; d++) cd[d] = Z.get_shard() ? Z.get_chunk_shape(d) : Z.get_chunks(d);
                const auto data = makeData(ws, cd, t.bytes, rng, 6);
                std::string gpuErr;
                if(writeBoth(a, b, data, s, e, true, 1, nullptr, "test array "+name, &gpuErr)){
                    if(gpuErr.empty()) sameArrays(a, b, "test array "+name);
                    else printf("test array %s: both writes failed (%s)\n", name.c_str(), gpuErr.substr(0, gpuErr.find('\n')).c_str());
                }
                // and the whole array written anew (not a crop write) with its metadata: a sharded
                // one's shards then written whole on the GPU's side (index at the start or the end)
                if(!gpuErr.empty()) continue;
                fs::remove_all(a, ec);
                fs::remove_all(b, ec);
                fs::copy(en.path(), a, fs::copy_options::recursive);
                fs::copy(en.path(), b, fs::copy_options::recursive);
                std::vector<uint64_t> zero(n, 0), whole(n);
                for(uint64_t d = 0; d < n; d++) whole[d] = Z.get_shape(d);
                const auto all = makeData(whole, cd, t.bytes, rng, 8);
                if(writeBoth(a, b, all, zero, whole, false, 2, nullptr, "test array "+name+" anew")) sameArrays(a, b, "test array "+name+" anew");
            }
        }
        printf("%d of %d checks of other writers' arrays matched\n", (total-before)-(failures-failedBefore), total-before);
    }
#endif

    // errors: data that is not GPU memory
    {
        total++;
        const std::string a = dir+"/w_err_cpu.zarr", b = dir+"/w_err_gpu.zarr";
        if(makeArrays(a, b, shape, chunks, inner, false, 1, "zstd", "blosc", 1, "<u2", 2, "F", "0")){
            zarr Zb(b);
            std::vector<uint16_t> host(40*24*18, 1);
            Zb.set_chunkInfo({0, 0, 0}, shape);
            if(!gpuWriteZarr(Zb, host.data(), {0, 0, 0}, shape, shape, {1, 40, 960}, 16, true, false))
                fail("host memory", "the GPU write of host memory did not fail");
            else printf("host memory: %s", Zb.get_errString().c_str());
        }
    }
    printf("\n%d/%d GPU write checks matched\n", total-failures, total);
    return failures ? 1 : 0;
}
