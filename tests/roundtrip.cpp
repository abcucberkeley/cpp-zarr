// Minimal smoke test for the cpp-zarr library: write small random volumes, read
// them back, and check the bytes survive the round trip. Covers every supported
// dtype (signed/unsigned 8-64 bit ints, f4/f8), each compressor, and both storage
// orders (F and C), using shapes that do not divide the chunk size so partial edge
// chunks are exercised, plus chunks large enough for the full-tile F<->C paths,
// 1D/2D/4D/5D arrays (region reads, crop writes, sharding, subfolders, and C-order
// input and output), and sharded and sparse writes (see that section).
//
// Exits 0 if every case passes, 1 otherwise, so CTest reports pass/fail.
// Usage: roundtripTest [output_dir]   (defaults to the current directory)
//
// Progress is written (flushed) to stderr before each operation so that if a
// build segfaults, the CTest log shows exactly which step/dtype/compressor died.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "zarr.h"
#include "helperfunctions.h"
#include "parallelreadzarr.h"
#include "parallelwritezarr.h"

struct Case { const char* name; const char* dtype; uint64_t bits; };

static void step(const char* name, const char* comp, const char* order, const char* what){
    std::fprintf(stderr, "  [%s/%s/%s] %s\n", name, comp, order, what);
    std::fflush(stderr);
}

int main(int argc, char** argv){
    const std::string dir = (argc > 1) ? argv[1] : ".";

    struct Config { const char* tag; std::vector<uint64_t> shape, chunks; std::vector<const char*> comps; };
    const Config configs[] = {
        // Non-chunk-aligned shape so read/write hit partial edge chunks; every compressor.
        {"", {40, 24, 18}, {16, 16, 16}, {"lz4", "blosclz", "lz4hc", "zlib", "zstd", "gzip", "none"}},
        // Chunks >= 128 along x and z so C-order reads/writes take the full-tile
        // (vectorized) transpose path for every element size, plus partial edges.
        {"big", {300, 9, 270}, {128, 4, 128}, {"lz4"}},
    };

    const Case cases[]  = {
        {"uint8",  "<u1", 8},  {"int8",   "<i1", 8},
        {"uint16", "<u2", 16}, {"int16",  "<i2", 16},
        {"uint32", "<u4", 32}, {"int32",  "<i4", 32},
        {"uint64", "<u8", 64}, {"int64",  "<i8", 64},
        {"float",  "<f4", 32}, {"double", "<f8", 64},
    };
    const char* orders[] = {"F", "C"};

    std::mt19937 rng(1234567u);
    std::uniform_int_distribution<int> byteDist(0, 255);

    int total = 0, failures = 0;
    for (const Config& cfg : configs){
    const std::vector<uint64_t>& shape = cfg.shape;
    const std::vector<uint64_t>& chunks = cfg.chunks;
    const uint64_t n = shape[0] * shape[1] * shape[2];
    for (const Case& c : cases){
        const uint64_t nbytes = n * (c.bits / 8);
        std::vector<uint8_t> orig(nbytes);
        for (uint64_t i = 0; i < nbytes; i++) orig[i] = static_cast<uint8_t>(byteDist(rng));

        for (const char* comp : cfg.comps){
            for (const char* order : orders){
            for (const int fmt : {2, 3}){
                total++;
                const char* vtag = fmt == 3 ? " v3" : "";
                const std::string path = dir + "/rt_" + c.name + "_" + comp + "_" + order + cfg.tag + (fmt == 3 ? "_v3" : "") + ".zarr";
                std::error_code ec; std::filesystem::remove_all(path, ec);

                bool metaOK = false, dataOK = false;
                try {
                    step(c.name, comp, order, "write");
                    zarr Zw;
                    Zw.set_fileName(path);
                    // (Zarr v3 with its default chunk keys, c/0/0/0)
                    if (fmt == 3) Zw.set_zarr_format(3);
                    Zw.set_cname(comp);
                    Zw.set_clevel(5);
                    Zw.set_order(order);
                    Zw.set_chunks(chunks);
                    Zw.set_dimension_separator(fmt == 3 ? "/" : ".");
                    Zw.set_dtype(c.dtype);
                    Zw.set_shape(shape);
                    Zw.write_zarray();
                    Zw.set_chunkInfo({0,0,0}, shape);
                    uint8_t werr = parallelWriteZarr(Zw, orig.data(), {0,0,0}, shape, shape,
                                                     c.bits, /*useUuid*/false, /*crop*/false, /*sparse*/false);
                    if (werr) throw std::string("write error: ") + Zw.get_errString();

                    // Metadata (.zarray, or zarr.json) must round trip.
                    step(c.name, comp, order, "read metadata");
                    zarr Zr(path);
                    // (1-byte types have no byte order: "|u1" in v3)
                    const std::string wantType = fmt == 3 && c.bits == 8 ? std::string("|") + (c.dtype+1) : c.dtype;
                    metaOK = Zr.get_dtype() == wantType && Zr.get_cname() == comp &&
                             Zr.get_shape(0) == shape[0] && Zr.get_shape(1) == shape[1] &&
                             Zr.get_shape(2) == shape[2] && Zr.get_order() == order &&
                             Zr.get_zarr_format() == (uint64_t)fmt &&
                             fileExists(path + (fmt == 3 ? "/zarr.json" : "/.zarray")) &&
                             !fileExists(path + (fmt == 3 ? "/.zarray" : "/zarr.json")) &&
                             (fmt == 2 || fileExists(path + "/c/0/0/0"));

                    // Data must be byte-identical to what we wrote.
                    step(c.name, comp, order, "read data");
                    Zr.set_chunkInfo({0,0,0}, shape);
                    std::vector<uint8_t> rbuf(nbytes, 0);
                    uint8_t rerr = parallelReadZarr(Zr, rbuf.data(), {0,0,0}, shape, shape,
                                                    c.bits, /*useCtx*/true, /*sparse*/false);
                    dataOK = !rerr && std::memcmp(rbuf.data(), orig.data(), nbytes) == 0;
                } catch (const std::string& e) {
                    std::fprintf(stderr, "    exception: %s\n", e.c_str());
                } catch (const std::exception& e) {
                    std::fprintf(stderr, "    exception: %s\n", e.what());
                }

                std::filesystem::remove_all(path, ec);

                if (metaOK && dataOK){
                    std::printf("PASS  %-6s %-8s %s %s%s\n", c.name, comp, order, cfg.tag, vtag);
                } else {
                    std::printf("FAIL  %-6s %-8s %s %s%s  (meta=%d data=%d)\n",
                                c.name, comp, order, cfg.tag, vtag, (int)metaOK, (int)dataOK);
                    failures++;
                }
            }
            }
        }
    }
    }

    // Oversize-chunk guard: a blosc chunk >= ~2 GB uncompressed must be rejected
    // (not silently written as empty chunks). Uses a tiny shape with huge chunks
    // so the guard fires before any large allocation.
    {
        total++;
        const std::vector<uint64_t> tinyShape = {2, 2, 2};
        const std::vector<uint64_t> hugeChunks = {1024, 1024, 1024}; // 1024^3 * 2B = 2 GiB > limit
        const std::string path = dir + "/rt_oversize.zarr";
        std::error_code ec; std::filesystem::remove_all(path, ec);
        std::vector<uint16_t> tiny(2 * 2 * 2, 0);
        uint8_t werr = 1;
        try {
            zarr Zw;
            Zw.set_fileName(path); Zw.set_cname("zstd"); Zw.set_order("F");
            Zw.set_chunks(hugeChunks); Zw.set_dtype("<u2"); Zw.set_shape(tinyShape);
            Zw.write_zarray();
            Zw.set_chunkInfo({0,0,0}, tinyShape);
            werr = parallelWriteZarr(Zw, tiny.data(), {0,0,0}, tinyShape, tinyShape,
                                     16, /*useUuid*/false, /*crop*/false, /*sparse*/false);
        } catch (...) { werr = 1; }
        std::filesystem::remove_all(path, ec);
        if (werr) {
            std::printf("PASS  oversize-chunk rejected\n");
        } else {
            std::printf("FAIL  oversize-chunk not rejected (would silently lose data)\n");
            failures++;
        }
    }

    // fill_value robustness: zarr metadata legally contains "Infinity"/"NaN"
    // fill values (zarr-python writes them for float arrays). Infinity used to
    // funnel into std::stoi and throw out_of_range on every read of the store.
    {
        total++;
        const std::string path = dir + "/rt_fillinf.zarr";
        std::error_code ec; std::filesystem::remove_all(path, ec);
        std::filesystem::create_directories(path);
        {
            std::ofstream z(path + "/.zarray");
            z << "{\"chunks\":[16,16,16],\"compressor\":{\"blocksize\":0,\"clevel\":5,"
                 "\"cname\":\"lz4\",\"id\":\"blosc\",\"shuffle\":1},\"dtype\":\"<f4\","
                 "\"fill_value\":\"Infinity\",\"filters\":null,\"order\":\"F\","
                 "\"shape\":[8,8,8],\"zarr_format\":2}";
        }
        bool ok = false;
        try {
            zarr Zr(path);
            void* buf = parallelReadZarrWriteWrapper(Zr, false, {0,0,0}, {0,0,0});
            ok = buf != NULL;
            free(buf);
        } catch (...) { ok = false; }
        std::filesystem::remove_all(path, ec);
        if (ok) std::printf("PASS  Infinity fill_value read\n");
        else  { std::printf("FAIL  Infinity fill_value read (threw or returned NULL)\n"); failures++; }
    }

    // readZarrParallelHelper (ImageJ path): a windowed read must transpose using
    // the window's dimensions (it used to index with the full array shape), and
    // an unsupported dtype must return NULL instead of crashing.
    {
        total++;
        const uint64_t sx = 24, sy = 20, sz = 12;
        const std::string path = dir + "/rt_imagej.zarr";
        std::error_code ec; std::filesystem::remove_all(path, ec);
        std::vector<uint16_t> src(sx*sy*sz);
        for (uint64_t v = 0; v < src.size(); v++) src[v] = (uint16_t)v;
        bool ok = false;
        try {
            zarr Zw;
            Zw.set_fileName(path); Zw.set_cname("lz4"); Zw.set_order("F");
            Zw.set_chunks({16,16,16}); Zw.set_dtype("<u2"); Zw.set_shape({sx,sy,sz});
            Zw.write_zarray();
            Zw.set_chunkInfo({0,0,0}, {sx,sy,sz});
            if (parallelWriteZarr(Zw, src.data(), {0,0,0}, {sx,sy,sz}, {sx,sy,sz},
                                  16, false, false, false)) throw std::string("write failed");

            const uint64_t x0 = 4, y0 = 2, z0 = 1, x1 = 20, y1 = 18, z1 = 9;
            const uint64_t rx = x1-x0, ry = y1-y0, rz = z1-z0;
            uint16_t* out = (uint16_t*)readZarrParallelHelper(path.c_str(), x0, y0, z0, x1, y1, z1, 1);
            if (!out) throw std::string("helper returned NULL");
            ok = true;
            for (uint64_t k = 0; k < rz && ok; k++)
                for (uint64_t j = 0; j < ry && ok; j++)
                    for (uint64_t i = 0; i < rx && ok; i++)
                        ok = out[j + i*ry + k*ry*rx] == src[(x0+i) + (y0+j)*sx + (z0+k)*sx*sy];
            free(out);

            // Unsupported dtype (complex64) must yield NULL, not a crash.
            const std::string cpath = dir + "/rt_imagej_c8.zarr";
            std::filesystem::remove_all(cpath, ec);
            std::filesystem::create_directories(cpath);
            {
                std::ofstream z(cpath + "/.zarray");
                z << "{\"chunks\":[8,8,8],\"compressor\":{\"blocksize\":0,\"clevel\":5,"
                     "\"cname\":\"lz4\",\"id\":\"blosc\",\"shuffle\":1},\"dtype\":\"<c8\","
                     "\"fill_value\":0,\"filters\":null,\"order\":\"F\","
                     "\"shape\":[8,8,8],\"zarr_format\":2}";
            }
            ok = ok && readZarrParallelHelper(cpath.c_str(), 0, 0, 0, 8, 8, 8, 1) == NULL;
            std::filesystem::remove_all(cpath, ec);
        } catch (const std::string& e) {
            std::fprintf(stderr, "    imagej exception: %s\n", e.c_str()); ok = false;
        } catch (...) { ok = false; }
        std::filesystem::remove_all(path, ec);
        if (ok) std::printf("PASS  ImageJ windowed read + NULL guard\n");
        else  { std::printf("FAIL  ImageJ windowed read + NULL guard\n"); failures++; }
    }

    // Bbox (crop) writes into an existing array must keep the existing data around
    // the box in partially covered chunks, in both orders:
    //   partial  : chunk-aligned start, box ends mid-chunk
    //   unaligned: start not on a chunk boundary
    //   edge     : box ends inside the array's last, partial chunk (y and z)
    // and a missing chunk file must read back as the fill value in both orders.
    {
        typedef std::vector<uint64_t> V;
        const V shape = {40, 24, 18};
        const uint64_t n = shape[0]*shape[1]*shape[2];
        std::mt19937 srng(99);
        std::vector<uint16_t> A(n);
        for (auto& v : A) v = 1 + srng() % 60000;   // never 0, so lost data is visible

        auto writeBox = [&](const std::string& path, const char* order, const std::vector<uint16_t>& data,
                            const V& s, const V& e, bool create) {
            zarr Z;
            if (create) {
                Z.set_fileName(path); Z.set_cname("lz4"); Z.set_order(order); Z.set_chunks({16,16,16});
                Z.set_dimension_separator("."); Z.set_dtype("<u2"); Z.set_shape(shape); Z.write_zarray();
            } else {
                Z = zarr(path);   // crop into the existing array, as the bbox writers do
            }
            Z.set_chunkInfo(s, e);
            // uuid temp files + rename, as the MEX writer does by default: rewriting
            // existing chunks must replace them (Windows' rename() cannot)
            return parallelWriteZarr(Z, (void*)data.data(), s, e, {e[0]-s[0], e[1]-s[1], e[2]-s[2]},
                                     16, /*useUuid*/true, /*crop*/!create, false) == 0;
        };

        struct Box { const char* name; V s, e; };
        const Box boxes[] = { {"partial", {16,0,0}, {29,13,11}}, {"unaligned", {5,3,2}, {29,13,11}},
                              {"edge", {0,0,0}, {40,20,17}} };
        for (const char* order : {"F", "C"}) {
            for (const Box& b : boxes) {
                total++;
                const std::string path = dir + "/rt_crop_" + b.name + "_" + order + ".zarr";
                std::error_code ec; std::filesystem::remove_all(path, ec);
                const V ws = {b.e[0]-b.s[0], b.e[1]-b.s[1], b.e[2]-b.s[2]};
                std::vector<uint16_t> B(ws[0]*ws[1]*ws[2]);
                for (auto& v : B) v = 60001 + srng() % 5000;
                std::vector<uint16_t> expect = A;
                for (uint64_t z = 0; z < ws[2]; z++) for (uint64_t y = 0; y < ws[1]; y++) for (uint64_t x = 0; x < ws[0]; x++)
                    expect[(b.s[0]+x) + (b.s[1]+y)*shape[0] + (b.s[2]+z)*shape[0]*shape[1]] = B[x + y*ws[0] + z*ws[0]*ws[1]];
                bool ok = false;
                try {
                    if (writeBox(path, order, A, {0,0,0}, shape, true) && writeBox(path, order, B, b.s, b.e, false)) {
                        zarr Zr(path);
                        std::vector<uint16_t> got(n, 0);
                        Zr.set_chunkInfo({0,0,0}, shape);
                        ok = parallelReadZarr(Zr, got.data(), {0,0,0}, shape, shape, 16, true, false) == 0 && got == expect;
                    }
                } catch (...) { ok = false; }
                std::filesystem::remove_all(path, ec);
                std::printf("%s  crop write (%s) %s\n", ok ? "PASS" : "FAIL", b.name, order);
                if (!ok) failures++;
            }

            total++;
            const std::string path = dir + std::string("/rt_missing_") + order + ".zarr";
            std::error_code ec; std::filesystem::remove_all(path, ec);
            bool ok = false;
            try {
                writeBox(path, order, A, {0,0,0}, shape, true);
                std::string meta;
                { std::ifstream f(path + "/.zarray"); meta.assign(std::istreambuf_iterator<char>(f), {}); }
                const size_t p = meta.find("\"fill_value\"");
                meta = meta.substr(0, p) + "\"fill_value\": 7" + meta.substr(meta.find_first_of(",}", p));
                { std::ofstream f(path + "/.zarray"); f << meta; }
                std::filesystem::remove(path + "/1.0.0");
                zarr Zr(path);
                uint16_t* got = (uint16_t*)parallelReadZarrWriteWrapper(Zr, false, {0,0,0}, shape);
                ok = got != nullptr;
                for (uint64_t z = 0; ok && z < shape[2]; z++) for (uint64_t y = 0; ok && y < shape[1]; y++)
                    for (uint64_t x = 0; ok && x < shape[0]; x++) {
                        const size_t i = x + y*shape[0] + z*shape[0]*shape[1];
                        const bool missing = x >= 16 && x < 32 && y < 16 && z < 16;
                        ok = got[i] == (missing ? 7 : A[i]);   // the fill value, as zarr-python reads it
                    }
                free(got);
            } catch (...) { ok = false; }
            std::filesystem::remove_all(path, ec);
            std::printf("%s  missing chunk reads as fill_value %s\n", ok ? "PASS" : "FAIL", order);
            if (!ok) failures++;
        }
    }

    // Arrays of other numbers of dimensions: 1D, 2D, 4D and 5D, both orders, with
    // partial edge chunks; sub-region reads; an unaligned crop write; and the
    // sharded, "/" separator and subfolder layouts
    {
        typedef std::vector<uint64_t> V;
        // Region [s, e) of an F-order array with the given shape
        auto extract = [](const std::vector<uint8_t>& full, const V& shape, const V& s, const V& e, uint64_t bytes){
            const uint64_t n = shape.size();
            uint64_t count = 1; for (uint64_t d = 0; d < n; d++) count *= e[d]-s[d];
            std::vector<uint8_t> out(count*bytes);
            for (uint64_t i = 0; i < count; i++){
                uint64_t r = i, off = 0, stride = 1;
                for (uint64_t d = 0; d < n; d++){
                    const uint64_t ext = e[d]-s[d];
                    off += (s[d] + r%ext)*stride;
                    r /= ext; stride *= shape[d];
                }
                std::memcpy(&out[i*bytes], &full[off*bytes], bytes);
            }
            return out;
        };
        struct NdCase { const char* name; V shape, chunks, inner, sub; const char* sep; };
        V shape70(70, 1), chunks70(70, 1);
        shape70[2] = 5;  shape70[35] = 3; shape70[66] = 4; shape70[69] = 6;
        chunks70[2] = 2; chunks70[35] = 2; chunks70[66] = 3; chunks70[69] = 4;
        const NdCase ndCases[] = {
            {"0d", {}, {}, {}, {}, "."},
            {"1d", {1000}, {128}, {}, {}, "."},
            {"2d", {300, 170}, {128, 64}, {}, {}, "."},
            {"4d", {5, 30, 40, 50}, {2, 16, 16, 32}, {}, {}, "."},
            {"5d", {2, 3, 20, 30, 40}, {1, 2, 16, 16, 32}, {}, {}, "."},
            {"4d_slash", {5, 30, 40, 50}, {2, 16, 16, 32}, {}, {}, "/"},
            {"4d_subf", {5, 30, 40, 50}, {2, 16, 16, 32}, {}, {2, 1, 1, 2}, "."},
            {"4d_shard", {5, 30, 40, 50}, {4, 32, 32, 64}, {2, 16, 16, 32}, {}, "."},
            {"2d_shard", {300, 170}, {256, 128}, {128, 64}, {}, "."},
            {"70d", shape70, chunks70, {}, {}, "."},
        };
        std::mt19937 nrng(4242);
        for (const NdCase& c : ndCases){
            for (const char* dtype : {"<u2", "<f8"}){
                for (const char* order : {"F", "C"}){
                for (const int fmt : {2, 3}){
                    if (fmt == 3 && !c.sub.empty()) continue;
                    total++;
                    const uint64_t bytes = dtype[2]-'0', n = c.shape.size();
                    uint64_t count = 1; for (uint64_t v : c.shape) count *= v;
                    std::vector<uint8_t> orig(count*bytes);
                    for (auto& b : orig) b = (uint8_t)byteDist(rng);
                    const std::string path = dir + "/rt_" + c.name + "_" + (dtype+1) + "_" + order + (fmt == 3 ? "_v3" : "") + ".zarr";
                    std::error_code ec; std::filesystem::remove_all(path, ec);
                    bool ok = false;
                    try {
                        zarr Zw;
                        Zw.set_fileName(path);
                        if (fmt == 3) Zw.set_zarr_format(3);
                        Zw.set_cname("lz4"); Zw.set_order(order); Zw.set_chunks(c.chunks);
                        Zw.set_dimension_separator(c.sep); Zw.set_dtype(dtype); Zw.set_shape(c.shape);
                        if (!c.sub.empty()) Zw.set_subfolders(c.sub);
                        if (!c.inner.empty()){ Zw.set_shard(true); Zw.set_chunk_shape(c.inner); }
                        Zw.write_zarray();
                        const V zeros(n, 0);
                        Zw.set_chunkInfo(zeros, c.shape);
                        if (parallelWriteZarr(Zw, orig.data(), zeros, c.shape, c.shape, bytes*8, true, false, false))
                            throw std::string("write error: ") + Zw.get_errString();
                        zarr Zr(path);
                        ok = Zr.get_ndims() == n;
                        Zr.set_chunkInfo(zeros, c.shape);
                        std::vector<uint8_t> back(count*bytes, 0);
                        ok = ok && parallelReadZarr(Zr, back.data(), zeros, c.shape, c.shape, bytes*8, true, false) == 0 && back == orig;
                        // random sub-regions
                        for (int t = 0; ok && t < 3; t++){
                            V s(n), e(n), rs(n);
                            for (uint64_t d = 0; d < n; d++){
                                s[d] = nrng() % c.shape[d];
                                e[d] = s[d] + 1 + nrng() % (c.shape[d]-s[d]);
                                rs[d] = e[d]-s[d];
                            }
                            const std::vector<uint8_t> exp = extract(orig, c.shape, s, e, bytes);
                            std::vector<uint8_t> got(exp.size(), 0);
                            zarr Zs(path);
                            Zs.set_chunkInfo(s, e);
                            ok = parallelReadZarr(Zs, got.data(), s, e, rs, bytes*8, true, false) == 0 && got == exp;
                        }
                        // The same array given in C order (read through its strides) must store
                        // the same data, and a C-order read must return that buffer exactly
                        if (ok){
                            V cs(n);
                            { uint64_t acc = 1; for (int64_t d = (int64_t)n-1; d >= 0; d--){ cs[d] = acc; acc *= c.shape[d]; } }
                            std::vector<uint8_t> corig(count*bytes);
                            for (uint64_t i = 0; i < count; i++){
                                uint64_t r = i, coff = 0;
                                for (uint64_t d = 0; d < n; d++){ coff += (r%c.shape[d])*cs[d]; r /= c.shape[d]; }
                                std::memcpy(&corig[coff*bytes], &orig[i*bytes], bytes);
                            }
                            const std::string cpath = path + "_cin";
                            std::filesystem::remove_all(cpath, ec);
                            zarr Zc;
                            Zc.set_fileName(cpath);
                            if (fmt == 3) Zc.set_zarr_format(3);
                            Zc.set_cname("lz4"); Zc.set_order(order); Zc.set_chunks(c.chunks);
                            Zc.set_dimension_separator(c.sep); Zc.set_dtype(dtype); Zc.set_shape(c.shape);
                            if (!c.sub.empty()) Zc.set_subfolders(c.sub);
                            if (!c.inner.empty()){ Zc.set_shard(true); Zc.set_chunk_shape(c.inner); }
                            Zc.write_zarray();
                            Zc.set_chunkInfo(zeros, c.shape);
                            if (parallelWriteZarr(Zc, corig.data(), zeros, c.shape, c.shape, cs, bytes*8, true, false, false))
                                throw std::string("C-order input write error: ") + Zc.get_errString();
                            zarr Zcr(cpath);
                            Zcr.set_chunkInfo(zeros, c.shape);
                            std::fill(back.begin(), back.end(), 0);
                            ok = parallelReadZarr(Zcr, back.data(), zeros, c.shape, c.shape, bytes*8, true, false) == 0 && back == orig;
                            zarr Zco(cpath);
                            void* cback = parallelReadZarrWriteWrapper(Zco, false, zeros, c.shape, true);
                            ok = ok && cback && std::memcmp(cback, corig.data(), count*bytes) == 0;
                            free(cback);
                            std::filesystem::remove_all(cpath, ec);
                        }
                        // unaligned crop write into the existing array (non-sharded layouts)
                        if (ok && c.inner.empty() && n > 1){
                            V s(n), e(n), rs(n);
                            uint64_t pc = 1;
                            for (uint64_t d = 0; d < n; d++){
                                s[d] = std::min<uint64_t>(3, c.shape[d]-1);
                                e[d] = c.shape[d] > 3 ? std::max<uint64_t>(s[d]+1, c.shape[d]-2) : c.shape[d];
                                rs[d] = e[d]-s[d]; pc *= rs[d];
                            }
                            std::vector<uint8_t> patch(pc*bytes);
                            for (auto& b : patch) b = (uint8_t)byteDist(rng);
                            zarr Zc(path);
                            Zc.set_chunkInfo(s, e);
                            if (parallelWriteZarr(Zc, patch.data(), s, e, rs, bytes*8, true, true, false))
                                throw std::string("crop error: ") + Zc.get_errString();
                            std::vector<uint8_t> exp = orig;
                            for (uint64_t i = 0; i < pc; i++){
                                uint64_t r = i, off = 0, stride = 1;
                                for (uint64_t d = 0; d < n; d++){ off += (s[d] + r%rs[d])*stride; r /= rs[d]; stride *= c.shape[d]; }
                                std::memcpy(&exp[off*bytes], &patch[i*bytes], bytes);
                            }
                            zarr Zr2(path);
                            Zr2.set_chunkInfo(zeros, c.shape);
                            std::fill(back.begin(), back.end(), 0);
                            ok = parallelReadZarr(Zr2, back.data(), zeros, c.shape, c.shape, bytes*8, true, false) == 0 && back == exp;
                            // the same region again, from a C-order patch
                            if (ok){
                                V pcs(n);
                                { uint64_t acc = 1; for (int64_t d = (int64_t)n-1; d >= 0; d--){ pcs[d] = acc; acc *= rs[d]; } }
                                std::vector<uint8_t> patch2(pc*bytes), patch2C(pc*bytes);
                                for (auto& b : patch2) b = (uint8_t)byteDist(rng);
                                for (uint64_t i = 0; i < pc; i++){
                                    uint64_t r = i, coff = 0, off = 0, stride = 1;
                                    for (uint64_t d = 0; d < n; d++){
                                        coff += (r%rs[d])*pcs[d];
                                        off += (s[d] + r%rs[d])*stride;
                                        r /= rs[d]; stride *= c.shape[d];
                                    }
                                    std::memcpy(&patch2C[coff*bytes], &patch2[i*bytes], bytes);
                                    std::memcpy(&exp[off*bytes], &patch2[i*bytes], bytes);
                                }
                                zarr Zc2(path);
                                Zc2.set_chunkInfo(s, e);
                                if (parallelWriteZarr(Zc2, patch2C.data(), s, e, rs, pcs, bytes*8, true, true, false))
                                    throw std::string("C-order crop error: ") + Zc2.get_errString();
                                zarr Zr3(path);
                                Zr3.set_chunkInfo(zeros, c.shape);
                                std::fill(back.begin(), back.end(), 0);
                                ok = parallelReadZarr(Zr3, back.data(), zeros, c.shape, c.shape, bytes*8, true, false) == 0 && back == exp;
                            }
                        }
                    } catch (const std::string& e) {
                        std::fprintf(stderr, "    exception: %s\n", e.c_str()); ok = false;
                    } catch (...) { ok = false; }
                    std::filesystem::remove_all(path, ec);
                    std::printf("%s  %-9s %s %s%s\n", ok ? "PASS" : "FAIL", c.name, dtype+1, order, fmt == 3 ? " v3" : "");
                    if (!ok) failures++;
                }
                }
            }
        }
    }

    // Sharding and sparse writes: crop writes into sharded arrays, empty inner chunks
    // (including the first one of a shard, and whole empty shards), rewrites without
    // temporary files, sharding with subfolders, shard shapes that are not a multiple of
    // the inner chunk shape, writes into arrays whose folders are missing, shard indexes
    // at the start of the file and without a checksum (read and written), damaged shards
    // (an error, not a crash), and rewrites in which chunks become all zero (fill value
    // 0, and 7). In Zarr v2 and v3 arrays (v3 has no subfolders, and needs shard shapes
    // that are a multiple of the inner chunk shape).
    {
        typedef std::vector<uint64_t> V;
        std::mt19937 srng(5150);
        // the format of the arrays being tested, their metadata file, and the name of
        // the shard file at shard indices i, j, k
        int fmt = 2;
        auto metaName = [&](){ return std::string(fmt == 3 ? "zarr.json" : ".zarray"); };
        auto shardKey = [&](int i, int j, int k){
            const std::string sep = fmt == 3 ? "/" : ".";
            return (fmt == 3 ? "c/" : "") + std::to_string(i) + sep + std::to_string(j) + sep + std::to_string(k);
        };
        // F-order uint16 data; zero marks inner chunks (of size `inner`) to leave empty
        auto makeData = [&](const V& shape, const V& inner, std::function<bool(const V&)> zero){
            std::vector<uint16_t> d(shape[0]*shape[1]*shape[2]);
            for (uint64_t z = 0; z < shape[2]; z++) for (uint64_t y = 0; y < shape[1]; y++) for (uint64_t x = 0; x < shape[0]; x++)
                d[x + y*shape[0] + z*shape[0]*shape[1]] = zero({x/inner[0], y/inner[1], z/inner[2]}) ? 0 : (uint16_t)(1 + srng() % 60000);
            return d;
        };
        // (sep: "." in v2 and "/" in v3 when not given)
        auto create = [&](const std::string& path, const V& shape, const V& chunks, const V& inner, const V& sub,
                          const char* order, const std::string& fill, const char* sep = nullptr){
            std::error_code ec; std::filesystem::remove_all(path, ec);
            zarr Z;
            Z.set_fileName(path);
            if (fmt == 3) Z.set_zarr_format(3);
            Z.set_cname("zstd"); Z.set_clevel(1); Z.set_order(order); Z.set_chunks(chunks);
            Z.set_dtype("<u2"); Z.set_shape(shape); Z.set_fill_value(fill);
            Z.set_dimension_separator(sep ? sep : fmt == 3 ? "/" : ".");
            if (!inner.empty()){ Z.set_shard(true); Z.set_chunk_shape(inner); }
            if (!sub.empty()) Z.set_subfolders(sub);
            Z.write_zarray();
        };
        auto write = [&](const std::string& path, std::vector<uint16_t>& d, const V& s, const V& e, bool crop, bool uuid){
            zarr Z(path);
            Z.set_chunkInfo(s, e);
            const V ws = {e[0]-s[0], e[1]-s[1], e[2]-s[2]};
            if (parallelWriteZarr(Z, d.data(), s, e, ws, 16, uuid, crop, true)) throw std::string("write: ") + Z.get_errString();
        };
        // read the whole array; empty if the read fails (the error goes in errOut)
        auto read = [&](const std::string& path, std::string* errOut = nullptr){
            zarr Z(path);
            const V shape = {Z.get_shape(0), Z.get_shape(1), Z.get_shape(2)};
            std::vector<uint16_t> d(shape[0]*shape[1]*shape[2], 0);
            Z.set_chunkInfo({0, 0, 0}, shape);
            if (parallelReadZarr(Z, d.data(), {0, 0, 0}, shape, shape, 16, true, false)){
                if (errOut) *errOut = Z.get_errString();
                d.clear();
            }
            return d;
        };
        // read the region [s, e); empty if the read fails
        auto readRegion = [&](const std::string& path, const V& s, const V& e){
            zarr Z(path);
            const V ws = {e[0]-s[0], e[1]-s[1], e[2]-s[2]};
            std::vector<uint16_t> d(ws[0]*ws[1]*ws[2], 0);
            Z.set_chunkInfo(s, e);
            if (parallelReadZarr(Z, d.data(), s, e, ws, 16, true, false)) d.clear();
            return d;
        };
        auto cut = [](const std::vector<uint16_t>& full, const V& shape, const V& s, const V& e){
            const V ws = {e[0]-s[0], e[1]-s[1], e[2]-s[2]};
            std::vector<uint16_t> out(ws[0]*ws[1]*ws[2]);
            for (uint64_t z = 0; z < ws[2]; z++) for (uint64_t y = 0; y < ws[1]; y++) for (uint64_t x = 0; x < ws[0]; x++)
                out[x + y*ws[0] + z*ws[0]*ws[1]] = full[(s[0]+x) + (s[1]+y)*shape[0] + (s[2]+z)*shape[0]*shape[1]];
            return out;
        };
        auto paste = [](std::vector<uint16_t>& full, const V& shape, const std::vector<uint16_t>& patch, const V& s, const V& e){
            const V ws = {e[0]-s[0], e[1]-s[1], e[2]-s[2]};
            for (uint64_t z = 0; z < ws[2]; z++) for (uint64_t y = 0; y < ws[1]; y++) for (uint64_t x = 0; x < ws[0]; x++)
                full[(s[0]+x) + (s[1]+y)*shape[0] + (s[2]+z)*shape[0]*shape[1]] = patch[x + y*ws[0] + z*ws[0]*ws[1]];
        };
        // shard files of an array (every file but the metadata)
        auto shardFiles = [](const std::string& path){
            std::vector<std::string> out;
            for (const auto& p : std::filesystem::recursive_directory_iterator(path))
                if (p.is_regular_file() && p.path().filename() != ".zarray" && p.path().filename() != "zarr.json")
                    out.push_back(p.path().string());
            return out;
        };
        auto readFile = [](const std::string& p){
            std::ifstream f(p, std::ios::binary);
            return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        };
        auto writeFile = [](const std::string& p, const std::vector<uint8_t>& b){
            std::ofstream(p, std::ios::binary | std::ios::trunc).write((const char*)b.data(), b.size());
        };
        // set a key of the sharding codec's configuration in the metadata
        auto setShardConfig = [&](const std::string& path, const std::string& key, const json& value){
            json meta;
            { std::ifstream f(path + "/" + metaName()); f >> meta; }
            meta["codecs"][0]["configuration"][key] = value;
            std::ofstream(path + "/" + metaName()) << meta.dump();
        };
        auto check = [&](const char* name, std::function<bool()> test){
            total++;
            bool ok = false;
            try { ok = test(); }
            catch (const std::string& e) { std::fprintf(stderr, "    exception: %s\n", e.c_str()); }
            catch (...) {}
            std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
            if (!ok) failures++;
        };
        const V shape = {70, 45, 33}, zero3 = {0, 0, 0};
        auto none = [](const V&){ return false; };

        for (const int f : {2, 3}) for (const char* order : {"F", "C"}){
            fmt = f;
            const std::string base = dir + "/rt_shard_" + order + (fmt == 3 ? "_v3" : "");
            const std::string tag = std::string(" (") + order + " order" + (fmt == 3 ? ", v3)" : ")");

            check(("shard crop writes: unaligned, across shards, at the edge, with and without temporary files" + tag).c_str(), [&](){
                bool ok = true;
                for (bool uuid : {true, false}){
                    const std::string p = base + "_crop.zarr";
                    create(p, shape, {32, 32, 20}, {16, 16, 10}, {}, order, "0");
                    std::vector<uint16_t> exp = makeData(shape, {16, 16, 10}, none);
                    write(p, exp, zero3, shape, false, uuid);
                    ok = ok && read(p) == exp;
                    // (the first region covers several inner chunks of a shard partly)
                    for (const auto& region : std::vector<std::pair<V, V>>{{{5, 7, 3}, {61, 40, 29}}, {{64, 32, 30}, {70, 45, 33}}, {{0, 0, 0}, {16, 16, 10}}}){
                        const V s = region.first, e = region.second;
                        std::vector<uint16_t> patch = makeData({e[0]-s[0], e[1]-s[1], e[2]-s[2]}, {1, 1, 1}, none);
                        write(p, patch, s, e, true, uuid);
                        paste(exp, shape, patch, s, e);
                        ok = ok && read(p) == exp;
                    }
                    // and no temporary files are left behind: only the metadata and the 3x2x2 shards
                    std::set<std::string> names, want = {metaName()};
                    for (const auto& f : std::filesystem::recursive_directory_iterator(p)){
                        if (!f.is_regular_file()) continue;
                        std::string name = f.path().string().substr(p.size() + 1);
                        std::replace(name.begin(), name.end(), '\\', '/');
                        names.insert(name);
                    }
                    for (int i = 0; i < 3; i++) for (int j = 0; j < 2; j++) for (int k = 0; k < 2; k++)
                        want.insert(shardKey(i, j, k));
                    ok = ok && names == want;
                }
                return ok;
            });

            check(("shard empty inner chunks: first of each shard, a whole shard" + tag).c_str(), [&](){
                const std::string p = base + "_sparse.zarr";
                create(p, shape, {32, 32, 20}, {16, 16, 10}, {}, order, "0");
                // inner chunk grid is 5x3x4: zero the first inner chunk of every shard,
                // every other chunk of the second shard row, and the whole shard (0, 1, 0)
                std::vector<uint16_t> exp = makeData(shape, {16, 16, 10}, [](const V& c){
                    return (c[0] % 2 == 0 && c[1] % 2 == 0 && c[2] % 2 == 0) || (c[1] == 1 && (c[0] + c[2]) % 2 == 1) ||
                           (c[0] < 2 && c[1] >= 2 && c[2] < 2);
                });
                write(p, exp, zero3, shape, false, true);
                return read(p) == exp;
            });

            check(("shard rewrite without temporary files, first inner chunks emptied" + tag).c_str(), [&](){
                const std::string p = base + "_rewrite.zarr";
                create(p, shape, {32, 32, 20}, {16, 16, 10}, {}, order, "0");
                std::vector<uint16_t> a = makeData(shape, {16, 16, 10}, none);
                write(p, a, zero3, shape, false, false);
                std::vector<uint16_t> b = makeData(shape, {16, 16, 10}, [](const V& c){ return c[0] % 2 == 0 && c[1] % 2 == 0 && c[2] % 2 == 0; });
                write(p, b, zero3, shape, false, false);
                return read(p) == b;
            });

            check(("shards in subfolders" + tag).c_str(), [&](){
                const std::string p = base + "_subf.zarr";
                // (Zarr v3 cannot describe subfolders)
                if (fmt == 3){
                    try { create(p, shape, {16, 16, 16}, {8, 8, 8}, {2, 2, 1}, order, "0"); return false; }
                    catch (const std::string& e){ return e.rfind("v3Unsupported:", 0) == 0 && !fileExists(p + "/zarr.json"); }
                }
                create(p, shape, {16, 16, 16}, {8, 8, 8}, {2, 2, 1}, order, "0");
                std::vector<uint16_t> exp = makeData(shape, {8, 8, 8}, none);
                write(p, exp, zero3, shape, false, true);
                std::vector<uint16_t> patch = makeData({30, 20, 10}, {1, 1, 1}, none);
                write(p, patch, {20, 10, 5}, {50, 30, 15}, true, true);
                paste(exp, shape, patch, {20, 10, 5}, {50, 30, 15});
                return read(p) == exp;
            });

            check(("shard shapes that are not a multiple of the inner chunk shape" + tag).c_str(), [&](){
                // a 24x20x15 shard of 16x16x10 inner chunks holds 2x2x2 whole inner chunks,
                // so it spans 32x32x20; these regions start in a different shard than
                // their coordinates divided by the shard shape would say
                const std::string p = base + "_nondiv.zarr";
                // (Zarr v3 needs whole inner chunks in a shard)
                if (fmt == 3){
                    try { create(p, shape, {24, 20, 15}, {16, 16, 10}, {}, order, "0"); return false; }
                    catch (const std::string& e){ return e.rfind("v3Unsupported:", 0) == 0 && !fileExists(p + "/zarr.json"); }
                }
                create(p, shape, {24, 20, 15}, {16, 16, 10}, {}, order, "0");
                std::vector<uint16_t> exp = makeData(shape, {16, 16, 10}, none);
                write(p, exp, zero3, shape, false, true);
                bool ok = read(p) == exp;
                for (const auto& region : std::vector<std::pair<V, V>>{{{50, 21, 16}, {60, 30, 19}}, {{40, 30, 25}, {70, 45, 33}}}){
                    const V s = region.first, e = region.second;
                    ok = ok && readRegion(p, s, e) == cut(exp, shape, s, e);
                    std::vector<uint16_t> patch = makeData({e[0]-s[0], e[1]-s[1], e[2]-s[2]}, {1, 1, 1}, none);
                    write(p, patch, s, e, true, true);
                    paste(exp, shape, patch, s, e);
                    ok = ok && read(p) == exp && readRegion(p, s, e) == patch;
                }
                return ok;
            });

            check(("writes into arrays whose folders are missing ('/' separator, subfolders)" + tag).c_str(), [&](){
                struct Layout { V chunks, inner, sub; const char* sep; };
                bool ok = true;
                for (const Layout& l : {Layout{{16, 16, 10}, {}, {}, "/"}, Layout{{32, 32, 20}, {16, 16, 10}, {}, "/"},
                                        Layout{{16, 16, 10}, {}, {2, 2, 1}, "."}, Layout{{32, 32, 20}, {16, 16, 10}, {2, 1, 2}, "/"}}){
                    if (fmt == 3 && !l.sub.empty()) continue;
                    const std::string p = base + "_nofolders.zarr";
                    // remove the folders write_zarray made, as if another program had made the array
                    auto removeFolders = [&](){
                        std::vector<std::filesystem::path> folders;
                        for (const auto& e : std::filesystem::directory_iterator(p)) if (e.is_directory()) folders.push_back(e.path());
                        for (const auto& f : folders) std::filesystem::remove_all(f);
                    };
                    create(p, shape, l.chunks, l.inner, l.sub, order, "0", l.sep);
                    removeFolders();
                    std::vector<uint16_t> exp = makeData(shape, {16, 16, 10}, none);
                    write(p, exp, zero3, shape, false, true);
                    ok = ok && read(p) == exp;
                    // and a crop write into an empty one
                    create(p, shape, l.chunks, l.inner, l.sub, order, "0", l.sep);
                    removeFolders();
                    std::vector<uint16_t> patch = makeData({30, 20, 10}, {1, 1, 1}, none);
                    write(p, patch, {20, 10, 5}, {50, 30, 15}, true, true);
                    exp.assign(exp.size(), 0);
                    paste(exp, shape, patch, {20, 10, 5}, {50, 30, 15});
                    ok = ok && read(p) == exp;
                }
                return ok;
            });

            check(("shard index at the start, and without a checksum" + tag).c_str(), [&](){
                const std::string p = base + "_sparse.zarr";   // written above, some inner chunks empty
                const std::vector<uint16_t> exp = read(p);
                const uint64_t n = 2*2*2, ib = n*16;            // 2x2x2 inner chunks per shard
                // move each index to the front of its shard (offsets shift by the index size)
                for (const std::string& f : shardFiles(p)){
                    const std::vector<uint8_t> b = readFile(f);
                    std::vector<uint64_t> idx(2*n);
                    std::memcpy(idx.data(), &b[b.size()-ib-4], ib);
                    for (uint64_t i = 0; i < n; i++) if (idx[2*i] != UINT64_MAX) idx[2*i] += ib + 4;
                    const uint32_t crc = crc32c((const uint8_t*)idx.data(), ib);
                    std::vector<uint8_t> out(ib + 4);
                    std::memcpy(out.data(), idx.data(), ib);
                    std::memcpy(&out[ib], &crc, 4);
                    out.insert(out.end(), b.begin(), b.end()-ib-4);
                    writeFile(f, out);
                }
                setShardConfig(p, "index_location", "start");
                bool ok = !exp.empty() && read(p) == exp;
                // and with no checksum after the index
                for (const std::string& f : shardFiles(p)){
                    std::vector<uint8_t> b = readFile(f);
                    std::vector<uint64_t> idx(2*n);
                    std::memcpy(idx.data(), b.data(), ib);
                    for (uint64_t i = 0; i < n; i++) if (idx[2*i] != UINT64_MAX) idx[2*i] -= 4;
                    std::memcpy(b.data(), idx.data(), ib);
                    b.erase(b.begin()+ib, b.begin()+ib+4);
                    writeFile(f, b);
                }
                setShardConfig(p, "index_codecs", json::array({{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}}));
                return ok && read(p) == exp;
            });

            check(("writes keep the array's index settings: at the start, without a checksum" + tag).c_str(), [&](){
                bool ok = true;
                for (int config = 0; config < 3; config++){
                    const bool atStart = config < 2, checksum = config != 1;
                    const std::string p = base + "_index.zarr";
                    create(p, shape, {32, 32, 20}, {16, 16, 10}, {}, order, "0");
                    if (atStart) setShardConfig(p, "index_location", "start");
                    json codecs = json::array({{{"name", "bytes"}, {"configuration", {{"endian", "little"}}}}});
                    if (checksum) codecs.push_back({{"name", "crc32c"}});
                    setShardConfig(p, "index_codecs", codecs);
                    // some empty inner chunks, then a crop write without temporary files
                    std::vector<uint16_t> exp = makeData(shape, {16, 16, 10}, [](const V& c){ return (c[0] + c[1] + c[2]) % 3 == 0; });
                    write(p, exp, zero3, shape, false, true);
                    ok = ok && read(p) == exp;
                    std::vector<uint16_t> patch = makeData({30, 20, 10}, {1, 1, 1}, none);
                    write(p, patch, {20, 10, 5}, {50, 30, 15}, true, false);
                    paste(exp, shape, patch, {20, 10, 5}, {50, 30, 15});
                    ok = ok && read(p) == exp;
                    // each shard has its index where the metadata says, with or without a
                    // checksum, and its inner chunks fill the rest of the file
                    const uint64_t n = 2*2*2, ib = n*16, total = ib + (checksum ? 4 : 0);
                    for (const std::string& f : shardFiles(p)){
                        const std::vector<uint8_t> b = readFile(f);
                        if (b.size() < total){ ok = false; continue; }
                        const uint64_t at = atStart ? 0 : b.size() - total;
                        std::vector<uint64_t> idx(2*n);
                        std::memcpy(idx.data(), &b[at], ib);
                        if (checksum){
                            uint32_t crc;
                            std::memcpy(&crc, &b[at + ib], 4);
                            ok = ok && crc == crc32c((const uint8_t*)idx.data(), ib);
                        }
                        uint64_t data = 0;
                        for (uint64_t i = 0; i < n; i++){
                            if (idx[2*i] == UINT64_MAX && idx[2*i+1] == UINT64_MAX) continue;
                            ok = ok && idx[2*i] >= (atStart ? total : 0) && idx[2*i] + idx[2*i+1] <= (atStart ? b.size() : b.size() - total);
                            data += idx[2*i+1];
                        }
                        ok = ok && data + total == b.size();
                    }
                }
                return ok;
            });

            check(("damaged shards are reported, not read" + tag).c_str(), [&](){
                const std::string p = base + "_damaged.zarr";
                create(p, shape, {32, 32, 20}, {16, 16, 10}, {}, order, "0");
                std::vector<uint16_t> d = makeData(shape, {16, 16, 10}, none);
                write(p, d, zero3, shape, false, true);
                const std::string f = p + "/" + shardKey(1, 0, 0);
                const std::vector<uint8_t> good = readFile(f);
                const uint64_t ib = 8*16;
                std::string e1, e2, e3;
                std::vector<uint8_t> b = good;
                b[b.size()-ib-4+3] ^= 0x40;                     // a flipped bit in the index
                writeFile(f, b);
                bool ok = read(p, &e1).empty() && e1.find("checksum") != std::string::npos;
                b = good;                                        // an entry past the end, with a valid checksum
                std::vector<uint64_t> idx(16);
                std::memcpy(idx.data(), &b[b.size()-ib-4], ib);
                idx[1] = 1ULL << 40;
                const uint32_t crc = crc32c((const uint8_t*)idx.data(), ib);
                std::memcpy(&b[b.size()-ib-4], idx.data(), ib);
                std::memcpy(&b[b.size()-4], &crc, 4);
                writeFile(f, b);
                ok = ok && read(p, &e2).empty() && e2.find("outside the file") != std::string::npos;
                writeFile(f, std::vector<uint8_t>(good.begin(), good.begin()+10));   // truncated
                ok = ok && read(p, &e3).empty() && e3.find("smaller than its index") != std::string::npos;
                writeFile(f, good);
                return ok && read(p) == d;
            });

            check(("chunks rewritten as all zeros read back as zeros, sharded or not" + tag).c_str(), [&](){
                bool ok = true;
                for (bool sharded : {false, true}){
                    const std::string p = base + (sharded ? "_zeros_shard.zarr" : "_zeros.zarr");
                    create(p, shape, sharded ? V{32, 32, 20} : V{16, 16, 10}, sharded ? V{16, 16, 10} : V{}, {}, order, "0");
                    std::vector<uint16_t> a = makeData(shape, {16, 16, 10}, none), z(a.size(), 0);
                    write(p, a, zero3, shape, false, true);
                    write(p, z, zero3, shape, false, true);
                    ok = ok && read(p) == z;
                    // and a crop that zeroes a region spanning whole chunks
                    write(p, a, zero3, shape, false, true);
                    std::vector<uint16_t> zp(48*32*20, 0);
                    write(p, zp, {0, 0, 0}, {48, 32, 20}, true, true);
                    paste(a, shape, zp, {0, 0, 0}, {48, 32, 20});
                    ok = ok && read(p) == a;
                }
                return ok;
            });

            check(("all-zero data in an array with fill_value 7 reads back as zeros" + tag).c_str(), [&](){
                const std::string p = base + "_fill7.zarr";
                create(p, shape, {16, 16, 10}, {}, {}, order, "7");
                std::vector<uint16_t> z(shape[0]*shape[1]*shape[2], 0);
                write(p, z, zero3, shape, false, true);
                return read(p) == z;
            });
        }
        std::error_code ec;
        for (const auto& p : std::filesystem::directory_iterator(dir))
            if (p.path().filename().string().rfind("rt_shard_", 0) == 0) std::filesystem::remove_all(p.path(), ec);
    }

    // Sharding does not apply to a 0-dimensional array: the write must fail cleanly
    {
        total++;
        const std::string path = dir + "/rt_0d_shard.zarr";
        std::error_code ec; std::filesystem::remove_all(path, ec);
        bool ok = false;
        try {
            zarr Zw;
            Zw.set_fileName(path); Zw.set_shape({}); Zw.set_chunks({}); Zw.set_dtype("<u2");
            Zw.set_shard(true); Zw.set_chunk_shape({});
            Zw.write_zarray();
            Zw.set_chunkInfo({}, {});
            uint16_t v = 7;
            ok = parallelWriteZarr(Zw, &v, {}, {}, {}, 16, false, false, false) != 0 &&
                 Zw.get_errString().find("0-dimensional") != std::string::npos;
        } catch (...) { ok = false; }
        std::filesystem::remove_all(path, ec);
        std::printf("%s  0-D sharded write rejected\n", ok ? "PASS" : "FAIL");
        if (!ok) failures++;
    }

    // Zarr v3 metadata and conversion: the zarr.json of new arrays; v2 arrays cpp-zarr
    // wrote, converted to v3 (only the metadata changes, and the attributes move into
    // zarr.json) and written again; what cannot be converted or written as v3; a v2 write
    // over a v3 array is refused, and a v3 one over a v2 array replaces its metadata
    {
        typedef std::vector<uint64_t> V;
        auto check = [&](const char* name, std::function<bool()> test){
            total++;
            bool ok = false;
            try { ok = test(); }
            catch (const std::string& e) { std::fprintf(stderr, "    exception: %s\n", e.c_str()); }
            catch (const std::exception& e) { std::fprintf(stderr, "    exception: %s\n", e.what()); }
            std::printf("%s  %s\n", ok ? "PASS" : "FAIL", name);
            if (!ok) failures++;
        };
        auto readJson = [](const std::string& p){ std::ifstream f(p); return json::parse(f); };
        // every file of an array but its metadata, with its contents
        auto chunkFiles = [](const std::string& p){
            std::vector<std::pair<std::string, std::string>> files;
            for (const auto& f : std::filesystem::recursive_directory_iterator(p)){
                const std::string name = f.path().filename().string();
                if (!f.is_regular_file() || name == ".zarray" || name == "zarr.json" || name == ".zattrs") continue;
                std::ifstream in(f.path().string(), std::ios::binary);
                files.push_back({f.path().string(), std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>())});
            }
            std::sort(files.begin(), files.end());
            return files;
        };
        struct Made { std::string path; V shape; std::vector<uint8_t> data; uint64_t bytes; };
        // an array with random data (F order); sharded when inner is given. Over what is
        // at its path, unless fresh.
        auto make = [&](const std::string& name, int fmt, const V& shape, const V& chunks, const V& inner, const char* order,
                        const char* comp, const char* sep, const char* dtype, const std::string& fill = "0", bool fresh = true){
            Made m{dir + "/rt_v3_" + name + ".zarr", shape, {}, (uint64_t)(dtype[2] - '0')};
            std::error_code ec;
            if (fresh) std::filesystem::remove_all(m.path, ec);
            zarr Z;
            Z.set_fileName(m.path);
            if (fmt == 3) Z.set_zarr_format(3);
            Z.set_cname(comp); Z.set_order(order); Z.set_chunks(chunks); Z.set_dimension_separator(sep);
            Z.set_dtype(dtype); Z.set_shape(shape); Z.set_fill_value(fill);
            if (!inner.empty()){ Z.set_shard(true); Z.set_chunk_shape(inner); }
            Z.write_zarray();
            uint64_t count = 1;
            for (uint64_t v : shape) count *= v;
            m.data.resize(count * m.bytes);
            for (auto& b : m.data) b = (uint8_t)byteDist(rng);
            const V zeros(shape.size(), 0);
            Z.set_chunkInfo(zeros, shape);
            if (parallelWriteZarr(Z, m.data.data(), zeros, shape, shape, m.bytes * 8, true, false, false))
                throw std::string("write: ") + Z.get_errString();
            return m;
        };
        auto readsBack = [&](const Made& m){
            zarr Z(m.path);
            const V zeros(m.shape.size(), 0);
            void* d = parallelReadZarrWriteWrapper(Z, false, zeros, m.shape);
            const bool same = d && !std::memcmp(d, m.data.data(), m.data.size());
            free(d);
            return same;
        };

        check("Zarr v3 metadata of new arrays (C order, '/' keys by default; F order, shards, gzip, fill values)", [&](){
            zarr D;
            D.set_zarr_format(3);
            bool ok = D.get_order() == "C" && D.get_dimension_separator() == "/" && D.get_zarr_format() == 3;
            // (an array set to v3 but not written yet has v2-style metadata in memory)
            ok = ok && D.chunkKey("0.0.0") == "0.0.0" && !D.get_chunkChecksum() && D.get_chunkAxisOrder().size() == 3;
            try { D.set_zarr_format(4); ok = false; } catch (const std::string& e){ ok = ok && e.rfind("zarrFormatUnsupported:", 0) == 0; }

            Made a = make("meta_default", 3, {20, 10, 6}, {8, 8, 6}, {}, "C", "zstd", "/", "<u2");
            json m = readJson(a.path + "/zarr.json");
            ok = ok && m.at("zarr_format") == 3 && m.at("node_type") == "array" && m.at("shape") == json({20, 10, 6}) &&
                 m.at("data_type") == "uint16" && m.at("chunk_grid") == json({{"name", "regular"}, {"configuration", {{"chunk_shape", {8, 8, 6}}}}}) &&
                 m.at("chunk_key_encoding") == json({{"name", "default"}, {"configuration", {{"separator", "/"}}}}) &&
                 m.at("fill_value") == 0 && m.at("attributes") == json::object() &&
                 m.at("codecs") == json::parse(R"([{"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "blosc", "configuration": {"cname": "zstd", "clevel": 5, "shuffle": "shuffle", "typesize": 2, "blocksize": 0}}])") &&
                 fileExists(a.path + "/c/2/1/0") && !fileExists(a.path + "/.zarray") && readsBack(a);

            Made b = make("meta_sharded_f", 3, {20, 10, 6}, {16, 8, 6}, {8, 4, 3}, "F", "gzip", ".", "<f4", "NaN");
            m = readJson(b.path + "/zarr.json");
            ok = ok && m.at("data_type") == "float32" && m.at("fill_value") == "NaN" &&
                 m.at("chunk_key_encoding") == json({{"name", "default"}, {"configuration", {{"separator", "."}}}}) &&
                 m.at("codecs") == json::parse(R"([{"name": "sharding_indexed", "configuration": {"chunk_shape": [8, 4, 3],
                    "codecs": [{"name": "transpose", "configuration": {"order": [2, 1, 0]}}, {"name": "bytes", "configuration": {"endian": "little"}},
                               {"name": "gzip", "configuration": {"level": 5}}],
                    "index_codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "crc32c"}],
                    "index_location": "end"}}])") &&
                 fileExists(b.path + "/c.1.1.0") && readsBack(b);

            // 1-byte types have no byte order, uncompressed chunks no compressor; a 0-D
            // array's one chunk is "c"
            Made c = make("meta_int8", 3, {5, 4}, {4, 4}, {}, "C", "none", "/", "<i1", "-3");
            m = readJson(c.path + "/zarr.json");
            ok = ok && m.at("data_type") == "int8" && m.at("fill_value") == -3 && m.at("codecs") == json::parse(R"([{"name": "bytes"}])") && readsBack(c);
            Made d = make("meta_0d", 3, {}, {}, {}, "C", "lz4", "/", "<f8", "1.5");
            m = readJson(d.path + "/zarr.json");
            ok = ok && m.at("shape") == json::array() && m.at("fill_value") == 1.5 && fileExists(d.path + "/c") && readsBack(d);
            for (const Made* x : {&a, &b, &c, &d}){ std::error_code ec; std::filesystem::remove_all(x->path, ec); }
            return ok;
        });

        check("shard and inner chunk shapes set before the shape (as the MATLAB and Python writers do) keep every axis", [&](){
            bool ok = true;
            for (const int fmt : {2, 3}) for (const V& shape : {V{8, 12, 16, 20}, V{6, 8, 10, 12, 14}}){
                const uint64_t n = shape.size();
                V shards(n), inner(n);
                for (uint64_t d = 0; d < n; d++){ inner[d] = 2 + d; shards[d] = 2*inner[d]; }
                const std::string p = dir + "/rt_v3_axes.zarr";
                std::error_code ec; std::filesystem::remove_all(p, ec);
                zarr Z;
                Z.set_fileName(p);
                if (fmt == 3) Z.set_zarr_format(3);
                Z.set_chunks(shards); Z.set_shard(true); Z.set_chunk_shape(inner);
                Z.set_shape(shape); Z.set_dtype("<u2");
                Z.write_zarray();
                const json m = readJson(p + (fmt == 3 ? "/zarr.json" : "/.zarray"));
                const json gotShards = fmt == 3 ? m.at("chunk_grid").at("configuration").at("chunk_shape") : m.at("chunks");
                ok = ok && gotShards == json(shards) && m.at("codecs").at(0).at("configuration").at("chunk_shape") == json(inner);
                // and the data round trips
                uint64_t count = 1;
                for (uint64_t v : shape) count *= v;
                Made a{p, shape, std::vector<uint8_t>(count*2), 2};
                for (auto& b : a.data) b = (uint8_t)byteDist(rng);
                const V zeros(n, 0);
                Z.set_chunkInfo(zeros, shape);
                ok = ok && parallelWriteZarr(Z, a.data.data(), zeros, shape, shape, 16, true, false, false) == 0 && readsBack(a);
                std::filesystem::remove_all(p, ec);
            }
            return ok;
        });

        check("v2 arrays converted to Zarr v3 keep their chunk files, read the same and can be written", [&](){
            struct Conv { const char* name; V shape, chunks, inner; const char* order; const char* comp; const char* sep; const char* dtype; const char* fill; };
            const Conv convs[] = {
                {"f_dot_lz4", {30, 20, 7}, {16, 8, 4}, {}, "F", "lz4", ".", "<u2", "0"},
                {"c_slash_zstd", {30, 20, 7}, {16, 8, 4}, {}, "C", "zstd", "/", "<i4", "7"},
                {"f_gzip", {30, 20, 7}, {16, 8, 4}, {}, "F", "gzip", ".", "<f8", "0"},
                {"c_none", {30, 20, 7}, {16, 8, 4}, {}, "C", "none", ".", "<u1", "0"},
                {"sharded_f", {30, 20, 7}, {16, 8, 4}, {8, 4, 2}, "F", "zstd", ".", "<u2", "0"},
                {"sharded_c_slash", {30, 20, 7}, {32, 16, 8}, {16, 8, 4}, "C", "lz4", "/", "<f4", "0"},
                {"1d", {1000}, {128}, {}, "F", "zstd", ".", "<u8", "0"},
                {"4d", {5, 6, 7, 8}, {2, 4, 4, 3}, {}, "F", "zlib", ".", "<i2", "0"},
                {"0d", {}, {}, {}, "F", "lz4", ".", "<f4", "0"},
            };
            bool ok = true;
            for (const Conv& c : convs){
                Made m = make(std::string("conv_") + c.name, 2, c.shape, c.chunks, c.inner, c.order, c.comp, c.sep, c.dtype, c.fill);
                std::ofstream(m.path + "/.zattrs") << R"({"units": "um", "n": 3})";
                const auto before = chunkFiles(m.path);
                zarr Z(m.path);
                Z.convert_to_v3();
                const json meta = readJson(m.path + "/zarr.json");
                bool good = Z.get_zarr_format() == 3 && !fileExists(m.path + "/.zarray") && !fileExists(m.path + "/.zattrs") &&
                            meta.at("attributes") == json({{"units", "um"}, {"n", 3}}) && meta.at("chunk_key_encoding").at("name") == "v2" &&
                            chunkFiles(m.path) == before && zarr(m.path).get_zarr_format() == 3 && readsBack(m);
                // and written again (a region, in its v2-named chunk files)
                const uint64_t n = c.shape.size();
                if (good && n){
                    V s(n), e(n), ws(n);
                    uint64_t pc = 1;
                    for (uint64_t d = 0; d < n; d++){ s[d] = c.shape[d] / 3; e[d] = c.shape[d] - c.shape[d] / 4; ws[d] = e[d] - s[d]; pc *= ws[d]; }
                    std::vector<uint8_t> patch(pc * m.bytes);
                    for (auto& b : patch) b = (uint8_t)byteDist(rng);
                    for (uint64_t i = 0; i < pc; i++){
                        uint64_t r = i, off = 0, stride = 1;
                        for (uint64_t d = 0; d < n; d++){ off += (s[d] + r % ws[d]) * stride; r /= ws[d]; stride *= c.shape[d]; }
                        std::memcpy(&m.data[off * m.bytes], &patch[i * m.bytes], m.bytes);
                    }
                    zarr W(m.path);
                    W.set_chunkInfo(s, e);
                    good = parallelWriteZarr(W, patch.data(), s, e, ws, m.bytes * 8, true, true, false) == 0 && readsBack(m) &&
                           chunkFiles(m.path).size() == before.size();
                }
                if (!good) std::fprintf(stderr, "    conversion %s failed\n", c.name);
                ok = ok && good;
                std::error_code ec; std::filesystem::remove_all(m.path, ec);
            }
            return ok;
        });

        check("conversion keeps negative zstd levels and removes an empty .zattrs; rewritten metadata takes a new order", [&](){
            // a v2 array with numcodecs Zstd at level -3 (as zarr-python writes it)
            const std::string p = dir + "/rt_v3_zstd_negative.zarr";
            std::error_code ec; std::filesystem::remove_all(p, ec);
            std::filesystem::create_directories(p);
            std::ofstream(p + "/.zarray") << R"({"chunks": [8, 8, 8], "compressor": {"id": "zstd", "level": -3, "checksum": false},
                "dimension_separator": ".", "dtype": "<u2", "fill_value": 0, "filters": null, "order": "C", "shape": [20, 10, 6],
                "zarr_format": 2})";
            std::ofstream(p + "/.zattrs") << "{}";
            Made m{p, {20, 10, 6}, std::vector<uint8_t>(20*10*6*2), 2};
            for (auto& b : m.data) b = (uint8_t)byteDist(rng);
            zarr W(p);
            W.set_chunkInfo({0, 0, 0}, m.shape);
            bool ok = parallelWriteZarr(W, m.data.data(), {0, 0, 0}, m.shape, m.shape, 16, true, false, false) == 0;
            zarr(p).convert_to_v3();
            json meta = readJson(p + "/zarr.json");
            ok = ok && meta.at("codecs").back() == json({{"name", "zstd"}, {"configuration", {{"level", -3}, {"checksum", false}}}}) &&
                 !fileExists(p + "/.zattrs") && !fileExists(p + "/.zarray") && readsBack(m);
            std::filesystem::remove_all(p, ec);
            // an F-order v3 array whose metadata is rewritten in C order (as a full
            // rewrite from MATLAB with 'order', 'C' does): no transpose
            Made f = make("rewrite_order", 3, {20, 10, 6}, {8, 8, 6}, {}, "F", "lz4", "/", "<u2");
            ok = ok && readJson(f.path + "/zarr.json").dump().find("transpose") != std::string::npos;
            zarr R(f.path);
            R.set_order("C");
            R.write_zarray();
            ok = ok && readJson(f.path + "/zarr.json").dump().find("transpose") == std::string::npos && R.get_chunkAxisOrder() == V({0, 1, 2});
            std::filesystem::remove_all(f.path, ec);
            return ok;
        });

        check("what Zarr v3 cannot describe is refused, and changes nothing", [&](){
            bool ok = true;
            // conversion: subfolders, shard shapes that are not a multiple of the inner
            // chunk shape, and an array that is already v3
            for (int k = 0; k < 2; k++){
                const std::string p = dir + "/rt_v3_refused.zarr";
                std::error_code ec; std::filesystem::remove_all(p, ec);
                zarr Z;
                Z.set_fileName(p); Z.set_shape({30, 20, 7}); Z.set_chunks({16, 8, 4});
                if (k == 0) Z.set_subfolders({2, 1, 1});
                else { Z.set_shard(true); Z.set_chunk_shape({6, 8, 4}); }
                Z.write_zarray();
                const std::string meta = readJson(p + "/.zarray").dump();
                zarr R(p);
                try { R.convert_to_v3(); ok = false; }
                catch (const std::string& e){ ok = ok && e.rfind("v3Unsupported:", 0) == 0; }
                ok = ok && readJson(p + "/.zarray").dump() == meta && !fileExists(p + "/zarr.json");
                // nor written as v3 from the start
                std::filesystem::remove_all(p, ec);
                Z.set_zarr_format(3);
                try { Z.write_zarray(); ok = false; }
                catch (const std::string& e){ ok = ok && e.rfind("v3Unsupported:", 0) == 0; }
                ok = ok && !fileExists(p + "/zarr.json");
                std::filesystem::remove_all(p, ec);
            }
            Made v3 = make("refused_v3", 3, {10, 7, 5}, {4, 4, 5}, {}, "C", "lz4", "/", "<u2");
            zarr Z3(v3.path);
            try { Z3.convert_to_v3(); ok = false; }
            catch (const std::string& e){ ok = ok && e.rfind("v3Unsupported:", 0) == 0; }
            // an unknown compressor
            zarr B;
            B.set_fileName(dir + "/rt_v3_badcname.zarr"); B.set_zarr_format(3); B.set_cname("snappy"); B.set_shape({4, 4, 4});
            try { B.write_zarray(); ok = false; }
            catch (const std::string& e){ ok = ok && e == "unsupportedCompressor"; }
            // a v2 write over a v3 array is refused before anything changes
            const auto before = chunkFiles(v3.path);
            const std::string meta = readJson(v3.path + "/zarr.json").dump();
            zarr N;
            N.set_fileName(v3.path); N.set_shape({10, 7, 5});
            try { N.write_zarray(); ok = false; }
            catch (const std::string& e){ ok = ok && e.rfind("zarrV3NotWritable:", 0) == 0; }
            ok = ok && chunkFiles(v3.path) == before && readJson(v3.path + "/zarr.json").dump() == meta && !fileExists(v3.path + "/.zarray") &&
                 readsBack(v3);
            std::error_code ec; std::filesystem::remove_all(v3.path, ec);
            return ok;
        });

        check("a Zarr v3 array written over a v2 one replaces its .zarray, keeping its attributes", [&](){
            Made v2 = make("over_v2", 2, {10, 7, 5}, {4, 4, 5}, {}, "F", "lz4", ".", "<u2");
            std::ofstream(v2.path + "/.zattrs") << R"({"a": [1, 2]})";
            Made v3 = make("over_v2", 3, {10, 7, 5}, {4, 4, 5}, {}, "C", "zstd", "/", "<u2", "0", false);
            (void)v2;
            const json m = readJson(v3.path + "/zarr.json");
            bool ok = !fileExists(v3.path + "/.zarray") && !fileExists(v3.path + "/.zattrs") && m.at("attributes") == json({{"a", {1, 2}}}) &&
                      readsBack(v3);
            // and rewritten as v3 again, it keeps them
            Made again = make("over_v2", 3, {10, 7, 5}, {4, 4, 5}, {}, "F", "gzip", ".", "<u2", "0", false);
            ok = ok && readJson(again.path + "/zarr.json").at("attributes") == json({{"a", {1, 2}}}) && readsBack(again);
            std::error_code ec; std::filesystem::remove_all(v3.path, ec);
            return ok;
        });
    }

    // Test arrays written by zarr-python 3 and TensorStore (tests/test_arrays, made by
    // tests/make_test_arrays.py): Zarr v3 arrays, and v2 arrays with codecs and fill
    // values cpp-zarr does not write itself. Full reads in C and F order and a region read
    // must match the values zarr-python reads, and the arrays cpp-zarr cannot read must
    // be rejected with an error that names the reason. Then writes into copies of them.
    {
#ifdef CPPZARR_TEST_ARRAYS
        const std::string v3dir = CPPZARR_TEST_ARRAYS;
#else
        const std::string v3dir = "tests/test_arrays";
#endif
        std::ifstream mf(v3dir + "/arrays.json");
        if (!mf){
            // made by tests/make_test_arrays.py; CI makes them and requires them
            const bool required = std::getenv("CPPZARR_REQUIRE_TEST_ARRAYS") != nullptr;
            if (required){ total++; failures++; }
            std::printf("%s  test arrays not found in %s (make them with tests/make_test_arrays.py)\n",
                        required ? "FAIL" : "SKIP", v3dir.c_str());
        }
        else{
            const json manifest = json::parse(mf);
            for (const json& fx : manifest.at("arrays")){
                total++;
                const std::string name = fx.at("name"), path = v3dir + "/" + name + ".zarr";
                bool ok = false;
                std::string detail;
                if (fx.contains("error")){
                    const std::string want = fx.at("error");
                    try { zarr Z(path); detail = "was not rejected"; }
                    catch (const std::string& e){ ok = e.find(want) != std::string::npos; detail = e; }
                    catch (...){ detail = "threw something other than an error message"; }
                    std::printf("%s  test array %s rejected%s%s\n", ok ? "PASS" : "FAIL", name.c_str(),
                                ok ? "" : ": ", ok ? "" : detail.c_str());
                    if (!ok) failures++;
                    continue;
                }
                try{
                    const std::string dt = fx.at("dtype");
                    const std::vector<uint64_t> shape = fx.at("shape").get<std::vector<uint64_t>>();
                    const uint64_t nd = shape.size(), bytes = dt[2] - '0';
                    uint64_t count = 1;
                    for (uint64_t v : shape) count *= v;
                    std::ifstream bf(v3dir + "/" + name + ".bin", std::ios::binary);
                    const std::vector<uint8_t> want((std::istreambuf_iterator<char>(bf)), std::istreambuf_iterator<char>());
                    // equal elements, NaN equal to NaN
                    auto same = [&](const uint8_t* a, const uint8_t* b){
                        if (!std::memcmp(a, b, bytes)) return true;
                        if (dt[1] != 'f') return false;
                        if (bytes == 4){ float x, y; std::memcpy(&x, a, 4); std::memcpy(&y, b, 4); return std::isnan(x) && std::isnan(y); }
                        double x, y; std::memcpy(&x, a, 8); std::memcpy(&y, b, 8); return std::isnan(x) && std::isnan(y);
                    };
                    // the expected element at C-order coordinates idx
                    auto at = [&](const std::vector<uint64_t>& idx){
                        uint64_t off = 0;
                        for (uint64_t d = 0; d < nd; d++) off = off * shape[d] + idx[d];
                        return &want[off * bytes];
                    };
                    // compare a read of [s, e) in C or F order
                    auto check = [&](const uint8_t* got, const std::vector<uint64_t>& s, const std::vector<uint64_t>& e, bool cOrder){
                        std::vector<uint64_t> ext(nd), idx(nd);
                        uint64_t n = 1;
                        for (uint64_t d = 0; d < nd; d++){ ext[d] = e[d] - s[d]; n *= ext[d]; }
                        for (uint64_t i = 0; i < n; i++){
                            uint64_t r = i;
                            if (cOrder) for (int64_t d = (int64_t)nd - 1; d >= 0; d--){ idx[d] = s[d] + r % ext[d]; r /= ext[d]; }
                            else for (uint64_t d = 0; d < nd; d++){ idx[d] = s[d] + r % ext[d]; r /= ext[d]; }
                            if (!same(got + i * bytes, at(idx))) return false;
                        }
                        return true;
                    };
                    zarr Z(path);
                    const std::vector<uint64_t> zeros(nd, 0);
                    const uint64_t format = name.rfind("v3_", 0) == 0 ? 3 : 2;
                    ok = want.size() == count * bytes && Z.get_ndims() == nd && Z.get_zarr_format() == format;
                    for (bool cOrder : {true, false}){
                        void* d = parallelReadZarrWriteWrapper(Z, false, zeros, shape, cOrder);
                        ok = ok && d && check((const uint8_t*)d, zeros, shape, cOrder);
                        if (!d) detail = "read failed";
                        free(d);
                    }
                    // a region inside the array
                    std::vector<uint64_t> s(nd), e(nd);
                    for (uint64_t d = 0; d < nd; d++){ s[d] = shape[d] > 2 ? 1 : 0; e[d] = shape[d] > 2 ? shape[d] - 1 : shape[d]; }
                    void* r = parallelReadZarrWriteWrapper(Z, true, s, e, true);
                    ok = ok && r && check((const uint8_t*)r, s, e, true);
                    free(r);
                }
                catch (const std::string& e){ ok = false; detail = e; }
                catch (const std::exception& e){ ok = false; detail = e.what(); }
                std::printf("%s  test array %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(), ok ? "" : ": ", detail.c_str());
                if (!ok) failures++;
            }

            // Writes into copies of the test arrays, which cpp-zarr did not make: a region
            // (partly covering chunks, and shards in the sharded arrays), then the whole
            // array, read back by cpp-zarr; every codec, chunk key encoding, axis order and
            // shard layout above. Arrays in the opposite byte order are refused before
            // anything is written. Then the v2 arrays are converted to Zarr v3 (numcodecs
            // zlib cannot be) and read and written again, and rewriting a v3 array's
            // metadata keeps its chunk key encoding, checksums, axis order and attributes.
            // (Arrays are copied file by file with streams: on macOS, std::filesystem::copy
            // aborts in this program, which has Homebrew's libstdc++ next to the copy
            // libcppZarr exports.)
            auto copyArray = [](const std::string& src, const std::string& to){
                std::error_code ec;
                std::filesystem::remove_all(to, ec);
                std::filesystem::create_directories(to);
                for (const auto& f : std::filesystem::recursive_directory_iterator(src)){
                    const std::string t = to + f.path().string().substr(src.size());
                    if (f.is_directory()){ std::filesystem::create_directories(t); continue; }
                    std::ifstream in(f.path().string(), std::ios::binary);
                    std::ofstream out(t, std::ios::binary);
                    out << in.rdbuf();
                }
            };
            // every file of an array, with its contents
            auto snapshot = [](const std::string& p){
                std::vector<std::pair<std::string, std::string>> files;
                for (const auto& f : std::filesystem::recursive_directory_iterator(p)){
                    if (!f.is_regular_file()) continue;
                    std::ifstream in(f.path().string(), std::ios::binary);
                    files.push_back({f.path().string(), std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>())});
                }
                std::sort(files.begin(), files.end());
                return files;
            };
            auto readText = [](const std::string& p){
                std::ifstream in(p, std::ios::binary);
                return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            };
            std::mt19937 wrng(20261002u);
            for (const json& fx : manifest.at("arrays")){
                if (fx.contains("error")) continue;
                total++;
                const std::string name = fx.at("name"), path = dir + "/rt_ta_" + name + ".zarr";
                const std::string dt = fx.at("dtype");
                const std::vector<uint64_t> shape = fx.at("shape").get<std::vector<uint64_t>>();
                const uint64_t nd = shape.size(), bytes = dt[2] - '0';
                uint64_t count = 1;
                for (uint64_t v : shape) count *= v;
                std::ifstream bf(v3dir + "/" + name + ".bin", std::ios::binary);
                std::vector<uint8_t> exp((std::istreambuf_iterator<char>(bf)), std::istreambuf_iterator<char>());
                bool ok = exp.size() == count * bytes;
                std::string detail;
                // new values, F order (booleans 0 or 1)
                auto values = [&](uint64_t n){
                    std::vector<uint8_t> v(n * bytes);
                    for (auto& b : v) b = dt[1] == 'b' ? (uint8_t)(wrng() & 1) : (uint8_t)wrng();
                    return v;
                };
                // put an F-order region [s, e) into exp (C order)
                auto paste = [&](const std::vector<uint8_t>& f, const std::vector<uint64_t>& s, const std::vector<uint64_t>& e){
                    uint64_t n = 1;
                    for (uint64_t d = 0; d < nd; d++) n *= e[d] - s[d];
                    for (uint64_t i = 0; i < n; i++){
                        uint64_t r = i, off = 0;
                        std::vector<uint64_t> idx(nd);
                        for (uint64_t d = 0; d < nd; d++){ idx[d] = s[d] + r % (e[d] - s[d]); r /= e[d] - s[d]; }
                        for (uint64_t d = 0; d < nd; d++) off = off * shape[d] + idx[d];
                        std::memcpy(&exp[off * bytes], &f[i * bytes], bytes);
                    }
                };
                // the whole array read by cpp-zarr matches exp (NaN equal to NaN)
                auto matches = [&](const std::string& p){
                    zarr Z(p);
                    const std::vector<uint64_t> zeros(nd, 0);
                    uint8_t* got = (uint8_t*)parallelReadZarrWriteWrapper(Z, false, zeros, shape, true);
                    bool same = got != nullptr;
                    for (uint64_t i = 0; same && i < count; i++){
                        const uint8_t *a = got + i * bytes, *b = &exp[i * bytes];
                        same = !std::memcmp(a, b, bytes);
                        if (!same && dt[1] == 'f'){
                            if (bytes == 4){ float x, y; std::memcpy(&x, a, 4); std::memcpy(&y, b, 4); same = std::isnan(x) && std::isnan(y); }
                            else { double x, y; std::memcpy(&x, a, 8); std::memcpy(&y, b, 8); same = std::isnan(x) && std::isnan(y); }
                        }
                    }
                    free(got);
                    return same;
                };
                // write F-order data into the region [s, e) (the whole array when not crop)
                auto writeRegion = [&](const std::string& p, const std::vector<uint8_t>& f, const std::vector<uint64_t>& s,
                                       const std::vector<uint64_t>& e, bool crop, std::string* err){
                    zarr Z(p);
                    std::vector<uint64_t> ws(nd);
                    for (uint64_t d = 0; d < nd; d++) ws[d] = e[d] - s[d];
                    Z.set_chunkInfo(s, e);
                    const bool failed = parallelWriteZarr(Z, (void*)f.data(), s, e, ws, bytes * 8, true, crop, true) != 0;
                    if (failed && err) *err = Z.get_errString();
                    return !failed;
                };
                try{
                    copyArray(v3dir + "/" + name + ".zarr", path);
                    const std::string metaFile = path + (name.rfind("v3_", 0) == 0 ? "/zarr.json" : "/.zarray");
                    const std::string meta = readText(metaFile);
                    std::vector<uint64_t> s(nd), e(nd), zeros(nd, 0);
                    uint64_t n = 1;
                    for (uint64_t d = 0; d < nd; d++){
                        s[d] = shape[d] > 2 ? 1 : 0; e[d] = shape[d] > 2 ? shape[d] - 1 : shape[d];
                        n *= e[d] - s[d];
                    }
                    const std::vector<uint8_t> patch = values(n);
                    if (oppositeEndianness(zarr(path).get_dtype())){
                        // refused, and nothing changes
                        const auto before = snapshot(path);
                        std::string err;
                        ok = ok && !writeRegion(path, patch, s, e, nd > 0, &err) && err.find("byte order") != std::string::npos &&
                             snapshot(path) == before;
                        if (!ok) detail = "write into an array in the opposite byte order: " + err;
                    }
                    else{
                        std::string err;
                        ok = ok && writeRegion(path, patch, s, e, nd > 0, &err);
                        paste(patch, s, e);
                        ok = ok && matches(path) && readText(metaFile) == meta;
                        if (!ok && detail.empty()) detail = "region write: " + err;
                        const std::vector<uint8_t> all = values(count);
                        ok = ok && writeRegion(path, all, zeros, shape, false, &err);
                        paste(all, zeros, shape);
                        ok = ok && matches(path) && readText(metaFile) == meta;
                        if (!ok && detail.empty()) detail = "whole array write: " + err;
                        // a v2 array converted to Zarr v3 reads the same and can be written
                        if (ok && name.rfind("v2_", 0) == 0){
                            zarr Z(path);
                            bool converted = false;
                            try { Z.convert_to_v3(); converted = true; }
                            catch (const std::string& x){
                                // (numcodecs zlib streams have no Zarr v3 codec)
                                ok = name.find("zlib") != std::string::npos && x.rfind("v3Unsupported:", 0) == 0 &&
                                     readText(metaFile) == meta && !fileExists(path + "/zarr.json");
                                if (!ok) detail = "conversion: " + x;
                            }
                            if (converted){
                                ok = name.find("zlib") == std::string::npos && !fileExists(metaFile) &&
                                     zarr(path).get_zarr_format() == 3 && matches(path);
                                const std::vector<uint8_t> patch2 = values(n);
                                ok = ok && writeRegion(path, patch2, s, e, nd > 0, &err);
                                paste(patch2, s, e);
                                ok = ok && matches(path);
                                if (!ok && detail.empty()) detail = "converted array: " + err;
                            }
                        }
                        // rewriting a v3 array's metadata (with attributes and dimension names
                        // added) keeps what describes its chunks, and its attributes
                        if (ok && name.rfind("v3_", 0) == 0){
                            json m = json::parse(meta);
                            m["attributes"] = {{"note", "kept"}};
                            if (nd) m["dimension_names"] = std::vector<std::string>(nd, "axis");
                            std::ofstream(path + "/zarr.json") << m.dump();
                            zarr(path).write_zarray();
                            const json r = json::parse(readText(path + "/zarr.json"));
                            ok = r.at("attributes") == m.at("attributes") && r.value("dimension_names", json()) == m.value("dimension_names", json()) &&
                                 r.at("chunk_key_encoding").at("name") == m.at("chunk_key_encoding").at("name") &&
                                 r.at("shape") == m.at("shape") && r.at("data_type") == m.at("data_type") && matches(path);
                            if (!ok) detail = "rewritten metadata: " + r.dump();
                        }
                    }
                }
                catch (const std::string& x){ ok = false; detail = x; }
                catch (const std::exception& x){ ok = false; detail = x.what(); }
                std::printf("%s  writes into test array %s%s%s\n", ok ? "PASS" : "FAIL", name.c_str(), ok ? "" : ": ", detail.c_str());
                if (!ok) failures++;
                std::error_code ec;
                std::filesystem::remove_all(path, ec);
            }
        }
    }

    std::printf("\n%d/%d round trips passed\n", total - failures, total);
    return failures ? 1 : 0;
}
