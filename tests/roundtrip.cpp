// Minimal smoke test for the cpp-zarr library: write small random volumes, read
// them back, and check the bytes survive the round trip. Covers every supported
// dtype (signed/unsigned 8-64 bit ints, f4/f8), each compressor, and both storage
// orders (F and C), using shapes that do not divide the chunk size so partial edge
// chunks are exercised, plus chunks large enough for the full-tile F<->C paths.
//
// Exits 0 if every case passes, 1 otherwise, so CTest reports pass/fail.
// Usage: roundtripTest [output_dir]   (defaults to the current directory)
//
// Progress is written (flushed) to stderr before each operation so that if a
// build segfaults, the CTest log shows exactly which step/dtype/compressor died.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "zarr.h"
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
        {"", {40, 24, 18}, {16, 16, 16}, {"lz4", "blosclz", "lz4hc", "zlib", "zstd", "gzip"}},
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
                total++;
                const std::string path = dir + "/rt_" + c.name + "_" + comp + "_" + order + cfg.tag + ".zarr";
                std::error_code ec; std::filesystem::remove_all(path, ec);

                bool metaOK = false, dataOK = false;
                try {
                    step(c.name, comp, order, "write");
                    zarr Zw;
                    Zw.set_fileName(path);
                    Zw.set_cname(comp);
                    Zw.set_clevel(5);
                    Zw.set_order(order);
                    Zw.set_chunks(chunks);
                    Zw.set_dimension_separator(".");
                    Zw.set_dtype(c.dtype);
                    Zw.set_shape(shape);
                    Zw.write_zarray();
                    Zw.set_chunkInfo({0,0,0}, shape);
                    uint8_t werr = parallelWriteZarr(Zw, orig.data(), {0,0,0}, shape, shape,
                                                     c.bits, /*useUuid*/false, /*crop*/false, /*sparse*/false);
                    if (werr) throw std::string("write error: ") + Zw.get_errString();

                    // Metadata (.zarray) must round trip.
                    step(c.name, comp, order, "read metadata");
                    zarr Zr(path);
                    metaOK = Zr.get_dtype() == c.dtype && Zr.get_cname() == comp &&
                             Zr.get_shape(0) == shape[0] && Zr.get_shape(1) == shape[1] &&
                             Zr.get_shape(2) == shape[2];

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
                    std::printf("PASS  %-6s %-8s %s %s\n", c.name, comp, order, cfg.tag);
                } else {
                    std::printf("FAIL  %-6s %-8s %s %s  (meta=%d data=%d)\n",
                                c.name, comp, order, cfg.tag, (int)metaOK, (int)dataOK);
                    failures++;
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
            return parallelWriteZarr(Z, (void*)data.data(), s, e, {e[0]-s[0], e[1]-s[1], e[2]-s[2]},
                                     16, false, /*crop*/!create, false) == 0;
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
                        ok = got[i] == (missing ? 0x0707 : A[i]);   // fill is applied byte-wise
                    }
                free(got);
            } catch (...) { ok = false; }
            std::filesystem::remove_all(path, ec);
            std::printf("%s  missing chunk reads as fill_value %s\n", ok ? "PASS" : "FAIL", order);
            if (!ok) failures++;
        }
    }

    std::printf("\n%d/%d round trips passed\n", total - failures, total);
    return failures ? 1 : 0;
}
