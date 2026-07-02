// Minimal smoke test for the cpp-zarr library: write small random volumes, read
// them back, and check the bytes survive the round trip. Covers every supported
// dtype (u1/u2/f4/f8), each compressor, and both storage orders (F and C), using
// a shape that does not divide the chunk size so partial edge chunks are exercised.
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

    // Non-chunk-aligned shape so read/write hit partial edge chunks.
    const std::vector<uint64_t> shape  = {40, 24, 18};
    const std::vector<uint64_t> chunks = {16, 16, 16};
    const uint64_t n = shape[0] * shape[1] * shape[2];

    const Case cases[]  = {
        {"uint8", "<u1", 8}, {"uint16", "<u2", 16}, {"float", "<f4", 32}, {"double", "<f8", 64},
    };
    const char* comps[]  = {"lz4", "blosclz", "lz4hc", "zlib", "zstd", "gzip"};
    const char* orders[] = {"F", "C"};

    std::mt19937 rng(1234567u);
    std::uniform_int_distribution<int> byteDist(0, 255);

    int total = 0, failures = 0;
    for (const Case& c : cases){
        const uint64_t nbytes = n * (c.bits / 8);
        std::vector<uint8_t> orig(nbytes);
        for (uint64_t i = 0; i < nbytes; i++) orig[i] = static_cast<uint8_t>(byteDist(rng));

        for (const char* comp : comps){
            for (const char* order : orders){
                total++;
                const std::string path = dir + "/rt_" + c.name + "_" + comp + "_" + order + ".zarr";
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
                    std::printf("PASS  %-6s %-8s %s\n", c.name, comp, order);
                } else {
                    std::printf("FAIL  %-6s %-8s %s  (meta=%d data=%d)\n",
                                c.name, comp, order, (int)metaOK, (int)dataOK);
                    failures++;
                }
            }
        }
    }

    std::printf("\n%d/%d round trips passed\n", total - failures, total);
    return failures ? 1 : 0;
}
