#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <exception>
#include <thread>
#include <tuple>
#include <unordered_map>
#include <vector>
#ifdef __linux__
#include <cstdio>
#include <ctime>
#include <fcntl.h>
#include <unistd.h>
#endif
#include "zarr.h"
#include "parallelreadzarr.h"
#include "parallelwritezarr.h"
#include "helperfunctions.h"
#include "gpureadzarr.h"
#include "gpuwritezarr.h"

// The NumPy dtype for an array's data type: bfloat16 is ml_dtypes' type, the others
// NumPy's own
static pybind11::dtype numpyDtype(const DtypeInfo &t, const std::string &dtype) {
    if (t.bfloat16) {
        try {
            return pybind11::dtype::from_args(pybind11::module_::import("ml_dtypes").attr("bfloat16"));
        }
        catch (pybind11::error_already_set &) {
            throw std::runtime_error("Reading bfloat16 arrays needs the ml_dtypes package (pip install ml_dtypes)");
        }
    }
    const std::string bits = std::to_string(8*t.bytes);
    switch (t.kind) {
        case 'b': return pybind11::dtype("bool");
        case 'i': return pybind11::dtype("int" + bits);
        case 'u': return pybind11::dtype("uint" + bits);
        case 'f': return pybind11::dtype("float" + bits);
        case 'c': return pybind11::dtype("complex" + bits);
    }
    throw std::runtime_error("Unsupported data type: " + dtype);
}

// A NumPy array over data (malloc'd: freed with the array), in F order (the first
// axis contiguous) or C order (the last)
static pybind11::array createArray(void* data, const std::vector<uint64_t> &dims, const bool cOrder,
                                   const pybind11::dtype &dt) {
    const size_t n = dims.size();
    std::vector<ssize_t> shape(n), strides(n);
    ssize_t stride = dt.itemsize();
    for (size_t i = 0; i < n; i++) {
        const size_t d = cOrder ? n-1-i : i;
        shape[d] = static_cast<ssize_t>(dims[d]);
        strides[d] = stride;
        stride *= static_cast<ssize_t>(dims[d]);
    }
    return pybind11::array(dt, shape, strides, data, pybind11::capsule(data, [](void* ptr) { free(ptr); }));
}

// A call's threads (numThreads; 0 for the thread's default: OMP_NUM_THREADS, or every
// CPU it may run on), set for the calling thread for the whole call and then restored:
// every OpenMP region of the call (finding the region's chunks too) then has a team of
// that size. (A region with the default team in between had OpenMP end and start
// threads in every call: pooled reads of small regions went 4 times as slow.)
struct CallThreads {
    int prev = 0;
    explicit CallThreads(const int64_t numThreads) {
        if (numThreads > 0) {
            prev = callThreads();
            setCallThreads((int)numThreads);
        }
    }
    ~CallThreads() {
        if (prev > 0) setCallThreads(prev);
    }
};

// The zarr class reports metadata problems by throwing std::string; turn them
// into Python exceptions with a readable message
static zarr openZarr(const std::string &fileName) {
    try {
        return zarr(fileName);
    }
    catch (const std::string &e) {
        if (e.rfind("metadataFileMissing:", 0) == 0) {
            throw std::runtime_error(fileName + " does not exist. No .zarray or zarr.json metadata file was found");
        }
        if (e.rfind("metadataUnsupported:", 0) == 0) {
            throw std::runtime_error("Cannot read " + fileName + ": " + e.substr(e.find(':') + 1));
        }
        throw std::runtime_error("Cannot read the zarr metadata of " + fileName + " (" + e + ")");
    }
}

// Errors the zarr class throws (as strings) when writing metadata or converting
static std::runtime_error metadataError(const std::string &fileName, const std::string &e) {
    if (e.rfind("zarrV3NotWritable:", 0) == 0) {
        return std::runtime_error(fileName + " is a Zarr v3 array. Write it with zarr_format=3, or delete it first");
    }
    if (e.rfind("v3Unsupported:", 0) == 0) {
        return std::runtime_error("Cannot write " + fileName + " as a Zarr v3 array: " + e.substr(e.find(':') + 1));
    }
    if (e == "unsupportedCompressor") {
        return std::runtime_error("Cannot write " + fileName + ": unsupported compressor");
    }
    return std::runtime_error("Cannot write the zarr metadata of " + fileName + " (" + e + ")");
}

static void writeZarray(zarr &Zarr) {
    try {
        Zarr.write_zarray();
    }
    catch (const std::string &e) {
        throw metadataError(Zarr.get_fileName(), e);
    }
}

// Fit user coordinates to an array with nDims axes. Extra trailing values are
// accepted when they describe singleton axes (start 0, end 0 or 1), so 3-value
// coordinates keep working on 1D/2D arrays.
static std::vector<uint64_t> fitCoords(const std::vector<uint64_t> &coords, const uint64_t nDims,
                                       const bool isEnd, const char* name) {
    if (coords.size() == nDims) return coords;
    if (coords.size() > nDims) {
        for (size_t d = nDims; d < coords.size(); d++) {
            if (coords[d] > (isEnd ? 1u : 0u)) {
                throw std::runtime_error(std::string(name) + " has " + std::to_string(coords.size()) +
                                         " values but the array has " + std::to_string(nDims) + " dimensions");
            }
        }
        return std::vector<uint64_t>(coords.begin(), coords.begin() + nDims);
    }
    throw std::runtime_error(std::string(name) + " has " + std::to_string(coords.size()) +
                             " values but the array has " + std::to_string(nDims) + " dimensions");
}

// The calling thread's clocks (seconds): wall, CPU, and time waiting for a CPU (Linux:
// /proc/thread-self/schedstat; ok false without it)
struct ThreadClocks {
    double wall = 0, cpu = 0, queued = 0;
    bool ok = false;
};
static ThreadClocks threadClocks() {
    ThreadClocks t;
#ifdef __linux__
    timespec w, c;
    if (clock_gettime(CLOCK_MONOTONIC, &w) != 0 || clock_gettime(CLOCK_THREAD_CPUTIME_ID, &c) != 0) return t;
    t.wall = (double)w.tv_sec + w.tv_nsec * 1e-9;
    t.cpu = (double)c.tv_sec + c.tv_nsec * 1e-9;
    const int fd = open("/proc/thread-self/schedstat", O_RDONLY | O_CLOEXEC);
    if (fd < 0) return t;
    char buf[128];
    const ssize_t n = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    unsigned long long run = 0, waited = 0;
    if (n > 0) {
        buf[n] = 0;
        if (sscanf(buf, "%llu %llu", &run, &waited) == 2) {
            t.queued = waited * 1e-9;
            t.ok = true;
        }
    }
#endif
    return t;
}

// The calling thread's last measured read (pybind11_read_zarr_opened with measure): its CPU time and
// its time waiting for the storage (neither on a CPU nor waiting for one), or -1 for that unknown.
// (Only the read: not the wait for the GIL, nor for the array's metadata.)
static thread_local double lastReadCpu = 0, lastReadWait = -1;

// A region of an opened array (whose chunk information the read sets) read into a new NumPy array,
// or into out (see pybind11_read_zarr); measure: time the read (see lastReadWait)
static pybind11::array readOpened(zarr &Zarr, const std::vector<uint64_t> &startCoords, const std::vector<uint64_t> &endCoords,
                                  const std::string &order, const pybind11::object &out, const bool measure = false) {
    ThreadClocks t0;
    auto measured = [&](){
        if (!measure) return;
        const ThreadClocks t1 = threadClocks();
        lastReadCpu = t1.cpu - t0.cpu;
        lastReadWait = t0.ok && t1.ok ? std::max(0.0, (t1.wall - t0.wall) - (t1.cpu - t0.cpu) - (t1.queued - t0.queued)) : -1;
    };
    bool cOrder = order.empty() ? Zarr.get_order() == "C" : order == "C";
    // (before reading: bfloat16 needs ml_dtypes)
    const pybind11::dtype dt = numpyDtype(dtypeInfo(Zarr.get_dtype()), Zarr.get_dtype());
    const uint64_t nDims = Zarr.get_ndims();
    std::vector<uint64_t> dims(nDims);
    for (uint64_t d = 0; d < nDims; d++) dims[d] = Zarr.get_shape(d);

    // All-zero (or empty) coordinates mean the whole array
    const bool startGiven = !std::all_of(startCoords.begin(), startCoords.end(), [](uint64_t i) { return i==0; });
    const bool endGiven = !std::all_of(endCoords.begin(), endCoords.end(), [](uint64_t i) { return i==0; });
    const bool crop = startGiven || endGiven;
    std::vector<uint64_t> start(nDims, 0), end = dims;
    if (startGiven) start = fitCoords(startCoords, nDims, false, "start_coords");
    // If we are only using user defined startCoords then the end is the axis size
    if (endGiven) end = fitCoords(endCoords, nDims, true, "end_coords");

    if(crop){
        for (uint64_t d = 0; d < nDims; d++) {
            if (end[d] > Zarr.get_shape(d) || start[d] >= end[d]) {
                throw std::runtime_error("Invalid start_coords or end_coords for axis " + std::to_string(d));
            }
            dims[d] = end[d]-start[d];
        }
        pybind11::gil_scoped_release release;
        Zarr.set_chunkInfo(start, end);
    }

    // Into the caller's array (out): of the region's shape and the array's data type,
    // writeable, and contiguous in order when it is given; with no order also any other
    // layout of positive strides (a view: a channel's planes, a region of a bigger array)
    if (!out.is_none()) {
        if (!pybind11::isinstance<pybind11::array>(out)) throw std::runtime_error("out must be a NumPy array");
        pybind11::array a = out.cast<pybind11::array>();
        std::string shapeText;
        for (uint64_t d = 0; d < nDims; d++) shapeText += (d ? ", " : "") + std::to_string(dims[d]);
        if (!a.dtype().equal(dt)) {
            throw std::runtime_error("out has data type " + pybind11::str(a.dtype()).cast<std::string>() +
                                     ", not the array's " + pybind11::str(dt).cast<std::string>());
        }
        bool sameShape = (uint64_t)a.ndim() == nDims;
        for (uint64_t d = 0; sameShape && d < nDims; d++) sameShape = (uint64_t)a.shape(d) == dims[d];
        if (!sameShape) throw std::runtime_error("out must have the shape of the region read, (" + shapeText + ")");
        const pybind11::object flags = a.attr("flags");
        const bool cContig = flags.attr("c_contiguous").cast<bool>(), fContig = flags.attr("f_contiguous").cast<bool>();
        if (!flags.attr("writeable").cast<bool>()) throw std::runtime_error("out must be writeable");
        if (!order.empty() && !(cOrder ? cContig : fContig)) {
            throw std::runtime_error("out must be " + order + "-contiguous for order '" + order + "'");
        }
        std::vector<uint64_t> outStrides;
        if (order.empty() && !cContig && !fContig) {
            const uint64_t itemsize = (uint64_t)a.itemsize();
            outStrides.assign(nDims, 0);
            for (uint64_t d = 0; d < nDims; d++) {
                if (dims[d] <= 1) continue;
                const ssize_t st = a.strides(d);
                if (st <= 0 || (uint64_t)st % itemsize) {
                    throw std::runtime_error("out's strides must be positive multiples of its element size");
                }
                outStrides[d] = (uint64_t)st / itemsize;
            }
        }
        // (its own layout, unless order says; one that is both, the file's order)
        if (order.empty() && cContig != fContig) cOrder = cContig;
        uint8_t fillElem[16];
        fillValueElement(Zarr.get_fill_value(), Zarr.get_dtype(), fillElem);
        void* ptr = a.mutable_data();
        uint8_t err = 0;
        {
            pybind11::gil_scoped_release release;
            if (measure) t0 = threadClocks();
            if (!crop) Zarr.set_chunkInfo(start, end);
            err = outStrides.empty() ? parallelReadZarr(Zarr, ptr, start, end, dims, Zarr.dtypeBytes()*8, true, false, cOrder, fillElem)
                                     : parallelReadZarr(Zarr, ptr, start, end, dims, Zarr.dtypeBytes()*8, true, false, outStrides, fillElem);
            measured();
        }
        if (err) throw std::runtime_error(Zarr.get_errString());
        return a;
    }

    void* data = nullptr;
    {
        pybind11::gil_scoped_release release;
        if (measure) t0 = threadClocks();
        data = parallelReadZarrWriteWrapper(Zarr, crop, start, end, cOrder);
        measured();
    }
    if (!data) throw std::runtime_error("Failed to read the zarr file (unsupported dtype or read error)");
    return createArray(data, dims, cOrder, dt);
}

pybind11::array pybind11_read_zarr(const std::string &fileName, const std::vector<uint64_t> &startCoords = std::vector<uint64_t>{0, 0, 0},
                                   const std::vector<uint64_t> &endCoords = std::vector<uint64_t>{0, 0, 0}, const std::string &order = "",
                                   const int64_t numThreads = 0, const pybind11::object &out = pybind11::none()){
    // order is the memory layout of the returned array: "F" (first axis
    // contiguous), "C" (last axis contiguous), or "" for the file's storage order
    // (or out's). The GIL is released while the array is opened and read, so other
    // Python threads run, and read other arrays, meanwhile.
    if (!order.empty() && order != "F" && order != "C") throw std::runtime_error("order must be 'F' or 'C', not '" + order + "'");
    if (numThreads < 0) throw std::runtime_error("num_threads must be at least 1");
    const CallThreads threads(numThreads);
    zarr Zarr;
    {
        pybind11::gil_scoped_release release;
        Zarr = openZarr(fileName);
    }
    return readOpened(Zarr, startCoords, endCoords, order, out);
}

// An array's metadata, read once for all of a batch's reads of it (read_zarr_batch): on network
// storage each read of an array's metadata waits on the server (about 1 ms from VAST), and a
// batch often reads many regions of one array
struct OpenedZarr {
    zarr z;
};

// As pybind11_read_zarr, from an array opened before (pybind11_open_zarrs): each read has its own
// copy of the metadata, so that reads of one array run at once. measure: the read's CPU time and
// wait for the storage are kept for pybind11_last_read_times
pybind11::array pybind11_read_zarr_opened(const OpenedZarr &opened, const std::vector<uint64_t> &startCoords,
                                          const std::vector<uint64_t> &endCoords, const std::string &order,
                                          const int64_t numThreads, const pybind11::object &out, const bool measure) {
    if (!order.empty() && order != "F" && order != "C") throw std::runtime_error("order must be 'F' or 'C', not '" + order + "'");
    if (numThreads < 0) throw std::runtime_error("num_threads must be at least 1");
    const CallThreads threads(numThreads);
    zarr Zarr;
    {
        pybind11::gil_scoped_release release;
        Zarr = opened.z;
    }
    return readOpened(Zarr, startCoords, endCoords, order, out, measure);
}

// A new write's array settings: format first (it sets the Zarr v3 defaults for order and
// dimension_separator), then the compressor, order, chunk file names and chunks or shards
static void configureWrite(zarr &Zarr, const std::string &fileName, const std::string &cname, const uint64_t clevel,
                           const std::string &order, const std::vector<uint64_t> &chunks, const std::string &dimension_separator,
                           const uint64_t zarr_format, const std::vector<uint64_t> &shards) {
    Zarr.set_fileName(fileName);
    try {
        Zarr.set_zarr_format(zarr_format);
    }
    catch (const std::string &) {
        throw std::runtime_error("zarr_format must be 2 or 3, not " + std::to_string(zarr_format));
    }
    Zarr.set_cname(cname);
    Zarr.set_clevel(clevel);
    Zarr.set_order(order);
    Zarr.set_dimension_separator(dimension_separator);
    // Zarr v3 shards (shards: the shard shape) of chunks (the inner chunk shape)
    if (shards.empty()) Zarr.set_chunks(chunks);
    else {
        Zarr.set_chunks(shards);
        Zarr.set_shard(true);
        Zarr.set_chunk_shape(chunks);
    }
}

// Once the data type is set: the shape, then the new metadata (.zarray or zarr.json) written,
// unless the write goes into an existing array (crop), whose metadata is read instead and whose
// data type the data must have; then the array set up for the region. (The GIL is released for
// the file system work.)
static void startWrite(zarr &Zarr, const std::string &fileName, const std::string &arrayDtype, const std::vector<uint64_t> &startCoords,
                       const std::vector<uint64_t> &endCoords, const bool crop) {
    Zarr.set_shape(endCoords);
    if(!crop || !(fileExists(fileName+"/.zarray") || fileExists(fileName+"/zarr.json"))){
        pybind11::gil_scoped_release release;
        writeZarray(Zarr);
    }
    else{
        {
            pybind11::gil_scoped_release release;
            Zarr = openZarr(fileName);
        }
        if (Zarr.get_ndims() != endCoords.size()) {
            throw std::runtime_error("The coordinates have " + std::to_string(endCoords.size()) +
                                     " values but the existing array has " + std::to_string(Zarr.get_ndims()) + " dimensions");
        }
        // The chunks are written with the data's element type, so it has to be the
        // array's (write_zarr converts the data to it)
        const DtypeInfo want = dtypeInfo(Zarr.get_dtype()), have = dtypeInfo(arrayDtype);
        if (want.kind != have.kind || want.bytes != have.bytes || want.bfloat16 != have.bfloat16) {
            throw std::runtime_error("The data's type (" + arrayDtype + ") does not match the existing array's (" +
                                     Zarr.get_dtype() + ")");
        }
    }
    pybind11::gil_scoped_release release;
    Zarr.set_chunkInfo(startCoords, endCoords);
}

void pybind11_write_zarr(const std::string &fileName, const pybind11::array &data, const std::vector<uint64_t> &startCoords = std::vector<uint64_t>{0, 0, 0},
                         const std::vector<uint64_t> endCoords = std::vector<uint64_t>{0, 0, 0}, const std::string &cname = "zstd",
                         const uint64_t clevel = 1, const std::string &order = "F", const std::vector<uint64_t> &chunks = std::vector<uint64_t>{256, 256, 256},
                         const std::string &dimension_separator = ".", const bool crop = false, const uint64_t zarr_format = 2,
                         const std::vector<uint64_t> &shards = std::vector<uint64_t>{}, const int64_t numThreads = 0,
                         const bool useUuid = true, const bool sparse = true){
    if (numThreads < 0) throw std::runtime_error("num_threads must be at least 1");
    const CallThreads threads(numThreads);

    zarr Zarr;
    configureWrite(Zarr, fileName, cname, clevel, order, chunks, dimension_separator, zarr_format, shards);

    // Map the numpy dtype via kind ('b' bool, 'u' unsigned int, 'i' signed int,
    // 'f' float, 'c' complex) and element size, and ml_dtypes' bfloat16 by name.
    // This is robust across platforms, unlike buffer-format strings (e.g. numpy
    // int64 reports 'l' on Linux while int64_t formats as 'q').
    const char kind = data.dtype().kind();
    const uint64_t itemsize = (uint64_t)data.dtype().itemsize();
    const bool bfloat16 = itemsize == 2 && kind == 'V' && data.dtype().attr("name").cast<std::string>() == "bfloat16";
    std::string arrayDtype;
    if (bfloat16) arrayDtype = "bfloat16";
    else if (kind == 'b' && itemsize == 1) arrayDtype = "|b1";
    else if (((kind == 'u' || kind == 'i') && (itemsize == 1 || itemsize == 2 || itemsize == 4 || itemsize == 8)) ||
             (kind == 'f' && (itemsize == 2 || itemsize == 4 || itemsize == 8)) ||
             (kind == 'c' && (itemsize == 8 || itemsize == 16))) {
        arrayDtype = std::string("<") + kind + std::to_string(itemsize);
    }
    else {
        throw std::runtime_error(std::string("Unsupported data type: kind '") + kind +
                                 "' with " + std::to_string(itemsize) + " bytes per element");
    }
    // (write_zarr converts data in the other byte order)
    if (!data.dtype().attr("isnative").cast<bool>()) {
        throw std::runtime_error("The data is not in this machine's byte order");
    }
    // The data's memory (bfloat16 through a uint16 view: NumPy cannot hand out
    // ml_dtypes' types as buffers)
    const pybind11::array buffer = bfloat16 ? data.attr("view")(pybind11::dtype("uint16")).cast<pybind11::array>() : data;
    pybind11::buffer_info info = buffer.request();
    const uint64_t dtype = itemsize * 8;
    Zarr.set_dtype(arrayDtype);

    // The writer reads the array in place through its strides (in elements), so
    // any memory layout works without a copy. The Python wrapper copies arrays
    // with reversed or repeated axes (negative or zero strides) first. A stride
    // is never used for an axis of length 1, nor for omitted trailing axes.
    std::vector<uint64_t> inStrides(endCoords.size(), 0);
    for (ssize_t d = 0; d < info.ndim && d < (ssize_t)inStrides.size(); d++) {
        if (info.shape[d] <= 1) continue;
        if (info.strides[d] <= 0 || info.strides[d] % (ssize_t)itemsize) {
            throw std::runtime_error("Unsupported memory layout: strides must be positive multiples of the element size");
        }
        inStrides[d] = (uint64_t)info.strides[d] / itemsize;
    }

    startWrite(Zarr, fileName, arrayDtype, startCoords, endCoords, crop);
    std::vector<uint64_t> writeShape(endCoords.size());
    for (size_t d = 0; d < endCoords.size(); d++) writeShape[d] = endCoords[d]-startCoords[d];

    // Write out the data
    uint8_t err = 0;
    {
        pybind11::gil_scoped_release release;
        err = parallelWriteZarr(Zarr, info.ptr, startCoords, endCoords, writeShape, inStrides, dtype, useUuid, crop, sparse);
    }
    if(err) throw std::runtime_error(Zarr.get_errString());
}

// Writing from GPU memory: the data at the device pointer ptr, of the data type dtype (a typestr in
// this machine's byte order, or "bfloat16"), shape and element strides, compressed on the GPU (see
// gpuwritezarr.h) with the GPU work on stream (a CUDA stream handle, 0 for the default stream);
// otherwise as pybind11_write_zarr. The GIL is released for the file system work and the write.
void pybind11_write_zarr_gpu(const std::string &fileName, const uint64_t ptr, const std::string &dtype, const std::vector<uint64_t> &shape,
                             const std::vector<int64_t> &strides, const std::vector<uint64_t> &startCoords,
                             const std::vector<uint64_t> &endCoords, const std::string &cname, const uint64_t clevel, const std::string &order,
                             const std::vector<uint64_t> &chunks, const std::string &dimension_separator, const bool crop,
                             const uint64_t zarr_format, const std::vector<uint64_t> &shards, const int64_t numThreads, const uint64_t stream,
                             const bool useUuid, const bool sparse){
    if (numThreads < 0) throw std::runtime_error("num_threads must be at least 1");
    const CallThreads threads(numThreads);
    zarr Zarr;
    configureWrite(Zarr, fileName, cname, clevel, order, chunks, dimension_separator, zarr_format, shards);
    const DtypeInfo t = dtypeInfo(dtype);
    if (!t.kind || t.bigEndian) throw std::runtime_error("Unsupported data type for a write from GPU memory: " + dtype);
    Zarr.set_dtype(dtype);
    // (strides along axes past the data's, and along axes of length 1, are never used)
    std::vector<uint64_t> inStrides(endCoords.size(), 0);
    for (size_t d = 0; d < shape.size() && d < inStrides.size(); d++) {
        if (shape[d] <= 1) continue;
        if (strides[d] <= 0) throw std::runtime_error("Unsupported memory layout: strides must be positive");
        inStrides[d] = (uint64_t)strides[d];
    }
    startWrite(Zarr, fileName, dtype, startCoords, endCoords, crop);
    std::vector<uint64_t> writeShape(endCoords.size());
    for (size_t d = 0; d < endCoords.size(); d++) writeShape[d] = endCoords[d]-startCoords[d];
    uint8_t err = 0;
    {
        pybind11::gil_scoped_release release;
        err = gpuWriteZarr(Zarr, reinterpret_cast<const void*>(static_cast<uintptr_t>(ptr)), startCoords, endCoords, writeShape, inStrides,
                           t.bytes*8, useUuid, crop, sparse, reinterpret_cast<void*>(static_cast<uintptr_t>(stream)));
    }
    if (err) {
        std::string e = Zarr.get_errString();
        while (!e.empty() && e.back() == '\n') e.pop_back();
        throw std::runtime_error(e);
    }
}

// A new array's metadata alone (.zarray or zarr.json), with no chunks, as a write of data of the data type dtype
// (a typestr in this machine's byte order, or "bfloat16") and of this shape makes it; the regions written into
// it later fill it. The chunk files already at fileName are left as they are.
void pybind11_create_zarr(const std::string &fileName, const std::vector<uint64_t> &shape, const std::string &dtype,
                          const std::string &cname, const uint64_t clevel, const std::string &order, const std::vector<uint64_t> &chunks,
                          const std::string &dimension_separator, const uint64_t zarr_format, const std::vector<uint64_t> &shards){
    zarr Zarr;
    configureWrite(Zarr, fileName, cname, clevel, order, chunks, dimension_separator, zarr_format, shards);
    const DtypeInfo t = dtypeInfo(dtype);
    if (!t.kind || t.bigEndian) throw std::runtime_error("Unsupported data type: " + dtype);
    Zarr.set_dtype(dtype);
    Zarr.set_shape(shape);
    pybind11::gil_scoped_release release;
    writeZarray(Zarr);
}

// data (C- or F-contiguous, of a type MATLAB has: bool, integers, float32 or float64) converted to the data type
// dtype (a typestr, or "bfloat16", returned as its bits: uint16) as MATLAB converts between classes (see
// convertValues), into a new array of the same shape and memory order
pybind11::array pybind11_convert_values(const pybind11::array &data, const std::string &dtype, const int64_t numThreads){
    if (numThreads < 0) throw std::runtime_error("num_threads must be at least 1");
    const DtypeInfo t = dtypeInfo(dtype);
    if (!t.kind || t.bigEndian) throw std::runtime_error("Unsupported data type: " + dtype);
    const bool cOrder = data.flags() & pybind11::array::c_style, fOrder = data.flags() & pybind11::array::f_style;
    if (!cOrder && !fOrder) throw std::runtime_error("The data to convert must be C- or F-contiguous");
    const char kind = data.dtype().kind();
    const uint64_t itemsize = (uint64_t)data.dtype().itemsize();
    const std::string from = kind == 'b' ? "|b1" : std::string("<") + kind + std::to_string(itemsize);
    const std::vector<ssize_t> shape(data.shape(), data.shape() + data.ndim());
    const pybind11::dtype outType(t.bfloat16 ? std::string("uint16") : dtype);
    std::vector<ssize_t> strides(shape.size());
    ssize_t acc = (ssize_t)outType.itemsize();
    if (cOrder) for (ssize_t d = (ssize_t)shape.size()-1; d >= 0; d--) { strides[d] = acc; acc *= shape[d]; }
    else for (size_t d = 0; d < shape.size(); d++) { strides[d] = acc; acc *= shape[d]; }
    pybind11::array out(outType, shape, strides);
    const void* src = data.data();
    void* dst = out.mutable_data();
    const uint64_t n = (uint64_t)data.size();
    bool ok = false;
    {
        pybind11::gil_scoped_release release;
        const CallThreads threads(numThreads);
        ok = convertValues(src, from, dst, dtype, n);
    }
    if (!ok) throw std::runtime_error("Cannot convert data of the type " + from + " to " + dtype);
    return out;
}

// Turn a Zarr v2 array into a Zarr v3 one in place (only its metadata is rewritten)
void pybind11_convert_to_v3(const std::string &fileName){
    zarr Zarr = openZarr(fileName);
    try {
        Zarr.convert_to_v3();
    }
    catch (const std::string &e) {
        if (e.rfind("v3Unsupported:", 0) == 0) {
            throw std::runtime_error("Cannot convert " + fileName + " to Zarr v3: " + e.substr(e.find(':') + 1));
        }
        throw metadataError(fileName, e);
    }
}

// Reading into GPU memory: requests of (file, start, end, the output's device pointer, its data
// type (a typestr, or "bfloat16") and shape), all of one order ("C" or "F"), read in one GPU batch.
// stream: a CUDA stream handle (0 for the default stream). The GIL is released throughout.
// Arrays opened at once (on network file systems each open waits on the server), each once
// however often the names give it: which[i] is the place of fileNames[i]'s array in the result
// (a batch of 16,384 single time points read from 1,430 arrays opened each 11 times); call
// without the GIL. An array that cannot be opened fails the call, the first one by name's
static std::vector<zarr> openZarrs(const std::vector<std::string> &fileNames, std::vector<size_t> &which) {
    std::vector<const std::string*> distinct;
    std::unordered_map<std::string, size_t> at;
    which.resize(fileNames.size());
    for (size_t i = 0; i < fileNames.size(); i++) {
        const auto e = at.emplace(fileNames[i], distinct.size());
        if (e.second) distinct.push_back(&fileNames[i]);
        which[i] = e.first->second;
    }
    const size_t n = distinct.size(), nt = std::min<size_t>(n, 16);
    std::vector<zarr> arrays(n);
    std::vector<std::exception_ptr> failed(n);
    auto open = [&](const size_t first) {
        for (size_t i = first; i < n; i += nt) {
            try { arrays[i] = openZarr(*distinct[i]); }
            catch (...) { failed[i] = std::current_exception(); }
        }
    };
    std::vector<std::thread> openers;
    for (size_t t = 1; t < nt; t++) openers.emplace_back(open, t);
    if (n) open(0);
    for (auto &t : openers) t.join();
    for (const auto &f : failed) if (f) std::rethrow_exception(f);
    return arrays;
}

// The region [start, end) that coordinates name (all-zero ones: the whole array), checked
static void regionOf(zarr &Z, const std::vector<uint64_t> &sc, const std::vector<uint64_t> &ec, std::vector<uint64_t> &start,
                     std::vector<uint64_t> &end) {
    const uint64_t nDims = Z.get_ndims();
    start.assign(nDims, 0);
    end.resize(nDims);
    for (uint64_t d = 0; d < nDims; d++) end[d] = Z.get_shape(d);
    if (!std::all_of(sc.begin(), sc.end(), [](uint64_t v) { return v == 0; })) start = fitCoords(sc, nDims, false, "start_coords");
    if (!std::all_of(ec.begin(), ec.end(), [](uint64_t v) { return v == 0; })) end = fitCoords(ec, nDims, true, "end_coords");
    for (uint64_t d = 0; d < nDims; d++) {
        if (end[d] > Z.get_shape(d) || start[d] >= end[d]) throw std::runtime_error("Invalid start_coords or end_coords for axis " + std::to_string(d));
    }
}

// The regions' shapes, NumPy data type names ("bfloat16" for bfloat16) and storage orders ("C"
// or "F"), as pybind11_read_zarr returns them: for arrays made elsewhere (in GPU memory) to read
// into. The arrays' metadata is read at once.
std::vector<std::tuple<std::vector<uint64_t>, std::string, std::string>> pybind11_region_info(
        const std::vector<std::tuple<std::string, std::vector<uint64_t>, std::vector<uint64_t>>> &requests) {
    std::vector<std::tuple<std::vector<uint64_t>, std::string, std::string>> info;
    pybind11::gil_scoped_release release;
    std::vector<std::string> names;
    for (const auto &q : requests) names.push_back(std::get<0>(q));
    std::vector<size_t> which;
    std::vector<zarr> arrays = openZarrs(names, which);
    for (size_t i = 0; i < requests.size(); i++) {
        zarr &Z = arrays[which[i]];
        std::vector<uint64_t> start, end, shape;
        regionOf(Z, std::get<1>(requests[i]), std::get<2>(requests[i]), start, end);
        for (size_t d = 0; d < start.size(); d++) shape.push_back(end[d] - start[d]);
        const DtypeInfo t = dtypeInfo(Z.get_dtype());
        const std::string bits = std::to_string(8*t.bytes);
        std::string name;
        if (t.bfloat16) name = "bfloat16";
        else if (t.kind == 'b') name = "bool";
        else if (t.kind == 'i') name = "int" + bits;
        else if (t.kind == 'u') name = "uint" + bits;
        else if (t.kind == 'f') name = "float" + bits;
        else if (t.kind == 'c') name = "complex" + bits;
        else throw std::runtime_error("Unsupported data type: " + Z.get_dtype());
        info.emplace_back(shape, name, Z.get_order() == "C" ? "C" : "F");
    }
    return info;
}

void pybind11_read_zarr_gpu(const std::vector<std::tuple<std::string, std::vector<uint64_t>, std::vector<uint64_t>, uint64_t,
                                                         std::string, std::vector<uint64_t>>> &requests,
                            const std::string &order, const int64_t numThreads, const uint64_t stream) {
    if (order != "C" && order != "F") throw std::runtime_error("order must be 'C' or 'F', not '" + order + "'");
    if (numThreads < 0) throw std::runtime_error("num_threads must be at least 1");
    std::string err;
    uint8_t r = 0;
    {
        pybind11::gil_scoped_release release;
        std::vector<std::string> names;
        for (const auto &q : requests) names.push_back(std::get<0>(q));
        // (the regions of one array share its metadata: the GPU reads only look at it)
        std::vector<size_t> which;
        std::vector<zarr> arrays = openZarrs(names, which);
        std::vector<GpuRegion> regions;
        for (size_t i = 0; i < requests.size(); i++) {
            const auto &q = requests[i];
            zarr &Z = arrays[which[i]];
            const uint64_t nDims = Z.get_ndims();
            std::vector<uint64_t> start, end;
            regionOf(Z, std::get<1>(q), std::get<2>(q), start, end);
            std::string shapeText;
            bool sameShape = std::get<5>(q).size() == nDims;
            for (uint64_t d = 0; d < nDims; d++) {
                shapeText += (d ? ", " : "") + std::to_string(end[d] - start[d]);
                sameShape = sameShape && std::get<5>(q)[d] == end[d] - start[d];
            }
            if (!sameShape) throw std::runtime_error("out must have the shape of the region read, (" + shapeText + ")");
            const DtypeInfo want = dtypeInfo(Z.get_dtype()), have = dtypeInfo(std::get<4>(q));
            if (!have.kind || have.kind != want.kind || have.bytes != want.bytes || have.bfloat16 != want.bfloat16 || have.bigEndian) {
                throw std::runtime_error("out has data type " + std::get<4>(q) + ", not the array's " + Z.get_dtype() +
                                         (want.bigEndian ? " (in this machine's byte order)" : ""));
            }
            regions.push_back({&Z, start, end, reinterpret_cast<void*>(static_cast<uintptr_t>(std::get<3>(q)))});
        }
        r = gpuReadZarr(regions, order == "C", reinterpret_cast<void*>(static_cast<uintptr_t>(stream)), (int)numThreads, err);
    }
    if (r) throw std::runtime_error(err);
}

PYBIND11_MODULE(cppzarr, m) {
	pybind11::module::import("numpy");

    m.doc() = "cpp-zarr python bindings";

	m.def("pybind11_read_zarr", &pybind11_read_zarr, pybind11::arg("fileName"), pybind11::arg("startCoords"),
	      pybind11::arg("endCoords"), pybind11::arg("order"), pybind11::arg("numThreads") = 0,
	      pybind11::arg("out") = pybind11::none(), "Read a zarr file");

	pybind11::class_<OpenedZarr>(m, "_OpenedZarr", "An array's metadata, read once for all of a batch's reads of it");

	m.def("pybind11_open_zarrs", [](const std::vector<std::string> &fileNames) {
	          std::vector<OpenedZarr> opened;
	          {
	              pybind11::gil_scoped_release release;
	              std::vector<size_t> which;
	              std::vector<zarr> arrays = openZarrs(fileNames, which);
	              opened.reserve(fileNames.size());
	              for (size_t i = 0; i < fileNames.size(); i++) opened.push_back({arrays[which[i]]});
	          }
	          return opened;
	      }, pybind11::arg("fileNames"), "Open arrays' metadata at once, each array once (for read_zarr_batch)");

	m.def("pybind11_on_network", [](const std::string &folder) {
	          pybind11::gil_scoped_release release;
	          return readsFromNetwork(folder);
	      }, pybind11::arg("folder"), "Whether an array's folder is on network storage, as the reads find it (Linux; false elsewhere)");

	m.def("pybind11_read_zarr_opened", &pybind11_read_zarr_opened, pybind11::arg("opened"), pybind11::arg("startCoords"),
	      pybind11::arg("endCoords"), pybind11::arg("order"), pybind11::arg("numThreads") = 0,
	      pybind11::arg("out") = pybind11::none(), pybind11::arg("measure") = false,
	      "Read a region of an array opened with pybind11_open_zarrs");

	m.def("pybind11_last_read_times", []() { return std::make_pair(lastReadCpu, lastReadWait); },
	      "The calling thread's last measured read: its CPU time and its wait for the storage in seconds (-1: unknown)");

	m.def("pybind11_read_times_available", []() { return threadClocks().ok; },
	      "Whether reads can be measured here (their wait for the storage)");

	m.def("pybind11_write_zarr", &pybind11_write_zarr, pybind11::arg("fileName"), pybind11::arg("data"), pybind11::arg("startCoords"),
	      pybind11::arg("endCoords"), pybind11::arg("cname"), pybind11::arg("clevel"), pybind11::arg("order"), pybind11::arg("chunks"),
	      pybind11::arg("dimension_separator"), pybind11::arg("crop"), pybind11::arg("zarr_format"), pybind11::arg("shards"),
	      pybind11::arg("numThreads") = 0, pybind11::arg("useUuid") = true, pybind11::arg("sparse") = true, "Write a zarr file");

	m.def("pybind11_write_zarr_gpu", &pybind11_write_zarr_gpu, pybind11::arg("fileName"), pybind11::arg("ptr"), pybind11::arg("dtype"),
	      pybind11::arg("shape"), pybind11::arg("strides"), pybind11::arg("startCoords"), pybind11::arg("endCoords"), pybind11::arg("cname"),
	      pybind11::arg("clevel"), pybind11::arg("order"), pybind11::arg("chunks"), pybind11::arg("dimension_separator"),
	      pybind11::arg("crop"), pybind11::arg("zarr_format"), pybind11::arg("shards"), pybind11::arg("numThreads") = 0,
	      pybind11::arg("stream") = 0, pybind11::arg("useUuid") = true, pybind11::arg("sparse") = true, "Write a zarr file from GPU memory");

	m.def("pybind11_create_zarr", &pybind11_create_zarr, pybind11::arg("fileName"), pybind11::arg("shape"), pybind11::arg("dtype"),
	      pybind11::arg("cname"), pybind11::arg("clevel"), pybind11::arg("order"), pybind11::arg("chunks"), pybind11::arg("dimension_separator"),
	      pybind11::arg("zarr_format"), pybind11::arg("shards"), "Write a new array's metadata alone");

	m.def("pybind11_convert_values", &pybind11_convert_values, pybind11::arg("data"), pybind11::arg("dtype"), pybind11::arg("numThreads") = 0,
	      "Convert values as MATLAB converts between classes");

	m.def("pybind11_convert_to_v3", &pybind11_convert_to_v3, pybind11::arg("fileName"),
	      "Convert a Zarr v2 array to Zarr v3 in place (metadata only)");

	m.def("pybind11_release_buffers", []() {
	          pybind11::gil_scoped_release nogil;
	          releaseBuffers();
	      }, "Free the buffers reads and writes keep between calls");

	m.def("pybind11_kept_bytes", []() { return std::make_tuple(keptBufferBytes(), keptReadBufferBytes(), keptBufferCap()); },
	      "The bytes of the buffers reads and writes keep between calls, those of the reads, and the most kept");

	m.def("pybind11_read_zarr_gpu", &pybind11_read_zarr_gpu, pybind11::arg("requests"), pybind11::arg("order"),
	      pybind11::arg("numThreads") = 0, pybind11::arg("stream") = 0, "Read regions of zarr files into GPU memory");

	m.def("pybind11_region_info", &pybind11_region_info, pybind11::arg("requests"),
	      "Shapes, data types and storage orders of regions of zarr files, for arrays to read into");

	m.def("pybind11_gpu_available", []() { std::string why; const bool ok = gpuReadAvailable(why); return std::make_pair(ok, why); },
	      "Whether GPU reads can run here, and why not");

	m.def("pybind11_gpu_set_nvcomp", &gpuSetNvcompLibrary, pybind11::arg("path"),
	      "Where to load nvCOMP from, before the first GPU read");

	m.def("pybind11_gpu_set_cufile", &gpuSetCufileLibrary, pybind11::arg("path"),
	      "Where to load cuFile (GPUDirect Storage) from, before the first GPU read");

	m.def("pybind11_default_threads", &callThreads,
	      "The threads a call uses by default from the calling thread (OMP_NUM_THREADS, or every CPU it may run on)");
}
