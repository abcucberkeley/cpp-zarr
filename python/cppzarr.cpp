#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "zarr.h"
#include "parallelreadzarr.h"
#include "parallelwritezarr.h"
#include "helperfunctions.h"

template <typename T>
pybind11::array_t<T> create_pybind11_array(void* data, const std::vector<uint64_t> &dims, const bool cOrder) {
    auto deleter = [](void* ptr) { free(ptr); };

    // The data is in F order (the first axis is contiguous) or C order (the last)
    const size_t n = dims.size();
    std::vector<ssize_t> shape(n), strides(n);
    ssize_t stride = sizeof(T);
    for (size_t i = 0; i < n; i++) {
        const size_t d = cOrder ? n-1-i : i;
        shape[d] = static_cast<ssize_t>(dims[d]);
        strides[d] = stride;
        stride *= static_cast<ssize_t>(dims[d]);
    }

    return pybind11::array_t<T>(
        shape,
        strides,
        static_cast<T*>(data),
        pybind11::capsule(data, deleter)
    );
}

// The zarr class reports metadata problems by throwing std::string; turn them
// into Python exceptions with a readable message
static zarr openZarr(const std::string &fileName) {
    try {
        return zarr(fileName);
    }
    catch (const std::string &e) {
        throw std::runtime_error("Cannot read the zarr metadata of " + fileName + " (" + e + ")");
    }
}

static void writeZarray(zarr &Zarr) {
    try {
        Zarr.write_zarray();
    }
    catch (const std::string &e) {
        throw std::runtime_error("Cannot write the zarr metadata of " + Zarr.get_fileName() + " (" + e + ")");
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

pybind11::array pybind11_read_zarr(const std::string &fileName, const std::vector<uint64_t> &startCoords = std::vector<uint64_t>{0, 0, 0},
                                   std::vector<uint64_t> endCoords = std::vector<uint64_t>{0, 0, 0}, const std::string &order = ""){
    // order is the memory layout of the returned array: "F" (first axis
    // contiguous), "C" (last axis contiguous), or "" for the file's storage order
    if (!order.empty() && order != "F" && order != "C") throw std::runtime_error("order must be 'F' or 'C', not '" + order + "'");
    zarr Zarr = openZarr(fileName);
    const bool cOrder = order.empty() ? Zarr.get_order() == "C" : order == "C";
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
        Zarr.set_chunkInfo(start, end);
    }
    void* data = parallelReadZarrWriteWrapper(Zarr, crop, start, end, cOrder);
    if (!data) throw std::runtime_error("Failed to read the zarr file (unsupported dtype or read error)");

    // Dispatch on the full dtype: byte width alone is ambiguous now that signed,
    // unsigned, and float types of the same size are all supported.
    const std::string &dtype = Zarr.get_dtype();
    const char kind = dtype.size() == 3 ? dtype[1] : '\0';
    const char dsize = dtype.size() == 3 ? dtype[2] : '\0';
    if (kind == 'u') {
        if (dsize == '1') return create_pybind11_array<uint8_t>(data, dims, cOrder);
        if (dsize == '2') return create_pybind11_array<uint16_t>(data, dims, cOrder);
        if (dsize == '4') return create_pybind11_array<uint32_t>(data, dims, cOrder);
        if (dsize == '8') return create_pybind11_array<uint64_t>(data, dims, cOrder);
    }
    else if (kind == 'i') {
        if (dsize == '1') return create_pybind11_array<int8_t>(data, dims, cOrder);
        if (dsize == '2') return create_pybind11_array<int16_t>(data, dims, cOrder);
        if (dsize == '4') return create_pybind11_array<int32_t>(data, dims, cOrder);
        if (dsize == '8') return create_pybind11_array<int64_t>(data, dims, cOrder);
    }
    else if (kind == 'f') {
        if (dsize == '4') return create_pybind11_array<float>(data, dims, cOrder);
        if (dsize == '8') return create_pybind11_array<double>(data, dims, cOrder);
    }
    free(data);
    throw std::runtime_error("Unsupported data type: " + dtype);
}

void pybind11_write_zarr(const std::string &fileName, const pybind11::array &data, const std::vector<uint64_t> &startCoords = std::vector<uint64_t>{0, 0, 0},
                         const std::vector<uint64_t> endCoords = std::vector<uint64_t>{0, 0, 0}, const std::string &cname = "zstd",
                         const uint64_t clevel = 1, const std::string &order = "F", const std::vector<uint64_t> &chunks = std::vector<uint64_t>{256, 256, 256},
                         const std::string &dimension_separator = ".", const bool crop = false){
    // Determine the dtype based on the NumPy array type
    pybind11::buffer_info info = data.request();

    zarr Zarr;
    Zarr.set_fileName(fileName);
    Zarr.set_cname(cname);
    Zarr.set_clevel(clevel);
    Zarr.set_order(order);
    Zarr.set_chunks(chunks);
    Zarr.set_dimension_separator(dimension_separator);

    // Map the numpy dtype via kind ('u' unsigned int, 'i' signed int, 'f' float)
    // and element size. This is robust across platforms, unlike buffer-format
    // strings (e.g. numpy int64 reports 'l' on Linux while int64_t formats as 'q').
    const char kind = data.dtype().kind();
    const uint64_t itemsize = (uint64_t)data.dtype().itemsize();
    const bool intKind = (kind == 'u' || kind == 'i');
    if (!((intKind && (itemsize == 1 || itemsize == 2 || itemsize == 4 || itemsize == 8)) ||
          (kind == 'f' && (itemsize == 4 || itemsize == 8)))) {
        throw std::runtime_error(std::string("Unsupported data type: kind '") + kind +
                                 "' with " + std::to_string(itemsize) + " bytes per element");
    }
    const uint64_t dtype = itemsize * 8;
    Zarr.set_dtype(std::string("<") + kind + std::to_string(itemsize));

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

    Zarr.set_shape(endCoords);

    // Write out the new .zarray file
    if(!crop || !fileExists(fileName+"/.zarray")) writeZarray(Zarr);
    else{
        Zarr = openZarr(fileName);
        if (Zarr.get_ndims() != endCoords.size()) {
            throw std::runtime_error("The coordinates have " + std::to_string(endCoords.size()) +
                                     " values but the existing array has " + std::to_string(Zarr.get_ndims()) + " dimensions");
        }
    }

    std::vector<uint64_t> writeShape(endCoords.size());
    for (size_t d = 0; d < endCoords.size(); d++) writeShape[d] = endCoords[d]-startCoords[d];
    Zarr.set_chunkInfo(startCoords, endCoords);

    // Write out the data
    uint8_t err = parallelWriteZarr(Zarr, info.ptr, startCoords, endCoords, writeShape, inStrides, dtype, true, crop);
    if(err) throw std::runtime_error(Zarr.get_errString());
}

PYBIND11_MODULE(cppzarr, m) {
	pybind11::module::import("numpy");

    m.doc() = "cpp-zarr python bindings";

	m.def("pybind11_read_zarr", &pybind11_read_zarr, pybind11::arg("fileName"), pybind11::arg("startCoords"),
	      pybind11::arg("endCoords"), pybind11::arg("order"), "Read a zarr file");

	m.def("pybind11_write_zarr", &pybind11_write_zarr, pybind11::arg("fileName"), pybind11::arg("data"), pybind11::arg("startCoords"),
	      pybind11::arg("endCoords"), pybind11::arg("cname"), pybind11::arg("clevel"), pybind11::arg("order"), pybind11::arg("chunks"),
	      pybind11::arg("dimension_separator"), pybind11::arg("crop"), "Write a zarr file");
}
