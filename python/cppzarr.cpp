#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include "zarr.h"
#include "parallelreadzarr.h"
#include "parallelwritezarr.h"
#include "helperfunctions.h"

template <typename T>
pybind11::array_t<T> create_pybind11_array(void* data, const uint64_t* dims) {
    auto deleter = [](void* ptr) { free(ptr); };

    std::vector<ssize_t> strides = {
        static_cast<ssize_t>(sizeof(T)),
        static_cast<ssize_t>(dims[0] * sizeof(T)),
        static_cast<ssize_t>(dims[1] * dims[0] * sizeof(T))
    };

    return pybind11::array_t<T>(
        {dims[0], dims[1], dims[2]},  // shape (y, x, z)
        strides,
        static_cast<T*>(data),
        pybind11::capsule(data, deleter)
    );
}

pybind11::array pybind11_read_zarr(const std::string &fileName, const std::vector<uint64_t> &startCoords = std::vector<uint64_t>{0, 0, 0},
                                   std::vector<uint64_t> endCoords = std::vector<uint64_t>{0, 0, 0}){
    zarr Zarr(fileName);
    uint64_t dims[3] = {Zarr.get_shape(0), Zarr.get_shape(1), Zarr.get_shape(2)};
    bool crop = false;

    // If the startCoords are not all zero then we are using user defined startCoords
    if (!std::all_of(startCoords.begin(), startCoords.end(), [](int i) { return i==0; })){
        crop = true;
    }

    // If the endCoords are not all zero then we are using user defined endCoords
    // If we are only using user defined startCoords then we want to set the endCoords to the axis size
    if (!std::all_of(endCoords.begin(), endCoords.end(), [](int i) { return i==0; })){
        crop = true;
    }
    else if(crop) endCoords.assign({dims[0], dims[1], dims[2]});

    if(crop){
        Zarr.set_chunkInfo(startCoords, endCoords);
        dims[0] = endCoords[0]-startCoords[0];
        dims[1] = endCoords[1]-startCoords[1];
        dims[2] = endCoords[2]-startCoords[2];
    }
    void* data = parallelReadZarrWriteWrapper(Zarr, crop, startCoords, endCoords);
    if (!data) throw std::runtime_error("Failed to read the zarr file (unsupported dtype or read error)");

    // Dispatch on the full dtype: byte width alone is ambiguous now that signed,
    // unsigned, and float types of the same size are all supported.
    const std::string &dtype = Zarr.get_dtype();
    const char kind = dtype.size() == 3 ? dtype[1] : '\0';
    const char dsize = dtype.size() == 3 ? dtype[2] : '\0';
    if (kind == 'u') {
        if (dsize == '1') return create_pybind11_array<uint8_t>(data, dims);
        if (dsize == '2') return create_pybind11_array<uint16_t>(data, dims);
        if (dsize == '4') return create_pybind11_array<uint32_t>(data, dims);
        if (dsize == '8') return create_pybind11_array<uint64_t>(data, dims);
    }
    else if (kind == 'i') {
        if (dsize == '1') return create_pybind11_array<int8_t>(data, dims);
        if (dsize == '2') return create_pybind11_array<int16_t>(data, dims);
        if (dsize == '4') return create_pybind11_array<int32_t>(data, dims);
        if (dsize == '8') return create_pybind11_array<int64_t>(data, dims);
    }
    else if (kind == 'f') {
        if (dsize == '4') return create_pybind11_array<float>(data, dims);
        if (dsize == '8') return create_pybind11_array<double>(data, dims);
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

    Zarr.set_shape(endCoords);

    // Write out the new .zarray file
    if(!crop || !fileExists(fileName+"/.zarray")) Zarr.write_zarray();
    else{
        Zarr = zarr(fileName);
    }

    const std::vector<uint64_t> writeShape({endCoords[0]-startCoords[0],
                                  endCoords[1]-startCoords[1],
                                  endCoords[2]-startCoords[2]});
    Zarr.set_chunkInfo(startCoords, endCoords);

    // Write out the data
    uint8_t err = parallelWriteZarr(Zarr, info.ptr, startCoords, endCoords, writeShape, dtype, true, crop);
    if(err) throw std::runtime_error(Zarr.get_errString());
}

PYBIND11_MODULE(cppzarr, m) {
	pybind11::module::import("numpy");

    m.doc() = "cpp-zarr python bindings";

	m.def("pybind11_read_zarr", &pybind11_read_zarr, "Read a zarr file");

	m.def("pybind11_write_zarr", &pybind11_write_zarr, pybind11::arg("fileName"), pybind11::arg("startCoords"), pybind11::arg("endCoords"),
	      pybind11::arg("data"), pybind11::arg("cname"), pybind11::arg("clevel"), pybind11::arg("order"), pybind11::arg("chunks"),
	      pybind11::arg("dimension_separator"), pybind11::arg("crop"), "Write a zarr file");
}
