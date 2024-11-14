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

pybind11::array pybind11_read_zarr(const std::string& fileName){
    zarr Zarr(fileName);
    uint64_t dims[3] = {Zarr.get_shape(0), Zarr.get_shape(1), Zarr.get_shape(2)};
    std::vector<uint64_t> startCoords = {0, 0, 0};
    std::vector<uint64_t> endCoords = {dims[0], dims[1], dims[2]};
    void* data = parallelReadZarrWriteWrapper(Zarr, true, startCoords, endCoords);

	switch (Zarr.dtypeBytes()) {
        case 1:  // 8-bit unsigned int
            return create_pybind11_array<uint8_t>(data, dims);
		case 2: // 16-bit unsigned int
            return create_pybind11_array<uint16_t>(data, dims);
        case 4: // 32-bit float
            return create_pybind11_array<float>(data, dims);
        case 8: // 64-bit double
            return create_pybind11_array<double>(data, dims);
        default:
            throw std::runtime_error("Unsupported data type");
    }
}

void pybind11_write_zarr(const std::string &fileName, const pybind11::array &data, const std::string &cname = "zstd",
                         const uint64_t clevel = 1, const std::string &order = "F", const std::vector<uint64_t> chunks = std::vector<uint64_t>{256, 256, 256},
                         const std::string &dimension_separator = ".") {
    // Determine the dtype based on the NumPy array type
    pybind11::buffer_info info = data.request();

    // Get the dimensions of the array
    uint64_t dims[3] = {static_cast<uint64_t>(info.shape[0]), static_cast<uint64_t>(info.shape[1]), static_cast<uint64_t>(info.shape[2])};

    // Set up the shape of the data that will be written out
	const std::vector<uint64_t> startCoords{0, 0, 0};
    const std::vector<uint64_t> endCoords{dims[0], dims[1], dims[2]};
	const std::vector<uint64_t> writeShape({endCoords[0]-startCoords[0],
                                      endCoords[1]-startCoords[1],
                                      endCoords[2]-startCoords[2]});
    bool crop = false;

    zarr Zarr;
    Zarr.set_fileName(fileName);
    Zarr.set_cname(cname);
    Zarr.set_clevel(clevel);
    Zarr.set_order(order);
    Zarr.set_chunks(chunks);
    Zarr.set_dimension_separator(dimension_separator);

    uint64_t dtype;
    if (info.format == pybind11::format_descriptor<uint8_t>::format()) {
        dtype = 8;
        Zarr.set_dtype("<u1");
    }
	else if (info.format == pybind11::format_descriptor<uint16_t>::format()) {
        dtype = 16;
        Zarr.set_dtype("<u2");
    }
	else if (info.format == pybind11::format_descriptor<float>::format()) {
        dtype = 32;
        Zarr.set_dtype("<f4");
    }
	else if (info.format == pybind11::format_descriptor<double>::format()) {
        dtype = 64;
        Zarr.set_dtype("<f8");
    }
	else {
        throw std::runtime_error("Unsupported data type");
    }

    Zarr.set_shape(writeShape);
    Zarr.set_chunks({256,256,256});
    Zarr.set_chunkInfo(startCoords, endCoords);

    // Write out the new .zarray file
    Zarr.write_zarray();

    // Write out the data
    parallelWriteZarr(Zarr, info.ptr, startCoords, endCoords, writeShape, dtype, true, crop);

    // Call the function to write the data to a TIFF file
    //writeTiffParallelHelper(fileName.c_str(), info.ptr, dtype, "w", dims[0], dims[1], dims[2], 0, 0, compression);
}

/*
pybind11::tuple pybind11_get_image_shape(const std::string& fileName){
    uint64_t* dims = getImageSize(fileName.c_str());
	pybind11::tuple shape = pybind11::make_tuple(dims[2], dims[0], dims[1]);
    free(dims);
	return shape;
}
*/

PYBIND11_MODULE(cppzarr, m) {
	pybind11::module::import("numpy");

    m.doc() = "cpp-zarr python bindings";

	m.def("pybind11_read_zarr", &pybind11_read_zarr, "Read a zarr file");

	m.def("pybind11_write_zarr", &pybind11_write_zarr, pybind11::arg("fileName"), pybind11::arg("data"), pybind11::arg("cname"),
	      pybind11::arg("clevel"), pybind11::arg("order"), pybind11::arg("chunks"), pybind11::arg("dimension_separator"), "Write a zarr file");

	//m.def("pybind11_get_image_shape", &pybind11_get_image_shape, "Get the image shape without reading the entire image");
}
