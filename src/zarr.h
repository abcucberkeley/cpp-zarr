#ifndef ZARR_H
#define ZARR_H
#include <stdint.h>
#include <vector>
#include <string>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

class zarr
{
public:
    zarr();
    zarr(const std::string &fileName);
    zarr(const std::string &fileName, const std::vector<uint64_t> &chunks,
         uint64_t blocksize, uint64_t clevel, const std::string &cname,
         const std::string &id, uint64_t shuffle, const std::string &dimension_separator, const std::string &dtype,
         const int64_t &fill_value, const std::vector<std::string> &filters,
         const std::string &order, const std::vector<uint64_t> &shape,
         uint64_t zarr_format, const std::vector<uint64_t> subfolders);
    ~zarr();
    void write_zarray();
    const std::string &get_fileName() const;
    void set_fileName(const std::string &fileName);
    const uint64_t &get_chunks(const uint64_t &index) const;
    void set_chunks(const std::vector<uint64_t> &chunks);
    const uint64_t &get_clevel() const;
    void set_clevel(const uint64_t &clevel);
    const std::string &get_cname() const;
    void set_cname(const std::string &cname);
    const std::string &get_dimension_separator() const;
    void set_dimension_separator(const std::string &dimension_separator);
    const std::string &get_dtype() const;
    void set_dtype(const std::string &dtype);
    const int64_t &get_fill_value() const;
    void set_fill_value(const int64_t &fill_value);
    const std::string &get_order() const;
    void set_order(const std::string &order);
    const uint64_t &get_shape(const uint64_t &index) const;
    void set_shape(const std::vector<uint64_t> &shape);
    const std::string get_subfoldersString(const std::vector<uint64_t> &cAV) const;
    void set_subfolders(const std::vector<uint64_t> &subfolders);
    const std::vector<uint64_t> get_chunkAxisVals(const std::string &fileName) const;
    void set_chunkInfo(const std::vector<uint64_t> &startCoords,
                             const std::vector<uint64_t> &endCoords);
    const std::string &get_chunkNames(const uint64_t &index) const;
    const uint64_t &get_numChunks() const;
    const std::string &get_errString() const;
    void set_errString(const std::string &errString);
private:
    void set_jsonValues();
    void write_jsonValues();
    uint64_t fastCeilDiv(uint64_t num, uint64_t denom);
    void createSubfolders();
    std::string fileName;
    json zarray;
    std::vector<uint64_t> chunks;
    uint64_t blocksize;
    uint64_t clevel;
    std::string cname;
    std::string id;
    uint64_t shuffle;
    std::string dimension_separator;
    std::string dtype;
    int64_t fill_value;
    std::vector<std::string> filters;
    std::string order;
    std::vector<uint64_t> shape;
    uint64_t zarr_format;
    std::vector<uint64_t> subfolders;

    std::vector<std::string> chunkNames;
    uint64_t numChunks;

    std::string errString;
};
#endif