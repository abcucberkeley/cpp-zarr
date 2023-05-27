#include <stdio.h>
#include <cstdlib>
#include <cstddef>
#include <sys/stat.h>
#ifdef _WIN32
#include <stdarg.h>
#include <sys/time.h>
#else
#include <uuid/uuid.h>
#endif
#include <iomanip>
#include <iostream>
#include <fstream>
#include "zarr.h"
#include "helperfunctions.h"

// Create a blank zarr object with default values
zarr::zarr() :
fileName(""), chunks({256,256,256}), blocksize(0),
clevel(5), cname("lz4"), id("blosc"), shuffle(1), dtype("<u2"),
fill_value(""), filters({}), order("F"), shape({0,0,0}),
zarr_format(2), subfolders({0,0,0})
{
    set_jsonValues();
}

zarr::zarr(const std::string &fileName) :
fileName(fileName), chunks({256,256,256}), blocksize(0),
clevel(5), cname("lz4"), id("blosc"), shuffle(1), dtype("<u2"),
fill_value(""), filters({}), order("F"), shape({0,0,0}),
zarr_format(2), subfolders({0,0,0})
{
    if(!fileExists(fileName+"/.zarray")){
        throw std::string("metadataFileMissing:"+fileName);
        //mexErrMsgIdAndTxt("zarr:zarrayError","Metadata file in \"%s\" is missing. Does the file exist?",fileName.c_str());
    }
    std::ifstream f(fileName+"/.zarray");
    zarray = json::parse(f);

    try{
        chunks = zarray.at("chunks").get<std::vector<uint64_t>>();
        try{
            // Try blosc compression types
            cname = zarray.at("compressor").at("cname");
            clevel = zarray.at("compressor").at("clevel");
            blocksize = zarray.at("compressor").at("blocksize");
            id = zarray.at("compressor").at("id");
            shuffle = zarray.at("compressor").at("shuffle");
        }
        catch(...){
            // Try gzip
            clevel = zarray.at("compressor").at("level");
            cname = zarray.at("compressor").at("id");
            blocksize = 0;
            id = "";
            shuffle = 0;
        }
        dtype = zarray.at("dtype");
        //fill_value = "";
        //filters = "";
        order = zarray.at("order");
        shape = zarray.at("shape").get<std::vector<uint64_t>>();
        zarr_format = zarray.at("zarr_format");
    }
    catch(...){
        throw std::string("metadataIncomplete");
        //mexErrMsgIdAndTxt("zarr:zarrayError","Metadata is incomplete. Check the .zarray file");
    }
    try{
        subfolders = zarray.at("subfolders").get<std::vector<uint64_t>>();
    }
    catch(...){
        subfolders = {0,0,0};
    }
}

// Create a new zarr file with .zarray metadata file (no actual data)
zarr::zarr(const std::string &fileName, const std::vector<uint64_t> &chunks,
           uint64_t blocksize, uint64_t clevel, const std::string &cname,
           const std::string &id, uint64_t shuffle, const std::string &dtype,
           const std::string &fill_value, const std::vector<std::string> &filters,
           const std::string &order, const std::vector<uint64_t> &shape,
           uint64_t zarr_format, const std::vector<uint64_t> subfolders) :
fileName(fileName), chunks(chunks), blocksize(blocksize),
clevel(clevel), cname(cname), id(id), shuffle(shuffle), dtype(dtype),
fill_value(fill_value), filters(filters), order(order), shape(shape),
zarr_format(zarr_format), subfolders(subfolders)
{
    // Handle the tilde character in filenames on Linux/Mac
    #ifndef _WIN32
    this->fileName = expandTilde(this->fileName.c_str());
    #endif
    set_jsonValues();
}

zarr::~zarr(){

}

// Write the current Metadata to the .zarray and create subfolders if needed
void zarr::write_zarray(){
    createSubfolders();
    set_jsonValues();
    write_jsonValues();
}

const std::string &zarr::get_fileName() const{
    return fileName;
}

void zarr::set_fileName(const std::string &fileName){
    // Handle the tilde character in filenames on Linux/Mac
    this->fileName = fileName;
    #ifndef _WIN32
    this->fileName = expandTilde(this->fileName.c_str());
    #endif
}

const uint64_t &zarr::get_chunks(const uint64_t &index) const{
    return chunks[index];
}

void zarr::set_chunks(const std::vector<uint64_t> &chunks){
    this->chunks = chunks;
}

const uint64_t &zarr::get_clevel() const{
    return clevel;
}

void zarr::set_clevel(const uint64_t &clevel){
    this->clevel = clevel;
}

const std::string &zarr::get_cname() const{
    return cname;
}

void zarr::set_cname(const std::string &cname){
    this->cname = cname;
}

const std::string &zarr::get_dtype() const{
    return dtype;
}

void zarr::set_dtype(const std::string &dtype){
    this->dtype = dtype;
}

const std::string &zarr::get_order() const{
    return order;
}

void zarr::set_order(const std::string &order){
    this->order = order;
}

const uint64_t &zarr::get_shape(const uint64_t &index) const{
    return shape[index];
}

void zarr::set_shape(const std::vector<uint64_t> &shape){
    this->shape = shape;
}

// Set the values of the JSON file to the current member values
void zarr::set_jsonValues(){
    zarray.clear();
    zarray["chunks"] = chunks;

    if(cname == "lz4" || cname == "blosclz" || cname == "lz4hc" || cname == "zlib" || cname == "zstd"){
        zarray["compressor"]["blocksize"] = blocksize;
        zarray["compressor"]["clevel"] = clevel;
        zarray["compressor"]["cname"] = cname;
        zarray["compressor"]["id"] = id;
        zarray["compressor"]["shuffle"] = shuffle;
    }
    else if(cname == "gzip"){
        zarray["compressor"]["id"] = id;
        zarray["compressor"]["level"] = clevel;
    }
    else throw std::string("unsupportedCompressor"); 
        //mexErrMsgIdAndTxt("zarr:zarrayError","Compressor: \"%s\" is not currently supported\n",cname.c_str());

    // fill_value just 0 and filters null for now
    zarray["dtype"] = dtype;
    zarray["fill_value"] = 0;
    zarray["filters"] = nullptr;
    zarray["order"] = order;

    // zarr_format just 2 for now
    zarray["shape"] = shape;
    zarray["zarr_format"] = 2;

    zarray["subfolders"] = subfolders;
}

// Write the current JSON to disk
void zarr::write_jsonValues(){
    // If the .zarray file does not exist then build the zarr fileName path recursively
    const std::string fileNameFinal(fileName+"/.zarray");

    //if(!std::filesystem::exists(fileNameFinal)){
    if(!fileExists(fileNameFinal)){
        mkdirRecursive(fileName.c_str());
        chmod(fileName.c_str(), 0775);
    }

    const std::string uuid = generateUUID();
    const std::string fnFull(fileName+"/.zarray"+uuid);

    std::ofstream o(fnFull);
    if(!o.good()) throw std::string("cannotOpenZarray:"+fnFull);
        //mexErrMsgIdAndTxt("zarr:zarrayError","Cannot open %s\n",fnFull.c_str());
    o << std::setw(4) << zarray << std::endl;
    o.close();

    rename(fnFull.c_str(),fileNameFinal.c_str());
}

const std::string zarr::get_subfoldersString(const std::vector<uint64_t> &cAV) const{
    if(subfolders[0] == 0 && subfolders[1] == 0 && subfolders[2] == 0) return "";

    std::vector<uint64_t> currVals = {0,0,0};
    if(subfolders[0] > 0) currVals[0] = cAV[0]/subfolders[0];
    if(subfolders[1] > 0) currVals[1] = cAV[1]/subfolders[1];
    if(subfolders[2] > 0) currVals[2] = cAV[2]/subfolders[2];

    return std::string(std::to_string(currVals[0])+"_"+
                       std::to_string(currVals[1])+"_"+
                       std::to_string(currVals[2]));
}

void zarr::set_subfolders(const std::vector<uint64_t> &subfolders){
    this->subfolders = subfolders;
    zarray["subfolders"] = subfolders;
}

// Fast ceiling for the subfolder function
uint64_t zarr::fastCeilDiv(uint64_t num, uint64_t denom){
    return 1 + ((num - 1) / denom);
}

// Create subfolder "chunks"
void zarr::createSubfolders(){
    // If all elements are zero then we don't make subfolders
    if(std::all_of(subfolders.begin(),
                   subfolders.end(),
                   [](int i){return !i;}))
    {
        return;
    }

    std::vector<uint64_t> nChunks = {fastCeilDiv(shape[0],chunks[0]),
                                     fastCeilDiv(shape[1],chunks[1]),
                                     fastCeilDiv(shape[2],chunks[2])};
    std::vector<uint64_t> nSubfolders = {1,1,1};
    for(uint64_t i = 0; i < nSubfolders.size(); i++){
        if(subfolders[i] > 0){
            nSubfolders[i] = fastCeilDiv(nChunks[i],subfolders[i]);
        }
    }

    // Create subfolders
    #pragma omp parallel for collapse(3)
    for(uint64_t x = 0; x < nSubfolders[0]; x++){
        for(uint64_t y = 0; y < nSubfolders[1]; y++){
            for(uint64_t z = 0; z < nSubfolders[2]; z++){
                std::string currName(fileName+"/"+std::to_string(x)+"_"+
                                     std::to_string(y)+"_"+std::to_string(z));
                mkdirRecursive(currName.c_str());
            }
        }
    }
}

const std::vector<uint64_t> zarr::get_chunkAxisVals(const std::string &fileName) const{
    std::vector<uint64_t> cAV(3);
    char* ptr;
    cAV[0] = strtol(fileName.c_str(), &ptr, 10);
    ptr++;
    cAV[1] = strtol(ptr, &ptr, 10);
    ptr++;
    cAV[2] = strtol(ptr, &ptr, 10);
    return cAV;
}

void zarr::set_chunkInfo(const std::vector<uint64_t> &startCoords,
                         const std::vector<uint64_t> &endCoords)
{

    uint64_t xStartAligned = startCoords[0]-(startCoords[0]%chunks[0]);
    uint64_t yStartAligned = startCoords[1]-(startCoords[1]%chunks[1]);
    uint64_t zStartAligned = startCoords[2]-(startCoords[2]%chunks[2]);
    uint64_t xStartChunk = (xStartAligned/chunks[0]);
    uint64_t yStartChunk = (yStartAligned/chunks[1]);
    uint64_t zStartChunk = (zStartAligned/chunks[2]);

    uint64_t xEndAligned = endCoords[0];
    uint64_t yEndAligned = endCoords[1];
    uint64_t zEndAligned = endCoords[2];

    if(xEndAligned%chunks[0]) xEndAligned = endCoords[0]-(endCoords[0]%chunks[0])+chunks[0];
    if(yEndAligned%chunks[1]) yEndAligned = endCoords[1]-(endCoords[1]%chunks[1])+chunks[1];
    if(zEndAligned%chunks[2]) zEndAligned = endCoords[2]-(endCoords[2]%chunks[2])+chunks[2];
    uint64_t xEndChunk = (xEndAligned/chunks[0]);
    uint64_t yEndChunk = (yEndAligned/chunks[1]);
    uint64_t zEndChunk = (zEndAligned/chunks[2]);

    uint64_t xChunks = (xEndChunk-xStartChunk);
    uint64_t yChunks = (yEndChunk-yStartChunk);
    uint64_t zChunks = (zEndChunk-zStartChunk);
    numChunks = xChunks*yChunks*zChunks;
    chunkNames = std::vector<std::string>(numChunks);
    #pragma omp parallel for collapse(3)
    for(uint64_t x = xStartChunk; x < xEndChunk; x++){
        for(uint64_t y = yStartChunk; y < yEndChunk; y++){
            for(uint64_t z = zStartChunk; z < zEndChunk; z++){
                uint64_t currFile = (z-zStartChunk)+((y-yStartChunk)*zChunks)+((x-xStartChunk)*yChunks*zChunks);
                chunkNames[currFile] = std::to_string(x)+"."+std::to_string(y)+"."+std::to_string(z);
            }
        }
    }
}

const std::string &zarr::get_chunkNames(const uint64_t &index) const{
    return chunkNames[index];
}

const uint64_t &zarr::get_numChunks() const{
    return numChunks;
}

const std::string &zarr::get_errString() const{
    return errString;
}

void zarr::set_errString(const std::string &errString){
    this->errString = errString;
}