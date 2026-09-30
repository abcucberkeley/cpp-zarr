#include <cstdint>
#include <omp.h>
#include <sys/stat.h>
#ifdef _WIN32
#include <stdarg.h>
#include <sys/time.h>
#else
#include <uuid/uuid.h>
#endif
#include <fstream>
#include "zarr.h"
#include "helperfunctions.h"

// Create a blank zarr object with default values
zarr::zarr() :
fileName(""), chunks({256,256,256}), blocksize(0),
clevel(5), cname("lz4"), id("blosc"), shuffle(1), dtype("<u2"),
dimension_separator("."), fill_value("0"), filters({}), order("F"), 
shape({0,0,0}), zarr_format(2), subfolders({0,0,0}), shard(false),
chunk_shape({1,1,1})
{
    set_jsonValues();
}

zarr::zarr(const std::string &fileName) :
fileName(fileName), chunks({256,256,256}), blocksize(0),
clevel(5), cname("lz4"), id("blosc"), shuffle(1), dtype("<u2"),
fill_value("0"), filters({}), order("F"), shape({0,0,0}),
zarr_format(2), subfolders({0,0,0}), shard(false), chunk_shape({1,1,1})
{
    if(!fileExists(fileName+"/.zarray")){
        throw std::string("metadataFileMissing:"+fileName);
        //mexErrMsgIdAndTxt("zarr:zarrayError","Metadata file in \"%s\" is missing. Does the file exist?",fileName.c_str());
    }
    std::ifstream f(fileName+"/.zarray");
    zarray = json::parse(f);

    try{
        // Check for Sharding
        // Sharding should be false by default
        try{
            // Assuming the shard info is at the beginning for now
            if(zarray.at("codecs").at(0).at("name") == "sharding_indexed"){
                chunk_shape = zarray.at("codecs").at(0).at("configuration").at("chunk_shape").get<std::vector<uint64_t>>();
                shard = true;
            }
        }
        catch(...){

        }
        chunks = zarray.at("chunks").get<std::vector<uint64_t>>();
        // A null compressor means the chunks are stored uncompressed (zarr-python
        // always does this for 0-dimensional arrays)
        if(zarray.at("compressor").is_null()){
            cname = "none";
            clevel = 0;
            blocksize = 0;
            id = "";
            shuffle = 0;
        }
        else try{
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
        // If dimension_separator does not exist then assume it is "."
        try{
            dimension_separator = zarray.at("dimension_separator");
            if(dimension_separator != "/" && dimension_separator != "."){
                throw std::string("metadataIncomplete:dimension_separatorIncorrect");
            }
        }
        catch(...){
            dimension_separator = ".";
        }

        dtype = zarray.at("dtype");
        if(zarray.at("fill_value").empty()) fill_value = "0";
        else{
            if(zarray.at("fill_value").type() == json::value_t::number_integer || 
               zarray.at("fill_value").type() == json::value_t::number_unsigned || 
               zarray.at("fill_value").type() == json::value_t::number_float)
            {
                fill_value = std::to_string((int64_t)zarray.at("fill_value"));
            }
            else fill_value = zarray.at("fill_value");
            // TODO: Make NaN actually NaN here and in other functions
            if(fill_value == "null" || fill_value == "NaN") fill_value = "0";
            else if(fill_value == "Infinity") fill_value = std::to_string(std::numeric_limits<int64_t>::max());
            else if(fill_value == "-Infinity") fill_value = std::to_string(std::numeric_limits<int64_t>::min());
        }
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
           const std::string &id, uint64_t shuffle, const std::string &dimension_separator, const std::string &dtype,
           const std::string &fill_value, const std::vector<std::string> &filters,
           const std::string &order, const std::vector<uint64_t> &shape,
           uint64_t zarr_format, const std::vector<uint64_t> &subfolders,
		   const bool shard, const std::vector<uint64_t> &chunk_shape) :
fileName(fileName), chunks(chunks), blocksize(blocksize), clevel(clevel),
cname(cname), id(id), shuffle(shuffle), dimension_separator(dimension_separator),
dtype(dtype), fill_value(fill_value), filters(filters), order(order), shape(shape),
zarr_format(zarr_format), subfolders(subfolders), shard(shard), chunk_shape(chunk_shape)
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
    normalizeDims();
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

const std::string &zarr::get_dimension_separator() const{
    return dimension_separator;
}

void zarr::set_dimension_separator(const std::string &dimension_separator){
    this->dimension_separator = dimension_separator;
}

const std::string &zarr::get_dtype() const{
    return dtype;
}

void zarr::set_dtype(const std::string &dtype){
    this->dtype = dtype;
}

const std::string &zarr::get_fill_value() const{
    return fill_value;
}

void zarr::set_fill_value(const std::string &fill_value){
    this->fill_value = fill_value;
}

void zarr::set_fill_value(const int64_t &fill_value){
    this->fill_value = std::to_string(fill_value);
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

const uint64_t zarr::dtypeBytes() const{
    if(dtype.size() != 3) return 0;
    else if(dtype[2] == '1') return 1;
    else if(dtype[2] == '2') return 2;
    else if(dtype[2] == '4') return 4;
    else if(dtype[2] == '8') return 8;
    else return 0;
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
        zarray["compressor"]["id"] = cname;
        zarray["compressor"]["level"] = clevel;
    }
    // Uncompressed chunks
    else if(cname == "none") zarray["compressor"] = nullptr;
    else throw std::string("unsupportedCompressor"); 
    
    // dimension_separator only if dimension_separator is "/"
    if(dimension_separator == "/") zarray["dimension_separator"] = dimension_separator;

    zarray["dtype"] = dtype;
    if(fill_value == "NaN") zarray["fill_value"] = fill_value;
    else if(fill_value == "null") zarray["fill_value"] = nullptr;
    else if(fill_value == "Infinity") zarray["fill_value"] = std::numeric_limits<int64_t>::max();
    else if(fill_value == "-Infinity") zarray["fill_value"] = std::numeric_limits<int64_t>::min();
    else zarray["fill_value"] = std::stoll(fill_value);
    zarray["filters"] = nullptr;
    zarray["order"] = order;    
    zarray["shape"] = shape;

    // zarr_format just 2 for now
    zarray["zarr_format"] = 2;
    
    // Only add the subfolder parameter if subfolders is not all zeros
    if(!std::all_of(subfolders.begin(),
               subfolders.end(),
               [](int i){return !i;})){
        zarray["subfolders"] = subfolders;
    }
    

    // Sharding
    if(shard){
        // Create the inner JSON objects
        json innerConfig;
        json innerCodec;
        json innerCodecConfig;

        // Set the inner JSON values
        // blosc
        if(cname == "lz4" || cname == "blosclz" || cname == "lz4hc" || cname == "zlib" || cname == "zstd"){
            innerCodec["name"] = id;
            innerCodecConfig["cname"] = cname;
            innerCodecConfig["clevel"] = clevel;
            innerCodecConfig["shuffle"] = "shuffle";
            innerCodecConfig["typesize"] = dtypeBytes();
            innerCodecConfig["blocksize"] = blocksize;
        }
        // gzip
        else if(cname == "gzip"){
            innerCodec["name"] = cname;
            innerCodecConfig["level"] = clevel;
        }
        
        innerCodec["configuration"] = innerCodecConfig;

        innerConfig["chunk_shape"] = chunk_shape;
        innerConfig["codecs"] = {innerCodec};
        
        // codecs is an array of json objects
        zarray["codecs"] = {{{"name", "sharding_indexed"}, 
                            {"configuration", innerConfig}}};
    }
}

// Write the current JSON to disk
void zarr::write_jsonValues(){
    // If the .zarray file does not exist then build the zarr fileName path recursively
    const std::string fileNameFinal(fileName+"/.zarray");

    if(!fileExists(fileNameFinal)){
        mkdirRecursive(fileName.c_str());
    }

    const std::string uuid = generateUUID();
    const std::string fnFull(fileName+"/.zarray"+uuid);

    std::ofstream o(fnFull);
    if(!o.good()) throw std::string("cannotOpenZarray:"+fnFull);
    o << std::setw(4) << zarray << std::endl;
    o.close();

    if(!renameReplace(fnFull, fileNameFinal)){
        remove(fnFull.c_str());
        throw std::string("cannotOpenZarray:"+fileNameFinal);
    }
}

const std::string zarr::get_subfoldersString(const std::vector<uint64_t> &cAV) const{
    if(std::all_of(subfolders.begin(), subfolders.end(), [](uint64_t i){return !i;})) return "";

    std::string name;
    for(uint64_t d = 0; d < cAV.size(); d++){
        const uint64_t v = (d < subfolders.size() && subfolders[d] > 0) ? cAV[d]/subfolders[d] : 0;
        if(d) name += "_";
        name += std::to_string(v);
    }
    return name;
}

void zarr::set_subfolders(const std::vector<uint64_t> &subfolders){
    this->subfolders = subfolders;
    zarray["subfolders"] = subfolders;
}

void zarr::set_shardData(){
    normalizeDims();
    const uint64_t n = shape.size();
    shards = std::vector<uint64_t>(n);
    chunksPerShard = std::vector<uint64_t>(n);
    numShards = 1;
    numChunksPerShard = 1;
    for(uint64_t i = 0; i < n; i++){
        chunksPerShard[i] = fastCeilDiv(chunks[i],chunk_shape[i]);
        shards[i] = ceil((double)shape[i]/(double)chunks[i]);
        numShards *= shards[i];
        numChunksPerShard *= chunksPerShard[i];
    }
}

const bool &zarr::get_shard() const{
    return shard;
}

void zarr::set_shard(const bool shard){
    this->shard = shard;
}

const uint64_t &zarr::get_chunk_shape(const uint64_t &index) const{
    return chunk_shape[index];
}

void zarr::set_chunk_shape(const std::vector<uint64_t> &chunk_shape){
    this->chunk_shape = chunk_shape;
    set_shardData();
}

const uint64_t &zarr::get_numShards() const{
    return numShards;
}

const uint64_t &zarr::get_numChunksPerShard() const{
    return numChunksPerShard;
}

// Fast ceiling for the subfolder function
uint64_t zarr::fastCeilDiv(uint64_t num, uint64_t denom){
    return 1 + ((num - 1) / denom);
}

// Name of the chunk (or subfolder) at flat index i of a grid of `count` indices
// starting at `first`, last axis fastest, joined by `sep`
static std::string gridIndexName(uint64_t i, const std::vector<uint64_t> &first,
                                 const std::vector<uint64_t> &count, const std::string &sep){
    const uint64_t n = count.size();
    // Stack storage for the usual numbers of dimensions, heap beyond that
    uint64_t local[64];
    std::vector<uint64_t> heap;
    uint64_t* idx = local;
    if(n > 64){ heap.resize(n); idx = heap.data(); }
    for(int64_t d = (int64_t)n-1; d >= 0; d--){
        idx[d] = first[d] + i % count[d];
        i /= count[d];
    }
    std::string name;
    for(uint64_t d = 0; d < n; d++){
        if(d) name += sep;
        name += std::to_string(idx[d]);
    }
    return name;
}

// Create subfolder "chunks"
void zarr::createSubfolders(){
    normalizeDims();
    const uint64_t n = shape.size();

    // dimension_separator subfolders
    if(dimension_separator == "/"){
        set_chunkInfo(std::vector<uint64_t>(n,0),shape);
        #pragma omp parallel for
        for(uint64_t i = 0; i < chunkNames.size(); i++){
            makeDimensionFolders(fileName+"/"+chunkNames[i]);
        }
    }

    // If all elements are zero then we don't make subfolders
    if(std::all_of(subfolders.begin(),
                   subfolders.end(),
                   [](uint64_t i){return !i;}))
    {
        return;
    }

    // Use shards instead of chunks when sharding
    if(shard) set_shardData();
    std::vector<uint64_t> nSubfolders(n,1);
    uint64_t total = 1;
    for(uint64_t d = 0; d < n; d++){
        const uint64_t nChunks = shard ? shards[d] : fastCeilDiv(shape[d],chunks[d]);
        if(subfolders[d] > 0) nSubfolders[d] = fastCeilDiv(nChunks,subfolders[d]);
        total *= nSubfolders[d];
    }

    // Create subfolders
    const std::vector<uint64_t> zeros(n,0);
    #pragma omp parallel for
    for(uint64_t i = 0; i < total; i++){
        const std::string currName(fileName+"/"+gridIndexName(i,zeros,nSubfolders,"_"));
        mkdirRecursive(currName.c_str());
    }
}

const std::string zarr::chunkNameToShardName(const std::string &chunkName) const{
    const std::vector<uint64_t> cAV = get_chunkAxisVals(chunkName);
    if(cAV.empty()) return "0";
    std::string name;
    for(uint64_t d = 0; d < cAV.size() && d < chunksPerShard.size(); d++){
        if(d) name += dimension_separator;
        name += std::to_string(cAV[d]/chunksPerShard[d]);
    }
    return name;
}

const std::vector<uint64_t> zarr::chunkToShard(const std::vector<uint64_t> &cAV) const{
    std::vector<uint64_t> s(cAV.size());
    for(uint64_t d = 0; d < cAV.size(); d++){
        uint64_t v = ceil((double)cAV[d]/ceil((double)chunks[d]/(double)chunk_shape[d]));
        if(v) v--;
        s[d] = v;
    }
    return s;
}

const uint64_t zarr::get_ShardPosition(const std::vector<uint64_t> &cAV) const{
    uint64_t pos = 0;
    for(uint64_t d = 0; d < cAV.size(); d++) pos = pos*shards[d] + cAV[d];
    return pos;
}

// Position of a chunk inside its shard, last axis fastest
const uint64_t zarr::get_chunkShardPosition(const std::vector<uint64_t> &cAV) const{
    uint64_t pos = 0;
    for(uint64_t d = 0; d < cAV.size(); d++) pos = pos*chunksPerShard[d] + (cAV[d]%chunksPerShard[d]);
    return pos;
}

// Parse a chunk name ("x.y.z", "a/b/c/d", ...) into its per-axis indices
const std::vector<uint64_t> zarr::get_chunkAxisVals(const std::string &fileName) const{
    std::vector<uint64_t> cAV;
    // A 0-dimensional array's single chunk ("0") has no indices
    if(shape.empty()) return cAV;
    cAV.reserve(shape.size());
    const char* p = fileName.c_str();
    while(true){
        char* end;
        cAV.push_back(strtoull(p, &end, 10));
        if(*end == '\0' || end == p) break;
        p = end+1;
    }
    return cAV;
}

void zarr::set_chunkInfo(const std::vector<uint64_t> &startCoords,
                         const std::vector<uint64_t> &endCoords)
{
    normalizeDims();
    const uint64_t n = shape.size();
    // A 0-dimensional array (scalar) is a single chunk named "0"
    if(n == 0){
        numChunks = 1;
        chunkNames = std::vector<std::string>(1, "0");
        return;
    }

    // Chunks (or shards when sharded) that the region touches along each axis
    std::vector<uint64_t> first(n), count(n);
    uint64_t numOuter = 1;
    for(uint64_t d = 0; d < n; d++){
        first[d] = startCoords[d]/chunks[d];
        const uint64_t last = endCoords[d]/chunks[d] + (endCoords[d]%chunks[d] ? 1 : 0);
        count[d] = last > first[d] ? last-first[d] : 0;
        numOuter *= count[d];
    }

    // Default behavior for when chunks are not sharded: every chunk, last axis fastest
    if(!shard){
        numChunks = numOuter;
        chunkNames = std::vector<std::string>(numChunks);
        #pragma omp parallel for
        for(uint64_t i = 0; i < numChunks; i++){
            chunkNames[i] = gridIndexName(i,first,count,dimension_separator);
        }
    }
    // Sharding: every inner chunk of every shard, shard by shard (both last axis fastest)
    else{
        set_shardData();
        numChunks = numOuter*numChunksPerShard;
        chunkNames = std::vector<std::string>(numChunks);
        #pragma omp parallel for
        for(uint64_t i = 0; i < numChunks; i++){
            uint64_t o = i/numChunksPerShard, c = i%numChunksPerShard;
            uint64_t local[64];
            std::vector<uint64_t> heap;
            uint64_t* idx = local;
            if(n > 64){ heap.resize(n); idx = heap.data(); }
            for(int64_t d = (int64_t)n-1; d >= 0; d--){
                const uint64_t s = first[d] + o%count[d];
                o /= count[d];
                idx[d] = s*chunksPerShard[d] + c%chunksPerShard[d];
                c /= chunksPerShard[d];
            }
            std::string name;
            for(uint64_t d = 0; d < n; d++){
                if(d) name += dimension_separator;
                name += std::to_string(idx[d]);
            }
            chunkNames[i] = name;
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

uint64_t zarr::get_ndims() const{
    return shape.size();
}

void zarr::normalizeDims(){
    const uint64_t n = shape.size();
    if(chunks.size() > n) chunks.resize(n);
    while(chunks.size() < n) chunks.push_back(chunks.size() < 3 ? 256 : 1);
    if(subfolders.size() > n) subfolders.resize(n);
    while(subfolders.size() < n) subfolders.push_back(0);
    if(chunk_shape.size() > n) chunk_shape.resize(n);
    while(chunk_shape.size() < n) chunk_shape.push_back(1);
}
