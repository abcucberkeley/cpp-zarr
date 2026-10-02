#include <algorithm>
#include <fstream>
#include <cstdint>
#include <omp.h>
#include "parallelreadzarr.h"
#include "blosc2.h"
#include "zarr.h"
#include "helperfunctions.h"
#include "zlib.h"
#include "zstd.h"

// zarrArr should be initialized to all zeros if you have empty chunks
uint8_t parallelReadZarr(zarr &Zarr, void* zarrArr,
                         const std::vector<uint64_t> &startCoords, 
                         const std::vector<uint64_t> &endCoords,
                         const std::vector<uint64_t> &readShape,
                         const uint64_t bits,
                         const bool useCtx,
                         const bool sparse)
{
    return parallelReadZarr(Zarr, zarrArr, startCoords, endCoords, readShape, bits, useCtx, sparse, false);
}

uint8_t parallelReadZarr(zarr &Zarr, void* zarrArr,
                         const std::vector<uint64_t> &startCoords,
                         const std::vector<uint64_t> &endCoords,
                         const std::vector<uint64_t> &readShape,
                         const uint64_t bits,
                         const bool useCtx,
                         const bool sparse,
                         const bool cOrder)
{
    // A 0-dimensional array is a single element: sharding does not apply
    if(Zarr.get_ndims() == 0 && Zarr.get_shard()){
        Zarr.set_errString("Sharding is not supported for 0-dimensional arrays\n");
        return 1;
    }
    const uint64_t bytes = (bits/8);
    
    int32_t numWorkers = omp_get_max_threads();

    // nBloscThreads used when using blosc_ctx
    uint32_t nBloscThreads = 1;
    if(!useCtx){
        blosc2_init();
        blosc2_set_nthreads(numWorkers);
    }

    // no ctx
    /*
    if(numWorkers>Zarr.get_numChunks()){
        blosc_set_nthreads(std::ceil(((double)numWorkers)/((double)Zarr.get_numChunks())));
        numWorkers = Zarr.get_numChunks();
    }
    else {
        blosc_set_nthreads(numWorkers);
    }
    */
    
    // ctx
    /*
    if(numWorkers>Zarr.get_numChunks()){
        nBloscThreads = std::ceil(((double)numWorkers)/((double)Zarr.get_numChunks()));
        numWorkers = Zarr.get_numChunks();
    }
    */

    // A sharded array is read with its inner chunk shape (the shard shape is
    // restored on return). Each shard's index is at the start or, by default, the
    // end of its file, optionally followed by a CRC32C of it.
    const uint64_t nDims = Zarr.get_ndims();
    std::vector<uint64_t> shardChunks;
    if(Zarr.get_shard()){
        std::vector<uint64_t> innerChunks(nDims);
        for(uint64_t d = 0; d < nDims; d++){
            shardChunks.push_back(Zarr.get_chunks(d));
            innerChunks[d] = Zarr.get_chunk_shape(d);
        }
        Zarr.set_chunks(innerChunks);
    }
    struct RestoreChunks {
        zarr &z;
        const std::vector<uint64_t> chunks;
        ~RestoreChunks(){ if(!chunks.empty()) z.set_chunks(chunks); }
    } restoreChunks{Zarr, shardChunks};
    const uint64_t nInner = Zarr.get_shard() ? Zarr.get_numChunksPerShard() : 0;
    const bool indexAtStart = Zarr.get_shardIndexAtStart();
    const bool indexChecksum = Zarr.get_shardIndexChecksum();
    const uint64_t emptyEntry = std::numeric_limits<uint64_t>::max();
    // How each stored chunk is encoded: a compressor (v2's, or a Zarr v3 codec),
    // and a CRC32C checksum after it (Zarr v3 crc32c codec)
    const std::string compressor = Zarr.get_compressor();
    const bool chunkChecksum = Zarr.get_chunkChecksum();
    if(compressor != "none" && compressor != "blosc" && compressor != "gzip" && compressor != "zstd"){
        Zarr.set_errString("The \""+compressor+"\" compressor is not supported\n");
        return 1;
    }
    
    const int32_t batchSize = (Zarr.get_numChunks()-1)/numWorkers+1;
    uint64_t s = 1;
    for(uint64_t d = 0; d < nDims; d++) s *= Zarr.get_chunks(d);
    const uint64_t sB = s*bytes;

    // Element strides of a decompressed chunk (its stored axes, slowest first: C or
    // F order, or a Zarr v3 transpose) and of the output (F or C order)
    std::vector<uint64_t> chunkStrides(nDims), outStrides(nDims);
    {
        const std::vector<uint64_t> axes = Zarr.get_chunkAxisOrder();
        uint64_t acc = 1;
        for(int64_t i = (int64_t)nDims-1; i >= 0; i--){ chunkStrides[axes[i]] = acc; acc *= Zarr.get_chunks(axes[i]); }
        acc = 1;
        if(cOrder){
            for(int64_t d = (int64_t)nDims-1; d >= 0; d--){ outStrides[d] = acc; acc *= readShape[d]; }
        }
        else{
            for(uint64_t d = 0; d < nDims; d++){ outStrides[d] = acc; acc *= readShape[d]; }
        }
    }

    // Sparse reads skip all-zero chunks, which the output already holds when the
    // fill value is zero
    void* zeroChunkUnc = NULL;
    if(sparse && fillValueIsZero(Zarr.get_fill_value())){
        zeroChunkUnc = calloc(s,bytes);
    }

    int err = 0;
    std::string errString;

    #pragma omp parallel for
    for(int32_t w = 0; w < numWorkers; w++){
        void* bufferDest = nullptr;
        void* buffer = NULL;
        // An exception cannot leave the parallel loop: report it as an error
        try{
        bufferDest = operator new(sB);
        // The index of the shard this worker read last
        std::vector<uint64_t> shardIndex(2*nInner);
        std::string indexShard;
        std::streamsize lastFileLen = 0;
        int64_t dsize = -1;
        int uncErr = 0;
        std::vector<uint64_t> boxExt(nDims);
        for(int64_t f = w*batchSize; f < (w+1)*batchSize; f++){
            if(f>=Zarr.get_numChunks() || err) break;
            const std::vector<uint64_t> cAV = Zarr.get_chunkAxisVals(Zarr.get_chunkNames(f));
            // (a shard's subfolder comes from the shard's own indices)
            const std::string subfolderName = Zarr.get_subfoldersString(Zarr.get_shard() ? Zarr.chunkToShard(cAV) : cAV);

            std::string fileName;
            
            if(!Zarr.get_shard()){
                fileName = (Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.chunkKey(Zarr.get_chunkNames(f)));
            }
            else{
                // Can change this to the check for zeros maybe
                bool pad = false;
                for(uint64_t d = 0; d < nDims; d++){
                    pad = pad || cAV[d] > endCoords[d]/Zarr.get_chunk_shape(d) ||
                                 cAV[d] < startCoords[d]/Zarr.get_chunk_shape(d);
                }
                if(pad) {
                    continue;
                }
                fileName = Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.chunkKey(Zarr.chunkNameToShardName(Zarr.get_chunkNames(f)));
            }
            // If we cannot open the file then set to all zeros
            // Can make this better by checking the errno
            std::ifstream file(fileName, std::ios::binary);
            if(!file.is_open()){
                continue;
                //memset(bufferDest,0,sB);
            }
            else{
                std::streamsize fileLen; 
                if(!Zarr.get_shard()){
                    file.seekg(0, std::ios::end);
                    fileLen = file.tellg();
                    if(lastFileLen < fileLen){
                        operator delete(buffer);
                        buffer = operator new(fileLen);
                        lastFileLen = fileLen;
                    }
                    file.seekg(0, std::ios::beg);
                    file.read(reinterpret_cast<char*>(buffer), fileLen);
                    file.close();
                }
                // Sharding
                else{
                    // Load the shard's index once per shard, checking its checksum and
                    // that every entry lies inside the file
                    if(fileName != indexShard){
                        indexShard = fileName;
                        file.seekg(0, std::ios::end);
                        const uint64_t shardLen = (uint64_t)file.tellg();
                        const uint64_t entryBytes = 2*nInner*sizeof(uint64_t);
                        const uint64_t indexBytes = entryBytes+(indexChecksum ? 4 : 0);
                        std::string problem;
                        if(shardLen < indexBytes) problem = "is smaller than its index";
                        else{
                            file.seekg(indexAtStart ? 0 : (std::streamoff)(shardLen-indexBytes), std::ios::beg);
                            file.read(reinterpret_cast<char*>(shardIndex.data()), entryBytes);
                            uint32_t crc = 0;
                            if(indexChecksum) file.read(reinterpret_cast<char*>(&crc), sizeof(crc));
                            if(!file) problem = "could not be read";
                            else if(indexChecksum && crc != crc32c(reinterpret_cast<const uint8_t*>(shardIndex.data()), entryBytes)){
                                problem = "has an index that does not match its checksum";
                            }
                            else{
                                // chunk data lies after the index when it is at the start, else before it
                                const uint64_t lo = indexAtStart ? indexBytes : 0;
                                const uint64_t hi = indexAtStart ? shardLen : shardLen-indexBytes;
                                for(uint64_t i = 0; i < nInner && problem.empty(); i++){
                                    const uint64_t off = shardIndex[2*i], n = shardIndex[(2*i)+1];
                                    if(off == emptyEntry && n == emptyEntry) continue;
                                    if(off < lo || off > hi || n > hi-off) problem = "has an index entry outside the file";
                                }
                            }
                        }
                        if(!problem.empty()){
                            indexShard.clear();
                            #pragma omp critical
                            {
                                err = 1;
                                errString = "The shard "+fileName+" "+problem+
                                            " (it may be damaged, or not match the array's metadata)\n";
                            }
                            break;
                        }
                    }
                    const uint64_t pos = Zarr.get_chunkShardPosition(cAV);
                    const uint64_t offset = shardIndex[2*pos], nbytes = shardIndex[(2*pos)+1];
                    // An inner chunk that was never written reads as the fill value
                    if(offset == emptyEntry && nbytes == emptyEntry){
                        file.close();
                        continue;
                    }
                    fileLen = (std::streamsize)nbytes;
                    if(lastFileLen < fileLen){
                        operator delete(buffer);
                        buffer = operator new(fileLen);
                        lastFileLen = fileLen;
                    }
                    file.clear();
                    file.seekg((std::streamoff)offset, std::ios::beg);
                    file.read(reinterpret_cast<char*>(buffer), fileLen);

                    file.close();
                }
                
                // A crc32c codec's checksum follows the chunk's bytes
                if(chunkChecksum){
                    uint32_t stored = 0;
                    if(fileLen >= 4) memcpy(&stored, (const uint8_t*)buffer+fileLen-4, 4);
                    if(fileLen < 4 || stored != crc32c((const uint8_t*)buffer, (size_t)fileLen-4)){
                        #pragma omp critical
                        {
                        err = 1;
                        errString = "The chunk "+fileName+" does not match its checksum (it may be damaged)\n";
                        }
                        break;
                    }
                    fileLen -= 4;
                }

                // Decompress
                if(compressor == "none"){
                    // Uncompressed chunk: used straight from the file buffer
                    if(fileLen != (std::streamsize)sB){
                        #pragma omp critical
                        {
                        err = 1;
                        errString = "Uncompressed chunk is "+std::to_string(fileLen)+
                                     " bytes instead of "+std::to_string(sB)+". ChunkName: "+
                                     Zarr.get_fileName()+"/"+subfolderName+"/"+
                                     Zarr.get_chunkNames(f)+"\n";
                        }
                        break;
                    }
                    dsize = sB;
                }
                else if(compressor == "zstd"){
                    const size_t zsize = ZSTD_decompress(bufferDest, sB, buffer, (size_t)fileLen);
                    dsize = (!ZSTD_isError(zsize) && zsize == sB) ? (int64_t)sB : -1;
                }
                else if(compressor == "blosc"){
                    if(!useCtx){
                        dsize = blosc2_decompress(buffer, fileLen, bufferDest, sB);
                    }
                    else{
                        blosc2_context *dctx;
                        blosc2_dparams dparams = {(int16_t)nBloscThreads,NULL,NULL,NULL};
                        dctx = blosc2_create_dctx(dparams);
                        dsize = blosc2_decompress_ctx(dctx, buffer, fileLen, bufferDest, sB);
                        blosc2_free_ctx(dctx);
                    }
                }
                else{
                    dsize = sB;
                    z_stream stream;
                    stream.zalloc = Z_NULL;
                    stream.zfree = Z_NULL;
                    stream.opaque = Z_NULL;
                    stream.avail_in = (uInt)fileLen;
                    stream.avail_out = (uInt)dsize;
                    while(stream.avail_in > 0){
    
                        dsize = sB;
    
                        stream.next_in = (uint8_t*)buffer+(fileLen-stream.avail_in);
                        stream.next_out = (uint8_t*)bufferDest+(sB-stream.avail_out);
    
                        uncErr = inflateInit2(&stream, 32);
                        if(uncErr){
                        #pragma omp critical
                        {
                        err = 1;
                        errString = "Decompression error. Error code: "+
                                     std::to_string(uncErr)+" ChunkName: "+
                                     Zarr.get_fileName()+"/"+subfolderName+"/"+
                                     Zarr.get_chunkNames(f)+"\n";
                        }
                        break;
                        }
        
                        uncErr = inflate(&stream, Z_NO_FLUSH);
        
                        if(uncErr != Z_STREAM_END){
                        #pragma omp critical
                        {
                        err = 1;
                        errString = "Decompression error. Error code: "+
                                     std::to_string(uncErr)+" ChunkName: "+
                                     Zarr.get_fileName()+"/"+subfolderName+"/"+
                                     Zarr.get_chunkNames(f)+"\n";
                        }
                        break;
                        }
                    }
                    if(inflateEnd(&stream)){
                        #pragma omp critical
                        {
                        err = 1;
                        errString = "Decompression error. Error code: "+
                                     std::to_string(uncErr)+" ChunkName: "+
                                     Zarr.get_fileName()+"/"+subfolderName+"/"+
                                     Zarr.get_chunkNames(f)+"\n";
                        }
                        break;
                    }
                }
                
                
                if(dsize < 0){
                    #pragma omp critical
                    {
                    err = 1;
                    errString = "Decompression error. Error code: "+
                                     std::to_string(uncErr)+" ChunkName: "+
                                     Zarr.get_fileName()+"/"+subfolderName+"/"+
                                     Zarr.get_chunkNames(f)+"\n";
                    }
                    break;
                }
            }
            // Uncompressed chunks are used straight from the file buffer
            const void* chunkData = compressor == "none" ? buffer : bufferDest;
            if(zeroChunkUnc){
                // If the chunk is all zeros (memcmp == 0) then we skip it
                const bool allZeros = memcmp(zeroChunkUnc,chunkData,sB);
                if(!allZeros) continue;
            }
            
            // Copy the part of the chunk inside the read region into the F-order
            // output: contiguous runs for F-order chunks, a tiled transpose for
            // C-order chunks (last axis fastest -> first axis fastest)
            uint64_t srcOff = 0, dstOff = 0;
            for(uint64_t d = 0; d < nDims; d++){
                const uint64_t b = cAV[d]*Zarr.get_chunks(d);
                const uint64_t lo = std::max(b, startCoords[d]);
                const uint64_t hi = std::min(b+Zarr.get_chunks(d), endCoords[d]);
                boxExt[d] = hi > lo ? hi-lo : 0;
                srcOff += (lo-b)*chunkStrides[d];
                dstOff += (lo-startCoords[d])*outStrides[d];
            }
            copyBoxND(bytes, (const uint8_t*)chunkData+srcOff*bytes, (uint8_t*)zarrArr+dstOff*bytes,
                      boxExt, chunkStrides, outStrides, true);
        }
        }
        catch(const std::exception &e){
            #pragma omp critical
            {
                err = 1;
                errString = std::string("Read error: ")+e.what()+"\n";
            }
        }
        operator delete(bufferDest);
        operator delete(buffer);
    }
    if(!useCtx){
        blosc2_destroy();
    }
    free(zeroChunkUnc);

    if(err){
        Zarr.set_errString(errString);
        return 1;
    }
    uint64_t readSize = 1;
    for(uint64_t d = 0; d < nDims; d++) readSize *= readShape[d];
    if(oppositeEndianness(Zarr.get_dtype())) swapArrayEndianness(zarrArr,bytes,readSize);
    return 0;
}

// TODO: FIX MEMORY LEAKS
// Wrapper used by parallelWriteZarr
void* parallelReadZarrWriteWrapper(zarr Zarr, const bool &crop,
                              std::vector<uint64_t> startCoords, 
                              std::vector<uint64_t> endCoords){
    return parallelReadZarrWriteWrapper(Zarr, crop, startCoords, endCoords, false);
}

void* parallelReadZarrWriteWrapper(zarr Zarr, const bool &crop,
                              std::vector<uint64_t> startCoords,
                              std::vector<uint64_t> endCoords,
                              const bool cOrder){
   
    const uint64_t nDims = Zarr.get_ndims();
    if(!crop){
        startCoords.assign(nDims, 0);
        endCoords.assign(nDims, 0);
        for(uint64_t d = 0; d < nDims; d++) endCoords[d] = Zarr.get_shape(d);
    }

    std::vector<uint64_t> readShape(nDims);
    uint64_t readSize = 1;
    for(uint64_t d = 0; d < nDims; d++){
        readShape[d] = endCoords[d]-startCoords[d];
        readSize *= readShape[d];
    }

    Zarr.set_chunkInfo(startCoords, endCoords);

    // The chunk read/copy machinery is element-width based, so one generic path
    // covers every supported dtype: signed/unsigned 8/16/32/64-bit integers,
    // 32/64-bit floats and booleans.
    const std::string &dtype = Zarr.get_dtype();
    const uint64_t bytes = Zarr.dtypeBytes();
    const char kind = dtype.size() == 3 ? dtype[1] : '\0';
    if(!bytes || (kind != 'u' && kind != 'i' && kind != 'f' && kind != 'b') ||
       (kind == 'f' && bytes < 4) || (kind == 'b' && bytes != 1)){
        return NULL;
    }

    // The output starts as the fill value, which is what missing chunks read as
    uint8_t fillElem[8];
    fillValueElement(Zarr.get_fill_value(), dtype, fillElem);
    void* zarrArr = nullptr;
    if(std::any_of(fillElem, fillElem+bytes, [](uint8_t b){ return b != 0; })){
        zarrArr = malloc(readSize*bytes);
        if(zarrArr) fillElements(bytes, zarrArr, readSize, fillElem);
    }
    else zarrArr = calloc(readSize,bytes);
    if(!zarrArr) return NULL;
    uint8_t err = parallelReadZarr(Zarr, zarrArr,startCoords,endCoords,readShape,bytes*8,true,false,cOrder);
    if(err){
        free(zarrArr);
        return NULL;
    }
    return zarrArr;
}

void* readZarrParallelHelper(const char* folderName, uint64_t startX, uint64_t startY, uint64_t startZ, uint64_t endX, uint64_t endY, uint64_t endZ, uint8_t imageJIm){
    zarr Zarr(folderName);
    // This helper's interface is 3D
    if(Zarr.get_ndims() != 3) return NULL;
    void* zarrArr = parallelReadZarrWriteWrapper(Zarr, true,
                              {startX, startY, startZ},
                              {endX, endY, endZ});
    // Unsupported dtype or read error: return NULL rather than crashing below
    if(!zarrArr) return NULL;
    // May need to add a check for if the data is f order or c order for ImageJ
    // For the c order images I have tested, we also have to do this flip for now
    if(imageJIm /*&& (order == 'F' || order == 'f')*/){
        // The buffer holds the read window, not the whole array, so allocate and
        // index with the window's dimensions
        const uint64_t rx = endX-startX;
        const uint64_t ry = endY-startY;
        const uint64_t rz = endZ-startZ;
        const uint64_t es = Zarr.dtypeBytes();
        void* zarrArrC = malloc(rx*ry*rz*es);
		#pragma omp parallel for
        for(uint64_t k = 0; k < rz; k++){
            for(uint64_t j = 0; j < ry; j++){
                for(uint64_t i = 0; i < rx; i++){
                    switch(es){
                        case 1:
                            ((uint8_t*)zarrArrC)[j+(i*ry)+(k*ry*rx)] = ((uint8_t*)zarrArr)[i+(j*rx)+(k*rx*ry)];
                            break;
                        case 2:
                            ((uint16_t*)zarrArrC)[j+(i*ry)+(k*ry*rx)] = ((uint16_t*)zarrArr)[i+(j*rx)+(k*rx*ry)];
                            break;
                        case 4:
                            ((float*)zarrArrC)[j+(i*ry)+(k*ry*rx)] = ((float*)zarrArr)[i+(j*rx)+(k*rx*ry)];
                            break;
                        case 8:
                            ((double*)zarrArrC)[j+(i*ry)+(k*ry*rx)] = ((double*)zarrArr)[i+(j*rx)+(k*rx*ry)];
                            break;
                    }
                }
            }
        }
		free(zarrArr);
        return zarrArrC;
    }
    return zarrArr;
}

