#include <algorithm>
#include <fstream>
#include <cstdint>
#include <omp.h>
#include "parallelreadzarr.h"
#include "blosc2.h"
#include "zarr.h"
#include "helperfunctions.h"
#include "zlib.h"

// zarrArr should be initialized to all zeros if you have empty chunks
uint8_t parallelReadZarr(zarr &Zarr, void* zarrArr,
                         const std::vector<uint64_t> &startCoords, 
                         const std::vector<uint64_t> &endCoords,
                         const std::vector<uint64_t> &readShape,
                         const uint64_t bits,
                         const bool useCtx,
                         const bool sparse)
{
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

    // The chunk size is actually the inner chunk size if the zarr file is sharded
    if(Zarr.get_shard()){
        Zarr.set_chunks({Zarr.get_chunk_shape(0),Zarr.get_chunk_shape(1),Zarr.get_chunk_shape(2)});
    }
    
    const int32_t batchSize = (Zarr.get_numChunks()-1)/numWorkers+1;
    const uint64_t s = Zarr.get_chunks(0)*Zarr.get_chunks(1)*Zarr.get_chunks(2);
    const uint64_t sB = s*bytes;

    void* zeroChunkUnc = NULL;
    if(sparse){
        zeroChunkUnc = calloc(s,bytes);
    }

    int err = 0;
    std::string errString;

    #pragma omp parallel for
    for(int32_t w = 0; w < numWorkers; w++){
        void* bufferDest = operator new(sB);
        void* buffer = NULL;
        std::streamsize lastFileLen = 0;
        int64_t dsize = -1;
        int uncErr = 0;
        for(int64_t f = w*batchSize; f < (w+1)*batchSize; f++){
            if(f>=Zarr.get_numChunks() || err) break;
            const std::vector<uint64_t> cAV = Zarr.get_chunkAxisVals(Zarr.get_chunkNames(f));
            const std::string subfolderName = Zarr.get_subfoldersString(cAV);

            std::string fileName;
            
            if(!Zarr.get_shard()){
                fileName = (Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.get_chunkNames(f));
            }
            else{
                // Can change this to the check for zeros maybe
                bool pad = cAV[0] > endCoords[0]/Zarr.get_chunk_shape(0) ||
                    cAV[1] > endCoords[1]/Zarr.get_chunk_shape(1) ||
                    cAV[2] > endCoords[2]/Zarr.get_chunk_shape(2) ||
                    cAV[0] < startCoords[0]/Zarr.get_chunk_shape(0) ||
                    cAV[1] < startCoords[1]/Zarr.get_chunk_shape(1) ||
                    cAV[2] < startCoords[2]/Zarr.get_chunk_shape(2);
                if(pad) {
                    continue;
                }
                fileName = Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.chunkNameToShardName(Zarr.get_chunkNames(f));
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
                    uint64_t currChunkShardPosition = Zarr.get_chunkShardPosition(cAV);
                    uint64_t offsetNBytes[2];
                    
                    file.seekg(-(int64_t)(((Zarr.get_numChunksPerShard()*2*sizeof(uint64_t))+4)-(currChunkShardPosition*2*sizeof(uint64_t))), std::ios::end);
                    file.read(reinterpret_cast<char*>(offsetNBytes), sizeof(offsetNBytes));
                    
                    // All zeros or skippable chunk
                    if(offsetNBytes[0]== std::numeric_limits<uint64_t>::max() &&
                       offsetNBytes[1] == std::numeric_limits<uint64_t>::max()){
                        file.close();
                        continue;
                    }
                    fileLen = offsetNBytes[1];
                    if(lastFileLen < fileLen){
                        operator delete(buffer);
                        buffer = operator new(fileLen);
                        lastFileLen = fileLen;
                    }
                    file.seekg(offsetNBytes[0], std::ios::beg);
                    file.read(reinterpret_cast<char*>(buffer), fileLen);

                    file.close();
                }
                
                // Decompress
                if(Zarr.get_cname() != "gzip"){
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
            if(sparse){
                // If the chunk is all zeros (memcmp == 0) then we skip it
                const bool allZeros = memcmp(zeroChunkUnc,bufferDest,sB);
                if(!allZeros) continue;
            }
            
            // F->F
            if(Zarr.get_order() == "F"){  
                for(int64_t y = cAV[1]*Zarr.get_chunks(1); y < (cAV[1]+1)*Zarr.get_chunks(1); y++){
                    if(y>=endCoords[1]) break;
                    else if(y<startCoords[1]) continue;
                    for(int64_t z = cAV[2]*Zarr.get_chunks(2); z < (cAV[2]+1)*Zarr.get_chunks(2); z++){
                        if(z>=endCoords[2]) break;
                        else if(z<startCoords[2]) continue;
                        if(((cAV[0]*Zarr.get_chunks(0)) < startCoords[0] && ((cAV[0]+1)*Zarr.get_chunks(0)) > startCoords[0]) || (cAV[0]+1)*Zarr.get_chunks(0)>endCoords[0]){
                            if(((cAV[0]*Zarr.get_chunks(0)) < startCoords[0] && ((cAV[0]+1)*Zarr.get_chunks(0)) > startCoords[0]) && (cAV[0]+1)*Zarr.get_chunks(0)>endCoords[0]){
                                memcpy((uint8_t*)zarrArr+((((cAV[0]*Zarr.get_chunks(0))-startCoords[0]+(startCoords[0]%Zarr.get_chunks(0)))+((y-startCoords[1])*readShape[0])+((z-startCoords[2])*readShape[0]*readShape[1]))*bytes),(uint8_t*)bufferDest+(((startCoords[0]%Zarr.get_chunks(0))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),((endCoords[0]%Zarr.get_chunks(0))-(startCoords[0]%Zarr.get_chunks(0)))*bytes);
                            }
                            else if((cAV[0]+1)*Zarr.get_chunks(0)>endCoords[0]){
                                memcpy((uint8_t*)zarrArr+((((cAV[0]*Zarr.get_chunks(0))-startCoords[0])+((y-startCoords[1])*readShape[0])+((z-startCoords[2])*readShape[0]*readShape[1]))*bytes),(uint8_t*)bufferDest+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(endCoords[0]%Zarr.get_chunks(0))*bytes);
                            }
                            else if((cAV[0]*Zarr.get_chunks(0)) < startCoords[0] && ((cAV[0]+1)*Zarr.get_chunks(0)) > startCoords[0]){
                                memcpy((uint8_t*)zarrArr+((((cAV[0]*Zarr.get_chunks(0)-startCoords[0]+(startCoords[0]%Zarr.get_chunks(0))))+((y-startCoords[1])*readShape[0])+((z-startCoords[2])*readShape[0]*readShape[1]))*bytes),(uint8_t*)bufferDest+(((startCoords[0]%Zarr.get_chunks(0))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(Zarr.get_chunks(0)-(startCoords[0]%Zarr.get_chunks(0)))*bytes);
                            }
                        }
                        else{
                            memcpy((uint8_t*)zarrArr+((((cAV[0]*Zarr.get_chunks(0))-startCoords[0])+((y-startCoords[1])*readShape[0])+((z-startCoords[2])*readShape[0]*readShape[1]))*bytes),(uint8_t*)bufferDest+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),Zarr.get_chunks(0)*bytes);
                        }
                    }
                }
                
            }
            // C->F: transpose the chunk (z fastest) straight into the F-order
            // output (x fastest). Only the part of the chunk inside the read
            // region is copied, one cache-sized tile at a time.
            else if (Zarr.get_order() == "C"){
                const uint64_t C0 = Zarr.get_chunks(0), C1 = Zarr.get_chunks(1), C2 = Zarr.get_chunks(2);
                const uint64_t b0 = cAV[0]*C0, b1 = cAV[1]*C1, b2 = cAV[2]*C2;
                const uint64_t x0 = std::max(b0, startCoords[0]), x1 = std::min(b0+C0, endCoords[0]);
                const uint64_t y0 = std::max(b1, startCoords[1]), y1 = std::min(b1+C1, endCoords[1]);
                const uint64_t z0 = std::max(b2, startCoords[2]), z1 = std::min(b2+C2, endCoords[2]);
                if(x0 < x1 && y0 < y1 && z0 < z1){
                    // i = x (contiguous in the output), j = y, k = z (contiguous in the chunk)
                    copyBoxTransposed(bytes,
                        (const uint8_t*)bufferDest+(((x0-b0)*C1*C2)+((y0-b1)*C2)+(z0-b2))*bytes,
                        (uint8_t*)zarrArr+((x0-startCoords[0])+((y0-startCoords[1])*readShape[0])+((z0-startCoords[2])*readShape[0]*readShape[1]))*bytes,
                        x1-x0, y1-y0, z1-z0,
                        C1*C2, C2, 1,
                        1, readShape[0], readShape[0]*readShape[1]);
                }
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
    if(oppositeEndianness(Zarr.get_dtype())) swapArrayEndianness(zarrArr,bytes,readShape[0]*readShape[1]*readShape[2]);
    return 0;
}

// TODO: FIX MEMORY LEAKS
// Wrapper used by parallelWriteZarr
void* parallelReadZarrWriteWrapper(zarr Zarr, const bool &crop,
                              std::vector<uint64_t> startCoords, 
                              std::vector<uint64_t> endCoords){
   
    if(!crop){
        startCoords[0] = 0;
        startCoords[1] = 0;
        startCoords[2] = 0;
        endCoords[0] = Zarr.get_shape(0);
        endCoords[1] = Zarr.get_shape(1);
        endCoords[2] = Zarr.get_shape(2);
    }

    
    std::vector<uint64_t> readShape = {endCoords[0]-startCoords[0],
                                       endCoords[1]-startCoords[1],
                                       endCoords[2]-startCoords[2]};

    Zarr.set_chunkInfo(startCoords, endCoords);
    uint64_t readSize = readShape[0]*readShape[1]*readShape[2];

    // The chunk read/copy machinery is element-width based, so one generic path
    // covers every supported dtype: signed/unsigned 8/16/32/64-bit integers and
    // 32/64-bit floats.
    const std::string &dtype = Zarr.get_dtype();
    const uint64_t bytes = Zarr.dtypeBytes();
    const char kind = dtype.size() == 3 ? dtype[1] : '\0';
    if(!bytes || (kind != 'u' && kind != 'i' && kind != 'f') ||
       (kind == 'f' && bytes < 4)){
        return NULL;
    }

    void* zarrArr = nullptr;
    const int fillValue = fillValueToInt(Zarr.get_fill_value());
    if(fillValue){
        zarrArr = malloc(readSize*bytes);
        memset(zarrArr,fillValue,readSize*bytes);
    }
    else zarrArr = calloc(readSize,bytes);
    uint8_t err = parallelReadZarr(Zarr, zarrArr,startCoords,endCoords,readShape,bytes*8,true);
    if(err){
        free(zarrArr);
        return NULL;
    }
    return zarrArr;
}

void* readZarrParallelHelper(const char* folderName, uint64_t startX, uint64_t startY, uint64_t startZ, uint64_t endX, uint64_t endY, uint64_t endZ, uint8_t imageJIm){
    zarr Zarr(folderName);
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

