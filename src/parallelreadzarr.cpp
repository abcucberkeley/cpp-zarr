#include <fstream>
#include <stdint.h>
#include <omp.h>
#include "parallelreadzarr.h"
#include "blosc2.h"
#include "blosc.h"
#include "zarr.h"
#include "helperfunctions.h"
#include "zlib.h"
//#include <iostream>

// zarrArr should be initialized to all zeros if you have empty chunks
uint8_t parallelReadZarr(zarr &Zarr, void* zarrArr,
                         const std::vector<uint64_t> &startCoords, 
                         const std::vector<uint64_t> &endCoords,
                         const std::vector<uint64_t> &readShape,
                         const uint64_t bits,
                         const bool useCtx,
                         const bool sparse)
{
    void* zarrArrC = NULL;
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
    
    const int32_t batchSize = (Zarr.get_numChunks()-1)/numWorkers+1;
    const uint64_t s = Zarr.get_chunks(0)*Zarr.get_chunks(1)*Zarr.get_chunks(2);
    const uint64_t sB = s*bytes;

    // If C->F order then we need a temp C order array
    if(Zarr.get_order() == "C") zarrArrC = calloc(readShape[0]*readShape[1]*readShape[2],bytes);
    
    void* zeroChunkUnc = NULL;
    if(sparse){
        zeroChunkUnc = calloc(s,bytes);
    }

    int err = 0;
    std::string errString;

    #pragma omp parallel for
    for(int32_t w = 0; w < numWorkers; w++){
        //void* bufferDest = malloc(sB);
        //void* buffer = malloc(sB);
        void* bufferDest = operator new(sB);
        std::streamsize lastFileLen = 0;
        void* buffer = NULL;
        int64_t dsize = -1;
        int uncErr = 0;
        for(int64_t f = w*batchSize; f < (w+1)*batchSize; f++){
            if(f>=Zarr.get_numChunks() || err) break;
            const std::vector<uint64_t> cAV = Zarr.get_chunkAxisVals(Zarr.get_chunkNames(f));
            const std::string subfolderName = Zarr.get_subfoldersString(cAV);

            const std::string fileName(Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.get_chunkNames(f));
            
            // If we cannot open the file then set to all zeros
            // Can make this better by checking the errno
            std::ifstream file(fileName, std::ios::binary);
            if(!file.is_open()){
                continue;
                //memset(bufferDest,0,sB);
            }
            else{
                file.seekg(0, std::ios::end);
                const std::streamsize fileLen = file.tellg();
                if(lastFileLen < fileLen){
                    operator delete(buffer);
                    buffer = operator new(fileLen);
                    lastFileLen = fileLen;
                }
                file.seekg(0, std::ios::beg);
                file.read(reinterpret_cast<char*>(buffer), fileLen);
                file.close();
                
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
            // C->C (x and z are flipped) then we flip to F below
            else if (Zarr.get_order() == "C"){

                for(int64_t y = cAV[1]*Zarr.get_chunks(1); y < (cAV[1]+1)*Zarr.get_chunks(1); y++){
                    if(y>=endCoords[1]) break;
                    else if(y<startCoords[1]) continue;
                    for(int64_t z = cAV[0]*Zarr.get_chunks(0); z < (cAV[0]+1)*Zarr.get_chunks(0); z++){
                        if(z>=endCoords[0]) break;
                        else if(z<startCoords[0]) continue;
                        if(((cAV[2]*Zarr.get_chunks(2)) < startCoords[2] && ((cAV[2]+1)*Zarr.get_chunks(2)) > startCoords[2]) || (cAV[2]+1)*Zarr.get_chunks(2)>endCoords[2]){
                            if(((cAV[2]*Zarr.get_chunks(2)) < startCoords[2] && ((cAV[2]+1)*Zarr.get_chunks(2)) > startCoords[2]) && (cAV[2]+1)*Zarr.get_chunks(2)>endCoords[2]){
                                memcpy((uint8_t*)zarrArrC+((((cAV[2]*Zarr.get_chunks(2))-startCoords[2]+(startCoords[2]%Zarr.get_chunks(2)))+((y-startCoords[1])*readShape[2])+((z-startCoords[0])*readShape[2]*readShape[1]))*bytes),(uint8_t*)bufferDest+(((startCoords[2]%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((z%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes),((endCoords[2]%Zarr.get_chunks(2))-(startCoords[2]%Zarr.get_chunks(2)))*bytes);
                            }
                            else if((cAV[2]+1)*Zarr.get_chunks(2)>endCoords[2]){
                                memcpy((uint8_t*)zarrArrC+((((cAV[2]*Zarr.get_chunks(2))-startCoords[2])+((y-startCoords[1])*readShape[2])+((z-startCoords[0])*readShape[2]*readShape[1]))*bytes),(uint8_t*)bufferDest+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((z%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes),(endCoords[2]%Zarr.get_chunks(2))*bytes);
                            }
                            else if((cAV[2]*Zarr.get_chunks(2)) < startCoords[2] && ((cAV[2]+1)*Zarr.get_chunks(2)) > startCoords[2]){
                                memcpy((uint8_t*)zarrArrC+((((cAV[2]*Zarr.get_chunks(2)-startCoords[2]+(startCoords[2]%Zarr.get_chunks(2))))+((y-startCoords[1])*readShape[2])+((z-startCoords[0])*readShape[2]*readShape[1]))*bytes),(uint8_t*)bufferDest+(((startCoords[2]%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((z%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes),(Zarr.get_chunks(2)-(startCoords[2]%Zarr.get_chunks(2)))*bytes);
                            }
                        }
                        else{
                            memcpy((uint8_t*)zarrArrC+((((cAV[2]*Zarr.get_chunks(2))-startCoords[2])+((y-startCoords[1])*readShape[2])+((z-startCoords[0])*readShape[2]*readShape[1]))*bytes),(uint8_t*)bufferDest+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((z%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes),Zarr.get_chunks(2)*bytes);
                        }
                    }
                }  
                
            }
            
        }
        operator delete(bufferDest);
        operator delete(buffer);
        //free(bufferDest);
        //free(buffer);
    }
    if(!useCtx){
        blosc2_destroy();
    }
    free(zeroChunkUnc);

    if(err){
        Zarr.set_errString(errString);
        return 1;
    }
    else if (Zarr.get_order() == "C"){
        #pragma omp parallel for
        for(size_t j = 0; j < readShape[1]; j++) {
            for(size_t i = 0; i < readShape[0]; i++) {
                for(size_t k = 0; k < readShape[2]; k++) {
                    switch(bytes){
                        case 1:
                            *(((uint8_t*)zarrArr)+(k*readShape[0]*readShape[1])+(j*readShape[0])+i) = *((uint8_t*)zarrArrC+(i*readShape[1]*readShape[2])+(j*readShape[2])+k);
                            break;
                        case 2:
                            *(((uint16_t*)zarrArr)+(k*readShape[0]*readShape[1])+(j*readShape[0])+i) = *((uint16_t*)zarrArrC+(i*readShape[1]*readShape[2])+(j*readShape[2])+k);
                            break;
                        case 4:
                            *(((float*)zarrArr)+(k*readShape[0]*readShape[1])+(j*readShape[0])+i) = *((float*)zarrArrC+(i*readShape[1]*readShape[2])+(j*readShape[2])+k);
                            break;
                        case 8:
                            *(((double*)zarrArr)+(k*readShape[0]*readShape[1])+(j*readShape[0])+i) = *((double*)zarrArrC+(i*readShape[1]*readShape[2])+(j*readShape[2])+k);
                            break;
                    }
                }
            }
        }
        
        
        free(zarrArrC);

        
    }
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
    else{
        startCoords[0]--;
        startCoords[1]--;
        startCoords[2]--;
    }

    
    std::vector<uint64_t> readShape = {endCoords[0]-startCoords[0],
                                       endCoords[1]-startCoords[1],
                                       endCoords[2]-startCoords[2]};

    Zarr.set_chunkInfo(startCoords, endCoords);
    uint8_t err = 0;
    uint64_t readSize = readShape[0]*readShape[1]*readShape[2];
    if(Zarr.get_dtype() == "<u1"){
        uint64_t bits = 8;
        uint8_t* zarrArr;
        if(Zarr.get_fill_value()){
            zarrArr = (uint8_t*)malloc(readSize*sizeof(uint8_t));
            memset(zarrArr,Zarr.get_fill_value(),readSize*sizeof(uint8_t));
        }
        else zarrArr = (uint8_t*)calloc(readSize,sizeof(uint8_t));
        err = parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits,true);
        if(err){
            free(zarrArr);
            return NULL;
        }
        else return (void*)zarrArr;
    }
    else if(Zarr.get_dtype() == "<u2"){
        uint64_t bits = 16;
        uint16_t* zarrArr;
        if(Zarr.get_fill_value()){
            zarrArr = (uint16_t*)malloc(readSize*(uint64_t)(sizeof(uint16_t)));
            memset(zarrArr,Zarr.get_fill_value(),readSize*sizeof(uint16_t));
        }
        else zarrArr = (uint16_t*)calloc(readSize,(uint64_t)(sizeof(uint16_t)));
        err = parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits,true);
        if(err){
            free(zarrArr);
            return NULL;
        }
        else return (void*)zarrArr;
    }
    else if(Zarr.get_dtype() == "<f4"){
        uint64_t bits = 32;
        float* zarrArr;
        if(Zarr.get_fill_value()){
            zarrArr = (float*)malloc(readSize*(sizeof(float)));
            memset(zarrArr,Zarr.get_fill_value(),readSize*sizeof(float));
        }
        else zarrArr = (float*)calloc(readSize,(sizeof(float)));
        err = parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits,true);
        if(err){
            free(zarrArr);
            return NULL;
        }
        else return (void*)zarrArr;
    }
    else if(Zarr.get_dtype() == "<f8"){
        uint64_t bits = 64;
        double* zarrArr;
        if(Zarr.get_fill_value()){
            zarrArr = (double*)malloc(readSize*(sizeof(double)));
            memset(zarrArr,Zarr.get_fill_value(),readSize*sizeof(double));
        }
        else zarrArr = (double*)calloc(readSize,(sizeof(double)));
        err = parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits,true);
        if(err){
            free(zarrArr);
            return NULL;
        }
        else return (void*)zarrArr;
    }
    else{
        return NULL;
    }
}
