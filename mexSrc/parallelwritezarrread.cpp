#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <omp.h>
#include <stdlib.h>
#include "parallelwritezarrread.h"
#include "blosc2.h"
#include "mex.h"
#include "zarr.h"
#include "helperfunctions.h"
#include "zlib.h"

void parallelReadZarrMex(zarr &Zarr, void* zarrArr,
                         const std::vector<uint64_t> &startCoords, 
                         const std::vector<uint64_t> &endCoords,
                         const std::vector<uint64_t> &readShape, uint64_t bits)
{
    uint64_t bytes = (bits/8);
    
    int32_t numWorkers = omp_get_max_threads();
    
    Zarr.set_chunkInfo(startCoords, endCoords);
    
    const int32_t batchSize = (Zarr.get_numChunks()-1)/numWorkers+1;
    const uint64_t s = Zarr.get_chunks(0)*Zarr.get_chunks(1)*Zarr.get_chunks(2);
    const uint64_t sB = s*bytes;
    int32_t w;
    int err = 0;
    std::string errString;


    #pragma omp parallel for if(numWorkers<=Zarr.get_numChunks())
    for(w = 0; w < numWorkers; w++){
        void* bufferDest = mallocDynamic(s,bits);
        uint64_t lastFileLen = 0;
        char *buffer = NULL;
        for(int64_t f = w*batchSize; f < (w+1)*batchSize; f++){
            if(f>=Zarr.get_numChunks() || err) break;
            std::vector<uint64_t> cAV = Zarr.get_chunkAxisVals(Zarr.get_chunkNames(f));
            const std::string subfolderName = Zarr.get_subfoldersString(cAV);

            //malloc +2 for null term and filesep
            const std::string fileName(Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.get_chunkNames(f));
            
            // If we cannot open the file then set to all zeros
            // Can make this better by checking the errno
            FILE *fileptr = fopen(fileName.c_str(), "rb");
            if(!fileptr){
                memset(bufferDest,0,sB);
            }
            else{
                fseek(fileptr, 0, SEEK_END);
                long filelen = ftell(fileptr);
                rewind(fileptr);
                if(lastFileLen < filelen){
                    free(buffer);
                    buffer = (char*) malloc(filelen);
                    lastFileLen = filelen;
                }
                fread(buffer, filelen, 1, fileptr);
                fclose(fileptr);
    
                
                // Decompress
                int64_t dsize = -1;
                int uncErr = 0;
                if(Zarr.get_cname() != "gzip"){
                    blosc2_context *dctx;
                    blosc2_dparams dparams = {(int16_t)numWorkers,NULL,NULL,NULL};
                    dctx = blosc2_create_dctx(dparams);
                    dsize = blosc2_decompress_ctx(dctx,buffer, filelen, bufferDest, sB);
                    blosc2_free_ctx(dctx);
                }
                else{
                    dsize = sB;
                    z_stream stream;
                    stream.zalloc = Z_NULL;
                    stream.zfree = Z_NULL;
                    stream.opaque = Z_NULL;
                    stream.avail_in = (uInt)filelen;
                    stream.avail_out = (uInt)dsize;
                    while(stream.avail_in > 0){
    
                        dsize = sB;
    
                        stream.next_in = (uint8_t*)buffer+(filelen-stream.avail_in);
                        stream.next_out = (uint8_t*)bufferDest+(sB-stream.avail_out);
    
                        uncErr = inflateInit2(&stream, 32);
                        if(uncErr){
                        #pragma omp critical
                        {
                        err = 1;
                        errString = "Decompression error. Error code: "+
                                     std::to_string(uncErr)+" ChunkName: "+
                                     Zarr.get_fileName()+"/"+Zarr.get_chunkNames(f)+"\n";
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
                                     Zarr.get_fileName()+"/"+Zarr.get_chunkNames(f)+"\n";
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
                                     Zarr.get_fileName()+"/"+Zarr.get_chunkNames(f)+"\n";
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
                                     Zarr.get_fileName()+"/"+Zarr.get_chunkNames(f)+"\n";
                    }
                    break;
                }
            }
            
            if(Zarr.get_order() == "F"){
                for(int64_t z = cAV[2]*Zarr.get_chunks(2); z < (cAV[2]+1)*Zarr.get_chunks(2); z++){
                    if(z>=endCoords[2]) break;
                    else if(z<startCoords[2]) continue;
                    for(int64_t y = cAV[1]*Zarr.get_chunks(1); y < (cAV[1]+1)*Zarr.get_chunks(1); y++){
                        if(y>=endCoords[1]) break;
                        else if(y<startCoords[1]) continue;
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
            else if (Zarr.get_order() == "C"){
                for(int64_t x = cAV[0]*Zarr.get_chunks(0); x < (cAV[0]+1)*Zarr.get_chunks(0); x++){
                    if(x>=endCoords[0]) break;
                    else if(x<startCoords[0]) continue;
                    for(int64_t y = cAV[1]*Zarr.get_chunks(1); y < (cAV[1]+1)*Zarr.get_chunks(1); y++){
                        if(y>=endCoords[1]) break;
                        else if(y<startCoords[1]) continue;
                        for(int64_t z = cAV[2]*Zarr.get_chunks(2); z < (cAV[2]+1)*Zarr.get_chunks(2); z++){
                            if(z>=endCoords[2]) break;
                            else if(z<startCoords[2]) continue;
                            switch(bytes){
                                case 1:
                                    ((uint8_t*)zarrArr)[((x-startCoords[0])+((y-startCoords[1])*readShape[0])+((z-startCoords[2])*readShape[0]*readShape[1]))] = ((uint8_t*)bufferDest)[((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))];
                                    break;
                                case 2:
                                    ((uint16_t*)zarrArr)[((x-startCoords[0])+((y-startCoords[1])*readShape[0])+((z-startCoords[2])*readShape[0]*readShape[1]))] = ((uint16_t*)bufferDest)[((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))];
                                    break;
                                case 4:
                                    ((float*)zarrArr)[((x-startCoords[0])+((y-startCoords[1])*readShape[0])+((z-startCoords[2])*readShape[0]*readShape[1]))] = ((float*)bufferDest)[((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))];
                                    break;
                                case 8:
                                    ((double*)zarrArr)[((x-startCoords[0])+((y-startCoords[1])*readShape[0])+((z-startCoords[2])*readShape[0]*readShape[1]))] = ((double*)bufferDest)[((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))];
                                    break;
                            }
                            
                        }
                    }
                }
                
            }
            
        }
        free(bufferDest);
        free(buffer);
    }
    
    if(err) mexErrMsgIdAndTxt("zarr:threadError",errString.c_str());
}

// TODO: FIX MEMORY LEAKS
void* parallelReadZarrWrapper(zarr &Zarr, const bool &crop,
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
    
    if(Zarr.get_dtype() == "<u1"){
        uint64_t bits = 8;
        uint8_t* zarrArr = (uint8_t*)malloc(sizeof(uint8_t)*readShape[0]*readShape[1]*readShape[2]);
        parallelReadZarrMex(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits);
        return (void*)zarrArr;
    }
    else if(Zarr.get_dtype() == "<u2"){
        uint64_t bits = 16;
        uint16_t* zarrArr = (uint16_t*)malloc((uint64_t)(sizeof(uint16_t)*readShape[0]*readShape[1]*readShape[2]));
        parallelReadZarrMex(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits);
        return (void*)zarrArr;
    }
    else if(Zarr.get_dtype() == "<f4"){
        uint64_t bits = 32;
        float* zarrArr = (float*)malloc(sizeof(float)*readShape[0]*readShape[1]*readShape[2]);
        parallelReadZarrMex(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits);
        return (void*)zarrArr;
    }
    else if(Zarr.get_dtype() == "<f8"){
        uint64_t bits = 64;
        double* zarrArr = (double*)malloc(sizeof(double)*readShape[0]*readShape[1]*readShape[2]);
        parallelReadZarrMex(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits);
        return (void*)zarrArr;
    }
    else{
        return NULL;
    }
}
