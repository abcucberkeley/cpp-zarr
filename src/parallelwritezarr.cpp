#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <dirent.h>
#include <string.h>
#include <omp.h>
#include <stddef.h>
#ifdef _WIN32
#include <sys/time.h>
#else
#include <uuid/uuid.h>
#endif
#include <sys/stat.h>
#include <fstream>
#include <algorithm>
#include "blosc.h"
#include "mex.h"
#include "parallelreadzarr.h"
#include "parallelwritezarr.h"
#include "helperfunctions.h"
#include "zarr.h"
#include "zlib.h"

void parallelWriteZarr(zarr &Zarr, void* zarrArr,
                          const std::vector<uint64_t> &startCoords,
                          const std::vector<uint64_t> &endCoords,
                          const std::vector<uint64_t> &writeShape,
                          const uint64_t bits, const bool useUuid, const bool crop){
    //printf("%s startCoords[0]yz: %d %d %d endCoords[0]yz: %d %d %d chunkxyz: %d %d %d writeShape[0]yz: %d %d %d bits: %d\n",Zarr.get_fileName().c_str(),startCoords[0],startCoords[1],startCoords[2],endCoords[0],endCoords[1],endCoords[2],Zarr.get_chunks(0),Zarr.get_chunks(1),Zarr.get_chunks(2),writeShape[0],writeShape[1],writeShape[2],bits);
    const uint64_t bytes = (bits/8);

    int32_t numWorkers = omp_get_max_threads();

    int32_t nBloscThreads = 1;
    if(numWorkers>Zarr.get_numChunks()){
        nBloscThreads = std::ceil(((double)numWorkers)/((double)Zarr.get_numChunks()));
        numWorkers = Zarr.get_numChunks();
    }

    const int32_t batchSize = (Zarr.get_numChunks()-1)/numWorkers+1;
    const uint64_t s = Zarr.get_chunks(0)*Zarr.get_chunks(1)*Zarr.get_chunks(2);
    const uint64_t sB = s*bytes;

    const std::string uuid(generateUUID());

    int err = 0;
    std::string errString;

    #pragma omp parallel for
    for(int32_t w = 0; w < numWorkers; w++){
        void* chunkUnC = malloc(sB);
        void* chunkC = malloc(sB+BLOSC_MAX_OVERHEAD);
        for(int64_t f = w*batchSize; f < (w+1)*batchSize; f++){
            if(f>=Zarr.get_numChunks()  || err) break;
            const std::vector<uint64_t> cAV = Zarr.get_chunkAxisVals(Zarr.get_chunkNames(f));
            void* cRegion = nullptr;

            if(crop && ((((cAV[0])*Zarr.get_chunks(0)) < startCoords[0] || ((cAV[0]+1)*Zarr.get_chunks(0) > endCoords[0] && endCoords[0] < Zarr.get_shape(0)))
                        || (((cAV[1])*Zarr.get_chunks(1)) < startCoords[1] || ((cAV[1]+1)*Zarr.get_chunks(1) > endCoords[1] && endCoords[1] < Zarr.get_shape(1)))
                        || (((cAV[2])*Zarr.get_chunks(2)) < startCoords[2] || ((cAV[2]+1)*Zarr.get_chunks(2) > endCoords[2] && endCoords[2] < Zarr.get_shape(2))))){
                cRegion = parallelReadZarrWrapper(Zarr, crop,
                                                  {((cAV[0])*Zarr.get_chunks(0))+1,
                                                   ((cAV[1])*Zarr.get_chunks(1))+1,
                                                   ((cAV[2])*Zarr.get_chunks(2))+1},
                                                  {(cAV[0]+1)*Zarr.get_chunks(0),
                                                   (cAV[1]+1)*Zarr.get_chunks(1),
                                                   (cAV[2]+1)*Zarr.get_chunks(2)});
            }
            if(Zarr.get_order() == "F"){
                for(int64_t z = cAV[2]*Zarr.get_chunks(2); z < (cAV[2]+1)*Zarr.get_chunks(2); z++){
                    if(z>=endCoords[2]){
                        if(crop){
                            if((cAV[2]+1)*Zarr.get_chunks(2) > Zarr.get_shape(2)){
                                memcpy((uint8_t*)chunkUnC+((((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)cRegion+((((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),((Zarr.get_shape(2)-z)*Zarr.get_chunks(0)*Zarr.get_chunks(1))*bytes);
                                uint64_t zRest = ((cAV[2]+1)*Zarr.get_chunks(2))-Zarr.get_shape(2);
                                memset((uint8_t*)chunkUnC+(((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1))*bytes),0,(zRest*(Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes);
                            }
                            else{
                                memcpy((uint8_t*)chunkUnC+((((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)cRegion+((((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),((((cAV[2]+1)*Zarr.get_chunks(2))-z)*Zarr.get_chunks(0)*Zarr.get_chunks(1))*bytes);
                            }
                        }
                        else{
                            uint64_t zRest = ((cAV[2]+1)*Zarr.get_chunks(2))-z;
                            memset((uint8_t*)chunkUnC+(((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1))*bytes),0,(zRest*(Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes);
                        }
                        break;
                    }
                    else if(z<startCoords[2]){
                        if(crop){
                            memcpy((uint8_t*)chunkUnC+(((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1))*bytes),(uint8_t*)cRegion+(((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1))*bytes),((startCoords[2]-z)*Zarr.get_chunks(0)*Zarr.get_chunks(1))*bytes);
                        }
                        else{
                            memset((uint8_t*)chunkUnC+(((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1))*bytes),0,((startCoords[2]-z)*(Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes);
                        }
                        z = startCoords[2]-1;
                        continue;
                    }
                    for(int64_t y = cAV[1]*Zarr.get_chunks(1); y < (cAV[1]+1)*Zarr.get_chunks(1); y++){
                        if(y>=endCoords[1]){
                            if(crop){
                                if((cAV[1]+1)*Zarr.get_chunks(1) > Zarr.get_shape(1)){
                                    memcpy((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)cRegion+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),((Zarr.get_shape(1)-y)*Zarr.get_chunks(0))*bytes);
                                    uint64_t yRest = ((cAV[1]+1)*Zarr.get_chunks(1))-Zarr.get_shape(1);
                                    memset((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),0,(yRest*(Zarr.get_chunks(0)))*bytes);
                                }
                                else{
                                    memcpy((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)cRegion+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),((((cAV[1]+1)*Zarr.get_chunks(1))-y)*Zarr.get_chunks(0))*bytes);
                                }
                            }
                            else{
                                uint64_t yRest = ((cAV[1]+1)*Zarr.get_chunks(1))-y;
                                memset((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),0,(yRest*Zarr.get_chunks(0))*bytes);
                            }
                            break;
                        }
                        else if(y<startCoords[1]){
                            if(crop){
                                memcpy((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)cRegion+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),((startCoords[1]-y)*Zarr.get_chunks(0))*bytes);
                            }
                            else{
                                memset((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),0,(startCoords[1]-y)*bytes);
                            }
                            y = startCoords[1]-1;
                            continue;
                        }

                        if(((cAV[0]*Zarr.get_chunks(0)) < startCoords[0] && ((cAV[0]+1)*Zarr.get_chunks(0)) > startCoords[0]) || (cAV[0]+1)*Zarr.get_chunks(0)>endCoords[0]){
                            if(((cAV[0]*Zarr.get_chunks(0)) < startCoords[0] && ((cAV[0]+1)*Zarr.get_chunks(0)) > startCoords[0]) && (cAV[0]+1)*Zarr.get_chunks(0)>endCoords[0]){
                                if(crop){
                                    memcpy((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)cRegion+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(startCoords[0]%Zarr.get_chunks(0))*bytes);
                                    memcpy((uint8_t*)chunkUnC+(((startCoords[0]%Zarr.get_chunks(0))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)zarrArr+((((cAV[0]*Zarr.get_chunks(0))-startCoords[0]+(startCoords[0]%Zarr.get_chunks(0)))+((y-startCoords[1])*writeShape[0])+((z-startCoords[2])*writeShape[0]*writeShape[1]))*bytes),((endCoords[0]%Zarr.get_chunks(0))-(startCoords[0]%Zarr.get_chunks(0)))*bytes);
                                    memcpy((uint8_t*)chunkUnC+(((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))+(endCoords[0]%Zarr.get_chunks(0)))*bytes),(uint8_t*)cRegion+(((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))+(endCoords[0]%Zarr.get_chunks(0)))*bytes),(Zarr.get_chunks(0)-(endCoords[0]%Zarr.get_chunks(0)))*bytes);
                                }
                                else{
                                    memset((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),0,(startCoords[0]%Zarr.get_chunks(0))*bytes);
                                    memcpy((uint8_t*)chunkUnC+(((startCoords[0]%Zarr.get_chunks(0))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)zarrArr+((((cAV[0]*Zarr.get_chunks(0))-startCoords[0]+(startCoords[0]%Zarr.get_chunks(0)))+((y-startCoords[1])*writeShape[0])+((z-startCoords[2])*writeShape[0]*writeShape[1]))*bytes),((endCoords[0]%Zarr.get_chunks(0))-(startCoords[0]%Zarr.get_chunks(0)))*bytes);
                                    memset((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))+(endCoords[0]%Zarr.get_chunks(0))*bytes),0,(Zarr.get_chunks(0)-(endCoords[0]%Zarr.get_chunks(0)))*bytes);
                                }
                            }
                            else if((cAV[0]+1)*Zarr.get_chunks(0)>endCoords[0]){
                                if(crop){
                                    memcpy((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)zarrArr+((((cAV[0]*Zarr.get_chunks(0))-startCoords[0])+((y-startCoords[1])*writeShape[0])+((z-startCoords[2])*writeShape[0]*writeShape[1]))*bytes),(endCoords[0]-(cAV[0]*Zarr.get_chunks(0)))*bytes);

                                    if((cAV[0]+1)*Zarr.get_chunks(0) > Zarr.get_shape(0)){
                                        memcpy((uint8_t*)chunkUnC+((((endCoords[0]-(cAV[0]*Zarr.get_chunks(0))))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)cRegion+((((endCoords[0]-(cAV[0]*Zarr.get_chunks(0))))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(Zarr.get_shape(0)-endCoords[0])*bytes);
                                        uint64_t xRest = ((cAV[0]+1)*Zarr.get_chunks(0))-Zarr.get_shape(0);
                                        memset((uint8_t*)chunkUnC+(((Zarr.get_shape(0)-(cAV[0]*Zarr.get_chunks(0)))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),0,(xRest)*bytes);
                                    }
                                    else{
                                        memcpy((uint8_t*)chunkUnC+((((endCoords[0]-(cAV[0]*Zarr.get_chunks(0))))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)cRegion+((((endCoords[0]-(cAV[0]*Zarr.get_chunks(0))))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(((cAV[0]+1)*Zarr.get_chunks(0))-endCoords[0])*bytes);
                                    }
                                }
                                else{
                                    memcpy((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)zarrArr+((((cAV[0]*Zarr.get_chunks(0))-startCoords[0])+((y-startCoords[1])*writeShape[0])+((z-startCoords[2])*writeShape[0]*writeShape[1]))*bytes),(endCoords[0]%Zarr.get_chunks(0))*bytes);
                                    memset((uint8_t*)chunkUnC+(((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))+(endCoords[0]%Zarr.get_chunks(0)))*bytes),0,(Zarr.get_chunks(0)-(endCoords[0]%Zarr.get_chunks(0)))*bytes);
                                }
                            }
                            else if((cAV[0]*Zarr.get_chunks(0)) < startCoords[0] && ((cAV[0]+1)*Zarr.get_chunks(0)) > startCoords[0]){
                                if(crop){
                                    memcpy((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)cRegion+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(startCoords[0]%Zarr.get_chunks(0))*bytes);
                                    memcpy((uint8_t*)chunkUnC+(((startCoords[0]%Zarr.get_chunks(0))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)zarrArr+((((cAV[0]*Zarr.get_chunks(0))-startCoords[0]+(startCoords[0]%Zarr.get_chunks(0)))+((y-startCoords[1])*writeShape[0])+((z-startCoords[2])*writeShape[0]*writeShape[1]))*bytes),(Zarr.get_chunks(0)-(startCoords[0]%Zarr.get_chunks(0)))*bytes);
                                }
                                else{
                                    memset((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),0,(startCoords[0]%Zarr.get_chunks(0))*bytes);
                                    memcpy((uint8_t*)chunkUnC+(((startCoords[0]%Zarr.get_chunks(0))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)zarrArr+((((cAV[0]*Zarr.get_chunks(0))-startCoords[0]+(startCoords[0]%Zarr.get_chunks(0)))+((y-startCoords[1])*writeShape[0])+((z-startCoords[2])*writeShape[0]*writeShape[1]))*bytes),(Zarr.get_chunks(0)-(startCoords[0]%Zarr.get_chunks(0)))*bytes);
                                }
                            }
                        }
                        else{
                            memcpy((uint8_t*)chunkUnC+((((y%Zarr.get_chunks(1))*Zarr.get_chunks(0))+((z%Zarr.get_chunks(2))*Zarr.get_chunks(0)*Zarr.get_chunks(1)))*bytes),(uint8_t*)zarrArr+((((cAV[0]*Zarr.get_chunks(0))-startCoords[0])+((y-startCoords[1])*writeShape[0])+((z-startCoords[2])*writeShape[0]*writeShape[1]))*bytes),Zarr.get_chunks(0)*bytes);
                        }
                    }
                }
            }
            else if (Zarr.get_order() == "C"){
                for(int64_t x = cAV[0]*Zarr.get_chunks(2); x < (cAV[0]+1)*Zarr.get_chunks(0); x++){
                    for(int64_t y = cAV[1]*Zarr.get_chunks(1); y < (cAV[1]+1)*Zarr.get_chunks(1); y++){
                        for(int64_t z = cAV[2]*Zarr.get_chunks(2); z < (cAV[2]+1)*Zarr.get_chunks(2); z++){
                            switch(bytes){
                                case 1:
                                    if(x>=endCoords[0] || x<startCoords[0] || y>= endCoords[1] || y<startCoords[1] || z>=endCoords[2] || z<startCoords[2]){
                                        ((uint8_t*)chunkUnC)[(((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes)] = 0;
                                        continue;
                                    }
                                        ((uint8_t*)chunkUnC)[(((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes)] = ((uint8_t*)zarrArr)[((x+(y*writeShape[0])+(z*writeShape[0]*writeShape[1]))*bytes)];
                                        break;
                                case 2:
                                    if(x>=endCoords[0] || x<startCoords[0] || y>= endCoords[1] || y<startCoords[1] || z>=endCoords[2] || z<startCoords[2]){
                                        ((uint16_t*)chunkUnC)[(((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes)] = 0;
                                        continue;
                                    }
                                        ((uint16_t*)chunkUnC)[(((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes)] = ((uint16_t*)zarrArr)[((x+(y*writeShape[0])+(z*writeShape[0]*writeShape[1]))*bytes)];
                                        break;
                                case 4:
                                    if(x>=endCoords[0] || x<startCoords[0] || y>= endCoords[1] || y<startCoords[1] || z>=endCoords[2] || z<startCoords[2]){
                                        ((float*)chunkUnC)[(((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes)] = 0;
                                        continue;
                                    }
                                        ((float*)chunkUnC)[(((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes)] = ((float*)zarrArr)[((x+(y*writeShape[0])+(z*writeShape[0]*writeShape[1]))*bytes)];
                                        break;
                                case 8:
                                    if(x>=endCoords[0] || x<startCoords[0] || y>= endCoords[1] || y<startCoords[1] || z>=endCoords[2] || z<startCoords[2]){
                                        ((double*)chunkUnC)[(((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes)] = 0;
                                        continue;
                                    }
                                        ((double*)chunkUnC)[(((z%Zarr.get_chunks(2))+((y%Zarr.get_chunks(1))*Zarr.get_chunks(2))+((x%Zarr.get_chunks(0))*Zarr.get_chunks(2)*Zarr.get_chunks(1)))*bytes)] = ((double*)zarrArr)[((x+(y*writeShape[0])+(z*writeShape[0]*writeShape[1]))*bytes)];
                                        break;
                            }

                        }
                    }
                }
            }
            //char* compressor = blosc_get_compressor();
            //printf("Thread: %d Compressor: %s\n",w,compressor);

            // Use the same blosc compress as Zarr
            //int64_t csize = blosc_compress(5, BLOSC_SHUFFLE, bytes, sB, chunkUnC, chunkC, sB+BLOSC_MAX_OVERHEAD);
            const std::string subfolderName = Zarr.get_subfoldersString(cAV);
            int64_t csize = 0;
            if(Zarr.get_cname() != "gzip"){
                /*
                if(numWorkers<=Zarr.get_numChunks()){
                    csize = blosc_compress_ctx(Zarr.get_clevel(), BLOSC_SHUFFLE, bytes, sB, chunkUnC, chunkC, sB+BLOSC_MAX_OVERHEAD,Zarr.get_cname().c_str(),0,1);
                }
                else{
                    csize = blosc_compress_ctx(Zarr.get_clevel(), BLOSC_SHUFFLE, bytes, sB, chunkUnC, chunkC, sB+BLOSC_MAX_OVERHEAD,Zarr.get_cname().c_str(),0,numWorkers);
                }*/
                csize = blosc_compress_ctx(Zarr.get_clevel(), BLOSC_SHUFFLE, bytes, sB, chunkUnC, chunkC, sB+BLOSC_MAX_OVERHEAD,Zarr.get_cname().c_str(),0,nBloscThreads);
            }
            else{
                //uint64_t sLength = sB;
                csize = sB+BLOSC_MAX_OVERHEAD;
                //compress2((Bytef*)chunkC,&csize,(Bytef*)chunkUnC,sB,1);
                z_stream stream;
                //memset(&stream, 0, sizeof(stream));
                stream.zalloc = Z_NULL;
                stream.zfree = Z_NULL;
                stream.opaque = Z_NULL;

                //char dummy = '\0';  // zlib does not like NULL output buffers (even if the uncompressed data is empty)
                stream.next_in = (uint8_t*)chunkUnC;
                stream.next_out = (uint8_t*)chunkC;

                stream.avail_in = sB;
                stream.avail_out = csize;
                int cErr = deflateInit2(&stream, Zarr.get_clevel(), Z_DEFLATED, MAX_WBITS + 16, MAX_MEM_LEVEL, Z_DEFAULT_STRATEGY);
                if(cErr){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Compression error. Error code: "+
                            std::to_string(cErr)+" ChunkName: "+
                            Zarr.get_fileName()+"/"+subfolderName+"/"+
                            Zarr.get_chunkNames(f)+"\n";
                    }
                    break;
                }

                cErr = deflate(&stream, Z_FINISH);

                if(cErr != Z_STREAM_END){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Compression error. Error code: "+
                            std::to_string(cErr)+" ChunkName: "+
                            Zarr.get_fileName()+"/"+subfolderName+"/"+
                            Zarr.get_chunkNames(f)+"\n";                }
                    break;
                }

                if(deflateEnd(&stream)){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Compression error. Error code: "+
                            std::to_string(cErr)+" ChunkName: "+
                            Zarr.get_fileName()+"/"+subfolderName+"/"+
                            Zarr.get_chunkNames(f)+"\n";
                    }
                    break;
                }
                csize = csize - stream.avail_out;
            }
            
            std::string fileName(Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.get_chunkNames(f));

            if(useUuid){
                const std::string fileNameFinal(fileName);
                fileName.append(uuid);
                std::ofstream file(fileName, std::ios::binary | std::ios::trunc);

                if(!file.is_open()){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Check permissions or filepath. Cannot write to path: "+
                            fileName+"\n";
                    }
                    break;
                }
                file.write(reinterpret_cast<char*>(chunkC),csize);
                file.close();
                rename(fileName.c_str(),fileNameFinal.c_str());
            }
            else{
                std::ofstream file(fileName, std::ios::binary | std::ios::trunc);
                if(!file.is_open()){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Check permissions or filepath. Cannot write to path: "+
                            Zarr.get_fileName()+"\n";
                    }
                    break;
                }
                file.write(reinterpret_cast<char*>(chunkC),csize);
                file.close();
            }
            free(cRegion);
        }
        free(chunkUnC);
        free(chunkC);

    }

    if(err) mexErrMsgIdAndTxt("zarr:threadError",errString.c_str());
}