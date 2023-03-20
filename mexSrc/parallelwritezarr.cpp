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
#include "blosc.h"
#include "mex.h"
#include "helperfunctions.h"
#include "zarr.h"
#include "parallelwritezarrread.h"
#include "zlib.h"

//compile
//mex -v COPTIMFLAGS="-DNDEBUG -O3" CFLAGS='$CFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp -O3' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/lib' -lblosc '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/lib64' -lblosc2 '-L/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/lib64' -lcjson -luuid parallelWriteZarr.c helperFunctions.c parallelReadZarr.c

//With zlib
//mex -v COPTIMFLAGS="-DNDEBUG -O3" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O3 -DNDEBUG" CFLAGS='$CFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp -O3' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/lib' -lblosc '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/lib64' -lblosc2 '-L/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/lib64' -lcjson -luuid -lz parallelWriteZarr.c helperFunctions.c parallelReadZarr.c

//mex -v COPTIMFLAGS="-O3 -fwrapv -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/lib64' -lblosc2 zarrMex.c
//
//Windows
//mex -v COPTIMFLAGS="-O3 -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-IC:\Program Files (x86)\bloscZarr\include' '-LC:\Program Files (x86)\bloscZarr\lib' -lblosc '-IC:\Program Files (x86)\cJSON\include\' '-LC:\Program Files (x86)\cJSON\lib' -lcjson '-IC:\Program Files (x86)\blosc\include' '-LC:\Program Files (x86)\blosc\lib' -lblosc2 parallelWriteZarr.c parallelReadZarr.c helperFunctions.c

void parallelWriteZarrMex(zarr &Zarr, void* zarrArr,
                          const std::vector<uint64_t> &startCoords,
                          const std::vector<uint64_t> &endCoords,
                          const std::vector<uint64_t> &writeShape,
                          const uint64_t bits, const bool useUuid, const bool crop){

    //printf("%s startCoords[0]yz: %d %d %d endCoords[0]yz: %d %d %d chunkxyz: %d %d %d writeShape[0]yz: %d %d %d bits: %d\n",folderName,startCoords[0],startCoords[1],startCoords[2],endCoords[0],endCoords[1],endCoords[2],Zarr.get_chunks(0),Zarr.get_chunks(1),Zarr.get_chunks(2),writeShape[0],writeShape[1],shapeZ,bits);


    const uint64_t bytes = (bits/8);

    const int32_t numWorkers = omp_get_max_threads();

    Zarr.set_chunkInfo(startCoords, endCoords);

    const int32_t batchSize = (Zarr.get_numChunks()-1)/numWorkers+1;
    const uint64_t s = Zarr.get_chunks(0)*Zarr.get_chunks(1)*Zarr.get_chunks(2);
    const uint64_t sB = s*bytes;

    const std::string uuid(generateUUID());

    int err = 0;
    std::string errString;
    #pragma omp parallel for if(numWorkers<=Zarr.get_numChunks())
    for(int32_t w = 0; w < numWorkers; w++){
        void* chunkUnC = mallocDynamic(s,bits);
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
            int64_t csize = 0;
            if(Zarr.get_cname() != "gzip"){
                if(numWorkers<=Zarr.get_numChunks()){
                    csize = blosc_compress_ctx(Zarr.get_clevel(), BLOSC_SHUFFLE, bytes, sB, chunkUnC, chunkC, sB+BLOSC_MAX_OVERHEAD,Zarr.get_cname().c_str(),0,1);
                }
                else{
                    csize = blosc_compress_ctx(Zarr.get_clevel(), BLOSC_SHUFFLE, bytes, sB, chunkUnC, chunkC, sB+BLOSC_MAX_OVERHEAD,Zarr.get_cname().c_str(),0,numWorkers);
                }
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
                            Zarr.get_fileName()+"/"+Zarr.get_chunkNames(f)+"\n";
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
                            Zarr.get_fileName()+"/"+Zarr.get_chunkNames(f)+"\n";                }
                    break;
                }

                if(deflateEnd(&stream)){
                    #pragma omp critical
                    {
                        err = 1;
                        errString = "Compression error. Error code: "+
                            std::to_string(cErr)+" ChunkName: "+
                            Zarr.get_fileName()+"/"+Zarr.get_chunkNames(f)+"\n";                }
                    break;
                }
                csize = csize - stream.avail_out;
            }
            //malloc +2 for null term and filesep

            const std::string subfolderName = Zarr.get_subfoldersString(cAV);
            std::string fileName(Zarr.get_fileName()+"/"+subfolderName+"/"+Zarr.get_chunkNames(f));

            //FILE *fileptr = fopen(fileName, "r+b");

            if(useUuid){
                const std::string fileNameFinal(fileName);
                fileName.append(uuid);
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

    /* After using it, destroy the Blosc environment */
    //blosc_destroy();
}

// TODO: FIX MEMORY LEAKS
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    std::vector<uint64_t> startCoords = {0,0,0};
    std::vector<uint64_t> endCoords = {0,0,0};
    bool crop = false;

    // Dims are 1 by default
    uint64_t iDims[3] = {1,1,1};

    if(nrhs < 3) mexErrMsgIdAndTxt("zarr:inputError","This functions requires at least 3 arguments");
    else if(nrhs == 4 || nrhs == 5 || nrhs == 6){
        if(mxGetN(prhs[3]) == 6){
            crop = true;
            startCoords[0] = (uint64_t)*(mxGetPr(prhs[3]))-1;
            startCoords[1] = (uint64_t)*((mxGetPr(prhs[3])+1))-1;
            startCoords[2] = (uint64_t)*((mxGetPr(prhs[3])+2))-1;
            endCoords[0] = (uint64_t)*((mxGetPr(prhs[3])+3));
            endCoords[1] = (uint64_t)*((mxGetPr(prhs[3])+4));
            endCoords[2] = (uint64_t)*((mxGetPr(prhs[3])+5));


            uint64_t* iDimsT = (uint64_t*)mxGetDimensions(prhs[1]);
            uint64_t niDims = (uint64_t) mxGetNumberOfDimensions(prhs[1]);
            for(uint64_t i = 0; i < niDims; i++) iDims[i] = iDimsT[i];

            if(startCoords[0]+1 < 1 ||
               startCoords[1]+1 < 1 ||
               startCoords[2]+1 < 1) mexErrMsgIdAndTxt("zarr:inputError","Lower bounds must be at least 1");

            if(endCoords[0]-startCoords[0] > iDims[0] ||
               endCoords[1]-startCoords[1] > iDims[1] ||
               endCoords[2]-startCoords[2] > iDims[2]) mexErrMsgIdAndTxt("zarr:inputError","Bounds are invalid for the input data size");
        }
        else if(mxGetN(prhs[3]) != 3) mexErrMsgIdAndTxt("zarr:inputError","Input range is not 6 or 3");
    }
    else if (nrhs > 6) mexErrMsgIdAndTxt("zarr:inputError","Number of input arguments must be 4 or less");
    if(!mxIsChar(prhs[0])) mexErrMsgIdAndTxt("zarr:inputError","The first argument must be a string");
    std::string folderName(mxArrayToString(prhs[0]));
    // Handle the tilde character in filenames on Linux/Mac
    #ifndef _WIN32
    folderName = expandTilde(folderName.c_str());
    #endif

    zarr Zarr(folderName);

    if(nrhs >= 5){
        Zarr.set_cname(mxArrayToString(prhs[4]));
    }
    if(nrhs == 6){
        if(mxGetN(prhs[5]) != 3) mexErrMsgIdAndTxt("zarr:inputError","subfolders must be an array of 3 numbers\n");
        Zarr.set_subfolders({(uint64_t)*(mxGetPr(prhs[5])),
                             (uint64_t)*((mxGetPr(prhs[5])+1)),
                             (uint64_t)*((mxGetPr(prhs[5])+2))});
    }
    bool useUuid = (bool)*(mxGetPr(prhs[2]));

    void* zarrC = NULL;


    mxClassID mDType = mxGetClassID(prhs[1]);
    if(mDType == mxUINT8_CLASS){
        Zarr.set_dtype("<u1");
    }
    else if(mDType == mxUINT16_CLASS){
        Zarr.set_dtype("<u2");
    }
    else if(mDType == mxSINGLE_CLASS){
        Zarr.set_dtype("<f4");
    }
    else if(mDType == mxDOUBLE_CLASS){
        Zarr.set_dtype("<f8");
    }

    if(!crop){
        uint64_t nDims = (uint64_t)mxGetNumberOfDimensions(prhs[1]);
        if(nDims < 2 || nDims > 3) mexErrMsgIdAndTxt("zarr:inputError","Input data must be 2D or 3D");

        uint64_t* dims = (uint64_t*)mxGetDimensions(prhs[1]);
        if(nDims == 3) Zarr.set_shape({dims[0],dims[1],dims[2]});
        else Zarr.set_shape({dims[0],dims[1],1});

        Zarr.set_chunks({(uint64_t)*(mxGetPr(prhs[3])),
                         (uint64_t)*((mxGetPr(prhs[3])+1)),
                         (uint64_t)*((mxGetPr(prhs[3])+2))});
        Zarr.write_zarray();
    }
    else{
        Zarr.set_shape({endCoords[0],endCoords[1],endCoords[2]});

        if(fileExists(folderName+"/.zarray")){
            if(!iDims) mexErrMsgIdAndTxt("zarr:inputError","Unable to get input dimensions");
            if(endCoords[0]-startCoords[0] != iDims[0] ||
               endCoords[1]-startCoords[1] != iDims[1] ||
               endCoords[2]-startCoords[2] != iDims[2]) mexErrMsgIdAndTxt("zarr:inputError","Bounding box size does not match the size of the input data");
        }
        else {
            Zarr.write_zarray();
        }

        const std::string dtypeT(Zarr.get_dtype());
        Zarr = zarr(folderName);
        if(dtypeT != Zarr.get_dtype()){
            uint64_t size = (endCoords[0]-startCoords[0])*
                (endCoords[1]-startCoords[1])*
                (endCoords[2]-startCoords[2]);

            uint64_t bitsT = 0;
            if(dtypeT == "<u1") bitsT = 8;
            else if(dtypeT == "<u2") bitsT = 16;
            else if(dtypeT == "<f4") bitsT = 32;
            else if(dtypeT == "<f8") bitsT = 64;
            else mexErrMsgIdAndTxt("tiff:dataTypeError","Cannont convert to passed in data type. Data type not suppported");


            if(Zarr.get_dtype() == "<u1"){
                zarrC = malloc(size*sizeof(uint8_t));
                if(bitsT == 16){
                    uint16_t* zarrT = (uint16_t*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((uint8_t*)zarrC)[i] = (uint8_t)zarrT[i];
                    }
                }
                else if(bitsT == 32){
                    float* zarrT = (float*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((uint8_t*)zarrC)[i] = (uint8_t)zarrT[i];
                    }
                }
                else if(bitsT == 64){
                    double* zarrT = (double*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((uint8_t*)zarrC)[i] = (uint8_t)zarrT[i];
                    }
                }
            }
            else if(Zarr.get_dtype() == "<u2"){
                zarrC = malloc(size*sizeof(uint16_t));
                if(bitsT == 8){
                    uint8_t* zarrT = (uint8_t*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((uint16_t*)zarrC)[i] = (uint16_t)zarrT[i];
                    }
                }
                else if (bitsT == 32){
                    float* zarrT = (float*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((uint16_t*)zarrC)[i] = (uint16_t)zarrT[i];
                    }
                }
                else if (bitsT == 64){
                    double* zarrT = (double*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((uint16_t*)zarrC)[i] = (uint16_t)zarrT[i];
                    }
                }
            }
            else if(Zarr.get_dtype() == "<f4"){
                zarrC = malloc(size*sizeof(float));
                if(bitsT == 8){
                    uint8_t* zarrT = (uint8_t*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((float*)zarrC)[i] = (float)zarrT[i];
                    }
                }
                else if(bitsT == 16){
                    uint16_t* zarrT = (uint16_t*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((float*)zarrC)[i] = (float)zarrT[i];
                    }
                }
                else if(bitsT == 64){
                    double* zarrT = (double*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((float*)zarrC)[i] = (float)zarrT[i];
                    }
                }
            }
            else if(Zarr.get_dtype() == "<f8"){
                zarrC = malloc(size*sizeof(double));
                if(bitsT == 8){
                    uint8_t* zarrT = (uint8_t*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((double*)zarrC)[i] = (double)zarrT[i];
                    }
                }
                else if(bitsT == 16){
                    uint16_t* zarrT = (uint16_t*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((double*)zarrC)[i] = (double)zarrT[i];
                    }
                }
                else if(bitsT == 32){
                    float* zarrT = (float*)mxGetPr(prhs[1]);
                    #pragma omp parallel for
                    for(uint64_t i = 0; i < size; i++){
                        ((double*)zarrC)[i] = (double)zarrT[i];
                    }
                }
            }
            else{
                mexErrMsgIdAndTxt("zarr:dataTypeError","Cannont convert to passed in data type. Data type not suppported");
            }
        }
    }


    //endCoords[0]-startCoords[0]
    if(endCoords[0] > Zarr.get_shape(0) ||
       endCoords[1] > Zarr.get_shape(1) ||
       endCoords[2] > Zarr.get_shape(2)) mexErrMsgIdAndTxt("zarr:inputError","Upper bound is invalid");
    if(!crop){
        endCoords = {Zarr.get_shape(0),Zarr.get_shape(1),Zarr.get_shape(2)};
        startCoords = {0,0,0};
    }
    const std::vector<uint64_t> writeShape({endCoords[0]-startCoords[0],
                                      endCoords[1]-startCoords[1],
                                      endCoords[2]-startCoords[2]});

    if(Zarr.get_dtype() == "<u1"){
        uint64_t bits = 8;
        uint8_t* zarrArr;
        if(zarrC) zarrArr = (uint8_t*)zarrC;
        else zarrArr =  (uint8_t*)mxGetPr(prhs[1]);
        parallelWriteZarrMex(Zarr, (void*)zarrArr, startCoords, endCoords, writeShape, bits, useUuid, crop);
    }
    else if(Zarr.get_dtype() == "<u2"){
        uint64_t bits = 16;
        uint16_t* zarrArr;
        if(zarrC) zarrArr = (uint16_t*)zarrC;
        else zarrArr = (uint16_t*)mxGetPr(prhs[1]);
        parallelWriteZarrMex(Zarr, (void*)zarrArr, startCoords, endCoords, writeShape, bits, useUuid, crop);
    }
    else if(Zarr.get_dtype() == "<f4"){
        uint64_t bits = 32;
        float* zarrArr;
        if(zarrC) zarrArr = (float*)zarrC;
        else zarrArr = (float*)mxGetPr(prhs[1]);
        parallelWriteZarrMex(Zarr, (void*)zarrArr, startCoords, endCoords, writeShape, bits, useUuid, crop);
    }
    else if(Zarr.get_dtype() == "<f8"){
        uint64_t bits = 64;
        double* zarrArr;
        if(zarrC) zarrArr = (double*)zarrC;
        else zarrArr = (double*)mxGetPr(prhs[1]);
        parallelWriteZarrMex(Zarr, (void*)zarrArr, startCoords, endCoords, writeShape, bits, useUuid, crop);
    }
    else{
        free(zarrC);
        mexErrMsgIdAndTxt("tiff:dataTypeError","Data type not suppported");
    }

    // zarrC is either a copy for data conversion or NULL
    free(zarrC);
}