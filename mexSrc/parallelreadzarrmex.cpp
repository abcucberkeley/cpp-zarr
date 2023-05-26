#include <omp.h>
#include <fstream>
#include "../src/parallelreadzarr.h"
#include "blosc2.h"
#include "mex.h"
#include "../src/zarr.h"
#include "../src/helperfunctions.h"
#include "zlib.h"

// TODO: FIX MEMORY LEAKS
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    std::vector<uint64_t> startCoords = {0,0,0};
    std::vector<uint64_t> endCoords = {0,0,0};

    if(!nrhs) mexErrMsgIdAndTxt("zarr:inputError","This functions requires at least 1 argument");
    else if(nrhs == 2){
        if(mxGetN(prhs[1]) != 6) mexErrMsgIdAndTxt("zarr:inputError","Input range is not 6");
        startCoords[0] = (uint64_t)*(mxGetPr(prhs[1]))-1;
        startCoords[1] = (uint64_t)*((mxGetPr(prhs[1])+1))-1;
        startCoords[2] = (uint64_t)*((mxGetPr(prhs[1])+2))-1;
        endCoords[0] = (uint64_t)*((mxGetPr(prhs[1])+3));
        endCoords[1] = (uint64_t)*((mxGetPr(prhs[1])+4));
        endCoords[2] = (uint64_t)*((mxGetPr(prhs[1])+5));
        
        if(startCoords[0]+1 < 1 || startCoords[1]+1 < 1 || startCoords[2]+1 < 1) mexErrMsgIdAndTxt("zarr:inputError","Lower bounds must be at least 1");
    }
    else if (nrhs > 2) mexErrMsgIdAndTxt("zarr:inputError","Number of input arguments must be 1 or 2");
    if(!mxIsChar(prhs[0])) mexErrMsgIdAndTxt("zarr:inputError","The first argument must be a string");
    std::string folderName(mxArrayToString(prhs[0]));
    
    // Handle the tilde character in filenames on Linux/Mac
    #ifndef _WIN32
    folderName = expandTilde(folderName.c_str());
    #endif

    zarr Zarr(folderName);
    

    if(endCoords[0] > Zarr.get_shape(0) || 
       endCoords[1] > Zarr.get_shape(1) || 
       endCoords[2] > Zarr.get_shape(2)) mexErrMsgIdAndTxt("zarr:inputError","Upper bound is invalid");
    if(nrhs == 1){
        endCoords[0] = Zarr.get_shape(0);
        endCoords[1] = Zarr.get_shape(1);
        endCoords[2] = Zarr.get_shape(2);
    }
    const std::vector<uint64_t> readShape = {endCoords[0]-startCoords[0],
                                             endCoords[1]-startCoords[1],
                                             endCoords[2]-startCoords[2]};
    uint64_t dim[3] = {readShape[0],readShape[1],readShape[2]};
    
    Zarr.set_chunkInfo(startCoords, endCoords);
    if(Zarr.get_dtype().find("u1") != std::string::npos){
        uint64_t bits = 8;
        plhs[0] = mxCreateNumericArray(3,(mwSize*)dim,mxUINT8_CLASS, mxREAL);
        uint8_t* zarrArr = (uint8_t*)mxGetPr(plhs[0]);
        parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits);
    }
    else if(Zarr.get_dtype().find("u2") != std::string::npos){
        uint64_t bits = 16;
        plhs[0] = mxCreateNumericArray(3,(mwSize*)dim,mxUINT16_CLASS, mxREAL);
        uint16_t* zarrArr = (uint16_t*)mxGetPr(plhs[0]);
        parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits);
    }
    else if(Zarr.get_dtype().find("f4") != std::string::npos){
        uint64_t bits = 32;
        plhs[0] = mxCreateNumericArray(3,(mwSize*)dim,mxSINGLE_CLASS, mxREAL);
        float* zarrArr = (float*)mxGetPr(plhs[0]);
        parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits);
    }
    else if(Zarr.get_dtype().find("f8") != std::string::npos){
        uint64_t bits = 64;
        plhs[0] = mxCreateNumericArray(3,(mwSize*)dim,mxDOUBLE_CLASS, mxREAL);
        double* zarrArr = (double*)mxGetPr(plhs[0]);
        parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits);
    }
    else{
        mexErrMsgIdAndTxt("tiff:dataTypeError","Data type not suppported");
    }

}
