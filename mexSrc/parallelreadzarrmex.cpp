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
    bool bbox = false;
    bool sparse = false;
    if(!nrhs) mexErrMsgIdAndTxt("zarr:inputError","This functions requires at least 1 argument");
    /*
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
    */
    if(!mxIsChar(prhs[0])) mexErrMsgIdAndTxt("zarr:inputError","The first argument must be a string");
    std::string folderName(mxArrayToString(prhs[0]));

    for(int i = 1; i < nrhs; i+=2){
        if(i+1 == nrhs) mexErrMsgIdAndTxt("zarr:inputError","Mismatched argument pair for input number %d\n",i+1);
        if(!mxIsChar(prhs[i])) mexErrMsgIdAndTxt("zarr:inputError","The argument in input location %d is not a string\n",i+1);
        std::string currInput = mxArrayToString(prhs[i]);

        if(currInput == "bbox"){
            // Skip bbox if it is empty
            if(!mxGetN(prhs[i+1])) continue;
            else if(mxGetN(prhs[i+1]) != 6) mexErrMsgIdAndTxt("zarr:inputError","Input range is not 6");
            bbox = true;
            startCoords[0] = (uint64_t)*(mxGetPr(prhs[i+1]))-1;
            startCoords[1] = (uint64_t)*((mxGetPr(prhs[i+1])+1))-1;
            startCoords[2] = (uint64_t)*((mxGetPr(prhs[i+1])+2))-1;
            endCoords[0] = (uint64_t)*((mxGetPr(prhs[i+1])+3));
            endCoords[1] = (uint64_t)*((mxGetPr(prhs[i+1])+4));
            endCoords[2] = (uint64_t)*((mxGetPr(prhs[i+1])+5));
            
            if(startCoords[0]+1 < 1 || startCoords[1]+1 < 1 || startCoords[2]+1 < 1) mexErrMsgIdAndTxt("zarr:inputError","Lower bounds must be at least 1");
        
        }
        else if(currInput == "sparse"){
            sparse = (bool)*((mxGetPr(prhs[i+1])));
        }
        else{
            mexErrMsgIdAndTxt("zarr:inputError","The argument \"%s\" does not match the name of any supported input name.\n \
            Currently Supported Names: bbox, sparse\n",currInput.c_str());
        }
    }
    
    // Handle the tilde character in filenames on Linux/Mac
    #ifndef _WIN32
    folderName = expandTilde(folderName.c_str());
    #endif
    zarr Zarr;
    try{
        Zarr = zarr(folderName);
    }
    catch(const std::string &e){
        if(e.find("metadataFileMissing") != std::string::npos){
            mexErrMsgIdAndTxt("zarr:zarrayError","Cannot open %s for writing. Try checking permissions or the file path.\n",e.substr(e.find(':')+1).c_str());
        }
        else if(e == "metadataIncomplete"){
            mexErrMsgIdAndTxt("zarr:zarrayError","Metadata is incomplete. Check the .zarray file");
        }
        else mexErrMsgIdAndTxt("zarr:zarrayError","Unknown error occurred\n");
    }

    if(endCoords[0] > Zarr.get_shape(0) || 
       endCoords[1] > Zarr.get_shape(1) || 
       endCoords[2] > Zarr.get_shape(2)) mexErrMsgIdAndTxt("zarr:inputError","Upper bound is invalid");
    if(!bbox){
        endCoords[0] = Zarr.get_shape(0);
        endCoords[1] = Zarr.get_shape(1);
        endCoords[2] = Zarr.get_shape(2);
    }
    const std::vector<uint64_t> readShape = {endCoords[0]-startCoords[0],
                                             endCoords[1]-startCoords[1],
                                             endCoords[2]-startCoords[2]};
    uint64_t dim[3] = {readShape[0],readShape[1],readShape[2]};
    
    Zarr.set_chunkInfo(startCoords, endCoords);

    bool err = 0;
    if(Zarr.get_dtype().find("u1") != std::string::npos){
        uint64_t bits = 8;
        // if sparse else use mxCreateUninitNumericArray
        plhs[0] = mxCreateNumericArray(3,(mwSize*)dim,mxUINT8_CLASS, mxREAL);
        uint8_t* zarrArr = (uint8_t*)mxGetPr(plhs[0]);
        err = parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits,false,sparse);
    }
    else if(Zarr.get_dtype().find("u2") != std::string::npos){
        uint64_t bits = 16;
        plhs[0] = mxCreateNumericArray(3,(mwSize*)dim,mxUINT16_CLASS, mxREAL);
        uint16_t* zarrArr = (uint16_t*)mxGetPr(plhs[0]);
        err = parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits,false,sparse);
    }
    else if(Zarr.get_dtype().find("f4") != std::string::npos){
        uint64_t bits = 32;
        plhs[0] = mxCreateNumericArray(3,(mwSize*)dim,mxSINGLE_CLASS, mxREAL);
        float* zarrArr = (float*)mxGetPr(plhs[0]);
        err = parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits,false,sparse);
    }
    else if(Zarr.get_dtype().find("f8") != std::string::npos){
        uint64_t bits = 64;
        plhs[0] = mxCreateNumericArray(3,(mwSize*)dim,mxDOUBLE_CLASS, mxREAL);
        double* zarrArr = (double*)mxGetPr(plhs[0]);
        err = parallelReadZarr(Zarr, (void*)zarrArr,startCoords,endCoords,readShape,bits,false,sparse);
    }
    else{
        mexErrMsgIdAndTxt("tiff:dataTypeError","Data type not suppported");
    }
    
    if(err) mexErrMsgIdAndTxt("zarr:readError",Zarr.get_errString().c_str());

}
