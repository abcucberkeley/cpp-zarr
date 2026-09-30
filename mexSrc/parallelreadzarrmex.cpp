#include "mex.h"
#include "../src/helperfunctions.h"
#include "../src/parallelreadzarr.h"
#include "../src/zarr.h"
#include "zarrmexhelpers.h"

// TODO: FIX MEMORY LEAKS
void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{   
    if(!nrhs) mexErrMsgIdAndTxt("zarr:inputError","This functions requires at least 1 argument");
    if(!mxIsChar(prhs[0])) mexErrMsgIdAndTxt("zarr:inputError","The first argument must be a string");  

    std::vector<uint64_t> startCoords;
    std::vector<uint64_t> endCoords;
    std::vector<uint64_t> bboxVals;
    bool bbox = false;
    bool useCtx = true;
    bool sparse = false;
    std::string folderName(mxArrayToString(prhs[0]));

    for(int i = 1; i < nrhs; i+=2){
        if(i+1 == nrhs) mexErrMsgIdAndTxt("zarr:inputError","Mismatched argument pair for input number %d\n",i+1);
        if(!mxIsChar(prhs[i])) mexErrMsgIdAndTxt("zarr:inputError","The argument in input location %d is not a string\n",i+1);
        std::string currInput = mxArrayToString(prhs[i]);

        if(currInput == "bbox"){
            // Skip bbox if it is empty
            if(!mxGetN(prhs[i+1])) continue;
            // Interpreted once the array's number of dimensions is known
            bbox = true;
            bboxVals = mexVector(prhs[i+1]);
        
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
            mexErrMsgIdAndTxt("zarr:zarrayError","Cannot open %s for reading. Try checking permissions or the file path.\n",e.substr(e.find(':')+1).c_str());
        }
        else if(e == "metadataIncomplete"){
            mexErrMsgIdAndTxt("zarr:zarrayError","Metadata is incomplete. Check the .zarray file");
        }
        else mexErrMsgIdAndTxt("zarr:zarrayError","Unknown error occurred\n");
    }

    const uint64_t nDims = Zarr.get_ndims();
    if(bbox){
        std::vector<uint64_t> shape(nDims);
        for(uint64_t d = 0; d < nDims; d++) shape[d] = Zarr.get_shape(d);
        if(!mexParseRegion(bboxVals, nDims, startCoords, endCoords, shape)){
            mexErrMsgIdAndTxt("zarr:inputError","bbox must have %d values ([starts ends]) for this %dD array",(int)(2*nDims),(int)nDims);
        }
    }
    else{
        startCoords.assign(nDims,0);
        endCoords.assign(nDims,0);
        for(uint64_t d = 0; d < nDims; d++) endCoords[d] = Zarr.get_shape(d);
    }
    for(uint64_t d = 0; d < nDims; d++){
        if(endCoords[d] > Zarr.get_shape(d)) mexErrMsgIdAndTxt("zarr:inputError","Upper bound is invalid");
    }
    std::vector<uint64_t> readShape(nDims);
    for(uint64_t d = 0; d < nDims; d++) readShape[d] = endCoords[d]-startCoords[d];
    // MATLAB arrays have at least 2 dimensions
    std::vector<mwSize> dim(nDims < 2 ? 2 : nDims, 1);
    for(uint64_t d = 0; d < nDims; d++) dim[d] = readShape[d];
    // TESTING
    /*
    if(Zarr.get_order() == "C"){
        //Zarr.set_order("F");
        dim[0] = readShape[2];
        dim[2] = readShape[0];
        
    }
    */
    
    
    
    
    Zarr.set_chunkInfo(startCoords, endCoords);

    bool err = 0;
    uint64_t readSize = 1;
    for(uint64_t d = 0; d < nDims; d++) readSize *= readShape[d];

    // Map the zarr dtype to the matching MATLAB class. The read machinery is
    // element-width based, so one generic path covers every supported dtype:
    // signed/unsigned 8/16/32/64-bit integers and 32/64-bit floats.
    const std::string dtype = Zarr.get_dtype();
    const char kind = dtype.size() == 3 ? dtype[1] : '\0';
    const char dsize = dtype.size() == 3 ? dtype[2] : '\0';
    mxClassID mxClass = mxUNKNOWN_CLASS;
    if(kind == 'u'){
        if(dsize == '1') mxClass = mxUINT8_CLASS;
        else if(dsize == '2') mxClass = mxUINT16_CLASS;
        else if(dsize == '4') mxClass = mxUINT32_CLASS;
        else if(dsize == '8') mxClass = mxUINT64_CLASS;
    }
    else if(kind == 'i'){
        if(dsize == '1') mxClass = mxINT8_CLASS;
        else if(dsize == '2') mxClass = mxINT16_CLASS;
        else if(dsize == '4') mxClass = mxINT32_CLASS;
        else if(dsize == '8') mxClass = mxINT64_CLASS;
    }
    else if(kind == 'f'){
        if(dsize == '4') mxClass = mxSINGLE_CLASS;
        else if(dsize == '8') mxClass = mxDOUBLE_CLASS;
    }
    if(mxClass == mxUNKNOWN_CLASS) mexErrMsgIdAndTxt("zarr:dataTypeError","Data type \"%s\" is not supported",dtype.c_str());

    const uint64_t bytes = Zarr.dtypeBytes();
    void* zarrArr = NULL;
    const int fillValue = fillValueToInt(Zarr.get_fill_value());
    if(fillValue){
        plhs[0] = mxCreateUninitNumericArray(dim.size(),dim.data(),mxClass, mxREAL);
        zarrArr = mxGetData(plhs[0]);
        memset(zarrArr,fillValue,readSize*bytes);
    }
    else{
        plhs[0] = mxCreateNumericArray(dim.size(),dim.data(),mxClass, mxREAL);
        zarrArr = mxGetData(plhs[0]);
    }
    err = parallelReadZarr(Zarr, zarrArr,startCoords,endCoords,readShape,bytes*8,useCtx,sparse);

    if(err) mexErrMsgIdAndTxt("zarr:readError",Zarr.get_errString().c_str());

}
