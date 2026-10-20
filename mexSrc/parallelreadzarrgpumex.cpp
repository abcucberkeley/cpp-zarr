// parallelReadZarrGPU(file, 'bbox', [starts ends]): parallelReadZarr's read, into a gpuArray (the
// Parallel Computing Toolbox's GPU arrays), for data that is needed on the GPU anyway: the chunks'
// stored bytes are read here and decompressed on the GPU with nvCOMP (gpureadzarr.h). float16 and
// bfloat16 arrays read as single, as parallelReadZarr reads them. Needs an NVIDIA GPU and nvCOMP
// (see gpureadzarr.h; setenv('CPPZARR_NVCOMP_LIBRARY', <libnvcomp file>) before the first read
// tells it where nvCOMP is).
#include <cstring>
#include "mex.h"
#include "gpu/mxGPUArray.h"
#include "../src/gpureadzarr.h"
#include "../src/helperfunctions.h"
#include "../src/zarr.h"
#include "zarrmexhelpers.h"

void mexFunction(int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[])
{
    if(!nrhs) mexErrMsgIdAndTxt("zarr:inputError","This functions requires at least 1 argument");
    if(!mxIsChar(prhs[0])) mexErrMsgIdAndTxt("zarr:inputError","The first argument must be a string");

    std::vector<uint64_t> startCoords;
    std::vector<uint64_t> endCoords;
    std::vector<uint64_t> bboxVals;
    bool bbox = false;
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
            // (accepted as parallelReadZarr takes it: missing chunks always read as the fill value)
        }
        else{
            mexErrMsgIdAndTxt("zarr:inputError","The argument \"%s\" does not match the name of any supported input name.\n \
            Currently Supported Names: bbox, sparse\n",currInput.c_str());
        }
    }

    if(mxInitGPU() != MX_GPU_SUCCESS) mexErrMsgIdAndTxt("zarr:gpuError","No GPU MATLAB can use (mxInitGPU failed)");
    std::string why;
    if(!gpuReadAvailable(why)) mexErrMsgIdAndTxt("zarr:gpuError","GPU reads are not available: %s",why.c_str());

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
            mexErrMsgIdAndTxt("zarr:zarrayError","Metadata is incomplete. Check the .zarray or zarr.json file");
        }
        else mexZarrError(e);
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
        if(startCoords[d] >= endCoords[d]) mexErrMsgIdAndTxt("zarr:inputError","The region is empty on axis %d",(int)d+1);
    }
    // MATLAB arrays have at least 2 dimensions
    std::vector<mwSize> dim(nDims < 2 ? 2 : nDims, 1);
    for(uint64_t d = 0; d < nDims; d++) dim[d] = endCoords[d]-startCoords[d];

    // The MATLAB class of the dtype, as parallelReadZarr maps it (gpuArrays are interleaved
    // complex, as complex64 and complex128 are)
    const std::string dtype = Zarr.get_dtype();
    const DtypeInfo t = dtypeInfo(dtype);
    mxClassID mxClass = mxUNKNOWN_CLASS;
    if(t.kind == 'u'){
        if(t.bytes == 1) mxClass = mxUINT8_CLASS;
        else if(t.bytes == 2) mxClass = mxUINT16_CLASS;
        else if(t.bytes == 4) mxClass = mxUINT32_CLASS;
        else if(t.bytes == 8) mxClass = mxUINT64_CLASS;
    }
    else if(t.kind == 'i'){
        if(t.bytes == 1) mxClass = mxINT8_CLASS;
        else if(t.bytes == 2) mxClass = mxINT16_CLASS;
        else if(t.bytes == 4) mxClass = mxINT32_CLASS;
        else if(t.bytes == 8) mxClass = mxINT64_CLASS;
    }
    else if(t.kind == 'f') mxClass = t.bytes == 8 ? mxDOUBLE_CLASS : mxSINGLE_CLASS;
    else if(t.kind == 'c') mxClass = t.bytes == 16 ? mxDOUBLE_CLASS : mxSINGLE_CLASS;
    else if(t.kind == 'b') mxClass = mxLOGICAL_CLASS;
    if(mxClass == mxUNKNOWN_CLASS) mexErrMsgIdAndTxt("zarr:dataTypeError","Data type \"%s\" is not supported",dtype.c_str());

    mxGPUArray* out = mxGPUCreateGPUArray((mwSize)dim.size(), dim.data(), mxClass, t.kind == 'c' ? mxCOMPLEX : mxREAL,
                                          MX_GPU_DO_NOT_INITIALIZE);
    if(!out) mexErrMsgIdAndTxt("zarr:readError","Out of GPU memory");
    // (F order, MATLAB's; on MATLAB's current stream's device, the default stream, and done when
    // gpuReadZarr returns)
    std::vector<GpuRegion> regions{{&Zarr, startCoords, endCoords, mxGPUGetData(out)}};
    std::string err;
    if(gpuReadZarr(regions, false, nullptr, 0, err, true)){
        mxGPUDestroyGPUArray(out);
        mexErrMsgIdAndTxt("zarr:readError","%s",err.c_str());
    }
    plhs[0] = mxGPUCreateMxArrayOnGPU(out);
    mxGPUDestroyGPUArray(out);
}
