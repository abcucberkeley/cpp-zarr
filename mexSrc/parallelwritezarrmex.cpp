#include <cstdint>
#include <cstring>
#include "mex.h"
#include "../src/helperfunctions.h"
#include "../src/parallelwritezarr.h"
#include "../src/parallelreadzarr.h"
#include "../src/zarr.h"
#include "zarrmexhelpers.h"
#ifdef CPPZARR_MEX_GPU
// (parallelWriteZarrGPU: this file built with CPPZARR_MEX_GPU, see parallelwritezarrgpumex.cpp)
#include "gpu/mxGPUArray.h"
#include "../src/gpuwritezarr.h"

// A gpuArray to write: its dimensions, data type (as mexClassDtype names a class) and GPU memory
struct MexGpuData {
    const mxGPUArray* g = nullptr;
    std::vector<uint64_t> dims;
    std::string dtype;
    uint64_t elems = 0;
    explicit MexGpuData(const mxArray* a){
        g = mxGPUCreateFromMxArray(a);
        const mwSize n = mxGPUGetNumberOfDimensions(g);
        const mwSize* d = mxGPUGetDimensions(g);
        dims.assign(d, d+n);
        mxFree((void*)d);
        elems = (uint64_t)mxGPUGetNumberOfElements(g);
        const bool cplx = mxGPUGetComplexity(g) == mxCOMPLEX;
        switch(mxGPUGetClassID(g)){
            case mxLOGICAL_CLASS: dtype = "|b1"; break;
            case mxUINT8_CLASS:  dtype = cplx ? "" : "<u1"; break;
            case mxINT8_CLASS:   dtype = cplx ? "" : "<i1"; break;
            case mxUINT16_CLASS: dtype = cplx ? "" : "<u2"; break;
            case mxINT16_CLASS:  dtype = cplx ? "" : "<i2"; break;
            case mxUINT32_CLASS: dtype = cplx ? "" : "<u4"; break;
            case mxINT32_CLASS:  dtype = cplx ? "" : "<i4"; break;
            case mxUINT64_CLASS: dtype = cplx ? "" : "<u8"; break;
            case mxINT64_CLASS:  dtype = cplx ? "" : "<i8"; break;
            case mxSINGLE_CLASS: dtype = cplx ? "<c8" : "<f4"; break;
            case mxDOUBLE_CLASS: dtype = cplx ? "<c16" : "<f8"; break;
            default: break;
        }
    }
    ~MexGpuData(){ if(g) mxGPUDestroyGPUArray(g); }
    MexGpuData(const MexGpuData&) = delete;
    MexGpuData& operator=(const MexGpuData&) = delete;
};
#endif

//compile
//mex -v COPTIMFLAGS="-DNDEBUG -O3" CFLAGS='$CFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp -O3' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/lib' -lblosc '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/lib64' -lblosc2 '-L/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/lib64' -lcjson -luuid parallelWriteZarr.c helperFunctions.c parallelReadZarr.c

//With zlib
//mex -v COPTIMFLAGS="-DNDEBUG -O3" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O3 -DNDEBUG" CFLAGS='$CFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp -O3' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/lib' -lblosc '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/lib64' -lblosc2 '-L/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/lib64' -lcjson -luuid -lz parallelWriteZarr.c helperFunctions.c parallelReadZarr.c

//mex -v COPTIMFLAGS="-O3 -fwrapv -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/lib64' -lblosc2 zarrMex.c
//
//Windows
//mex -v COPTIMFLAGS="-O3 -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-IC:\Program Files (x86)\bloscZarr\include' '-LC:\Program Files (x86)\bloscZarr\lib' -lblosc '-IC:\Program Files (x86)\cJSON\include\' '-LC:\Program Files (x86)\cJSON\lib' -lcjson '-IC:\Program Files (x86)\blosc\include' '-LC:\Program Files (x86)\blosc\lib' -lblosc2 parallelWriteZarr.c parallelReadZarr.c helperFunctions.c

// TODO: FIX MEMORY LEAKS
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
#if defined(__APPLE__)
    // keep the mex file loaded until MATLAB exits; unloading it crashes MATLAB on exit
    if (!mexIsLocked()) {
        mexLock();
    }
#endif
    if(nrhs < 2) mexErrMsgIdAndTxt("zarr:inputError","This functions requires at least 2 arguments");
    if(!mxIsChar(prhs[0])) mexErrMsgIdAndTxt("zarr:inputError","The first argument must be a string");
#ifdef CPPZARR_MEX_GPU
    // (the data: a gpuArray, whose chunks are put together and compressed on the GPU)
    if(!mxIsGPUArray(prhs[1])) mexErrMsgIdAndTxt("zarr:inputError","parallelWriteZarrGPU writes gpuArrays: write other arrays with parallelWriteZarr");
    if(mxInitGPU() != MX_GPU_SUCCESS) mexErrMsgIdAndTxt("zarr:gpuError","No GPU MATLAB can use (mxInitGPU failed)");
    {
        std::string why;
        if(!gpuWriteAvailable(why)) mexErrMsgIdAndTxt("zarr:gpuError","GPU writes are not available: %s",why.c_str());
    }
    const MexGpuData gpuData(prhs[1]);
    if(!gpuData.elems) mexErrMsgIdAndTxt("zarr:inputError","All input data axes must be of at least size 1");
#else
    if(mxIsEmpty(prhs[1])) mexErrMsgIdAndTxt("zarr:inputError","All input data axes must be of at least size 1");
#endif

    std::vector<uint64_t> startCoords;
    std::vector<uint64_t> endCoords;
    bool crop = false;
    bool useUuid = true;
    bool sparse = true;
    int threads = 0;
    // Per-axis options are collected first and fit to the array's number of
    // dimensions once it is known
    std::vector<uint64_t> bboxVals, chunksVals, subfoldersVals, chunkShapeVals;
    bool hasChunks = false, hasSubfolders = false, hasChunkShape = false;
    // Applied after the Zarr format, which sets their Zarr v3 defaults
    uint64_t zarrFormat = 0;
    std::string order, dimension_separator;
    // The data type of a new array (by default the data's)
    std::string dtypeOption;

    // Input data dimensions (MATLAB arrays have at least 2)
#ifdef CPPZARR_MEX_GPU
    const std::vector<uint64_t> dataDims = gpuData.dims;
    const uint64_t nDataDims = dataDims.size();
#else
    const uint64_t nDataDims = (uint64_t)mxGetNumberOfDimensions(prhs[1]);
    const mwSize* dataDimsT = mxGetDimensions(prhs[1]);
    const std::vector<uint64_t> dataDims(dataDimsT, dataDimsT+nDataDims);
#endif

    std::string folderName(mxArrayToString(prhs[0]));
    // Handle the tilde character in filenames on Linux/Mac
    #ifndef _WIN32
    folderName = expandTilde(folderName.c_str());
    #endif
    
    // Check if metadata exists that we can use or if we have to create new metadata
    zarr Zarr;
    const bool metadataExists = fileExists(folderName+"/.zarray") || fileExists(folderName+"/zarr.json");
    if(!metadataExists){
        Zarr.set_fileName(folderName);
    }
    else{
        try{
            Zarr = zarr(folderName);
        }
        catch(const std::string &e){
            if(e.find("metadataFileMissing") != std::string::npos){
                mexErrMsgIdAndTxt("zarr:zarrayError","Cannot open %s for writing. Try checking permissions or the file path.\n",e.substr(e.find(':')+1).c_str());
            }
            else if(e == "metadataIncomplete"){
                mexErrMsgIdAndTxt("zarr:zarrayError","Metadata is incomplete. Check the .zarray or zarr.json file");
            }
            else mexZarrError(e);
        }
    }

    for(int i = 2; i < nrhs; i+=2){
        if(i+1 == nrhs) mexErrMsgIdAndTxt("zarr:inputError","Mismatched argument pair for input number %d\n",i+1);
        if(!mxIsChar(prhs[i])) mexErrMsgIdAndTxt("zarr:inputError","The argument in input location %d is not a string\n",i+1);
        std::string currInput = mxArrayToString(prhs[i]);

        if(currInput == "uuid"){
            useUuid = (bool)*((mxGetPr(prhs[i+1])));
        }
        else if(currInput == "bbox"){
            // Skip bbox if it is empty
            if(!mxGetN(prhs[i+1])) continue;
            // A region [starts ends], or one value per axis for the chunk size
            bboxVals = mexVector(prhs[i+1]);
        }
        else if(currInput == "cname"){
            if(!mxIsChar(prhs[i+1])) mexErrMsgIdAndTxt("zarr:inputError","cname must be a string\n");
            Zarr.set_cname(mxArrayToString(prhs[i+1]));
        }
        // (like cname, used for a new array; a region written into an existing array
        // keeps the array's)
        else if(currInput == "clevel"){
            Zarr.set_clevel(mexClevel(prhs[i+1]));
        }
        else if(currInput == "subfolders"){
            subfoldersVals = mexVector(prhs[i+1]);
            hasSubfolders = true;
        }
        else if(currInput == "chunks"){
            chunksVals = mexVector(prhs[i+1]);
            hasChunks = true;
        }
        else if(currInput == "chunk_shape"){
            chunkShapeVals = mexVector(prhs[i+1]);
            hasChunkShape = true;
        }
        else if(currInput == "sparse"){
            sparse = (bool)*((mxGetPr(prhs[i+1])));
        }
        else if(currInput == "dimension_separator"){
            if(!mxIsChar(prhs[i+1])) mexErrMsgIdAndTxt("zarr:inputError","dimension_separator must be a string\n");
            dimension_separator = mxArrayToString(prhs[i+1]);
            if(dimension_separator != "." && dimension_separator != "/") mexErrMsgIdAndTxt("zarr:inputError","dimension_separator must be a . or /\n");
        }
        else if(currInput == "order"){
            order = mexOrder(prhs[i+1]);
        }
        else if(currInput == "zarr_format"){
            zarrFormat = mexZarrFormat(prhs[i+1]);
        }
        else if(currInput == "dtype"){
            dtypeOption = mexDtype(prhs[i+1]);
        }
        else if(currInput == "threads"){
            threads = mexThreads(prhs[i+1]);
        }
        else{
            mexErrMsgIdAndTxt("zarr:inputError","The argument \"%s\" does not match the name of any supported input name.\n \
            Currently Supported Names: uuid, bbox, cname, clevel, subfolders, chunks, chunk_shape, sparse, dimension_separator, order, zarr_format, dtype, threads\n",currInput.c_str());
        }
    }
    const MexCallThreads callThreadsSet(threads);

    // The Zarr format: as given, or the existing array's (2 for a new array). A new
    // format (a new array, or an existing one rewritten in the other format) starts
    // from that format's defaults for order and dimension_separator.
    const uint64_t existingFormat = metadataExists ? Zarr.get_zarr_format() : 0;
    if(zarrFormat && zarrFormat != existingFormat) Zarr.set_zarr_format(zarrFormat);
    if(!order.empty()) Zarr.set_order(order);
    if(!dimension_separator.empty()) Zarr.set_dimension_separator(dimension_separator);

    // Number of dimensions of the array being written: a region written into an
    // existing array uses that array's; otherwise the input data's
    uint64_t nDims = nDataDims;
    std::vector<uint64_t> chunksFromBbox;
    if(!bboxVals.empty()){
        uint64_t nRegion = nDataDims;
        std::vector<uint64_t> fileShape;
        if(metadataExists){
            nRegion = Zarr.get_ndims();
            for(uint64_t d = 0; d < nRegion; d++) fileShape.push_back(Zarr.get_shape(d));
        }
        if(mexParseRegion(bboxVals, nRegion, startCoords, endCoords, fileShape)){
            crop = true;
            nDims = nRegion;
        }
        else if(bboxVals.size() == nDataDims || (bboxVals.size() == 3 && nDataDims < 3)){
            // One value per axis: the chunk size
            chunksFromBbox = mexFitAxes(bboxVals, nDataDims, "bbox");
        }
        else{
            mexErrMsgIdAndTxt("zarr:inputError","bbox must have %d values ([starts ends]) or %d values (chunk size)",
                              (int)(2*nRegion),(int)nDataDims);
        }
    }

    // The data may omit trailing singleton axes (MATLAB drops them)
    std::vector<uint64_t> iDims(nDims,1);
    for(uint64_t d = 0; d < nDataDims; d++){
        if(d < nDims) iDims[d] = dataDims[d];
        else if(dataDims[d] != 1) mexErrMsgIdAndTxt("zarr:inputError","Bounds are invalid for the input data size");
    }
    if(crop){
        for(uint64_t d = 0; d < nDims; d++){
            if(endCoords[d]-startCoords[d] > iDims[d]) mexErrMsgIdAndTxt("zarr:inputError","Bounds are invalid for the input data size");
        }
        // Writing a region into an existing array keeps its format
        if(metadataExists && zarrFormat && zarrFormat != existingFormat){
            mexErrMsgIdAndTxt("zarr:inputError","%s is a Zarr v%d array; writing a region keeps its format%s\n",folderName.c_str(),
                              (int)existingFormat,existingFormat == 2 ? " (convert it with convertZarrToV3 first)" : "");
        }
    }

    if(hasSubfolders) Zarr.set_subfolders(mexFitAxes(subfoldersVals, nDims, "subfolders"));
    if(hasChunks) Zarr.set_chunks(mexFitAxes(chunksVals, nDims, "chunks"));
    if(hasChunkShape){
        Zarr.set_shard(true);
        Zarr.set_chunk_shape(mexFitAxes(chunkShapeVals, nDims, "chunk_shape"));
    }

    void* zarrC = NULL;

    // The data's own type (logical is bool, complex single and double are complex64
    // and complex128), which a new array gets unless the dtype option says otherwise
#ifdef CPPZARR_MEX_GPU
    const std::string dataDtype = gpuData.dtype;
#else
    const std::string dataDtype = mexClassDtype(prhs[1]);
#endif
    if(dataDtype.empty()) mexErrMsgIdAndTxt("zarr:dataTypeError","The input data type is not supported");
    const bool cropExisting = crop && metadataExists;
    if(!cropExisting) Zarr.set_dtype(dtypeOption.empty() ? dataDtype : dtypeOption);

    if(!crop){
        Zarr.set_shape(iDims);
        if(!chunksFromBbox.empty()) Zarr.set_chunks(chunksFromBbox);
        try{
            Zarr.write_zarray();
        }
        catch(const std::string &e){
            if(e == "unsupportedCompressor"){
                mexErrMsgIdAndTxt("zarr:zarrayError","Compressor: \"%s\" is not currently supported\n",Zarr.get_cname().c_str());
            }
            else if(e.find("cannotOpenZarray") != std::string::npos){
                mexErrMsgIdAndTxt("zarr:zarrayError","Cannot open %s for writing. Try checking permissions and path.\n",e.substr(e.find(':')+1).c_str());
            }
            else mexZarrError(e);
        }
    }
    else{
        Zarr.set_shape(endCoords);

        if(metadataExists){
            for(uint64_t d = 0; d < nDims; d++){
                if(endCoords[d]-startCoords[d] != iDims[d]) mexErrMsgIdAndTxt("zarr:inputError","Bounding box size does not match the size of the input data");
            }
        }
        else {
            try{
                Zarr.write_zarray();
            }
            catch(const std::string &e){
                if(e == "unsupportedCompressor"){
                    mexErrMsgIdAndTxt("zarr:zarrayError","Compressor: \"%s\" is not currently supported\n",Zarr.get_cname().c_str());
                }
                else if(e.find("cannotOpenZarray") != std::string::npos){
                    mexErrMsgIdAndTxt("zarr:zarrayError","Cannot open %s for writing. Try checking permissions and path.\n",e.substr(e.find(':')+1).c_str());
                }
                else mexZarrError(e);
            }
        }

        try{
            Zarr = zarr(folderName);
        }
        catch(const std::string &e){
            if(e.find("metadataFileMissing") != std::string::npos){
                mexErrMsgIdAndTxt("zarr:zarrayError","Cannot open %s for writing. Try checking permissions or the file path.\n",e.substr(e.find(':')+1).c_str());
            }
            else if(e == "metadataIncomplete"){
                mexErrMsgIdAndTxt("zarr:zarrayError","Metadata is incomplete. Check the .zarray or zarr.json file");
            }
            else mexZarrError(e);
        }
        
        // (the dtype option names the type of a new array: an existing one keeps its own)
        if(cropExisting && !dtypeOption.empty()){
            const DtypeInfo want = dtypeInfo(dtypeOption), have = dtypeInfo(Zarr.get_dtype());
            if(want.kind != have.kind || want.bytes != have.bytes || want.bfloat16 != have.bfloat16){
                mexErrMsgIdAndTxt("zarr:dataTypeError","%s has data type \"%s\", not \"%s\" (dtype is the data type of a new array)",
                                  folderName.c_str(),Zarr.get_dtype().c_str(),dtypeOption.c_str());
            }
        }
    }

    // The data in the array's data type: converted as MATLAB converts between classes
    // (integers rounded and saturated) when the types differ, and complex data with
    // its real and imaginary parts interleaved, as Zarr stores them
    {
        const DtypeInfo out = dtypeInfo(Zarr.get_dtype()), in = dtypeInfo(dataDtype);
        if(in.kind == 'c' && out.kind != 'c'){
            mexErrMsgIdAndTxt("zarr:dataTypeError","Complex data can only be written into complex64 or complex128 arrays, not \"%s\"",
                              Zarr.get_dtype().c_str());
        }
        bool convert = out.kind != in.kind || out.bytes != in.bytes || out.bfloat16 != in.bfloat16;
#if !(MX_HAS_INTERLEAVED_COMPLEX) && !defined(CPPZARR_MEX_GPU)
        convert = convert || in.kind == 'c';
#endif
        if(convert && out.kind){
            uint64_t size = 1;
            for(uint64_t d = 0; d < nDims; d++) size *= iDims[d];
#ifdef CPPZARR_MEX_GPU
            // (converted as parallelWriteZarr converts data, in host memory, and written from there:
            // data cast on the GPU to the array's class is written from the GPU)
            mxArray* gathered = nullptr;
            mxArray* gin = const_cast<mxArray*>(prhs[1]);
            if(mexCallMATLAB(1, &gathered, 1, &gin, "gather") || !gathered)
                mexErrMsgIdAndTxt("zarr:dataTypeError","Cannot copy the input data to host memory");
            zarrC = mexConvertToDtype(gathered, Zarr.get_dtype(), size);
            mxDestroyArray(gathered);
#else
            zarrC = mexConvertToDtype(prhs[1], Zarr.get_dtype(), size);
#endif
            if(!zarrC) mexErrMsgIdAndTxt("zarr:dataTypeError","Cannot convert the input data to the data type \"%s\"",Zarr.get_dtype().c_str());
        }
    }

    if(!crop){
        startCoords.assign(nDims,0);
        endCoords.assign(nDims,0);
        for(uint64_t d = 0; d < nDims; d++) endCoords[d] = Zarr.get_shape(d);
    }
    for(uint64_t d = 0; d < nDims; d++){
        if(endCoords[d] > Zarr.get_shape(d)) mexErrMsgIdAndTxt("zarr:inputError","Upper bound is invalid");
    }
    std::vector<uint64_t> writeShape(nDims);
    for(uint64_t d = 0; d < nDims; d++) writeShape[d] = endCoords[d]-startCoords[d];

    Zarr.set_chunkInfo(startCoords, endCoords);
    bool err = 0;
    // The write machinery is element-width based, so one generic path covers
    // every supported dtype
    const std::string &dtypeOut = Zarr.get_dtype();
    const uint64_t outBytes = Zarr.dtypeBytes();
    if(!dtypeInfo(dtypeOut).kind){
        free(zarrC);
        mexErrMsgIdAndTxt("zarr:dataTypeError","Data type \"%s\" is not supported",dtypeOut.c_str());
    }
#ifdef CPPZARR_MEX_GPU
    if(zarrC) err = parallelWriteZarr(Zarr, zarrC, startCoords, endCoords, writeShape, outBytes*8, useUuid, crop, sparse);
    else{
        // (F order, MATLAB's; on MATLAB's current stream's device, the default stream)
        std::vector<uint64_t> inStrides(nDims);
        uint64_t acc = 1;
        for(uint64_t d = 0; d < nDims; d++){ inStrides[d] = acc; acc *= iDims[d]; }
        err = gpuWriteZarr(Zarr, mxGPUGetDataReadOnly(gpuData.g), startCoords, endCoords, writeShape, inStrides, outBytes*8, useUuid, crop,
                           sparse, nullptr);
    }
#else
    void* zarrArr = zarrC ? zarrC : mxGetData(prhs[1]);
    err = parallelWriteZarr(Zarr, zarrArr, startCoords, endCoords, writeShape, outBytes*8, useUuid, crop, sparse);
#endif

    // zarrC is either a copy for data conversion or NULL
    free(zarrC);

    if(err) mexErrMsgIdAndTxt("zarr:writeError",Zarr.get_errString().c_str());
    
}
