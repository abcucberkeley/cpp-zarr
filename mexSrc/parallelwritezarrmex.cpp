#include <cstdint>
#include <cstring>
#include "mex.h"
#include "../src/helperfunctions.h"
#include "../src/parallelwritezarr.h"
#include "../src/parallelreadzarr.h"
#include "../src/zarr.h"

//compile
//mex -v COPTIMFLAGS="-DNDEBUG -O3" CFLAGS='$CFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp -O3' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/lib' -lblosc '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/lib64' -lblosc2 '-L/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/lib64' -lcjson -luuid parallelWriteZarr.c helperFunctions.c parallelReadZarr.c

//With zlib
//mex -v COPTIMFLAGS="-DNDEBUG -O3" LDOPTIMFLAGS="-Wl',-rpath='''$ORIGIN'''' -O3 -DNDEBUG" CFLAGS='$CFLAGS -fopenmp -O3' LDFLAGS='$LDFLAGS -fopenmp -O3' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/include/' '-I/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/zarr/lib' -lblosc '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/lib64' -lblosc2 '-L/global/home/groups/software/sl-7.x86_64/modules/cJSON/1.7.15/lib64' -lcjson -luuid -lz parallelWriteZarr.c helperFunctions.c parallelReadZarr.c

//mex -v COPTIMFLAGS="-O3 -fwrapv -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-I/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/include/' '-L/global/home/groups/software/sl-7.x86_64/modules/cBlosc/2.0.4/lib64' -lblosc2 zarrMex.c
//
//Windows
//mex -v COPTIMFLAGS="-O3 -DNDEBUG" CFLAGS='$CFLAGS -O3 -fopenmp' LDFLAGS='$LDFLAGS -O3 -fopenmp' '-IC:\Program Files (x86)\bloscZarr\include' '-LC:\Program Files (x86)\bloscZarr\lib' -lblosc '-IC:\Program Files (x86)\cJSON\include\' '-LC:\Program Files (x86)\cJSON\lib' -lcjson '-IC:\Program Files (x86)\blosc\include' '-LC:\Program Files (x86)\blosc\lib' -lblosc2 parallelWriteZarr.c parallelReadZarr.c helperFunctions.c

// Cast-copy between any two supported element types, used when writing into an
// existing zarr file whose dtype differs from the MATLAB input's type.
template <typename TDst, typename TSrc>
static void castCopy(void* dstV, const void* srcV, const uint64_t n){
    TDst* dst = (TDst*)dstV;
    const TSrc* src = (const TSrc*)srcV;
    #pragma omp parallel for
    for(uint64_t i = 0; i < n; i++){
        dst[i] = (TDst)src[i];
    }
}

template <typename TSrc>
static void* castToDtype(const std::string &dtype, const void* src, const uint64_t n){
    const char kind = dtype.size() == 3 ? dtype[1] : '\0';
    const char dsize = dtype.size() == 3 ? dtype[2] : '\0';
    void* dst = NULL;
    if(kind == 'u'){
        if(dsize == '1'){ dst = malloc(n*sizeof(uint8_t));  castCopy<uint8_t,TSrc>(dst,src,n); }
        else if(dsize == '2'){ dst = malloc(n*sizeof(uint16_t)); castCopy<uint16_t,TSrc>(dst,src,n); }
        else if(dsize == '4'){ dst = malloc(n*sizeof(uint32_t)); castCopy<uint32_t,TSrc>(dst,src,n); }
        else if(dsize == '8'){ dst = malloc(n*sizeof(uint64_t)); castCopy<uint64_t,TSrc>(dst,src,n); }
    }
    else if(kind == 'i'){
        if(dsize == '1'){ dst = malloc(n*sizeof(int8_t));  castCopy<int8_t,TSrc>(dst,src,n); }
        else if(dsize == '2'){ dst = malloc(n*sizeof(int16_t)); castCopy<int16_t,TSrc>(dst,src,n); }
        else if(dsize == '4'){ dst = malloc(n*sizeof(int32_t)); castCopy<int32_t,TSrc>(dst,src,n); }
        else if(dsize == '8'){ dst = malloc(n*sizeof(int64_t)); castCopy<int64_t,TSrc>(dst,src,n); }
    }
    else if(kind == 'f'){
        if(dsize == '4'){ dst = malloc(n*sizeof(float));  castCopy<float,TSrc>(dst,src,n); }
        else if(dsize == '8'){ dst = malloc(n*sizeof(double)); castCopy<double,TSrc>(dst,src,n); }
    }
    return dst;
}

// Convert a MATLAB array's data to the given zarr dtype. Returns a malloc'd
// buffer the caller frees, or NULL if either side is unsupported.
static void* convertMxToDtype(const std::string &dtype, const mxArray* arr, const uint64_t n){
    switch(mxGetClassID(arr)){
        case mxUINT8_CLASS:  return castToDtype<uint8_t>(dtype, mxGetData(arr), n);
        case mxINT8_CLASS:   return castToDtype<int8_t>(dtype, mxGetData(arr), n);
        case mxUINT16_CLASS: return castToDtype<uint16_t>(dtype, mxGetData(arr), n);
        case mxINT16_CLASS:  return castToDtype<int16_t>(dtype, mxGetData(arr), n);
        case mxUINT32_CLASS: return castToDtype<uint32_t>(dtype, mxGetData(arr), n);
        case mxINT32_CLASS:  return castToDtype<int32_t>(dtype, mxGetData(arr), n);
        case mxUINT64_CLASS: return castToDtype<uint64_t>(dtype, mxGetData(arr), n);
        case mxINT64_CLASS:  return castToDtype<int64_t>(dtype, mxGetData(arr), n);
        case mxSINGLE_CLASS: return castToDtype<float>(dtype, mxGetData(arr), n);
        case mxDOUBLE_CLASS: return castToDtype<double>(dtype, mxGetData(arr), n);
        default: return NULL;
    }
}

// TODO: FIX MEMORY LEAKS
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    if(nrhs < 2) mexErrMsgIdAndTxt("zarr:inputError","This functions requires at least 2 arguments");
    if(!mxIsChar(prhs[0])) mexErrMsgIdAndTxt("zarr:inputError","The first argument must be a string");
    if(mxIsEmpty(prhs[1])) mexErrMsgIdAndTxt("zarr:inputError","All input data axes must be of at least size 1");

    std::vector<uint64_t> startCoords = {0,0,0};
    std::vector<uint64_t> endCoords = {0,0,0};
    bool crop = false;
    bool useUuid = true;
    uint8_t bboxIndex = 0;
    bool sparse = true;

    // Dims are 1 by default
    uint64_t iDims[3] = {1,1,1};

    std::string folderName(mxArrayToString(prhs[0]));
    // Handle the tilde character in filenames on Linux/Mac
    #ifndef _WIN32
    folderName = expandTilde(folderName.c_str());
    #endif
    
    // Check if metadata exists that we can use or if we have to create new metadata
    zarr Zarr;
    if(!fileExists(folderName+"/.zarray")){
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
                mexErrMsgIdAndTxt("zarr:zarrayError","Metadata is incomplete. Check the .zarray file");
            }
            else mexErrMsgIdAndTxt("zarr:zarrayError","Unknown error occurred\n");
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
            else if(mxGetN(prhs[i+1]) == 6){
                crop = true;
                startCoords[0] = (uint64_t)*(mxGetPr(prhs[i+1]))-1;
                startCoords[1] = (uint64_t)*((mxGetPr(prhs[i+1])+1))-1;
                startCoords[2] = (uint64_t)*((mxGetPr(prhs[i+1])+2))-1;
                endCoords[0] = (uint64_t)*((mxGetPr(prhs[i+1])+3));
                endCoords[1] = (uint64_t)*((mxGetPr(prhs[i+1])+4));
                endCoords[2] = (uint64_t)*((mxGetPr(prhs[i+1])+5));
    
    
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
            else if(mxGetN(prhs[i+1]) == 3) bboxIndex = i+1; 
            else mexErrMsgIdAndTxt("zarr:inputError","Input range is not 6 or 3");
        }
        else if(currInput == "cname"){
            if(!mxIsChar(prhs[i+1])) mexErrMsgIdAndTxt("zarr:inputError","cname must be a string\n");
            Zarr.set_cname(mxArrayToString(prhs[i+1]));
        }
        else if(currInput == "subfolders"){
            if(mxGetN(prhs[i+1]) != 3) mexErrMsgIdAndTxt("zarr:inputError","subfolders must be an array of 3 numbers\n");
            Zarr.set_subfolders({(uint64_t)*(mxGetPr(prhs[i+1])),
                               (uint64_t)*((mxGetPr(prhs[i+1])+1)),
                               (uint64_t)*((mxGetPr(prhs[i+1])+2))});
        }
        else if(currInput == "chunks"){
            if(mxGetN(prhs[i+1]) != 3) mexErrMsgIdAndTxt("zarr:inputError","chunks must be an array of 3 numbers\n");
            Zarr.set_chunks({(uint64_t)*(mxGetPr(prhs[i+1])),
                             (uint64_t)*((mxGetPr(prhs[i+1])+1)),
                             (uint64_t)*((mxGetPr(prhs[i+1])+2))});
        }
        else if(currInput == "chunk_shape"){
            if(mxGetN(prhs[i+1]) != 3) mexErrMsgIdAndTxt("zarr:inputError","chunk_shape must be an array of 3 numbers\n");
            Zarr.set_shard(true);
            Zarr.set_chunk_shape({(uint64_t)*(mxGetPr(prhs[i+1])),
                               (uint64_t)*((mxGetPr(prhs[i+1])+1)),
                               (uint64_t)*((mxGetPr(prhs[i+1])+2))});
        }
        else if(currInput == "sparse"){
            sparse = (bool)*((mxGetPr(prhs[i+1])));
        }
        else if(currInput == "dimension_separator"){
            if(!mxIsChar(prhs[i+1])) mexErrMsgIdAndTxt("zarr:inputError","dimension_separator must be a string\n");
            const std::string dimension_separator(mxArrayToString(prhs[i+1]));
            if(dimension_separator != "." && dimension_separator != "/") mexErrMsgIdAndTxt("zarr:inputError","dimension_separator must be a . or /\n");
            Zarr.set_dimension_separator(dimension_separator);
        }
        else{
            mexErrMsgIdAndTxt("zarr:inputError","The argument \"%s\" does not match the name of any supported input name.\n \
            Currently Supported Names: uuid, bbox, cname, subfolders, chunks, chunk_shape, sparse\n",currInput.c_str());
        }
    }

    void* zarrC = NULL;

    mxClassID mDType = mxGetClassID(prhs[1]);
    switch(mDType){
        case mxUINT8_CLASS:  Zarr.set_dtype("<u1"); break;
        case mxINT8_CLASS:   Zarr.set_dtype("<i1"); break;
        case mxUINT16_CLASS: Zarr.set_dtype("<u2"); break;
        case mxINT16_CLASS:  Zarr.set_dtype("<i2"); break;
        case mxUINT32_CLASS: Zarr.set_dtype("<u4"); break;
        case mxINT32_CLASS:  Zarr.set_dtype("<i4"); break;
        case mxUINT64_CLASS: Zarr.set_dtype("<u8"); break;
        case mxINT64_CLASS:  Zarr.set_dtype("<i8"); break;
        case mxSINGLE_CLASS: Zarr.set_dtype("<f4"); break;
        case mxDOUBLE_CLASS: Zarr.set_dtype("<f8"); break;
        default: mexErrMsgIdAndTxt("zarr:dataTypeError","The input data type is not supported");
    }

    if(!crop){
        uint64_t nDims = (uint64_t)mxGetNumberOfDimensions(prhs[1]);
        if(nDims < 2 || nDims > 3) mexErrMsgIdAndTxt("zarr:inputError","Input data must be 2D or 3D");

        uint64_t* dims = (uint64_t*)mxGetDimensions(prhs[1]);
        if(nDims == 3) Zarr.set_shape({dims[0],dims[1],dims[2]});
        else Zarr.set_shape({dims[0],dims[1],1});
        if(bboxIndex){
            Zarr.set_chunks({(uint64_t)*(mxGetPr(prhs[bboxIndex])),
                            (uint64_t)*((mxGetPr(prhs[bboxIndex])+1)),
                            (uint64_t)*((mxGetPr(prhs[bboxIndex])+2))});
        }
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
            else mexErrMsgIdAndTxt("zarr:zarrayError","Unknown error occurred\n");
        }
    }
    else{
        Zarr.set_shape({endCoords[0],endCoords[1],endCoords[2]});

        if(fileExists(folderName+"/.zarray")){
            if(endCoords[0]-startCoords[0] != iDims[0] ||
               endCoords[1]-startCoords[1] != iDims[1] ||
               endCoords[2]-startCoords[2] != iDims[2]) mexErrMsgIdAndTxt("zarr:inputError","Bounding box size does not match the size of the input data");
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
                else mexErrMsgIdAndTxt("zarr:zarrayError","Unknown error occurred\n");
            }
        }

        const std::string dtypeT(Zarr.get_dtype());
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
        
        if(dtypeT != Zarr.get_dtype()){
            const uint64_t size = (endCoords[0]-startCoords[0])*
                (endCoords[1]-startCoords[1])*
                (endCoords[2]-startCoords[2]);
            zarrC = convertMxToDtype(Zarr.get_dtype(), prhs[1], size);
            if(!zarrC) mexErrMsgIdAndTxt("zarr:dataTypeError","Cannot convert the input data to the existing file's data type \"%s\"",Zarr.get_dtype().c_str());
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
    //printf("%s startCoords[0]yz: %d %d %d endCoords[0]yz: %d %d %d chunkxyz: %d %d %d writeShape[0]yz: %d %d %d\n",Zarr.get_fileName().c_str(),startCoords[0],startCoords[1],startCoords[2],endCoords[0],endCoords[1],endCoords[2],Zarr.get_chunks(0),Zarr.get_chunks(1),Zarr.get_chunks(2),writeShape[0],writeShape[1],writeShape[2]);

    Zarr.set_chunkInfo(startCoords, endCoords);
    bool err = 0;
    // The write machinery is element-width based, so one generic path covers
    // every supported dtype: signed/unsigned 8/16/32/64-bit ints, 32/64-bit floats.
    const std::string &dtypeOut = Zarr.get_dtype();
    const uint64_t outBytes = Zarr.dtypeBytes();
    const char outKind = dtypeOut.size() == 3 ? dtypeOut[1] : '\0';
    if(!outBytes || (outKind != 'u' && outKind != 'i' && outKind != 'f') ||
       (outKind == 'f' && outBytes < 4)){
        free(zarrC);
        mexErrMsgIdAndTxt("zarr:dataTypeError","Data type \"%s\" is not supported",dtypeOut.c_str());
    }
    void* zarrArr = zarrC ? zarrC : mxGetData(prhs[1]);
    err = parallelWriteZarr(Zarr, zarrArr, startCoords, endCoords, writeShape, outBytes*8, useUuid, crop, sparse);

    // zarrC is either a copy for data conversion or NULL
    free(zarrC);

    if(err) mexErrMsgIdAndTxt("zarr:writeError",Zarr.get_errString().c_str());
    
}