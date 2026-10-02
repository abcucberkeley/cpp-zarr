#ifndef CPPZARR_ZARRMEXHELPERS_H
#define CPPZARR_ZARRMEXHELPERS_H
// Argument handling shared by the MEX functions for arrays of any number of
// dimensions. Existing 3D calls behave exactly as before; lower-dimensional
// arrays also accept the old 3D-style forms (extra trailing singleton axes).
#include <cstdint>
#include <string>
#include <vector>
#include "mex.h"

// Numeric vector argument (read as double, like mxGetPr)
static std::vector<uint64_t> mexVector(const mxArray* a){
    const size_t n = mxGetNumberOfElements(a);
    std::vector<uint64_t> v(n);
    const double* p = mxGetPr(a);
    for(size_t i = 0; i < n; i++) v[i] = (uint64_t)p[i];
    return v;
}

// Fit a per-axis option (chunks, subfolders, chunk_shape) to nDims axes. Fewer
// values are fine (the library fills in its defaults for the remaining axes); 3
// values are accepted for 1D/2D arrays (the old 3D form); otherwise more values
// than axes is an error.
static std::vector<uint64_t> mexFitAxes(const std::vector<uint64_t> &v, const uint64_t nDims, const char* name){
    if(v.size() <= nDims) return v;
    if(v.size() == 3 && nDims < 3) return std::vector<uint64_t>(v.begin(), v.begin()+nDims);
    mexErrMsgIdAndTxt("zarr:inputError","%s has %d values but the array has %d dimensions\n",
                      name, (int)v.size(), (int)nDims);
    return v;
}

// Parse a region given as [starts ends] (1-based starts, inclusive ends) for an
// array with nDims axes: 2*nDims values; or 2*M values with M > nDims whose extra
// axes are singleton (start 1, end 1); or, when the array's shape is given, 2*M
// values with M < nDims if the array's remaining axes have size 1 (e.g. a 2D
// region on a [m n 1] array). Returns false if the length does not fit.
static bool mexParseRegion(const std::vector<uint64_t> &b, const uint64_t nDims,
                           std::vector<uint64_t> &start, std::vector<uint64_t> &end,
                           const std::vector<uint64_t> &shape = std::vector<uint64_t>()){
    if(b.empty() || b.size()%2) return false;
    const uint64_t m = b.size()/2;
    for(uint64_t d = nDims; d < m; d++){
        if(b[d] != 1 || b[m+d] != 1) return false;
    }
    for(uint64_t d = m; d < nDims; d++){
        if(d >= shape.size() || shape[d] != 1) return false;
    }
    start.assign(nDims,0);
    end.assign(nDims,1);
    for(uint64_t d = 0; d < nDims && d < m; d++){
        if(b[d] < 1) mexErrMsgIdAndTxt("zarr:inputError","Lower bounds must be at least 1");
        start[d] = b[d]-1;
        end[d] = b[m+d];
    }
    return true;
}
// Report an error the zarr class threw (as a string) that the caller does not
// handle itself
static void mexZarrError(const std::string &e){
    const std::string detail = e.substr(e.find(':')+1);
    if(e.rfind("metadataUnsupported:", 0) == 0){
        mexErrMsgIdAndTxt("zarr:zarrayError","This zarr array cannot be read: %s\n",detail.c_str());
    }
    if(e.rfind("zarrV3NotWritable:", 0) == 0){
        mexErrMsgIdAndTxt("zarr:zarrayError","%s is a Zarr v3 array. Writing Zarr v3 arrays is not supported yet\n",detail.c_str());
    }
    mexErrMsgIdAndTxt("zarr:zarrayError","Unknown error occurred (%s)\n",e.c_str());
}
#endif
