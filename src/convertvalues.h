#ifndef CPPZARR_CONVERTVALUES_H
#define CPPZARR_CONVERTVALUES_H
// Values converted between data types as MATLAB converts between classes, for data written into an array of
// another data type: the library converts with these the MATLAB writer's data (convertValueParts) and Python's
// write_zarr's (convertValues). Only the library compiles them: the mex files call those functions, so that
// they run no OpenMP of their own (see helperfunctions.h).
#include <cmath>
#include <cstdint>
#include <limits>
#include "helperfunctions.h"

// To an integer type: rounded to the nearest integer (halves away from zero) and saturated at the type's
// limits, NaN to 0
template <typename D> static inline D czRoundToInt(const double v){
    if(std::isnan(v)) return 0;
    const double r = std::round(v);
    if(r <= (double)std::numeric_limits<D>::min()) return std::numeric_limits<D>::min();
    if(r >= (double)std::numeric_limits<D>::max()) return std::numeric_limits<D>::max();
    return (D)r;
}
template <typename D> static inline D czSignedToInt(const int64_t v){
    if(std::numeric_limits<D>::is_signed){
        if(v < (int64_t)std::numeric_limits<D>::min()) return std::numeric_limits<D>::min();
        if(v > (int64_t)std::numeric_limits<D>::max()) return std::numeric_limits<D>::max();
        return (D)v;
    }
    if(v < 0) return 0;
    return (uint64_t)v > (uint64_t)std::numeric_limits<D>::max() ? std::numeric_limits<D>::max() : (D)v;
}
template <typename D> static inline D czUnsignedToInt(const uint64_t v){
    return v > (uint64_t)std::numeric_limits<D>::max() ? std::numeric_limits<D>::max() : (D)v;
}
template <typename S, typename D> static inline D czToInt(const S v){
    if(!std::numeric_limits<S>::is_integer) return czRoundToInt<D>((double)v);
    if(std::numeric_limits<S>::is_signed) return czSignedToInt<D>((int64_t)v);
    return czUnsignedToInt<D>((uint64_t)v);
}

template <typename S, typename D>
static void czConvertInts(const S* re, const uint64_t stride, const int64_t n, void* out){
    D* o = (D*)out;
    #pragma omp parallel for
    for(int64_t i = 0; i < n; i++) o[i] = czToInt<S, D>(re[i*stride]);
}

// n elements (real parts re, imaginary parts im or NULL, stride elements apart) converted to the data type t;
// any nonzero value is true
template <typename S>
static void czConvertValues(const S* re, const S* im, const uint64_t stride, const uint64_t n, const DtypeInfo &t, void* out){
    const int64_t N = (int64_t)n;
    if(t.kind == 'b'){
        uint8_t* o = (uint8_t*)out;
        #pragma omp parallel for
        for(int64_t i = 0; i < N; i++) o[i] = re[i*stride] != 0 || (im && im[i*stride] != 0);
    }
    else if(t.kind == 'u'){
        if(t.bytes == 1) czConvertInts<S, uint8_t>(re, stride, N, out);
        else if(t.bytes == 2) czConvertInts<S, uint16_t>(re, stride, N, out);
        else if(t.bytes == 4) czConvertInts<S, uint32_t>(re, stride, N, out);
        else czConvertInts<S, uint64_t>(re, stride, N, out);
    }
    else if(t.kind == 'i'){
        if(t.bytes == 1) czConvertInts<S, int8_t>(re, stride, N, out);
        else if(t.bytes == 2) czConvertInts<S, int16_t>(re, stride, N, out);
        else if(t.bytes == 4) czConvertInts<S, int32_t>(re, stride, N, out);
        else czConvertInts<S, int64_t>(re, stride, N, out);
    }
    else if(t.kind == 'f' && t.bytes == 2){
        uint16_t* o = (uint16_t*)out;
        const bool bf = t.bfloat16;
        #pragma omp parallel for
        for(int64_t i = 0; i < N; i++) o[i] = bf ? floatToBfloat16((float)re[i*stride]) : doubleToHalf((double)re[i*stride]);
    }
    else if(t.kind == 'f' && t.bytes == 4){
        float* o = (float*)out;
        #pragma omp parallel for
        for(int64_t i = 0; i < N; i++) o[i] = (float)re[i*stride];
    }
    else if(t.kind == 'f'){
        double* o = (double*)out;
        #pragma omp parallel for
        for(int64_t i = 0; i < N; i++) o[i] = (double)re[i*stride];
    }
    else if(t.kind == 'c' && t.bytes == 8){
        float* o = (float*)out;
        #pragma omp parallel for
        for(int64_t i = 0; i < N; i++){
            o[2*i] = (float)re[i*stride];
            o[2*i+1] = im ? (float)im[i*stride] : 0.0f;
        }
    }
    else if(t.kind == 'c'){
        double* o = (double*)out;
        #pragma omp parallel for
        for(int64_t i = 0; i < N; i++){
            o[2*i] = (double)re[i*stride];
            o[2*i+1] = im ? (double)im[i*stride] : 0.0;
        }
    }
}
#endif
