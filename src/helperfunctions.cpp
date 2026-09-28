#ifdef _WIN32
#include <stdarg.h> 
#include <sys/time.h>
// For std::replace
#include <algorithm>
#else
#include <uuid/uuid.h>
#endif
#include <sys/stat.h>
#include <omp.h>
#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <cstring>
#include "helperfunctions.h"
#include "zarr.h"

#ifndef _WIN32
#include <wordexp.h>
// Expand the tilde to the home directory on Mac and Linux
const char* expandTilde(const char* path) {
    if(strchr(path,'~')){
        wordexp_t expPath;
        wordexp(path, &expPath, 0);
        return expPath.we_wordv[0];
    }
    else return path;
}
#endif
    
#ifdef _WIN32
char* strndup (const char *s, size_t n)
{
  size_t len = strnlen (s, n);
  char *newS = (char *) malloc (len + 1);
  if (newS == NULL)
    return NULL;
  newS[len] = '\0';
  return (char *) memcpy (newS, s, len);
}
int _vscprintf_so(const char * format, va_list pargs) {
    int retval;
    va_list argcopy;
    va_copy(argcopy, pargs);
    retval = vsnprintf(NULL, 0, format, argcopy);
    va_end(argcopy);
    return retval;
}

int vasprintf(char **strp, const char *fmt, va_list ap) {
    int len = _vscprintf_so(fmt, ap);
    if (len == -1) return -1;
    char *str = (char*)malloc((size_t) len + 1);
    if (!str) return -1;
    int r = vsnprintf(str, len + 1, fmt, ap); /* "secure" version of vsprintf */
    if (r == -1) return free(str), -1;
    *strp = str;
    return r;
}

int asprintf(char *strp[], const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int r = vasprintf(strp, fmt, ap);
    va_end(ap);
    return r;
}
#endif

std::string generateUUID(){
    #ifdef _WIN32
    // uuid of length 5 for windows
    char uuidC[6];
    char *seedArr = (char*)malloc(1000);
    struct timeval cSeed;
    gettimeofday(&cSeed,NULL);
    int nChars = sprintf(seedArr,"%d%d",cSeed.tv_sec,cSeed.tv_usec);
    int aSeed = 0;
    char* ptr;
    if(nChars > 9)
        aSeed = strtol(seedArr+nChars-10, &ptr, 9);
    else aSeed = strtol(seedArr, &ptr, 9);
    srand(aSeed);
    sprintf(uuidC,"%.5d",rand() % 99999);
    free(seedArr);
    #else
    uuid_t binuuid;
    uuid_generate_random(binuuid);
    char uuidC[37];
    uuid_unparse(binuuid, uuidC);
    #endif
    return std::string(uuidC);
}

bool folderExists(const std::string &folderName){
    struct stat info;

    if(stat(folderName.c_str(), &info) != 0)
        return false;
    else if(info.st_mode & S_IFDIR)
        return true;
    else
        return false;
}

// Recursively make a directory
void mkdirRecursive(const char *dir) {
    if(folderExists(dir)) return;

    std::string dirPath(dir);
    if(!dirPath.size()) return;

    // Convert all \\ to / if on Windows
    #ifdef _WIN32
    std::replace(dirPath.begin(), dirPath.end(), '\\', '/');
    #endif

    // If there is a slash at the end, remove it
    if(dirPath.back() == '/') dirPath.pop_back();
    
    for(size_t i = 0; i < dirPath.size(); i++){
        if(dirPath[i] == '/'){
            dirPath[i] = '\0';

            #ifdef _WIN32
            mkdir(dirPath.c_str());
            #else
            mkdir(dirPath.c_str(), 0777);
            #endif

            dirPath[i] = '/';
        }
    }

    #ifdef _WIN32
    mkdir(dirPath.c_str());
    #else
    mkdir(dirPath.c_str(), 0777);
    #endif
}

bool fileExists(const std::string &fileName){
    if (FILE *file = fopen(fileName.c_str(), "r")){
        fclose(file);
        return true;
    }
    else return false;
}

void makeDimensionFolders(const std::string &fileName){
    size_t lastSlash = fileName.find_last_of("/");
    mkdirRecursive(fileName.substr(0,lastSlash).c_str());
}

bool isLittleEndian(){
    uint16_t number = 1;
    return *reinterpret_cast<uint8_t*>(&number) == 1;
}

bool oppositeEndianness(const std::string &dtype){
    // '|' means byte order is not applicable (single-byte dtypes like |u1/|i1)
    if(dtype.empty() || dtype[0] == '|') return false;
    if(isLittleEndian()){
        if(dtype[0] == '<') return false;
    }
    else{
        if(dtype[0] == '>') return false;
    }
    return true;
}

int fillValueToInt(const std::string &fillValue){
    errno = 0;
    char* end = NULL;
    const long v = strtol(fillValue.c_str(), &end, 10);
    if(errno || end == fillValue.c_str() || v > INT_MAX || v < INT_MIN) return 0;
    return (int)v;
}

void swapArrayEndianness(void* array, const size_t elementSize, const size_t numElements){
    uint8_t* data = reinterpret_cast<uint8_t*>(array); // Cast array to byte pointer

    #pragma omp parallel for
    for (size_t i = 0; i < numElements; ++i) {
        uint8_t* elementPtr = data + i * elementSize; // Pointer to the current element
        std::reverse(elementPtr, elementPtr + elementSize); // Reverse bytes within the element
    }
}

// In-register transposes of small square blocks with portable vector extensions
// (SSE2 on x86, NEON on ARM); compilers without __builtin_shufflevector use the
// scalar path in copyBoxTiled.
#if defined(__has_builtin)
#if __has_builtin(__builtin_shufflevector)
#define CPPZARR_VECTOR_TRANSPOSE 1
#endif
#endif

#ifdef CPPZARR_VECTOR_TRANSPOSE
typedef uint8_t  czV16u8 __attribute__((vector_size(16)));
typedef uint16_t czV8u16 __attribute__((vector_size(16)));
typedef uint32_t czV4u32 __attribute__((vector_size(16)));
typedef uint64_t czV2u64 __attribute__((vector_size(16)));

// Interleave G-byte lanes from the low (Lo) or high (Hi) halves of a and b.
template <int G> static inline czV16u8 czUnpLo(czV16u8 a, czV16u8 b);
template <int G> static inline czV16u8 czUnpHi(czV16u8 a, czV16u8 b);
template <> inline czV16u8 czUnpLo<1>(czV16u8 a, czV16u8 b){ return __builtin_shufflevector(a, b, 0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23); }
template <> inline czV16u8 czUnpHi<1>(czV16u8 a, czV16u8 b){ return __builtin_shufflevector(a, b, 8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31); }
template <> inline czV16u8 czUnpLo<2>(czV16u8 a, czV16u8 b){ return (czV16u8)__builtin_shufflevector((czV8u16)a, (czV8u16)b, 0,8,1,9,2,10,3,11); }
template <> inline czV16u8 czUnpHi<2>(czV16u8 a, czV16u8 b){ return (czV16u8)__builtin_shufflevector((czV8u16)a, (czV8u16)b, 4,12,5,13,6,14,7,15); }
template <> inline czV16u8 czUnpLo<4>(czV16u8 a, czV16u8 b){ return (czV16u8)__builtin_shufflevector((czV4u32)a, (czV4u32)b, 0,4,1,5); }
template <> inline czV16u8 czUnpHi<4>(czV16u8 a, czV16u8 b){ return (czV16u8)__builtin_shufflevector((czV4u32)a, (czV4u32)b, 2,6,3,7); }
template <> inline czV16u8 czUnpLo<8>(czV16u8 a, czV16u8 b){ return (czV16u8)__builtin_shufflevector((czV2u64)a, (czV2u64)b, 0,2); }
template <> inline czV16u8 czUnpHi<8>(czV16u8 a, czV16u8 b){ return (czV16u8)__builtin_shufflevector((czV2u64)a, (czV2u64)b, 1,3); }

// One butterfly stage over N rows: pair rows (2p, 2p+1), low halves go to p and
// high halves to p+N/2. Stages at G = element size, 2x, ... up to 8 bytes leave
// column c of the block in row bitreverse(c).
template <int G, int N> static inline void czStage(czV16u8* r){
    czV16u8 t[N];
#pragma GCC unroll 16
    for(int p = 0; p < N/2; p++){
        t[p] = czUnpLo<G>(r[2*p], r[2*p+1]);
        t[p+N/2] = czUnpHi<G>(r[2*p], r[2*p+1]);
    }
#pragma GCC unroll 16
    for(int q = 0; q < N; q++) r[q] = t[q];
}

static constexpr int czBitRev(int v, int bits){ return bits == 0 ? 0 : ((v & 1) << (bits-1)) | czBitRev(v >> 1, bits-1); }

// Transpose one N x N block (N = 16/SZ elements of SZ bytes): rows of src (16
// contiguous bytes each, srcStride bytes apart) become rows of dst.
template <int SZ> static inline void czTransposeBlock(const uint8_t* src, const uint64_t srcStride,
                                                     uint8_t* dst, const uint64_t dstStride){
    constexpr int N = 16/SZ;
    constexpr int BITS = (N == 16) ? 4 : (N == 8) ? 3 : (N == 4) ? 2 : 1;
    czV16u8 r[N];
#pragma GCC unroll 16
    for(int q = 0; q < N; q++) memcpy(&r[q], src+q*srcStride, 16);
    czStage<SZ, N>(r);
    if(2*SZ <= 8) czStage<(2*SZ <= 8) ? 2*SZ : 8, N>(r);
    if(4*SZ <= 8) czStage<(4*SZ <= 8) ? 4*SZ : 8, N>(r);
    if(8*SZ <= 8) czStage<8, N>(r);
#pragma GCC unroll 16
    for(int q = 0; q < N; q++) memcpy(dst+czBitRev(q, BITS)*dstStride, &r[q], 16);
}
#endif

template <typename T>
static void copyBoxTiled(const T* __restrict src, T* __restrict dst,
                         const uint64_t n0, const uint64_t n1, const uint64_t n2,
                         const uint64_t s0, const uint64_t s1, const uint64_t s2,
                         const uint64_t d0, const uint64_t d1, const uint64_t d2){
    // Tile edge: four 64-byte cache lines of elements (two for 1-byte types, so
    // the tile buffer stays within L1)
    constexpr uint64_t TB = (sizeof(T) == 1) ? 128 : 256/sizeof(T);

    // Common case (all F<->C conversions): source contiguous along k, destination
    // contiguous along i. Each tile is staged through a small buffer so every
    // source and destination cache line is touched once, in a single burst.
    // Chunk strides are usually powers of two, which map a tile's rows onto the
    // same cache sets; working on them directly makes the rows evict each other.
    if(s2 == 1 && d0 == 1){
        T buf[TB*TB];
        for(uint64_t j = 0; j < n1; j++){
            const T* sj = src + j*s1;
            T* dj = dst + j*d1;
            for(uint64_t it = 0; it < n0; it += TB){
                const uint64_t in = (it+TB < n0) ? TB : n0-it;
                for(uint64_t kt = 0; kt < n2; kt += TB){
                    const uint64_t kn = (kt+TB < n2) ? TB : n2-kt;
                    if(kn == TB){
                        for(uint64_t ii = 0; ii < in; ii++) memcpy(buf+ii*TB, sj+(it+ii)*s0+kt, TB*sizeof(T));
                    }
                    else{
                        for(uint64_t ii = 0; ii < in; ii++) memcpy(buf+ii*TB, sj+(it+ii)*s0+kt, kn*sizeof(T));
                    }
#ifdef CPPZARR_VECTOR_TRANSPOSE
                    if(in == TB && kn == TB){
                        // Full tile: blocks of N x N elements (16 bytes per block row),
                        // each transposed in registers; a block row of destination
                        // lines is finished before moving on, so each line is
                        // written in one burst.
                        constexpr uint64_t N = 16/sizeof(T);
                        for(uint64_t kb = 0; kb < TB; kb += N){
                            for(uint64_t ib = 0; ib < TB; ib += N){
                                czTransposeBlock<sizeof(T)>((const uint8_t*)(buf+ib*TB+kb), TB*sizeof(T),
                                                            (uint8_t*)(dj+(kt+kb)*d2+it+ib), d2*sizeof(T));
                            }
                        }
                        continue;
                    }
#endif
                    for(uint64_t kk = 0; kk < kn; kk++){
                        T* d = dj+(kt+kk)*d2+it;
                        for(uint64_t ii = 0; ii < in; ii++) d[ii] = buf[ii*TB+kk];
                    }
                }
            }
        }
        return;
    }

    for(uint64_t j = 0; j < n1; j++){
        const T* sj = src + j*s1;
        T* dj = dst + j*d1;
        for(uint64_t kt = 0; kt < n2; kt += TB){
            const uint64_t ke = (kt+TB < n2) ? kt+TB : n2;
            for(uint64_t it = 0; it < n0; it += TB){
                const uint64_t ie = (it+TB < n0) ? it+TB : n0;
                for(uint64_t k = kt; k < ke; k++){
                    const T* s = sj + k*s2;
                    T* d = dj + k*d2;
                    for(uint64_t i = it; i < ie; i++) d[i*d0] = s[i*s0];
                }
            }
        }
    }
}

void copyBoxTransposed(const uint64_t bytes, const void* src, void* dst,
                       const uint64_t n0, const uint64_t n1, const uint64_t n2,
                       const uint64_t s0, const uint64_t s1, const uint64_t s2,
                       const uint64_t d0, const uint64_t d1, const uint64_t d2){
    switch(bytes){
        case 1: copyBoxTiled((const uint8_t*)src, (uint8_t*)dst, n0, n1, n2, s0, s1, s2, d0, d1, d2); break;
        case 2: copyBoxTiled((const uint16_t*)src, (uint16_t*)dst, n0, n1, n2, s0, s1, s2, d0, d1, d2); break;
        case 4: copyBoxTiled((const uint32_t*)src, (uint32_t*)dst, n0, n1, n2, s0, s1, s2, d0, d1, d2); break;
        case 8: copyBoxTiled((const uint64_t*)src, (uint64_t*)dst, n0, n1, n2, s0, s1, s2, d0, d1, d2); break;
    }
}

template <typename T>
static void fillTyped(void* dst, const uint64_t n, const void* elem){
    T v;
    memcpy(&v, elem, sizeof(T));
    std::fill_n((T*)dst, n, v);
}

void fillElements(const uint64_t bytes, void* dst, const uint64_t n, const void* elem){
    switch(bytes){
        case 1: fillTyped<uint8_t>(dst, n, elem); break;
        case 2: fillTyped<uint16_t>(dst, n, elem); break;
        case 4: fillTyped<uint32_t>(dst, n, elem); break;
        case 8: fillTyped<uint64_t>(dst, n, elem); break;
    }
}
