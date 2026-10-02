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
#include <cstdio>
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

// CRC32C lookup table (reflected Castagnoli polynomial 0x82F63B78), one byte at a
// time. A constant array, not a table built on first use: a function-local static
// built at run time needs thread-safe initialization, which crashes on Windows
// (MinGW, static libstdc++ in the DLL) when OpenMP threads reach it at once.
static const uint32_t crc32cTable[256] = {
    0x00000000, 0xF26B8303, 0xE13B70F7, 0x1350F3F4, 0xC79A971F, 0x35F1141C, 0x26A1E7E8, 0xD4CA64EB,
    0x8AD958CF, 0x78B2DBCC, 0x6BE22838, 0x9989AB3B, 0x4D43CFD0, 0xBF284CD3, 0xAC78BF27, 0x5E133C24,
    0x105EC76F, 0xE235446C, 0xF165B798, 0x030E349B, 0xD7C45070, 0x25AFD373, 0x36FF2087, 0xC494A384,
    0x9A879FA0, 0x68EC1CA3, 0x7BBCEF57, 0x89D76C54, 0x5D1D08BF, 0xAF768BBC, 0xBC267848, 0x4E4DFB4B,
    0x20BD8EDE, 0xD2D60DDD, 0xC186FE29, 0x33ED7D2A, 0xE72719C1, 0x154C9AC2, 0x061C6936, 0xF477EA35,
    0xAA64D611, 0x580F5512, 0x4B5FA6E6, 0xB93425E5, 0x6DFE410E, 0x9F95C20D, 0x8CC531F9, 0x7EAEB2FA,
    0x30E349B1, 0xC288CAB2, 0xD1D83946, 0x23B3BA45, 0xF779DEAE, 0x05125DAD, 0x1642AE59, 0xE4292D5A,
    0xBA3A117E, 0x4851927D, 0x5B016189, 0xA96AE28A, 0x7DA08661, 0x8FCB0562, 0x9C9BF696, 0x6EF07595,
    0x417B1DBC, 0xB3109EBF, 0xA0406D4B, 0x522BEE48, 0x86E18AA3, 0x748A09A0, 0x67DAFA54, 0x95B17957,
    0xCBA24573, 0x39C9C670, 0x2A993584, 0xD8F2B687, 0x0C38D26C, 0xFE53516F, 0xED03A29B, 0x1F682198,
    0x5125DAD3, 0xA34E59D0, 0xB01EAA24, 0x42752927, 0x96BF4DCC, 0x64D4CECF, 0x77843D3B, 0x85EFBE38,
    0xDBFC821C, 0x2997011F, 0x3AC7F2EB, 0xC8AC71E8, 0x1C661503, 0xEE0D9600, 0xFD5D65F4, 0x0F36E6F7,
    0x61C69362, 0x93AD1061, 0x80FDE395, 0x72966096, 0xA65C047D, 0x5437877E, 0x4767748A, 0xB50CF789,
    0xEB1FCBAD, 0x197448AE, 0x0A24BB5A, 0xF84F3859, 0x2C855CB2, 0xDEEEDFB1, 0xCDBE2C45, 0x3FD5AF46,
    0x7198540D, 0x83F3D70E, 0x90A324FA, 0x62C8A7F9, 0xB602C312, 0x44694011, 0x5739B3E5, 0xA55230E6,
    0xFB410CC2, 0x092A8FC1, 0x1A7A7C35, 0xE811FF36, 0x3CDB9BDD, 0xCEB018DE, 0xDDE0EB2A, 0x2F8B6829,
    0x82F63B78, 0x709DB87B, 0x63CD4B8F, 0x91A6C88C, 0x456CAC67, 0xB7072F64, 0xA457DC90, 0x563C5F93,
    0x082F63B7, 0xFA44E0B4, 0xE9141340, 0x1B7F9043, 0xCFB5F4A8, 0x3DDE77AB, 0x2E8E845F, 0xDCE5075C,
    0x92A8FC17, 0x60C37F14, 0x73938CE0, 0x81F80FE3, 0x55326B08, 0xA759E80B, 0xB4091BFF, 0x466298FC,
    0x1871A4D8, 0xEA1A27DB, 0xF94AD42F, 0x0B21572C, 0xDFEB33C7, 0x2D80B0C4, 0x3ED04330, 0xCCBBC033,
    0xA24BB5A6, 0x502036A5, 0x4370C551, 0xB11B4652, 0x65D122B9, 0x97BAA1BA, 0x84EA524E, 0x7681D14D,
    0x2892ED69, 0xDAF96E6A, 0xC9A99D9E, 0x3BC21E9D, 0xEF087A76, 0x1D63F975, 0x0E330A81, 0xFC588982,
    0xB21572C9, 0x407EF1CA, 0x532E023E, 0xA145813D, 0x758FE5D6, 0x87E466D5, 0x94B49521, 0x66DF1622,
    0x38CC2A06, 0xCAA7A905, 0xD9F75AF1, 0x2B9CD9F2, 0xFF56BD19, 0x0D3D3E1A, 0x1E6DCDEE, 0xEC064EED,
    0xC38D26C4, 0x31E6A5C7, 0x22B65633, 0xD0DDD530, 0x0417B1DB, 0xF67C32D8, 0xE52CC12C, 0x1747422F,
    0x49547E0B, 0xBB3FFD08, 0xA86F0EFC, 0x5A048DFF, 0x8ECEE914, 0x7CA56A17, 0x6FF599E3, 0x9D9E1AE0,
    0xD3D3E1AB, 0x21B862A8, 0x32E8915C, 0xC083125F, 0x144976B4, 0xE622F5B7, 0xF5720643, 0x07198540,
    0x590AB964, 0xAB613A67, 0xB831C993, 0x4A5A4A90, 0x9E902E7B, 0x6CFBAD78, 0x7FAB5E8C, 0x8DC0DD8F,
    0xE330A81A, 0x115B2B19, 0x020BD8ED, 0xF0605BEE, 0x24AA3F05, 0xD6C1BC06, 0xC5914FF2, 0x37FACCF1,
    0x69E9F0D5, 0x9B8273D6, 0x88D28022, 0x7AB90321, 0xAE7367CA, 0x5C18E4C9, 0x4F48173D, 0xBD23943E,
    0xF36E6F75, 0x0105EC76, 0x12551F82, 0xE03E9C81, 0x34F4F86A, 0xC69F7B69, 0xD5CF889D, 0x27A40B9E,
    0x79B737BA, 0x8BDCB4B9, 0x988C474D, 0x6AE7C44E, 0xBE2DA0A5, 0x4C4623A6, 0x5F16D052, 0xAD7D5351
};

uint32_t crc32c(const uint8_t* data, size_t length){
    uint32_t crc = 0xFFFFFFFF;
    for(size_t i = 0; i < length; i++) crc = (crc >> 8) ^ crc32cTable[(crc ^ data[i]) & 0xFF];
    return ~crc;
}

bool fillValueIsZero(const std::string &fillValue){
    char* end = NULL;
    const double v = strtod(fillValue.c_str(), &end);
    return end != fillValue.c_str() && *end == '\0' && v == 0.0;
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
    // Elements whose bytes are all equal (e.g. 0) are a plain memset
    const uint8_t* e = (const uint8_t*)elem;
    bool uniform = true;
    for(uint64_t b = 1; b < bytes; b++) uniform = uniform && e[b] == e[0];
    if(uniform){
        memset(dst, e[0], n*bytes);
        return;
    }
    switch(bytes){
        case 1: fillTyped<uint8_t>(dst, n, elem); break;
        case 2: fillTyped<uint16_t>(dst, n, elem); break;
        case 4: fillTyped<uint32_t>(dst, n, elem); break;
        case 8: fillTyped<uint64_t>(dst, n, elem); break;
    }
}

// Zarr puts no limit on the number of dimensions. The N-D routines keep their
// per-axis scratch arrays on the stack for up to CZ_FAST_DIMS dimensions and on
// the heap beyond that.
#define CZ_FAST_DIMS 64
struct CzScratch {
    uint64_t local[CZ_FAST_DIMS*12];
    std::vector<uint64_t> heap;
    // count arrays of n values each, zero-initialized
    uint64_t* get(const uint64_t n, const uint64_t count){
        uint64_t* p = local;
        if(n > CZ_FAST_DIMS){
            heap.assign(n*count, 0);
            p = heap.data();
        }
        else std::fill(local, local+n*count, 0);
        return p;
    }
};

// Copy one element-strided box with a loop over every element (used only for
// layouts that have no contiguous axis to exploit)
static void copyBoxElementwise(const uint64_t bytes, const uint8_t* src, uint8_t* dst, const uint64_t m,
                               const uint64_t* ext, const uint64_t* ss, const uint64_t* ds){
    CzScratch scratch;
    uint64_t* idx = scratch.get(m, 1);
    uint64_t sOff = 0, dOff = 0;
    while(true){
        memcpy(dst+dOff*bytes, src+sOff*bytes, bytes);
        uint64_t l = 0;
        for(; l < m; l++){
            idx[l]++; sOff += ss[l]; dOff += ds[l];
            if(idx[l] < ext[l]) break;
            sOff -= ext[l]*ss[l]; dOff -= ext[l]*ds[l]; idx[l] = 0;
        }
        if(l == m) return;
    }
}

void copyBoxND(const uint64_t bytes, const void* src, void* dst,
               const std::vector<uint64_t> &extents,
               const std::vector<uint64_t> &srcStrides,
               const std::vector<uint64_t> &dstStrides,
               const bool spreadWrites){
    // Per-axis scratch: extents and strides of the axes that move, then the
    // loop bookkeeping below
    const uint64_t n = extents.size();
    CzScratch scratch;
    uint64_t* buf = scratch.get(n, 11);
    uint64_t *ext = buf, *ss = buf+n, *ds = buf+2*n, *used = buf+3*n, *idx = buf+4*n;
    uint64_t *oe = buf+5*n, *os = buf+6*n, *od = buf+7*n;
    uint64_t *me = buf+8*n, *ms = buf+9*n, *md = buf+10*n;

    // Axes of extent 1 do not move either pointer, so drop them (remaining axes
    // stay in their original order)
    uint64_t m = 0;
    for(uint64_t d = 0; d < n; d++){
        if(extents[d] == 0) return;
        if(extents[d] == 1) continue;
        ext[m] = extents[d]; ss[m] = srcStrides[d]; ds[m] = dstStrides[d]; m++;
    }
    const uint8_t* s = (const uint8_t*)src;
    uint8_t* dp = (uint8_t*)dst;
    if(m == 0){
        memcpy(dp, s, bytes);
        return;
    }

    // Axes that are contiguous in the source (a) and in the destination (b)
    int64_t a = -1, b = -1;
    for(uint64_t d = 0; d < m; d++){
        if(a < 0 && ss[d] == 1) a = d;
        if(b < 0 && ds[d] == 1) b = d;
    }

    if(a >= 0 && a == b){
        // Same contiguous axis on both sides: copy contiguous runs, growing the
        // run with any axis that continues it on both sides
        used[a] = 1;
        uint64_t run = ext[a];
        for(bool merged = true; merged;){
            merged = false;
            for(uint64_t d = 0; d < m; d++){
                if(!used[d] && ss[d] == run && ds[d] == run){
                    run *= ext[d];
                    used[d] = 1;
                    merged = true;
                }
            }
        }
        // Remaining axes, innermost first: the smallest destination stride, or
        // with spreadWrites the largest
        uint64_t k = 0;
        for(uint64_t d = 0; d < m; d++){
            if(used[d]) continue;
            uint64_t p = k++;
            if(spreadWrites){
                while(p > 0 && od[p-1] < ds[d]){ oe[p] = oe[p-1]; os[p] = os[p-1]; od[p] = od[p-1]; p--; }
            }
            else{
                while(p > 0 && od[p-1] > ds[d]){ oe[p] = oe[p-1]; os[p] = os[p-1]; od[p] = od[p-1]; p--; }
            }
            oe[p] = ext[d]; os[p] = ss[d]; od[p] = ds[d];
        }
        const uint64_t runBytes = run*bytes;
        if(k == 0){
            memcpy(dp, s, runBytes);
            return;
        }
        uint64_t sOff = 0, dOff = 0;
        while(true){
            const uint8_t* sp = s+sOff*bytes;
            uint8_t* dq = dp+dOff*bytes;
            for(uint64_t i = 0; i < oe[0]; i++){
                memcpy(dq, sp, runBytes);
                sp += os[0]*bytes;
                dq += od[0]*bytes;
            }
            uint64_t l = 1;
            for(; l < k; l++){
                idx[l]++; sOff += os[l]; dOff += od[l];
                if(idx[l] < oe[l]) break;
                sOff -= oe[l]*os[l]; dOff -= oe[l]*od[l]; idx[l] = 0;
            }
            if(l == k) return;
        }
    }

    // Different layouts (F <-> C): transpose tiles of the plane spanned by the
    // destination-contiguous axis (i) and the source-contiguous axis (k). Without
    // a contiguous axis on a side, use that side's smallest stride instead.
    uint64_t i = 0, kk = 0;
    if(b >= 0) i = b;
    else for(uint64_t d = 1; d < m; d++) if(ds[d] < ds[i]) i = d;
    if(a >= 0) kk = a;
    else{
        kk = (i == 0 && m > 1) ? 1 : 0;
        for(uint64_t d = 0; d < m; d++) if(d != i && ss[d] < ss[kk]) kk = d;
    }
    if(i == kk || m == 1){
        copyBoxElementwise(bytes, s, dp, m, ext, ss, ds);
        return;
    }
    // Middle axes: the one with the smallest destination stride is the kernel's
    // plane loop (j); any others are iterated here
    uint64_t k = 0;
    for(uint64_t d = 0; d < m; d++){
        if(d == i || d == kk) continue;
        uint64_t p = k++;
        while(p > 0 && md[p-1] > ds[d]){ me[p] = me[p-1]; ms[p] = ms[p-1]; md[p] = md[p-1]; p--; }
        me[p] = ext[d]; ms[p] = ss[d]; md[p] = ds[d];
    }
    const uint64_t n1 = k ? me[0] : 1, s1 = k ? ms[0] : 0, d1 = k ? md[0] : 0;
    uint64_t sOff = 0, dOff = 0;
    while(true){
        copyBoxTransposed(bytes, s+sOff*bytes, dp+dOff*bytes, ext[i], n1, ext[kk],
                          ss[i], s1, ss[kk], ds[i], d1, ds[kk]);
        uint64_t l = 1;
        for(; l < k; l++){
            idx[l]++; sOff += ms[l]; dOff += md[l];
            if(idx[l] < me[l]) break;
            sOff -= me[l]*ms[l]; dOff -= me[l]*md[l]; idx[l] = 0;
        }
        if(l >= k) return;
    }
}

// Fill the part of a packed buffer outside [lo, hi), from the outermost axis in:
// the slabs before and after the box along axis order[level] are contiguous
// blocks, and inside the box range the next axis is handled the same way.
static void fillOutsideRec(const uint64_t bytes, uint8_t* base, const uint64_t level, const uint64_t m,
                           const uint64_t* order, const std::vector<uint64_t> &full,
                           const std::vector<uint64_t> &strides, const std::vector<uint64_t> &lo,
                           const std::vector<uint64_t> &hi, const void* elem){
    const uint64_t d = order[level];
    const uint64_t blk = strides[d];
    if(lo[d] > 0) fillElements(bytes, base, lo[d]*blk, elem);
    if(hi[d] < full[d]) fillElements(bytes, base+hi[d]*blk*bytes, (full[d]-hi[d])*blk, elem);
    if(level+1 < m){
        for(uint64_t i = lo[d]; i < hi[d]; i++){
            fillOutsideRec(bytes, base+i*blk*bytes, level+1, m, order, full, strides, lo, hi, elem);
        }
    }
}

void fillOutsideBoxND(const uint64_t bytes, void* buf,
                      const std::vector<uint64_t> &full,
                      const std::vector<uint64_t> &strides,
                      const std::vector<uint64_t> &lo,
                      const std::vector<uint64_t> &hi,
                      const void* elem){
    const uint64_t m = full.size();
    if(!m) return;
    // Axes from the largest stride (outermost) to the smallest
    CzScratch scratch;
    uint64_t* order = scratch.get(m, 1);
    for(uint64_t d = 0; d < m; d++){
        uint64_t p = d;
        while(p > 0 && strides[order[p-1]] < strides[d]){ order[p] = order[p-1]; p--; }
        order[p] = d;
    }
    fillOutsideRec(bytes, (uint8_t*)buf, 0, m, order, full, strides, lo, hi, elem);
}

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

bool renameReplace(const std::string &src, const std::string &dst){
#ifdef _WIN32
    // rename() on Windows fails when dst exists, which silently kept the old
    // chunk/.zarray whenever an existing array was rewritten
    return MoveFileExA(src.c_str(), dst.c_str(), MOVEFILE_REPLACE_EXISTING) != 0;
#else
    return rename(src.c_str(), dst.c_str()) == 0;
#endif
}
