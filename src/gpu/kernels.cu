// GPU kernels of cpp-zarr's GPU reads (gpureadzarr.cpp), compiled to PTX by
// gen_kernels_ptx.py into kernels_ptx.h; the CUDA driver compiles that PTX for the GPU it
// runs on. Plain C kernels, no includes, so that NVRTC compiles them as they are.

// Each block of decoded chunk data (a blosc block, or a whole chunk) is n elements of ts bytes
// that start at chunk-local element `first` of its chunk, in the chunk's storage order. The
// kernel writes each element to its place in its region's output:
//  - the element's storage-order index is split into coordinates by the chunk's dimensions in
//    storage order (sdims, slowest first), which saxis maps to the array's axes;
//  - its place in the region is the chunk's origin there (corg, per array axis) plus those
//    coordinates, outside the region (a chunk the region cuts) it is left out;
//  - its offset in the output is the sum of the place's coordinates times the region's
//    element strides (rstride: C or F order).
// A block's bytes are byte-shuffled (blosc's shuffle: all elements' first bytes, then their
// second bytes, ...) when its flag says so, and big-endian data is byte-swapped in units of
// swapunit bytes (the element, or each half of a complex number) on the way.
// The block's source is in the decode buffer (dec + offset) or, with bit 63 of the offset set,
// in the copy of the host buffer (host + offset): blocks stored as they are.
// widen 1 or 2 writes 16-bit floats (float16, bfloat16) as float32s, for callers that have no
// 16-bit floats (MATLAB): as the CPU reader's halfToFloat and bfloat16ToFloat do.

__device__ __forceinline__ unsigned int cz_widen(unsigned int h, int widen)
{
    if (widen == 2) return h << 16;
    const unsigned int sign = (h & 0x8000u) << 16, e = h & 0x7c00u, m = h & 0x03ffu;
    if (e == 0) {
        if (!m) return sign;
        int k = 0;
        unsigned int t = m << 1;
        while (!(t & 0x0400u)) { t <<= 1; k++; }
        return sign + ((unsigned int)(127 - 15 - k) << 23) + ((t & 0x03ffu) << 13);
    }
    if (e == 0x7c00u) return sign + 0x7f800000u + (m << 13);
    return sign + (((h & 0x7fffu) + 0x1c000u) << 13);
}

#define CZ_SCATTER(NAME, IDX)                                                                         \
extern "C" __global__ void NAME(                                                                      \
    const unsigned long long* __restrict__ bsrc, const unsigned int* __restrict__ bnel,             \
    const unsigned long long* __restrict__ bfirst, const unsigned int* __restrict__ bchunk,         \
    const unsigned char* __restrict__ bflags, const unsigned int* __restrict__ creg,                \
    const long long* __restrict__ corg, const unsigned long long* __restrict__ rout,                \
    const long long* __restrict__ rshape, const long long* __restrict__ rstride,                    \
    const long long* __restrict__ sdims, const int* __restrict__ saxis,                             \
    const unsigned char* __restrict__ host, const unsigned char* __restrict__ dec,                  \
    int ndim, int ts, int swapunit, int widen)                                                        \
{                                                                                                     \
    const unsigned int b = blockIdx.x;                                                                \
    const unsigned long long so = bsrc[b];                                                            \
    const unsigned char* s = (so >> 63) ? host + (so & 0x7fffffffffffffffULL) : dec + so;           \
    const unsigned int n = bnel[b], c = bchunk[b];                                                    \
    const IDX f = (IDX)bfirst[b];                                                                     \
    const unsigned int r = creg[c];                                                                   \
    const long long* o = corg + (long long)c * ndim;                                                  \
    const long long* rs = rshape + (long long)r * ndim;                                               \
    const long long* os = rstride + (long long)r * ndim;                                              \
    unsigned char* out = (unsigned char*)rout[r];                                                     \
    const int sh = bflags[b] & 1;                                                                     \
    /* (whole-element stores need an output aligned to the element, and loads of unshuffled      \
       elements a source aligned to it: blocks stored as they are can start anywhere) */          \
    const int ots = widen ? 4 : ts;                                                                   \
    const unsigned long long am = (unsigned long long)(ts - 1), om = (unsigned long long)(ots - 1);  \
    const int fast = swapunit == 0 && (ts & (ts - 1)) == 0 && !((unsigned long long)out & om) &&    \
                     (sh || !((unsigned long long)s & am));                                          \
    for (unsigned int e = threadIdx.x; e < n; e += blockDim.x) {                                      \
        IDX l = f + (IDX)e;                                                                           \
        long long off = 0;                                                                            \
        int inside = 1;                                                                               \
        for (int k = ndim - 1; k >= 0; k--) {                                                         \
            const IDX d = (IDX)sdims[k];                                                              \
            const int a = saxis[k];                                                                   \
            const long long p = o[a] + (long long)(l % d);                                            \
            l /= d;                                                                                   \
            if (p < 0 || p >= rs[a]) { inside = 0; break; }                                           \
            off += p * os[a];                                                                         \
        }                                                                                             \
        if (!inside) continue;                                                                        \
        unsigned char* dst = out + off * ots;                                                         \
        if (fast) {                                                                                   \
            if (ts == 1) { dst[0] = s[e]; continue; }                                                 \
            if (ts == 2) {                                                                            \
                const unsigned short v = sh ? (unsigned short)(s[e] | ((unsigned)s[n + e] << 8))      \
                                            : ((const unsigned short*)s)[e];                          \
                if (widen) *(unsigned int*)dst = cz_widen(v, widen);                                  \
                else *(unsigned short*)dst = v;                                                       \
                continue;                                                                             \
            }                                                                                         \
            if (ts == 4) {                                                                            \
                *(unsigned int*)dst = sh ? (s[e] | ((unsigned)s[n + e] << 8) | ((unsigned)s[2 * n + e] << 16) | \
                                            ((unsigned)s[3 * n + e] << 24))                           \
                                         : ((const unsigned int*)s)[e];                               \
                continue;                                                                             \
            }                                                                                         \
            if (ts == 8) {                                                                            \
                unsigned long long v;                                                                 \
                if (sh) {                                                                             \
                    v = 0;                                                                            \
                    for (int j = 0; j < 8; j++) v |= (unsigned long long)s[(unsigned long long)j * n + e] << (8 * j); \
                }                                                                                     \
                else v = ((const unsigned long long*)s)[e];                                           \
                *(unsigned long long*)dst = v;                                                        \
                continue;                                                                             \
            }                                                                                         \
        }                                                                                             \
        if (widen) {                                                                                  \
            unsigned int h = 0;                                                                       \
            for (int j = 0; j < 2; j++) {                                                             \
                const unsigned char v = sh ? s[(unsigned long long)j * n + e] : s[(unsigned long long)e * 2 + j]; \
                h |= (unsigned int)v << (8 * (swapunit ? 1 - j : j));                                 \
            }                                                                                         \
            const unsigned int f = cz_widen(h, widen);                                                \
            for (int j = 0; j < 4; j++) dst[j] = (unsigned char)(f >> (8 * j));                       \
            continue;                                                                                 \
        }                                                                                             \
        for (int j = 0; j < ts; j++) {                                                                \
            const unsigned char v = sh ? s[(unsigned long long)j * n + e] : s[(unsigned long long)e * ts + j]; \
            const int t = swapunit ? (j / swapunit) * swapunit + (swapunit - 1 - j % swapunit) : j;  \
            dst[t] = v;                                                                               \
        }                                                                                             \
    }                                                                                                 \
}

// (32-bit index arithmetic for chunks of fewer than 2^32 elements, which is much faster)
CZ_SCATTER(cz_scatter32, unsigned int)
CZ_SCATTER(cz_scatter64, unsigned long long)

// The pointer arrays nvCOMP takes, from offsets: in[i] = host + inoff[i], out[i] = dec + outoff[i]
extern "C" __global__ void cz_resolve(const unsigned long long* __restrict__ inoff, const unsigned long long* __restrict__ outoff,
                                      unsigned long long* __restrict__ inp, unsigned long long* __restrict__ outp,
                                      unsigned long long host, unsigned long long dec, unsigned long long n)
{
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        inp[i] = host + inoff[i];
        outp[i] = dec + outoff[i];
    }
}

// n elements of ts (at most 16) bytes set to the pattern lo, hi (little-endian bytes)
extern "C" __global__ void cz_fill(unsigned char* __restrict__ out, unsigned long long n, int ts,
                                   unsigned long long lo, unsigned long long hi)
{
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned char* d = out + i * ts;
        for (int j = 0; j < ts; j++) d[j] = (unsigned char)(((j < 8) ? (lo >> (8 * j)) : (hi >> (8 * (j - 8)))) & 0xff);
    }
}

// Whether every decompressed stream came out whole: counts the streams whose status is not 0 or
// whose size differs from the expected one into *bad, and keeps the first such stream's index
extern "C" __global__ void cz_check(const int* __restrict__ status, const unsigned long long* __restrict__ got,
                                    const unsigned long long* __restrict__ want, unsigned long long n,
                                    unsigned long long* __restrict__ bad)
{
    for (unsigned long long i = blockIdx.x * (unsigned long long)blockDim.x + threadIdx.x; i < n;
         i += (unsigned long long)gridDim.x * blockDim.x) {
        if (status[i] != 0 || got[i] != want[i]) {
            atomicAdd(bad, 1ULL);
            atomicMin(bad + 1, i);
        }
    }
}
