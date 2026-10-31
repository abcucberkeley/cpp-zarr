// GPU kernels of cpp-zarr's GPU reads (gpureadzarr.cpp) and writes (gpuwritezarr.cpp), compiled
// to PTX by gen_kernels_ptx.py into kernels_ptx.h; the CUDA driver compiles that PTX for the GPU
// it runs on. Plain C kernels, no includes, so that NVRTC compiles them as they are.

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

// GPU writes (gpuwritezarr.cpp): the chunks of a part of a write, each put together from the input
// in its storage order, as the CPU writer puts it together. An element inside the written region
// comes from the input (in: element strides istride along each array axis, the region from wstart,
// wshape long); one outside it from the chunk's current contents (at base + cbase[c] in storage
// order, for a crop write's chunks the region cuts; cbase[c] -1 for none) or the fill value (lo,
// hi: little-endian bytes, zero past the element), which elements past the array's edge (ashape)
// get anyway when fillpast is set. Chunk c goes to out + cout[c] in blocks of be elements (blosc's
// blocks, the last one maybe shorter; one block of the whole chunk for other codecs), block b at
// b * bstride: its elements one after another, or shuffled as blosc shuffles a block (its elements'
// first bytes, then their second bytes, ...). A block blosc splits into ts streams (with split,
// every block but a shorter last one) has its streams pstride apart: the shuffled block's runs of
// each byte or, without shuffle, its bytes cut into ts runs. Every chunk with a byte that is not
// zero is flagged in nonzero (sparse writes leave out all-zero chunks). A chunk's elements are
// shared out among `pieces` thread blocks; ts is at most 16. Outputs of whole elements (no shuffle
// and no split) are 16-byte aligned.
#define CZ_GATHER(NAME, IDX)                                                                          \
extern "C" __global__ void NAME(                                                                      \
    const unsigned char* __restrict__ in, const long long* __restrict__ istride,                     \
    const long long* __restrict__ wstart, const long long* __restrict__ wshape,                      \
    const long long* __restrict__ ashape, const long long* __restrict__ sdims,                       \
    const int* __restrict__ saxis, const long long* __restrict__ corg,                               \
    const unsigned long long* __restrict__ cout, const long long* __restrict__ cbase,                \
    const unsigned char* __restrict__ base, unsigned char* __restrict__ out,                         \
    unsigned int* __restrict__ nonzero, unsigned long long celems, unsigned long long be,            \
    unsigned long long nblocks, unsigned long long bstride, unsigned long long pstride,              \
    int pieces, int ndim, int ts, int shuffle, int split, int fillpast,                              \
    unsigned long long lo, unsigned long long hi)                                                    \
{                                                                                                     \
    const unsigned int c = blockIdx.x / (unsigned int)pieces;                                         \
    const unsigned int piece = blockIdx.x % (unsigned int)pieces;                                     \
    const long long* o = corg + (long long)c * ndim;                                                  \
    unsigned char* dst = out + cout[c];                                                               \
    const long long bo = cbase[c];                                                                    \
    const unsigned char* b = bo >= 0 ? base + bo : 0;                                                 \
    const unsigned long long per = (celems + pieces - 1) / pieces;                                    \
    const unsigned long long a0 = per * piece;                                                        \
    const unsigned long long a1 = a0 + per < celems ? a0 + per : celems;                              \
    const IDX bel = (IDX)be;                                                                          \
    const IDX lastb = (IDX)(nblocks - 1);                                                             \
    const IDX lastn = (IDX)(celems - (nblocks - 1) * be);                                             \
    /* (whole-element loads and stores for element sizes of 1, 2, 4 and 8 bytes, where aligned) */   \
    const int pow2 = (ts & (ts - 1)) == 0 && ts <= 8;                                                 \
    const int wide = pow2 && !((unsigned long long)in & (unsigned long long)(ts - 1));                \
    unsigned int any = 0;                                                                             \
    for (IDX l = (IDX)a0 + threadIdx.x; l < (IDX)a1; l += blockDim.x) {                               \
        IDX r = l;                                                                                    \
        long long src = 0;                                                                            \
        int inside = 1, past = 0;                                                                     \
        for (int k = ndim - 1; k >= 0; k--) {                                                         \
            const IDX d = (IDX)sdims[k];                                                              \
            const int a = saxis[k];                                                                   \
            const long long p = o[a] + (long long)(r % d);                                            \
            r /= d;                                                                                   \
            if (p >= ashape[a]) past = 1;                                                             \
            const long long q = p - wstart[a];                                                        \
            if (q < 0 || q >= wshape[a]) inside = 0;                                                  \
            else src += q * istride[a];                                                               \
        }                                                                                             \
        unsigned long long v0 = 0, v1 = 0;                                                            \
        const unsigned char* s = 0;                                                                   \
        if (inside) s = in + src * ts;                                                                \
        else if (b && !(fillpast && past)) s = b + (unsigned long long)l * ts;                        \
        else { v0 = lo; v1 = hi; }                                                                    \
        if (s) {                                                                                      \
            if (inside && wide) {                                                                     \
                if (ts == 1) v0 = s[0];                                                               \
                else if (ts == 2) v0 = *(const unsigned short*)s;                                     \
                else if (ts == 4) v0 = *(const unsigned int*)s;                                       \
                else v0 = *(const unsigned long long*)s;                                              \
            }                                                                                         \
            else for (int j = 0; j < ts; j++) {                                                       \
                if (j < 8) v0 |= (unsigned long long)s[j] << (8 * j);                                 \
                else v1 |= (unsigned long long)s[j] << (8 * (j - 8));                                 \
            }                                                                                         \
        }                                                                                             \
        any |= (v0 | v1) != 0;                                                                        \
        const IDX blk = l / bel, e = l - blk * bel;                                                   \
        const IDX n = blk == lastb ? lastn : bel;                                                     \
        unsigned char* bp = dst + (unsigned long long)blk * bstride;                                  \
        const int sp = split && n == bel;                                                             \
        if (!shuffle && !sp) {                                                                        \
            unsigned char* t = bp + (unsigned long long)e * ts;                                       \
            if (pow2) {                                                                               \
                if (ts == 1) t[0] = (unsigned char)v0;                                                \
                else if (ts == 2) *(unsigned short*)t = (unsigned short)v0;                           \
                else if (ts == 4) *(unsigned int*)t = (unsigned int)v0;                               \
                else *(unsigned long long*)t = v0;                                                    \
            }                                                                                         \
            else for (int j = 0; j < ts; j++)                                                         \
                t[j] = (unsigned char)((j < 8 ? v0 >> (8 * j) : v1 >> (8 * (j - 8))) & 0xff);         \
            continue;                                                                                 \
        }                                                                                             \
        for (int j = 0; j < ts; j++) {                                                                \
            const unsigned char x = (unsigned char)((j < 8 ? v0 >> (8 * j) : v1 >> (8 * (j - 8))) & 0xff); \
            unsigned long long at;                                                                    \
            if (shuffle) at = sp ? (unsigned long long)j * pstride + e : (unsigned long long)j * n + e; \
            else {                                                                                    \
                const unsigned long long q = (unsigned long long)e * ts + j;                          \
                at = (q / bel) * pstride + q % bel;                                                   \
            }                                                                                         \
            bp[at] = x;                                                                               \
        }                                                                                             \
    }                                                                                                 \
    if (__syncthreads_or(any) && threadIdx.x == 0) atomicOr(nonzero + c, 1u);                         \
}

// (32-bit index arithmetic for chunks of fewer than 2^32 elements)
CZ_GATHER(cz_gather32, unsigned int)
CZ_GATHER(cz_gather64, unsigned long long)

// GPU writes: the stored chunks of a part put together for the host. Piece i copies n[i] bytes to
// out + dst[i] from comp, unc or tab (kind 0, 1 or 2) + src[i], after writing the 4-byte
// little-endian number pre[i] just before them when kind has bit 2 set (blosc's stream sizes).
extern "C" __global__ void cz_pack(const unsigned long long* __restrict__ psrc, const unsigned long long* __restrict__ pdst,
                                   const unsigned int* __restrict__ pn, const unsigned int* __restrict__ ppre,
                                   const unsigned char* __restrict__ pkind, const unsigned char* __restrict__ comp,
                                   const unsigned char* __restrict__ unc, const unsigned char* __restrict__ tab,
                                   unsigned char* __restrict__ out)
{
    const unsigned int i = blockIdx.x;
    const unsigned int kind = pkind[i];
    const unsigned char* s = ((kind & 3) == 0 ? comp : (kind & 3) == 1 ? unc : tab) + psrc[i];
    unsigned char* d = out + pdst[i];
    const unsigned int n = pn[i];
    if ((kind & 4) && threadIdx.x < 4) d[(int)threadIdx.x - 4] = (unsigned char)((ppre[i] >> (8 * threadIdx.x)) & 0xff);
    // (8 bytes at a time where the source and destination are as far from 8-byte alignment)
    const unsigned int sa = (unsigned int)((unsigned long long)s & 7), da = (unsigned int)((unsigned long long)d & 7);
    if (sa == da && n >= 64) {
        const unsigned int head = (8 - sa) & 7, nw = (n - head) / 8;
        for (unsigned int k = threadIdx.x; k < head; k += blockDim.x) d[k] = s[k];
        const unsigned long long* sw = (const unsigned long long*)(s + head);
        unsigned long long* dw = (unsigned long long*)(d + head);
        for (unsigned int k = threadIdx.x; k < nw; k += blockDim.x) dw[k] = sw[k];
        for (unsigned int k = head + nw * 8 + threadIdx.x; k < n; k += blockDim.x) d[k] = s[k];
    }
    else for (unsigned int k = threadIdx.x; k < n; k += blockDim.x) d[k] = s[k];
}
