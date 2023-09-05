/*********************************************************************
  Blosc - Blocked Shuffling and Compression Library

  Copyright (c) 2021  The Blosc Development Team <blosc@blosc.org>
  https://blosc.org
  License: BSD 3-Clause (see LICENSE.txt)

  See LICENSE.txt for details about copyright and rights to use.
**********************************************************************/

/* AVX2-accelerated shuffle/unshuffle routines. */

#ifndef BLOSC_BITSHUFFLE_AVX2_H
#define BLOSC_BITSHUFFLE_AVX2_H

#include "blosc2/blosc2-common.h"

#include <stddef.h>
#include <stdint.h>

/**
  AVX2-accelerated bitshuffle routine.
*/
BLOSC_NO_EXPORT int64_t
    bshuf_trans_bit_elem_avx2(void* in, void* out, const size_t size,
                              const size_t elem_size, void* tmp_buf);

/**
  AVX2-accelerated bitunshuffle routine.
*/
BLOSC_NO_EXPORT int64_t
    bshuf_untrans_bit_elem_avx2(void* in, void* out, const size_t size,
                                const size_t elem_size, void* tmp_buf);

#endif /* BLOSC_BITSHUFFLE_AVX2_H */
