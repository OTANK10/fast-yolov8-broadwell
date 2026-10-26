#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "kernel.h"

void kernel
(
  const conv_desc_t* d,
  const float*       in,
  const float*       w,
  float*             out
){
  //TODO: Implement a fast 3x3 stride-2 pad-1 convolution here

  /*
    Assume the following
        - in   is NCHW8c with a 1-pixel zero halo (padded coords: logical + 1)
        - w    is [K/8][C/8][3][3][8 ic][8 oc]
        - out  is NCHW8c with a 1-pixel halo; write only the interior
        - out[k] = sum_{c,r,s} in[c][2*oh+r][2*ow+s] * w[k][c][r][s]
          (in padded coordinates, since in_halo == pad)
  */

  // Placeholder: correct scalar version on the blocked layout so the harness
  // has something to check. Replace it with the AVX2 kernel.
  const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);
  const int Hp = d->H + 2 * d->in_halo, Wp = d->W + 2 * d->in_halo;
  const int OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo;

  for (int n = 0; n < d->N; ++n)
    for (int kb = 0; kb < Kb; ++kb)
      for (int oh = 0; oh < d->OH; ++oh)
        for (int ow = 0; ow < d->OW; ++ow) {
          float acc[CB];
          for (int oc = 0; oc < CB; ++oc) acc[oc] = 0.0f;

          for (int cb = 0; cb < Cb; ++cb)
            for (int r = 0; r < 3; ++r)
              for (int s = 0; s < 3; ++s) {
                const float* ip = in + blk_idx(Cb, Hp, Wp, n, cb, 2 * oh + r, 2 * ow + s);
                const float* wp = w + wpk_idx(Cb, kb, cb, r, s, 0);
                for (int ic = 0; ic < CB; ++ic)
                  for (int oc = 0; oc < CB; ++oc)
                    acc[oc] += ip[ic] * wp[ic * CB + oc];
              }

          float* op = out + blk_idx(Kb, OHp, OWp, n, kb, oh + d->out_halo, ow + d->out_halo);
          for (int oc = 0; oc < CB; ++oc)
            op[oc] = acc[oc];
        }
}

/* ---------------------------------------------------------------------------
   AVX2 kernels: the conv as a small matmul, same shape as hw2's
   m x k * k x n kernel.

   For one output row oh, a run of output pixels ow.. and an oc block kb:

       C (m x n)  +=  A (m x k)  *  B (k x n)

     m = output pixels along ow                 -> broadcast (like a[p*m + i])
     n = output channels, 8 per YMM             -> SIMD      (like b[p*n + j])
     k = C*3*3 reduction, p = (cb, r, s, ic) over the real input channels
         only, so the zero channels padding C up to a multiple of 8 (the
         3-channel stem) are never multiplied

     B: the packed weights [kb][cb][r][s][ic][oc] are, for a fixed kb, a
        k x 8 row-major matrix, so row p is just b[p*CB]. The next oc block
        is ldb = k*CB floats further on.
     A: element (i, p) is the input value pixel i multiplies at step p. Pixel
        i reads padded x = 2*(ow+i)+s, so the pixels are 2*CB floats apart and
        a(i, p) = a[aoff[p] + i*2*CB], with aoff[p] = offset of (cb, r, s, ic).
     C: NCHW8c output, pixel i of oc block j is c[i*CB + j*ldc].
        C starts at zero instead of being loaded.

   A YMM register block is named by the oc blocks x pixels it covers. The
   accumulator vcIJ is pixel I, oc block J, as in the matmul's row I,
   vector column J.
   --------------------------------------------------------------------------- */

/* real channels in input block cb (C = 3 -> 3, otherwise CB) */
static inline int real_ch(int C, int cb) { return C - cb * CB < CB ? C - cb * CB : CB; }

/* aoff[p] for p over (cb, r, s, ic) with ic a real channel, relative to
   (cb 0, 2*oh, 2*ow). Returns k, the number of steps. */
static int build_aoff(int C, int Cb, int Hp, int Wp, int* aoff) {
  int p = 0;
  for (int cb = 0; cb < Cb; ++cb)
    for (int r = 0; r < 3; ++r)
      for (int s = 0; s < 3; ++s)
        for (int ic = 0; ic < real_ch(C, cb); ++ic)
          aoff[p++] = (int)blk_idx(Cb, Hp, Wp, 0, cb, r, s) + ic;
  return p;
}

/* B with the same rows as aoff: when C % CB != 0 the weight rows of the
   padding channels are dropped (k x 8 per oc block). Otherwise the packed
   weights already are that matrix and are used as is. */
static const float* pack_b(int C, int Cb, int Kb, int k, const float* w) {
  if (C % CB == 0) return w;
  float* b = _mm_malloc((size_t)Kb * k * CB * sizeof(float), 32);
  for (int kb = 0; kb < Kb; ++kb) {
    int p = 0;
    for (int cb = 0; cb < Cb; ++cb)
      for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s)
          for (int ic = 0; ic < real_ch(C, cb); ++ic)
            memcpy(&b[((size_t)kb * k + p++) * CB], w + wpk_idx(Cb, kb, cb, r, s, ic),
                   CB * sizeof(float));
  }
  return b;
}

/* Generic nk x nw block (nk <= 3 oc blocks, nw <= 6 pixels) for the
   leftovers. It runs the same loop; always inlined with constant nk/nw so
   the vc array stays in registers. */
static inline __attribute__((always_inline)) void conv_tail
(
  const int        nk,
  const int        nw,
  int              k,
  const int*       aoff,
  const float*     a,
  const float*     b,
  size_t           ldb,
  float*           c,
  size_t           ldc
){
  __m256 vc[6][3];

#pragma GCC unroll 6
  for (int i = 0; i < nw; ++i)
#pragma GCC unroll 3
    for (int j = 0; j < nk; ++j)
      vc[i][j] = _mm256_setzero_ps();

  for (int p = 0; p != k; ++p) {

    __m256 vb[3];
#pragma GCC unroll 3
    for (int j = 0; j < nk; ++j)
      vb[j] = _mm256_load_ps(&b[j*ldb + p*CB]);

    const float* ap = &a[aoff[p]];

#pragma GCC unroll 6
    for (int i = 0; i < nw; ++i) {
      __m256 va = _mm256_broadcast_ss(&ap[i*2*CB]);
#pragma GCC unroll 3
      for (int j = 0; j < nk; ++j)
        vc[i][j] = _mm256_fmadd_ps(va, vb[j], vc[i][j]);
    }
  }

#pragma GCC unroll 6
  for (int i = 0; i < nw; ++i)
#pragma GCC unroll 3
    for (int j = 0; j < nk; ++j)
      _mm256_storeu_ps(&c[i*CB + j*ldc], vc[i][j]);
}

#define CONV_SETUP                                                            \
  const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);                 \
  const int Hp = d->H + 2 * d->in_halo, Wp = d->W + 2 * d->in_halo;           \
  const int OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo;     \
  int aoff[Cb * 3 * 3 * CB];                                                  \
  const int k = build_aoff(d->C, Cb, Hp, Wp, aoff);  /* real channels only */ \
  const size_t ldb = (size_t)k * CB;          /* next oc block of weights */  \
  const size_t ldc = (size_t)OHp * OWp * CB;  /* next oc block of output  */  \
  const float* wk = pack_b(d->C, Cb, Kb, k, w);

#define CONV_DONE \
  if (wk != w) _mm_free((void*)wk);

/* operands of the block starting at output pixel (n, oh, ow), oc block kb */
#define A_AT  (in   + blk_idx(Cb, Hp, Wp, n, 0, 2 * oh, 2 * ow))
#define B_AT  (wk   + kb * ldb)
#define C_AT  (out  + blk_idx(Kb, OHp, OWp, n, kb, oh + d->out_halo, ow + d->out_halo))

#define TAIL(NK, NW) \
  conv_tail(NK, NW, k, aoff, A_AT, B_AT, ldb, C_AT, ldc)

/* ---------------------------------------------------------------------------
   kernel_v2: 2x5 block, 2 oc blocks (n = 16) x 5 pixels (m = 5)
     10 accumulators + 2 vb + 1 va = 13 of 16 YMM registers
   --------------------------------------------------------------------------- */

static inline void conv_2x5
(
  int              k,
  const int*       aoff,
  const float*     a,
  const float*     b,
  size_t           ldb,
  float*           c,
  size_t           ldc
){
  const float* b0 = b;
  const float* b1 = b + ldb;

  __m256 vc00 = _mm256_setzero_ps();
  __m256 vc01 = _mm256_setzero_ps();

  __m256 vc10 = vc00;
  __m256 vc11 = vc01;

  __m256 vc20 = vc00;
  __m256 vc21 = vc01;

  __m256 vc30 = vc00;
  __m256 vc31 = vc01;

  __m256 vc40 = vc00;
  __m256 vc41 = vc01;

  for (int p = 0; p != k; ++p) { // p is column idx in A and row idx in B

    __m256 vb0 = _mm256_load_ps(&b0[p*CB]);
    __m256 vb1 = _mm256_load_ps(&b1[p*CB]);

    const float* ap = &a[aoff[p]];
    __m256 va;

    va = _mm256_broadcast_ss(&ap[0*2*CB]);
    vc00 = _mm256_fmadd_ps(va, vb0, vc00);
    vc01 = _mm256_fmadd_ps(va, vb1, vc01);

    va = _mm256_broadcast_ss(&ap[1*2*CB]);
    vc10 = _mm256_fmadd_ps(va, vb0, vc10);
    vc11 = _mm256_fmadd_ps(va, vb1, vc11);

    va = _mm256_broadcast_ss(&ap[2*2*CB]);
    vc20 = _mm256_fmadd_ps(va, vb0, vc20);
    vc21 = _mm256_fmadd_ps(va, vb1, vc21);

    va = _mm256_broadcast_ss(&ap[3*2*CB]);
    vc30 = _mm256_fmadd_ps(va, vb0, vc30);
    vc31 = _mm256_fmadd_ps(va, vb1, vc31);

    va = _mm256_broadcast_ss(&ap[4*2*CB]);
    vc40 = _mm256_fmadd_ps(va, vb0, vc40);
    vc41 = _mm256_fmadd_ps(va, vb1, vc41);

  }

  _mm256_storeu_ps(&c[0*CB + 0*ldc], vc00);
  _mm256_storeu_ps(&c[0*CB + 1*ldc], vc01);

  _mm256_storeu_ps(&c[1*CB + 0*ldc], vc10);
  _mm256_storeu_ps(&c[1*CB + 1*ldc], vc11);

  _mm256_storeu_ps(&c[2*CB + 0*ldc], vc20);
  _mm256_storeu_ps(&c[2*CB + 1*ldc], vc21);

  _mm256_storeu_ps(&c[3*CB + 0*ldc], vc30);
  _mm256_storeu_ps(&c[3*CB + 1*ldc], vc31);

  _mm256_storeu_ps(&c[4*CB + 0*ldc], vc40);
  _mm256_storeu_ps(&c[4*CB + 1*ldc], vc41);
}

void kernel_v2
(
  const conv_desc_t* d,
  const float*       in,
  const float*       w,
  float*             out
){
  CONV_SETUP

  for (int n = 0; n < d->N; ++n)
    for (int kb = 0; kb < Kb; kb += 2)
      for (int oh = 0; oh < d->OH; ++oh) {
        int ow = 0;
        if (Kb - kb >= 2) {
          for (; ow + 5 <= d->OW; ow += 5)
            conv_2x5(k, aoff, A_AT, B_AT, ldb, C_AT, ldc);
          for (; ow < d->OW; ++ow)         TAIL(2, 1);   /* OW % 5 */
        } else {
          for (; ow + 5 <= d->OW; ow += 5) TAIL(1, 5);   /* odd Kb */
          for (; ow < d->OW; ++ow)         TAIL(1, 1);
        }
      }

  CONV_DONE
}

/* ---------------------------------------------------------------------------
   kernel_v3: 2x6 block, 2 oc blocks (n = 16) x 6 pixels (m = 6)
     12 accumulators + 2 vb + 1 va = 15 of 16 YMM registers
   --------------------------------------------------------------------------- */

static inline void conv_2x6
(
  int              k,
  const int*       aoff,
  const float*     a,
  const float*     b,
  size_t           ldb,
  float*           c,
  size_t           ldc
){
  const float* b0 = b;
  const float* b1 = b + ldb;

  __m256 vc00 = _mm256_setzero_ps();
  __m256 vc01 = _mm256_setzero_ps();

  __m256 vc10 = vc00;
  __m256 vc11 = vc01;

  __m256 vc20 = vc00;
  __m256 vc21 = vc01;

  __m256 vc30 = vc00;
  __m256 vc31 = vc01;

  __m256 vc40 = vc00;
  __m256 vc41 = vc01;

  __m256 vc50 = vc00;
  __m256 vc51 = vc01;

  for (int p = 0; p != k; ++p) { // p is column idx in A and row idx in B

    __m256 vb0 = _mm256_load_ps(&b0[p*CB]);
    __m256 vb1 = _mm256_load_ps(&b1[p*CB]);

    const float* ap = &a[aoff[p]];
    __m256 va;

    va = _mm256_broadcast_ss(&ap[0*2*CB]);
    vc00 = _mm256_fmadd_ps(va, vb0, vc00);
    vc01 = _mm256_fmadd_ps(va, vb1, vc01);

    va = _mm256_broadcast_ss(&ap[1*2*CB]);
    vc10 = _mm256_fmadd_ps(va, vb0, vc10);
    vc11 = _mm256_fmadd_ps(va, vb1, vc11);

    va = _mm256_broadcast_ss(&ap[2*2*CB]);
    vc20 = _mm256_fmadd_ps(va, vb0, vc20);
    vc21 = _mm256_fmadd_ps(va, vb1, vc21);

    va = _mm256_broadcast_ss(&ap[3*2*CB]);
    vc30 = _mm256_fmadd_ps(va, vb0, vc30);
    vc31 = _mm256_fmadd_ps(va, vb1, vc31);

    va = _mm256_broadcast_ss(&ap[4*2*CB]);
    vc40 = _mm256_fmadd_ps(va, vb0, vc40);
    vc41 = _mm256_fmadd_ps(va, vb1, vc41);

    va = _mm256_broadcast_ss(&ap[5*2*CB]);
    vc50 = _mm256_fmadd_ps(va, vb0, vc50);
    vc51 = _mm256_fmadd_ps(va, vb1, vc51);

  }

  _mm256_storeu_ps(&c[0*CB + 0*ldc], vc00);
  _mm256_storeu_ps(&c[0*CB + 1*ldc], vc01);

  _mm256_storeu_ps(&c[1*CB + 0*ldc], vc10);
  _mm256_storeu_ps(&c[1*CB + 1*ldc], vc11);

  _mm256_storeu_ps(&c[2*CB + 0*ldc], vc20);
  _mm256_storeu_ps(&c[2*CB + 1*ldc], vc21);

  _mm256_storeu_ps(&c[3*CB + 0*ldc], vc30);
  _mm256_storeu_ps(&c[3*CB + 1*ldc], vc31);

  _mm256_storeu_ps(&c[4*CB + 0*ldc], vc40);
  _mm256_storeu_ps(&c[4*CB + 1*ldc], vc41);

  _mm256_storeu_ps(&c[5*CB + 0*ldc], vc50);
  _mm256_storeu_ps(&c[5*CB + 1*ldc], vc51);
}

void kernel_v3
(
  const conv_desc_t* d,
  const float*       in,
  const float*       w,
  float*             out
){
  CONV_SETUP

  for (int n = 0; n < d->N; ++n)
    for (int kb = 0; kb < Kb; kb += 2)
      for (int oh = 0; oh < d->OH; ++oh) {
        int ow = 0;
        if (Kb - kb >= 2) {
          for (; ow + 6 <= d->OW; ow += 6)
            conv_2x6(k, aoff, A_AT, B_AT, ldb, C_AT, ldc);
          switch (d->OW - ow) {                          /* OW % 6, one block */
            case 5: TAIL(2, 5); break;
            case 4: TAIL(2, 4); break;
            case 3: TAIL(2, 3); break;
            case 2: TAIL(2, 2); break;
            case 1: TAIL(2, 1); break;
          }
        } else {
          for (; ow + 6 <= d->OW; ow += 6) TAIL(1, 6);   /* odd Kb */
          for (; ow < d->OW; ++ow)         TAIL(1, 1);
        }
      }

  CONV_DONE
}

/* ---------------------------------------------------------------------------
   kernel_v4: 3x4 block, 3 oc blocks (n = 24) x 4 pixels (m = 4)
     12 accumulators + 3 vb + 1 va = all 16 YMM registers
   --------------------------------------------------------------------------- */

static inline void conv_3x4
(
  int              k,
  const int*       aoff,
  const float*     a,
  const float*     b,
  size_t           ldb,
  float*           c,
  size_t           ldc
){
  const float* b0 = b;
  const float* b1 = b + ldb;
  const float* b2 = b + 2*ldb;

  __m256 vc00 = _mm256_setzero_ps();
  __m256 vc01 = _mm256_setzero_ps();
  __m256 vc02 = _mm256_setzero_ps();

  __m256 vc10 = vc00;
  __m256 vc11 = vc01;
  __m256 vc12 = vc02;

  __m256 vc20 = vc00;
  __m256 vc21 = vc01;
  __m256 vc22 = vc02;

  __m256 vc30 = vc00;
  __m256 vc31 = vc01;
  __m256 vc32 = vc02;

  for (int p = 0; p != k; ++p) { // p is column idx in A and row idx in B

    __m256 vb0 = _mm256_load_ps(&b0[p*CB]);
    __m256 vb1 = _mm256_load_ps(&b1[p*CB]);
    __m256 vb2 = _mm256_load_ps(&b2[p*CB]);

    const float* ap = &a[aoff[p]];
    __m256 va;

    va = _mm256_broadcast_ss(&ap[0*2*CB]);
    vc00 = _mm256_fmadd_ps(va, vb0, vc00);
    vc01 = _mm256_fmadd_ps(va, vb1, vc01);
    vc02 = _mm256_fmadd_ps(va, vb2, vc02);

    va = _mm256_broadcast_ss(&ap[1*2*CB]);
    vc10 = _mm256_fmadd_ps(va, vb0, vc10);
    vc11 = _mm256_fmadd_ps(va, vb1, vc11);
    vc12 = _mm256_fmadd_ps(va, vb2, vc12);

    va = _mm256_broadcast_ss(&ap[2*2*CB]);
    vc20 = _mm256_fmadd_ps(va, vb0, vc20);
    vc21 = _mm256_fmadd_ps(va, vb1, vc21);
    vc22 = _mm256_fmadd_ps(va, vb2, vc22);

    va = _mm256_broadcast_ss(&ap[3*2*CB]);
    vc30 = _mm256_fmadd_ps(va, vb0, vc30);
    vc31 = _mm256_fmadd_ps(va, vb1, vc31);
    vc32 = _mm256_fmadd_ps(va, vb2, vc32);

  }

  _mm256_storeu_ps(&c[0*CB + 0*ldc], vc00);
  _mm256_storeu_ps(&c[0*CB + 1*ldc], vc01);
  _mm256_storeu_ps(&c[0*CB + 2*ldc], vc02);

  _mm256_storeu_ps(&c[1*CB + 0*ldc], vc10);
  _mm256_storeu_ps(&c[1*CB + 1*ldc], vc11);
  _mm256_storeu_ps(&c[1*CB + 2*ldc], vc12);

  _mm256_storeu_ps(&c[2*CB + 0*ldc], vc20);
  _mm256_storeu_ps(&c[2*CB + 1*ldc], vc21);
  _mm256_storeu_ps(&c[2*CB + 2*ldc], vc22);

  _mm256_storeu_ps(&c[3*CB + 0*ldc], vc30);
  _mm256_storeu_ps(&c[3*CB + 1*ldc], vc31);
  _mm256_storeu_ps(&c[3*CB + 2*ldc], vc32);
}

void kernel_v4
(
  const conv_desc_t* d,
  const float*       in,
  const float*       w,
  float*             out
){
  CONV_SETUP

  for (int n = 0; n < d->N; ++n)
    for (int kb = 0; kb < Kb; ) {
      const int nk = Kb - kb >= 3 ? 3 : Kb - kb;
      for (int oh = 0; oh < d->OH; ++oh) {
        int ow = 0;
        if (nk == 3) {
          for (; ow + 4 <= d->OW; ow += 4)
            conv_3x4(k, aoff, A_AT, B_AT, ldb, C_AT, ldc);
          for (; ow < d->OW; ++ow)         TAIL(3, 1);   /* OW % 4 */
        } else if (nk == 2) {
          for (; ow + 4 <= d->OW; ow += 4) TAIL(2, 4);   /* Kb % 3 == 2 */
          for (; ow < d->OW; ++ow)         TAIL(2, 1);
        } else {
          for (; ow + 4 <= d->OW; ow += 4) TAIL(1, 4);   /* Kb % 3 == 1 */
          for (; ow < d->OW; ++ow)         TAIL(1, 1);
        }
      }
      kb += nk;
    }

  CONV_DONE
}
