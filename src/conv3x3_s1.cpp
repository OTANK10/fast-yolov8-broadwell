/*
 * conv3x3_s1.c  --  3x3, stride-1, pad-1 convolution (C2f bottlenecks + Detect head).
 * Owner: Om.
 *
 * Read docs/STRIDE1_GUIDE.md alongside this file; it derives every number here.
 *
 *   v0  scalar, blocked layouts, loop order identical to v1. Read this first:
 *       it is v1 with the registers replaced by plain arrays.
 *   v1  AVX2 register-blocked microkernel: 2 output-channel blocks x 5 output
 *       pixels = 10 YMM accumulators (= 2 FMA ports x 5-cycle latency), plus
 *       2 weight registers and 1 broadcast register = 13 of 16 YMM.
 *       OpenMP over (image, output-channel pair, output row).
 *
 * Next steps (yours): v2 cache blocking over input-channel blocks so the
 * weight slab stays in L1/L2 for the 64/128-channel layers; NUMA placement;
 * a work split that keeps 20 cores busy on the 20x20 layers.
 */
#include <cstdio>
#include "common.h"
#include "epilogue.h"
#include "kernels.h"

namespace fastyolo {

#define TILE_KB 2 /* output-channel blocks per microkernel call (2 x 8 = 16 channels) */
#define TILE_OW 5 /* output pixels per microkernel call                           */

/* ------------------------------------------------------------------ v0 ---- */
void conv3x3_s1_v0(const conv_desc_t *d, const float *in, const float *wp,
                   const float *bias, float *out) {
    const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);
    const int Hp = d->H + 2 * d->in_halo, Wp = d->W + 2 * d->in_halo;
    const int OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo;
    const int off = d->in_halo - d->pad; /* 0 when the halo is exactly the padding */

    for (int n = 0; n < d->N; n++)
        for (int kb = 0; kb < Kb; kb++)
            for (int oh = 0; oh < d->OH; oh++)
                for (int ow = 0; ow < d->OW; ow++) {
                    float acc[CB]; /* in v1 this is ONE ymm register */
                    for (int ko = 0; ko < CB; ko++) acc[ko] = bias[kb * CB + ko];

                    for (int cb = 0; cb < Cb; cb++)
                        for (int r = 0; r < 3; r++)
                            for (int s = 0; s < 3; s++) {
                                const float *ip = in + blk_idx(Cb, Hp, Wp, n, cb, oh + r + off, ow + s + off);
                                const float *wq = wp + ((((size_t)kb * Cb + cb) * 3 + r) * 3 + s) * CB * CB;
                                for (int ci = 0; ci < CB; ci++)     /* v1: broadcast ip[ci]   */
                                    for (int ko = 0; ko < CB; ko++) /* v1: one 8-wide FMA     */
                                        acc[ko] += ip[ci] * wq[ci * CB + ko];
                            }

                    float *op = out + blk_idx(Kb, OHp, OWp, n, kb, oh + d->out_halo, ow + d->out_halo);
                    for (int ko = 0; ko < CB; ko++)
                        op[ko] = d->fuse_silu ? silu_scalar(acc[ko]) : acc[ko];
                }
}

/* ------------------------------------------------------------------ v1 ---- */
#ifdef FASTYOLO_HAVE_AVX2
/*
 * One call computes a 16-channel x 5-pixel output tile, reducing over ALL
 * input channels and the 3x3 window, then stores it once.
 *
 *   ip      input at (cb=0, row oh, col ow) in padded coordinates
 *   w0, w1  packed weights for output blocks kb and kb+1 at cb=0
 *   o0, o1  output for blocks kb and kb+1 at (oh, ow)
 */
static inline __attribute__((always_inline)) void
ukr_2x5(int Cb, const float *ip, size_t in_plane, int Wp,
        const float *w0, const float *w1, const float *b,
        float *o0, float *o1, int fuse_silu) {
    __m256 c0[TILE_OW], c1[TILE_OW]; /* 10 accumulators */
    const __m256 bias0 = _mm256_loadu_ps(b), bias1 = _mm256_loadu_ps(b + CB);
    for (int j = 0; j < TILE_OW; j++) { c0[j] = bias0; c1[j] = bias1; }

    for (int cb = 0; cb < Cb; cb++) {
        const float *icb = ip + cb * in_plane;
        const float *wa = w0 + (size_t)cb * 9 * CB * CB;
        const float *wb = w1 + (size_t)cb * 9 * CB * CB;
        for (int r = 0; r < 3; r++)
            for (int s = 0; s < 3; s++) {
                const float *row = icb + ((size_t)r * Wp + s) * CB;
                const float *wra = wa + (r * 3 + s) * CB * CB;
                const float *wrb = wb + (r * 3 + s) * CB * CB;
#pragma GCC unroll 8
                for (int ci = 0; ci < CB; ci++) {
                    const __m256 wv0 = _mm256_load_ps(wra + ci * CB); /* 8 output channels */
                    const __m256 wv1 = _mm256_load_ps(wrb + ci * CB);
#pragma GCC unroll 5
                    for (int j = 0; j < TILE_OW; j++) {
                        const __m256 x = _mm256_broadcast_ss(row + j * CB + ci); /* 1 input value */
                        c0[j] = _mm256_fmadd_ps(x, wv0, c0[j]);
                        c1[j] = _mm256_fmadd_ps(x, wv1, c1[j]);
                    }
                }
            }
    }
    for (int j = 0; j < TILE_OW; j++) {
        _mm256_store_ps(o0 + j * CB, epilogue_ps(c0[j], fuse_silu));
        _mm256_store_ps(o1 + j * CB, epilogue_ps(c1[j], fuse_silu));
    }
}

void conv3x3_s1_v1(const conv_desc_t *d, const float *in, const float *wp,
                   const float *bias, float *out) {
    const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);
    if (d->R != 3 || d->S != 3 || d->stride != 1 || d->in_halo != d->pad ||
        Kb % TILE_KB != 0 || d->OW % TILE_OW != 0) {
        static int warned = 0;
        if (!warned++) fprintf(stderr, "conv3x3_s1_v1: shape not supported, using generic\n");
        conv_blk_generic(d, in, wp, bias, out);
        return;
    }
    const int Hp = d->H + 2 * d->in_halo, Wp = d->W + 2 * d->in_halo;
    const int OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo;
    const size_t in_plane = (size_t)Hp * Wp * CB, out_plane = (size_t)OHp * OWp * CB;
    const size_t w_kb = (size_t)Cb * 9 * CB * CB; /* floats per output-channel block */
    const int N = d->N, OH = d->OH, OW = d->OW, KP = Kb / TILE_KB, oh_ = d->out_halo;

#pragma omp parallel for collapse(3) schedule(static)
    for (int n = 0; n < N; n++)
        for (int kp = 0; kp < KP; kp++)
            for (int oh = 0; oh < OH; oh++) {
                const int kb = kp * TILE_KB;
                const float *irow = in + blk_idx(Cb, Hp, Wp, n, 0, oh, 0);
                float *orow = out + blk_idx(Kb, OHp, OWp, n, kb, oh + oh_, oh_);
                for (int ow = 0; ow < OW; ow += TILE_OW)
                    ukr_2x5(Cb, irow + (size_t)ow * CB, in_plane, Wp,
                            wp + kb * w_kb, wp + (kb + 1) * w_kb, bias + kb * CB,
                            orow + (size_t)ow * CB, orow + out_plane + (size_t)ow * CB,
                            d->fuse_silu);
            }
}
#else /* no AVX2: keep the build working on non-Broadwell machines */
void conv3x3_s1_v1(const conv_desc_t *d, const float *in, const float *wp,
                   const float *bias, float *out) {
    conv3x3_s1_v0(d, in, wp, bias, out);
}
#endif

} // namespace fastyolo
