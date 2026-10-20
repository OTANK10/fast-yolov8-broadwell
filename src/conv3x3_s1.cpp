/*
 * conv3x3_s1.cpp  --  3x3, stride-1, pad-1 convolution (C2f bottlenecks + Detect head).
 * Owner: Om.   Read docs/STRIDE1_GUIDE.md alongside this file.
 *
 * The versions are a staircase. Each one removes exactly one bottleneck, so
 * each step is one slide: what limited the previous version, what changed,
 * what the number did.
 *
 *   v0  scalar C on the blocked layouts. Correct, slow, and the loop order
 *       every later version keeps.
 *   v1  SIMD: the 8 output channels of a block live in ONE ymm register and
 *       one FMA updates all 8. Only one accumulator, so every FMA waits for
 *       the previous one: latency-bound at ~1/10 of peak.
 *   v2  register tile: 2 output-channel blocks x 5 output pixels = 10
 *       independent accumulators = 2 FMA ports x 5-cycle latency. Single
 *       thread. This is the "microkernel".
 *   v3  v2 + OpenMP across cores.
 *
 *   s1_t<KB>x<OW>  the same microkernel with other tile shapes, single thread,
 *       for the "how many accumulators do I need?" experiment
 *       (scripts/tile_sweep.sh). v1 is t1x1 and v2 is t2x5.
 *
 * Next (yours): v4 cache-blocks the input-channel loop for C >= 64.
 */
#include <cstdio>
#include "common.h"
#include "epilogue.h"
#include "kernels.h"

namespace fastyolo {

/* ------------------------------------------------------------------ v0 ---- */
/* out[kb][oh][ow][ko] = bias + sum over (cb, r, s, ci) of in[cb][oh+r][ow+s][ci] * W[kb][cb][r][s][ci][ko]
 * Every array index here is the same one the AVX2 versions use; only the
 * innermost "for ko" loop becomes a single vector instruction in v1. */
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
                    float acc[CB]; /* in v1 this becomes ONE ymm register */
                    for (int ko = 0; ko < CB; ko++) acc[ko] = bias[kb * CB + ko];

                    for (int cb = 0; cb < Cb; cb++)
                        for (int r = 0; r < 3; r++)
                            for (int s = 0; s < 3; s++) {
                                const float *ip = in + blk_idx(Cb, Hp, Wp, n, cb, oh + r + off, ow + s + off);
                                const float *wq = wp + ((((size_t)kb * Cb + cb) * 3 + r) * 3 + s) * CB * CB;
                                for (int ci = 0; ci < CB; ci++)     /* v1: broadcast ip[ci]           */
                                    for (int ko = 0; ko < CB; ko++) /* v1: one 8-wide FMA             */
                                        acc[ko] += ip[ci] * wq[ci * CB + ko];
                            }

                    float *op = out + blk_idx(Kb, OHp, OWp, n, kb, oh + d->out_halo, ow + d->out_halo);
                    for (int ko = 0; ko < CB; ko++)
                        op[ko] = d->fuse_silu ? silu_scalar(acc[ko]) : acc[ko];
                }
}

#ifdef FASTYOLO_HAVE_AVX2
/* ---------------------------------------------------------- microkernel ---- */
/*
 * One call computes a KB-block x OW-pixel output tile:
 *     KB * 8 output channels  x  OW consecutive output pixels in one row
 *   = KB * OW ymm accumulators, each holding 8 outputs.
 * It reduces over ALL input channels and the whole 3x3 window, then stores
 * each accumulator exactly once.
 *
 * Per inner step (one input channel ci, one tap (r, s)) it does:
 *   KB  loads      of 8 weights        (one per output-channel block)
 *   OW  broadcasts of 1 input value    (one per output pixel)
 *   KB * OW FMAs                       (every weight meets every input)
 * Registers: KB*OW accumulators + KB weights + 1 broadcast. Must be <= 16.
 *
 * KB and OW are template parameters, so the compiler fully unrolls the two
 * inner loops and keeps every accumulator in its own register. With runtime
 * loop bounds it would have to keep them in memory instead.
 *
 *   ip      input at (cb=0, padded row oh, padded col ow)
 *   w       packed weights for output block kb at cb=0
 *   w_kb    floats between consecutive output blocks in the packed weights
 *   o       output at (block kb, row oh, col ow); o_kb = floats between blocks
 */
template <int KB, int OW>
static inline __attribute__((always_inline)) void
ukr(int Cb, const float *ip, size_t in_plane, int Wp, const float *w, size_t w_kb,
    const float *b, float *o, size_t o_kb, int fuse_silu) {
    static_assert(KB * OW + KB + 1 <= 16, "tile needs more than 16 ymm registers");
    __m256 acc[KB][OW];
#pragma GCC unroll 4
    for (int k = 0; k < KB; k++) {
        const __m256 bv = _mm256_loadu_ps(b + k * CB); /* bias = initial value: free add */
#pragma GCC unroll 12
        for (int j = 0; j < OW; j++) acc[k][j] = bv;
    }

    for (int cb = 0; cb < Cb; cb++) {
        const float *icb = ip + cb * in_plane;
        const float *wcb = w + (size_t)cb * 9 * CB * CB;
        for (int r = 0; r < 3; r++)
            for (int s = 0; s < 3; s++) {
                const float *row = icb + ((size_t)r * Wp + s) * CB; /* pixel (oh+r, ow+s) */
                const float *wrs = wcb + (r * 3 + s) * CB * CB;
#pragma GCC unroll 8
                for (int ci = 0; ci < CB; ci++) {
                    __m256 wv[KB];
#pragma GCC unroll 4
                    for (int k = 0; k < KB; k++) wv[k] = _mm256_load_ps(wrs + k * w_kb + ci * CB);
#pragma GCC unroll 12
                    for (int j = 0; j < OW; j++) {
                        const __m256 x = _mm256_broadcast_ss(row + j * CB + ci);
#pragma GCC unroll 4
                        for (int k = 0; k < KB; k++) acc[k][j] = _mm256_fmadd_ps(x, wv[k], acc[k][j]);
                    }
                }
            }
    }
#pragma GCC unroll 4
    for (int k = 0; k < KB; k++)
#pragma GCC unroll 12
        for (int j = 0; j < OW; j++)
            _mm256_store_ps(o + k * o_kb + j * CB, epilogue_ps(acc[k][j], fuse_silu));
}

/* Walks the whole output in KB x OW tiles. `threads` = false keeps it on one
 * core so the tile experiment measures the microkernel alone. */
template <int KB, int OW>
static void conv3x3_s1_tiled(const conv_desc_t *d, const float *in, const float *wp,
                             const float *bias, float *out, bool threads) {
    const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);
    if (d->R != 3 || d->S != 3 || d->stride != 1 || d->in_halo != d->pad ||
        Kb % KB != 0 || d->OW % OW != 0) {
        static bool warned = false;
        if (!warned) { warned = true; std::fprintf(stderr, "conv3x3_s1 %dx%d: shape not supported, using generic\n", KB, OW); }
        conv_blk_generic(d, in, wp, bias, out);
        return;
    }
    const int Hp = d->H + 2 * d->in_halo, Wp = d->W + 2 * d->in_halo;
    const int OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo;
    const size_t in_plane = (size_t)Hp * Wp * CB, out_plane = (size_t)OHp * OWp * CB;
    const size_t w_kb = (size_t)Cb * 9 * CB * CB; /* floats per output-channel block */
    const int N = d->N, OH = d->OH, OWd = d->OW, KT = Kb / KB, oh_ = d->out_halo;

#pragma omp parallel for collapse(3) schedule(static) if (threads)
    for (int n = 0; n < N; n++)
        for (int kt = 0; kt < KT; kt++)
            for (int oh = 0; oh < OH; oh++) {
                const int kb = kt * KB;
                const float *irow = in + blk_idx(Cb, Hp, Wp, n, 0, oh, 0);
                float *orow = out + blk_idx(Kb, OHp, OWp, n, kb, oh + oh_, oh_);
                for (int ow = 0; ow < OWd; ow += OW)
                    ukr<KB, OW>(Cb, irow + (size_t)ow * CB, in_plane, Wp, wp + kb * w_kb, w_kb,
                                bias + kb * CB, orow + (size_t)ow * CB, out_plane, d->fuse_silu);
            }
}

/* ------------------------------------------------------------- versions ---- */
void conv3x3_s1_v1(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_tiled<1, 1>(d, in, wp, bias, out, false); /* 1 accumulator: latency-bound */
}
void conv3x3_s1_v2(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_tiled<2, 5>(d, in, wp, bias, out, false); /* 10 accumulators, 1 core */
}
void conv3x3_s1_v3(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_tiled<2, 5>(d, in, wp, bias, out, true);  /* 10 accumulators, all cores */
}

/* Tile-shape sweep. Every yolov8n stride-1 width (160, 80, 40, 20) divides by
 * 1, 2, 4, 5 and 10; every K/8 (2, 4, 8, 10, 16) divides by 1 and 2. */
#define S1_TILE(KB, OW) \
    void conv3x3_s1_t##KB##x##OW(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) { \
        conv3x3_s1_tiled<KB, OW>(d, in, wp, bias, out, false); }
S1_TILE(1, 1) S1_TILE(1, 2) S1_TILE(1, 4) S1_TILE(1, 5) S1_TILE(1, 10)
S1_TILE(2, 1) S1_TILE(2, 2) S1_TILE(2, 4) S1_TILE(2, 5)

#else /* no AVX2 (e.g. a Mac): every version falls back to v0 so the build and tests still work */
#define S1_FALLBACK(name) \
    void name(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) { \
        conv3x3_s1_v0(d, in, wp, bias, out); }
S1_FALLBACK(conv3x3_s1_v1) S1_FALLBACK(conv3x3_s1_v2) S1_FALLBACK(conv3x3_s1_v3)
S1_FALLBACK(conv3x3_s1_t1x1) S1_FALLBACK(conv3x3_s1_t1x2) S1_FALLBACK(conv3x3_s1_t1x4)
S1_FALLBACK(conv3x3_s1_t1x5) S1_FALLBACK(conv3x3_s1_t1x10) S1_FALLBACK(conv3x3_s1_t2x1)
S1_FALLBACK(conv3x3_s1_t2x2) S1_FALLBACK(conv3x3_s1_t2x4) S1_FALLBACK(conv3x3_s1_t2x5)
#endif

} // namespace fastyolo
