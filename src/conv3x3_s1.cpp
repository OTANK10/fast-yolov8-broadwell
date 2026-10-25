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
 *   v4  2 x 6 tile = 12 accumulators: 2 more than latency x throughput, so a
 *       late load or a busy port no longer costs an FMA slot (v2 sat at 80%
 *       with zero slack). No width is a multiple of 6, so each row ends in a
 *       tail of smaller tiles. v4b: the 4+4 tail variant.
 *   v5  v4 + weights packed in the microkernel's reading order (one stream).
 *   v6  v2's 2x5 tile, all 9 taps unrolled: every load is [base + constant],
 *       no per-tap pointer math or branch competing with the FMAs. v6b
 *       unrolls only s (smaller loop body, fits the uop cache).
 *   v7  v6 + software pipelining: step ci+1's weight loads issue before step
 *       ci's FMAs (15 registers). v7b = v6b + the same. v7mt = v7 on all cores.
 *       Measured on ece028 (GCC 8.5): v6 71%, v6b 61%, v7 73% vs v2 77%. v6's
 *       body overflows the uop cache (38% of uops from legacy decode); GCC
 *       spills accumulators to the stack in all of them. Kept as evidence.
 *   v8  4 x 3 broadcast-stationary tile: 12 accumulators, 7 loads / 12 FMAs
 *       (v2: 7 / 10), so fewer issued uops per FMA. GCC spills it, so v8a
 *       writes the inner step in inline asm with the registers fixed by hand.
 *       v8mt = v8a on all cores. v2a = v2's 2x5 tile in the same inline asm
 *       (control: separates register allocation from tile shape).
 *       Measured: v8a 80-82% on the wide head layers (best yet), but loses on
 *       20/40-wide rows (4x2/4x1 tails) and C >= 128 (4-block weights > L2).
 *   v8b v8a with row remainders done by 2x5 tiles (no small tiles).
 *   v9  per-layer dispatch: v8b for C <= 80, v2 for C >= 128. v9mt = all cores.
 *
 *   s1_t<KB>x<OW>  the same microkernel with other tile shapes, single thread,
 *       for the "how many accumulators do I need?" experiment
 *       (scripts/tile_sweep.sh). v1 is t1x1 and v2 is t2x5.
 *
 * Next: cache-block the input-channel loop for the C = 256 head layers.
 * Every version is the same algorithm and FLOP count; only the instruction
 * stream changes, so differences are purely microkernel quality.
 */
#include <cstdio>
#include <cstdlib>
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
 *
 * PACKED (v5) selects the weight order. false: the team layout
 * [Kb][Cb][3][3][8 ic][8 oc], so the KB weight vectors of one step live w_kb
 * floats apart (KB separate streams). true: pack_weights_s1_kb2's order
 * [Kb/KB][Cb][3][3][8 ic][KB][8 oc], so they are adjacent and the whole
 * microkernel reads its weights as ONE contiguous stream, in exactly the
 * order it uses them.
 */
template <int KB, int OW, bool PACKED = false>
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

    constexpr size_t W_TAP = PACKED ? CB * KB * CB : CB * CB; /* floats per (cb, r, s) step */
    constexpr size_t W_CI = PACKED ? KB * CB : CB;            /* floats per input channel   */
    for (int cb = 0; cb < Cb; cb++) {
        const float *icb = ip + cb * in_plane;
        const float *wcb = w + (size_t)cb * 9 * W_TAP;
        for (int r = 0; r < 3; r++)
            for (int s = 0; s < 3; s++) {
                const float *row = icb + ((size_t)r * Wp + s) * CB; /* pixel (oh+r, ow+s) */
                const float *wrs = wcb + (r * 3 + s) * W_TAP;
#pragma GCC unroll 8
                for (int ci = 0; ci < CB; ci++) {
                    __m256 wv[KB];
#pragma GCC unroll 4
                    for (int k = 0; k < KB; k++)
                        wv[k] = _mm256_load_ps(wrs + (PACKED ? k * CB : k * w_kb) + ci * W_CI);
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

/* ------------------------------------------------- main tile + row tail ---- */
/*
 * v4/v5 driver. The main tile is KB x OW_MAIN, but no yolov8n width (160, 80,
 * 40, 20) is a multiple of 6, so each row is split as
 *     [ OW_MAIN | OW_MAIN | ... | tail ]        tail = OW % OW_MAIN pixels,
 * and the tail is done with smaller tiles of the same microkernel: 4, then 2,
 * then 1 pixel(s). Same KB, so the weights are shared with the main tiles.
 * SPLIT2: when the tail would be 2 pixels, give back one main tile and do
 * 4 + 4 instead of 6 + 2 (a 2x2 tile has only 4 accumulators, so it runs at
 * ~40% of peak; a 2x4 tile has 8 and runs at ~65%). Which is faster is a
 * measurement, so both exist (v4 = 6+2, v4b = 4+4).
 */
template <int KB, bool PACKED>
static inline void ukr_tail(int ow, int OWd, int Cb, const float *irow, size_t in_plane, int Wp,
                            const float *w, size_t w_kb, const float *b, float *orow, size_t o_kb, int fs) {
    while (OWd - ow >= 4) {
        ukr<KB, 4, PACKED>(Cb, irow + (size_t)ow * CB, in_plane, Wp, w, w_kb, b, orow + (size_t)ow * CB, o_kb, fs);
        ow += 4;
    }
    if (OWd - ow >= 2) {
        ukr<KB, 2, PACKED>(Cb, irow + (size_t)ow * CB, in_plane, Wp, w, w_kb, b, orow + (size_t)ow * CB, o_kb, fs);
        ow += 2;
    }
    if (OWd - ow == 1)
        ukr<KB, 1, PACKED>(Cb, irow + (size_t)ow * CB, in_plane, Wp, w, w_kb, b, orow + (size_t)ow * CB, o_kb, fs);
}

template <int KB, int OW_MAIN, bool PACKED, bool SPLIT2>
static void conv3x3_s1_rowtail(const conv_desc_t *d, const float *in, const float *wp,
                               const float *bias, float *out, bool threads) {
    const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);
    if (d->R != 3 || d->S != 3 || d->stride != 1 || d->in_halo != d->pad || Kb % KB != 0) {
        if (PACKED) { /* weights are in v5's order, which conv_blk_generic cannot read */
            std::fprintf(stderr, "conv3x3_s1 packed: shape not supported (needs 3x3 s1, K/8 %% %d == 0)\n", KB);
            std::abort();
        }
        static bool warned = false;
        if (!warned) { warned = true; std::fprintf(stderr, "conv3x3_s1 %dx%d: shape not supported, using generic\n", KB, OW_MAIN); }
        conv_blk_generic(d, in, wp, bias, out);
        return;
    }
    const int Hp = d->H + 2 * d->in_halo, Wp = d->W + 2 * d->in_halo;
    const int OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo;
    const size_t in_plane = (size_t)Hp * Wp * CB, out_plane = (size_t)OHp * OWp * CB;
    const size_t w_kb = (size_t)Cb * 9 * CB * CB; /* floats per output-channel block (both orders) */
    const int N = d->N, OH = d->OH, OWd = d->OW, KT = Kb / KB, oh_ = d->out_halo;

    int main_end = OWd - OWd % OW_MAIN;                               /* pixels done by main tiles */
    if (SPLIT2 && OWd % OW_MAIN == 2 && main_end >= OW_MAIN) main_end -= OW_MAIN; /* 6+2 -> 4+4 */

#pragma omp parallel for collapse(3) schedule(static) if (threads)
    for (int n = 0; n < N; n++)
        for (int kt = 0; kt < KT; kt++)
            for (int oh = 0; oh < OH; oh++) {
                const int kb = kt * KB;
                const float *irow = in + blk_idx(Cb, Hp, Wp, n, 0, oh, 0);
                float *orow = out + blk_idx(Kb, OHp, OWp, n, kb, oh + oh_, oh_);
                const float *w = wp + kb * w_kb, *b = bias + kb * CB;
                for (int ow = 0; ow < main_end; ow += OW_MAIN)
                    ukr<KB, OW_MAIN, PACKED>(Cb, irow + (size_t)ow * CB, in_plane, Wp, w, w_kb, b,
                                             orow + (size_t)ow * CB, out_plane, d->fuse_silu);
                ukr_tail<KB, PACKED>(main_end, OWd, Cb, irow, in_plane, Wp, w, w_kb, b, orow, out_plane, d->fuse_silu);
            }
}

/* ----------------------------------------------- v6/v7: tuned microkernel ---- */
/*
 * Same 2x5 tile, same layouts, same FLOPs as v2. What changes is the
 * instruction stream inside the microkernel:
 *
 *   UT    taps unrolled. v2 runs the 3x3 (r, s) taps as two real loops, so each
 *         tap pays pointer arithmetic (r*Wp + s), a compare and a branch, and
 *         those integer uops compete with the FMAs for ports 0/1 and for
 *         scheduler entries. UT = 3 unrolls s (loop over r only); UT = 9
 *         unrolls all nine taps, so every load is [base + constant] from three
 *         row pointers set once per input-channel block. The 8 input channels
 *         are always unrolled (as in v2).
 *   PIPE  software pipelining. The 2 weight vectors for step ci+1 are loaded
 *         BEFORE the FMAs of step ci, so their load latency overlaps this
 *         step's work instead of sitting in front of the next step's FMAs.
 *         Costs 2 registers: 10 acc + 2x2 weights + 1 broadcast = 15. (Also
 *         double-buffering the broadcast needs all 16, and GCC 8 then spills
 *         accumulators to the stack, which is slower; checked in the asm.)
 *
 * The weight pointer walks the team layout front to back: for one output
 * block, the (cb, r, s, ci) steps are contiguous 8-float vectors, so every
 * weight address is wcb + constant.
 */
template <int UT, bool PIPE>
static inline __attribute__((always_inline)) void
ukr_2x5_tuned(int Cb, const float *ip, size_t in_plane, size_t in_row, const float *w, size_t w_kb,
              const float *b, float *o, size_t o_kb, int fuse_silu) {
    static_assert(UT == 3 || UT == 9, "UT = taps unrolled: 3 (s) or 9 (r and s)");
    constexpr int OW = 5;
    const __m256 b0 = _mm256_loadu_ps(b), b1 = _mm256_loadu_ps(b + CB);
    __m256 a00 = b0, a01 = b0, a02 = b0, a03 = b0, a04 = b0; /* block kb   */
    __m256 a10 = b1, a11 = b1, a12 = b1, a13 = b1, a14 = b1; /* block kb+1 */

    /* one (tap, ci) step: 2 weight loads, 5 broadcasts, 10 FMAs */
#define S1_FMA2(X, J) a0##J = _mm256_fmadd_ps(X, w0, a0##J); a1##J = _mm256_fmadd_ps(X, w1, a1##J);
    for (int cb = 0; cb < Cb; cb++) {
        const float *icb = ip + (size_t)cb * in_plane;
        const float *wa = w + (size_t)cb * 9 * CB * CB, *wb = wa + w_kb;
        for (int t0 = 0; t0 < 9; t0 += UT) {
#pragma GCC unroll 9
            for (int tt = 0; tt < UT; tt++) {
                const int tap = t0 + tt, r = tap / 3, s = tap % 3;
                const float *row = icb + (size_t)r * in_row + s * CB; /* pixel (oh+r, ow+s) */
                const float *wta = wa + tap * CB * CB, *wtb = wb + tap * CB * CB;
                if (!PIPE) {
#pragma GCC unroll 8
                    for (int ci = 0; ci < CB; ci++) {
                        const __m256 w0 = _mm256_load_ps(wta + ci * CB), w1 = _mm256_load_ps(wtb + ci * CB);
                        __m256 x;
                        x = _mm256_broadcast_ss(row + 0 * CB + ci); S1_FMA2(x, 0)
                        x = _mm256_broadcast_ss(row + 1 * CB + ci); S1_FMA2(x, 1)
                        x = _mm256_broadcast_ss(row + 2 * CB + ci); S1_FMA2(x, 2)
                        x = _mm256_broadcast_ss(row + 3 * CB + ci); S1_FMA2(x, 3)
                        x = _mm256_broadcast_ss(row + 4 * CB + ci); S1_FMA2(x, 4)
                    }
                } else {
                    /* prologue: weights for ci = 0 */
                    __m256 w0 = _mm256_load_ps(wta), w1 = _mm256_load_ps(wtb);
#pragma GCC unroll 8
                    for (int ci = 0; ci < CB; ci++) {
                        __m256 n0 = w0, n1 = w1, x;
                        if (ci + 1 < CB) { /* next step's weights go in before this step's FMAs */
                            n0 = _mm256_load_ps(wta + (ci + 1) * CB);
                            n1 = _mm256_load_ps(wtb + (ci + 1) * CB);
                        }
                        x = _mm256_broadcast_ss(row + 0 * CB + ci); S1_FMA2(x, 0)
                        x = _mm256_broadcast_ss(row + 1 * CB + ci); S1_FMA2(x, 1)
                        x = _mm256_broadcast_ss(row + 2 * CB + ci); S1_FMA2(x, 2)
                        x = _mm256_broadcast_ss(row + 3 * CB + ci); S1_FMA2(x, 3)
                        x = _mm256_broadcast_ss(row + 4 * CB + ci); S1_FMA2(x, 4)
                        w0 = n0; w1 = n1;
                    }
                }
            }
        }
    }
#undef S1_FMA2
    (void)OW;
    _mm256_store_ps(o + 0 * CB, epilogue_ps(a00, fuse_silu));
    _mm256_store_ps(o + 1 * CB, epilogue_ps(a01, fuse_silu));
    _mm256_store_ps(o + 2 * CB, epilogue_ps(a02, fuse_silu));
    _mm256_store_ps(o + 3 * CB, epilogue_ps(a03, fuse_silu));
    _mm256_store_ps(o + 4 * CB, epilogue_ps(a04, fuse_silu));
    _mm256_store_ps(o + o_kb + 0 * CB, epilogue_ps(a10, fuse_silu));
    _mm256_store_ps(o + o_kb + 1 * CB, epilogue_ps(a11, fuse_silu));
    _mm256_store_ps(o + o_kb + 2 * CB, epilogue_ps(a12, fuse_silu));
    _mm256_store_ps(o + o_kb + 3 * CB, epilogue_ps(a13, fuse_silu));
    _mm256_store_ps(o + o_kb + 4 * CB, epilogue_ps(a14, fuse_silu));
}

/* Driver: identical to conv3x3_s1_tiled<2, 5> except for the microkernel. */
template <int UT, bool PIPE>
static void conv3x3_s1_tuned(const conv_desc_t *d, const float *in, const float *wp,
                             const float *bias, float *out, bool threads) {
    const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);
    if (d->R != 3 || d->S != 3 || d->stride != 1 || d->in_halo != d->pad || Kb % 2 != 0 || d->OW % 5 != 0) {
        static bool warned = false;
        if (!warned) { warned = true; std::fprintf(stderr, "conv3x3_s1 tuned: shape not supported, using generic\n"); }
        conv_blk_generic(d, in, wp, bias, out);
        return;
    }
    const int Hp = d->H + 2 * d->in_halo, Wp = d->W + 2 * d->in_halo;
    const int OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo;
    const size_t in_plane = (size_t)Hp * Wp * CB, out_plane = (size_t)OHp * OWp * CB;
    const size_t in_row = (size_t)Wp * CB, w_kb = (size_t)Cb * 9 * CB * CB;
    const int N = d->N, OH = d->OH, OWd = d->OW, KT = Kb / 2, oh_ = d->out_halo;

#pragma omp parallel for collapse(3) schedule(static) if (threads)
    for (int n = 0; n < N; n++)
        for (int kt = 0; kt < KT; kt++)
            for (int oh = 0; oh < OH; oh++) {
                const int kb = kt * 2;
                const float *irow = in + blk_idx(Cb, Hp, Wp, n, 0, oh, 0);
                float *orow = out + blk_idx(Kb, OHp, OWp, n, kb, oh + oh_, oh_);
                for (int ow = 0; ow < OWd; ow += 5)
                    ukr_2x5_tuned<UT, PIPE>(Cb, irow + (size_t)ow * CB, in_plane, in_row, wp + kb * w_kb, w_kb,
                                            bias + kb * CB, orow + (size_t)ow * CB, out_plane, d->fuse_silu);
            }
}

/* ------------------------------------- v8: broadcast-stationary 4 x 3 tile ---- */
/*
 * v2-v7 keep the 2 weight vectors in registers and stream the 5 broadcasts:
 * 7 loads per 10 FMAs, about 1.76 instructions per FMA in GCC 8.5's loop, so
 * peak needs ~3.5 issued uops/cycle of Broadwell's 4 and we sustain ~3.
 * v8 flips it: OW = 3 broadcasts stay in registers and the KB = 4 weight
 * vectors stream through ONE register:
 *     per step: 3 broadcasts + 4 weight loads, 12 FMAs  -> 7 loads / 12 FMAs
 *     registers: 12 accumulators + 3 broadcasts + 1 weight = 16
 * 12 accumulators also give 2 chains of slack over latency x throughput.
 * Every weight vector is used 3 times (once per pixel) straight after its
 * load, which keeps register pressure flat (the order that made GCC spill in
 * v6/v7 was 5 broadcasts live at once on top of 12 other values).
 * Shapes: K/8 % 4 != 0 (the 16- and 80-channel layers) does its last 2 blocks
 * with v2's 2x5 microkernel; widths are not multiples of 3, so each row ends
 * with a 4x2 or 4x1 tile.
 */
template <int KB, int OW>
static inline __attribute__((always_inline)) void
ukr_bs(int Cb, const float *ip, size_t in_plane, size_t in_row, const float *w, size_t w_kb,
       const float *b, float *o, size_t o_kb, int fuse_silu) {
    static_assert(KB * OW + OW + 1 <= 16, "tile needs more than 16 ymm registers");
    __m256 acc[KB][OW];
#pragma GCC unroll 4
    for (int k = 0; k < KB; k++) {
        const __m256 bv = _mm256_loadu_ps(b + k * CB);
#pragma GCC unroll 3
        for (int j = 0; j < OW; j++) acc[k][j] = bv;
    }
    for (int cb = 0; cb < Cb; cb++) {
        const float *icb = ip + (size_t)cb * in_plane;
        const float *wcb = w + (size_t)cb * 9 * CB * CB;
        for (int r = 0; r < 3; r++)
            for (int s = 0; s < 3; s++) {
                const float *row = icb + (size_t)r * in_row + s * CB;
                const float *wrs = wcb + (r * 3 + s) * CB * CB;
#pragma GCC unroll 8
                for (int ci = 0; ci < CB; ci++) {
                    __m256 x[OW];
#pragma GCC unroll 3
                    for (int j = 0; j < OW; j++) x[j] = _mm256_broadcast_ss(row + j * CB + ci);
#pragma GCC unroll 4
                    for (int k = 0; k < KB; k++) {
                        const __m256 wv = _mm256_load_ps(wrs + k * w_kb + ci * CB);
#pragma GCC unroll 3
                        for (int j = 0; j < OW; j++) acc[k][j] = _mm256_fmadd_ps(x[j], wv, acc[k][j]);
                    }
                }
            }
    }
#pragma GCC unroll 4
    for (int k = 0; k < KB; k++)
#pragma GCC unroll 3
        for (int j = 0; j < OW; j++)
            _mm256_store_ps(o + k * o_kb + j * CB, epilogue_ps(acc[k][j], fuse_silu));
}

/*
 * v8a: the same 4 x 3 tile with its inner step in inline assembly. GCC (8 and
 * 13) cannot keep 12 accumulators + 3 broadcasts + 1 weight in the 16 ymm
 * registers and spills to the stack, which is exactly what sank v6/v7. Here
 * the allocation is fixed by hand (BLIS-style): the 12 accumulators are "+x"
 * operands, so the compiler must give each its own register, and the asm uses
 * ymm12-14 for the broadcasts and ymm15 for the weight (declared clobbered).
 * One asm block = one tap (8 input channels): 24 broadcasts, 32 weight loads,
 * 96 FMAs, 152 instructions, about v2's loop size, so it stays in the uop cache.
 * Weight block k of the tile is at w + k*S bytes (S = floats per block * 4).
 */
static inline __attribute__((always_inline)) void
ukr_4x3_asm(int Cb, const float *ip, size_t in_plane, size_t in_row, const float *w, size_t w_kb,
            const float *b, float *o, size_t o_kb, int fuse_silu) {
    __m256 a00, a01, a02, a10, a11, a12, a20, a21, a22, a30, a31, a32;
    {
        const __m256 b0 = _mm256_loadu_ps(b), b1 = _mm256_loadu_ps(b + CB);
        const __m256 b2 = _mm256_loadu_ps(b + 2 * CB), b3 = _mm256_loadu_ps(b + 3 * CB);
        a00 = a01 = a02 = b0; a10 = a11 = a12 = b1; a20 = a21 = a22 = b2; a30 = a31 = a32 = b3;
    }
    const long S = (long)(w_kb * sizeof(float));
    for (int cb = 0; cb < Cb; cb++) {
        const float *icb = ip + (size_t)cb * in_plane;
        const float *wcb = w + (size_t)cb * 9 * CB * CB;
        for (int r = 0; r < 3; r++)
            for (int s = 0; s < 3; s++) {
                const float *row = icb + (size_t)r * in_row + s * CB;
                const float *wt = wcb + (r * 3 + s) * CB * CB;
                const float *w3 = wt + 3 * w_kb;
                __asm__ volatile(
        /* ci = 0: 3 broadcasts, then per weight block 1 load + 3 FMAs */
        "vbroadcastss 0(%[row]), %%ymm12\n\t"
        "vbroadcastss 32(%[row]), %%ymm13\n\t"
        "vbroadcastss 64(%[row]), %%ymm14\n\t"
        "vmovaps 0(%[w]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a02]\n\t"
        "vmovaps 0(%[w],%[S]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a10]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a12]\n\t"
        "vmovaps 0(%[w],%[S],2), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a20]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a21]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a22]\n\t"
        "vmovaps 0(%[w3]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a30]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a31]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a32]\n\t"
        /* ci = 1: 3 broadcasts, then per weight block 1 load + 3 FMAs */
        "vbroadcastss 4(%[row]), %%ymm12\n\t"
        "vbroadcastss 36(%[row]), %%ymm13\n\t"
        "vbroadcastss 68(%[row]), %%ymm14\n\t"
        "vmovaps 32(%[w]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a02]\n\t"
        "vmovaps 32(%[w],%[S]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a10]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a12]\n\t"
        "vmovaps 32(%[w],%[S],2), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a20]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a21]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a22]\n\t"
        "vmovaps 32(%[w3]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a30]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a31]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a32]\n\t"
        /* ci = 2: 3 broadcasts, then per weight block 1 load + 3 FMAs */
        "vbroadcastss 8(%[row]), %%ymm12\n\t"
        "vbroadcastss 40(%[row]), %%ymm13\n\t"
        "vbroadcastss 72(%[row]), %%ymm14\n\t"
        "vmovaps 64(%[w]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a02]\n\t"
        "vmovaps 64(%[w],%[S]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a10]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a12]\n\t"
        "vmovaps 64(%[w],%[S],2), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a20]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a21]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a22]\n\t"
        "vmovaps 64(%[w3]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a30]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a31]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a32]\n\t"
        /* ci = 3: 3 broadcasts, then per weight block 1 load + 3 FMAs */
        "vbroadcastss 12(%[row]), %%ymm12\n\t"
        "vbroadcastss 44(%[row]), %%ymm13\n\t"
        "vbroadcastss 76(%[row]), %%ymm14\n\t"
        "vmovaps 96(%[w]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a02]\n\t"
        "vmovaps 96(%[w],%[S]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a10]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a12]\n\t"
        "vmovaps 96(%[w],%[S],2), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a20]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a21]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a22]\n\t"
        "vmovaps 96(%[w3]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a30]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a31]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a32]\n\t"
        /* ci = 4: 3 broadcasts, then per weight block 1 load + 3 FMAs */
        "vbroadcastss 16(%[row]), %%ymm12\n\t"
        "vbroadcastss 48(%[row]), %%ymm13\n\t"
        "vbroadcastss 80(%[row]), %%ymm14\n\t"
        "vmovaps 128(%[w]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a02]\n\t"
        "vmovaps 128(%[w],%[S]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a10]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a12]\n\t"
        "vmovaps 128(%[w],%[S],2), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a20]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a21]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a22]\n\t"
        "vmovaps 128(%[w3]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a30]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a31]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a32]\n\t"
        /* ci = 5: 3 broadcasts, then per weight block 1 load + 3 FMAs */
        "vbroadcastss 20(%[row]), %%ymm12\n\t"
        "vbroadcastss 52(%[row]), %%ymm13\n\t"
        "vbroadcastss 84(%[row]), %%ymm14\n\t"
        "vmovaps 160(%[w]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a02]\n\t"
        "vmovaps 160(%[w],%[S]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a10]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a12]\n\t"
        "vmovaps 160(%[w],%[S],2), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a20]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a21]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a22]\n\t"
        "vmovaps 160(%[w3]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a30]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a31]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a32]\n\t"
        /* ci = 6: 3 broadcasts, then per weight block 1 load + 3 FMAs */
        "vbroadcastss 24(%[row]), %%ymm12\n\t"
        "vbroadcastss 56(%[row]), %%ymm13\n\t"
        "vbroadcastss 88(%[row]), %%ymm14\n\t"
        "vmovaps 192(%[w]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a02]\n\t"
        "vmovaps 192(%[w],%[S]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a10]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a12]\n\t"
        "vmovaps 192(%[w],%[S],2), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a20]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a21]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a22]\n\t"
        "vmovaps 192(%[w3]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a30]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a31]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a32]\n\t"
        /* ci = 7: 3 broadcasts, then per weight block 1 load + 3 FMAs */
        "vbroadcastss 28(%[row]), %%ymm12\n\t"
        "vbroadcastss 60(%[row]), %%ymm13\n\t"
        "vbroadcastss 92(%[row]), %%ymm14\n\t"
        "vmovaps 224(%[w]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a02]\n\t"
        "vmovaps 224(%[w],%[S]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a10]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a12]\n\t"
        "vmovaps 224(%[w],%[S],2), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a20]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a21]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a22]\n\t"
        "vmovaps 224(%[w3]), %%ymm15\n\t"
        "vfmadd231ps %%ymm15, %%ymm12, %[a30]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a31]\n\t"
        "vfmadd231ps %%ymm15, %%ymm14, %[a32]\n\t"
                    : [a00] "+x"(a00), [a01] "+x"(a01), [a02] "+x"(a02),
                      [a10] "+x"(a10), [a11] "+x"(a11), [a12] "+x"(a12),
                      [a20] "+x"(a20), [a21] "+x"(a21), [a22] "+x"(a22),
                      [a30] "+x"(a30), [a31] "+x"(a31), [a32] "+x"(a32)
                    : [row] "r"(row), [w] "r"(wt), [S] "r"(S), [w3] "r"(w3)
                    : "xmm12", "xmm13", "xmm14", "xmm15", "memory");
            }
    }
    _mm256_store_ps(o + 0 * CB, epilogue_ps(a00, fuse_silu));
    _mm256_store_ps(o + 1 * CB, epilogue_ps(a01, fuse_silu));
    _mm256_store_ps(o + 2 * CB, epilogue_ps(a02, fuse_silu));
    _mm256_store_ps(o + o_kb + 0 * CB, epilogue_ps(a10, fuse_silu));
    _mm256_store_ps(o + o_kb + 1 * CB, epilogue_ps(a11, fuse_silu));
    _mm256_store_ps(o + o_kb + 2 * CB, epilogue_ps(a12, fuse_silu));
    _mm256_store_ps(o + 2 * o_kb + 0 * CB, epilogue_ps(a20, fuse_silu));
    _mm256_store_ps(o + 2 * o_kb + 1 * CB, epilogue_ps(a21, fuse_silu));
    _mm256_store_ps(o + 2 * o_kb + 2 * CB, epilogue_ps(a22, fuse_silu));
    _mm256_store_ps(o + 3 * o_kb + 0 * CB, epilogue_ps(a30, fuse_silu));
    _mm256_store_ps(o + 3 * o_kb + 1 * CB, epilogue_ps(a31, fuse_silu));
    _mm256_store_ps(o + 3 * o_kb + 2 * CB, epilogue_ps(a32, fuse_silu));
}

/*
 * v2a: control experiment. v2's exact 2 x 5 tile and instruction mix, but the
 * inner step in inline asm with fixed registers (ymm13 broadcast, ymm14/15
 * weights). v2 vs v2a isolates "hand register allocation"; v2a vs v8a isolates
 * "4 x 3 tile, fewer loads per FMA".
 */
static inline __attribute__((always_inline)) void
ukr_2x5_asm(int Cb, const float *ip, size_t in_plane, size_t in_row, const float *w, size_t w_kb,
            const float *b, float *o, size_t o_kb, int fuse_silu) {
    const __m256 b0 = _mm256_loadu_ps(b), b1 = _mm256_loadu_ps(b + CB);
    __m256 a00 = b0, a01 = b0, a02 = b0, a03 = b0, a04 = b0;
    __m256 a10 = b1, a11 = b1, a12 = b1, a13 = b1, a14 = b1;
    const long S = (long)(w_kb * sizeof(float));
    for (int cb = 0; cb < Cb; cb++) {
        const float *icb = ip + (size_t)cb * in_plane;
        const float *wcb = w + (size_t)cb * 9 * CB * CB;
        for (int r = 0; r < 3; r++)
            for (int s = 0; s < 3; s++) {
                const float *row = icb + (size_t)r * in_row + s * CB;
                const float *wt = wcb + (r * 3 + s) * CB * CB;
                __asm__ volatile(
        /* ci = 0: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 0(%[w]), %%ymm14\n\t"
        "vmovaps 0(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 0(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 32(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 64(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 96(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 128(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        /* ci = 1: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 32(%[w]), %%ymm14\n\t"
        "vmovaps 32(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 4(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 36(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 68(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 100(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 132(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        /* ci = 2: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 64(%[w]), %%ymm14\n\t"
        "vmovaps 64(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 8(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 40(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 72(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 104(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 136(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        /* ci = 3: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 96(%[w]), %%ymm14\n\t"
        "vmovaps 96(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 12(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 44(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 76(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 108(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 140(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        /* ci = 4: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 128(%[w]), %%ymm14\n\t"
        "vmovaps 128(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 16(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 48(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 80(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 112(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 144(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        /* ci = 5: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 160(%[w]), %%ymm14\n\t"
        "vmovaps 160(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 20(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 52(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 84(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 116(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 148(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        /* ci = 6: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 192(%[w]), %%ymm14\n\t"
        "vmovaps 192(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 24(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 56(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 88(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 120(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 152(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        /* ci = 7: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 224(%[w]), %%ymm14\n\t"
        "vmovaps 224(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 28(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 60(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 92(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 124(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 156(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
                    : [a00] "+x"(a00), [a01] "+x"(a01), [a02] "+x"(a02), [a03] "+x"(a03), [a04] "+x"(a04),
                      [a10] "+x"(a10), [a11] "+x"(a11), [a12] "+x"(a12), [a13] "+x"(a13), [a14] "+x"(a14)
                    : [row] "r"(row), [w] "r"(wt), [S] "r"(S)
                    : "xmm13", "xmm14", "xmm15", "memory");
            }
    }
    _mm256_store_ps(o + 0 * CB, epilogue_ps(a00, fuse_silu));
    _mm256_store_ps(o + 1 * CB, epilogue_ps(a01, fuse_silu));
    _mm256_store_ps(o + 2 * CB, epilogue_ps(a02, fuse_silu));
    _mm256_store_ps(o + 3 * CB, epilogue_ps(a03, fuse_silu));
    _mm256_store_ps(o + 4 * CB, epilogue_ps(a04, fuse_silu));
    _mm256_store_ps(o + o_kb + 0 * CB, epilogue_ps(a10, fuse_silu));
    _mm256_store_ps(o + o_kb + 1 * CB, epilogue_ps(a11, fuse_silu));
    _mm256_store_ps(o + o_kb + 2 * CB, epilogue_ps(a12, fuse_silu));
    _mm256_store_ps(o + o_kb + 3 * CB, epilogue_ps(a13, fuse_silu));
    _mm256_store_ps(o + o_kb + 4 * CB, epilogue_ps(a14, fuse_silu));
}

static void conv3x3_s1_2x5asm(const conv_desc_t *d, const float *in, const float *wp,
                              const float *bias, float *out) {
    const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);
    if (d->R != 3 || d->S != 3 || d->stride != 1 || d->in_halo != d->pad || Kb % 2 != 0 || d->OW % 5 != 0) {
        conv_blk_generic(d, in, wp, bias, out);
        return;
    }
    const int Hp = d->H + 2 * d->in_halo, Wp = d->W + 2 * d->in_halo;
    const int OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo;
    const size_t in_plane = (size_t)Hp * Wp * CB, out_plane = (size_t)OHp * OWp * CB;
    const size_t in_row = (size_t)Wp * CB, w_kb = (size_t)Cb * 9 * CB * CB;
    for (int n = 0; n < d->N; n++)
        for (int kb = 0; kb < Kb; kb += 2)
            for (int oh = 0; oh < d->OH; oh++) {
                const float *irow = in + blk_idx(Cb, Hp, Wp, n, 0, oh, 0);
                float *orow = out + blk_idx(Kb, OHp, OWp, n, kb, oh + d->out_halo, d->out_halo);
                for (int ow = 0; ow < d->OW; ow += 5)
                    ukr_2x5_asm(Cb, irow + (size_t)ow * CB, in_plane, in_row, wp + kb * w_kb, w_kb,
                                bias + kb * CB, orow + (size_t)ow * CB, out_plane, d->fuse_silu);
            }
}

template <bool ASM, bool TAIL5 = false>
static void conv3x3_s1_bs43(const conv_desc_t *d, const float *in, const float *wp,
                            const float *bias, float *out, bool threads) {
    const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);
    if (d->R != 3 || d->S != 3 || d->stride != 1 || d->in_halo != d->pad || Kb % 2 != 0 ||
        (Kb % 4 != 0 && d->OW % 5 != 0)) {
        static bool warned = false;
        if (!warned) { warned = true; std::fprintf(stderr, "conv3x3_s1 v8: shape not supported, using generic\n"); }
        conv_blk_generic(d, in, wp, bias, out);
        return;
    }
    const int Hp = d->H + 2 * d->in_halo, Wp = d->W + 2 * d->in_halo;
    const int OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo;
    const size_t in_plane = (size_t)Hp * Wp * CB, out_plane = (size_t)OHp * OWp * CB;
    const size_t in_row = (size_t)Wp * CB, w_kb = (size_t)Cb * 9 * CB * CB;
    const int N = d->N, OH = d->OH, OWd = d->OW, oh_ = d->out_halo;
    const int G4 = Kb / 4, G = G4 + (Kb % 4 ? 1 : 0); /* groups of 4 blocks, then one pair */
    /* TAIL5 (v8b): end the 3-wide tiles where the remainder is a multiple of 5,
     * and finish the row with v2's 2x5 tile (two pairs of blocks), so no row
     * ever runs a 4x2 or 4x1 tile: 20 = 15+5, 40 = 30+10, 80 = 75+5, 160 = 150+10. */
    int main_end = OWd - OWd % 3, m5 = main_end;
    if (TAIL5)
        while (m5 > 0 && (OWd - m5) % 5 != 0) m5 -= 3;
    const bool tail5 = TAIL5 && (OWd - m5) % 5 == 0;
    if (tail5) main_end = m5; /* otherwise (e.g. width 7) keep the 4x2 / 4x1 tails */

#pragma omp parallel for collapse(3) schedule(static) if (threads)
    for (int n = 0; n < N; n++)
        for (int g = 0; g < G; g++)
            for (int oh = 0; oh < OH; oh++) {
                const int kb = g * 4;
                const float *irow = in + blk_idx(Cb, Hp, Wp, n, 0, oh, 0);
                float *orow = out + blk_idx(Kb, OHp, OWp, n, kb, oh + oh_, oh_);
                const float *w = wp + kb * w_kb, *b = bias + kb * CB;
                if (g < G4) {
                    for (int ow = 0; ow < main_end; ow += 3) {
                        if (ASM)
                            ukr_4x3_asm(Cb, irow + (size_t)ow * CB, in_plane, in_row, w, w_kb, b,
                                        orow + (size_t)ow * CB, out_plane, d->fuse_silu);
                        else
                            ukr_bs<4, 3>(Cb, irow + (size_t)ow * CB, in_plane, in_row, w, w_kb, b,
                                         orow + (size_t)ow * CB, out_plane, d->fuse_silu);
                    }
                    if (tail5) {
                        for (int ow = main_end; ow < OWd; ow += 5)
                            for (int p = 0; p < 4; p += 2)
                                ukr<2, 5>(Cb, irow + (size_t)ow * CB, in_plane, Wp, w + p * w_kb, w_kb, b + p * CB,
                                          orow + (size_t)ow * CB + p * out_plane, out_plane, d->fuse_silu);
                    } else if (OWd - main_end == 2)
                        ukr_bs<4, 2>(Cb, irow + (size_t)main_end * CB, in_plane, in_row, w, w_kb, b,
                                     orow + (size_t)main_end * CB, out_plane, d->fuse_silu);
                    else if (OWd - main_end == 1)
                        ukr_bs<4, 1>(Cb, irow + (size_t)main_end * CB, in_plane, in_row, w, w_kb, b,
                                     orow + (size_t)main_end * CB, out_plane, d->fuse_silu);
                } else { /* last 2 blocks when K/8 % 4 == 2: v2's 2x5 microkernel */
                    for (int ow = 0; ow < OWd; ow += 5)
                        ukr<2, 5>(Cb, irow + (size_t)ow * CB, in_plane, Wp, w, w_kb, b,
                                  orow + (size_t)ow * CB, out_plane, d->fuse_silu);
                }
            }
}

/* ------------------------------------------------------------- versions ---- */
void conv3x3_s1_v8(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_bs43<false>(d, in, wp, bias, out, false); /* 4x3 broadcast-stationary, intrinsics, 1 core */
}
void conv3x3_s1_v2a(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_2x5asm(d, in, wp, bias, out); /* v2's tile, inner step in inline asm, 1 core */
}
void conv3x3_s1_v8a(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_bs43<true>(d, in, wp, bias, out, false);  /* 4x3, inner step in inline asm, 1 core */
}
void conv3x3_s1_v8b(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_bs43<true, true>(d, in, wp, bias, out, false); /* v8a, row remainders done by 2x5 tiles */
}
/* v9: pick the microkernel per layer. One output block's weights are
 * C*9*8 floats = 288*C bytes, so v8b's 4-block tile streams 1152*C bytes per
 * call: 144 KB at C = 128 and 288 KB at C = 256, against a 256 KB L2 that also
 * holds the input rows. Those weights come from L2 or beyond, and the 4x3 tile
 * reads 4 weight vectors per 12 FMAs (v2: 2 per 10, 1.67x less). Measured:
 * v8a 54-67% vs v2 69-79% at C >= 128; below that v8b's fewer loads per FMA
 * win. Until we cache-block the input channels (phase 2), deep layers use v2. */
void conv3x3_s1_v9(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    if (d->C >= 128) conv3x3_s1_tiled<2, 5>(d, in, wp, bias, out, false);
    else conv3x3_s1_bs43<true, true>(d, in, wp, bias, out, false);
}
void conv3x3_s1_v9mt(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    if (d->C >= 128) conv3x3_s1_tiled<2, 5>(d, in, wp, bias, out, true);
    else conv3x3_s1_bs43<true, true>(d, in, wp, bias, out, true);
}
void conv3x3_s1_v8mt(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_bs43<true>(d, in, wp, bias, out, true);   /* v8a + OpenMP */
}
void conv3x3_s1_v6(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_tuned<9, false>(d, in, wp, bias, out, false); /* all 9 taps unrolled */
}
void conv3x3_s1_v6b(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_tuned<3, false>(d, in, wp, bias, out, false); /* s unrolled, loop over r (smaller code) */
}
void conv3x3_s1_v7(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_tuned<9, true>(d, in, wp, bias, out, false);  /* v6 + software-pipelined loads */
}
void conv3x3_s1_v7b(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_tuned<3, true>(d, in, wp, bias, out, false);  /* v6b + software-pipelined loads */
}
void conv3x3_s1_v7mt(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_tuned<9, true>(d, in, wp, bias, out, true);   /* v7 on all cores (compare with v3) */
}

void conv3x3_s1_v1(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_tiled<1, 1>(d, in, wp, bias, out, false); /* 1 accumulator: latency-bound */
}
void conv3x3_s1_v2(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_tiled<2, 5>(d, in, wp, bias, out, false); /* 10 accumulators, 1 core */
}
void conv3x3_s1_v3(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_tiled<2, 5>(d, in, wp, bias, out, true);  /* 10 accumulators, all cores */
}

void conv3x3_s1_v4(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_rowtail<2, 6, false, false>(d, in, wp, bias, out, false); /* 12 accumulators, 6+2 tail */
}
void conv3x3_s1_v4b(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_rowtail<2, 6, false, true>(d, in, wp, bias, out, false);  /* 12 accumulators, 4+4 tail */
}
void conv3x3_s1_v5(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    conv3x3_s1_rowtail<2, 6, true, false>(d, in, wp, bias, out, false);  /* v4 + microkernel-order weights */
}

/* Tile-shape sweep. Every yolov8n stride-1 width (160, 80, 40, 20) divides by
 * 1, 2, 4, 5 and 10; every K/8 (2, 4, 8, 10, 16) divides by 1 and 2. */
#define S1_TILE(KB, OW) \
    void conv3x3_s1_t##KB##x##OW(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) { \
        conv3x3_s1_tiled<KB, OW>(d, in, wp, bias, out, false); }
S1_TILE(1, 1) S1_TILE(1, 2) S1_TILE(1, 4) S1_TILE(1, 5) S1_TILE(1, 10)
S1_TILE(2, 1) S1_TILE(2, 2) S1_TILE(2, 4) S1_TILE(2, 5)


/* ------------------------------------------- isolated microkernels ---- */
/*
 * Every register tile as a plain function with ONE signature, so
 * benchmarks/bench_ukr.cpp can time the microkernel alone: one call = one
 * full KB x OW output tile (bias, C*9 reduction, SiLU, store), no layer
 * loops, no row tails, no layout traversal. in_row = floats between input
 * rows (Wp * 8); ukr<> wants Wp, so the wrapper converts.
 */
template <int KB, int OW>
static void ukr_entry(int Cb, const float *ip, size_t in_plane, size_t in_row, const float *w, size_t w_kb,
                      const float *b, float *o, size_t o_kb, int fs) {
    ukr<KB, OW>(Cb, ip, in_plane, (int)(in_row / CB), w, w_kb, b, o, o_kb, fs);
}
template <int KB, int OW>
static void ukr_bs_entry(int Cb, const float *ip, size_t in_plane, size_t in_row, const float *w, size_t w_kb,
                         const float *b, float *o, size_t o_kb, int fs) {
    ukr_bs<KB, OW>(Cb, ip, in_plane, in_row, w, w_kb, b, o, o_kb, fs);
}
static void ukr_2x5_asm_entry(int Cb, const float *ip, size_t in_plane, size_t in_row, const float *w, size_t w_kb,
                              const float *b, float *o, size_t o_kb, int fs) {
    ukr_2x5_asm(Cb, ip, in_plane, in_row, w, w_kb, b, o, o_kb, fs);
}
static void ukr_4x3_asm_entry(int Cb, const float *ip, size_t in_plane, size_t in_row, const float *w, size_t w_kb,
                              const float *b, float *o, size_t o_kb, int fs) {
    ukr_4x3_asm(Cb, ip, in_plane, in_row, w, w_kb, b, o, o_kb, fs);
}
/*
 * 2x6 tile in inline asm (isolated-microkernel experiment). GCC 8.5's 2x6
 * loop (v4) is spill-free but adds 11 register-to-register copies per tap
 * (v2's 2x5: 2), shuffling accumulators at 15 live registers. Here the
 * registers are fixed: 12 accumulators, ymm13 broadcast, ymm14/15 weights.
 */
static void ukr_2x6_asm_entry(int Cb, const float *ip, size_t in_plane, size_t in_row, const float *w, size_t w_kb,
                              const float *b, float *o, size_t o_kb, int fuse_silu) {
    const __m256 b0 = _mm256_loadu_ps(b), b1 = _mm256_loadu_ps(b + CB);
    __m256 a00 = b0, a01 = b0, a02 = b0, a03 = b0, a04 = b0, a05 = b0;
    __m256 a10 = b1, a11 = b1, a12 = b1, a13 = b1, a14 = b1, a15 = b1;
    const long S = (long)(w_kb * sizeof(float));
    for (int cb = 0; cb < Cb; cb++) {
        const float *icb = ip + (size_t)cb * in_plane;
        const float *wcb = w + (size_t)cb * 9 * CB * CB;
        for (int r = 0; r < 3; r++)
            for (int s = 0; s < 3; s++) {
                const float *row = icb + (size_t)r * in_row + s * CB;
                const float *wt = wcb + (r * 3 + s) * CB * CB;
                __asm__ volatile(
        /* ci = 0: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 0(%[w]), %%ymm14\n\t"
        "vmovaps 0(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 0(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 32(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 64(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 96(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 128(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        "vbroadcastss 160(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a05]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a15]\n\t"
        /* ci = 1: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 32(%[w]), %%ymm14\n\t"
        "vmovaps 32(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 4(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 36(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 68(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 100(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 132(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        "vbroadcastss 164(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a05]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a15]\n\t"
        /* ci = 2: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 64(%[w]), %%ymm14\n\t"
        "vmovaps 64(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 8(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 40(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 72(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 104(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 136(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        "vbroadcastss 168(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a05]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a15]\n\t"
        /* ci = 3: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 96(%[w]), %%ymm14\n\t"
        "vmovaps 96(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 12(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 44(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 76(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 108(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 140(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        "vbroadcastss 172(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a05]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a15]\n\t"
        /* ci = 4: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 128(%[w]), %%ymm14\n\t"
        "vmovaps 128(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 16(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 48(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 80(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 112(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 144(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        "vbroadcastss 176(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a05]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a15]\n\t"
        /* ci = 5: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 160(%[w]), %%ymm14\n\t"
        "vmovaps 160(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 20(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 52(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 84(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 116(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 148(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        "vbroadcastss 180(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a05]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a15]\n\t"
        /* ci = 6: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 192(%[w]), %%ymm14\n\t"
        "vmovaps 192(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 24(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 56(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 88(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 120(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 152(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        "vbroadcastss 184(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a05]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a15]\n\t"
        /* ci = 7: 2 weight loads, then per pixel 1 broadcast + 2 FMAs */
        "vmovaps 224(%[w]), %%ymm14\n\t"
        "vmovaps 224(%[w],%[S]), %%ymm15\n\t"
        "vbroadcastss 28(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a00]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a10]\n\t"
        "vbroadcastss 60(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a01]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a11]\n\t"
        "vbroadcastss 92(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a02]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a12]\n\t"
        "vbroadcastss 124(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a03]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a13]\n\t"
        "vbroadcastss 156(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a04]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a14]\n\t"
        "vbroadcastss 188(%[row]), %%ymm13\n\t"
        "vfmadd231ps %%ymm14, %%ymm13, %[a05]\n\t"
        "vfmadd231ps %%ymm15, %%ymm13, %[a15]\n\t"
                    : [a00] "+x"(a00), [a01] "+x"(a01), [a02] "+x"(a02), [a03] "+x"(a03), [a04] "+x"(a04), [a05] "+x"(a05),
                      [a10] "+x"(a10), [a11] "+x"(a11), [a12] "+x"(a12), [a13] "+x"(a13), [a14] "+x"(a14), [a15] "+x"(a15)
                    : [row] "r"(row), [w] "r"(wt), [S] "r"(S)
                    : "xmm13", "xmm14", "xmm15", "memory");
            }
    }
    _mm256_store_ps(o + 0 * CB, epilogue_ps(a00, fuse_silu));
    _mm256_store_ps(o + 1 * CB, epilogue_ps(a01, fuse_silu));
    _mm256_store_ps(o + 2 * CB, epilogue_ps(a02, fuse_silu));
    _mm256_store_ps(o + 3 * CB, epilogue_ps(a03, fuse_silu));
    _mm256_store_ps(o + 4 * CB, epilogue_ps(a04, fuse_silu));
    _mm256_store_ps(o + 5 * CB, epilogue_ps(a05, fuse_silu));
    _mm256_store_ps(o + o_kb + 0 * CB, epilogue_ps(a10, fuse_silu));
    _mm256_store_ps(o + o_kb + 1 * CB, epilogue_ps(a11, fuse_silu));
    _mm256_store_ps(o + o_kb + 2 * CB, epilogue_ps(a12, fuse_silu));
    _mm256_store_ps(o + o_kb + 3 * CB, epilogue_ps(a13, fuse_silu));
    _mm256_store_ps(o + o_kb + 4 * CB, epilogue_ps(a14, fuse_silu));
    _mm256_store_ps(o + o_kb + 5 * CB, epilogue_ps(a15, fuse_silu));
}
/* regs = accumulators + weight registers + broadcast registers */
const s1_ukr_entry_t S1_UKRS[] = {
    {"1x1", 1, 1, 3, "v1's tile", ukr_entry<1, 1>},
    {"1x5", 1, 5, 7, "sweep", ukr_entry<1, 5>},
    {"1x10", 1, 10, 12, "sweep", ukr_entry<1, 10>},
    {"2x2", 2, 2, 7, "sweep", ukr_entry<2, 2>},
    {"2x4", 2, 4, 11, "sweep", ukr_entry<2, 4>},
    {"2x5", 2, 5, 13, "v2's tile (intrinsics)", ukr_entry<2, 5>},
    {"2x6", 2, 6, 15, "v4/v5's tile (intrinsics)", ukr_entry<2, 6>},
    {"2x5asm", 2, 5, 13, "v2a's tile (inline asm)", ukr_2x5_asm_entry},
    {"2x6asm", 2, 6, 15, "2x6 tile (inline asm)", ukr_2x6_asm_entry},
    {"4x3", 4, 3, 16, "v8's tile (intrinsics)", ukr_bs_entry<4, 3>},
    {"4x3asm", 4, 3, 16, "v8a/v8b's tile (inline asm)", ukr_4x3_asm_entry},
};
const int S1_NUKRS = (int)(sizeof(S1_UKRS) / sizeof(S1_UKRS[0]));

#endif /* AVX2 */

/* ------------------------------------------------------ weight packing ---- */
/*
 * v5's weight order: [Kb/2][Cb][3][3][8 ic][2 kb][8 oc]. Same floats as the
 * team layout, reordered so the 2x6 microkernel reads them front to back with
 * no gaps: for each (cb, r, s, ic) it needs the 8 weights of block kb and the 8
 * of block kb+1, and here those 16 floats are adjacent. Runs once, offline,
 * outside the timed region (like BatchNorm folding). Falls back to the team
 * layout when K/8 is odd so the generic kernel still gets what it expects.
 */
void pack_weights_s1_kb2(const float *w, int K, int C, int R, int S, float *dst) {
    const int Kb = ceil_div(K, CB), Cb = ceil_div(C, CB);
    if (Kb % 2 != 0 || R != 3 || S != 3) { pack_weights(w, K, C, R, S, dst); return; }
    for (int k = 0; k < K; k++)
        for (int c = 0; c < C; c++)
            for (int r = 0; r < 3; r++)
                for (int s = 0; s < 3; s++) {
                    const int kb = k / CB, ko = k % CB, kp = kb / 2, kk = kb % 2, cb = c / CB, ci = c % CB;
                    const size_t o = ((((((size_t)kp * Cb + cb) * 3 + r) * 3 + s) * CB + ci) * 2 + kk) * CB + ko;
                    dst[o] = w[(((size_t)k * C + c) * R + r) * S + s];
                }
}

/* Scalar reference for ONE microkernel call (same arguments), used by
 * bench_ukr and the tests to check every tile. */
void s1_ukr_reference(int KB, int OW, int Cb, const float *ip, size_t in_plane, size_t in_row,
                      const float *w, size_t w_kb, const float *b, float *o, size_t o_kb, int fs) {
    for (int k = 0; k < KB; k++)
        for (int j = 0; j < OW; j++)
            for (int ko = 0; ko < CB; ko++) {
                float acc = b[k * CB + ko];
                for (int cb = 0; cb < Cb; cb++)
                    for (int r = 0; r < 3; r++)
                        for (int s = 0; s < 3; s++)
                            for (int ci = 0; ci < CB; ci++)
                                acc += ip[cb * in_plane + r * in_row + (size_t)(j + s) * CB + ci] *
                                       w[k * w_kb + (size_t)cb * 9 * CB * CB + (r * 3 + s) * CB * CB + ci * CB + ko];
                o[k * o_kb + (size_t)j * CB + ko] = fs ? silu_scalar(acc) : acc;
            }
}

#ifndef FASTYOLO_HAVE_AVX2
const s1_ukr_entry_t S1_UKRS[] = {{"none", 0, 0, 0, "no AVX2 build", nullptr}};
const int S1_NUKRS = 0;
#endif

#ifndef FASTYOLO_HAVE_AVX2 /* no AVX2 (e.g. a Mac): every version falls back to v0 so the build and tests still work */
#define S1_FALLBACK(name) \
    void name(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) { \
        conv3x3_s1_v0(d, in, wp, bias, out); }
S1_FALLBACK(conv3x3_s1_v1) S1_FALLBACK(conv3x3_s1_v2) S1_FALLBACK(conv3x3_s1_v3)
S1_FALLBACK(conv3x3_s1_v4) S1_FALLBACK(conv3x3_s1_v4b)
S1_FALLBACK(conv3x3_s1_v6) S1_FALLBACK(conv3x3_s1_v6b) S1_FALLBACK(conv3x3_s1_v7) S1_FALLBACK(conv3x3_s1_v7b) S1_FALLBACK(conv3x3_s1_v7mt)
S1_FALLBACK(conv3x3_s1_v2a) S1_FALLBACK(conv3x3_s1_v8) S1_FALLBACK(conv3x3_s1_v8a) S1_FALLBACK(conv3x3_s1_v8mt)
S1_FALLBACK(conv3x3_s1_v8b) S1_FALLBACK(conv3x3_s1_v9) S1_FALLBACK(conv3x3_s1_v9mt)
/* v5 reads the v5 weight order, so its fallback must too: a plain scalar loop over that order */
void conv3x3_s1_v5(const conv_desc_t *d, const float *in, const float *wp, const float *bias, float *out) {
    const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);
    if (Kb % 2 != 0) { conv3x3_s1_v0(d, in, wp, bias, out); return; }
    const int Hp = d->H + 2 * d->in_halo, Wp = d->W + 2 * d->in_halo;
    const int OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo, off = d->in_halo - d->pad;
    for (int n = 0; n < d->N; n++)
        for (int kb = 0; kb < Kb; kb++)
            for (int oh = 0; oh < d->OH; oh++)
                for (int ow = 0; ow < d->OW; ow++) {
                    float acc[CB];
                    for (int ko = 0; ko < CB; ko++) acc[ko] = bias[kb * CB + ko];
                    for (int cb = 0; cb < Cb; cb++)
                        for (int r = 0; r < 3; r++)
                            for (int s = 0; s < 3; s++) {
                                const float *ip = in + blk_idx(Cb, Hp, Wp, n, cb, oh + r + off, ow + s + off);
                                const float *wq = wp + (((((size_t)(kb / 2) * Cb + cb) * 3 + r) * 3 + s) * CB * 2 + (kb % 2)) * CB;
                                for (int ci = 0; ci < CB; ci++)
                                    for (int ko = 0; ko < CB; ko++) acc[ko] += ip[ci] * wq[ci * 2 * CB + ko];
                            }
                    float *op = out + blk_idx(Kb, OHp, OWp, n, kb, oh + d->out_halo, ow + d->out_halo);
                    for (int ko = 0; ko < CB; ko++) op[ko] = d->fuse_silu ? silu_scalar(acc[ko]) : acc[ko];
                }
}
S1_FALLBACK(conv3x3_s1_t1x1) S1_FALLBACK(conv3x3_s1_t1x2) S1_FALLBACK(conv3x3_s1_t1x4)
S1_FALLBACK(conv3x3_s1_t1x5) S1_FALLBACK(conv3x3_s1_t1x10) S1_FALLBACK(conv3x3_s1_t2x1)
S1_FALLBACK(conv3x3_s1_t2x2) S1_FALLBACK(conv3x3_s1_t2x4) S1_FALLBACK(conv3x3_s1_t2x5)
#endif

} // namespace fastyolo
