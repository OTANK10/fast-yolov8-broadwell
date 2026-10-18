/*
 * common.h  --  the shared contract for every kernel in this repo.
 *
 * If all three of us code against THIS header, nobody has to wait on anybody:
 * the harness (benchmarks/bench_layer.cpp) feeds every kernel the same blocked
 * tensors, and each person only edits their own file in src/.
 * Change this file only through a small PR that everyone sees.
 *
 * Layouts (all FP32, all buffers 32-byte aligned):
 *
 *   Activations: NCHW8c with a zero halo.
 *     logical  [N][C][H][W]
 *     stored   [N][Cb][H+2*halo][W+2*halo][8]      Cb = ceil(C/8)
 *     The 8 innermost floats are 8 consecutive channels = one YMM register.
 *     The halo is the conv zero-padding stored explicitly, so a 3x3 kernel
 *     never needs an "am I at the image edge?" branch. Producers write only
 *     the interior; the halo stays zero.
 *
 *   Weights: packed once, offline (like BN folding, this is not inference work).
 *     logical  [K][C][R][S]   (ONNX OIHW)
 *     stored   [Kb][Cb][R][S][8 ic][8 oc]
 *     For a fixed (kb, cb, r, s, ic) the 8 output-channel weights are one
 *     contiguous YMM load.
 *
 *   Bias: [K] plain floats (BatchNorm is ALREADY folded into W and bias by the
 *   Ultralytics ONNX export; there is no BatchNormalization node at runtime).
 */
#ifndef FASTYOLO_COMMON_H
#define FASTYOLO_COMMON_H

#include <cstddef>
#include <cstdint>
#if !defined(__x86_64__) && !defined(__i386__)
#include <chrono>
#endif

namespace fastyolo {

#define CB 8 /* channel block = number of FP32 lanes in a 256-bit AVX2 register */

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

typedef struct {
    int N, C, H, W;   /* input tensor (logical) */
    int K, R, S;      /* output channels, kernel height, kernel width */
    int stride, pad;  /* conv stride and zero padding */
    int OH, OW;       /* output spatial size */
    int in_halo;      /* halo stored around the input planes (must equal pad for 3x3) */
    int out_halo;     /* halo the kernel must leave around the output planes */
    int fuse_silu;    /* 1: out = SiLU(conv + bias), 0: out = conv + bias */
} conv_desc_t;

/* Every conv kernel has exactly this signature. */
typedef void (*conv_fn_t)(const conv_desc_t *d,
                          const float *in_blk,  /* NCHW8c, halo d->in_halo  */
                          const float *w_pack,  /* [Kb][Cb][R][S][8][8]     */
                          const float *bias,    /* [K]                      */
                          float *out_blk);      /* NCHW8c, halo d->out_halo, pre-zeroed */

/* ---- index helpers (blocked layouts) ---- */
static inline size_t blk_size(int N, int C, int H, int W, int halo) {
    return (size_t)N * ceil_div(C, CB) * (H + 2 * halo) * (W + 2 * halo) * CB;
}
/* offset of element (n, cb, y, x, lane) where y, x are PADDED coordinates */
static inline size_t blk_idx(int Cb, int Hp, int Wp, int n, int cb, int y, int x) {
    return ((((size_t)n * Cb + cb) * Hp + y) * Wp + x) * CB;
}
static inline size_t wpack_size(int K, int C, int R, int S) {
    return (size_t)ceil_div(K, CB) * ceil_div(C, CB) * R * S * CB * CB;
}

/* ---- src/layout.cpp ---- */
float *alloc_aligned(size_t nfloats);  /* 64B aligned, zero-filled; free with free() */
void nchw_to_blk(const float *src, int N, int C, int H, int W, int halo, float *dst);
void blk_to_nchw(const float *src, int N, int C, int H, int W, int halo, float *dst);
void pack_weights(const float *oihw, int K, int C, int R, int S, float *dst);

/* ---- timing ---- */
/* Timestamp counter (x86). On other CPUs (e.g. a Mac for correctness work) it falls
 * back to nanoseconds, so numbers there are NOT cycles; report only ece007 runs. */
static inline uint64_t rdtsc(void) {
#if defined(__x86_64__) || defined(__i386__)
    unsigned lo, hi;
    __asm__ __volatile__("lfence\n\trdtsc" : "=a"(lo), "=d"(hi)::"memory");
    return ((uint64_t)hi << 32) | lo;
#else
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
#endif
}

/* Broadwell-EP E5-2640 v4: 2 FMA ports x 8 FP32 lanes x 2 flops = 32 flops/cycle/core */
#define PEAK_FLOPS_PER_CYCLE_PER_CORE 32.0
#define NOMINAL_GHZ 2.4

} // namespace fastyolo
#endif
