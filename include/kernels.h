#pragma once
// kernels.h -- every kernel the harness and tests know about.
// Layouts, conv_desc_t and the shared signature live in common.h.
// To add a version: declare it here, add one line to the table in
// benchmarks/bench_layer.cpp, and one line to tests/test_kernels.cpp.

#include <vector>
#include "common.h"

namespace fastyolo {

// ---- reference (src/reference_conv.cpp): never optimize these ----
// Plain NCHW / OIHW, any stride, padding and kernel size, bias + optional SiLU.
void conv_ref_nchw(const conv_desc_t *d, const float *in, const float *w,
                   const float *bias, float *out);
// Scalar conv on the blocked layouts. The correct fallback for any kernel
// that is not written yet.
void conv_blk_generic(const conv_desc_t *d, const float *in_blk, const float *w_pack,
                      const float *bias, float *out_blk);

// ---- 3x3 stride 1 (Om, src/conv3x3_s1.cpp) ----
#define FASTYOLO_CONV_DECL(name) \
    void name(const conv_desc_t *d, const float *in, const float *w, const float *bias, float *out);
FASTYOLO_CONV_DECL(conv3x3_s1_v0)   // scalar, the loop order every later version keeps
FASTYOLO_CONV_DECL(conv3x3_s1_v1)   // AVX2, 1 accumulator (latency-bound)
FASTYOLO_CONV_DECL(conv3x3_s1_v2)   // AVX2 2x5 register tile, 10 accumulators, 1 core
FASTYOLO_CONV_DECL(conv3x3_s1_v3)   // v2 + OpenMP
FASTYOLO_CONV_DECL(conv3x3_s1_v4)   // 2x6 tile, 12 accumulators, 6+2 row tail, 1 core
FASTYOLO_CONV_DECL(conv3x3_s1_v4b)  // same, 4+4 row tail
FASTYOLO_CONV_DECL(conv3x3_s1_v5)   // v4 + weights in microkernel order (needs pack_weights_s1_kb2)
FASTYOLO_CONV_DECL(conv3x3_s1_v6)   // 2x5, all 9 taps unrolled (constant offsets), 1 core
FASTYOLO_CONV_DECL(conv3x3_s1_v6b)  // 2x5, s unrolled, loop over r, 1 core
FASTYOLO_CONV_DECL(conv3x3_s1_v7)   // v6 + software-pipelined loads, 1 core
FASTYOLO_CONV_DECL(conv3x3_s1_v7b)  // v6b + software-pipelined loads, 1 core
FASTYOLO_CONV_DECL(conv3x3_s1_v7mt) // v7 + OpenMP
FASTYOLO_CONV_DECL(conv3x3_s1_v8)   // 4x3 broadcast-stationary tile (12 acc, 7 loads/12 FMA), 1 core
FASTYOLO_CONV_DECL(conv3x3_s1_v2a)  // v2's 2x5 tile, inner step in inline asm (control for v8a)
FASTYOLO_CONV_DECL(conv3x3_s1_v8a)  // v8 with the inner step in inline asm (no spills)
FASTYOLO_CONV_DECL(conv3x3_s1_v8mt) // v8a + OpenMP
FASTYOLO_CONV_DECL(conv3x3_s1_v8b)  // v8a, row remainders by 2x5 tiles
FASTYOLO_CONV_DECL(conv3x3_s1_v9)   // per-layer dispatch: v8b (C <= 80) or v2 (C >= 128), 1 core
FASTYOLO_CONV_DECL(conv3x3_s1_v9mt) // v9 + OpenMP
FASTYOLO_CONV_DECL(conv3x3_s1_v10)  // 2x6 + 2x5 asm microkernels called noinline (as in bench_ukr), 1 core
FASTYOLO_CONV_DECL(conv3x3_s1_v10mt) // v10 + OpenMP
FASTYOLO_CONV_DECL(conv3x3_s1_v2b)  // 2x5 asm microkernel called noinline (control for v2a)
// Weight packer for v5: [Kb/2][Cb][3][3][8 ic][2][8 oc]. Same size as pack_weights.
void pack_weights_s1_kb2(const float *oihw, int K, int C, int R, int S, float *dst);
typedef void (*pack_fn_t)(const float *oihw, int K, int C, int R, int S, float *dst);
// Isolated stride-1 microkernels (benchmarks/bench_ukr.cpp): one call = one KB x OW output tile.
typedef void (*s1_ukr_fn_t)(int Cb, const float *ip, size_t in_plane, size_t in_row, const float *w,
                            size_t w_kb, const float *b, float *o, size_t o_kb, int fuse_silu);
typedef struct { const char *name; int KB, OW, regs; const char *note; s1_ukr_fn_t fn; } s1_ukr_entry_t;
extern const s1_ukr_entry_t S1_UKRS[];
extern const int S1_NUKRS;
void s1_ukr_reference(int KB, int OW, int Cb, const float *ip, size_t in_plane, size_t in_row,
                      const float *w, size_t w_kb, const float *b, float *o, size_t o_kb, int fuse_silu);
// tile-shape sweep (single core): KB output-channel blocks x OW pixels
FASTYOLO_CONV_DECL(conv3x3_s1_t1x1) FASTYOLO_CONV_DECL(conv3x3_s1_t1x2) FASTYOLO_CONV_DECL(conv3x3_s1_t1x4)
FASTYOLO_CONV_DECL(conv3x3_s1_t1x5) FASTYOLO_CONV_DECL(conv3x3_s1_t1x10)
FASTYOLO_CONV_DECL(conv3x3_s1_t2x1) FASTYOLO_CONV_DECL(conv3x3_s1_t2x2) FASTYOLO_CONV_DECL(conv3x3_s1_t2x4)
FASTYOLO_CONV_DECL(conv3x3_s1_t2x5)

// ---- 3x3 stride 2 (O'Malley, src/conv3x3_s2.cpp) ----
void conv3x3_s2_v1(const conv_desc_t *d, const float *in, const float *w,
                   const float *bias, float *out);

// ---- 1x1 (src/conv1x1.cpp, if the team takes it on) ----
void conv1x1_v1(const conv_desc_t *d, const float *in, const float *w,
                const float *bias, float *out);

// ---- SPPF (Simon, src/sppf.cpp) ----
// Pooling core only: concatenate X, P(X), P(P(X)), P(P(P(X))) along C,
// where P is 5x5 stride-1 maxpool with padding 2. Plain NCHW, batch 1.
std::vector<float> sppf_pool_reference(const std::vector<float> &input, int channels, int h, int w);

} // namespace fastyolo
