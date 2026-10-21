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
// Weight packer for v5: [Kb/2][Cb][3][3][8 ic][2][8 oc]. Same size as pack_weights.
void pack_weights_s1_kb2(const float *oihw, int K, int C, int R, int S, float *dst);
typedef void (*pack_fn_t)(const float *oihw, int K, int C, int R, int S, float *dst);
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
