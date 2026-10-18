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
void conv3x3_s1_v0(const conv_desc_t *d, const float *in, const float *w,
                   const float *bias, float *out);  // scalar, same loop order as v1
void conv3x3_s1_v1(const conv_desc_t *d, const float *in, const float *w,
                   const float *bias, float *out);  // AVX2 2x5 microkernel + OpenMP

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
