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
// Same result on NCHW8c, batch N, C a multiple of 8. in: [N][C/8][H+2*in_halo][W+2*in_halo][8];
// out: [N][4C/8][H+2*out_halo][W+2*out_halo][8], channel slices X, P1, P2, P3 (the Concat
// model.9.cv2 reads). Writes only the interior; the out halo stays as it was (zero).
typedef void (*pool_fn_t)(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo);
void sppf_pool_v1(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo); // AVX2, separable, 8 vmaxps/output
void sppf_pool_v2(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo); // v1 + shared pair maxes, 6 vmaxps/output
void sppf_pool_v3(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo); // v2 rows + straight-line edges, columns pass row by row
void sppf_pool_v4(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo); // v3 with a 5-row ring instead of the tmp plane (fits L1)
void sppf_pool_v5(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo); // v4 with rotated row pointers (no r % 5 on port 1)
// Isolated pool microkernels: one 5x5 pool of one 8-channel plane. s, d point at pixel (0, 0) of the
// interior, sr/dr are row strides in floats; t is scratch: H*W*8 floats, or 5*W*8 if scratch_rows = 5.
typedef void (*pool_ukr_fn_t)(const float *s, size_t sr, float *d, size_t dr, float *t, int H, int W);
typedef struct { const char *name; int rows_mode, cols_mode, scratch_rows; const char *note; pool_ukr_fn_t fn; } pool_ukr_entry_t;
extern const pool_ukr_entry_t POOL_UKRS[];
extern const int POOL_NUKRS;
long sppf_vmax_per_line(int n, int pair); // vmaxps per 1-D pass of length n: 0 direct, 1 v2 pairs, 2 v3 rows (for bench_pool)

} // namespace fastyolo
