/* layout.c -- conversions between ONNX's plain layouts and our blocked layouts.
 * These run OUTSIDE the timed region (weights are packed once; activations
 * would already be blocked if the whole network used NCHW8c). */
#include <cstdlib>
#include <cstring>
#include "common.h"

namespace fastyolo {

float *alloc_aligned(size_t nfloats) {
    size_t bytes = ((nfloats * sizeof(float) + 63) / 64) * 64;
    if (bytes == 0) bytes = 64;
    float *p = static_cast<float *>(std::aligned_alloc(64, bytes));
    if (p) std::memset(p, 0, bytes);
    return p;
}

void nchw_to_blk(const float *src, int N, int C, int H, int W, int halo, float *dst) {
    int Cb = ceil_div(C, CB), Hp = H + 2 * halo, Wp = W + 2 * halo;
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    dst[blk_idx(Cb, Hp, Wp, n, c / CB, y + halo, x + halo) + c % CB] =
                        src[(((size_t)n * C + c) * H + y) * W + x];
    /* channels C..Cb*8-1 and the halo stay zero (dst was zero-filled) */
}

void blk_to_nchw(const float *src, int N, int C, int H, int W, int halo, float *dst) {
    int Cb = ceil_div(C, CB), Hp = H + 2 * halo, Wp = W + 2 * halo;
    for (int n = 0; n < N; n++)
        for (int c = 0; c < C; c++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    dst[(((size_t)n * C + c) * H + y) * W + x] =
                        src[blk_idx(Cb, Hp, Wp, n, c / CB, y + halo, x + halo) + c % CB];
}

/* OIHW -> [Kb][Cb][R][S][8 ic][8 oc] */
void pack_weights(const float *w, int K, int C, int R, int S, float *dst) {
    int Cb = ceil_div(C, CB);
    for (int k = 0; k < K; k++)
        for (int c = 0; c < C; c++)
            for (int r = 0; r < R; r++)
                for (int s = 0; s < S; s++) {
                    size_t o = ((((size_t)(k / CB) * Cb + c / CB) * R + r) * S + s) * CB * CB
                             + (c % CB) * CB + (k % CB);
                    dst[o] = w[(((size_t)k * C + c) * R + r) * S + s];
                }
}

} // namespace fastyolo
