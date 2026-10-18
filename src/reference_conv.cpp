/* ref_conv.c -- slow, obviously-correct convolutions. Never optimize these;
 * they exist so every fast kernel has something to be checked against. */
#include "common.h"
#include "epilogue.h"
#include "kernels.h"

namespace fastyolo {

void conv_ref_nchw(const conv_desc_t *d, const float *in, const float *w,
                   const float *bias, float *out) {
    for (int n = 0; n < d->N; n++)
        for (int k = 0; k < d->K; k++)
            for (int oh = 0; oh < d->OH; oh++)
                for (int ow = 0; ow < d->OW; ow++) {
                    float acc = bias ? bias[k] : 0.0f;
                    for (int c = 0; c < d->C; c++)
                        for (int r = 0; r < d->R; r++)
                            for (int s = 0; s < d->S; s++) {
                                int ih = oh * d->stride - d->pad + r;
                                int iw = ow * d->stride - d->pad + s;
                                if (ih < 0 || ih >= d->H || iw < 0 || iw >= d->W) continue;
                                acc += in[(((size_t)n * d->C + c) * d->H + ih) * d->W + iw] *
                                       w[(((size_t)k * d->C + c) * d->R + r) * d->S + s];
                            }
                    out[(((size_t)n * d->K + k) * d->OH + oh) * d->OW + ow] =
                        d->fuse_silu ? silu_scalar(acc) : acc;
                }
}

void conv_blk_generic(const conv_desc_t *d, const float *in, const float *wp,
                      const float *bias, float *out) {
    const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);
    const int Hp = d->H + 2 * d->in_halo, Wp = d->W + 2 * d->in_halo;
    const int OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo;
    const int off = d->in_halo - d->pad; /* padded-coordinate shift */

    for (int n = 0; n < d->N; n++)
        for (int kb = 0; kb < Kb; kb++)
            for (int oh = 0; oh < d->OH; oh++)
                for (int ow = 0; ow < d->OW; ow++) {
                    float acc[CB];
                    for (int ko = 0; ko < CB; ko++) {
                        int k = kb * CB + ko;
                        acc[ko] = (bias && k < d->K) ? bias[k] : 0.0f;
                    }
                    for (int cb = 0; cb < Cb; cb++)
                        for (int r = 0; r < d->R; r++)
                            for (int s = 0; s < d->S; s++) {
                                int y = oh * d->stride + r + off, x = ow * d->stride + s + off;
                                if (y < 0 || y >= Hp || x < 0 || x >= Wp) continue;
                                const float *ip = in + blk_idx(Cb, Hp, Wp, n, cb, y, x);
                                const float *wq = wp + ((((size_t)kb * Cb + cb) * d->R + r) * d->S + s) * CB * CB;
                                for (int ci = 0; ci < CB; ci++)
                                    for (int ko = 0; ko < CB; ko++)
                                        acc[ko] += ip[ci] * wq[ci * CB + ko];
                            }
                    float *op = out + blk_idx(Kb, OHp, OWp, n, kb, oh + d->out_halo, ow + d->out_halo);
                    for (int ko = 0; ko < CB; ko++)
                        op[ko] = (kb * CB + ko < d->K) ? (d->fuse_silu ? silu_scalar(acc[ko]) : acc[ko]) : 0.0f;
                }
}

} // namespace fastyolo
