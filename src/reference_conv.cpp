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



// ---- explicit naming of variables for my understanding

int PADDING = 1;
int KERNEL_SIZE = 3;
int STRIDE = 2;


void kernel_verbose(
  const int n_input_channels, 
  const int n_output_channels,
  const int input_height,
  const int input_width,
  const float* input,
  const float* weights,
  float* output
) {

  int output_height = (input_height + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;
  int output_width  = (input_width  + 2*PADDING - KERNEL_SIZE) / STRIDE + 1;

  // loop for all channels, cols, rows in the output
  for (int out_channel_idx = 0; out_channel_idx < n_output_channels; ++out_channel_idx) {
    for (int out_row_idx = 0; out_row_idx < output_height; ++out_row_idx) {
      for (int out_col_idx = 0; out_col_idx < output_width; ++out_col_idx) {

        float running_pixel_sum = 0.0f;
        
        // loop over all input channels
        for (int in_channel_idx = 0; in_channel_idx < n_input_channels; ++in_channel_idx) {

          for (int filter_row = 0; filter_row < KERNEL_SIZE; ++filter_row) {
            int input_row_idx = out_row_idx*STRIDE - PADDING + filter_row;
            if (input_row_idx < 0 || input_row_idx >= input_height) {continue;}

            for (int filter_col = 0; filter_col < KERNEL_SIZE; ++filter_col) {
              int input_col_idx = out_col_idx*STRIDE - PADDING + filter_col;
              if (input_col_idx < 0 || input_col_idx >= input_width) {continue;}

                int input_index = (in_channel_idx*input_height + input_row_idx)*input_width + input_col_idx;
                int weight_index = ((out_channel_idx*n_input_channels + in_channel_idx)*KERNEL_SIZE+ filter_row)*KERNEL_SIZE + filter_col;
                running_pixel_sum += input[input_index] * weights[weight_index];

            }


          }


        }

      int output_index = (out_channel_idx*output_height + out_row_idx)*output_width + out_col_idx;
      output[output_index] = running_pixel_sum;

      }
    }
  }

}




} // namespace fastyolo
