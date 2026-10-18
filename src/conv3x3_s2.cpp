/*
 * conv3x3_s2.c  --  3x3, stride-2, pad-1 downsampling convs (backbone + neck).
 * Owner: O'Malley. Branch: sherlock/conv3x3-s2.
 *
 * Same signature and layouts as every other kernel (include/common.h), so the
 * harness already benchmarks this file. Right now it forwards to the generic
 * scalar conv so it is CORRECT on day one; replace the body with your kernel.
 * Note model.0 has C=3: the layout pads it to one 8-channel block of zeros.
 */
#include "common.h"
#include "kernels.h"

namespace fastyolo {

void conv3x3_s2_v1(const conv_desc_t *d, const float *in, const float *wp,
                   const float *bias, float *out) {
    conv_blk_generic(d, in, wp, bias, out); /* TODO(O'Malley): stride-2 microkernel */
}

} // namespace fastyolo
