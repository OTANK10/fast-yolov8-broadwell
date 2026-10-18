/*
 * conv1x1.c  --  1x1 convs (C2f cv1/cv2, SPPF cv1/cv2, Detect final layers).
 * Owner: Simon, if the team takes it on (suggested by Prof. Low's feedback: "implement the 1x1 to get a full model").
 *
 * In NCHW8c a 1x1 conv is a GEMM: [pixels x C] * [C x K]. Same signature and
 * layouts as everything else, so the harness benchmarks it already.
 */
#include "common.h"
#include "kernels.h"

namespace fastyolo {

void conv1x1_v1(const conv_desc_t *d, const float *in, const float *wp,
                const float *bias, float *out) {
    conv_blk_generic(d, in, wp, bias, out); /* TODO(Simon): 1x1 / GEMM microkernel */
}

} // namespace fastyolo
