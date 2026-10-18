#include "kernels.h"
#include <cstddef>
#include <stdexcept>

namespace fastyolo {
std::vector<float> conv3x3_reference(
    const std::vector<float>& input, const std::vector<float>& weights,
    int cin, int cout, int h, int w, int stride, int pad) {
    if (cin <= 0 || cout <= 0 || h <= 0 || w <= 0 || stride <= 0 || pad < 0)
        throw std::invalid_argument("Invalid convolution dimensions or stride/padding");
    if (h + 2 * pad < 3 || w + 2 * pad < 3)
        throw std::invalid_argument("Kernel does not fit padded input");
    if (input.size() != static_cast<std::size_t>(cin) * h * w ||
        weights.size() != static_cast<std::size_t>(cout) * cin * 3 * 3)
        throw std::invalid_argument("Input/weight tensor size mismatch");

    const int oh = (h + 2 * pad - 3) / stride + 1;
    const int ow = (w + 2 * pad - 3) / stride + 1;
    std::vector<float> out(static_cast<std::size_t>(cout) * oh * ow, 0.0f);
    for (int oc = 0; oc < cout; ++oc) {
        for (int oy = 0; oy < oh; ++oy) {
            for (int ox = 0; ox < ow; ++ox) {
                float sum = 0.0f;
                for (int ic = 0; ic < cin; ++ic) {
                    for (int ky = 0; ky < 3; ++ky) {
                        const int iy = oy * stride + ky - pad;
                        if (iy < 0 || iy >= h) continue;
                        for (int kx = 0; kx < 3; ++kx) {
                            const int ix = ox * stride + kx - pad;
                            if (ix < 0 || ix >= w) continue;
                            const std::size_t input_idx = (static_cast<std::size_t>(ic) * h + iy) * w + ix;
                            const std::size_t wt_idx = ((static_cast<std::size_t>(oc) * cin + ic) * 3 + ky) * 3 + kx;
                            sum += input[input_idx] * weights[wt_idx];
                        }
                    }
                }
                out[(static_cast<std::size_t>(oc) * oh + oy) * ow + ox] = sum;
            }
        }
    }
    return out;
}
} // namespace fastyolo
