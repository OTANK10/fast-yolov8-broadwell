#include "kernels.h"
namespace fastyolo {
std::vector<float> conv3x3_s1(
    const std::vector<float>& input, const std::vector<float>& weights,
    int cin, int cout, int h, int w) {
    // TODO(Om): replace reference call with a correctness-verified AVX2/FMA
    // direct-convolution kernel and document tile/register/data layout.
    return conv3x3_reference(input, weights, cin, cout, h, w, 1, 1);
}
} // namespace fastyolo
