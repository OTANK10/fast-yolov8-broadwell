#include "kernels.h"
namespace fastyolo {
std::vector<float> conv3x3_s2(
    const std::vector<float>& input, const std::vector<float>& weights,
    int cin, int cout, int h, int w) {
    // TODO(Sherlock): optimize stride-2 input addressing and vector blocking.
    return conv3x3_reference(input, weights, cin, cout, h, w, 2, 1);
}
} // namespace fastyolo
