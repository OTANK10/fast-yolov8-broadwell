#include "kernels.h"
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
namespace fastyolo {
namespace {
std::vector<float> maxpool5x5(const std::vector<float>& input, int c, int h, int w) {
    std::vector<float> out(input.size());
    for (int ch = 0; ch < c; ++ch)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float value = -std::numeric_limits<float>::infinity();
                for (int ky = -2; ky <= 2; ++ky)
                    for (int kx = -2; kx <= 2; ++kx) {
                        const int iy = y + ky, ix = x + kx;
                        if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                        const auto idx = (static_cast<std::size_t>(ch) * h + iy) * w + ix;
                        value = std::max(value, input[idx]);
                    }
                out[(static_cast<std::size_t>(ch) * h + y) * w + x] = value;
            }
    return out;
}
} // namespace
std::vector<float> sppf_pool_reference(const std::vector<float>& input, int channels, int h, int w) {
    if (channels <= 0 || h <= 0 || w <= 0 ||
        input.size() != static_cast<std::size_t>(channels) * h * w)
        throw std::invalid_argument("Invalid SPPF input tensor");
    auto p1 = maxpool5x5(input, channels, h, w);
    auto p2 = maxpool5x5(p1, channels, h, w);
    auto p3 = maxpool5x5(p2, channels, h, w);
    std::vector<float> result;
    result.reserve(input.size() * 4);
    result.insert(result.end(), input.begin(), input.end());
    result.insert(result.end(), p1.begin(), p1.end());
    result.insert(result.end(), p2.begin(), p2.end());
    result.insert(result.end(), p3.begin(), p3.end());
    return result;
}
} // namespace fastyolo
