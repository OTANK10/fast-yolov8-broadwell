#include "kernels.h"
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

static bool close(float a, float b, float tol = 1e-5f) {
    return std::abs(a - b) <= tol;
}
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << " " << #expr << "\n"; \
        std::exit(1); \
    } \
} while (false)
int main() {
    using namespace fastyolo;
    // One 3x3 input and a 3x3 all-ones kernel; padding=1.
    const std::vector<float> x{1,2,3,4,5,6,7,8,9};
    const std::vector<float> weights(9, 1.0f);
    auto y = conv3x3_s1(x, weights, 1, 1, 3, 3);
    CHECK(y.size() == 9);
    CHECK(close(y[0], 12.0f));  // 1+2+4+5
    CHECK(close(y[4], 45.0f));  // all nine pixels
    CHECK(close(y[8], 28.0f));  // 5+6+8+9
    auto ds = conv3x3_s2(x, weights, 1, 1, 3, 3);
    CHECK(ds.size() == 4);
    CHECK(close(ds[0], 12.0f));
    CHECK(close(ds[3], 28.0f));

    // Multi-channel/multi-output smoke test and weight indexing.
    const std::vector<float> two_channels{
        1,2,3,4,5,6,7,8,9,
        10,20,30,40,50,60,70,80,90
    };
    std::vector<float> two_weights(2 * 2 * 9, 0.0f);
    two_weights[4] = 1.0f;                // output 0, input 0, center only
    two_weights[2 * 9 + 9 + 4] = 2.0f;   // output 1, input 1, center only
    auto multi = conv3x3_s1(two_channels, two_weights, 2, 2, 3, 3);
    CHECK(multi.size() == 18);
    CHECK(close(multi[4], 5.0f));
    CHECK(close(multi[9 + 4], 100.0f));

    // SPPF's pooling-only reference: 1 x 3 x 3 => concat of 4 feature maps.
    auto sppf = sppf_pool_reference(x, 1, 3, 3);
    CHECK(sppf.size() == 36);
    CHECK(close(sppf[4], 5.0f));  // Original tensor preserved in first C block.
    for (std::size_t i = 9; i < sppf.size(); ++i)
        CHECK(close(sppf[i], 9.0f)); // Every 5x5 window covers max=9 on 3x3.

    bool rejected = false;
    try { (void) conv3x3_s1({}, weights, 1, 1, 3, 3); }
    catch (const std::invalid_argument&) { rejected = true; }
    CHECK(rejected);
    std::cout << "Starter correctness checks passed.\n";
}
