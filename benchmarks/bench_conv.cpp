#include "kernels.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <vector>

int main() {
    // Smoke-test benchmark only. This placeholder kernel calls the scalar
    // reference, so DO NOT present this as optimized or hardware-peak results.
    constexpr int CIN = 8, COUT = 16, H = 32, W = 32, REPEATS = 5;
    std::vector<float> x(CIN * H * W, 0.2f);
    std::vector<float> wt(COUT * CIN * 9, 0.03f);
    volatile float checksum = 0.0f;
    using clock = std::chrono::steady_clock;
    const auto start = clock::now();
    for (int i = 0; i < REPEATS; ++i) {
        const auto y = fastyolo::conv3x3_s1(x, wt, CIN, COUT, H, W);
        checksum = checksum + y[y.size() / 2];
    }
    const auto end = clock::now();
    const double millis = std::chrono::duration<double, std::milli>(end - start).count() / REPEATS;
    const double nominal_flops = 2.0 * CIN * COUT * H * W * 9.0;
    std::cout << "SMOKE TEST ONLY (scalar reference, not optimized)\n";
    std::cout << "CIN=" << CIN << " COUT=" << COUT << " H=" << H << " W=" << W << "\n";
    std::cout << "Approx. milliseconds/invocation: " << millis << "\n";
    std::cout << "Nominal FLOPs/invocation (counts padded MAC positions): " << nominal_flops << "\n";
    std::cout << "Checksum: " << checksum << "\n";
}
