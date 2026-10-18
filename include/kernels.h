#pragma once

#include <vector>

namespace fastyolo {
// Minimal, correctness-first interface for a batch of one.
// Input: [Cin, H, W] NCHW, weights: [Cout, Cin, 3, 3] OIHW.
// Output: [Cout, OH, OW]; no batch norm, bias or activation.
// SAME-style padding for 3x3 with pad=1.
std::vector<float> conv3x3_reference(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    int cin, int cout, int h, int w, int stride, int pad = 1);

// IMPORTANT: The owner kernels are currently placeholders that call the
// scalar reference. Replace these with verified optimized versions.
std::vector<float> conv3x3_s1(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    int cin, int cout, int h, int w);

std::vector<float> conv3x3_s2(
    const std::vector<float>& input,
    const std::vector<float>& weights,
    int cin, int cout, int h, int w);

// SPPF core only: concatenate X, P(X), P(P(X)), P(P(P(X))) along C,
// where P is 5x5 stride-1 maxpool with padding 2.
// Surrounding 1x1 convolutions and activations not included.
std::vector<float> sppf_pool_reference(
    const std::vector<float>& input, int channels, int h, int w);
} // namespace fastyolo
