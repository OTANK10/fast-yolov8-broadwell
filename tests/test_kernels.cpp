// test_kernels.cpp -- correctness tests that need no dumped data, so they run
// anywhere (CI, a Mac, ece007). Real-layer checks against the baseline's own
// outputs are done by bench_layer on data/tensors/.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include "kernels.h"

using namespace fastyolo;

static bool close(float a, float b, float tol = 1e-5f) { return std::abs(a - b) <= tol; }
#define CHECK(expr) do { \
    if (!(expr)) { \
        std::cerr << "FAILED " << __FILE__ << ":" << __LINE__ << " " << #expr << "\n"; \
        std::exit(1); \
    } \
} while (false)

static conv_desc_t make_desc(int N, int C, int H, int W, int K, int R, int stride, int silu) {
    conv_desc_t d{};
    d.N = N; d.C = C; d.H = H; d.W = W; d.K = K; d.R = R; d.S = R;
    d.stride = stride; d.pad = R / 2;
    d.OH = (H + 2 * d.pad - R) / stride + 1;
    d.OW = (W + 2 * d.pad - R) / stride + 1;
    d.in_halo = 1; d.out_halo = 1; d.fuse_silu = silu;
    return d;
}

// Run `fn` through the blocked layouts and compare with the NCHW reference.
static double check_kernel(const char *name, conv_fn_t fn, conv_desc_t d, std::mt19937 &rng,
                           pack_fn_t pack = pack_weights) {
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t)d.N * d.C * d.H * d.W), w((size_t)d.K * d.C * d.R * d.S), b(d.K);
    for (auto &v : x) v = nd(rng);
    for (auto &v : w) v = nd(rng) / std::sqrt((float)(d.C * d.R * d.S));
    for (auto &v : b) v = 0.1f * nd(rng);
    size_t n_out = (size_t)d.N * d.K * d.OH * d.OW;
    std::vector<float> ref(n_out), got(n_out);
    conv_ref_nchw(&d, x.data(), w.data(), b.data(), ref.data());

    float *xin = alloc_aligned(blk_size(d.N, d.C, d.H, d.W, d.in_halo));
    float *wp = alloc_aligned(wpack_size(d.K, d.C, d.R, d.S));
    float *bp = alloc_aligned((size_t)ceil_div(d.K, CB) * CB);
    float *out = alloc_aligned(blk_size(d.N, d.K, d.OH, d.OW, d.out_halo));
    for (int k = 0; k < d.K; k++) bp[k] = b[k];
    nchw_to_blk(x.data(), d.N, d.C, d.H, d.W, d.in_halo, xin);
    pack(w.data(), d.K, d.C, d.R, d.S, wp);
    fn(&d, xin, wp, bp, out);
    blk_to_nchw(out, d.N, d.K, d.OH, d.OW, d.out_halo, got.data());

    // the output halo must stay zero: the next 3x3 layer reads it as padding
    int OHp = d.OH + 2 * d.out_halo, OWp = d.OW + 2 * d.out_halo, Kb = ceil_div(d.K, CB);
    for (int kb = 0; kb < Kb; kb++)
        for (int y = 0; y < OHp; y++)
            for (int xx = 0; xx < OWp; xx++) {
                bool interior = y >= d.out_halo && y < d.OH + d.out_halo && xx >= d.out_halo && xx < d.OW + d.out_halo;
                if (!interior)
                    for (int l = 0; l < CB; l++) CHECK(out[blk_idx(Kb, OHp, OWp, 0, kb, y, xx) + l] == 0.0f);
            }

    double err = 0;
    for (size_t i = 0; i < n_out; i++)
        err = std::fmax(err, std::fabs(got[i] - ref[i]) / std::fmax(std::fabs(ref[i]), 1.0));
    std::printf("  %-8s N=%d C=%-3d %dx%-3d K=%-3d %dx%d s%d silu=%d  max_rel_err=%.1e %s\n", name, d.N, d.C, d.H,
                d.W, d.K, d.R, d.S, d.stride, d.fuse_silu, err, err <= 1e-4 ? "ok" : "FAIL");
    free(xin); free(wp); free(bp); free(out);
    return err;
}

int main() {
    // ---- hand-computed checks of the reference (3x3 image, all-ones kernel) ----
    const std::vector<float> x{1, 2, 3, 4, 5, 6, 7, 8, 9};
    const std::vector<float> ones(9, 1.0f);
    std::vector<float> y(9), ds(4);
    conv_desc_t d = make_desc(1, 1, 3, 3, 1, 3, 1, 0);
    conv_ref_nchw(&d, x.data(), ones.data(), nullptr, y.data());
    CHECK(close(y[0], 12.0f)); CHECK(close(y[4], 45.0f)); CHECK(close(y[8], 28.0f));
    d = make_desc(1, 1, 3, 3, 1, 3, 2, 0);
    conv_ref_nchw(&d, x.data(), ones.data(), nullptr, ds.data());
    CHECK(close(ds[0], 12.0f)); CHECK(close(ds[3], 28.0f));

    // ---- SPPF pooling reference ----
    auto sppf = sppf_pool_reference(x, 1, 3, 3);
    CHECK(sppf.size() == 36);
    CHECK(close(sppf[4], 5.0f));
    for (std::size_t i = 9; i < sppf.size(); ++i) CHECK(close(sppf[i], 9.0f));

    // ---- every kernel vs the reference on yolov8n channel counts (small spatial size) ----
    std::mt19937 rng(645);
    double worst = 0;
    struct { int C, K, H; } s1[] = {{16, 16, 20}, {32, 32, 20}, {64, 64, 10}, {128, 128, 10}, {64, 80, 10}, {256, 64, 5}, {32, 32, 7}};
    for (auto s : s1) {
        worst = std::fmax(worst, check_kernel("s1_v0", conv3x3_s1_v0, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v1", conv3x3_s1_v1, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v2", conv3x3_s1_v2, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v3", conv3x3_s1_v3, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
    }
    // v4/v4b/v5 tails: widths that leave 0, 1, 2, 3, 4 and 5 pixels after the 6-wide tiles
    for (int Wd : {6, 7, 8, 9, 10, 11, 20, 40})
        for (int C : {16, 64}) {
            worst = std::fmax(worst, check_kernel("s1_v4", conv3x3_s1_v4, make_desc(1, C, 5, Wd, 32, 3, 1, 1), rng));
            worst = std::fmax(worst, check_kernel("s1_v4b", conv3x3_s1_v4b, make_desc(1, C, 5, Wd, 32, 3, 1, 1), rng));
            worst = std::fmax(worst, check_kernel("s1_v5", conv3x3_s1_v5, make_desc(1, C, 5, Wd, 32, 3, 1, 1), rng, pack_weights_s1_kb2));
        }
    worst = std::fmax(worst, check_kernel("s1_v5", conv3x3_s1_v5, make_desc(1, 256, 10, 10, 80, 3, 1, 1), rng, pack_weights_s1_kb2));
    worst = std::fmax(worst, check_kernel("s1_v5", conv3x3_s1_v5, make_desc(2, 32, 8, 8, 16, 3, 1, 0), rng, pack_weights_s1_kb2));
    worst = std::fmax(worst, check_kernel("s1_v3", conv3x3_s1_v3, make_desc(2, 32, 10, 10, 32, 3, 1, 0), rng));
    {   // every sweep tile on one shape where all of them apply (K/8 = 4, width 20)
        conv_fn_t tiles[] = {conv3x3_s1_t1x1, conv3x3_s1_t1x2, conv3x3_s1_t1x4, conv3x3_s1_t1x5, conv3x3_s1_t1x10,
                             conv3x3_s1_t2x1, conv3x3_s1_t2x2, conv3x3_s1_t2x4, conv3x3_s1_t2x5};
        for (conv_fn_t t : tiles) worst = std::fmax(worst, check_kernel("s1_tile", t, make_desc(1, 32, 20, 20, 32, 3, 1, 1), rng));
    }
    struct { int C, K, H; } s2[] = {{3, 16, 20}, {16, 32, 20}, {64, 128, 10}, {128, 256, 10}};
    for (auto s : s2) worst = std::fmax(worst, check_kernel("s2_v1", conv3x3_s2_v1, make_desc(1, s.C, s.H, s.H, s.K, 3, 2, 1), rng));
    struct { int C, K, H; } k1[] = {{32, 32, 20}, {48, 32, 20}, {384, 256, 5}, {64, 64, 10}};
    for (auto s : k1) worst = std::fmax(worst, check_kernel("1x1_v1", conv1x1_v1, make_desc(1, s.C, s.H, s.H, s.K, 1, 1, 1), rng));

    {   // SPPF pools on NCHW8c vs the NCHW reference. Max is exact, so the outputs must be identical.
        // Shapes: the real 128 x 20 x 20, odd sizes, lines shorter than the 5-wide window, batch 2,
        // and all-negative inputs (a zero instead of -inf padding would show up as 0 at the border).
        pool_fn_t pools[] = {sppf_pool_v1, sppf_pool_v2, sppf_pool_v3, sppf_pool_v4};
        const char *pnames[] = {"sppf_v1", "sppf_v2", "sppf_v3", "sppf_v4"};
        struct { int N, C, H, W; float shift; } ps[] = {{1, 128, 20, 20, 0.f}, {1, 16, 7, 9, 0.f}, {2, 8, 5, 5, 0.f},
                                                        {1, 8, 3, 4, 0.f}, {1, 8, 1, 6, 0.f}, {1, 24, 6, 11, -10.f}};
        std::normal_distribution<float> nd(0.0f, 1.0f);
        for (int v = 0; v < 4; v++)
            for (auto s : ps) {
                std::vector<float> x((size_t)s.N * s.C * s.H * s.W);
                for (auto &e : x) e = nd(rng) + s.shift;
                const int hi = 1, ho = 1;
                float *xb = alloc_aligned(blk_size(s.N, s.C, s.H, s.W, hi));
                float *ob = alloc_aligned(blk_size(s.N, 4 * s.C, s.H, s.W, ho));
                nchw_to_blk(x.data(), s.N, s.C, s.H, s.W, hi, xb);
                pools[v](s.N, s.C, s.H, s.W, xb, hi, ob, ho);
                std::vector<float> got((size_t)s.N * 4 * s.C * s.H * s.W);
                blk_to_nchw(ob, s.N, 4 * s.C, s.H, s.W, ho, got.data());
                const size_t img = (size_t)s.C * s.H * s.W;
                for (int n = 0; n < s.N; n++) {
                    std::vector<float> xi(x.begin() + n * img, x.begin() + (n + 1) * img);
                    auto ref = sppf_pool_reference(xi, s.C, s.H, s.W);
                    for (size_t i = 0; i < ref.size(); i++) CHECK(got[n * 4 * img + i] == ref[i]);
                }
                // the halo of every output plane must still be zero
                const int Hp = s.H + 2 * ho, Wp = s.W + 2 * ho;
                for (int n = 0; n < s.N; n++)
                    for (int cb = 0; cb < 4 * s.C / CB; cb++)
                        for (int y = 0; y < Hp; y++)
                            for (int xx = 0; xx < Wp; xx++)
                                if (y == 0 || y == Hp - 1 || xx == 0 || xx == Wp - 1)
                                    for (int l = 0; l < CB; l++) CHECK(ob[blk_idx(4 * s.C / CB, Hp, Wp, n, cb, y, xx) + l] == 0.f);
                free(xb); free(ob);
                std::printf("  %-8s N=%d C=%3d %2dx%-2d  exact match\n", pnames[v], s.N, s.C, s.H, s.W);
            }
    }

    CHECK(worst <= 1e-4);
    std::cout << "All kernel correctness checks passed.\n";
}
