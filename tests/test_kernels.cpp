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

    // ---- every isolated stride-1 microkernel vs the scalar one-tile reference ----
    for (int t = 0; t < S1_NUKRS; t++)
        for (int C : {16, 64}) {
            const s1_ukr_entry_t &u = S1_UKRS[t];
            const int Cb = C / 8; const size_t in_row = 12 * 8, in_plane = 3 * in_row, w_kb = (size_t)Cb * 9 * 64, o_kb = 80;
            std::vector<float> in(Cb * in_plane + 64), w(4 * w_kb + 64), b(32), o(4 * o_kb + 64), r(4 * o_kb + 64);
            std::mt19937 g(C + t);
            std::uniform_real_distribution<float> d(-1.f, 1.f);
            float *pi = alloc_aligned(in.size()), *pw = alloc_aligned(w.size()), *po = alloc_aligned(o.size());
            for (size_t i = 0; i < in.size(); i++) pi[i] = d(g);
            for (size_t i = 0; i < w.size(); i++) pw[i] = 0.1f * d(g);
            for (auto &x : b) x = d(g);
            u.fn(Cb, pi, in_plane, in_row, pw, w_kb, b.data(), po, o_kb, 1);
            s1_ukr_reference(u.KB, u.OW, Cb, pi, in_plane, in_row, pw, w_kb, b.data(), r.data(), o_kb, 1);
            double e = 0, m = 0;
            for (int k = 0; k < u.KB; k++)
                for (int i = 0; i < u.OW * 8; i++) { e = std::fmax(e, std::fabs(po[k * o_kb + i] - r[k * o_kb + i])); m = std::fmax(m, std::fabs(r[k * o_kb + i])); }
            std::printf("  ukr %-7s C=%-3d max_rel_err=%.1e %s\n", u.name, C, e / m, e / m <= 1e-4 ? "ok" : "FAIL");
            CHECK(e / m <= 1e-4);
            free(pi); free(pw); free(po);
        }

    // ---- every kernel vs the reference on yolov8n channel counts (small spatial size) ----
    std::mt19937 rng(645);
    double worst = 0;
    struct { int C, K, H; } s1[] = {{16, 16, 20}, {32, 32, 20}, {64, 64, 10}, {128, 128, 10}, {64, 80, 10}, {256, 64, 5}, {32, 32, 7}};
    for (auto s : s1) {
        worst = std::fmax(worst, check_kernel("s1_v0", conv3x3_s1_v0, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v1", conv3x3_s1_v1, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v2", conv3x3_s1_v2, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v3", conv3x3_s1_v3, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v6", conv3x3_s1_v6, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v6b", conv3x3_s1_v6b, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v7", conv3x3_s1_v7, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v7b", conv3x3_s1_v7b, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v7mt", conv3x3_s1_v7mt, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v8", conv3x3_s1_v8, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v2a", conv3x3_s1_v2a, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v8a", conv3x3_s1_v8a, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v8mt", conv3x3_s1_v8mt, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v8b", conv3x3_s1_v8b, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v9", conv3x3_s1_v9, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
        worst = std::fmax(worst, check_kernel("s1_v9mt", conv3x3_s1_v9mt, make_desc(1, s.C, s.H, s.H, s.K, 3, 1, 1), rng));
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
    worst = std::fmax(worst, check_kernel("s1_v7", conv3x3_s1_v7, make_desc(2, 32, 10, 10, 32, 3, 1, 0), rng));
    worst = std::fmax(worst, check_kernel("s1_v7mt", conv3x3_s1_v7mt, make_desc(2, 32, 10, 10, 32, 3, 1, 0), rng));
    for (int Wd : {6, 7, 8, 10, 15, 20, 25, 40})  // v8 row tails: 0, 1, 2 pixels after the 3-wide tiles; K/8 = 4, 10, 2
        for (int K : {32, 80, 16})
        {
            worst = std::fmax(worst, check_kernel("s1_v8", conv3x3_s1_v8, make_desc(1, 64, 5, Wd, K, 3, 1, 1), rng));
            worst = std::fmax(worst, check_kernel("s1_v8a", conv3x3_s1_v8a, make_desc(1, 64, 5, Wd, K, 3, 1, 1), rng));
            worst = std::fmax(worst, check_kernel("s1_v8b", conv3x3_s1_v8b, make_desc(1, 64, 5, Wd, K, 3, 1, 1), rng));
        }
    worst = std::fmax(worst, check_kernel("s1_v8mt", conv3x3_s1_v8mt, make_desc(2, 32, 10, 10, 64, 3, 1, 0), rng));
    {   // every sweep tile on one shape where all of them apply (K/8 = 4, width 20)
        conv_fn_t tiles[] = {conv3x3_s1_t1x1, conv3x3_s1_t1x2, conv3x3_s1_t1x4, conv3x3_s1_t1x5, conv3x3_s1_t1x10,
                             conv3x3_s1_t2x1, conv3x3_s1_t2x2, conv3x3_s1_t2x4, conv3x3_s1_t2x5};
        for (conv_fn_t t : tiles) worst = std::fmax(worst, check_kernel("s1_tile", t, make_desc(1, 32, 20, 20, 32, 3, 1, 1), rng));
    }
    struct { int C, K, H; } s2[] = {{3, 16, 20}, {16, 32, 20}, {64, 128, 10}, {128, 256, 10}};
    for (auto s : s2) worst = std::fmax(worst, check_kernel("s2_v1", conv3x3_s2_v1, make_desc(1, s.C, s.H, s.H, s.K, 3, 2, 1), rng));
    struct { int C, K, H; } k1[] = {{32, 32, 20}, {48, 32, 20}, {384, 256, 5}, {64, 64, 10}};
    for (auto s : k1) worst = std::fmax(worst, check_kernel("1x1_v1", conv1x1_v1, make_desc(1, s.C, s.H, s.H, s.K, 1, 1, 1), rng));

    CHECK(worst <= 1e-4);
    std::cout << "All kernel correctness checks passed.\n";
}
