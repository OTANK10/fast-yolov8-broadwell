/*
 * bench_pool.cpp  --  SPPF max-pool (model.9) on the real tensors: correctness vs
 * the baseline's own MaxPool outputs, then timing.
 *
 *   ./build/bench_pool data/tensors/b1 [v1|v2|ref] [--reps N] [--csv results/sppf_v2_b1_t1.csv]
 *
 * Reads data/tensors/b<N>/model.9.m*.MaxPool/ (written by scripts/dump_layers.py):
 * the first pool's input is X, the three outputs are P1, P2, P3. Layout conversion
 * happens outside the timed region; the timed region is only the pool call.
 *
 * Pooling has no FLOPs, so we report what bounds it instead:
 *   - vmaxps per cycle: Broadwell issues vmaxps only on port 1, so the peak is 1.0
 *   - cycles per output vector (8 channels of one pixel, per pool)
 *   - GB/s of compulsory traffic (read X, write the 4 slices of the concat)
 * Cycles = rdtsc ticks x 3.4 / 2.4 (TSC at the 2.4 GHz base, core assumed at 3.4 GHz
 * max turbo, as in Assignment 2). "ref" is the scalar NCHW reference, for scale.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <string>
#include <vector>
#include "common.h"
#include "kernels.h"

using namespace fastyolo;

static std::vector<float> read_bin(const std::string &p, size_t n) {
    std::vector<float> v(n);
    FILE *f = fopen(p.c_str(), "rb");
    if (!f || fread(v.data(), sizeof(float), n, f) != n) { fprintf(stderr, "cannot read %zu floats from %s\n", n, p.c_str()); exit(1); }
    fclose(f);
    return v;
}

static void busy_warmup(double seconds) {
    auto t0 = std::chrono::steady_clock::now();
    volatile double x = 1.0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) x = x * 1.0000001 + 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s data/tensors/b1 [v1|v2|ref] [--reps N] [--csv out.csv]\n", argv[0]); return 1; }
    std::string root = argv[1], ver = "v2";
    const char *csv = nullptr;
    long reps = 0;
    double turbo = 3.4, base = 2.4;
    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atol(argv[++i]);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!strcmp(argv[i], "--clock") && i + 1 < argc) turbo = atof(argv[++i]);
        else if (!strcmp(argv[i], "--base") && i + 1 < argc) base = atof(argv[++i]);
        else ver = argv[i];
    }

    /* the three chained pools: model.9.m.MaxPool, model.9.m_1.MaxPool, model.9.m_2.MaxPool */
    std::vector<std::string> dirs;
    if (DIR *dp = opendir(root.c_str())) {
        while (dirent *e = readdir(dp)) {
            std::string n = e->d_name;
            if (n.rfind("model.9.m", 0) == 0 && n.size() > 8 && n.find("MaxPool") != std::string::npos) dirs.push_back(n);
        }
        closedir(dp);
    }
    std::sort(dirs.begin(), dirs.end());
    if (dirs.size() != 3) { fprintf(stderr, "expected 3 model.9.m*.MaxPool dirs in %s, found %zu (run scripts/dump_layers.py)\n", root.c_str(), dirs.size()); return 1; }
    int N, C, H, W, k, st, pad, OH, OW;
    {
        FILE *f = fopen((root + "/" + dirs[0] + "/meta.txt").c_str(), "r");
        if (!f || fscanf(f, "%d %d %d %d %d %d %d %d %d", &N, &C, &H, &W, &k, &st, &pad, &OH, &OW) != 9) { fprintf(stderr, "bad meta.txt\n"); return 1; }
        fclose(f);
    }
    if (k != 5 || st != 1 || pad != 2 || OH != H || OW != W || C % CB) { fprintf(stderr, "unexpected pool shape\n"); return 1; }
    const size_t img = (size_t)N * C * H * W;
    std::vector<float> x = read_bin(root + "/" + dirs[0] + "/input.bin", img), want[3];
    for (int p = 0; p < 3; p++) want[p] = read_bin(root + "/" + dirs[p] + "/output.bin", img);

    pool_fn_t fn = ver == "v1" ? sppf_pool_v1 : ver == "v2" ? sppf_pool_v2 : nullptr;
    if (!fn && ver != "ref") { fprintf(stderr, "unknown version %s (v1, v2, ref)\n", ver.c_str()); return 1; }

    /* layouts, outside timing */
    const int halo = 1;
    float *xb = alloc_aligned(blk_size(N, C, H, W, halo)), *ob = alloc_aligned(blk_size(N, 4 * C, H, W, halo));
    nchw_to_blk(x.data(), N, C, H, W, halo, xb);
    std::vector<float> got((size_t)4 * img), nchw_out;

    /* correctness against the baseline's own MaxPool outputs (max is exact: expect 0) */
    double err = 0;
    if (fn) {
        fn(N, C, H, W, xb, halo, ob, halo);
        blk_to_nchw(ob, N, 4 * C, H, W, halo, got.data());
    }
    for (int n = 0; n < N; n++) {
        const size_t one = (size_t)C * H * W;
        std::vector<float> r;
        if (!fn) r = sppf_pool_reference(std::vector<float>(x.begin() + n * one, x.begin() + (n + 1) * one), C, H, W);
        const float *g = fn ? got.data() + n * 4 * one : r.data();
        for (size_t i = 0; i < one; i++) {
            err = std::max(err, (double)std::fabs(g[i] - x[n * one + i]));
            for (int p = 0; p < 3; p++) err = std::max(err, (double)std::fabs(g[(p + 1) * one + i] - want[p][n * one + i]));
        }
    }
    const bool pass = err <= 1e-6;
    printf("SPPF pools %s: N=%d C=%d %dx%d, max abs err vs baseline = %.1e  %s\n", ver.c_str(), N, C, H, W, err, pass ? "PASS" : "FAIL");

    /* timing: best of 9 trials, each the mean of n calls */
    busy_warmup(1.0);
    const long n = reps > 0 ? reps : (fn ? 2000 : 20);
    double best = 1e300;
    for (int t = 0; t < 9; t++) {
        const uint64_t t0 = rdtsc();
        for (long i = 0; i < n; i++) {
            if (fn) fn(N, C, H, W, xb, halo, ob, halo);
            else nchw_out = sppf_pool_reference(std::vector<float>(x.begin(), x.begin() + (size_t)C * H * W), C, H, W);
        }
        const uint64_t t1 = rdtsc();
        best = std::min(best, (double)(t1 - t0) / n);
    }
    const double cyc = best * turbo / base, us = best / (base * 1e3);
    const double vecs = 3.0 * N * (C / CB) * H * W;                       /* output vectors, all 3 pools */
    const double vmax = 3.0 * N * (C / CB) * ((double)H * sppf_vmax_per_line(W, ver == "v2") + (double)W * sppf_vmax_per_line(H, ver == "v2"));
    const double bytes = 5.0 * img * sizeof(float);                        /* read X + write X, P1, P2, P3 */
    printf("  %.2f us  %.0f cycles  %.2f cycles/output vector  ", us, cyc, cyc / vecs);
    if (fn) printf("%.0f vmaxps -> %.2f per cycle (port-1 peak 1.00)  ", vmax, vmax / cyc);
    printf("%.1f GB/s\n", bytes / (us * 1e3));
    if (csv) {
        FILE *f = fopen(csv, "w");
        fprintf(f, "kernel,N,C,H,W,us,cycles,cycles_per_vec,vmaxps,vmaxps_per_cycle,gb_s,max_abs_err,pass\n");
        fprintf(f, "sppf_%s,%d,%d,%d,%d,%.3f,%.0f,%.3f,%.0f,%.3f,%.2f,%.2e,%d\n", ver.c_str(), N, C, H, W, us, cyc, cyc / vecs,
                fn ? vmax : 0, fn ? vmax / cyc : 0, bytes / (us * 1e3), err, pass);
        fclose(f);
        printf("wrote %s\n", csv);
    }
    free(xb); free(ob);
    return pass ? 0 : 1;
}
