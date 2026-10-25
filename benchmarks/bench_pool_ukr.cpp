/*
 * bench_pool_ukr.cpp  --  the SPPF pool microkernel ALONE, Assignment 2 style.
 *
 * One call = one 5x5 stride-1 max-pool (pad 2, -inf semantics) of ONE 8-channel plane
 * of H x W pixels, src -> dst. No other channel blocks, no chaining of the 3 pools, no
 * concat copy, no layout traversal. The same plane is pooled over and over, so it stays
 * in cache, exactly like bench_ukr for the conv microkernel.
 *
 *   ./build/bench_pool_ukr                       H = W = 20 (model.9), every version
 *   ./build/bench_pool_ukr 20 --ukrs v2,v5
 *   ./build/bench_pool_ukr --sweep 5 40 1 --runs 3 --csv results/pool_ukr_sweep.csv
 *
 * There are no FLOPs. The work unit is vmaxps (one = 8 channel maxes), and Broadwell
 * issues vmaxps only on port 1, so the peak is 1 vmaxps per cycle. Reported:
 * vmaxps/cycle (% of that peak) and cycles per output vector.
 * Cycles = rdtsc ticks x 3.4 / 2.4 (TSC at the 2.4 GHz base, core assumed at its 3.4 GHz
 * max turbo, Assignment 2 convention). Warm-up 2 s busy loop. Pin with taskset.
 * Every version is checked against a scalar 5x5 max at every size first (exact match).
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include "common.h"
#include "kernels.h"

using namespace fastyolo;

static void busy_warmup(double seconds) {
    auto t0 = std::chrono::steady_clock::now();
    volatile double x = 1.0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) x = x * 1.0000001 + 1e-9;
}

int main(int argc, char **argv) {
    int S = 20, s0 = 0, s1 = 0, sstep = 1, runs = 1;
    double warm = 2.0, turbo = 3.4, base = 2.4;
    long reps = 0;
    const char *csv = nullptr;
    std::string ukrs;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--sweep") && i + 3 < argc) { s0 = atoi(argv[++i]); s1 = atoi(argv[++i]); sstep = atoi(argv[++i]); }
        else if (!strcmp(argv[i], "--runs") && i + 1 < argc) runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warm = atof(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atol(argv[++i]);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!strcmp(argv[i], "--ukrs") && i + 1 < argc) ukrs = std::string(",") + argv[++i] + ",";
        else if (!strcmp(argv[i], "--clock") && i + 1 < argc) turbo = atof(argv[++i]);
        else if (!strcmp(argv[i], "--base") && i + 1 < argc) base = atof(argv[++i]);
        else S = atoi(argv[i]);
    }
    {   /* the TSC x turbo/base rule needs the TSC to tick at `base` GHz (2.40 on ece028) */
        auto w0 = std::chrono::steady_clock::now(); const uint64_t t0 = rdtsc();
        busy_warmup(0.1);
        const double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - w0).count();
        const double ghz = (rdtsc() - t0) / ns;
        if (std::fabs(ghz / base - 1) > 0.02)
            fprintf(stderr, "warning: TSC runs at %.2f GHz here, not --base %.2f: cycles are off (on ece028 it is 2.40)\n", ghz, base);
    }
    std::vector<int> sizes;
    if (s0 > 0) for (int s = s0; s <= s1; s += sstep) sizes.push_back(s);
    else sizes.push_back(S);
    const bool sweep = sizes.size() > 1 || runs > 1;

    FILE *f = csv ? fopen(csv, "w") : nullptr;
    if (f) fprintf(f, "run,ukr,H,W,working_set_kb,vmaxps_per_call,best_cycles_per_call,vmaxps_per_cycle,pct_peak,cycles_per_output_vec,exact,note\n");
    printf("SPPF pool microkernel alone: one 5x5 pool of one 8-channel plane, cycles = TSC x %.1f/%.1f, peak 1 vmaxps/cycle (port 1)\n", turbo, base);
    if (!sweep) printf("%-4s %7s %9s %12s %12s %8s %13s  %s\n", "ukr", "H x W", "set (KB)", "vmaxps/call", "cycles/call", "vmax/cyc", "cyc/out vec", "note");

    int fails = 0;
    std::mt19937 rng(645);
    std::normal_distribution<float> nd(0.f, 1.f);
    for (int run = 1; run <= runs; run++) {
        busy_warmup(warm);
        if (sweep) { printf("  run %d/%d: H = W = %d ... %d\n", run, runs, sizes.front(), sizes.back()); fflush(stdout); }
        for (int s : sizes) {
            const int H = s, W = s, Wp = W + 2;
            const size_t sr = (size_t)Wp * CB;                               /* stored with the 1-pixel halo */
            float *src = alloc_aligned((size_t)(H + 2) * sr), *dst = alloc_aligned((size_t)(H + 2) * sr);
            float *scr = alloc_aligned((size_t)H * W * CB);
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    for (int l = 0; l < CB; l++) src[(y + 1) * sr + (x + 1) * CB + l] = nd(rng);   /* includes negatives */
            /* scalar reference: clamped window == -inf padding */
            std::vector<float> ref((size_t)H * W * CB);
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++)
                    for (int l = 0; l < CB; l++) {
                        float m = -std::numeric_limits<float>::infinity();
                        for (int yy = std::max(0, y - 2); yy <= std::min(H - 1, y + 2); yy++)
                            for (int xx = std::max(0, x - 2); xx <= std::min(W - 1, x + 2); xx++)
                                m = std::max(m, src[(yy + 1) * sr + (xx + 1) * CB + l]);
                        ref[((size_t)y * W + x) * CB + l] = m;
                    }
            const float *si = src + sr + CB;
            float *di = dst + sr + CB;
            for (int t = 0; t < POOL_NUKRS; t++) {
                const pool_ukr_entry_t &k = POOL_UKRS[t];
                if (!ukrs.empty() && ukrs.find(std::string(",") + k.name + ",") == std::string::npos) continue;
                memset(dst, 0, (size_t)(H + 2) * sr * sizeof(float));
                k.fn(si, sr, di, sr, scr, H, W);
                bool exact = true;
                for (int y = 0; y < H && exact; y++)
                    for (int x = 0; x < W * CB; x++)
                        if (di[y * sr + x] != ref[(size_t)y * W * CB + x]) { exact = false; break; }
                fails += !exact;
                const double vmax = (double)H * sppf_vmax_per_line(W, k.rows_mode) + (double)W * sppf_vmax_per_line(H, k.cols_mode);
                const long n = reps > 0 ? reps : std::max(200L, (long)(2e6 / vmax));   /* ~2-3 ms per trial */
                for (long i = 0; i < n / 4; i++) k.fn(si, sr, di, sr, scr, H, W);
                double best = 1e300;
                for (int tr = 0; tr < (sweep ? 5 : 9); tr++) {
                    const uint64_t t0 = rdtsc();
                    for (long i = 0; i < n; i++) k.fn(si, sr, di, sr, scr, H, W);
                    const uint64_t t1 = rdtsc();
                    best = std::min(best, (double)(t1 - t0) / n);
                }
                const double cyc = best * turbo / base, vpc = vmax / cyc;
                const double setkb = (2.0 * (H + 2) * sr + (k.scratch_rows ? (double)k.scratch_rows * W * CB : (double)H * W * CB)) * 4 / 1024;
                if (!sweep)
                    printf("%-4s %3dx%-3d %9.1f %12.0f %12.0f %8.2f %13.2f  %s%s\n", k.name, H, W, setkb, vmax, cyc, vpc,
                           cyc / ((double)H * W), k.note, exact ? "" : "  <-- WRONG");
                if (f) fprintf(f, "%d,%s,%d,%d,%.1f,%.0f,%.0f,%.4f,%.2f,%.3f,%d,\"%s\"\n", run, k.name, H, W, setkb, vmax, cyc, vpc,
                               100 * vpc, cyc / ((double)H * W), exact, k.note);
            }
            free(src); free(dst); free(scr);
        }
    }
    if (f) { fclose(f); printf("wrote %s\n", csv); }
    printf(fails ? "%d results WRONG\n" : "all versions exact vs the scalar 5x5 max\n", fails);
    return fails ? 1 : 0;
}
