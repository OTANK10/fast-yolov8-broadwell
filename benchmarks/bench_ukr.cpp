/*
 * bench_ukr.cpp  --  the stride-1 MICROKERNEL alone, benchmarked like the
 * Assignment 2 matrix-multiply kernel.
 *
 * The 2x5 tile IS an Assignment-2-style outer-product kernel:
 *     m = 5 output pixels (broadcast, like A)
 *     n = 16 output channels = 2 vectors of 8 (loaded, like the B row)
 *     k = C * 9 (input channels x 3x3 taps): the reduction
 *     C = the 10 accumulators, stored once at the end (plus bias and SiLU)
 * One call computes one m x n output tile over the whole k, nothing else:
 * no layer loops, no row tails, no walk over image planes. The same small
 * block of inputs and weights is reused on every call, so after warm-up it
 * sits in L1 (or L2 once it outgrows 32 KB, which the sweep shows).
 *
 *   ./build/bench_ukr 32                          one C, every tile, table
 *   ./build/bench_ukr --sweep 8 512 8 --runs 3 --tiles 2x5,2x6 --csv results/ukr_sweep.csv
 *        k = 72 ... 4608, like the assignment's k = 32 ... 512 sweep
 *   options: --no-silu  --reps N  --warmup SECONDS (default 2)
 *   --crosscheck: also time with rdtsc and wall clock; prints the core's real GHz and
 *        % of peak from wall time at max turbo (a floor that needs no perf counter)
 *
 * Cycles are real core cycles from perf_event (not TSC x 3.4/2.4 like the
 * assignment driver), so % of peak is exact under turbo. Pin with taskset.
 * Every tile is checked against a scalar reference at every C first.
 */
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <vector>
#include "common.h"
#include "kernels.h"
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

using namespace fastyolo;

static int perf_open_cycles(void) {
#ifndef __linux__
    return -1;
#else
    struct perf_event_attr pe;
    memset(&pe, 0, sizeof pe);
    pe.type = PERF_TYPE_HARDWARE;
    pe.size = sizeof pe;
    pe.config = PERF_COUNT_HW_CPU_CYCLES;
    pe.exclude_kernel = 1;
    pe.exclude_hv = 1;
    return (int)syscall(SYS_perf_event_open, &pe, 0, -1, -1, 0);
#endif
}
static uint64_t perf_read(int fd) {
    uint64_t v = 0;
#ifdef __linux__
    if (fd >= 0 && read(fd, &v, sizeof v) != (ssize_t)sizeof v) v = 0;
#else
    (void)fd;
#endif
    return v;
}

/* Spin so the core is at full clock before anything is timed (the
 * assignment's "2-second busy loop" fix). */
static void busy_warmup(double seconds) {
    auto t0 = std::chrono::steady_clock::now();
    volatile double x = 1.0;
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) x = x * 1.0000001 + 1e-9;
}

struct Data {
    int C, Cb;
    size_t in_row, in_plane, w_kb, o_kb;
    float *in, *w, *b, *o, *ref;
};
static const int KBMAX = 4, OWMAX = 10;

static Data make_data(int C) {
    Data d;
    d.C = C; d.Cb = C / CB;
    d.in_row = (size_t)(OWMAX + 2) * CB; d.in_plane = 3 * d.in_row;
    d.w_kb = (size_t)d.Cb * 9 * CB * CB; d.o_kb = (size_t)OWMAX * CB;
    d.in = alloc_aligned(d.Cb * d.in_plane); d.w = alloc_aligned(KBMAX * d.w_kb);
    d.b = alloc_aligned(KBMAX * CB); d.o = alloc_aligned(KBMAX * d.o_kb); d.ref = alloc_aligned(KBMAX * d.o_kb);
    std::mt19937 rng(645 + C);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (size_t i = 0; i < d.Cb * d.in_plane; i++) d.in[i] = u(rng);
    for (size_t i = 0; i < KBMAX * d.w_kb; i++) d.w[i] = u(rng) * 0.1f;
    for (int i = 0; i < KBMAX * CB; i++) d.b[i] = u(rng);
    return d;
}
static void free_data(Data &d) { free(d.in); free(d.w); free(d.b); free(d.o); free(d.ref); }

struct Result { double flops, best, med, fpc, pct, rel, wkb, wskb, ghz_core, ghz_tsc, gflops_wall; bool pass; };
static double now_ns() { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

static Result run_tile(const s1_ukr_entry_t &k, Data &d, int fs, long reps, int trials, int pfd, double peak) {
    Result r{};
    memset(d.o, 0, KBMAX * d.o_kb * sizeof(float));
    k.fn(d.Cb, d.in, d.in_plane, d.in_row, d.w, d.w_kb, d.b, d.o, d.o_kb, fs);
    s1_ukr_reference(k.KB, k.OW, d.Cb, d.in, d.in_plane, d.in_row, d.w, d.w_kb, d.b, d.ref, d.o_kb, fs);
    double err = 0, mx = 0;
    for (int kb = 0; kb < k.KB; kb++)
        for (int i = 0; i < k.OW * CB; i++) {
            const size_t ix = kb * d.o_kb + i;
            err = std::max(err, (double)std::fabs(d.o[ix] - d.ref[ix]));
            mx = std::max(mx, (double)std::fabs(d.ref[ix]));
        }
    r.rel = err / std::max(mx, 1e-30);
    r.pass = r.rel <= 1e-4;
    r.flops = 2.0 * k.KB * CB * k.OW * (double)d.C * 9;
    const long n = reps > 0 ? reps : std::max(1000L, (long)(1.5e8 / r.flops)); /* ~25-50 ms per trial */
    for (long i = 0; i < n / 4; i++) k.fn(d.Cb, d.in, d.in_plane, d.in_row, d.w, d.w_kb, d.b, d.o, d.o_kb, fs);
    /* Three clocks around the same loop: core cycles (perf), TSC ticks
     * (fixed base clock) and wall time. Their ratios are the cross-check. */
    std::vector<double> cyc;
    double best_ns = 0, best_ticks = 0, best_cyc = 1e300;
    for (int t = 0; t < trials; t++) {
        const double w0 = now_ns();
        const uint64_t t0 = rdtsc();
        uint64_t c0 = pfd >= 0 ? perf_read(pfd) : t0;
        for (long i = 0; i < n; i++) k.fn(d.Cb, d.in, d.in_plane, d.in_row, d.w, d.w_kb, d.b, d.o, d.o_kb, fs);
        const uint64_t t1 = rdtsc();
        uint64_t c1 = pfd >= 0 ? perf_read(pfd) : t1;
        const double w1 = now_ns();
        const double cpc = (double)(c1 - c0) / n;
        cyc.push_back(cpc);
        if (cpc < best_cyc) { best_cyc = cpc; best_ns = (w1 - w0) / n; best_ticks = (double)(t1 - t0) / n; }
    }
    std::sort(cyc.begin(), cyc.end());
    r.best = cyc[0]; r.med = cyc[cyc.size() / 2];
    r.ghz_core = r.best / best_ns;                 /* core cycles per ns = the clock the core really ran at */
    r.ghz_tsc = best_ticks / best_ns;              /* should be the TSC's fixed rate (2.40 on ece028) */
    r.gflops_wall = r.flops / best_ns;             /* FLOPs per ns = GFLOP/s, from wall time alone */
    r.fpc = r.best > 0 ? r.flops / r.best : 0;
    r.pct = 100 * r.fpc / peak;
    r.wkb = k.KB * (double)d.C * 288 / 1024;                                  /* weights read per call */
    r.wskb = r.wkb + 3.0 * (k.OW + 2) * d.C * 4 / 1024;                        /* + input rows touched */
    return r;
}

int main(int argc, char **argv) {
    int C = 32, fs = 1, runs = 1, c0 = 0, c1 = 0, cstep = 8;
    long reps = 0;
    double warm = 2.0;
    int cross = 0;
    const double max_ghz = getenv("MAX_GHZ") ? atof(getenv("MAX_GHZ")) : 3.4; /* E5-2640 v4 max turbo */
    const char *csv = nullptr;
    std::string tiles;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--no-silu")) fs = 0;
        else if (!strcmp(argv[i], "--crosscheck")) cross = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atol(argv[++i]);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else if (!strcmp(argv[i], "--runs") && i + 1 < argc) runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--warmup") && i + 1 < argc) warm = atof(argv[++i]);
        else if (!strcmp(argv[i], "--tiles") && i + 1 < argc) tiles = std::string(",") + argv[++i] + ",";
        else if (!strcmp(argv[i], "--sweep") && i + 3 < argc) { c0 = atoi(argv[++i]); c1 = atoi(argv[++i]); cstep = atoi(argv[++i]); }
        else C = atoi(argv[i]);
    }
    if (S1_NUKRS == 0) { fprintf(stderr, "built without AVX2 (-DYOLO_ENABLE_BROADWELL=ON): no microkernels\n"); return 1; }
    std::vector<int> Cs;
    if (c0 > 0) for (int c = c0; c <= c1; c += cstep) Cs.push_back(c);
    else Cs.push_back(C);
    for (int c : Cs) if (c < 8 || c % 8) { fprintf(stderr, "C must be a multiple of 8 (got %d)\n", c); return 1; }
    const double peak = getenv("PEAK_FPC") ? atof(getenv("PEAK_FPC")) : PEAK_FLOPS_PER_CYCLE_PER_CORE;
    const bool sweep = Cs.size() > 1 || runs > 1;
    const int trials = sweep ? 5 : 9;

    int pfd = perf_open_cycles();
    const char *src = pfd >= 0 ? "core" : "tsc";
    if (pfd < 0) fprintf(stderr, "warning: perf_event not available, using TSC ticks (base clock): %% of peak is inflated under turbo\n");

    FILE *f = csv ? fopen(csv, "w") : nullptr;
    if (f) fprintf(f, "run,tile,KB,OW,accumulators,registers,C,k,silu,weights_kb,working_set_kb,cycle_src,flops_per_call,best_cycles_per_call,median_cycles_per_call,flops_per_cycle,pct_peak,max_rel_err,pass,note\n");
    printf("stride-1 microkernel alone: SiLU %s, peak %.0f FLOPs/cycle, %s cycles, %d run(s)\n", fs ? "on" : "off", peak, src, runs);

    int fails = 0;
    for (int run = 1; run <= runs; run++) {
        busy_warmup(warm);
        if (sweep) { printf("  run %d/%d: C = %d ... %d\n", run, runs, Cs.front(), Cs.back()); fflush(stdout); }
        for (int c : Cs) {
            Data d = make_data(c);
            if (!sweep) printf("C = %d (k = C*9 = %d)\n%-8s %4s %5s %8s %9s %11s %10s %7s %9s\n", c, c * 9, "tile", "acc", "regs",
                               "weights", "FLOP/call", "cycles/call", "FLOPs/cyc", "%peak", "max_err");
            for (int t = 0; t < S1_NUKRS; t++) {
                const s1_ukr_entry_t &k = S1_UKRS[t];
                if (!tiles.empty() && tiles.find(std::string(",") + k.name + ",") == std::string::npos) continue;
                Result r = run_tile(k, d, fs, reps, trials, pfd, peak);
                fails += !r.pass;
                if (cross) {
                    /* % of peak three ways: from core cycles (what we report), from wall time if the core had run
                     * at max turbo the whole time (a floor: the true clock cannot be higher), and at base clock. */
                    const double pk_turbo = 100 * r.gflops_wall / (max_ghz * peak), pk_base = 100 * r.gflops_wall / (2.4 * peak);
                    printf("%-8s core clock %.2f GHz | TSC %.2f GHz | %6.1f GFLOP/s wall | %%peak: %5.1f%% (core cycles)  >= %5.1f%% (wall @ %.1f GHz max turbo)  %5.1f%% (wall @ 2.4 GHz base)%s\n",
                           k.name, r.ghz_core, r.ghz_tsc, r.gflops_wall, r.pct, pk_turbo, max_ghz, pk_base,
                           (r.ghz_core > max_ghz * 1.02 || r.ghz_core < 1.0) ? "  <-- core clock implausible, cycle count suspect" : "");
                }
                if (!sweep && !cross)
                    printf("%-8s %4d %5d %6.1fKB %9.0f %11.1f %10.2f %6.1f%% %9.1e  %s%s\n", k.name, k.KB * k.OW, k.regs, r.wkb, r.flops,
                           r.best, r.fpc, r.pct, r.rel, r.pass ? "" : "FAIL  ", k.note);
                if (f)
                    fprintf(f, "%d,%s,%d,%d,%d,%d,%d,%d,%d,%.1f,%.1f,%s,%.0f,%.2f,%.2f,%.3f,%.2f,%.2e,%d,%s\n", run, k.name, k.KB, k.OW,
                            k.KB * k.OW, k.regs, c, c * 9, fs, r.wkb, r.wskb, src, r.flops, r.best, r.med, r.fpc, r.pct, r.rel,
                            r.pass ? 1 : 0, k.note);
            }
            free_data(d);
        }
    }
    printf("%s; weights: L1D = 32 KB, L2 = 256 KB\n", fails ? "SOME TILES FAILED" : "all tiles PASS (max rel err <= 1e-4)");
    if (f) fclose(f);
    return fails ? 1 : 0;
}
