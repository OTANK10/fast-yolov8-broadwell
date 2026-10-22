/*
 * bench_ukr.cpp  --  the stride-1 MICROKERNEL alone, no layer around it.
 *
 *   ./build/bench_ukr [C] [--no-silu] [--reps N] [--csv results/ukr_C<C>.csv]
 *
 * Each register tile in S1_UKRS (src/conv3x3_s1.cpp) is called over and over
 * on the same small block of data: KB output-channel blocks x OW pixels, a
 * full reduction over C input channels x 3 x 3 taps, bias and SiLU included.
 * That is exactly one microkernel call inside a real layer, minus everything
 * the layer adds: no loops over rows and output blocks, no row tails, no
 * traversal of the NCHW8c planes, no cache misses (inputs and weights are
 * tiny and reused every call, so after warm-up they sit in L1/L2; the
 * "weights" column says which).
 *
 * Cycles are real core cycles (perf_event), so % of peak is honest under
 * turbo; pin it with `taskset -c 0`. Every tile is first checked against a
 * scalar reference on the same data (max relative error <= 1e-4).
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
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

int main(int argc, char **argv) {
    int C = 32, fs = 1;
    long reps = 0;
    const char *csv = nullptr;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--no-silu")) fs = 0;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atol(argv[++i]);
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc) csv = argv[++i];
        else C = atoi(argv[i]);
    }
    if (S1_NUKRS == 0) { fprintf(stderr, "built without AVX2 (-DYOLO_ENABLE_BROADWELL=ON): no microkernels\n"); return 1; }
    if (C < 8 || C % 8) { fprintf(stderr, "C must be a multiple of 8\n"); return 1; }
    const double peak = getenv("PEAK_FPC") ? atof(getenv("PEAK_FPC")) : PEAK_FLOPS_PER_CYCLE_PER_CORE;

    /* One tile's worth of data for the largest tile (KB <= 4, OW <= 10). */
    const int Cb = C / CB, KBMAX = 4, OWMAX = 10;
    const size_t in_row = (size_t)(OWMAX + 2) * CB, in_plane = 3 * in_row;
    const size_t w_kb = (size_t)Cb * 9 * CB * CB, o_kb = (size_t)OWMAX * CB;
    float *in = alloc_aligned(Cb * in_plane), *w = alloc_aligned(KBMAX * w_kb);
    float *b = alloc_aligned(KBMAX * CB), *o = alloc_aligned(KBMAX * o_kb), *ref = alloc_aligned(KBMAX * o_kb);
    std::mt19937 rng(645);
    std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (size_t i = 0; i < Cb * in_plane; i++) in[i] = u(rng);
    for (size_t i = 0; i < KBMAX * w_kb; i++) w[i] = u(rng) * 0.1f;
    for (int i = 0; i < KBMAX * CB; i++) b[i] = u(rng);

    int pfd = perf_open_cycles();
    const char *src = pfd >= 0 ? "core" : "tsc";
    if (pfd < 0) fprintf(stderr, "warning: perf_event not available, using TSC ticks (base clock): %% of peak is inflated under turbo\n");

    FILE *f = csv ? fopen(csv, "w") : nullptr;
    if (f) fprintf(f, "tile,KB,OW,accumulators,registers,C,silu,weights_kb,cycle_src,flops_per_call,best_cycles_per_call,median_cycles_per_call,flops_per_cycle,pct_peak,max_rel_err,pass,note\n");
    printf("microkernel alone: C = %d (reduction %d MACs per output), SiLU %s, peak %.0f FLOPs/cycle, %s cycles\n", C, C * 9, fs ? "on" : "off", peak, src);
    printf("%-8s %4s %5s %8s %9s %11s %10s %7s %9s  %s\n", "tile", "acc", "regs", "weights", "FLOP/call", "cycles/call", "FLOPs/cyc", "%peak", "max_err", "");

    for (int t = 0; t < S1_NUKRS; t++) {
        const s1_ukr_entry_t &k = S1_UKRS[t];
        /* correctness on this data */
        memset(o, 0, KBMAX * o_kb * sizeof(float));
        k.fn(Cb, in, in_plane, in_row, w, w_kb, b, o, o_kb, fs);
        s1_ukr_reference(k.KB, k.OW, Cb, in, in_plane, in_row, w, w_kb, b, ref, o_kb, fs);
        double err = 0, mx = 0;
        for (int kb = 0; kb < k.KB; kb++)
            for (int i = 0; i < k.OW * CB; i++) {
                const size_t ix = kb * o_kb + i;
                err = std::max(err, (double)std::fabs(o[ix] - ref[ix]));
                mx = std::max(mx, (double)std::fabs(ref[ix]));
            }
        const double rel = err / std::max(mx, 1e-30);

        const double flops = 2.0 * k.KB * CB * k.OW * (double)C * 9;
        const long n = reps > 0 ? reps : std::max(2000L, (long)(4e8 / flops)); /* ~0.1 s per trial */
        for (long i = 0; i < n / 4; i++) k.fn(Cb, in, in_plane, in_row, w, w_kb, b, o, o_kb, fs); /* warm-up */
        std::vector<double> cyc;
        for (int trial = 0; trial < 9; trial++) {
            uint64_t c0 = pfd >= 0 ? perf_read(pfd) : rdtsc();
            for (long i = 0; i < n; i++) k.fn(Cb, in, in_plane, in_row, w, w_kb, b, o, o_kb, fs);
            uint64_t c1 = pfd >= 0 ? perf_read(pfd) : rdtsc();
            cyc.push_back((double)(c1 - c0) / n);
        }
        std::sort(cyc.begin(), cyc.end());
        const double best = cyc[0], med = cyc[cyc.size() / 2];
        const double fpc = best > 0 ? flops / best : 0, pct = 100 * fpc / peak;
        const double wkb = k.KB * (double)C * 288 / 1024; /* KB output blocks x C x 9 x 8 floats */
        const bool pass = rel <= 1e-4;
        printf("%-8s %4d %5d %6.1fKB %9.0f %11.1f %10.2f %6.1f%% %9.1e  %s%s\n", k.name, k.KB * k.OW, k.regs, wkb, flops, best,
               fpc, pct, rel, pass ? "" : "FAIL  ", k.note);
        if (f)
            fprintf(f, "%s,%d,%d,%d,%d,%d,%d,%.1f,%s,%.0f,%.2f,%.2f,%.3f,%.2f,%.2e,%d,%s\n", k.name, k.KB, k.OW, k.KB * k.OW, k.regs, C, fs,
                    wkb, src, flops, best, med, fpc, pct, rel, pass ? 1 : 0, k.note);
    }
    printf("weights column: L1D = 32 KB, L2 = 256 KB. Above 32 KB the weights stream from L2 every call.\n");
    if (f) fclose(f);
    return 0;
}
