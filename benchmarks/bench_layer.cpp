/*
 * bench_layer.cpp  --  check one kernel on one real yolov8n layer, then time it.
 *
 *   ./build/bench_layer data/tensors/b1/model.2.m.0.cv1 s1_v1 [-r reps] [-t threads]
 *                       [--no-silu] [--in-halo H] [--out-halo H] [--cold] [--csv]
 *
 * Default halo is 1 on both input and output (the team layout contract), so a
 * 1x1 kernel sees the same padded buffers a 3x3 kernel does.
 *
 * A layer directory (written by scripts/dump_layers.py) holds:
 *   meta.txt     "N C H W K R S stride pad OH OW has_silu"
 *   input.bin    NCHW float32, the layer's real input activations
 *   weight.bin   KCRS float32 (BatchNorm already folded in by the ONNX export)
 *   bias.bin     K float32
 *   conv_out.bin NCHW float32, the baseline's Conv output (before SiLU)
 *   output.bin   NCHW float32, the baseline's output after SiLU (= conv_out if no SiLU)
 *
 * FLOPs/cycle uses core cycles (perf_event) for 1 thread when the kernel
 * allows it, otherwise TSC cycles (wall time at the base clock). The output says which.
 * Peak defaults to 32 FLOPs/cycle/core (AVX2); export PEAK_FPC=<n> to change it.
 *
 * Everything outside the kernel call (file I/O, layout conversion, weight
 * packing) is outside the timed region.
 */
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif
#include "common.h"
#include "kernels.h"

using namespace fastyolo;

/* pack = this kernel's weight packer; nullptr means the team layout (pack_weights). */
typedef struct { const char *name; conv_fn_t fn; pack_fn_t pack; } kernel_entry_t;

static const kernel_entry_t KERNELS[] = {
    {"generic", conv_blk_generic, nullptr},
    {"s1_v0", conv3x3_s1_v0, nullptr},
    {"s1_v1", conv3x3_s1_v1, nullptr},
    {"s1_v2", conv3x3_s1_v2, nullptr},
    {"s1_v3", conv3x3_s1_v3, nullptr},
    {"s1_v4", conv3x3_s1_v4, nullptr},
    {"s1_v4b", conv3x3_s1_v4b, nullptr},
    {"s1_v5", conv3x3_s1_v5, pack_weights_s1_kb2},
    {"s1_v6", conv3x3_s1_v6, nullptr},
    {"s1_v6b", conv3x3_s1_v6b, nullptr},
    {"s1_v7", conv3x3_s1_v7, nullptr},
    {"s1_v7b", conv3x3_s1_v7b, nullptr},
    {"s1_v7mt", conv3x3_s1_v7mt, nullptr},
    {"s1_v8", conv3x3_s1_v8, nullptr},
    {"s1_v2a", conv3x3_s1_v2a, nullptr},
    {"s1_v8a", conv3x3_s1_v8a, nullptr},
    {"s1_v8mt", conv3x3_s1_v8mt, nullptr},
    {"s1_v8b", conv3x3_s1_v8b, nullptr},
    {"s1_v9", conv3x3_s1_v9, nullptr},
    {"s1_v9mt", conv3x3_s1_v9mt, nullptr},
    {"s1_t1x1", conv3x3_s1_t1x1, nullptr}, {"s1_t1x2", conv3x3_s1_t1x2, nullptr}, {"s1_t1x4", conv3x3_s1_t1x4, nullptr},
    {"s1_t1x5", conv3x3_s1_t1x5, nullptr}, {"s1_t1x10", conv3x3_s1_t1x10, nullptr},
    {"s1_t2x1", conv3x3_s1_t2x1, nullptr}, {"s1_t2x2", conv3x3_s1_t2x2, nullptr}, {"s1_t2x4", conv3x3_s1_t2x4, nullptr},
    {"s1_t2x5", conv3x3_s1_t2x5, nullptr},
    {"s2_v1", conv3x3_s2_v1, nullptr},
    {"1x1_v1", conv1x1_v1, nullptr},
    /* add your kernel here */
};
#define NKERNELS (int)(sizeof(KERNELS) / sizeof(KERNELS[0]))

static float *read_bin(const char *dir, const char *file, size_t nfloats) {
    char path[1024];
    snprintf(path, sizeof path, "%s/%s", dir, file);
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    float *p = alloc_aligned(nfloats);
    size_t got = fread(p, sizeof(float), nfloats, f);
    if (got != nfloats || fgetc(f) != EOF) {
        fprintf(stderr, "%s: expected %zu floats\n", path, nfloats);
        exit(1);
    }
    fclose(f);
    return p;
}

/* Core clock cycles of THIS thread via perf_event (-1 if not permitted).
 * rdtsc ticks at the fixed base clock, but the core turbos higher,
 * so TSC "cycles" undercount real cycles and make you
 * look faster than peak. For single-thread runs we report core cycles. */
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

/* Measure the TSC rate once (should match the "@ x.xxGHz" in the CPU name). */
static double tsc_ghz(void) {
    struct timespec a{}, b{};
    clock_gettime(CLOCK_MONOTONIC, &a);
    uint64_t t0 = rdtsc();
    do clock_gettime(CLOCK_MONOTONIC, &b);
    while ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec) < 50e6);
    uint64_t t1 = rdtsc();
    return (t1 - t0) / ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec));
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return (x > y) - (x < y);
}

static void usage(void) {
    fprintf(stderr, "usage: bench_layer <layer_dir> <kernel|ref> [-r reps] [-t threads] "
                    "[--no-silu] [--in-halo H] [--out-halo H] [--cold] [--csv]\nkernels: ref");
    for (int i = 0; i < NKERNELS; i++) fprintf(stderr, " %s", KERNELS[i].name);
    fprintf(stderr, "\n");
    exit(1);
}

int main(int argc, char **argv) {
    if (argc < 3) usage();
    const char *dir = argv[1], *kname = argv[2];
    int reps = 20, threads = 1, no_silu = 0, in_halo = 1, out_halo = 1, cold = 0, csv = 0;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "-r") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-t") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--in-halo") && i + 1 < argc) in_halo = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--out-halo") && i + 1 < argc) out_halo = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-silu")) no_silu = 1;
        else if (!strcmp(argv[i], "--cold")) cold = 1;
        else if (!strcmp(argv[i], "--csv")) csv = 1;
        else usage();
    }

    conv_fn_t fn = NULL;
    pack_fn_t pack = NULL;
    int is_ref = !strcmp(kname, "ref");
    for (int i = 0; i < NKERNELS; i++)
        if (!strcmp(kname, KERNELS[i].name)) { fn = KERNELS[i].fn; pack = KERNELS[i].pack; }
    if (!fn && !is_ref) usage();

#ifdef _OPENMP
    omp_set_num_threads(threads);
#else
    threads = 1;
#endif

    /* ---- load the layer ---- */
    char path[1024];
    snprintf(path, sizeof path, "%s/meta.txt", dir);
    FILE *mf = fopen(path, "r");
    if (!mf) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    conv_desc_t d{};
    int has_silu = 0;
    if (fscanf(mf, "%d %d %d %d %d %d %d %d %d %d %d %d", &d.N, &d.C, &d.H, &d.W, &d.K, &d.R,
               &d.S, &d.stride, &d.pad, &d.OH, &d.OW, &has_silu) != 12) {
        fprintf(stderr, "bad meta.txt\n");
        return 1;
    }
    fclose(mf);
    d.fuse_silu = has_silu && !no_silu;
    d.in_halo = in_halo < d.pad ? d.pad : in_halo; /* team contract: halo 1 on every activation */
    d.out_halo = out_halo;

    size_t n_in = (size_t)d.N * d.C * d.H * d.W, n_out = (size_t)d.N * d.K * d.OH * d.OW;
    float *in = read_bin(dir, "input.bin", n_in);
    float *w = read_bin(dir, "weight.bin", (size_t)d.K * d.C * d.R * d.S);
    float *bias = read_bin(dir, "bias.bin", d.K);
    float *expect = read_bin(dir, d.fuse_silu ? "output.bin" : "conv_out.bin", n_out);
    float *bias_pad = alloc_aligned((size_t)ceil_div(d.K, CB) * CB); /* bias padded to a full block */
    memcpy(bias_pad, bias, d.K * sizeof(float));

    float *in_blk = alloc_aligned(blk_size(d.N, d.C, d.H, d.W, d.in_halo));
    float *w_pack = alloc_aligned(wpack_size(d.K, d.C, d.R, d.S));
    float *out_blk = alloc_aligned(blk_size(d.N, d.K, d.OH, d.OW, d.out_halo));
    float *got = alloc_aligned(n_out);
    nchw_to_blk(in, d.N, d.C, d.H, d.W, d.in_halo, in_blk);
    (pack ? pack : pack_weights)(w, d.K, d.C, d.R, d.S, w_pack);

#define RUN() do { if (is_ref) conv_ref_nchw(&d, in, w, bias, got); \
                   else fn(&d, in_blk, w_pack, bias_pad, out_blk); } while (0)

    /* ---- correctness: |got - expect| / max(|expect|, 1) must be <= 1e-4 ---- */
    RUN();
    if (!is_ref) blk_to_nchw(out_blk, d.N, d.K, d.OH, d.OW, d.out_halo, got);
    double max_err = 0;
    size_t worst = 0;
    for (size_t i = 0; i < n_out; i++) {
        double e = fabs((double)got[i] - expect[i]) / fmax(fabs((double)expect[i]), 1.0);
        if (!(e <= max_err)) { max_err = e; worst = i; } /* also catches NaN */
    }
    int pass = max_err <= 1e-4;
    if (!csv)
        printf("%s  %s  N=%d C=%d H=%d W=%d K=%d %dx%d s%d  silu=%d  max_err=%.2e (at %zu: got %g want %g)  %s\n",
               dir, kname, d.N, d.C, d.H, d.W, d.K, d.R, d.S, d.stride, d.fuse_silu, max_err, worst,
               got[worst], expect[worst], pass ? "PASS" : "FAIL");

    /* ---- timing ---- */
    const double flops = 2.0 * d.N * d.K * d.OH * d.OW * d.C * d.R * d.S;
    size_t flush_n = cold ? (64u << 20) / sizeof(float) : 0; /* 64 MB > 25 MB L3 */
    float *flush = cold ? alloc_aligned(flush_n) : NULL;
    uint64_t *tsc = (uint64_t *)malloc(sizeof(uint64_t) * reps), *core = (uint64_t *)malloc(sizeof(uint64_t) * reps);
    int pfd = threads == 1 ? perf_open_cycles() : -1;
    RUN(); /* warm-up */
    for (int r = 0; r < reps; r++) {
        if (cold) for (size_t i = 0; i < flush_n; i += 16) flush[i] += 1.0f;
        uint64_t p0 = perf_read(pfd), t0 = rdtsc();
        RUN();
        uint64_t t1 = rdtsc(), p1 = perf_read(pfd);
        tsc[r] = t1 - t0;
        core[r] = p1 - p0;
    }
    qsort(tsc, reps, sizeof(uint64_t), cmp_u64);
    qsort(core, reps, sizeof(uint64_t), cmp_u64);
    int use_core = pfd >= 0 && core[0] > 0;
    double best = use_core ? (double)core[0] : (double)tsc[0];
    double med = use_core ? (double)core[reps / 2] : (double)tsc[reps / 2];
    double ghz = tsc_ghz(), secs = (double)tsc[0] / (ghz * 1e9);
    const char *pk = getenv("PEAK_FPC"); /* peak FLOPs/cycle/core; default 32 = AVX2 with 2 FMA units */
    double peak_core = pk ? atof(pk) : PEAK_FLOPS_PER_CYCLE_PER_CORE;
    double fpc = flops / best, pct = 100.0 * fpc / (peak_core * threads);
    double gflops = flops / secs / 1e9;
    const char *src = use_core ? "core" : "tsc";

    const char *layer = strrchr(dir, '/') ? strrchr(dir, '/') + 1 : dir;
    if (csv)
        printf("%s,%s,%d,%d,%d,%d,%d,%d,%d,%d,%d,%.0f,%.0f,%.0f,%s,%.3f,%.2f,%.2f,%.3e,%d,%.1f\n", layer, kname,
               d.N, d.C, d.H, d.W, d.K, d.R, d.S, d.stride, threads, flops, best, med, src, fpc, pct, gflops,
               max_err, pass, secs * 1e6);
    else
        printf("  threads=%d  %.3g FLOPs  best %.0f %s-cycles (median %.0f)  time %.1f us  ->  %.2f FLOPs/cycle"
               "  = %.1f%% of %d-core peak  (%.1f GFLOP/s, TSC %.2f GHz)\n",
               threads, flops, best, src, med, secs * 1e6, fpc, pct, threads, gflops, ghz);
    return pass ? 0 : 2;
}
