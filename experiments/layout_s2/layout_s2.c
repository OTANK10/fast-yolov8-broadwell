/*
 * layout_s2.c  --  does NCHW8c actually help the stride-2 kernel?
 *
 * O'Malley's kernel_v2 (2x5 tile: 2 output-channel blocks x 5 pixels, broadcast input,
 * packed weights) addresses the input only through a per-step offset table aoff[p] and a
 * pixel stride. So the SAME microkernel can read either input layout; only those two
 * change. Weights (packed [K/8][C/8][3][3][8][8]) and the output (NCHW8c) are identical,
 * so the timing difference is the input activation layout alone.
 *
 *   A  NCHW8c + halo  [C/8][H+2][W+2][8]   pixel stride 2*8 floats, step p = (cb,r,s,ic)
 *   B  NCHW   + halo  [C][H+2][W+2]        pixel stride 2 floats,   same step order
 *   C  NCHW8c, NO halo [C/8][H][W][8]      same fast 2x5 for tiles whose 3x3 windows stay inside
 *      the image; tiles touching an edge use a checked path that skips out-of-image taps
 *      (what every kernel would need without the stored zero border)
 *
 * The microkernel below is a copy of conv_2x5 from omalley-kernel-code/kernel.c with the
 * pixel stride made a parameter. Build: gcc -O3 -mavx2 -mfma -std=c99 layout_s2.c -lm
 * Run on ece028 pinned: taskset -c 0 ./layout_s2.x [reps]
 * Cycles = rdtsc x 3.4 / 2.4 (Assignment 2 convention), best of `reps` runs per layer.
 */
#define _POSIX_C_SOURCE 200112L
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stddef.h>
#include <immintrin.h>
#include "kernel.h"

static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("lfence\n\trdtsc" : "=a"(lo), "=d"(hi)::"memory");
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

typedef struct { const char *name; int C, K, H, W; } layer_t;
static const layer_t LAYERS[] = {
  {"model.0", 3, 16, 640, 640}, {"model.1", 16, 32, 320, 320}, {"model.3", 32, 64, 160, 160},
  {"model.5", 64, 128, 80, 80}, {"model.7", 128, 256, 40, 40}, {"model.16", 64, 64, 80, 80},
  {"model.19", 128, 128, 40, 40},
};
#define NL ((int)(sizeof(LAYERS) / sizeof(LAYERS[0])))

/* conv_2x5 from O'Malley's kernel.c, pixel stride `ps` floats instead of 2*CB */
static inline __attribute__((always_inline)) void conv_2x5_ps(int k, const int *aoff, const float *a, int ps,
                                                              const float *b, size_t ldb, float *c, size_t ldc) {
  const float *b0 = b, *b1 = b + ldb;
  __m256 c00 = _mm256_setzero_ps(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00, c30 = c00, c31 = c00, c40 = c00, c41 = c00;
  for (int p = 0; p != k; ++p) {
    const __m256 vb0 = _mm256_load_ps(&b0[p * CB]), vb1 = _mm256_load_ps(&b1[p * CB]);
    const float *ap = &a[aoff[p]];
    __m256 va;
    va = _mm256_broadcast_ss(&ap[0 * ps]); c00 = _mm256_fmadd_ps(va, vb0, c00); c01 = _mm256_fmadd_ps(va, vb1, c01);
    va = _mm256_broadcast_ss(&ap[1 * ps]); c10 = _mm256_fmadd_ps(va, vb0, c10); c11 = _mm256_fmadd_ps(va, vb1, c11);
    va = _mm256_broadcast_ss(&ap[2 * ps]); c20 = _mm256_fmadd_ps(va, vb0, c20); c21 = _mm256_fmadd_ps(va, vb1, c21);
    va = _mm256_broadcast_ss(&ap[3 * ps]); c30 = _mm256_fmadd_ps(va, vb0, c30); c31 = _mm256_fmadd_ps(va, vb1, c31);
    va = _mm256_broadcast_ss(&ap[4 * ps]); c40 = _mm256_fmadd_ps(va, vb0, c40); c41 = _mm256_fmadd_ps(va, vb1, c41);
  }
  _mm256_storeu_ps(&c[0 * CB], c00); _mm256_storeu_ps(&c[0 * CB + ldc], c01);
  _mm256_storeu_ps(&c[1 * CB], c10); _mm256_storeu_ps(&c[1 * CB + ldc], c11);
  _mm256_storeu_ps(&c[2 * CB], c20); _mm256_storeu_ps(&c[2 * CB + ldc], c21);
  _mm256_storeu_ps(&c[3 * CB], c30); _mm256_storeu_ps(&c[3 * CB + ldc], c31);
  _mm256_storeu_ps(&c[4 * CB], c40); _mm256_storeu_ps(&c[4 * CB + ldc], c41);
}
/* scalar leftovers (OW % 5 and odd Kb): same math, one pixel x one oc block */
static void tail_1x1(int k, const int *aoff, const float *a, const float *b, float *c) {
  __m256 acc = _mm256_setzero_ps();
  for (int p = 0; p != k; ++p) acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&a[aoff[p]]), _mm256_load_ps(&b[p * CB]), acc);
  _mm256_storeu_ps(c, acc);
}

static inline int real_ch(int C, int cb) { return C - cb * CB < CB ? C - cb * CB : CB; }

/* layout 0 = NCHW8c, 1 = NCHW. Step order p = (cb, r, s, ic), real channels only, as in kernel.c */
static int build_aoff(int layout, int C, int Hp, int Wp, int *aoff) {
  const int Cb = ceil_div(C, CB);
  int p = 0;
  for (int cb = 0; cb < Cb; ++cb)
    for (int r = 0; r < 3; ++r)
      for (int s = 0; s < 3; ++s)
        for (int ic = 0; ic < real_ch(C, cb); ++ic)
          aoff[p++] = layout == 0 ? (int)blk_idx(Cb, Hp, Wp, 0, cb, r, s) + ic
                                  : ((cb * CB + ic) * Hp + r) * Wp + s;
  return p;
}

static void conv(int layout, const conv_desc_t *d, const float *in, const float *wk, float *out) {
  const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB);
  const int Hp = d->H + 2, Wp = d->W + 2, OHp = d->OH + 2, OWp = d->OW + 2;
  int aoff[Cb * 9 * CB];
  const int k = build_aoff(layout, d->C, Hp, Wp, aoff);
  const size_t ldb = (size_t)k * CB, ldc = (size_t)OHp * OWp * CB;
  const int ps = layout == 0 ? 2 * CB : 2;
  for (int kb = 0; kb < Kb; kb += 2)
    for (int oh = 0; oh < d->OH; ++oh) {
      const float *arow = in + (layout == 0 ? blk_idx(Cb, Hp, Wp, 0, 0, 2 * oh, 0) : (size_t)2 * oh * Wp);
      float *crow = out + blk_idx(Kb, OHp, OWp, 0, kb, oh + 1, 1);
      int ow = 0;
      if (Kb - kb >= 2)
        for (; ow + 5 <= d->OW; ow += 5)
          conv_2x5_ps(k, aoff, arow + (size_t)2 * ow * (layout == 0 ? CB : 1), ps, wk + kb * ldb, ldb, crow + (size_t)ow * CB, ldc);
      for (; ow < d->OW; ++ow)
        for (int j = 0; j < (Kb - kb >= 2 ? 2 : 1); ++j)
          tail_1x1(k, aoff, arow + (size_t)2 * ow * (layout == 0 ? CB : 1), wk + (kb + j) * ldb, crow + (size_t)ow * CB + j * ldc);
      if (Kb - kb < 2) /* odd Kb: the 2x5 loop above was skipped, so the tail did every pixel */;
    }
}

/* C: no halo. Tiles fully inside use the fast microkernel; edge tiles check every tap. */
static void conv_nohalo(const conv_desc_t *d, const float *in, const float *wk, float *out) {
  const int Cb = ceil_div(d->C, CB), Kb = ceil_div(d->K, CB), H = d->H, W = d->W;
  const int OHp = d->OH + 2, OWp = d->OW + 2;
  int aoff[Cb * 9 * CB], pr[Cb * 9 * CB], psx[Cb * 9 * CB];
  int k = 0;
  for (int cb = 0; cb < Cb; ++cb)
    for (int r = 0; r < 3; ++r)
      for (int s = 0; s < 3; ++s)
        for (int ic = 0; ic < real_ch(d->C, cb); ++ic) { aoff[k] = (int)blk_idx(Cb, H, W, 0, cb, r, s) + ic; pr[k] = r; psx[k] = s; ++k; }
  const size_t ldb = (size_t)k * CB, ldc = (size_t)OHp * OWp * CB;
  for (int kb = 0; kb < Kb; kb += 2) {
    const int nk = Kb - kb >= 2 ? 2 : 1;
    for (int oh = 0; oh < d->OH; ++oh) {
      const int y0 = 2 * oh - 1;                       /* window top row, may be -1 */
      float *crow = out + blk_idx(Kb, OHp, OWp, 0, kb, oh + 1, 1);
      const int rows_in = y0 >= 0 && y0 + 2 <= H - 1;
      for (int ow = 0; ow < d->OW; ow += 5) {
        const int n = d->OW - ow < 5 ? d->OW - ow : 5, x0 = 2 * ow - 1;
        if (nk == 2 && n == 5 && rows_in && x0 >= 0 && 2 * (ow + 4) + 1 <= W - 1) {   /* interior: fast path */
          conv_2x5_ps(k, aoff, in + blk_idx(Cb, H, W, 0, 0, y0, x0), 2 * CB, wk + kb * ldb, ldb, crow + (size_t)ow * CB, ldc);
          continue;
        }
        for (int i = 0; i < n; ++i)                    /* edge: check each tap */
          for (int j = 0; j < nk; ++j) {
            const float *b = wk + (kb + j) * ldb;
            __m256 acc = _mm256_setzero_ps();
            const int xi = 2 * (ow + i) - 1;
            for (int p = 0; p != k; ++p) {
              const int yy = y0 + pr[p], xx = xi + psx[p];
              if (yy < 0 || yy >= H || xx < 0 || xx >= W) continue;
              acc = _mm256_fmadd_ps(_mm256_broadcast_ss(&in[aoff[p] + ((ptrdiff_t)y0 * W + xi) * CB]), _mm256_load_ps(&b[p * CB]), acc);
            }
            _mm256_storeu_ps(crow + (size_t)(ow + i) * CB + j * ldc, acc);
          }
      }
    }
  }
}

static float *falloc(size_t n) { void *p; if (posix_memalign(&p, 64, n * sizeof(float))) exit(1); memset(p, 0, n * sizeof(float)); return p; }
static float frand(void) { return 2.0f * rand() / (float)RAND_MAX - 1.0f; }

int main(int argc, char **argv) {
  const int reps = argc > 1 ? atoi(argv[1]) : 5;
  printf("cycles = rdtsc x 3.4/2.4, best of %d; FLOPs/cycle counts real channels only. Ratios = cycles / NCHW8c+halo cycles\n", reps);
  printf("%-9s %4s %4s %4s | %-22s | %-22s | %-22s | %s\n", "layer", "C", "K", "H", "A NCHW8c + halo", "B NCHW + halo", "C NCHW8c, no halo", "check");
  printf("%-9s %4s %4s %4s | %8s %6s %6s | %8s %6s %6s | %8s %6s %6s |\n", "", "", "", "", "F/cyc", "%peak", "", "F/cyc", "%peak", "x A", "F/cyc", "%peak", "x A");
  for (int l = 0; l < NL; ++l) {
    const layer_t *L = &LAYERS[l];
    conv_desc_t d = {1, L->C, L->H, L->W, L->K, 3, 3, 2, 1, (L->H - 1) / 2 + 1, (L->W - 1) / 2 + 1, 1, 1, 0};
    const int Cb = ceil_div(d.C, CB), Kb = ceil_div(d.K, CB), Hp = d.H + 2, Wp = d.W + 2;
    float *x = falloc((size_t)d.C * d.H * d.W), *w = falloc((size_t)d.K * d.C * 9);
    for (size_t i = 0; i < (size_t)d.C * d.H * d.W; i++) x[i] = frand();
    for (size_t i = 0; i < (size_t)d.K * d.C * 9; i++) w[i] = frand() / sqrtf(9.f * d.C);
    /* inputs in both layouts (outside timing) */
    float *in8 = falloc((size_t)Cb * Hp * Wp * CB), *in1 = falloc((size_t)d.C * Hp * Wp);
    for (int c = 0; c < d.C; c++)
      for (int y = 0; y < d.H; y++)
        for (int xx = 0; xx < d.W; xx++) {
          const float v = x[((size_t)c * d.H + y) * d.W + xx];
          in8[blk_idx(Cb, Hp, Wp, 0, c / CB, y + 1, xx + 1) + c % CB] = v;
          in1[((size_t)c * Hp + y + 1) * Wp + xx + 1] = v;
        }
    /* weights packed and with padding channels dropped, as kernel.c's pack_b: k x 8 per oc block */
    int aoff_tmp[Cb * 9 * CB];
    const int k = build_aoff(0, d.C, Hp, Wp, aoff_tmp);
    float *wk = falloc((size_t)Kb * k * CB);
    for (int kb = 0; kb < Kb; kb++) {
      int p = 0;
      for (int cb = 0; cb < Cb; cb++)
        for (int r = 0; r < 3; r++)
          for (int s = 0; s < 3; s++)
            for (int ic = 0; ic < real_ch(d.C, cb); ic++, p++)
              for (int oc = 0; oc < CB; oc++)
                if (kb * CB + oc < d.K) wk[((size_t)kb * k + p) * CB + oc] = w[(((size_t)(kb * CB + oc) * d.C + cb * CB + ic) * 3 + r) * 3 + s];
    }
    const size_t on = (size_t)Kb * (d.OH + 2) * (d.OW + 2) * CB;
    float *o8 = falloc(on), *o1 = falloc(on), *o0 = falloc(on);
    float *in0 = falloc((size_t)Cb * d.H * d.W * CB);
    for (int c = 0; c < d.C; c++)
      for (int y = 0; y < d.H; y++)
        for (int xx = 0; xx < d.W; xx++) in0[blk_idx(Cb, d.H, d.W, 0, c / CB, y, xx) + c % CB] = x[((size_t)c * d.H + y) * d.W + xx];
    double best[3] = {1e300, 1e300, 1e300};
    for (int lay = 0; lay < 3; lay++) {
      for (int r = 0; r <= reps; r++) {               /* r = 0 is the warm-up */
        unsigned long long t0 = rdtsc();
        if (lay == 2) conv_nohalo(&d, in0, wk, o0);
        else conv(lay, &d, lay ? in1 : in8, wk, lay ? o1 : o8);
        unsigned long long t1 = rdtsc();
        if (r && (double)(t1 - t0) < best[lay]) best[lay] = (double)(t1 - t0);
      }
    }
    /* check: both layouts must give the same output, and match a direct sum on a sample */
    double err = 0;
    for (size_t i = 0; i < on; i++) err = fmax(err, fmax(fabs(o8[i] - o1[i]), fabs(o8[i] - o0[i])));
    for (int t = 0; t < 64; t++) {
      const int kk = rand() % d.K, oh = rand() % d.OH, ow = rand() % d.OW;
      double acc = 0;
      for (int c = 0; c < d.C; c++)
        for (int r = 0; r < 3; r++)
          for (int s = 0; s < 3; s++) {
            const int ih = 2 * oh - 1 + r, iw = 2 * ow - 1 + s;
            if (ih >= 0 && ih < d.H && iw >= 0 && iw < d.W) acc += (double)x[((size_t)c * d.H + ih) * d.W + iw] * w[(((size_t)kk * d.C + c) * 3 + r) * 3 + s];
          }
      err = fmax(err, fabs(acc - o8[blk_idx(Kb, d.OH + 2, d.OW + 2, 0, kk / CB, oh + 1, ow + 1) + kk % CB]));
    }
    const double flops = 2.0 * d.K * d.C * 9.0 * d.OH * d.OW;
    double cyc[3], fpc[3];
    for (int lay = 0; lay < 3; lay++) { cyc[lay] = best[lay] * 3.4 / 2.4; fpc[lay] = flops / cyc[lay]; }
    printf("%-9s %4d %4d %4d | %8.2f %5.1f%% %6s | %8.2f %5.1f%% %5.2fx | %8.2f %5.1f%% %5.2fx | %s (err %.1e)\n", L->name, d.C, d.K, d.H,
           fpc[0], 100 * fpc[0] / 32, "", fpc[1], 100 * fpc[1] / 32, cyc[1] / cyc[0], fpc[2], 100 * fpc[2] / 32, cyc[2] / cyc[0],
           err < 1e-3 ? "ok" : "MISMATCH", err);
    free(x); free(w); free(in8); free(in1); free(in0); free(wk); free(o8); free(o1); free(o0);
  }
  return 0;
}
