#define _POSIX_C_SOURCE 200112L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "immintrin.h"

#include "kernel.h"

#define MAX_FREQ 3.4
#define BASE_FREQ 2.4
#define PEAK_FLOPS_PER_CYCLE 32.0  // 2 FMA ports x 8 fp32 lanes x 2 flops

/* kernel under test; pick another with  make KERNEL=kernel_v3 */
#ifndef KERNEL_FN
#define KERNEL_FN kernel_v2
#endif

#define ATOL 1e-4
#define RTOL 1e-4

//timing routine for reading the time stamp counter
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__ ("lfence\n\trdtsc" : "=a"(lo), "=d"(hi) :: "memory");
  return ( (unsigned long long)lo)|( ((unsigned long long)hi)<<32 );
}

// the 7 YOLOv8n 3x3 stride-2 layers (batch 1, pad 1)
typedef struct { const char* name; int C, K, H, W; } layer_t;

static const layer_t LAYERS[] = {
  { "model.0",    3,  16, 640, 640 },  // stem: 3 real channels padded to one block
  { "model.1",   16,  32, 320, 320 },
  { "model.3",   32,  64, 160, 160 },
  { "model.5",   64, 128,  80,  80 },
  { "model.7",  128, 256,  40,  40 },
  { "model.16",  64,  64,  80,  80 },  // neck (PAN)
  { "model.19", 128, 128,  40,  40 },  // neck (PAN)
};
#define N_LAYERS ((int)(sizeof(LAYERS) / sizeof(LAYERS[0])))

static float* alloc_aligned(size_t n) {
  void* p;
  if (posix_memalign(&p, 64, n * sizeof(float))) { perror("posix_memalign"); exit(1); }
  memset(p, 0, n * sizeof(float));
  return (float*)p;
}

static float frand(void) { return 2.0f * ((float)rand() / (float)RAND_MAX) - 1.0f; }

static inline float silu(float x) { return x / (1.0f + expf(-x)); }

// plain NCHW / OIHW, explicit bounds checks instead of a halo
void naive
(
 const conv_desc_t* d,
 const float*       in,
 const float*       w,
 float*             out
 ){

  for (int n = 0; n != d->N; ++n)
    for (int k = 0; k != d->K; ++k)
      for (int oh = 0; oh != d->OH; ++oh)
        for (int ow = 0; ow != d->OW; ++ow)
	  {
	    double acc = 0.0;
	    for (int c = 0; c != d->C; ++c)
	      for (int r = 0; r != 3; ++r)
		for (int s = 0; s != 3; ++s)
		  {
		    int ih = oh * d->stride - d->pad + r;
		    int iw = ow * d->stride - d->pad + s;
		    if (ih < 0 || ih >= d->H || iw < 0 || iw >= d->W) continue;
		    acc += (double)in[(((size_t)n * d->C + c) * d->H + ih) * d->W + iw] *
		           (double)w[(((size_t)k * d->C + c) * 3 + r) * 3 + s];
		  }
	    float y = (float)acc;
	    out[(((size_t)n * d->K + k) * d->OH + oh) * d->OW + ow] = d->fuse_silu ? silu(y) : y;
	  }
}

// NCHW -> NCHW8c with halo (padding channels and halo stay zero)
static void nchw_to_blk(const float* src, int N, int C, int H, int W, int halo, float* dst) {
  int Cb = ceil_div(C, CB), Hp = H + 2 * halo, Wp = W + 2 * halo;
  for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
      for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
          dst[blk_idx(Cb, Hp, Wp, n, c / CB, y + halo, x + halo) + c % CB] =
            src[(((size_t)n * C + c) * H + y) * W + x];
}

// OIHW -> [K/8][C/8][3][3][8 ic][8 oc]
static void pack_weights(const float* oihw, int K, int C, float* dst) {
  int Cb = ceil_div(C, CB);
  for (int k = 0; k < K; ++k)
    for (int c = 0; c < C; ++c)
      for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s)
          dst[wpk_idx(Cb, k / CB, c / CB, r, s, c % CB) + k % CB] =
            oihw[(((size_t)k * C + c) * 3 + r) * 3 + s];
}

// compare blocked output against NCHW reference; also require the halo to stay zero
static int check(const conv_desc_t* d, const float* out_blk, const float* ref, double* max_err) {
  int Kb = ceil_div(d->K, CB), OHp = d->OH + 2 * d->out_halo, OWp = d->OW + 2 * d->out_halo;
  int correct = 1;
  *max_err = 0.0;
  for (int k = 0; k < Kb * CB; ++k)
    for (int y = 0; y < OHp; ++y)
      for (int x = 0; x < OWp; ++x) {
        float got = out_blk[blk_idx(Kb, OHp, OWp, 0, k / CB, y, x) + k % CB];
        int oy = y - d->out_halo, ox = x - d->out_halo;
        int interior = k < d->K && oy >= 0 && oy < d->OH && ox >= 0 && ox < d->OW;
        float want = interior ? ref[((size_t)k * d->OH + oy) * d->OW + ox] : 0.0f;
        double err = fabs((double)got - (double)want);
        if (!(err <= ATOL + RTOL * fabs((double)want))) correct = 0;  // also catches NaN
        if (err > *max_err || err != err) *max_err = err;
      }
  return correct;
}

// build, time and check one conv; prints one line
static void run(const layer_t* L){
    conv_desc_t d;
    d.N = 1; d.C = L->C; d.H = L->H; d.W = L->W;
    d.K = L->K; d.R = 3; d.S = 3;
    d.stride = 2; d.pad = 1;
    d.OH = (d.H + 2 * d.pad - 3) / d.stride + 1;
    d.OW = (d.W + 2 * d.pad - 3) / d.stride + 1;
    d.in_halo = 1; d.out_halo = 1;
    d.fuse_silu = 0;  // kernels compute the conv only: no bias, no SiLU

    int Cb = ceil_div(d.C, CB), Kb = ceil_div(d.K, CB);
    size_t in_n   = (size_t)d.C * d.H * d.W;
    size_t w_n    = (size_t)d.K * d.C * 9;
    size_t ref_n  = (size_t)d.K * d.OH * d.OW;
    size_t inb_n  = (size_t)Cb * (d.H + 2) * (d.W + 2) * CB;
    size_t wpk_n  = (size_t)Kb * Cb * 9 * CB * CB;
    size_t outb_n = (size_t)Kb * (d.OH + 2) * (d.OW + 2) * CB;

    //create memory aligned buffers
    float* in     = alloc_aligned(in_n);
    float* w      = alloc_aligned(w_n);
    float* ref    = alloc_aligned(ref_n);
    float* in_blk = alloc_aligned(inb_n);
    float* w_pack = alloc_aligned(wpk_n);
    float* out    = alloc_aligned(outb_n);

    //initialize input and weights (weights scaled so outputs stay O(1))
    float wscale = 1.0f / sqrtf((float)(d.C * 9));
    for (size_t i = 0; i != in_n; ++i) in[i] = frand();
    for (size_t i = 0; i != w_n; ++i)  w[i]  = frand() * wscale;

    naive(&d, in, w, ref);
    nchw_to_blk(in, 1, d.C, d.H, d.W, d.in_halo, in_blk);
    pack_weights(w, d.K, d.C, w_pack);

    // warm-up run (caches, page faults), then a fresh zeroed output for the timed run
    KERNEL_FN(&d, in_blk, w_pack, out);
    memset(out, 0, outb_n * sizeof(float));

    unsigned long long t0 = rdtsc();

    KERNEL_FN(&d, in_blk, w_pack, out);

    unsigned long long t1 = rdtsc();

    double max_err;
    int correct = check(&d, out, ref, &max_err);

    // useful flops only: the stem's 5 zero-padded channels are not counted
    double flops  = 2.0 * d.K * d.C * 9.0 * d.OH * d.OW;
    double cycles = (double)(t1 - t0) * MAX_FREQ / BASE_FREQ;  // rdtsc ticks at base clock
    double fpc    = flops / cycles;

    printf("%s\t %d\t %d\t %d\t %d\t %.0lf\t %lf\t %.1lf\t %.2e\t %d\n",
           L->name, d.C, d.K, d.H, d.W, cycles, fpc, 100.0 * fpc / PEAK_FLOPS_PER_CYCLE,
           max_err, correct);

    free(in); free(w); free(ref);
    free(in_blk); free(w_pack); free(out);
}

int main(int argc, char** argv){

  // optional: ./kernel_driver.x model.3   runs only that layer
  //           ./kernel_driver.x sweep     sweeps the reduction length k = C*3*3
  const char* only = argc > 1 ? argv[1] : NULL;

  srand(645);

  if (only && strcmp(only, "sweep") == 0){
    // like the hw2 matmul's k sweep: grow k = C*3*3 with everything else fixed.
    // K = 48 (6 oc blocks) and OW = 60 divide evenly by the 2x5, 2x6 and 3x4
    // blocks, so no kernel runs a tail and only the register block differs.
    // C = 2..7 covers small k (18..63): the kernels skip the zero padding
    // channels, so those points do only the real work.
    char name[16];
    for (int C = 2; C <= 256; C += C < 8 ? 1 : 8){
      snprintf(name, sizeof(name), "%d", C * 9);   // name = k
      layer_t L = { name, C, 48, 24, 120 };
      run(&L);
    }
    return 0;
  }

  for (int li = 0; li != N_LAYERS; ++li){
    if (only && strcmp(only, LAYERS[li].name) != 0) continue;
    run(&LAYERS[li]);
  }

  return 0;
}
