#ifndef KERNEL_H
#define KERNEL_H

#include <stddef.h>

/*
  Same layouts as the team repo (fast-yolov8-broadwell/include/common.h),
  so a kernel written here drops into src/conv3x3_s2.cpp unchanged.

    Activations: NCHW8c with a zero halo
      logical [N][C][H][W]
      stored  [N][C/8][H+2*halo][W+2*halo][8]   (8 channels = one YMM)

    Weights: [K/8][C/8][3][3][8 ic][8 oc]
      for fixed (kb, cb, r, s, ic) the 8 output-channel weights are one YMM load

*/

#define CB 8

typedef struct {
  int N, C, H, W;   /* input tensor (logical) */
  int K, R, S;      /* output channels, kernel height, kernel width */
  int stride, pad;  /* conv stride and zero padding */
  int OH, OW;       /* output spatial size */
  int in_halo;      /* halo stored around the input planes (== pad) */
  int out_halo;     /* halo the kernel must leave (zero) around the output planes */
  int fuse_silu;    /* 1: out = SiLU(conv), 0: out = conv (the kernels do conv only) */
} conv_desc_t;

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

/* offset of element (n, cb, y, x, lane 0) where y, x are PADDED coordinates */
static inline size_t blk_idx(int Cb, int Hp, int Wp, int n, int cb, int y, int x) {
  return ((((size_t)n * Cb + cb) * Hp + y) * Wp + x) * CB;
}

/* offset of weight (kb, cb, r, s, ic, oc 0) */
static inline size_t wpk_idx(int Cb, int kb, int cb, int r, int s, int ic) {
  return (((((size_t)kb * Cb + cb) * 3 + r) * 3 + s) * CB + ic) * CB;
}

void kernel
(
  const conv_desc_t* d,
  const float*       in,    /* NCHW8c, halo d->in_halo           */
  const float*       w,     /* [K/8][C/8][3][3][8][8]            */
  float*             out    /* NCHW8c, halo d->out_halo, zeroed  */
);

void kernel_v2
(
  const conv_desc_t* d,
  const float*       in,
  const float*       w,
  float*             out
);

void kernel_v3
(
  const conv_desc_t* d,
  const float*       in,
  const float*       w,
  float*             out
);

void kernel_v4
(
  const conv_desc_t* d,
  const float*       in,
  const float*       w,
  float*             out
);

#endif
