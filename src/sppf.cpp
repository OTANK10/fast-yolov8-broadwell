/*
 * sppf.cpp  --  SPPF max-pool (model.9): X, P(X), P(P(X)), P(P(P(X))) concatenated
 * along channels, P = 5x5 max-pool, stride 1, pad 2. Owner: Simon.
 *
 * sppf_pool_reference: plain NCHW, scalar, the ground truth (do not optimize).
 * sppf_pool_v1/v2:     NCHW8c (the layout every conv uses), AVX2, batch N.
 *
 * Design (both versions):
 *  - One 8-channel block of one pixel is one YMM register, so one vmaxps does
 *    8 channels at once. There are no FLOPs; the limit is vmaxps, which on
 *    Broadwell issues only on port 1: 1 per cycle.
 *  - A 5x5 max is separable: max over 5 columns (rows pass), then over 5 rows
 *    (columns pass). 2 x 4 = 8 vmaxps per output vector instead of 24.
 *  - Padding must act like -inf, not zero: the input comes after SiLU and can be
 *    negative. Instead of storing a -inf halo we clamp the window to the image
 *    at the 2 edge pixels of each line, which gives exactly the same result.
 *  - All three pools of one channel block run back to back, so the block's
 *    20x20x8 floats (12.8 KB) stay in L1 the whole time.
 *  - Each pool writes straight into its channel slice of the 4C-channel output
 *    buffer that model.9.cv2 reads, so the Concat is free. Slice 0 is a copy of X
 *    (in the full network cv1 would write it there directly).
 *  v1: direct 5-wide max, 4 vmaxps per output per pass (8 per pool).
 *  v5: v4 with rotated row pointers instead of r % 5 (imul shares port 1 with vmaxps).
 *  v4: v3 without the tmp plane (5-row ring), so one pool fits in L1 (see below).
 *  v3: v2's rows pass with straight-line edges, columns pass done row by row (see below).
 *  v2: reuses pair maxes P[j] = max(s[j], s[j+1]); out[i] = max(P[i-2], P[i], s[i+2]),
 *      3 vmaxps per output per pass (6 per pool), 25% fewer port-1 ops.
 */
#include "common.h"
#include "kernels.h"
#include <cstdlib>
#include <cstring>
#if defined(__AVX2__)
#include <immintrin.h>
#endif
#include <algorithm>
#include <cstddef>
#include <limits>
#include <stdexcept>
namespace fastyolo {
namespace {
std::vector<float> maxpool5x5(const std::vector<float>& input, int c, int h, int w) {
    std::vector<float> out(input.size());
    for (int ch = 0; ch < c; ++ch)
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) {
                float value = -std::numeric_limits<float>::infinity();
                for (int ky = -2; ky <= 2; ++ky)
                    for (int kx = -2; kx <= 2; ++kx) {
                        const int iy = y + ky, ix = x + kx;
                        if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                        const auto idx = (static_cast<std::size_t>(ch) * h + iy) * w + ix;
                        value = std::max(value, input[idx]);
                    }
                out[(static_cast<std::size_t>(ch) * h + y) * w + x] = value;
            }
    return out;
}

/* ---- one 8-channel vector: AVX2 register, or a scalar fallback for non-x86 builds ---- */
#if defined(__AVX2__)
typedef __m256 vec;
static inline vec vload(const float *p) { return _mm256_load_ps(p); }
static inline void vstore(float *p, vec v) { _mm256_store_ps(p, v); }
static inline vec vmax(vec a, vec b) { return _mm256_max_ps(a, b); }
#else
struct vec { float v[8]; };
static inline vec vload(const float *p) { vec r; std::memcpy(r.v, p, sizeof r.v); return r; }
static inline void vstore(float *p, vec v) { std::memcpy(p, v.v, sizeof v.v); }
static inline vec vmax(vec a, vec b) { for (int i = 0; i < 8; i++) a.v[i] = std::max(a.v[i], b.v[i]); return a; }
#endif

/* max of s[lo..hi] (vectors `ss` floats apart): the clamped window at a line's edge */
static inline vec vmax_range(const float *s, size_t ss, int lo, int hi) {
    vec m = vload(s + lo * ss);
    for (int j = lo + 1; j <= hi; j++) m = vmax(m, vload(s + j * ss));
    return m;
}

/* 1-D 5-wide max along a line of n vectors: d[i] = max s[i-2 .. i+2], clamped.
 * s and d step ss and ds floats per element (8 along a row, a row stride along a column). */
template <bool PAIR>
static inline void pool5_line(const float *s, size_t ss, float *d, size_t ds, int n) {
    if (n < 5) {
        for (int i = 0; i < n; i++) vstore(d + i * ds, vmax_range(s, ss, std::max(0, i - 2), std::min(n - 1, i + 2)));
        return;
    }
    vstore(d, vmax_range(s, ss, 0, 2));
    vstore(d + ds, vmax_range(s, ss, 0, 3));
    if (!PAIR) {
        for (int i = 2; i < n - 2; i++) {
            const float *p = s + (i - 2) * ss;
            const vec a = vmax(vload(p), vload(p + ss)), b = vmax(vload(p + 2 * ss), vload(p + 3 * ss));
            vstore(d + i * ds, vmax(vmax(a, b), vload(p + 4 * ss)));     /* 4 vmaxps */
        }
    } else {
        /* P[j] = max(s[j], s[j+1]) is shared by outputs j and j+2: keep the last two in registers */
        vec p0 = vmax(vload(s), vload(s + ss));              /* P[0] */
        vec p1 = vmax(vload(s + ss), vload(s + 2 * ss));     /* P[1] */
        for (int i = 2; i < n - 2; i++) {
            const vec si = vload(s + i * ss);
            const vec p2 = vmax(si, vload(s + (i + 1) * ss));                      /* P[i]: 1 vmaxps */
            vstore(d + i * ds, vmax(vmax(p0, p2), vload(s + (i + 2) * ss)));        /* + 2 vmaxps */
            p0 = p1; p1 = p2;
        }
    }
    vstore(d + (n - 2) * ds, vmax_range(s, ss, n - 4, n - 1));
    vstore(d + (n - 1) * ds, vmax_range(s, ss, n - 3, n - 1));
}

/* One 5x5 pool of one channel-block plane: rows pass into tmp ([H][W][8], dense), then
 * columns pass into dst. src/dst point at pixel (0, 0) of the interior, row strides in floats. */
template <bool PAIR>
static void pool5_plane(const float *src, size_t sr, float *dst, size_t dr, float *tmp, int H, int W) {
    for (int y = 0; y < H; y++) pool5_line<PAIR>(src + y * sr, CB, tmp + (size_t)y * W * CB, CB, W);
    for (int x = 0; x < W; x++) pool5_line<PAIR>(tmp + (size_t)x * CB, (size_t)W * CB, dst + (size_t)x * CB, dr, H);
}

/* v3: the same math with less overhead around it. Lines are short (W = H = 20), so in v1/v2
 * the 4 edge outputs of every line (20%) ran a variable-length loop, and the columns pass walked
 * down one column at a time. v3 writes the edges as straight-line code and does the columns pass
 * row by row: output row y = max of rows y-2..y+2, a long contiguous loop over the W pixels, with
 * the edge handled once per row (rows 0, 1, H-2, H-1 just use fewer rows). */
static inline void pool5_row_v3(const float *s, float *d, int n) {   /* along a row, pair reuse */
    auto L = [&](int i) { return vload(s + (size_t)i * CB); };
    auto S = [&](int i, vec v) { vstore(d + (size_t)i * CB, v); };
    if (n < 5) { pool5_line<true>(s, CB, d, CB, n); return; }
    const vec a0 = L(0), a1 = L(1), a2 = L(2), a3 = L(3);
    vec p0 = vmax(a0, a1), p1 = vmax(a1, a2);                /* P[0], P[1] */
    S(0, vmax(p0, a2));                                      /* max s0..s2 */
    S(1, vmax(p0, vmax(a2, a3)));                            /* max s0..s3 */
    for (int i = 2; i < n - 2; i++) {
        const vec p2 = vmax(L(i), L(i + 1));
        S(i, vmax(vmax(p0, p2), L(i + 2)));
        p0 = p1; p1 = p2;
    }
    /* p0 = P[n-4], p1 = P[n-3] */
    const vec b1 = L(n - 1), b2 = L(n - 2);
    S(n - 2, vmax(p0, vmax(b2, b1)));                        /* max s[n-4..n-1] */
    S(n - 1, vmax(p1, b1));                                  /* max s[n-3..n-1] */
}
static void pool5_plane_v3(const float *src, size_t sr, float *dst, size_t dr, float *tmp, int H, int W) {
    const size_t tr = (size_t)W * CB;
    for (int y = 0; y < H; y++) pool5_row_v3(src + y * sr, tmp + y * tr, W);
    for (int y = 0; y < H; y++) {
        const int lo = std::max(0, y - 2), hi = std::min(H - 1, y + 2);
        const float *t = tmp + lo * tr;
        float *o = dst + y * dr;
        if (hi - lo == 4) {
            for (int x = 0; x < W; x++) {
                const float *q = t + (size_t)x * CB;
                const vec a = vmax(vload(q), vload(q + tr)), b = vmax(vload(q + 2 * tr), vload(q + 3 * tr));
                vstore(o + (size_t)x * CB, vmax(vmax(a, b), vload(q + 4 * tr)));
            }
        } else {
            for (int x = 0; x < W; x++) vstore(o + (size_t)x * CB, vmax_range(t + (size_t)x * CB, tr, 0, hi - lo));
        }
    }
}

/* v4: fix what perf showed on ece028 for v2: 36% of cycles stalled on a full store buffer
 * (resource_stalls.sb) plus 5.3 G 4K-aliasing load blocks. The cause is L1 capacity: one pool
 * touches its source plane, the full tmp plane and its destination plane, ~44 KB, more than the
 * 32 KB L1, so the stores to tmp and dst go to L2 and back up the store buffer. v4 never builds
 * the tmp plane: the rows-pass results live in a ring of 5 rows (5 x W x 8 floats = 3.2 KB at
 * W = 20). Row r of the rows pass is computed once, when output row r-2 needs it, and the
 * columns pass reads the 5 ring rows. Working set per pool: source + destination + 3.2 KB. */
static void pool5_plane_v4(const float *src, size_t sr, float *dst, size_t dr, float *ring, int H, int W) {
    const size_t tr = (size_t)W * CB;
    auto R = [&](int r) { return ring + (size_t)(r % 5) * tr; };
    for (int r = 0; r < std::min(2, H); r++) pool5_row_v3(src + r * sr, R(r), W);
    for (int y = 0; y < H; y++) {
        if (y + 2 < H) pool5_row_v3(src + (y + 2) * sr, R(y + 2), W);   /* the row entering the window */
        const int lo = std::max(0, y - 2), hi = std::min(H - 1, y + 2);
        float *o = dst + y * dr;
        if (hi - lo == 4) {
            const float *r0 = R(y - 2), *r1 = R(y - 1), *r2 = R(y), *r3 = R(y + 1), *r4 = R(y + 2);
            for (int x = 0; x < W; x++) {
                const size_t q = (size_t)x * CB;
                const vec a = vmax(vload(r0 + q), vload(r1 + q)), b = vmax(vload(r2 + q), vload(r3 + q));
                vstore(o + q, vmax(vmax(a, b), vload(r4 + q)));
            }
        } else {
            for (int x = 0; x < W; x++) {
                const size_t q = (size_t)x * CB;
                vec m = vload(R(lo) + q);
                for (int r = lo + 1; r <= hi; r++) m = vmax(m, vload(R(r) + q));
                vstore(o + q, m);
            }
        }
    }
}

/* v5: v4 without its integer overhead. perf on ece028 for v4: store-buffer stalls fell from 36% to
 * 18% of cycles and 4K-alias blocks from 5.3 G to 1.7 G, but port 1 ran 29.4 G uops against v2's
 * 22.1 G for about the same 22 G vmaxps: the ring index r % 5 compiles to an imul sequence, and imul
 * runs on port 1, the same port as vmaxps. v5 keeps 5 row pointers and rotates them once per output
 * row, so the per-pixel loops have no index math beyond the pointer increments. */
static void pool5_plane_v5(const float *src, size_t sr, float *dst, size_t dr, float *ring, int H, int W) {
    const size_t tr = (size_t)W * CB;
    /* rp[k] = rows-pass result of row y-2+k (k = 0..4), slots of the ring */
    float *rp[5] = {ring, ring + tr, ring + 2 * tr, ring + 3 * tr, ring + 4 * tr};
    /* prime: rows 0 and 1 go to the slots of y-2+k for y = 0, i.e. k = 2, 3 */
    for (int r = 0; r < std::min(2, H); r++) pool5_row_v3(src + r * sr, rp[2 + r], W);
    for (int y = 0; y < H; y++) {
        if (y + 2 < H) pool5_row_v3(src + (y + 2) * sr, rp[4], W);   /* the row entering the window */
        const int lo = std::max(0, y - 2) - (y - 2), hi = std::min(H - 1, y + 2) - (y - 2);  /* k range */
        float *o = dst + y * dr;
        if (lo == 0 && hi == 4) {
            const float *r0 = rp[0], *r1 = rp[1], *r2 = rp[2], *r3 = rp[3], *r4 = rp[4];
            for (int x = 0; x < W; x++) {
                const size_t q = (size_t)x * CB;
                const vec a = vmax(vload(r0 + q), vload(r1 + q)), b = vmax(vload(r2 + q), vload(r3 + q));
                vstore(o + q, vmax(vmax(a, b), vload(r4 + q)));
            }
        } else {
            for (int x = 0; x < W; x++) {
                const size_t q = (size_t)x * CB;
                vec m = vload(rp[lo] + q);
                for (int k = lo + 1; k <= hi; k++) m = vmax(m, vload(rp[k] + q));
                vstore(o + q, m);
            }
        }
        float *t = rp[0]; rp[0] = rp[1]; rp[1] = rp[2]; rp[2] = rp[3]; rp[3] = rp[4]; rp[4] = t;  /* slide down one row */
    }
}

/* per-thread scratch for the rows pass, allocated once (never inside a timed call after the first) */
static float *scratch(size_t nfloats) {
    static thread_local float *buf = nullptr;
    static thread_local size_t cap = 0;
    if (nfloats > cap) { std::free(buf); buf = alloc_aligned(nfloats); cap = nfloats; }
    return buf;
}

template <int VER>
static void sppf_pool_blk(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo) {
    if (C % CB) throw std::invalid_argument("sppf_pool: C must be a multiple of 8");
    const int Cb = C / CB, Hpi = H + 2 * in_halo, Wpi = W + 2 * in_halo, Hpo = H + 2 * out_halo, Wpo = W + 2 * out_halo;
    const size_t ri = (size_t)Wpi * CB, ro = (size_t)Wpo * CB;
    float *tmp = scratch((size_t)H * W * CB);
    for (int n = 0; n < N; n++)
        for (int cb = 0; cb < Cb; cb++) {
            const float *x = in + blk_idx(Cb, Hpi, Wpi, n, cb, in_halo, in_halo);
            float *o[4];
            for (int k = 0; k < 4; k++) o[k] = out + blk_idx(4 * Cb, Hpo, Wpo, n, k * Cb + cb, out_halo, out_halo);
            for (int y = 0; y < H; y++) std::memcpy(o[0] + y * ro, x + y * ri, (size_t)W * CB * sizeof(float)); /* slice 0: X */
            if (VER == 5) pool5_plane_v5(x, ri, o[1], ro, tmp, H, W); else if (VER == 4) pool5_plane_v4(x, ri, o[1], ro, tmp, H, W); else if (VER == 3) pool5_plane_v3(x, ri, o[1], ro, tmp, H, W); else pool5_plane<VER == 2>(x, ri, o[1], ro, tmp, H, W);       /* P1 = P(X)  */
            if (VER == 5) pool5_plane_v5(o[1], ro, o[2], ro, tmp, H, W); else if (VER == 4) pool5_plane_v4(o[1], ro, o[2], ro, tmp, H, W); else if (VER == 3) pool5_plane_v3(o[1], ro, o[2], ro, tmp, H, W); else pool5_plane<VER == 2>(o[1], ro, o[2], ro, tmp, H, W);    /* P2 = P(P1) */
            if (VER == 5) pool5_plane_v5(o[2], ro, o[3], ro, tmp, H, W); else if (VER == 4) pool5_plane_v4(o[2], ro, o[3], ro, tmp, H, W); else if (VER == 3) pool5_plane_v3(o[2], ro, o[3], ro, tmp, H, W); else pool5_plane<VER == 2>(o[2], ro, o[3], ro, tmp, H, W);    /* P3 = P(P2) */
        }
}

/* ---- isolated microkernels (benchmarks/bench_pool_ukr.cpp): one 5x5 pool of ONE 8-channel
 * plane, src -> dst, nothing else (no other channel blocks, no chaining, no concat copy). */
static void ukr_v1(const float *s, size_t sr, float *d, size_t dr, float *t, int H, int W) { pool5_plane<false>(s, sr, d, dr, t, H, W); }
static void ukr_v2(const float *s, size_t sr, float *d, size_t dr, float *t, int H, int W) { pool5_plane<true>(s, sr, d, dr, t, H, W); }
static void ukr_v3(const float *s, size_t sr, float *d, size_t dr, float *t, int H, int W) { pool5_plane_v3(s, sr, d, dr, t, H, W); }
static void ukr_v4(const float *s, size_t sr, float *d, size_t dr, float *t, int H, int W) { pool5_plane_v4(s, sr, d, dr, t, H, W); }
static void ukr_v5(const float *s, size_t sr, float *d, size_t dr, float *t, int H, int W) { pool5_plane_v5(s, sr, d, dr, t, H, W); }
} // namespace
/* name, vmaxps mode of the rows pass and of the columns pass (see sppf_vmax_per_line), scratch rows, note */
const pool_ukr_entry_t POOL_UKRS[] = {
    {"v1", 0, 0, 0, "direct 5-wide max, rows then columns, tmp plane", ukr_v1},
    {"v2", 1, 1, 0, "shared pair maxes (6 vmaxps/output), tmp plane", ukr_v2},
    {"v3", 2, 0, 0, "v2 rows + straight-line edges, columns pass row by row", ukr_v3},
    {"v4", 2, 0, 5, "v3 with a 5-row ring instead of the tmp plane (r % 5 index)", ukr_v4},
    {"v5", 2, 0, 5, "v4 with rotated row pointers (no imul)", ukr_v5},
};
const int POOL_NUKRS = sizeof(POOL_UKRS) / sizeof(POOL_UKRS[0]);

void sppf_pool_v1(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo) {
    sppf_pool_blk<1>(N, C, H, W, in, in_halo, out, out_halo);
}
void sppf_pool_v2(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo) {
    sppf_pool_blk<2>(N, C, H, W, in, in_halo, out, out_halo);
}
void sppf_pool_v3(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo) {
    sppf_pool_blk<3>(N, C, H, W, in, in_halo, out, out_halo);
}
void sppf_pool_v4(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo) {
    sppf_pool_blk<4>(N, C, H, W, in, in_halo, out, out_halo);
}
void sppf_pool_v5(int N, int C, int H, int W, const float *in, int in_halo, float *out, int out_halo) {
    sppf_pool_blk<5>(N, C, H, W, in, in_halo, out, out_halo);
}
/* vmaxps one pool issues on an n-long line: pair 0 = direct, 1 = v2 pairs, 2 = v3 rows pass */
long sppf_vmax_per_line(int n, int pair) {
    auto rng = [&](int i) { return (long)(std::min(n - 1, i + 2) - std::max(0, i - 2)); };
    if (n < 5) { long t = 0; for (int i = 0; i < n; i++) t += rng(i); return t; }
    long t = rng(0) + rng(1) + rng(n - 2) + rng(n - 1);
    if (pair == 2) return 8 + 3L * (n - 4);              /* v3 rows pass: straight-line edges share P[0], P[1] */
    return t + (pair ? 2 + 3L * (n - 4) : 4L * (n - 4));
}

std::vector<float> sppf_pool_reference(const std::vector<float>& input, int channels, int h, int w) {
    if (channels <= 0 || h <= 0 || w <= 0 ||
        input.size() != static_cast<std::size_t>(channels) * h * w)
        throw std::invalid_argument("Invalid SPPF input tensor");
    auto p1 = maxpool5x5(input, channels, h, w);
    auto p2 = maxpool5x5(p1, channels, h, w);
    auto p3 = maxpool5x5(p2, channels, h, w);
    std::vector<float> result;
    result.reserve(input.size() * 4);
    result.insert(result.end(), input.begin(), input.end());
    result.insert(result.end(), p1.begin(), p1.end());
    result.insert(result.end(), p2.begin(), p2.end());
    result.insert(result.end(), p3.begin(), p3.end());
    return result;
}
} // namespace fastyolo
