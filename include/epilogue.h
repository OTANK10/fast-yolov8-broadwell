/*
 * epilogue.h  --  what happens to an accumulator register before it is stored.
 * Owner: Simon (fused epilogue). Everyone else just calls epilogue_ps().
 *
 * Bias is added by initializing the accumulators with it (free: no extra add).
 * SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x)) is applied while the 8 outputs
 * are still in a YMM register, so the output is written to memory exactly once.
 *
 * exp256_ps is the classic Cephes range reduction + degree-5 polynomial
 * (relative error ~1e-7, well inside our 1e-4 tolerance).
 *
 * The AVX2 part only compiles with -march=broadwell (cmake -DYOLO_ENABLE_BROADWELL=ON);
 * without it, kernels fall back to scalar code so the repo still builds on a Mac.
 */
#ifndef FASTYOLO_EPILOGUE_H
#define FASTYOLO_EPILOGUE_H

#include <cmath>
#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define FASTYOLO_HAVE_AVX2 1
#endif

namespace fastyolo {

static inline float silu_scalar(float x) { return x / (1.0f + std::exp(-x)); }

#ifdef FASTYOLO_HAVE_AVX2

static inline __m256 exp256_ps(__m256 x) {
    const __m256 hi = _mm256_set1_ps(88.3762626647949f);
    const __m256 lo = _mm256_set1_ps(-88.3762626647949f);
    x = _mm256_max_ps(_mm256_min_ps(x, hi), lo);

    /* exp(x) = 2^n * exp(g),  n = round(x / ln2),  g = x - n*ln2 in [-ln2/2, ln2/2] */
    __m256 fx = _mm256_fmadd_ps(x, _mm256_set1_ps(1.44269504088896341f), _mm256_set1_ps(0.5f));
    fx = _mm256_floor_ps(fx);
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(0.693359375f), x);    /* ln2 split in two   */
    x = _mm256_fnmadd_ps(fx, _mm256_set1_ps(-2.12194440e-4f), x); /* for extra accuracy */

    __m256 y = _mm256_set1_ps(1.9875691500E-4f);
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.3981999507E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(8.3334519073E-3f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(4.1665795894E-2f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(1.6666665459E-1f));
    y = _mm256_fmadd_ps(y, x, _mm256_set1_ps(5.0000001201E-1f));
    y = _mm256_fmadd_ps(y, _mm256_mul_ps(x, x), x);
    y = _mm256_add_ps(y, _mm256_set1_ps(1.0f));

    /* build 2^n by writing n+127 into the float exponent bits */
    __m256i n = _mm256_cvttps_epi32(fx);
    n = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);
    return _mm256_mul_ps(y, _mm256_castsi256_ps(n));
}

static inline __m256 silu256_ps(__m256 x) {
    __m256 e = exp256_ps(_mm256_sub_ps(_mm256_setzero_ps(), x)); /* exp(-x) */
    return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1.0f), e));
}

static inline __m256 epilogue_ps(__m256 acc, int fuse_silu) {
    return fuse_silu ? silu256_ps(acc) : acc;
}

#endif /* AVX2 */

} // namespace fastyolo
#endif
