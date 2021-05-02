#include <emmintrin.h>
#include <immintrin.h>

int main(void)
{
    __m128 a  = _mm_cvtph_ps(_mm_setzero_si128());
    __m256 a8 = _mm256_cvtph_ps(_mm_setzero_si128());
    return (int)(_mm_cvtss_f32(a) + _mm_cvtss_f32(_mm256_castps256_ps128(a8)));
}
