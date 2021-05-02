#include <xmmintrin.h>
#include <immintrin.h>

int main(void)
{
    __m256 a = _mm256_fmadd_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
    return (int)_mm_cvtss_f32(_mm256_castps256_ps128(a));
}
