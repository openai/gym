#include <immintrin.h>
#ifdef _MSC_VER
    #include <ammintrin.h>
#else
    #include <x86intrin.h>
#endif

int main(void)
{
    __m256 a = _mm256_macc_ps(_mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps());
    return (int)_mm_cvtss_f32(_mm256_castps256_ps128(a));
}
