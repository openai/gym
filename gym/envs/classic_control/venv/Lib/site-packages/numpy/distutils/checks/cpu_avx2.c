#include <immintrin.h>

int main(void)
{
    __m256i a = _mm256_abs_epi16(_mm256_setzero_si256());
    return _mm_cvtsi128_si32(_mm256_castsi256_si128(a));
}
