#include <immintrin.h>

int main(void)
{
    /* VL */
    __m256i a = _mm256_abs_epi64(_mm256_setzero_si256());
    /* DQ */
    __m512i b = _mm512_broadcast_i32x8(a);
    /* BW */
    b = _mm512_abs_epi16(b);
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(b));
}
