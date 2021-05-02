#include <immintrin.h>

int main(void)
{
    /* IFMA */
    __m512i a = _mm512_madd52hi_epu64(_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512());
    /* VMBI */
    a = _mm512_permutex2var_epi8(a, _mm512_setzero_si512(), _mm512_setzero_si512());
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
