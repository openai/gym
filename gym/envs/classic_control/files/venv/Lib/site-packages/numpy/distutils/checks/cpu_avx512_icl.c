#include <immintrin.h>

int main(void)
{
    /* VBMI2 */
    __m512i a = _mm512_shrdv_epi64(_mm512_setzero_si512(), _mm512_setzero_si512(), _mm512_setzero_si512());
    /* BITLAG */
    a = _mm512_popcnt_epi8(a);
    /* VPOPCNTDQ */
    a = _mm512_popcnt_epi64(a);
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
