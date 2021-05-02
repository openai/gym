#include <immintrin.h>

int main(void)
{
    __m512i a = _mm512_setzero_si512();
    __m512 b = _mm512_setzero_ps();

    /* 4FMAPS */
    b = _mm512_4fmadd_ps(b, b, b, b, b, NULL);
    /* 4VNNIW */
    a = _mm512_4dpwssd_epi32(a, a, a, a, a, NULL);
    /* VPOPCNTDQ */
    a = _mm512_popcnt_epi64(a);

    a = _mm512_add_epi32(a, _mm512_castps_si512(b));
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
