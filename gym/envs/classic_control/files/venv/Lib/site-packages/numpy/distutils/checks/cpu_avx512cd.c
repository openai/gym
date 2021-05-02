#include <immintrin.h>

int main(void)
{
    __m512i a = _mm512_lzcnt_epi32(_mm512_setzero_si512());
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
