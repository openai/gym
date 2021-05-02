#include <immintrin.h>

int main(void)
{
    __m512i a = _mm512_abs_epi32(_mm512_setzero_si512());
    return _mm_cvtsi128_si32(_mm512_castsi512_si128(a));
}
