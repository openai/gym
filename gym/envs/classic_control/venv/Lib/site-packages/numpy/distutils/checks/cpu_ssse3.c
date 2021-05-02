#include <tmmintrin.h>

int main(void)
{
    __m128i a = _mm_hadd_epi16(_mm_setzero_si128(), _mm_setzero_si128());
    return (int)_mm_cvtsi128_si32(a);
}
