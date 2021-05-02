#include <pmmintrin.h>

int main(void)
{
    __m128 a = _mm_hadd_ps(_mm_setzero_ps(), _mm_setzero_ps());
    return (int)_mm_cvtss_f32(a);
}
