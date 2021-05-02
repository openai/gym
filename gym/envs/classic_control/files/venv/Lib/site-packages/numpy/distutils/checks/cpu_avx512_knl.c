#include <immintrin.h>

int main(void)
{
    int base[128];
    /* ER */
    __m512i a = _mm512_castpd_si512(_mm512_exp2a23_pd(_mm512_setzero_pd()));
    /* PF */
    _mm512_mask_prefetch_i64scatter_pd(base, _mm512_cmpeq_epi64_mask(a, a), a, 1, _MM_HINT_T1);
    return base[0];
}
