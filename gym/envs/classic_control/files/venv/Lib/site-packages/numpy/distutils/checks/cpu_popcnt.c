#ifdef _MSC_VER
    #include <nmmintrin.h>
#else
    #include <popcntintrin.h>
#endif

int main(void)
{
    long long a = 0;
    int b;
#ifdef _MSC_VER
    #ifdef _M_X64
    a = _mm_popcnt_u64(1);
    #endif
    b = _mm_popcnt_u32(1);
#else
    #ifdef __x86_64__
    a = __builtin_popcountll(1);
    #endif
    b = __builtin_popcount(1);
#endif
    return (int)a + b;
}
