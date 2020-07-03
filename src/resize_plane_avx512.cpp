#include "JincRessize.h"

#if !defined(__AVX512F__ ) && !defined(__INTEL_COMPILER)
#error "AVX512 option needed"
#endif

template <typename T>
static __forceinline __m512i convert(void const* p)
{
    return _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
}

template <>
__forceinline __m512i convert<uint16_t>(void const* p)
{
    return _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
}

template <typename T>
static __forceinline __m128i pack(const __m128i& x, const __m128i& y)
{
    return _mm_packus_epi16(x, y);
}

template <>
__forceinline __m128i pack<uint16_t>(const __m128i& x, const __m128i& y)
{
    return _mm_packus_epi32(x, y);
}

template <typename T>
void resize_plane_avx512(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch)
{
    EWAPixelCoeffMeta* meta = coeff->meta;

    const T* src = reinterpret_cast<const T*>(src_);
    T* VS_RESTRICT dst = reinterpret_cast<T*>(dst_);

    src_pitch /= sizeof(T);
    dst_pitch /= sizeof(T);

    //  assert(fitler_size == coeff->filter_size);

    for (int y = 0; y < dst_height; y++)
    {
        for (int x = 0; x < dst_width; x++)
        {
            const T* src_ptr = src + (meta->start_y * static_cast<int64_t>(src_pitch)) + meta->start_x;
            const float* coeff_ptr = coeff->factor + meta->coeff_meta;

            if (!(std::is_same_v<T, float>))
            {
                __m512 result = _mm512_setzero_ps();

                for (int ly = 0; ly < coeff->filter_size; ly++)
                {
                    for (int lx = 0; lx < coeff->filter_size; lx += 16)
                    {
                        __m512 src_ps = _mm512_cvtepi32_ps(convert<T>(src_ptr + lx));
                        __m512 coeff = _mm512_load_ps(coeff_ptr + lx);

                        result = _mm512_fmadd_ps(src_ps, coeff, result);
                    }

                    coeff_ptr += coeff->coeff_stride;
                    src_ptr += src_pitch;
                }

                __m256 sum_lo = _mm256_hadd_ps(_mm512_castps512_ps256(result), _mm512_castps512_ps256(result));
                __m128 lo_sum = _mm_add_ss(_mm256_castps256_ps128(_mm256_hadd_ps(sum_lo, sum_lo)), _mm256_extractf128_ps(_mm256_hadd_ps(sum_lo, sum_lo), 1));

                __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(result), 1));
                __m128 sum_hi = _mm256_extractf128_ps(_mm256_hadd_ps(_mm256_hadd_ps(hi, hi), _mm256_hadd_ps(hi, hi)), 1);
                __m128 hi_sum = _mm_add_ss(_mm256_castps256_ps128(_mm256_hadd_ps(_mm256_hadd_ps(hi, hi), _mm256_hadd_ps(hi, hi))), sum_hi);

                // Convert back to interger + round + Save data
                __m128i result_i = _mm_cvtps_epi32(_mm_add_ss(lo_sum, hi_sum));

                dst[x] = _mm_cvtsi128_si32(pack<T>(result_i, _mm_setzero_si128()));

                meta++;
            }
            else
            {
                __m512 result = _mm512_setzero_ps();

                for (int ly = 0; ly < coeff->filter_size; ly++)
                {
                    for (int lx = 0; lx < coeff->filter_size; lx += 16)
                    {
                        __m512 src_ps = _mm512_loadu_ps(reinterpret_cast<const float*>(src_ptr + lx));

                        __m512 coeff = _mm512_load_ps(coeff_ptr + lx);

                        result = _mm512_fmadd_ps(src_ps, coeff, result);
                    }

                    coeff_ptr += coeff->coeff_stride;
                    src_ptr += src_pitch;
                }

                __m256 sum_lo = _mm256_hadd_ps(_mm512_castps512_ps256(result), _mm512_castps512_ps256(result));
                __m128 lo_sum = _mm_add_ss(_mm256_castps256_ps128(_mm256_hadd_ps(sum_lo, sum_lo)), _mm256_extractf128_ps(_mm256_hadd_ps(sum_lo, sum_lo), 1));

                __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(result), 1));
                __m128 sum_hi = _mm256_extractf128_ps(_mm256_hadd_ps(_mm256_hadd_ps(hi, hi), _mm256_hadd_ps(hi, hi)), 1);
                __m128 hi_sum = _mm_add_ss(_mm256_castps256_ps128(_mm256_hadd_ps(_mm256_hadd_ps(hi, hi), _mm256_hadd_ps(hi, hi))), sum_hi);

                // Save data
                _mm_storeu_ps(reinterpret_cast<float*>(dst + x), _mm_add_ss(lo_sum, hi_sum));

                meta++;
            }
        } // for (x)
        dst += dst_pitch;
    } // for (y)
}

template void resize_plane_avx512<uint8_t>(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch);
template void resize_plane_avx512<uint16_t>(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch);
template void resize_plane_avx512<float>(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch);
