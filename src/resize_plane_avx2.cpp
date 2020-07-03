#include "JincRessize.h"

#if !defined(__AVX2__)
#error "AVX2 option needed"
#endif

template <typename T>
static __forceinline __m256i convert(const __m128i& x)
{
    return _mm256_cvtepu8_epi32(x);
}

template <>
__forceinline __m256i convert<uint16_t>(const __m128i& x)
{
    return _mm256_cvtepu16_epi32(x);
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
void resize_plane_avx2(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch)
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
                __m256 result = _mm256_setzero_ps();

                for (int ly = 0; ly < coeff->filter_size; ly++)
                {
                    for (int lx = 0; lx < coeff->filter_size; lx += 8)
                    {
                        __m128i src_load = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + lx));

                        __m256 src_ps = _mm256_cvtepi32_ps(convert<T>(src_load));
                        __m256 coeff = _mm256_load_ps(coeff_ptr + lx);

                        result = _mm256_fmadd_ps(src_ps, coeff, result);
                    }

                    coeff_ptr += coeff->coeff_stride;
                    src_ptr += src_pitch;
                }

                __m256 sum = _mm256_hadd_ps(_mm256_hadd_ps(result, result), _mm256_hadd_ps(result, result));
                __m128 lo_sum = _mm_add_ss(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));

                // Convert back to interger + round + Save data
                dst[x] = _mm_cvtsi128_si32(pack<T>(_mm_cvtps_epi32(lo_sum), _mm_setzero_si128()));

                meta++;
            }
            else
            {
                __m256 result = _mm256_setzero_ps();

                for (int ly = 0; ly < coeff->filter_size; ly++) {
                    for (int lx = 0; lx < coeff->filter_size; lx += 8) {
                        __m256 src_ps = _mm256_loadu_ps(reinterpret_cast<const float*>(src_ptr + lx));

                        __m256 coeff = _mm256_load_ps(coeff_ptr + lx);

                        result = _mm256_fmadd_ps(src_ps, coeff, result);
                    }

                    coeff_ptr += coeff->coeff_stride;
                    src_ptr += src_pitch;
                }

                __m256 sum = _mm256_hadd_ps(_mm256_hadd_ps(result, result), _mm256_hadd_ps(result, result));
                __m128 lo_sum = _mm_add_ss(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));

                // Save data
                _mm_storeu_ps(reinterpret_cast<float*>(dst + x), lo_sum);

                meta++;
            }
        } // for (x)
        dst += dst_pitch;
    } // for (y)
}

template void resize_plane_avx2<uint8_t>(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch);
template void resize_plane_avx2<uint16_t>(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch);
template void resize_plane_avx2<float>(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch);
