#include "JincRessize.h"

#if !defined(__AVX2__)
#error "AVX2 option needed"
#endif

template <typename T>
void resize_plane_avx(EWAPixelCoeff* coeff, const T* src, T* VS_RESTRICT dst, int dst_width, int dst_height, int src_pitch, int dst_pitch) {};

template<>
void resize_plane_avx<uint8_t>(EWAPixelCoeff* coeff, const uint8_t* src, uint8_t* VS_RESTRICT dst, int dst_width, int dst_height, int src_pitch, int dst_pitch)
{
    EWAPixelCoeffMeta* meta = coeff->meta;

    //  assert(fitler_size == coeff->filter_size);

    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            const uint8_t* src_ptr = src + (meta->start_y * static_cast<int64_t>(src_pitch)) + meta->start_x;
            const float* coeff_ptr = coeff->factor + meta->coeff_meta;

            __m256 result = _mm256_setzero_ps();
            __m256i zero = _mm256_setzero_si256();

            for (int ly = 0; ly < coeff->filter_size; ly++) {
                for (int lx = 0; lx < coeff->filter_size; lx += 8) {
                    __m128i src_load = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + lx));
                    __m256i src_epi32 = _mm256_cvtepu8_epi32(src_load);

                    __m256 src_ps = _mm256_cvtepi32_ps(src_epi32);
                    __m256 coeff = _mm256_load_ps(coeff_ptr + lx);

                    result = _mm256_fmadd_ps(src_ps, coeff, result);
                }

                coeff_ptr += coeff->coeff_stride;
                src_ptr += src_pitch;
            }

            // Add to single float at the lower bit
            __m256 zero_ps = _mm256_setzero_ps();
            result = _mm256_hadd_ps(result, zero_ps);

            // I don't understand AVX2 at all...
            result = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(result), _MM_SHUFFLE(3, 1, 2, 0)));

            result = _mm256_hadd_ps(result, zero_ps);
            result = _mm256_hadd_ps(result, zero_ps);

            // Convert back to interger + clamp
            __m256i result_i = _mm256_cvtps_epi32(result);
            result_i = _mm256_packus_epi16(result_i, zero);

            // Save data
            dst[x] = _mm_cvtsi128_si32(_mm256_castsi256_si128(result_i));

            meta++;
        } // for (x)
        dst += dst_pitch;
    } // for (y)
}

template<>
void resize_plane_avx<uint16_t>(EWAPixelCoeff* coeff, const uint16_t* src, uint16_t* VS_RESTRICT dst, int dst_width, int dst_height, int src_pitch, int dst_pitch)
{
    EWAPixelCoeffMeta* meta = coeff->meta;

    //  assert(fitler_size == coeff->filter_size);

    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            const uint16_t* src_ptr = src + (meta->start_y * static_cast<int64_t>(src_pitch)) + meta->start_x;
            const float* coeff_ptr = coeff->factor + meta->coeff_meta;

            __m256 result = _mm256_setzero_ps();
            __m256i zero = _mm256_setzero_si256();

            for (int ly = 0; ly < coeff->filter_size; ly++) {
                for (int lx = 0; lx < coeff->filter_size; lx += 8) {
                    __m128i src_load = _mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + lx));
                    __m256i src_epi32 = _mm256_cvtepu16_epi32(src_load);

                    __m256 src_ps = _mm256_cvtepi32_ps(src_epi32);
                    __m256 coeff = _mm256_load_ps(coeff_ptr + lx);

                    result = _mm256_fmadd_ps(src_ps, coeff, result);
                }

                coeff_ptr += coeff->coeff_stride;
                src_ptr += src_pitch;
            }

            // Add to single float at the lower bit
            __m256 zero_ps = _mm256_setzero_ps();
            result = _mm256_hadd_ps(result, zero_ps);

            // I don't understand AVX2 at all...
            result = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(result), _MM_SHUFFLE(3, 1, 2, 0)));

            result = _mm256_hadd_ps(result, zero_ps);
            result = _mm256_hadd_ps(result, zero_ps);

            // Convert back to interger + clamp
            __m256i result_i = _mm256_cvtps_epi32(result);
            result_i = _mm256_packus_epi32(result_i, zero);

            // Save data
            dst[x] = _mm_cvtsi128_si32(_mm256_castsi256_si128(result_i));

            meta++;
        } // for (x)
        dst += dst_pitch;
    } // for (y)
}

void resize_plane_avx_float(EWAPixelCoeff* coeff, const float* src, float* VS_RESTRICT dst, int dst_width, int dst_height, int src_pitch, int dst_pitch)
{
    EWAPixelCoeffMeta* meta = coeff->meta;

    //  assert(fitler_size == coeff->filter_size);

    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            const float* src_ptr = src + (meta->start_y * static_cast<int64_t>(src_pitch)) + meta->start_x;
            const float* coeff_ptr = coeff->factor + meta->coeff_meta;

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

            // Add to single float at the lower bit
            __m256 zero_ps = _mm256_setzero_ps();
            result = _mm256_hadd_ps(result, zero_ps);

            // I don't understand AVX2 at all...
            result = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(result), _MM_SHUFFLE(3, 1, 2, 0)));

            result = _mm256_hadd_ps(result, zero_ps);
            result = _mm256_hadd_ps(result, zero_ps);

            // Save data
            _mm256_store_ps(reinterpret_cast<float*>(dst + x), result);

            meta++;
        } // for (x)
        dst += dst_pitch;
    } // for (y)
}
