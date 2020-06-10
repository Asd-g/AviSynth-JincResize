#include <smmintrin.h>

#include "JincRessize.h"

template <typename T>
#if defined(CLANG) || defined(GCC)
__attribute__((__target__("sse4.1")))
#endif
void resize_plane_sse(EWAPixelCoeff* coeff, const T* src, T* VS_RESTRICT dst, int dst_width, int dst_height, int src_pitch, int dst_pitch) {};

template<>
void resize_plane_sse<uint8_t>(EWAPixelCoeff* coeff, const uint8_t* src, uint8_t* VS_RESTRICT dst, int dst_width, int dst_height, int src_pitch, int dst_pitch)
{
    EWAPixelCoeffMeta* meta = coeff->meta;

    //  assert(fitler_size == coeff->filter_size);

    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            const uint8_t* src_ptr = src + (meta->start_y * static_cast<int64_t>(src_pitch)) + meta->start_x;
            const float* coeff_ptr = coeff->factor + meta->coeff_meta;

            __m128 result = _mm_setzero_ps();
            __m128i zero = _mm_setzero_si128();

            for (int ly = 0; ly < coeff->filter_size; ly++) {
                for (int lx = 0; lx < coeff->filter_size; lx += 4) {
                    __m128i src_epi64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr + lx));
                    __m128i src_epi32 = _mm_cvtepu8_epi32(src_epi64);

                    __m128 src_ps = _mm_cvtepi32_ps(src_epi32);
                    __m128 coeff = _mm_load_ps(coeff_ptr + lx);

                    __m128 multiplied = _mm_mul_ps(src_ps, coeff);
                    result = _mm_add_ps(result, multiplied);
                }
                coeff_ptr += coeff->coeff_stride;
                src_ptr += src_pitch;
            }

            // use 3xshuffle + 2xadd instead of 2xhadd
            __m128 result1 = result;
            __m128 result2 = _mm_shuffle_ps(result1, result1, _MM_SHUFFLE(1, 0, 3, 2));

            result1 = _mm_add_ps(result1, result2);
            result2 = _mm_shuffle_ps(result1, result1, _MM_SHUFFLE(2, 3, 0, 1));

            result = _mm_add_ss(result1, result2);

            // Convert back to interger + clamp
            __m128i result_i = _mm_cvtps_epi32(result);
            result_i = _mm_packus_epi16(result_i, zero);

            // Save data
            dst[x] = _mm_cvtsi128_si32(result_i);

            meta++;
        } // for (x)
        dst += dst_pitch;
    } // for (y)
}

template<>
void resize_plane_sse<uint16_t>(EWAPixelCoeff* coeff, const uint16_t* src, uint16_t* VS_RESTRICT dst, int dst_width, int dst_height, int src_pitch, int dst_pitch)
{
    EWAPixelCoeffMeta* meta = coeff->meta;

    //  assert(fitler_size == coeff->filter_size);

    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            const uint16_t* src_ptr = src + (meta->start_y * static_cast<int64_t>(src_pitch)) + meta->start_x;
            const float* coeff_ptr = coeff->factor + meta->coeff_meta;

            __m128 result = _mm_setzero_ps();
            __m128i zero = _mm_setzero_si128();

            for (int ly = 0; ly < coeff->filter_size; ly++) {
                for (int lx = 0; lx < coeff->filter_size; lx += 4) {
                    __m128i src_epi64 = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(src_ptr + lx));
                    __m128i src_epi32 = _mm_cvtepu16_epi32(src_epi64);

                    __m128 src_ps = _mm_cvtepi32_ps(src_epi32);
                    __m128 coeff = _mm_load_ps(coeff_ptr + lx);

                    __m128 multiplied = _mm_mul_ps(src_ps, coeff);
                    result = _mm_add_ps(result, multiplied);
                }
                coeff_ptr += coeff->coeff_stride;
                src_ptr += src_pitch;
            }

            // use 3xshuffle + 2xadd instead of 2xhadd
            __m128 result1 = result;
            __m128 result2 = _mm_shuffle_ps(result1, result1, _MM_SHUFFLE(1, 0, 3, 2));

            result1 = _mm_add_ps(result1, result2);
            result2 = _mm_shuffle_ps(result1, result1, _MM_SHUFFLE(2, 3, 0, 1));

            result = _mm_add_ss(result1, result2);

            // Convert back to interger + clamp
            __m128i result_i = _mm_cvtps_epi32(result);
            result_i = _mm_packus_epi32(result_i, zero);

            // Save data
            dst[x] = _mm_cvtsi128_si32(result_i);

            meta++;
        } // for (x)
        dst += dst_pitch;
    } // for (y)
}

#if defined(CLANG) || defined(GCC)
__attribute__((__target__("sse4.1")))
#endif
void resize_plane_sse_float(EWAPixelCoeff* coeff, const float* src, float* VS_RESTRICT dst, int dst_width, int dst_height, int src_pitch, int dst_pitch)
{
    EWAPixelCoeffMeta* meta = coeff->meta;

    //  assert(fitler_size == coeff->filter_size);

    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            const float* src_ptr = src + (meta->start_y * static_cast<int64_t>(src_pitch)) + meta->start_x;
            const float* coeff_ptr = coeff->factor + meta->coeff_meta;

            __m128 result = _mm_setzero_ps();

            for (int ly = 0; ly < coeff->filter_size; ly++) {
                for (int lx = 0; lx < coeff->filter_size; lx += 4) {

                    __m128 src_ps = _mm_loadu_ps(reinterpret_cast<const float*>(src_ptr + lx));

                    __m128 coeff = _mm_load_ps(coeff_ptr + lx);

                    __m128 multiplied = _mm_mul_ps(src_ps, coeff);
                    result = _mm_add_ps(result, multiplied);
                }
                coeff_ptr += coeff->coeff_stride;
                src_ptr += src_pitch;
            }

            // use 3xshuffle + 2xadd instead of 2xhadd
            __m128 result1 = result;
            __m128 result2 = _mm_shuffle_ps(result1, result1, _MM_SHUFFLE(1, 0, 3, 2));

            result1 = _mm_add_ps(result1, result2);
            result2 = _mm_shuffle_ps(result1, result1, _MM_SHUFFLE(2, 3, 0, 1));

            result = _mm_add_ss(result1, result2);

            // Save data
            _mm_store_ps(reinterpret_cast<float*>(dst + x), result);

            meta++;
        } // for (x)
        dst += dst_pitch;
    } // for (y)
}
