#include <immintrin.h>

#include "JincResize.h"

#if !defined(__AVX512F__ ) && !defined(__INTEL_COMPILER)
#error "AVX512 option needed"
#endif

template <typename T, int thr, int subsampled>
void JincResize::resize_plane_avx512(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi)
{
    const int planes_y[4] = { AVS_PLANAR_Y, AVS_PLANAR_U, AVS_PLANAR_V, AVS_PLANAR_A };
    const int planes_r[4] = { AVS_PLANAR_G, AVS_PLANAR_B, AVS_PLANAR_R, AVS_PLANAR_A };
    const int* current_planes = (avs_is_rgb(vi)) ? planes_r : planes_y;
    for (int i = 0; i < planecount; ++i)
    {
        const int plane = current_planes[i];

        const int src_stride = avs_get_pitch_p(src, plane) / sizeof(T);
        const int dst_stride = avs_get_pitch_p(dst, plane) / sizeof(T);
        const int dst_width = avs_get_row_size_p(dst, plane) / sizeof(T);
        const int dst_height = avs_get_height_p(dst, plane);
        const T* srcp = reinterpret_cast<const T*>(avs_get_read_ptr_p(src, plane));
        const __m512 min_val = (i && !avs_is_rgb(vi)) ? _mm512_set1_ps(-0.5f) : _mm512_setzero_ps();

        EWAPixelCoeff* out = [&]()
        {
            if constexpr (subsampled)
                return (i) ? (i == 3) ? JincResize::out[0] : JincResize::out[1] : JincResize::out[0];
            else
                return JincResize::out[0];
        }();

        auto loop = [&](int y)
        {
            T* __restrict dstp = reinterpret_cast<T*>(avs_get_write_ptr_p(dst, plane)) + static_cast<int64_t>(y) * dst_stride;

            for (int x = 0; x < dst_width; ++x)
            {
                EWAPixelCoeffMeta* meta = out->meta + static_cast<int64_t>(y) * dst_width + x;
                const T* src_ptr = srcp + (meta->start_y * static_cast<int64_t>(src_stride)) + meta->start_x;
                const float* coeff_ptr = out->factor + meta->coeff_meta;
                __m512 result = _mm512_setzero_ps();

                if constexpr (std::is_same_v<T, uint8_t>)
                {
                    for (int ly = 0; ly < out->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < out->filter_size; lx += 16)
                        {
                            const __m512 src_ps = _mm512_cvtepi32_ps(_mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + lx))));
                            const __m512 coeff = _mm512_load_ps(coeff_ptr + lx);
                            result = _mm512_fmadd_ps(src_ps, coeff, result);
                        }

                        coeff_ptr += out->coeff_stride;
                        src_ptr += src_stride;
                    }

                    const __m256 lo_hi_256 = _mm256_add_ps(_mm512_castps512_ps256(result), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(result), 1)));
                    __m128 hsum = _mm_add_ps(_mm256_castps256_ps128(lo_hi_256), _mm256_extractf128_ps(lo_hi_256, 1));
                    hsum = _mm_hadd_ps(_mm_hadd_ps(hsum, hsum), _mm_hadd_ps(hsum, hsum));
                    dstp[x] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi32(_mm_cvtps_epi32(hsum), _mm_setzero_si128()), _mm_setzero_si128()));
                }
                else if constexpr (std::is_same_v<T, uint16_t>)
                {
                    for (int ly = 0; ly < out->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < out->filter_size; lx += 16)
                        {
                            const __m512 src_ps = _mm512_cvtepi32_ps(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(src_ptr + lx))));
                            const __m512 coeff = _mm512_load_ps(coeff_ptr + lx);
                            result = _mm512_fmadd_ps(src_ps, coeff, result);
                        }

                        coeff_ptr += out->coeff_stride;
                        src_ptr += src_stride;
                    }

                    const __m256 lo_hi_256 = _mm256_add_ps(_mm512_castps512_ps256(result), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(result), 1)));
                    __m128 hsum = _mm_add_ps(_mm256_castps256_ps128(lo_hi_256), _mm256_extractf128_ps(lo_hi_256, 1));
                    hsum = _mm_hadd_ps(_mm_hadd_ps(hsum, hsum), _mm_hadd_ps(hsum, hsum));
                    dstp[x] = _mm_cvtsi128_si32(_mm_packus_epi32(_mm_cvtps_epi32(hsum), _mm_setzero_si128()));
                }
                else
                {
                    for (int ly = 0; ly < out->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < out->filter_size; lx += 16)
                        {
                            const __m512 src_ps = _mm512_max_ps(_mm512_loadu_ps(src_ptr + lx), min_val);
                            const __m512 coeff = _mm512_load_ps(coeff_ptr + lx);
                            result = _mm512_fmadd_ps(src_ps, coeff, result);
                        }

                        coeff_ptr += out->coeff_stride;
                        src_ptr += src_stride;
                    }

                    const __m256 lo_hi_256 = _mm256_add_ps(_mm512_castps512_ps256(result), _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(result), 1)));
                    __m128 hsum = _mm_add_ps(_mm256_castps256_ps128(lo_hi_256), _mm256_extractf128_ps(lo_hi_256, 1));
                    dstp[x] = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(hsum, hsum), _mm_hadd_ps(hsum, hsum)));
                }
            }
        };

        if constexpr (thr)
        {
            for (intptr_t i = 0; i < dst_height; ++i)
                loop(i);
        }
        else
        {
            std::vector<int> l(dst_height);
            std::iota(std::begin(l), std::end(l), 0);
            std::for_each(std::execution::par, std::begin(l), std::end(l), loop);
        }
    }
}

template void JincResize::resize_plane_avx512<uint8_t, 0, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx512<uint16_t, 0, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx512<float, 0, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);

template void JincResize::resize_plane_avx512<uint8_t, 1, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx512<uint16_t, 1, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx512<float, 1, 1>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);

template void JincResize::resize_plane_avx512<uint8_t, 0, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx512<uint16_t, 0, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx512<float, 0, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);

template void JincResize::resize_plane_avx512<uint8_t, 1, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx512<uint16_t, 1, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
template void JincResize::resize_plane_avx512<float, 1, 0>(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
