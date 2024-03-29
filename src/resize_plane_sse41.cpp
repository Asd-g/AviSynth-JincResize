#include <smmintrin.h>

#include "JincRessize.h"

template <typename T, int thr>
void JincResize::resize_plane_sse41(PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env)
{
    const int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* current_planes = (vi.IsRGB()) ? planes_r : planes_y;
    for (int i = 0; i < planecount; ++i)
    {
        const int plane = current_planes[i];

        const int src_stride = src->GetPitch(plane) / sizeof(T);
        const int dst_stride = dst->GetPitch(plane) / sizeof(T);
        const int dst_width = dst->GetRowSize(plane) / sizeof(T);
        const int dst_height = dst->GetHeight(plane);
        const T* srcp = reinterpret_cast<const T*>(src->GetReadPtr(plane));
        const __m128 min_val = (i && !vi.IsRGB()) ? _mm_set_ps1(-0.5f) : _mm_setzero_ps();

        auto loop = [&](int y)
        {
            T* __restrict dstp = reinterpret_cast<T*>(dst->GetWritePtr(plane)) + static_cast<int64_t>(y) * dst_stride;

            for (int x = 0; x < dst_width; ++x)
            {
                EWAPixelCoeffMeta* meta = out[i]->meta + static_cast<int64_t>(y) * dst_width + x;
                const T* src_ptr = srcp + (meta->start_y * static_cast<int64_t>(src_stride)) + meta->start_x;
                const float* coeff_ptr = out[i]->factor + meta->coeff_meta;
                __m128 result = _mm_setzero_ps();

                if constexpr (std::is_same_v<T, uint8_t>)
                {
                    for (int ly = 0; ly < out[i]->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < out[i]->filter_size; lx += 4)
                        {
                            const __m128 src_ps = _mm_cvtepi32_ps(_mm_cvtepu8_epi32(_mm_cvtsi32_si128(*(reinterpret_cast<const int32_t*>(src_ptr + lx)))));
                            const __m128 coeff = _mm_load_ps(coeff_ptr + lx);
                            result = _mm_add_ps(result, _mm_mul_ps(src_ps, coeff));
                        }

                        coeff_ptr += out[i]->coeff_stride;
                        src_ptr += src_stride;
                    }

                    const __m128 hsum = _mm_hadd_ps(_mm_hadd_ps(result, result), _mm_hadd_ps(result, result));
                    dstp[x] = _mm_cvtsi128_si32(_mm_packus_epi16(_mm_packus_epi32(_mm_cvtps_epi32(hsum), _mm_setzero_si128()), _mm_setzero_si128()));
                }
                else if constexpr (std::is_same_v<T, uint16_t>)
                {
                    for (int ly = 0; ly < out[i]->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < out[i]->filter_size; lx += 4)
                        {
                            const __m128 src_ps = _mm_cvtepi32_ps(_mm_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(src_ptr + lx))));
                            const __m128 coeff = _mm_load_ps(coeff_ptr + lx);
                            result = _mm_add_ps(result, _mm_mul_ps(src_ps, coeff));
                        }

                        coeff_ptr += out[i]->coeff_stride;
                        src_ptr += src_stride;
                    }

                    const __m128 hsum = _mm_hadd_ps(_mm_hadd_ps(result, result), _mm_hadd_ps(result, result));
                    dstp[x] = _mm_cvtsi128_si32(_mm_packus_epi32(_mm_cvtps_epi32(hsum), _mm_setzero_si128()));
                }
                else
                {
                    for (int ly = 0; ly < out[i]->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < out[i]->filter_size; lx += 4)
                        {
                            const __m128 src_ps = _mm_max_ps(_mm_loadu_ps(src_ptr + lx), min_val);
                            const __m128 coeff = _mm_load_ps(coeff_ptr + lx);
                            result = _mm_add_ps(result, _mm_mul_ps(src_ps, coeff));
                        }

                        coeff_ptr += out[i]->coeff_stride;
                        src_ptr += src_stride;
                    }

                    dstp[x] = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(result, result), _mm_hadd_ps(result, result)));
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

template void JincResize::resize_plane_sse41<uint8_t, 0>(PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);
template void JincResize::resize_plane_sse41<uint16_t, 0>(PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);
template void JincResize::resize_plane_sse41<float, 0>(PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);

template void JincResize::resize_plane_sse41<uint8_t, 1>(PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);
template void JincResize::resize_plane_sse41<uint16_t, 1>(PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);
template void JincResize::resize_plane_sse41<float, 1>(PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);
