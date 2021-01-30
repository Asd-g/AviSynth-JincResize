#include "JincRessize.h"

#if !defined(__AVX2__)
#error "AVX2 option needed"
#endif

template <typename T>
void JincResize::resize_plane_avx2(EWAPixelCoeff* coeff[3], PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env)
{
    const int pixel_size = sizeof(T);
    const int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* current_planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
    for (int i = 0; i < planecount; ++i)
    {
        const int plane = current_planes[i];

        const int src_stride = src->GetPitch(plane) / pixel_size;
        const int dst_stride = dst->GetPitch(plane) / pixel_size;
        const int dst_width = dst->GetRowSize(plane) / pixel_size;
        const int dst_height = dst->GetHeight(plane);
        const T* srcp = reinterpret_cast<const T*>(src->GetReadPtr(plane));
        const __m256 min_val = (i && !vi.IsRGB()) ? _mm256_set1_ps(-0.5f) : _mm256_setzero_ps();

        if (bAddProc)
        {
            const int src_width = src->GetRowSize(plane) / pixel_size;
            const int src_height = src->GetHeight(plane);
            unsigned char* dstp_add = reinterpret_cast<unsigned char*>(dst->GetWritePtr(plane));
            KernelProc((unsigned char*)srcp, src_stride, src_width, src_height, (unsigned char*)dstp_add, dst_stride);
            continue;
        }

#pragma omp parallel for num_threads(threads_)
        for (int y = 0; y < dst_height; ++y)
        {
            T* __restrict dstp = reinterpret_cast<T*>(dst->GetWritePtr(plane)) + static_cast<int64_t>(y) * dst_stride;

            for (int x = 0; x < dst_width; ++x)
            {
                EWAPixelCoeffMeta* meta = coeff[i]->meta + static_cast<int64_t>(y) * dst_width + x;
                const T* src_ptr = srcp + (meta->start_y * static_cast<int64_t>(src_stride)) + meta->start_x;
                const float* coeff_ptr = coeff[i]->factor + meta->coeff_meta;
                __m256 result = _mm256_setzero_ps();

//                if constexpr (std::is_same_v<T, uint8_t>)
				if (std::is_same_v<T, uint8_t>)
                {
                    for (int ly = 0; ly < coeff[i]->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < coeff[i]->filter_size; lx += 8)
                        {
                            const __m256 src_ps = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr + lx)))));
                            const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                            result = _mm256_fmadd_ps(src_ps, coeff, result);
                        }

                        coeff_ptr += coeff[i]->coeff_stride;
                        src_ptr += src_stride;
                    }

                    __m128 hsum = _mm_add_ps(_mm256_castps256_ps128(result), _mm256_extractf128_ps(result, 1));
                    hsum = _mm_hadd_ps(_mm_hadd_ps(hsum, hsum), _mm_hadd_ps(hsum, hsum));
                    const __m128i src_int = _mm_packus_epi16(_mm_packs_epi32(_mm_cvtps_epi32(hsum), _mm_setzero_si128()), _mm_setzero_si128());
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + x), src_int);

                }
//                else if constexpr (std::is_same_v<T, uint16_t>)
				else if (std::is_same_v<T, uint16_t>)
                {
                    for (int ly = 0; ly < coeff[i]->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < coeff[i]->filter_size; lx += 8)
                        {
                            const __m256 src_ps = _mm256_cvtepi32_ps(_mm256_cvtepu16_epi32(_mm_loadu_si128(const_cast<__m128i*>(reinterpret_cast<const __m128i*>(src_ptr + lx)))));
                            const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                            result = _mm256_fmadd_ps(src_ps, coeff, result);
                        }

                        coeff_ptr += coeff[i]->coeff_stride;
                        src_ptr += src_stride;
                    }

                    __m128 hsum = _mm_add_ps(_mm256_castps256_ps128(result), _mm256_extractf128_ps(result, 1));
                    hsum = _mm_hadd_ps(_mm_hadd_ps(hsum, hsum), _mm_hadd_ps(hsum, hsum));
                    const __m128i src_int = _mm_packus_epi32(_mm_cvtps_epi32(hsum), _mm_setzero_si128());
                    _mm_storeu_si128(reinterpret_cast<__m128i*>(dstp + x), src_int);

                }
                else
                {
                    for (int ly = 0; ly < coeff[i]->filter_size; ++ly)
                    {
                        for (int lx = 0; lx < coeff[i]->filter_size; lx += 8)
                        {
//                          const __m256 src_ps = _mm256_max_ps(_mm256_loadu_ps(src_ptr + lx), min_val);
							const __m256 src_ps = _mm256_max_ps(*(__m256*)(src_ptr + lx), min_val);
                            const __m256 coeff = _mm256_load_ps(coeff_ptr + lx);
                            result = _mm256_fmadd_ps(src_ps, coeff, result);
                        }

                        coeff_ptr += coeff[i]->coeff_stride;
                        src_ptr += src_stride;
                    }

                    __m128 hsum = _mm_add_ps(_mm256_castps256_ps128(result), _mm256_extractf128_ps(result, 1));
                    dstp[x] = _mm_cvtss_f32(_mm_hadd_ps(_mm_hadd_ps(hsum, hsum), _mm_hadd_ps(hsum, hsum)));
                }
            }
        }
    }
}

template void JincResize::resize_plane_avx2<uint8_t>(EWAPixelCoeff* coeff[3], PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);
template void JincResize::resize_plane_avx2<uint16_t>(EWAPixelCoeff* coeff[3], PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);
template void JincResize::resize_plane_avx2<float>(EWAPixelCoeff* coeff[3], PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);
