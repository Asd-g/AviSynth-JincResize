#include "JincRessize.h"
/*
#if !defined(__AVX2__)
#error "AVX2 option needed"
#endif
*/
void JincResize::KernelRow_avx2(int64_t iOutWidth)
{
    const int k_col8 = iKernelSize - (iKernelSize % 32);

#pragma omp parallel for num_threads(threads_) // do not works for x64 and VS2019 compiler still - need to fix (pointers ?)
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here
        //int64_t iProcPtr = static_cast<int64_t>((row * iMul - (iKernelSize / 2)) * iOutWidth) + (col * iMul - (iKernelSize / 2));
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);
        int64_t iInpPtrRowStart = row * iWidthEl;

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col+=2) // input cols counter
        {
            unsigned char ucInpSample = g_pElImageBuffer[(iInpPtrRowStart + col)];
            unsigned char ucInpSample2 = g_pElImageBuffer[(iInpPtrRowStart + col + 1)];

            // fast skip zero
//            if (ucInpSample == 0) continue;

            float* pfCurrKernel = g_pfKernelWeighted + static_cast<int64_t>(ucInpSample) * iKernelStride * iKernelSize + (iKernelStride - iKernelSize) / 2;
            float* pfCurrKernel2 = g_pfKernelWeighted + static_cast<int64_t>(ucInpSample2) * iKernelStride * iKernelSize + (iKernelStride - iKernelSize) / 2;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4, my_ymm5, my_ymm6, my_ymm7;

                // add full kernel row to output - AVX2
                for (int64_t k_col = 0; k_col < k_col8; k_col += 32)
                {
/*                    my_ymm0 =_mm256_loadu_ps(pfProc + k_col);
                    my_ymm1 = _mm256_loadu_ps(pfProc + k_col + 8);

                    my_ymm2 = _mm256_loadu_ps(pfCurrKernel + k_col);
                    my_ymm3 = _mm256_loadu_ps(pfCurrKernel + k_col + 8);

                    _mm256_storeu_ps(pfProc + k_col, _mm256_add_ps(my_ymm0,my_ymm2));
                    _mm256_storeu_ps(pfProc + k_col + 8, _mm256_add_ps(my_ymm1, my_ymm3)); */
                    my_ymm0 = _mm256_loadu_ps(pfProc + k_col);
                    my_ymm1 = _mm256_loadu_ps(pfProc + k_col + 8);
                    my_ymm2 = _mm256_loadu_ps(pfProc + k_col + 16);
                    my_ymm3 = _mm256_loadu_ps(pfProc + k_col + 24);

                    my_ymm4 = _mm256_loadu_ps(pfCurrKernel + k_col);
                    my_ymm5 = _mm256_loadu_ps(pfCurrKernel + k_col + 8);
                    my_ymm6 = _mm256_loadu_ps(pfCurrKernel + k_col + 16);
                    my_ymm7 = _mm256_loadu_ps(pfCurrKernel + k_col + 24);

                    my_ymm0 = _mm256_add_ps(my_ymm0, my_ymm4);
                    my_ymm1 = _mm256_add_ps(my_ymm1, my_ymm5);
                    my_ymm2 = _mm256_add_ps(my_ymm2, my_ymm6);
                    my_ymm3 = _mm256_add_ps(my_ymm3, my_ymm7);

                    my_ymm4 = _mm256_loadu_ps(pfCurrKernel2 + k_col - iMul);
                    my_ymm5 = _mm256_loadu_ps(pfCurrKernel2 + k_col - iMul + 8);
                    my_ymm6 = _mm256_loadu_ps(pfCurrKernel2 + k_col - iMul + 16);
                    my_ymm7 = _mm256_loadu_ps(pfCurrKernel2 + k_col - iMul + 24);

                    _mm256_storeu_ps(pfProc + k_col, _mm256_add_ps(my_ymm0, my_ymm4));
                    _mm256_storeu_ps(pfProc + k_col + 8, _mm256_add_ps(my_ymm1, my_ymm5));
                    _mm256_storeu_ps(pfProc + k_col + 16, _mm256_add_ps(my_ymm2, my_ymm6));
                    _mm256_storeu_ps(pfProc + k_col + 24, _mm256_add_ps(my_ymm3, my_ymm7));

 //                   _mm256_storeu_ps(pfProc + k_col, _mm256_add_ps(_mm256_loadu_ps(pfCurrKernel + k_col), _mm256_loadu_ps(pfProc + k_col)));
 
                }
                
                // need to process last up to 7 floats separately..
                for (int64_t k_col = k_col8; k_col < iKernelSize; ++k_col)
                    pfProc[k_col] += pfCurrKernel[k_col];

                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel += iKernelSize; // point to next kernel row now
                pfCurrKernel2 += iKernelSize;
            } // k_row
        } // col
    }
}


void JincResize::KernelRow_avx2_mul(int64_t iOutWidth)
{
//    const int k_col8 = iKernelSize - (iKernelSize % 8);
    const int k_col32 = iKernelSize - (iKernelSize % 32);

//    AVX2Row = &JincResize::AVX2Row32;

    float *pfCurrKernel = g_pfKernel + (iKernelStride-iKernelSize)/2; 

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);
        int64_t iInpPtrRowStart = row * iWidthEl;

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col+=2) // input cols counter
        {
            unsigned char ucInpSample = g_pElImageBuffer[(iInpPtrRowStart + col)];
            unsigned char ucInpSample2 = g_pElImageBuffer[(iInpPtrRowStart + col + 1)];

            float fSample = (float)ucInpSample;
            float fSample2 = (float)ucInpSample2;

            const __m256 inp256 = _mm256_broadcast_ss(&fSample);
            const __m256 inp256_2 = _mm256_broadcast_ss(&fSample2);

            float *pfCurrKernel_pos = pfCurrKernel;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4, my_ymm5, my_ymm6, my_ymm7, my_ymm8, my_ymm9;

                for (int64_t k_col = 0; k_col < k_col32; k_col += 32)
                {
                    my_ymm0 = _mm256_loadu_ps(pfProc + k_col);
                    my_ymm1 = _mm256_loadu_ps(pfProc + k_col + 8);
                    my_ymm2 = _mm256_loadu_ps(pfProc + k_col + 16);
                    my_ymm3 = _mm256_loadu_ps(pfProc + k_col + 24);
                    my_ymm9 = _mm256_loadu_ps(pfProc + k_col + 32);

                    my_ymm4 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col), inp256);
                    my_ymm5 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col + 8), inp256);
                    my_ymm6 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col + 16), inp256);
                    my_ymm7 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col + 24), inp256);

                    my_ymm0 = _mm256_add_ps(my_ymm0, my_ymm4);
                    my_ymm1 = _mm256_add_ps(my_ymm1, my_ymm5);
                    my_ymm2 = _mm256_add_ps(my_ymm2, my_ymm6);
                    my_ymm3 = _mm256_add_ps(my_ymm3, my_ymm7);

                    my_ymm4 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col - iMul), inp256_2); // decrement pointer to get kernel shift right after loading
                    my_ymm5 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col - iMul + 8), inp256_2);
                    my_ymm6 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col - iMul + 16), inp256_2);
                    my_ymm7 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col - iMul + 24), inp256_2);
                    my_ymm8 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col - iMul + 32), inp256_2);
          
                    _mm256_storeu_ps(pfProc + k_col, _mm256_add_ps(my_ymm0, my_ymm4));
                    _mm256_storeu_ps(pfProc + k_col + 8, _mm256_add_ps(my_ymm1, my_ymm5));
                    _mm256_storeu_ps(pfProc + k_col + 16, _mm256_add_ps(my_ymm2, my_ymm6));
                    _mm256_storeu_ps(pfProc + k_col + 24, _mm256_add_ps(my_ymm3, my_ymm7));
                    _mm256_storeu_ps(pfProc + k_col + 32, _mm256_add_ps(my_ymm9, my_ymm8)); // second sample last values
           //         _mm256_storeu_ps(pfProc + k_col, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col), inp256), _mm256_loadu_ps(pfProc + k_col)));

                }
                
                // need to process last (up to 7) floats separately..
                for (int64_t k_col = k_col32; k_col < iKernelSize; ++k_col)
                {
                    pfProc[k_col] += (pfCurrKernel_pos[k_col] * fSample);
                    pfProc[k_col] += (pfCurrKernel_pos[k_col - iMul] * fSample2);
                }

                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelStride; // point to next kernel row now
             } // k_row
        } // col
    }
}

void JincResize::AVX2Row32(int64_t k_col_x, float* pfProc, float* pfCurrKernel_pos, float *pfSample)
{
    const __m256 inp256 = _mm256_broadcast_ss(pfSample);

    for (int64_t k_col = 0; k_col < k_col_x; k_col += 32)
    {
        __m256 my_ymm0 = _mm256_loadu_ps(pfProc + k_col);
        __m256 my_ymm1 = _mm256_loadu_ps(pfProc + k_col + 8);
        __m256 my_ymm2 = _mm256_loadu_ps(pfProc + k_col + 16);
        __m256 my_ymm3 = _mm256_loadu_ps(pfProc + k_col + 24);

        __m256 my_ymm4 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col), inp256);
        __m256 my_ymm5 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col + 8), inp256);
        __m256 my_ymm6 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col + 16), inp256);
        __m256 my_ymm7 = _mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col + 24), inp256);

        _mm256_storeu_ps(pfProc + k_col, _mm256_add_ps(my_ymm0, my_ymm4));
        _mm256_storeu_ps(pfProc + k_col + 8, _mm256_add_ps(my_ymm1, my_ymm5));
        _mm256_storeu_ps(pfProc + k_col + 16, _mm256_add_ps(my_ymm2, my_ymm6));
        _mm256_storeu_ps(pfProc + k_col + 24, _mm256_add_ps(my_ymm3, my_ymm7));
    }
}

