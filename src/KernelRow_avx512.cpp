#include "JincRessize.h"

#if !defined(__AVX512F__) && !defined(__INTEL_COMPILER)
void JincResize::KernelRowAll_avx512_mul(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{} // do nothing

void JincResize::KernelRowAll_avx512_mul_cb(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{} // do nothing
#else

void JincResize::KernelRowAll_avx512_mul(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{
    // current input plane sizes
    iWidthEl = iInpWidth + 2 * iKernelSize;
    iHeightEl = iInpHeight + 2 * iKernelSize;

    float* pfCurrKernel = g_pfKernel;

    memset(g_pfFilteredImageBuffer, 0, iWidthEl * iHeightEl * iMul * iMul);
    const int k_col16 = iKernelSize - (iKernelSize % 16);

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iWidthEl * iMul - (iKernelSize / 2);

        // prepare float32 pre-converted row data for each threal separately
        int64_t tidx = omp_get_thread_num();
        float* pfInpRowSamplesFloatBufStart = pfInpFloatRow + tidx * iWidthEl;
        (this->*GetInpElRowAsFloat)(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);
//        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart); // still no avx512

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
        {
            float* pfCurrKernel_pos = g_pfKernel;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                for (int64_t k_col = 0; k_col < k_col16; k_col += 16)
                {
                    *(__m512*)(pfProc + k_col) = _mm512_fmadd_ps(*(__m512*)(pfCurrKernel_pos + k_col), _mm512_broadcast_f32x4(_mm_load_ps1(pfInpRowSamplesFloatBufStart + col)), *(__m512*)(pfProc + k_col));
                }

                // need to process last up to 15 floats separately..
                for (int64_t k_col = k_col16; k_col < iKernelSize; ++k_col)
                    pfProc[k_col] += pfCurrKernel_pos[k_col];

                pfProc += iWidthEl * iMul; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize; // point to next kernel row now
            } // k_row
        } // col
    }

    ConvertToInt_avx2(iInpWidth, iInpHeight, dst, iDstStride);
}

void JincResize::KernelRowAll_avx512_mul_cb(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{
    // current input plane sizes
    iWidthEl = iInpWidth + 2 * iKernelSize;
    iHeightEl = iInpHeight + 2 * iKernelSize;

    float* pfCurrKernel = g_pfKernel;

    memset(pfFilteredCirculatingBuf, 0, iWidthEl * iKernelSize * iMul * sizeof(float));

    const int k_col16 = iKernelSize - (iKernelSize % 16);

    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here
        // prepare float32 pre-converted row data for each threal separately
        //int64_t tidx = omp_get_thread_num();
        float* pfInpRowSamplesFloatBufStart = pfInpFloatRow; // +tidx * iWidthEl;
        (this->*GetInpElRowAsFloat)(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
        {
            float* pfCurrKernel_pos = pfCurrKernel;
            float* pfProc;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                pfProc = vpfRowsPointers[k_row] + col * iMul;
                for (int64_t k_col = 0; k_col < k_col16; k_col += 16)
                {
                    *(__m512*)(pfProc + k_col) = _mm512_fmadd_ps(*(__m512*)(pfCurrKernel_pos + k_col), _mm512_broadcast_f32x4(_mm_load_ps1(pfInpRowSamplesFloatBufStart + col)), *(__m512*)(pfProc + k_col));
                }

                // need to process last up to 15 floats separately..
                for (int64_t k_col = k_col16; k_col < iKernelSize; ++k_col)
                    pfProc[k_col] += pfCurrKernel_pos[k_col];

                pfCurrKernel_pos += iKernelSize; // point to next kernel row now
            } // k_row
        } // col

        int iOutStartRow = (row - (iTaps + iKernelSize)) * iMul;
        //iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
        if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight)*iMul)
        {
            ConvertiMulRowsToInt_avx2(iInpWidth, iOutStartRow, dst, iDstStride);
        }

        // circulate pointers to iMul rows upper
        std::rotate(vpfRowsPointers.begin(), vpfRowsPointers.begin() + iMul, vpfRowsPointers.end());

        // clear last iMul rows
        for (int i = iKernelSize - iMul; i < iKernelSize; i++)
        {
            memset(vpfRowsPointers[i], 0, iWidthEl * iMul * sizeof(float));
        }
    } // row
}

__forceinline void JincResize::ConvertiMulRowsToInt_avx2(int iInpWidth, int iOutStartRow, unsigned char* dst, int iDstStride) // still no avx512 version
{
    const int col32 = (iInpWidth * iMul) - ((iInpWidth * iMul) % 32);

    __m256 my_zero_ymm2;

    int row_float_buf_index = 0;

    for (int64_t row = iOutStartRow; row < iOutStartRow + iMul; row++)
    {
        my_zero_ymm2 = _mm256_setzero_ps();

        int64_t col = 0;
        float* pfProc = vpfRowsPointers[row_float_buf_index] + (iKernelSize + iTaps) * iMul;
        unsigned char* pucDst = dst + row * (int64_t)iDstStride;

        for (col = 0; col < col32; col += 32)
        {
            __m256 my_Val_ymm1 = _mm256_load_ps(pfProc);
            __m256 my_Val_ymm2 = _mm256_load_ps(pfProc + 8);
            __m256 my_Val_ymm3 = _mm256_load_ps(pfProc + 16);
            __m256 my_Val_ymm4 = _mm256_load_ps(pfProc + 24);

            my_Val_ymm1 = _mm256_max_ps(my_Val_ymm1, my_zero_ymm2);
            my_Val_ymm2 = _mm256_max_ps(my_Val_ymm2, my_zero_ymm2);
            my_Val_ymm3 = _mm256_max_ps(my_Val_ymm3, my_zero_ymm2);
            my_Val_ymm4 = _mm256_max_ps(my_Val_ymm4, my_zero_ymm2);

            __m256i my_iVal_ymm1 = _mm256_cvtps_epi32(my_Val_ymm1);
            __m256i my_iVal_ymm2 = _mm256_cvtps_epi32(my_Val_ymm2);
            __m256i my_iVal_ymm3 = _mm256_cvtps_epi32(my_Val_ymm3);
            __m256i my_iVal_ymm4 = _mm256_cvtps_epi32(my_Val_ymm4);

            __m256i my_iVal_12 = _mm256_packus_epi32(my_iVal_ymm1, my_iVal_ymm2);
            __m256i my_iVal_34 = _mm256_packus_epi32(my_iVal_ymm3, my_iVal_ymm4);

            __m256i my_iVal_1234 = _mm256_packus_epi16(my_iVal_12, my_iVal_34);

            __m256i my_iVal_ymm1234 = _mm256_permutevar8x32_epi32(my_iVal_1234, _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));

            _mm256_stream_si256((__m256i*)(pucDst), my_iVal_ymm1234);

            pucDst += 32;
            pfProc += 32;
        }
        // last up to 31..
        for (int64_t l_col = col32; l_col < iInpWidth * iMul; ++l_col)
        {

            float fVal = *pfProc;

            fVal += 0.5f;

            if (fVal > 255.0f)
            {
                fVal = 255.0f;
            }
            if (fVal < 0.0f)
            {
                fVal = 0.0f;
            }

            *pucDst = (unsigned char)fVal;

            pucDst++;
            pfProc++;
        } //l_col

        row_float_buf_index++;
    } // row
}
#endif