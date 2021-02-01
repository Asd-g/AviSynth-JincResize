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
#endif