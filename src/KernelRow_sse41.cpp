#include "JincRessize.h"

void JincResize::KernelRow_sse41_mul(int64_t iOutWidth)
{
    const int k_col4 = iKernelSize - (iKernelSize % 4);

    memset(g_pfFilteredImageBuffer, 0, iWidthEl * iHeightEl * iMul * iMul * sizeof(float));

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);

        // prepare float32 pre-converted row data for each threal separately
        int64_t tidx = omp_get_thread_num();
        float* pfInpRowSamplesFloatBufStart = pfInpFloatRow + tidx * iWidthEl;
        (this->*GetInpElRowAsFloat)(row, pfInpRowSamplesFloatBufStart);

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
        {
            float* pfCurrKernel_pos = g_pfKernel;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                for (int64_t k_col = 0; k_col < k_col4; k_col += 4)
                {
                    *(__m128*)(pfProc + k_col) = _mm_add_ps(_mm_mul_ps(*(__m128*)(pfCurrKernel_pos + k_col), _mm_load_ps1(pfInpRowSamplesFloatBufStart + col)), *(__m128*)(pfProc + k_col));
                }

                // need to process last up to 3 floats separately..
                for (int64_t k_col = k_col4; k_col < iKernelSize; ++k_col)
                    pfProc[k_col] += pfCurrKernel_pos[k_col];

                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize; // point to next kernel row now
            } // k_row
        } // col
    }
}
