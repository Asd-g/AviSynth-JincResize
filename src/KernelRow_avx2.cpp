#include "JincRessize.h"

#if !defined(__AVX2__)
#error "AVX2 option needed"
#endif

void JincResize::KernelRow_avx2(int64_t iOutWidth)
{
    const int k_col8 = iKernelSize - (iKernelSize % 8);

#pragma omp parallel for num_threads(threads_) // do not works for x64 and VS2019 compiler still - need to fix (pointers ?)
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here
        //int64_t iProcPtr = static_cast<int64_t>((row * iMul - (iKernelSize / 2)) * iOutWidth) + (col * iMul - (iKernelSize / 2));
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);
        int64_t iInpPtrRowStart = row * iWidthEl;

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
        {
            unsigned char ucInpSample = g_pElImageBuffer[(iInpPtrRowStart + col)];

            // fast skip zero
            if (ucInpSample == 0) continue;

            float* pfCurrKernel = g_pfKernelWeighted + static_cast<int64_t>(ucInpSample) * iKernelSize * iKernelSize;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                // add full kernel row to output - AVX2
                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
                    _mm256_storeu_ps(pfProc + k_col, _mm256_add_ps(_mm256_loadu_ps(pfCurrKernel + k_col), _mm256_loadu_ps(pfProc + k_col)));
                }
                
                // need to process last up to 7 floats separately..
                for (int64_t k_col = k_col8; k_col < iKernelSize; ++k_col)
                    pfProc[k_col] += pfCurrKernel[k_col];

                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel += iKernelSize; // point to next kernel row now
            } // k_row
        } // col
    }
}
