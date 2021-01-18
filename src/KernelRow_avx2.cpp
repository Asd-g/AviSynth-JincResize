#include "JincRessize.h"
/*
#if !defined(__AVX2__)
#error "AVX2 option needed"
#endif
*/ // VS2019 for unknown reason can not see __AVX2__ 

void JincResize::KernelRow_avx2(int64_t iOutWidth)
{
    const int k_col8 = iKernelSize - (iKernelSize % 8);

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here
        //int64_t iProcPtr = static_cast<int64_t>((row * iMul - (iKernelSize / 2)) * iOutWidth) + (col * iMul - (iKernelSize / 2));
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);
        int64_t iInpPtrRowStart = row * iWidthEl;

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
        {
            unsigned char ucInpSample = g_pElImageBuffer[(iInpPtrRowStart + col)];

            float* pfCurrKernel = g_pfKernelWeighted + static_cast<int64_t>(ucInpSample) * iKernelSize * iKernelSize;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {

                // add full kernel row to output - AVX2
                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
                    _mm256_storeu_ps(pfProc + k_col, _mm256_add_ps(*(__m256*)(pfCurrKernel + k_col), *(__m256*)(pfProc + k_col)));
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


void JincResize::KernelRow_avx2_mul(int64_t iOutWidth)
{
    const int k_col8 = iKernelSize - (iKernelSize % 8);


    float *pfCurrKernel = g_pfKernel; 

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);
        int64_t iInpPtrRowStart = row * iWidthEl;

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
        {

            __declspec(align(32)) float fSample = (float)g_pElImageBuffer[(iInpPtrRowStart + col)];

            const __m256 inp256 = _mm256_broadcast_ss(&fSample);

            float *pfCurrKernel_pos = pfCurrKernel;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;


            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {

                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
                    _mm256_storeu_ps(pfProc + k_col, _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), inp256, *(__m256*)(pfProc + k_col)));
                }
                
                // need to process last (up to 7) floats separately..
                for (int64_t k_col = k_col8; k_col < iKernelSize; ++k_col)
                {
                    pfProc[k_col] += (pfCurrKernel_pos[k_col] * (float)g_pElImageBuffer[(iInpPtrRowStart + col)]);
                }

                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize; // point to next kernel row now
             } // k_row
        } // col
    }
}

/*Better implementation of Mul2 will be with AVX512 instructions with shift between zmm registers instead of memory referencing with FMA*/
void JincResize::KernelRow_avx2_mul2_taps8(int64_t iOutWidth)
{
    float* pfCurrKernel = g_pfKernelWeighted + iKernelSize*iKernelSize; // needs unfinished g_pfKernelParProc with shifted or padded with zeroes kernel rows

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here 
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);
        int64_t iInpPtrRowStart = row * iWidthEl;

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col += 4) // input cols counter
        {
            __declspec(align(32)) float fSample = (float)g_pElImageBuffer[(iInpPtrRowStart + col)];
            __declspec(align(32)) float fSample2 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 1)];
            __declspec(align(32)) float fSample3 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 2)];
            __declspec(align(32)) float fSample4 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 3)];

            __m256 inp256 = _mm256_broadcast_ss(&fSample);
            __m256 inp256_2 = _mm256_broadcast_ss(&fSample2);
            __m256 inp256_3 = _mm256_broadcast_ss(&fSample3);
            __m256 inp256_4 = _mm256_broadcast_ss(&fSample4);

            float* pfCurrKernel_pos = pfCurrKernel;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4;

                my_ymm0 = _mm256_load_ps(pfProc);
                my_ymm1 = _mm256_load_ps(pfProc + 8);
                my_ymm2 = _mm256_load_ps(pfProc + 16);
                my_ymm3 = _mm256_load_ps(pfProc + 24);
                my_ymm4 = _mm256_load_ps(pfProc + 32);

                my_ymm0 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos), inp256, my_ymm0);  
                my_ymm1 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 8), inp256, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 16), inp256, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 24), inp256, my_ymm3);

                my_ymm0 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40), inp256_2, my_ymm0);
                my_ymm1 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 + 8), inp256_2, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 + 16), inp256_2, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 + 24), inp256_2, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 + 32), inp256_2, my_ymm4);

                my_ymm0 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 * 2), inp256_3, my_ymm0);
                my_ymm1 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 * 2 + 8), inp256_3, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 * 2 + 16), inp256_3, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 * 2 + 24), inp256_3, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 * 2 + 32), inp256_3, my_ymm4);

                my_ymm0 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 * 3), inp256_4, my_ymm0);
                my_ymm1 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 * 3 + 8), inp256_4, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 * 3 + 16), inp256_4, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 * 3 + 24), inp256_4, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + 40 * 3 + 32), inp256_4, my_ymm4);

                _mm256_store_ps(pfProc, my_ymm0);
                _mm256_store_ps(pfProc + 8, my_ymm1);
                _mm256_store_ps(pfProc + 16, my_ymm2);
                _mm256_store_ps(pfProc + 24, my_ymm3);
                _mm256_store_ps(pfProc + 32, my_ymm4);
                    
                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize;// *iParProc; // point to next kernel row now
            } // k_row
        } // col
    }
}


void JincResize::KernelRow_avx2_mul8_taps3(int64_t iOutWidth)
{
    float* pfCurrKernel = g_pfKernel;

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here 
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);
        int64_t iInpPtrRowStart = row * iWidthEl;

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col += 4) // input cols counter
        {
            __declspec(align(32)) float fSample = (float)g_pElImageBuffer[(iInpPtrRowStart + col)];
            __declspec(align(32)) float fSample2 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 1)];
            __declspec(align(32)) float fSample3 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 2)];
            __declspec(align(32)) float fSample4 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 3)];

            float* pfCurrKernel_pos = pfCurrKernel;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

             for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
               __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4, my_ymm5, my_ymm6, my_ymm7, my_ymm8; // out samples
               __m256 my_ymm9, my_ymm10, my_ymm11, my_ymm12, my_ymm13, my_ymm14; // kernel samples

               __m256 inp256 = _mm256_broadcast_ss(&fSample);

                my_ymm0 = _mm256_load_ps(pfProc);
                my_ymm1 = _mm256_load_ps(pfProc + 8);
                my_ymm2 = _mm256_load_ps(pfProc + 16);
                my_ymm3 = _mm256_load_ps(pfProc + 24);
                my_ymm4 = _mm256_load_ps(pfProc + 32);
                my_ymm5 = _mm256_load_ps(pfProc + 40);
                my_ymm6 = _mm256_load_ps(pfProc + 48);
                my_ymm7 = _mm256_load_ps(pfProc + 56);
                my_ymm8 = _mm256_load_ps(pfProc + 64);

                my_ymm9 = _mm256_load_ps(pfCurrKernel_pos);
                my_ymm10 = _mm256_load_ps(pfCurrKernel_pos + 8);
                my_ymm11 = _mm256_load_ps(pfCurrKernel_pos + 16);
                my_ymm12 = _mm256_load_ps(pfCurrKernel_pos + 24);
                my_ymm13 = _mm256_load_ps(pfCurrKernel_pos + 32);
                my_ymm14 = _mm256_load_ps(pfCurrKernel_pos + 40);
// 1st sample
                my_ymm0 = _mm256_fmadd_ps(my_ymm9, inp256, my_ymm0);
                my_ymm1 = _mm256_fmadd_ps(my_ymm10, inp256, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(my_ymm11, inp256, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm12, inp256, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm13, inp256, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm14, inp256, my_ymm5);
// 2nd sample
                inp256 = _mm256_broadcast_ss(&fSample2);
                my_ymm1 = _mm256_fmadd_ps(my_ymm9, inp256, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(my_ymm10, inp256, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm11, inp256, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm12, inp256, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm13, inp256, my_ymm5);
                my_ymm6 = _mm256_fmadd_ps(my_ymm14, inp256, my_ymm6);
// 3rd sample
                inp256 = _mm256_broadcast_ss(&fSample3);
                my_ymm2 = _mm256_fmadd_ps(my_ymm9, inp256, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm10, inp256, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm11, inp256, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm12, inp256, my_ymm5);
                my_ymm6 = _mm256_fmadd_ps(my_ymm13, inp256, my_ymm6);
                my_ymm7 = _mm256_fmadd_ps(my_ymm14, inp256, my_ymm7);
// 4th sample
                inp256 = _mm256_broadcast_ss(&fSample4);
                my_ymm3 = _mm256_fmadd_ps(my_ymm9, inp256, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm10, inp256, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm11, inp256, my_ymm5);
                my_ymm6 = _mm256_fmadd_ps(my_ymm12, inp256, my_ymm6);
                my_ymm7 = _mm256_fmadd_ps(my_ymm13, inp256, my_ymm7);
                my_ymm8 = _mm256_fmadd_ps(my_ymm14, inp256, my_ymm8);

                _mm256_store_ps(pfProc, my_ymm0);
                _mm256_store_ps(pfProc + 8, my_ymm1);
                _mm256_store_ps(pfProc + 16, my_ymm2);
                _mm256_store_ps(pfProc + 24, my_ymm3);
                _mm256_store_ps(pfProc + 32, my_ymm4);
                _mm256_store_ps(pfProc + 40, my_ymm5);
                _mm256_store_ps(pfProc + 48, my_ymm6);
                _mm256_store_ps(pfProc + 56, my_ymm7);
                _mm256_store_ps(pfProc + 64, my_ymm8);

                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize; // point to next kernel row now
            } // k_row
        } // col
    }
}

void JincResize::KernelRow_avx2_mul4_taps4(int64_t iOutWidth)
{
    float* pfCurrKernel = g_pfKernel;

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here 
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);
        int64_t iInpPtrRowStart = row * iWidthEl;

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col += 8) // input cols counter
        {
            __declspec(align(32)) float fSample = (float)g_pElImageBuffer[(iInpPtrRowStart + col)];
            __declspec(align(32)) float fSample2 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 1)];
            __declspec(align(32)) float fSample3 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 2)];
            __declspec(align(32)) float fSample4 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 3)];

            __declspec(align(32)) float fSample5 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 4)];
            __declspec(align(32)) float fSample6 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 5)];
            __declspec(align(32)) float fSample7 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 6)];
            __declspec(align(32)) float fSample8 = (float)g_pElImageBuffer[(iInpPtrRowStart + col + 7)];

            float* pfCurrKernel_pos = pfCurrKernel;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4, my_ymm5, my_ymm6, my_ymm7; // out samples
                __m256 my_ymm8, my_ymm9, my_ymm10, my_ymm11; // inp samples
                __m256 my_ymm12, my_ymm13, my_ymm14, my_ymm15; // kernel samples

                my_ymm12 = _mm256_load_ps(pfCurrKernel_pos);
                my_ymm13 = _mm256_load_ps(pfCurrKernel_pos + 8);
                my_ymm14 = _mm256_load_ps(pfCurrKernel_pos + 16);
                my_ymm15 = _mm256_load_ps(pfCurrKernel_pos + 24);

                my_ymm0 = _mm256_load_ps(pfProc);
                my_ymm1 = _mm256_load_ps(pfProc + 8);
                my_ymm2 = _mm256_load_ps(pfProc + 16);
                my_ymm3 = _mm256_load_ps(pfProc + 24);
                my_ymm4 = _mm256_load_ps(pfProc + 32);
                my_ymm5 = _mm256_load_ps(pfProc + 40);
                my_ymm6 = _mm256_load_ps(pfProc + 48);

                my_ymm8 = _mm256_broadcast_ss(&fSample); // 1
                my_ymm9 = _mm256_broadcast_ss(&fSample3); // 3
                my_ymm10 = _mm256_broadcast_ss(&fSample5); // 5
                my_ymm11 = _mm256_broadcast_ss(&fSample7); // 7            

                 // 1st sample
                my_ymm0 = _mm256_fmadd_ps(my_ymm12, my_ymm8, my_ymm0);
                my_ymm1 = _mm256_fmadd_ps(my_ymm13, my_ymm8, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(my_ymm14, my_ymm8, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm15, my_ymm8, my_ymm3);

                // 3rd sample
                my_ymm1 = _mm256_fmadd_ps(my_ymm12, my_ymm9, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(my_ymm13, my_ymm9, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm14, my_ymm9, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm15, my_ymm9, my_ymm4);

                // 5th sample
                my_ymm2 = _mm256_fmadd_ps(my_ymm12, my_ymm10, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm13, my_ymm10, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm14, my_ymm10, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm15, my_ymm10, my_ymm5);


                // 7th sample
                my_ymm3 = _mm256_fmadd_ps(my_ymm12, my_ymm11, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm13, my_ymm11, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm14, my_ymm11, my_ymm5);
                my_ymm6 = _mm256_fmadd_ps(my_ymm15, my_ymm11, my_ymm6);


                _mm_store_ps(pfProc, _mm256_castps256_ps128(my_ymm0));
                my_ymm0 = _mm256_permute2f128_ps(my_ymm0, my_ymm1, 33);
                my_ymm1 = _mm256_permute2f128_ps(my_ymm1, my_ymm2, 33);
                my_ymm2 = _mm256_permute2f128_ps(my_ymm2, my_ymm3, 33);
                my_ymm3 = _mm256_permute2f128_ps(my_ymm3, my_ymm4, 33);
                my_ymm4 = _mm256_permute2f128_ps(my_ymm4, my_ymm5, 33);
                my_ymm5 = _mm256_permute2f128_ps(my_ymm5, my_ymm6, 33);
                my_ymm6 = _mm256_permute2f128_ps(my_ymm6, my_ymm6, 49);
                my_ymm6 = _mm256_insertf128_ps(my_ymm6, *(__m128*)(pfProc + 56), 1);

                // even samples
                my_ymm8 = _mm256_broadcast_ss(&fSample2); // 2
                my_ymm9 = _mm256_broadcast_ss(&fSample4); // 4
                my_ymm10 = _mm256_broadcast_ss(&fSample6); // 6
                my_ymm11 = _mm256_broadcast_ss(&fSample8); // 8            

                 // 2nd sample
                my_ymm0 = _mm256_fmadd_ps(my_ymm12, my_ymm8, my_ymm0);
                my_ymm1 = _mm256_fmadd_ps(my_ymm13, my_ymm8, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(my_ymm14, my_ymm8, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm15, my_ymm8, my_ymm3);

                // 4th sample
                my_ymm1 = _mm256_fmadd_ps(my_ymm12, my_ymm9, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(my_ymm13, my_ymm9, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm14, my_ymm9, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm15, my_ymm9, my_ymm4);

                // 6th sample
                my_ymm2 = _mm256_fmadd_ps(my_ymm12, my_ymm10, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm13, my_ymm10, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm14, my_ymm10, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm15, my_ymm10, my_ymm5);

                // 7th sample
                my_ymm3 = _mm256_fmadd_ps(my_ymm12, my_ymm11, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm13, my_ymm11, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm14, my_ymm11, my_ymm5);
                my_ymm6 = _mm256_fmadd_ps(my_ymm15, my_ymm11, my_ymm6);

                _mm256_store_ps(pfProc + 4, my_ymm0);
                _mm256_store_ps(pfProc + 12, my_ymm1);
                _mm256_store_ps(pfProc + 20, my_ymm2);
                _mm256_store_ps(pfProc + 28, my_ymm3);
                _mm256_store_ps(pfProc + 36, my_ymm4);
                _mm256_store_ps(pfProc + 44, my_ymm5);
                _mm256_store_ps(pfProc + 52, my_ymm6);
 
                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize;// *iParProc; // point to next kernel row now
            } // k_row
        } // col
    }
}

void JincResize::KernelRow_avx2_mul4_taps4_fr(int64_t iOutWidth)
{
    float* pfCurrKernel = g_pfKernel;

    /* iMul=4, iTaps=4, KernelSize (row length in floats) is 32*/
    /* full row-walking, fixed at 19.01.2021 and need to be tested */

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here 
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);
        int64_t iInpPtrRowStart = row * iWidthEl;

        for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
        {

            float* pfCurrKernel_pos = pfCurrKernel + iKernelSize * k_row;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + iTaps * iMul + k_row * iOutWidth;

            __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4, my_ymm5, my_ymm6, my_ymm7; // out samples
            __m256 my_ymm8, my_ymm9, my_ymm10, my_ymm11; // inp samples
            __m256 my_ymm12, my_ymm13, my_ymm14, my_ymm15; // kernel samples

            my_ymm8 = _mm256_setzero_ps();
            my_ymm9 = _mm256_setzero_ps();
            my_ymm10 = _mm256_setzero_ps();
            my_ymm11 = _mm256_setzero_ps();

            my_ymm12 = _mm256_load_ps(pfCurrKernel_pos);
            my_ymm13 = _mm256_load_ps(pfCurrKernel_pos + 8);
            my_ymm14 = _mm256_load_ps(pfCurrKernel_pos + 16);
            my_ymm15 = _mm256_load_ps(pfCurrKernel_pos + 24);

            my_ymm0 = _mm256_load_ps(pfProc);
            my_ymm1 = _mm256_load_ps(pfProc + 8);
            my_ymm2 = _mm256_load_ps(pfProc + 16);
            my_ymm3 = _mm256_load_ps(pfProc + 24);
            my_ymm4 = _mm256_load_ps(pfProc + 32);
            my_ymm5 = _mm256_load_ps(pfProc + 40);
            my_ymm6 = _mm256_load_ps(pfProc + 48);

            for (int64_t col = iTaps; col < iWidthEl - iTaps; col += 8) // input cols counter
            {
                // odd samples
                my_ymm8 = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm8), g_pElImageBuffer[(iInpPtrRowStart + col)])); // 1
                my_ymm9 = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm9), g_pElImageBuffer[(iInpPtrRowStart + col + 2)])); // 3
                my_ymm10 = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm10), g_pElImageBuffer[(iInpPtrRowStart + col + 4)])); // 5
                my_ymm11 = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm11), g_pElImageBuffer[(iInpPtrRowStart + col + 6)])); // 7            

                 // 1st sample
                my_ymm0 = _mm256_fmadd_ps(my_ymm12, my_ymm8, my_ymm0);
                my_ymm1 = _mm256_fmadd_ps(my_ymm13, my_ymm8, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(my_ymm14, my_ymm8, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm15, my_ymm8, my_ymm3);

                // 3rd sample
                my_ymm1 = _mm256_fmadd_ps(my_ymm12, my_ymm9, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(my_ymm13, my_ymm9, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm14, my_ymm9, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm15, my_ymm9, my_ymm4);

                // 5th sample
                my_ymm2 = _mm256_fmadd_ps(my_ymm12, my_ymm10, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm13, my_ymm10, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm14, my_ymm10, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm15, my_ymm10, my_ymm5);


                // 7th sample
                my_ymm3 = _mm256_fmadd_ps(my_ymm12, my_ymm11, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm13, my_ymm11, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm14, my_ymm11, my_ymm5);
                my_ymm6 = _mm256_fmadd_ps(my_ymm15, my_ymm11, my_ymm6);


                _mm_store_ps(pfProc, _mm256_castps256_ps128(my_ymm0));
                my_ymm0 = _mm256_permute2f128_ps(my_ymm0, my_ymm1, 33);
                my_ymm1 = _mm256_permute2f128_ps(my_ymm1, my_ymm2, 33);
                my_ymm2 = _mm256_permute2f128_ps(my_ymm2, my_ymm3, 33);
                my_ymm3 = _mm256_permute2f128_ps(my_ymm3, my_ymm4, 33);
                my_ymm4 = _mm256_permute2f128_ps(my_ymm4, my_ymm5, 33);
                my_ymm5 = _mm256_permute2f128_ps(my_ymm5, my_ymm6, 33);
                my_ymm6 = _mm256_permute2f128_ps(my_ymm6, my_ymm6, 49);
                my_ymm6 = _mm256_insertf128_ps(my_ymm6, *(__m128*)(pfProc + 56), 1);

                // even samples
                my_ymm8 = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm8), g_pElImageBuffer[(iInpPtrRowStart + col + 1)])); // 2
                my_ymm9 = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm9), g_pElImageBuffer[(iInpPtrRowStart + col + 3)])); // 4
                my_ymm10 = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm10), g_pElImageBuffer[(iInpPtrRowStart + col + 5)])); // 6
                my_ymm11 = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm11), g_pElImageBuffer[(iInpPtrRowStart + col + 7)])); // 8            

                 // 2nd sample
                my_ymm0 = _mm256_fmadd_ps(my_ymm12, my_ymm8, my_ymm0);
                my_ymm1 = _mm256_fmadd_ps(my_ymm13, my_ymm8, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(my_ymm14, my_ymm8, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm15, my_ymm8, my_ymm3);

                // 4th sample
                my_ymm1 = _mm256_fmadd_ps(my_ymm12, my_ymm9, my_ymm1);
                my_ymm2 = _mm256_fmadd_ps(my_ymm13, my_ymm9, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm14, my_ymm9, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm15, my_ymm9, my_ymm4);

                // 6th sample
                my_ymm2 = _mm256_fmadd_ps(my_ymm12, my_ymm10, my_ymm2);
                my_ymm3 = _mm256_fmadd_ps(my_ymm13, my_ymm10, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm14, my_ymm10, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm15, my_ymm10, my_ymm5);

                // 7th sample
                my_ymm3 = _mm256_fmadd_ps(my_ymm12, my_ymm11, my_ymm3);
                my_ymm4 = _mm256_fmadd_ps(my_ymm13, my_ymm11, my_ymm4);
                my_ymm5 = _mm256_fmadd_ps(my_ymm14, my_ymm11, my_ymm5);
                my_ymm6 = _mm256_fmadd_ps(my_ymm15, my_ymm11, my_ymm6);

                _mm256_store_ps(pfProc + 4, my_ymm0);
                _mm256_store_ps(pfProc + 12, my_ymm1);
                _mm256_store_ps(pfProc + 20, my_ymm2);
                _mm_store_ps(pfProc + 28, _mm256_castps256_ps128(my_ymm3));

                pfProc += 8*iMul;

                my_ymm0 = my_ymm4;
                my_ymm1 = my_ymm5;
                my_ymm2 = my_ymm6;

                my_ymm3 = _mm256_load_ps(pfProc + 24); // may be read out of g_pfFilteredImageBuffer at last col - may be need of pad buffer or other fix
                my_ymm4 = _mm256_load_ps(pfProc + 32);
                my_ymm5 = _mm256_load_ps(pfProc + 40);
                my_ymm6 = _mm256_load_ps(pfProc + 48);
                
            } // col
            // need to process last up to 7 col..
        } // k_row
    } // row
}

void JincResize::ConvertToInt_avx2(int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{
    const int col32 = (iInpWidth*iMul) - ((iInpWidth*iMul) % 32);

    __m256 my_0d5f_ymm0, my_256f_ymm1, my_zero_ymm2;

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = 0; row < iInpHeight * iMul; row++)
    {
        my_zero_ymm2 = _mm256_setzero_ps();

        int64_t col = 0;
        float* pfProc = g_pfFilteredImageBuffer + (row + iKernelSize * iMul) * iWidthEl * iMul +col + iKernelSize * iMul;
        unsigned char* pucDst = dst + row * iDstStride;

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

           __m256i my_iVal_ymm1234 = _mm256_permutevar8x32_epi32(my_iVal_1234, _mm256_setr_epi32(0,4, 1,5, 2,6, 3,7));

           _mm256_storeu_si256((__m256i*)(pucDst), my_iVal_ymm1234);

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
        }
    }


}