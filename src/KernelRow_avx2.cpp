#include "JincRessize.h"
/*
#if !defined(__AVX2__)
#error "AVX2 option needed"
#endif
*/ // VS2019 for unknown reason can not see __AVX2__ 

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
//            if (ucInpSample == 0) continue;

            float* pfCurrKernel = g_pfKernelWeighted + static_cast<int64_t>(ucInpSample) * iKernelSize * iKernelSize;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {

                // add full kernel row to output - AVX2
                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
 //                   _mm256_storeu_ps(pfProc + k_col, _mm256_add_ps(_mm256_loadu_ps(pfCurrKernel + k_col), _mm256_loadu_ps(pfProc + k_col)));
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
                    _mm256_storeu_ps(pfProc + k_col, _mm256_add_ps(_mm256_mul_ps(*(__m256*)(pfCurrKernel_pos + k_col), inp256), *(__m256*)(pfProc + k_col)));
           
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

void JincResize::KernelRow_avx2_mul2_taps8(int64_t iOutWidth)
{
    float* pfCurrKernel = g_pfKernelWeighted; // needs unfinished g_pfKernelParProc with shifted kernel rows

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

            // some quick and bad attempt to align32
            pfProc = (float*)((unsigned char*)pfProc - ((int64_t)pfProc % 32));

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4;

/* If we can somehow shift long loaded kernel row in a number of ymm registers faster in compare with 4-cycles L1 cache access I think we can get more GFlops,
but for now I do not know how to shift, may be with shuffle/blend ? 
The only cheapest 'shift' is available for iMul=8 because of just renaming virtual-registers.*/
                    my_ymm0 = _mm256_load_ps(pfProc);
                    my_ymm1 = _mm256_load_ps(pfProc + 8);
                    my_ymm2 = _mm256_load_ps(pfProc + 16);
                    my_ymm3 = _mm256_load_ps(pfProc + 24);
                    my_ymm4 = _mm256_load_ps(pfProc + 32);

                    my_ymm0 = _mm256_fmadd_ps(my_ymm0, *(__m256*)(pfCurrKernel_pos), inp256);
                    my_ymm1 = _mm256_fmadd_ps(my_ymm1, *(__m256*)(pfCurrKernel_pos + 8), inp256);
                    my_ymm2 = _mm256_fmadd_ps(my_ymm2, *(__m256*)(pfCurrKernel_pos + 16), inp256);
                    my_ymm3 = _mm256_fmadd_ps(my_ymm3, *(__m256*)(pfCurrKernel_pos + 24), inp256);

                    my_ymm0 = _mm256_fmadd_ps(my_ymm0, *(__m256*)(pfCurrKernel_pos + 40), inp256_2);
                    my_ymm1 = _mm256_fmadd_ps(my_ymm1, *(__m256*)(pfCurrKernel_pos + 40 + 8), inp256_2);
                    my_ymm2 = _mm256_fmadd_ps(my_ymm2, *(__m256*)(pfCurrKernel_pos + 40 + 16), inp256_2);
                    my_ymm3 = _mm256_fmadd_ps(my_ymm3, *(__m256*)(pfCurrKernel_pos + 40 + 24), inp256_2);
                    my_ymm4 = _mm256_fmadd_ps(my_ymm4, *(__m256*)(pfCurrKernel_pos + 40 + 32), inp256_2);

                    my_ymm0 = _mm256_fmadd_ps(my_ymm0, *(__m256*)(pfCurrKernel_pos + 40 * 2), inp256_3);
                    my_ymm1 = _mm256_fmadd_ps(my_ymm1, *(__m256*)(pfCurrKernel_pos + 40 * 2 + 8), inp256_3);
                    my_ymm2 = _mm256_fmadd_ps(my_ymm2, *(__m256*)(pfCurrKernel_pos + 40 * 2 + 16), inp256_3);
                    my_ymm3 = _mm256_fmadd_ps(my_ymm3, *(__m256*)(pfCurrKernel_pos + 40 * 2 + 24), inp256_3);
                    my_ymm4 = _mm256_fmadd_ps(my_ymm4, *(__m256*)(pfCurrKernel_pos + 40 * 2 + 32), inp256_3);

                    my_ymm0 = _mm256_fmadd_ps(my_ymm0, *(__m256*)(pfCurrKernel_pos + 40 * 3), inp256_4);
                    my_ymm1 = _mm256_fmadd_ps(my_ymm1, *(__m256*)(pfCurrKernel_pos + 40 * 3 + 8), inp256_4);
                    my_ymm2 = _mm256_fmadd_ps(my_ymm2, *(__m256*)(pfCurrKernel_pos + 40 * 3 + 16), inp256_4);
                    my_ymm3 = _mm256_fmadd_ps(my_ymm3, *(__m256*)(pfCurrKernel_pos + 40 * 3 + 24), inp256_4);
                    my_ymm4 = _mm256_fmadd_ps(my_ymm4, *(__m256*)(pfCurrKernel_pos + 40 * 3 + 32), inp256_4);

                    _mm256_store_ps(pfProc, my_ymm0);
                    _mm256_store_ps(pfProc + 8, my_ymm1);
                    _mm256_store_ps(pfProc + 16, my_ymm2);
                    _mm256_store_ps(pfProc + 24, my_ymm3);
                    _mm256_store_ps(pfProc + 32, my_ymm4);
                    
                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize*iParProc; // point to next kernel row now
            } // k_row
        } // col
    }
}

void JincResize::KernelRow_avx2_mul8_taps2(int64_t iOutWidth)
{
    float* pfCurrKernel = g_pfKernelWeighted;

    /* iMul=8, iTaps=2, KernelSize (row length in floats) is 32*/

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

            // some quick and bad attempt to align32
            pfProc = (float*)((unsigned char*)pfProc - ((int64_t)pfProc % 32));

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4, my_ymm5, my_ymm6; // out samples
                __m256 my_ymm7, my_ymm8, my_ymm9, my_ymm10; // kernel samples

                my_ymm0 = _mm256_load_ps(pfProc);
                my_ymm1 = _mm256_load_ps(pfProc + 8);
                my_ymm2 = _mm256_load_ps(pfProc + 16);
                my_ymm3 = _mm256_load_ps(pfProc + 24);
                my_ymm4 = _mm256_load_ps(pfProc + 32);
                my_ymm5 = _mm256_load_ps(pfProc + 40);
                my_ymm6 = _mm256_load_ps(pfProc + 48);

                my_ymm7 = _mm256_load_ps(pfCurrKernel_pos);
                my_ymm8 = _mm256_load_ps(pfCurrKernel_pos + 8);
                my_ymm9 = _mm256_load_ps(pfCurrKernel_pos + 16);
                my_ymm10 = _mm256_load_ps(pfCurrKernel_pos + 24);

                my_ymm0 = _mm256_fmadd_ps(my_ymm0, my_ymm7, inp256);
                my_ymm1 = _mm256_fmadd_ps(my_ymm1, my_ymm8, inp256);
                my_ymm2 = _mm256_fmadd_ps(my_ymm2, my_ymm9, inp256);
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm10, inp256);

                //unfinished


                _mm256_store_ps(pfProc, my_ymm0);
                _mm256_store_ps(pfProc + 8, my_ymm1);
                _mm256_store_ps(pfProc + 16, my_ymm2);
                _mm256_store_ps(pfProc + 24, my_ymm3);
                _mm256_store_ps(pfProc + 32, my_ymm4);
                //         _mm256_storeu_ps(pfProc + k_col, _mm256_add_ps(_mm256_mul_ps(_mm256_loadu_ps(pfCurrKernel_pos + k_col), inp256), _mm256_loadu_ps(pfProc + k_col)));

                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize * iParProc; // point to next kernel row now
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

            // some quick and bad attempt to align32
            pfProc = (float*)((unsigned char*)pfProc - ((int64_t)pfProc % 32));

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
               __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4, my_ymm5, my_ymm6, my_ymm7, my_ymm8; // out samples
               __m256 my_ymm9, my_ymm10, my_ymm11, my_ymm12, my_ymm13, my_ymm14; // kernel samples
 //              register __m256 my_ymm15; //inp
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
                my_ymm0 = _mm256_fmadd_ps(my_ymm0, my_ymm9, inp256);
                my_ymm1 = _mm256_fmadd_ps(my_ymm1, my_ymm10, inp256);
                my_ymm2 = _mm256_fmadd_ps(my_ymm2, my_ymm11, inp256);
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm12, inp256);
                my_ymm4 = _mm256_fmadd_ps(my_ymm4, my_ymm13, inp256);
                my_ymm5 = _mm256_fmadd_ps(my_ymm5, my_ymm14, inp256);
// 2nd sample
                inp256 = _mm256_broadcast_ss(&fSample2);
                my_ymm1 = _mm256_fmadd_ps(my_ymm1, my_ymm9, inp256);
                my_ymm2 = _mm256_fmadd_ps(my_ymm2, my_ymm10, inp256);
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm11, inp256);
                my_ymm4 = _mm256_fmadd_ps(my_ymm4, my_ymm12, inp256);
                my_ymm5 = _mm256_fmadd_ps(my_ymm5, my_ymm13, inp256);
                my_ymm6 = _mm256_fmadd_ps(my_ymm6, my_ymm14, inp256);
// 3rd sample
                inp256 = _mm256_broadcast_ss(&fSample3);
                my_ymm2 = _mm256_fmadd_ps(my_ymm2, my_ymm9, inp256);
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm10, inp256);
                my_ymm4 = _mm256_fmadd_ps(my_ymm4, my_ymm11, inp256);
                my_ymm5 = _mm256_fmadd_ps(my_ymm5, my_ymm12, inp256);
                my_ymm6 = _mm256_fmadd_ps(my_ymm6, my_ymm13, inp256);
                my_ymm7 = _mm256_fmadd_ps(my_ymm7, my_ymm14, inp256);
// 4th sample
                inp256 = _mm256_broadcast_ss(&fSample4);
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm9, inp256);
                my_ymm4 = _mm256_fmadd_ps(my_ymm4, my_ymm10, inp256);
                my_ymm5 = _mm256_fmadd_ps(my_ymm5, my_ymm11, inp256);
                my_ymm6 = _mm256_fmadd_ps(my_ymm6, my_ymm12, inp256);
                my_ymm7 = _mm256_fmadd_ps(my_ymm7, my_ymm13, inp256);
                my_ymm8 = _mm256_fmadd_ps(my_ymm8, my_ymm14, inp256);

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
                pfCurrKernel_pos += iKernelSize;// *iParProc; // point to next kernel row now
            } // k_row
        } // col
    }
}

void JincResize::KernelRow_avx2_mul4_taps4(int64_t iOutWidth)
{
    float* pfCurrKernel = g_pfKernel;

    /* iMul=4, iTaps=4, KernelSize (row length in floats) is 32*/

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

            // some quick and bad attempt to align32
            pfProc = (float*)((unsigned char*)pfProc - ((int64_t)pfProc % 32));

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4, my_ymm5, my_ymm6, my_ymm7; // out samples
                __m256 my_ymm8, my_ymm9, my_ymm10, my_ymm11; // inp samples
                __m256 my_ymm12, my_ymm13, my_ymm14, my_ymm15; // kernel samples
 
                // odd samples
                my_ymm8 = _mm256_broadcast_ss(&fSample);
                my_ymm9 = _mm256_broadcast_ss(&fSample3);
                my_ymm10 = _mm256_broadcast_ss(&fSample5);
                my_ymm11 = _mm256_broadcast_ss(&fSample7);

                my_ymm0 = _mm256_load_ps(pfProc);
                my_ymm1 = _mm256_load_ps(pfProc + 8);
                my_ymm2 = _mm256_load_ps(pfProc + 16);
                my_ymm3 = _mm256_load_ps(pfProc + 24);
                my_ymm4 = _mm256_load_ps(pfProc + 32);
                my_ymm5 = _mm256_load_ps(pfProc + 40);
                my_ymm6 = _mm256_load_ps(pfProc + 48);

                my_ymm12 = _mm256_load_ps(pfCurrKernel_pos);
                my_ymm13 = _mm256_load_ps(pfCurrKernel_pos + 8);
                my_ymm14 = _mm256_load_ps(pfCurrKernel_pos + 16);
                my_ymm15 = _mm256_load_ps(pfCurrKernel_pos + 24);

                // 1st sample
                my_ymm0 = _mm256_fmadd_ps(my_ymm0, my_ymm12, my_ymm8);
                my_ymm1 = _mm256_fmadd_ps(my_ymm1, my_ymm13, my_ymm8);
                my_ymm2 = _mm256_fmadd_ps(my_ymm2, my_ymm14, my_ymm8);
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm15, my_ymm8);

                // 3rd sample
                my_ymm1 = _mm256_fmadd_ps(my_ymm1, my_ymm12, my_ymm9);
                my_ymm2 = _mm256_fmadd_ps(my_ymm2, my_ymm13, my_ymm9);
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm14, my_ymm9);
                my_ymm4 = _mm256_fmadd_ps(my_ymm4, my_ymm15, my_ymm9);

                // 5th sample
                my_ymm2 = _mm256_fmadd_ps(my_ymm2, my_ymm12, my_ymm10);
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm13, my_ymm10);
                my_ymm4 = _mm256_fmadd_ps(my_ymm4, my_ymm14, my_ymm10);
                my_ymm5 = _mm256_fmadd_ps(my_ymm5, my_ymm15, my_ymm10);

                // 7th sample
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm12, my_ymm11);
                my_ymm4 = _mm256_fmadd_ps(my_ymm4, my_ymm13, my_ymm11);
                my_ymm5 = _mm256_fmadd_ps(my_ymm5, my_ymm14, my_ymm11);
                my_ymm6 = _mm256_fmadd_ps(my_ymm6, my_ymm15, my_ymm11);

                _mm256_store_ps(pfProc, my_ymm0);
                _mm256_store_ps(pfProc + 8, my_ymm1);
                _mm256_store_ps(pfProc + 16, my_ymm2);
                _mm256_store_ps(pfProc + 24, my_ymm3);
                _mm256_store_ps(pfProc + 32, my_ymm4);
                _mm256_store_ps(pfProc + 40, my_ymm5);
                _mm256_store_ps(pfProc + 48, my_ymm6);

                // even samples
                my_ymm8 = _mm256_broadcast_ss(&fSample2);
                my_ymm9 = _mm256_broadcast_ss(&fSample4);
                my_ymm10 = _mm256_broadcast_ss(&fSample6);
                my_ymm11 = _mm256_broadcast_ss(&fSample8);

                my_ymm0 = _mm256_load_ps(pfProc + 4);
                my_ymm1 = _mm256_load_ps(pfProc + 8 + 4);
                my_ymm2 = _mm256_load_ps(pfProc + 16 + 4);
                my_ymm3 = _mm256_load_ps(pfProc + 24 + 4);
                my_ymm4 = _mm256_load_ps(pfProc + 32 + 4);
                my_ymm5 = _mm256_load_ps(pfProc + 40 + 4);
                my_ymm6 = _mm256_load_ps(pfProc + 48 + 4);

                // 2nd sample
                my_ymm0 = _mm256_fmadd_ps(my_ymm0, my_ymm12, my_ymm8);
                my_ymm1 = _mm256_fmadd_ps(my_ymm1, my_ymm13, my_ymm8);
                my_ymm2 = _mm256_fmadd_ps(my_ymm2, my_ymm14, my_ymm8);
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm15, my_ymm8);

                // 4th sample
                my_ymm1 = _mm256_fmadd_ps(my_ymm1, my_ymm12, my_ymm9);
                my_ymm2 = _mm256_fmadd_ps(my_ymm2, my_ymm13, my_ymm9);
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm14, my_ymm9);
                my_ymm4 = _mm256_fmadd_ps(my_ymm4, my_ymm15, my_ymm9);

                // 6th sample
                my_ymm2 = _mm256_fmadd_ps(my_ymm2, my_ymm12, my_ymm10);
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm13, my_ymm10);
                my_ymm4 = _mm256_fmadd_ps(my_ymm4, my_ymm14, my_ymm10);
                my_ymm5 = _mm256_fmadd_ps(my_ymm5, my_ymm15, my_ymm10);

                // 7th sample
                my_ymm3 = _mm256_fmadd_ps(my_ymm3, my_ymm12, my_ymm11);
                my_ymm4 = _mm256_fmadd_ps(my_ymm4, my_ymm13, my_ymm11);
                my_ymm5 = _mm256_fmadd_ps(my_ymm5, my_ymm14, my_ymm11);
                my_ymm6 = _mm256_fmadd_ps(my_ymm6, my_ymm15, my_ymm11);

                _mm256_store_ps(pfProc + 4, my_ymm0);
                _mm256_store_ps(pfProc + 8 + 4, my_ymm1);
                _mm256_store_ps(pfProc + 16 + 4, my_ymm2);
                _mm256_store_ps(pfProc + 24 + 4, my_ymm3);
                _mm256_store_ps(pfProc + 32 + 4, my_ymm4);
                _mm256_store_ps(pfProc + 40 + 4, my_ymm5);
                _mm256_store_ps(pfProc + 48 + 4, my_ymm6);

                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize;// *iParProc; // point to next kernel row now
            } // k_row
        } // col
    }
}
