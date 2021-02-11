#include "JincRessize.h"
/*
#if !defined(__AVX2__)
#error "AVX2 option needed"
#endif
*/ // VS2019 for unknown reason can not see __AVX2__ 


void JincResize::KernelRowAll_avx2_mul(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride)
{
	// current input plane sizes
	iWidthEl = iInpWidth + 2 * iKernelSize;
	iHeightEl = iInpHeight + 2 * iKernelSize;
	
	const int k_col8 = iKernelSize - (iKernelSize % 8);
    float *pfCurrKernel = g_pfKernel; 

	memset(g_pfFilteredImageBuffer, 0, iWidthEl * iHeightEl * iMul * iMul); 

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iWidthEl * iMul - (iKernelSize / 2);
 
        // prepare float32 pre-converted row data for each threal separately
        int64_t tidx = omp_get_thread_num();
        float* pfInpRowSamplesFloatBufStart = pfInpFloatRow + tidx * iWidthEl;
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
        {
            float *pfCurrKernel_pos = pfCurrKernel;
            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
                    *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart[col]), *(__m256*)(pfProc + k_col));
                }
                
                // need to process last (up to 7) floats separately..
                for (int64_t k_col = k_col8; k_col < iKernelSize; ++k_col)
                {
                    pfProc[k_col] += (pfCurrKernel_pos[k_col] * pfInpRowSamplesFloatBufStart[col]);
                }

                pfProc += iWidthEl * iMul; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize; // point to next kernel row now
             } // k_row
        } // col
    }

	ConvertToInt_avx2(iInpWidth, iInpHeight, dst, iDstStride);
}

void JincResize::KernelRowAll_avx2_mul_cb(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride)
{
	// current input plane sizes
	iWidthEl = iInpWidth + 2 * iKernelSize;
	iHeightEl = iInpHeight + 2 * iKernelSize;

	const int k_col8 = iKernelSize - (iKernelSize % 8);
	float *pfCurrKernel = g_pfKernel;

	memset(pfFilteredCirculatingBuf, 0, iWidthEl * iKernelSize * iMul * sizeof(float));

	// still 1 thread for now
	for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
	{
		// start all row-only dependent ptrs here

		// prepare float32 pre-converted row data for each threal separately
		//int64_t tidx = omp_get_thread_num();
		float* pfInpRowSamplesFloatBufStart = pfInpFloatRow; // +tidx * iWidthEl;
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);

		for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
		{
			float *pfCurrKernel_pos = pfCurrKernel;
			float* pfProc; 

			for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
			{
				pfProc = vpfRowsPointers[k_row] + col * iMul;
				for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
				{
					*(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart[col]), *(__m256*)(pfProc + k_col)); 
				}

				// need to process last (up to 7) floats separately..
				for (int64_t k_col = k_col8; k_col < iKernelSize; ++k_col)
				{
					pfProc[k_col] += (pfCurrKernel_pos[k_col] * pfInpRowSamplesFloatBufStart[col]);
				}
				pfCurrKernel_pos += iKernelSize; // point to next kernel row now
			} // k_row


		} // col

		int iOutStartRow = (row - (iTaps + iKernelSize))*iMul;
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
			memset(vpfRowsPointers[i], 0, iWidthEl*iMul * sizeof(float));
		}
        
	} // row
}

void JincResize::KernelRowAll_avx2_mul_cb_frw(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride)
{
	// current input plane sizes
	iWidthEl = iInpWidth + 2 * iKernelSize;
	iHeightEl = iInpHeight + 2 * iKernelSize;

	const int k_col8 = iKernelSize - (iKernelSize % 8);
	float *pfCurrKernel = g_pfKernel;

	memset(pfFilteredCirculatingBuf, 0, iWidthEl * iKernelSize * iMul * sizeof(float));

	// still 1 thread for now
	for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
	{
		// start all row-only dependent ptrs here

		// prepare float32 pre-converted row data for each threal separately
		//int64_t tidx = omp_get_thread_num();
		float* pfInpRowSamplesFloatBufStart = pfInpFloatRow; // +tidx * iWidthEl;
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);

		for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
		{
			float* pfCurrKernel_pos = g_pfKernel + k_row * iKernelSize;

			for (int64_t col = iTaps; col < iWidthEl - iTaps; col++)
			{
				float *pfProc = vpfRowsPointers[k_row] + col * iMul;
				float fInpSample = pfInpRowSamplesFloatBufStart[col];

				for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
				{
					*(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart[col]), *(__m256*)(pfProc + k_col));
				}

				// need to process last (up to 7) floats separately..
				for (int64_t k_col = k_col8; k_col < iKernelSize; ++k_col)
				{
					pfProc[k_col] += (pfCurrKernel_pos[k_col] * pfInpRowSamplesFloatBufStart[col]);
				}

			} // col
		} // r_row 

		int iOutStartRow = (row - (iTaps + iKernelSize))*iMul;
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
			memset(vpfRowsPointers[i], 0, iWidthEl*iMul * sizeof(float));
		}
        
	} // row
}



/*Better implementation of Mul2 will be with AVX512 instructions with shift between zmm registers instead of memory referencing with FMA*/
void JincResize::KernelRow_avx2_mul2_taps8(int64_t iOutWidth)
{
    float* pfCurrKernel = g_pfKernel + iKernelSize*iKernelSize; // needs unfinished g_pfKernelParProc with shifted or padded with zeroes kernel rows
/*
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
    }*/
}


void JincResize::KernelRow_avx2_mul8_taps3(int64_t iOutWidth)
{
    float* pfCurrKernel = g_pfKernel;
    /*
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
    }*/
}

void JincResize::KernelRowAll_avx2_mul4_taps4_cb(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{

    // current input plane sizes
    iWidthEl = iInpWidth + 2 * iKernelSize;
    iHeightEl = iInpHeight + 2 * iKernelSize;

    const int k_col8 = iKernelSize - (iKernelSize % 8);
    const int col8 = iWidthEl - iTaps - ((iWidthEl - iTaps) % 8);
    float* pfCurrKernel = g_pfKernel;

    int64_t col;

    memset(pfFilteredCirculatingBuf, 0, iWidthEl * iKernelSize * iMul * sizeof(float));

// still no MT for now - to be done later 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here 
        // prepare float32 pre-converted row data for each threal separately
//        int64_t tidx = omp_get_thread_num();
        float* pfInpRowSamplesFloatBufStart = pfInpFloatRow; // +tidx * iWidthEl;
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);

        for (col = iTaps; col < /*iWidthEl - iTaps*/col8; col += 8) // input cols counter
        {
            float* pfColStart = pfInpRowSamplesFloatBufStart + col;

            float* pfCurrKernel_pos = pfCurrKernel;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                float* pfProc = vpfRowsPointers[k_row] + col * iMul;

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

                my_ymm8 = _mm256_broadcast_ss(pfColStart + 0); // 1
                my_ymm9 = _mm256_broadcast_ss(pfColStart + 2); // 3
                my_ymm10 = _mm256_broadcast_ss(pfColStart + 4); // 5
                my_ymm11 = _mm256_broadcast_ss(pfColStart + 6); // 7 

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
                my_ymm8 = _mm256_broadcast_ss(pfColStart + 1); // 2
                my_ymm9 = _mm256_broadcast_ss(pfColStart + 3); // 4
                my_ymm10 = _mm256_broadcast_ss(pfColStart + 5); // 6
                my_ymm11 = _mm256_broadcast_ss(pfColStart + 7); // 8 

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

                pfCurrKernel_pos += iKernelSize;// *iParProc; // point to next kernel row now
            } // k_row
 
        } // col

        // need to process last up to 7 cols separately...
        for (col = col8 + iTaps; col < iWidthEl - iTaps; col++) // input cols counter
        {
            float* pfCurrKernel_pos = pfCurrKernel;
            float* pfProc;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                pfProc = vpfRowsPointers[k_row] + col * iMul;
                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
                    *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart[col]), *(__m256*)(pfProc + k_col));
                }

                // need to process last (up to 7) floats separately..
                for (int64_t k_col = k_col8; k_col < iKernelSize; ++k_col)
                {
                    pfProc[k_col] += (pfCurrKernel_pos[k_col] * pfInpRowSamplesFloatBufStart[col]);
                }
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

void JincResize::KernelRowAll_avx2_mul2_taps4_cb(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{
	// current input plane sizes
	iWidthEl = iInpWidth + 2 * iKernelSize;
	iHeightEl = iInpHeight + 2 * iKernelSize;

	const int k_col8 = iKernelSize - (iKernelSize % 8);
    const int col20 = iWidthEl - iTaps - ((iWidthEl - iTaps) % 20);
	float* pfCurrKernel = g_pfKernel;
    int64_t col;

	memset(pfFilteredCirculatingBuf, 0, iWidthEl * iKernelSize * iMul * sizeof(float));

	// still no MT for now - to be done later 
	for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
	{
		// start all row-only dependent ptrs here 
		// prepare float32 pre-converted row data for each threal separately
		//        int64_t tidx = omp_get_thread_num();
		float* pfInpRowSamplesFloatBufStart = pfInpFloatRow; // +tidx * iWidthEl;
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);

		for (col = iTaps; col < /*iWidthEl - iTaps*/col20; col += 20) // input cols counter
		{
            float* pfColStart = pfInpRowSamplesFloatBufStart + col;

			float* pfCurrKernel_pos = pfCurrKernel;

			for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
			{
                float* pfProc = vpfRowsPointers[k_row] + col * iMul;

				register __m256 my_ymm0, my_ymm1; // kernel samples
                register __m256 my_ymm2, my_ymm3, my_ymm4, my_ymm5, my_ymm6, my_ymm7; // out samples
				const register __m256i my_ymm8_main_circ = _mm256_set_epi32(1, 0, 7, 6, 5, 4, 3, 2); // main circulating cosst
                register __m256i my_imm_perm; // to temp ymm
                register __m256 my_ymm9, my_ymm10, my_ymm11, my_ymm12, my_ymm13; // inp samples
                register __m256 my_ymm15; //  temp

				my_ymm0 = _mm256_load_ps(pfCurrKernel_pos);
				my_ymm1 = _mm256_load_ps(pfCurrKernel_pos + 8);

				my_ymm2 = _mm256_load_ps(pfProc);
				my_ymm3 = _mm256_load_ps(pfProc + 8);
				my_ymm4 = _mm256_load_ps(pfProc + 16);
				my_ymm5 = _mm256_load_ps(pfProc + 24);
				my_ymm6 = _mm256_load_ps(pfProc + 32);
				my_ymm7 = _mm256_load_ps(pfProc + 40);
				my_ymm15 = _mm256_load_ps(pfProc + 48); //[in8_01;in8_23;in8_45;xx]

														// !! remove 	my_ymm15 = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0); - debug values for check how permutes work
				my_imm_perm = _mm256_set_epi32(1, 0, 5, 4, 3, 2, 7, 6);
				my_ymm15 = _mm256_permutevar8x32_ps(my_ymm15, my_imm_perm); // [xx;in8_23;in8_45;in8_01]

				my_ymm9 = _mm256_broadcast_ss(pfColStart + 0); // 1
				my_ymm10 = _mm256_broadcast_ss(pfColStart + 4); // 5
				my_ymm11 = _mm256_broadcast_ss(pfColStart + 8); // 9
				my_ymm12 = _mm256_broadcast_ss(pfColStart + 12); // 13 
				my_ymm13 = _mm256_broadcast_ss(pfColStart + 16); // 17 

				// 1st sample
				my_ymm2 = _mm256_fmadd_ps(my_ymm9, my_ymm0, my_ymm2);
				my_ymm3 = _mm256_fmadd_ps(my_ymm9, my_ymm1, my_ymm3);

				// 5 sample
				my_ymm3 = _mm256_fmadd_ps(my_ymm10, my_ymm0, my_ymm3);
				my_ymm4 = _mm256_fmadd_ps(my_ymm10, my_ymm1, my_ymm4);

				// 9 sample
				my_ymm4 = _mm256_fmadd_ps(my_ymm11, my_ymm0, my_ymm4);
				my_ymm5 = _mm256_fmadd_ps(my_ymm11, my_ymm1, my_ymm5);

				// 13 sample
				my_ymm5 = _mm256_fmadd_ps(my_ymm12, my_ymm0, my_ymm5);
				my_ymm6 = _mm256_fmadd_ps(my_ymm12, my_ymm1, my_ymm6);

				// 17 sample
				my_ymm6 = _mm256_fmadd_ps(my_ymm13, my_ymm0, my_ymm6);
				my_ymm7 = _mm256_fmadd_ps(my_ymm13, my_ymm1, my_ymm7);

				my_ymm15 = _mm256_blend_ps(my_ymm15, my_ymm2, 3); // store out 01 [out_01;in8_23;in8_45;in8_01]

				my_ymm2 = _mm256_permutevar8x32_ps(my_ymm2, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm3 = _mm256_permutevar8x32_ps(my_ymm3, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm2 = _mm256_blend_ps(my_ymm2, my_ymm3, 192); // copy higher 2 floats

				my_ymm4 = _mm256_permutevar8x32_ps(my_ymm4, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm3 = _mm256_blend_ps(my_ymm3, my_ymm4, 192); // copy higher 2 floats

				my_ymm5 = _mm256_permutevar8x32_ps(my_ymm5, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm4 = _mm256_blend_ps(my_ymm4, my_ymm5, 192); // copy higher 2 floats

				my_ymm6 = _mm256_permutevar8x32_ps(my_ymm6, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm5 = _mm256_blend_ps(my_ymm5, my_ymm6, 192); // copy higher 2 floats

				my_ymm7 = _mm256_permutevar8x32_ps(my_ymm7, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm6 = _mm256_blend_ps(my_ymm6, my_ymm7, 192); // copy higher 2 floats

																  // load next hi 2 ps from temp
				my_ymm7 = _mm256_blend_ps(my_ymm7, my_ymm15, 192); // copy higher 2 floats [out_01;in8_45;in8_23;xx]

																   // next samples
				my_ymm9 = _mm256_broadcast_ss(pfColStart + 1); // 2
				my_ymm10 = _mm256_broadcast_ss(pfColStart + 5); // 6
				my_ymm11 = _mm256_broadcast_ss(pfColStart + 9); // 10
				my_ymm12 = _mm256_broadcast_ss(pfColStart + 13); // 14 
				my_ymm13 = _mm256_broadcast_ss(pfColStart + 17); // 18 

				my_imm_perm = _mm256_set_epi32(3, 2, 1, 0, 5, 4, 7, 6);
				my_ymm15 = _mm256_permutevar8x32_ps(my_ymm15, my_imm_perm); // [xx;in8_45;out_01;in8_23]

				// 2 sample
				my_ymm2 = _mm256_fmadd_ps(my_ymm9, my_ymm0, my_ymm2);
				my_ymm3 = _mm256_fmadd_ps(my_ymm9, my_ymm1, my_ymm3);

				// 6 sample
				my_ymm3 = _mm256_fmadd_ps(my_ymm10, my_ymm0, my_ymm3);
				my_ymm4 = _mm256_fmadd_ps(my_ymm10, my_ymm1, my_ymm4);

				// 10 sample
				my_ymm4 = _mm256_fmadd_ps(my_ymm11, my_ymm0, my_ymm4);
				my_ymm5 = _mm256_fmadd_ps(my_ymm11, my_ymm1, my_ymm5);

				// 14 sample
				my_ymm5 = _mm256_fmadd_ps(my_ymm12, my_ymm0, my_ymm5);
				my_ymm6 = _mm256_fmadd_ps(my_ymm12, my_ymm1, my_ymm6);

				// 18 sample
				my_ymm6 = _mm256_fmadd_ps(my_ymm13, my_ymm0, my_ymm6);
				my_ymm7 = _mm256_fmadd_ps(my_ymm13, my_ymm1, my_ymm7);

				my_ymm15 = _mm256_blend_ps(my_ymm15, my_ymm2, 3); // store out 23 [out_23;in8_45;out_01;in8_23]

				my_ymm2 = _mm256_permutevar8x32_ps(my_ymm2, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm3 = _mm256_permutevar8x32_ps(my_ymm3, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm2 = _mm256_blend_ps(my_ymm2, my_ymm3, 192); // copy higher 2 floats

				my_ymm4 = _mm256_permutevar8x32_ps(my_ymm4, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm3 = _mm256_blend_ps(my_ymm3, my_ymm4, 192); // copy higher 2 floats

				my_ymm5 = _mm256_permutevar8x32_ps(my_ymm5, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm4 = _mm256_blend_ps(my_ymm4, my_ymm5, 192); // copy higher 2 floats

				my_ymm6 = _mm256_permutevar8x32_ps(my_ymm6, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm5 = _mm256_blend_ps(my_ymm5, my_ymm6, 192); // copy higher 2 floats

				my_ymm7 = _mm256_permutevar8x32_ps(my_ymm7, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm6 = _mm256_blend_ps(my_ymm6, my_ymm7, 192); // copy higher 2 floats

				// load next hi 2 ps from temp
				my_ymm7 = _mm256_blend_ps(my_ymm7, my_ymm15, 192); // copy higher 2 floats [out_23;in8_45;out_01;xx]

				// next samples
				my_ymm9 = _mm256_broadcast_ss(pfColStart + 2); // 3
				my_ymm10 = _mm256_broadcast_ss(pfColStart + 6); // 7
				my_ymm11 = _mm256_broadcast_ss(pfColStart + 10); // 11
				my_ymm12 = _mm256_broadcast_ss(pfColStart + 14); // 15 
				my_ymm13 = _mm256_broadcast_ss(pfColStart + 18); // 19 

				my_imm_perm = _mm256_set_epi32(3, 2, 5, 4, 1, 0, 7, 6);
				my_ymm15 = _mm256_permutevar8x32_ps(my_ymm15, my_imm_perm); // [xx;out_23;out_01;in8_45]

				// 3 sample
				my_ymm2 = _mm256_fmadd_ps(my_ymm9, my_ymm0, my_ymm2);
				my_ymm3 = _mm256_fmadd_ps(my_ymm9, my_ymm1, my_ymm3);

				// 7 sample
				my_ymm3 = _mm256_fmadd_ps(my_ymm10, my_ymm0, my_ymm3);
				my_ymm4 = _mm256_fmadd_ps(my_ymm10, my_ymm1, my_ymm4);

				// 11 sample
				my_ymm4 = _mm256_fmadd_ps(my_ymm11, my_ymm0, my_ymm4);
				my_ymm5 = _mm256_fmadd_ps(my_ymm11, my_ymm1, my_ymm5);

				// 15 sample
				my_ymm5 = _mm256_fmadd_ps(my_ymm12, my_ymm0, my_ymm5);
				my_ymm6 = _mm256_fmadd_ps(my_ymm12, my_ymm1, my_ymm6);

				// 19 sample
				my_ymm6 = _mm256_fmadd_ps(my_ymm13, my_ymm0, my_ymm6);
				my_ymm7 = _mm256_fmadd_ps(my_ymm13, my_ymm1, my_ymm7);

				my_ymm15 = _mm256_blend_ps(my_ymm15, my_ymm2, 3); // store out45 [out_45;out_23;out_01;in8_45]

				my_ymm2 = _mm256_permutevar8x32_ps(my_ymm2, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm3 = _mm256_permutevar8x32_ps(my_ymm3, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm2 = _mm256_blend_ps(my_ymm2, my_ymm3, 192); // copy higher 2 floats

				my_ymm4 = _mm256_permutevar8x32_ps(my_ymm4, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm3 = _mm256_blend_ps(my_ymm3, my_ymm4, 192); // copy higher 2 floats

				my_ymm5 = _mm256_permutevar8x32_ps(my_ymm5, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm4 = _mm256_blend_ps(my_ymm4, my_ymm5, 192); // copy higher 2 floats

				my_ymm6 = _mm256_permutevar8x32_ps(my_ymm6, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm5 = _mm256_blend_ps(my_ymm5, my_ymm6, 192); // copy higher 2 floats

				my_ymm7 = _mm256_permutevar8x32_ps(my_ymm7, my_ymm8_main_circ); // circulate by 2 ps to the left
				my_ymm6 = _mm256_blend_ps(my_ymm6, my_ymm7, 192); // copy higher 2 floats

				my_ymm7 = _mm256_blend_ps(my_ymm7, my_ymm15, 192); // copy higher 2 floats [out_45;out_23;out_01;xx]

																   // next samples
				my_ymm9 = _mm256_broadcast_ss(pfColStart + 3); // 4
				my_ymm10 = _mm256_broadcast_ss(pfColStart + 7); // 8
				my_ymm11 = _mm256_broadcast_ss(pfColStart + 11); // 12
				my_ymm12 = _mm256_broadcast_ss(pfColStart + 15); // 16 
				my_ymm13 = _mm256_broadcast_ss(pfColStart + 19); // 20 

				// final permute before store
				my_imm_perm = _mm256_set_epi32(7, 6, 1, 0, 3, 2, 5, 4);
				my_ymm15 = _mm256_permutevar8x32_ps(my_ymm15, my_imm_perm); // [out_01;out_23;out_45;xx]

				// 4 sample
				my_ymm2 = _mm256_fmadd_ps(my_ymm9, my_ymm0, my_ymm2);
				my_ymm3 = _mm256_fmadd_ps(my_ymm9, my_ymm1, my_ymm3);

				// 8 sample
				my_ymm3 = _mm256_fmadd_ps(my_ymm10, my_ymm0, my_ymm3);
				my_ymm4 = _mm256_fmadd_ps(my_ymm10, my_ymm1, my_ymm4);

				// 12 sample
				my_ymm4 = _mm256_fmadd_ps(my_ymm11, my_ymm0, my_ymm4);
				my_ymm5 = _mm256_fmadd_ps(my_ymm11, my_ymm1, my_ymm5);

				// 16 sample
				my_ymm5 = _mm256_fmadd_ps(my_ymm12, my_ymm0, my_ymm5);
				my_ymm6 = _mm256_fmadd_ps(my_ymm12, my_ymm1, my_ymm6);

				// 20 sample
				my_ymm6 = _mm256_fmadd_ps(my_ymm13, my_ymm0, my_ymm6);
				my_ymm7 = _mm256_fmadd_ps(my_ymm13, my_ymm1, my_ymm7);


				_mm256_store_ps(pfProc, my_ymm15);
				_mm256_store_ps(pfProc + 6, my_ymm2);
				_mm256_store_ps(pfProc + 14, my_ymm3);
				_mm256_store_ps(pfProc + 22, my_ymm4);
				_mm256_store_ps(pfProc + 30, my_ymm5);
				_mm256_store_ps(pfProc + 38, my_ymm6);
				_mm256_store_ps(pfProc + 46, my_ymm7);

				pfCurrKernel_pos += iKernelSize; // point to next kernel row now
			} // k_row

		} // col

        // need to process last up to 19 cols separately...
        for (col = col20 + iTaps; col < iWidthEl - iTaps; col++) // input cols counter
        {
            float* pfCurrKernel_pos = pfCurrKernel;
            float* pfProc;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                pfProc = vpfRowsPointers[k_row] + col * iMul;
                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
                    *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart[col]), *(__m256*)(pfProc + k_col));
                }

                // need to process last (up to 7) floats separately..
                for (int64_t k_col = k_col8; k_col < iKernelSize; ++k_col)
                {
                    pfProc[k_col] += (pfCurrKernel_pos[k_col] * pfInpRowSamplesFloatBufStart[col]);
                }
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


void JincResize::KernelRowAll_avx2_mul4_taps4(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{

    // current input plane sizes
    iWidthEl = iInpWidth + 2 * iKernelSize;
    iHeightEl = iInpHeight + 2 * iKernelSize;

    const int k_col8 = iKernelSize - (iKernelSize % 8);
    float* pfCurrKernel = g_pfKernel;

    memset(g_pfFilteredImageBuffer, 0, iWidthEl * iHeightEl * iMul * iMul);

#pragma omp parallel for num_threads(threads_)
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iWidthEl * iMul - (iKernelSize / 2);

        // prepare float32 pre-converted row data for each threal separately
        int64_t tidx = omp_get_thread_num();
        float* pfInpRowSamplesFloatBufStart = pfInpFloatRow + tidx * iWidthEl;
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col += 8) // input cols counter
        {
            float* pfColStart = pfInpRowSamplesFloatBufStart + col;

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

                my_ymm8 = _mm256_broadcast_ss(pfColStart + 0); // 1
                my_ymm9 = _mm256_broadcast_ss(pfColStart + 2); // 3
                my_ymm10 = _mm256_broadcast_ss(pfColStart + 4); // 5
                my_ymm11 = _mm256_broadcast_ss(pfColStart + 6); // 7

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
                my_ymm8 = _mm256_broadcast_ss(pfColStart + 1); // 2
                my_ymm9 = _mm256_broadcast_ss(pfColStart + 3); // 4
                my_ymm10 = _mm256_broadcast_ss(pfColStart + 5); // 6
                my_ymm11 = _mm256_broadcast_ss(pfColStart + 7); // 8

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

                pfProc += iWidthEl * iMul; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize;// *iParProc; // point to next kernel row now
            } // k_row
        } // col
    } // row

    ConvertToInt_avx2(iInpWidth, iInpHeight, dst, iDstStride);
}

void JincResize::KernelRow_avx2_mul4_taps4_fr(int64_t iOutWidth)
{
    float* pfCurrKernel = g_pfKernel;

	memset(g_pfFilteredImageBuffer, 0 , iWidthEl * iHeightEl * iMul * iMul);

    /* iMul=4, iTaps=4, KernelSize (row length in floats) is 32*/
    /* full row-walking, fixed at 19.01.2021 and need to be tested 
       added ConvertInpElRowToFloat stuff 22.01.2021 and need to be tested too */
/*
#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here 
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);

	// prepare float32 pre-converted row data for each threal separately
	    int64_t tidx = omp_get_thread_num();
        float *pfInpRowSamplesFloatBufStart = pfInpFloatRow + tidx * iWidthEl;
        (this->*GetInpElRowAsFloat)(row, pfInpRowSamplesFloatBufStart);
	// lets hope now converted to float32 one row inp samples in L1d cache.

        for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
        {

            float *pfCurrKernel_pos = pfCurrKernel + iKernelSize * k_row;

            float *pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + iTaps * iMul + k_row * iOutWidth;

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

            for (int64_t col = iTaps; col < iWidthEl - iTaps; col += 8) // input cols counter
            {
                // odd samples
		        my_ymm8 = _mm256_broadcast_ss(pfInpRowSamplesFloatBufStart + col); // 1
                my_ymm9 = _mm256_broadcast_ss(pfInpRowSamplesFloatBufStart + col + 2); // 3
                my_ymm10 = _mm256_broadcast_ss(pfInpRowSamplesFloatBufStart + col + 4); // 5
                my_ymm11 = _mm256_broadcast_ss(pfInpRowSamplesFloatBufStart + col + 6); // 7            

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
		        my_ymm8 = _mm256_broadcast_ss(pfInpRowSamplesFloatBufStart + col + 1); // 2
                my_ymm9 = _mm256_broadcast_ss(pfInpRowSamplesFloatBufStart + col + 3); // 4
                my_ymm10 = _mm256_broadcast_ss(pfInpRowSamplesFloatBufStart + col + 5); // 6
                my_ymm11 = _mm256_broadcast_ss(pfInpRowSamplesFloatBufStart + col + 7); // 8            


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

                // 8th sample
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
    } // row*/
}

void JincResize::ConvertToInt_avx2(int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{
    const int col32 = (iInpWidth*iMul) - ((iInpWidth*iMul) % 32);

    __m256 my_zero_ymm2;

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
        }
    }
}

__forceinline void JincResize::ConvertiMulRowsToInt_avx2(int iInpWidth, int iOutStartRow, unsigned char* dst, int iDstStride)
{
	const int col32 = (iInpWidth*iMul) - ((iInpWidth*iMul) % 32);

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

__forceinline void JincResize::GetInpElRowAsFloat_avx2(int iInpRow, int iCurrInpHeight, int iCurrInpWidth, unsigned char* pCurr_src, int iCurrSrcStride, float* dst)
{
    int64_t col;

    const int64_t col8 = iKernelSize - iKernelSize % 8;
    // row range from iTaps to iHeightEl - iTaps - 1
    // use iWidthEl and iHeightEl set in KernelRow()

    if (iInpRow < iKernelSize) iInpRow = iKernelSize;
    if (iInpRow > (iCurrInpHeight + iKernelSize - 1)) iInpRow = iCurrInpHeight + iKernelSize - 1;

    unsigned char* pCurrRowSrc = pCurr_src + (iInpRow - iKernelSize) * iCurrSrcStride;
        
    // start cols
    __m256 my_ymm_start = _mm256_setzero_ps();
    my_ymm_start = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm_start), *pCurrRowSrc));

    for (col = iKernelSize - 8; col >= 0; col-=8) // left
    {
        _mm256_store_ps(dst + col, my_ymm_start);
    }

    // mid cols
    unsigned char *pCurrColSrc = pCurrRowSrc;
    for (col = iKernelSize; col < iKernelSize + iCurrInpWidth; col+=8) 
    {
        __m256 src_ps = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)pCurrColSrc)));
        _mm256_store_ps(dst + col, src_ps);
        pCurrColSrc += 8;
    }

    // end cols
    __m256 my_ymm_end = _mm256_setzero_ps();
    my_ymm_end = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm_start), pCurrRowSrc[iCurrInpWidth - 1]));
    for (col = iKernelSize + iCurrInpWidth; col < iWidthEl; col+=8) // right
    {
        _mm256_store_ps(dst + col, my_ymm_end);
    }
 
}
