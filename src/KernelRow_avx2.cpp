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
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart, iKernelSize);

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

void JincResize::KernelRowAll_avx2_mul_cb(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride, const int ciMul, const int ciTaps, const int ciKS)
{
	// current input plane sizes
	iWidthEl = iInpWidth + 2 * ciKS;
	iHeightEl = iInpHeight + 2 * ciKS;

	const int k_col8 = ciKS - (ciKS % 8);
	float *pfCurrKernel = g_pfKernel;

	memset(pfFilteredCirculatingBuf, 0, iWidthEl * ciKS * ciMul * sizeof(float));

	// still 1 thread for now
	for (int64_t row = ciTaps; row < iHeightEl - ciTaps; row++) // input lines counter
	{
		// start all row-only dependent ptrs here

		// prepare float32 pre-converted row data for each threal separately
		//int64_t tidx = omp_get_thread_num();
		float* pfInpRowSamplesFloatBufStart = pfInpFloatRow; // +tidx * iWidthEl;
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart, ciKS);

		for (int64_t col = ciTaps; col < iWidthEl - ciTaps; col++) // input cols counter
		{
			float *pfCurrKernel_pos = pfCurrKernel;
			float* pfProc; 

			for (int64_t k_row = 0; k_row < ciKS; k_row++)
			{
				pfProc = vpfRowsPointers[k_row] + col * ciMul;
				for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
				{
					*(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart[col]), *(__m256*)(pfProc + k_col)); 
				}

				// need to process last (up to 7) floats separately..
				for (int64_t k_col = k_col8; k_col < ciKS; ++k_col)
				{
					pfProc[k_col] += (pfCurrKernel_pos[k_col] * pfInpRowSamplesFloatBufStart[col]);
				}
				pfCurrKernel_pos += ciKS; // point to next kernel row now
			} // k_row


		} // col

		int iOutStartRow = (row - (ciTaps + ciKS)) * ciMul;
		//iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
		if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight) * ciMul)
		{
			ConvertiMulRowsToInt_avx2(vpfRowsPointers, iInpWidth, iOutStartRow, dst, iDstStride);
		}
        
		// circulate pointers to iMul rows upper
		std::rotate(vpfRowsPointers.begin(), vpfRowsPointers.begin() + ciMul, vpfRowsPointers.end());
    
		// clear last iMul rows
		for (int i = ciKS - ciMul; i < ciKS; i++)
		{
			memset(vpfRowsPointers[i], 0, iWidthEl * ciMul * sizeof(float));
		}
        
	} // row
}

void JincResize::KernelRowAll_avx2_mul_cb_2r(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride, const int ciMul, const int ciTaps, const int ciKS)
{
    // current input plane sizes
    iWidthEl = iInpWidth + 2 * ciKS;
    iHeightEl = iInpHeight + 2 * ciKS;

    const int k_col8 = ciKS - (ciKS % 8);
    float* pfCurrKernel = g_pfKernel;

    memset(pfFilteredCirculatingBuf, 0, iWidthEl * (ciKS + ciMul * (iNumSimProcRows - 1)) * ciMul * sizeof(float));

    // still 1 thread for now
    for (int64_t row = ciTaps; row < iHeightEl - (ciTaps * iNumSimProcRows); row += iNumSimProcRows) // input lines counter, *iNumSimProcRows temp workaround ?
    {
        // start all row-only dependent ptrs here
        float* pfInpRowSamplesFloatBufStart_r1 = pfInpFloatRow; 
        float* pfInpRowSamplesFloatBufStart_r2 = pfInpFloatRow + iWidthEl;
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart_r1, ciKS);
        GetInpElRowAsFloat_avx2(row + 1, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart_r2, ciKS);

        for (int64_t col = ciTaps; col < iWidthEl - ciTaps; col++) // input cols counter
        {
            float* pfCurrKernel_pos1 = pfCurrKernel;
            float* pfCurrKernel_pos2 = pfCurrKernel;
            float* pfProc;

            for (int64_t k_row = 0; k_row < ciMul; k_row++) // r1 only proc
            {
                pfProc = vpfRowsPointers_nr[k_row] + col * ciMul;
#pragma unroll(2)
                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
                    *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos1 + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart_r1[col]), *(__m256*)(pfProc + k_col));
                }

                // need to process last (up to 7) floats separately..
                for (int64_t k_col = k_col8; k_col < ciKS; ++k_col)
                {
                    pfProc[k_col] += (pfCurrKernel_pos1[k_col] * pfInpRowSamplesFloatBufStart_r1[col]);
                }
                pfCurrKernel_pos1 += ciKS; // point to next kernel row now
            } // k_row

            for (int64_t k_row = ciMul; k_row < ciKS; k_row++) // r1  and r2 proc
            {
                pfProc = vpfRowsPointers_nr[k_row] + col * ciMul;
#pragma unroll(2)
                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
                    __m256 my_tmp = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos1 + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart_r1[col]), *(__m256*)(pfProc + k_col));
//                  *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos1 + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart_r1[col]), *(__m256*)(pfProc + k_col));
//                  *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos2 + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart_r2[col]), *(__m256*)(pfProc + k_col));
                    *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos2 + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart_r2[col]), my_tmp);
                }

                // need to process last (up to 7) floats separately..
                for (int64_t k_col = k_col8; k_col < ciKS; ++k_col)
                {
                    pfProc[k_col] += ((pfCurrKernel_pos1[k_col] * pfInpRowSamplesFloatBufStart_r1[col]) + (pfCurrKernel_pos2[k_col] * pfInpRowSamplesFloatBufStart_r2[col]));
                }
                pfCurrKernel_pos1 += ciKS; // point to next kernel row now
                pfCurrKernel_pos2 += ciKS; // point to next kernel row now
            } // k_row
#pragma unroll(2)
            for (int64_t k_row = ciKS; k_row < ciKS + ciMul; k_row++) // r2 proc
            {
                pfProc = vpfRowsPointers_nr[k_row] + col * ciMul;
                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
                    *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos2 + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart_r2[col]), *(__m256*)(pfProc + k_col));
                }

                // need to process last (up to 7) floats separately..
                for (int64_t k_col = k_col8; k_col < ciKS; ++k_col)
                {
                    pfProc[k_col] += (pfCurrKernel_pos2[k_col] * pfInpRowSamplesFloatBufStart_r2[col]);
                }
                pfCurrKernel_pos2 += ciKS; // point to next kernel row now
            } // k_row

        } // col

        int iOutStartRow = (row - (ciTaps + ciKS)) * ciMul;
        //iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
        if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight) * ciMul)
        {
            ConvertNRowsToInt_avx2(vpfRowsPointers_nr, iInpWidth, iOutStartRow, dst, iDstStride, iNumSimProcRows * ciMul);
        }

        // circulate pointers to iMul * iNumSimProcRows rows upper
        std::rotate(vpfRowsPointers_nr.begin(), vpfRowsPointers_nr.begin() + ciMul * iNumSimProcRows, vpfRowsPointers_nr.end());

        // clear last iMul * iNumSimProcRows rows
        for (int i = (ciKS + ciMul * (iNumSimProcRows - 1)) - (ciMul * iNumSimProcRows); i < (ciKS + ciMul * (iNumSimProcRows - 1)); i++)
        {
            memset(vpfRowsPointers_nr[i], 0, iWidthEl * ciMul * sizeof(float));
        }

    } // row
}

void JincResize::KernelRowAll_avx2_mul_cb_mt(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride, const int ciMul, const int ciTaps, const int ciKS)
{
    // current input plane sizes
    iWidthEl = iInpWidth + 2 * ciKS;
    iHeightEl = iInpHeight + 2 * ciKS;

    const int k_col8 = ciKS - (ciKS % 8);

    const int64_t iNumRowsPerThread = (iHeightEl - 2 * ciTaps) / threads_;

    // initial clearing
    for (int j = 0; j < threads_; j++)
    {
        for (int i = 0; i < ciKS; i++)
        {
            memset(vpvThreadsVectors[j][i], 0, iWidthEl * ciMul * sizeof(float));
        }
    }

#pragma omp parallel num_threads(threads_)
    {
        // start all thread dependent ptrs here
        int64_t tidx = omp_get_thread_num(); // our thread id here

        std::vector<float*> vpfThreadVector = vpvThreadsVectors[tidx];
        // it looks vpfThreadVector uses copy of vector from vpvThreadsVectors and it may be slower, TO DO - remake to pointer to vector

        // calculate rows to process in this thread
        int64_t iStartRow = tidx * iNumRowsPerThread;
        int64_t iThreadSkipRows = ciTaps * 2;
        // some check
        if (iStartRow < ciTaps) iStartRow = ciTaps;
        int64_t iEndRow = iStartRow + iNumRowsPerThread + 2 * ciTaps;
        if (iEndRow > iHeightEl - ciTaps) iEndRow = iHeightEl - ciTaps;

        for (int64_t row = iStartRow; row < iEndRow; row++)
        {
            // start all row-only dependent ptrs here

            // prepare float32 pre-converted row data for each threal separately
            float* pfInpRowSamplesFloatBufStart = pfInpFloatRow + tidx * iWidthEl;
            GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart, ciKS);

            for (int64_t col = ciTaps; col < iWidthEl - ciTaps; col++) // input cols counter
            {
                float* pfCurrKernel_pos = g_pfKernel;
                float* pfProc;

                for (int64_t k_row = 0; k_row < ciKS; k_row++)
                {
                    pfProc = vpfThreadVector[k_row] + col * ciMul;
                    for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                    {
                        *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart[col]), *(__m256*)(pfProc + k_col));
                    }

                    // need to process last (up to 7) floats separately..
                    for (int64_t k_col = k_col8; k_col < ciKS; ++k_col)
                    {
                        pfProc[k_col] += (pfCurrKernel_pos[k_col] * pfInpRowSamplesFloatBufStart[col]);
                    }
                    pfCurrKernel_pos += ciKS; // point to next kernel row now
                } // k_row


            } // col

            int iOutStartRow = (row - (ciTaps + ciKS)) * ciMul;
            //iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
            if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight) * ciMul && iThreadSkipRows <= 0)
            {
                // it looks vpfThreadVector uses copy of vector and it may be slower, TO DO - remake to pointer to vector
                ConvertiMulRowsToInt_avx2(vpfThreadVector, iInpWidth, iOutStartRow, dst, iDstStride);
            }

            iThreadSkipRows--;

            // circulate pointers to iMul rows upper
            std::rotate(vpfThreadVector.begin(), vpfThreadVector.begin() + ciMul, vpfThreadVector.end());

            // clear last iMul rows
            for (int i = ciKS - ciMul; i < ciKS; i++)
            {
                memset(vpfThreadVector[i], 0, iWidthEl * ciMul * sizeof(float));
            }

        } // row
    } // parallel section
}

void JincResize::KernelRowAll_avx2_mul4_taps4_cb(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride, const int ciMul, const int ciTaps, const int ciKS)
{

    // current input plane sizes
    iWidthEl = iInpWidth + 64/*2 * iKernelSize*/;
    iHeightEl = iInpHeight + 64/*2 * iKernelSize*/;

    const int k_col8 = 32/*iKernelSize*/ - (32/*iKernelSize*/ % 8);
    const int col8 = iWidthEl - 4/*iTaps*/ - ((iWidthEl - 4/*iTaps*/) % 8);
    float* pfCurrKernel = g_pfKernel;

    int64_t col;

    memset(pfFilteredCirculatingBuf, 0, iWidthEl * 32 * 4/*iKernelSize * iMul*/ * sizeof(float));

// no MT version 
    for (int64_t row = 4/*iTaps*/; row < iHeightEl - 4/*iTaps*/; row++) // input lines counter
    {
        // start all row-only dependent ptrs here 
        float* pfInpRowSamplesFloatBufStart = pfInpFloatRow; 
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart, 32/*iKernelSize*/);

        for (col = 4/*iTaps*/; col < col8; col += 8) // input cols counter
        {
            float* pfColStart = pfInpRowSamplesFloatBufStart + col;

            float* pfCurrKernel_pos = pfCurrKernel;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                float* pfProc = vpfRowsPointers[k_row] + col * 4/*iMul*/;

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

                pfCurrKernel_pos += 32/*iKernelSize*/; // point to next kernel row now
            } // k_row
 
        } // col

        // need to process last up to 7 cols separately...
        for (col = col8 + 4/*iTaps*/; col < iWidthEl - 4/*iTaps*/; col++) // input cols counter
        {
            float* pfCurrKernel_pos = pfCurrKernel;
            float* pfProc;

            for (int64_t k_row = 0; k_row < 32/*iKernelSize*/; k_row++)
            {
                pfProc = vpfRowsPointers[k_row] + col * 4/*iMul*/;
                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
                    *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart[col]), *(__m256*)(pfProc + k_col));
                }

                // need to process last (up to 7) floats separately..
                for (int64_t k_col = k_col8; k_col < 32/*iKernelSize*/; ++k_col)
                {
                    pfProc[k_col] += (pfCurrKernel_pos[k_col] * pfInpRowSamplesFloatBufStart[col]);
                }
                pfCurrKernel_pos += 32/*iKernelSize*/; // point to next kernel row now
            } // k_row

        } // col

        int iOutStartRow = (row - (4 + 32/*iTaps + iKernelSize*/)) * 4/*iMul*/;
        //iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
        if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight) * 4/*iMul*/)
        {
            ConvertiMulRowsToInt_avx2(vpfRowsPointers, iInpWidth, iOutStartRow, dst, iDstStride);
        }
        
        // circulate pointers to iMul rows upper
        std::rotate(vpfRowsPointers.begin(), vpfRowsPointers.begin() + 4/*iMul*/, vpfRowsPointers.end());
        
        // clear last iMul rows
        for (int i = 32 - 4/*iKernelSize - iMul*/; i < 32/*iKernelSize*/; i++)
        {
            memset(vpfRowsPointers[i], 0, iWidthEl * 4/*iMul*/ * sizeof(float));
        } 
    } // row
}

void JincResize::KernelRowAll_avx2_mul4_taps4_cb_2r(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride, const int ciMul, const int ciTaps, const int ciKS)
{

    // current input plane sizes
    iWidthEl = iInpWidth + 2 * 32/*iKernelSize*/;
    iHeightEl = iInpHeight + 2 * 32/*iKernelSize*/;

    const int k_col8 = 32/*iKernelSize*/ - (32/*iKernelSize*/ % 8);
    const int col8 = iWidthEl - 4/*iTaps*/ - ((iWidthEl - 4/*iTaps*/) % 8);
    float* pfCurrKernel = g_pfKernel;

    int64_t col;

    memset(pfFilteredCirculatingBuf, 0, iWidthEl * (32+4/*iKernelSize + iMul*/) * 4/*iMul*/ * sizeof(float));

    // no MT version 
    for (int64_t row = 4/*iTaps*/; row < (iHeightEl - 8/*iTaps*/); row += 2) // input lines counter  !!! temp -8 because of 2 rows proc ?
    {
        // start all row-only dependent ptrs here 
        float* pfInpRowSamplesFloatBufStart_r1 = pfInpFloatRow;
        float* pfInpRowSamplesFloatBufStart_r2 = pfInpFloatRow + iWidthEl;
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart_r1, 32/*iKernelSize*/);
        GetInpElRowAsFloat_avx2(row + 1, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart_r2, 32/*iKernelSize*/); // 

        for (col = 4/*iTaps*/; col < col8; col += 8) // input cols counter
        {
            float* pfColStart1 = pfInpRowSamplesFloatBufStart_r1 + col;// -iTaps; // temp -it for align 32 !!
            float* pfColStart2 = pfInpRowSamplesFloatBufStart_r2 + col;// -iTaps; // temp -it for align 32 !!

            float* pfCurrKernel_pos1 = pfCurrKernel;
            float* pfCurrKernel_pos2 = pfCurrKernel;

            __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4, my_ymm5, my_ymm6, my_ymm7; // out samples
            __m256 my_ymm8, my_ymm9, my_ymm10, my_ymm11; // inp samples
            __m256 my_ymm12, my_ymm13, my_ymm14, my_ymm15; // kernel samples

            for (int64_t k_row = 0; k_row < 4; k_row++) // 1st row only proc - iMul =4
            {
                float* pfProc = vpfRowsPointers_nr[k_row] + col * 4/*iMul*/;

                my_ymm12 = _mm256_load_ps(pfCurrKernel_pos1);
                my_ymm13 = _mm256_load_ps(pfCurrKernel_pos1 + 8);
                my_ymm14 = _mm256_load_ps(pfCurrKernel_pos1 + 16);
                my_ymm15 = _mm256_load_ps(pfCurrKernel_pos1 + 24);

                my_ymm0 = _mm256_load_ps(pfProc);
                my_ymm1 = _mm256_load_ps(pfProc + 8);
                my_ymm2 = _mm256_load_ps(pfProc + 16);
                my_ymm3 = _mm256_load_ps(pfProc + 24);
                my_ymm4 = _mm256_load_ps(pfProc + 32);
                my_ymm5 = _mm256_load_ps(pfProc + 40);
                my_ymm6 = _mm256_load_ps(pfProc + 48);

                my_ymm8 = _mm256_broadcast_ss(pfColStart1 + 0); // 1
                my_ymm9 = _mm256_broadcast_ss(pfColStart1 + 2); // 3
                my_ymm10 = _mm256_broadcast_ss(pfColStart1 + 4); // 5
                my_ymm11 = _mm256_broadcast_ss(pfColStart1 + 6); // 7 

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
                my_ymm8 = _mm256_broadcast_ss(pfColStart1 + 1); // 2
                my_ymm9 = _mm256_broadcast_ss(pfColStart1 + 3); // 4
                my_ymm10 = _mm256_broadcast_ss(pfColStart1 + 5); // 6
                my_ymm11 = _mm256_broadcast_ss(pfColStart1 + 7); // 8 

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
                _mm256_store_ps(pfProc + 28, my_ymm3);
                _mm256_store_ps(pfProc + 36, my_ymm4);
                _mm256_store_ps(pfProc + 44, my_ymm5);
                _mm256_store_ps(pfProc + 52, my_ymm6);

                pfCurrKernel_pos1 += 32/*iKernelSize*/;// point to next kernel row now
            } // k_row

            for (int64_t k_row = 4/*iMul*/; k_row < 32/*iKS*/; k_row++) // 1st and 2nd rows proc iMul=4 , iKS=32
            {
                float* pfProc = vpfRowsPointers_nr[k_row] + col * 4/*iMul*/;

                // 1st row

                my_ymm12 = _mm256_load_ps(pfCurrKernel_pos1);
                my_ymm13 = _mm256_load_ps(pfCurrKernel_pos1 + 8);
                my_ymm14 = _mm256_load_ps(pfCurrKernel_pos1 + 16);
                my_ymm15 = _mm256_load_ps(pfCurrKernel_pos1 + 24);

                my_ymm0 = _mm256_load_ps(pfProc);
                my_ymm1 = _mm256_load_ps(pfProc + 8);
                my_ymm2 = _mm256_load_ps(pfProc + 16);
                my_ymm3 = _mm256_load_ps(pfProc + 24);
                my_ymm4 = _mm256_load_ps(pfProc + 32);
                my_ymm5 = _mm256_load_ps(pfProc + 40);
                my_ymm6 = _mm256_load_ps(pfProc + 48);

                my_ymm8 = _mm256_broadcast_ss(pfColStart1 + 0); // 1
                my_ymm9 = _mm256_broadcast_ss(pfColStart1 + 2); // 3
                my_ymm10 = _mm256_broadcast_ss(pfColStart1 + 4); // 5
                my_ymm11 = _mm256_broadcast_ss(pfColStart1 + 6); // 7 

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
                
                // 2nd row

                my_ymm12 = _mm256_load_ps(pfCurrKernel_pos2);
                my_ymm13 = _mm256_load_ps(pfCurrKernel_pos2 + 8);
                my_ymm14 = _mm256_load_ps(pfCurrKernel_pos2 + 16);
                my_ymm15 = _mm256_load_ps(pfCurrKernel_pos2 + 24);

                my_ymm8 = _mm256_broadcast_ss(pfColStart2 + 0); // 1
                my_ymm9 = _mm256_broadcast_ss(pfColStart2 + 2); // 3
                my_ymm10 = _mm256_broadcast_ss(pfColStart2 + 4); // 5
                my_ymm11 = _mm256_broadcast_ss(pfColStart2 + 6); // 7 

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

                // 2nd row to not reload kernel again
                
                // even samples
                my_ymm8 = _mm256_broadcast_ss(pfColStart2 + 1); // 2
                my_ymm9 = _mm256_broadcast_ss(pfColStart2 + 3); // 4
                my_ymm10 = _mm256_broadcast_ss(pfColStart2 + 5); // 6
                my_ymm11 = _mm256_broadcast_ss(pfColStart2 + 7); // 8 

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
                
                // 1st row reload 1st row kernel again

                my_ymm12 = _mm256_load_ps(pfCurrKernel_pos1);
                my_ymm13 = _mm256_load_ps(pfCurrKernel_pos1 + 8);
                my_ymm14 = _mm256_load_ps(pfCurrKernel_pos1 + 16);
                my_ymm15 = _mm256_load_ps(pfCurrKernel_pos1 + 24);
                
                // even samples
                my_ymm8 = _mm256_broadcast_ss(pfColStart1 + 1); // 2
                my_ymm9 = _mm256_broadcast_ss(pfColStart1 + 3); // 4
                my_ymm10 = _mm256_broadcast_ss(pfColStart1 + 5); // 6
                my_ymm11 = _mm256_broadcast_ss(pfColStart1 + 7); // 8 

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
                _mm256_store_ps(pfProc + 28, my_ymm3);
                _mm256_store_ps(pfProc + 36, my_ymm4);
                _mm256_store_ps(pfProc + 44, my_ymm5);
                _mm256_store_ps(pfProc + 52, my_ymm6);

                pfCurrKernel_pos1 += 32/*iKernelSize*/;// point to next kernel row now
                pfCurrKernel_pos2 += 32/*iKernelSize*/;// point to next kernel row now
            } // k_row
            
            for (int64_t k_row = 32/*iKS*/; k_row < 36/*iKS+iMul*/; k_row++) // 2nd row only proc, KS to KS+iMul
            {
                float* pfProc = vpfRowsPointers_nr[k_row] + col * 4/*iMul*/;

                // 2nd row

                my_ymm12 = _mm256_load_ps(pfCurrKernel_pos2);
                my_ymm13 = _mm256_load_ps(pfCurrKernel_pos2 + 8);
                my_ymm14 = _mm256_load_ps(pfCurrKernel_pos2 + 16);
                my_ymm15 = _mm256_load_ps(pfCurrKernel_pos2 + 24);

                my_ymm0 = _mm256_load_ps(pfProc);
                my_ymm1 = _mm256_load_ps(pfProc + 8);
                my_ymm2 = _mm256_load_ps(pfProc + 16);
                my_ymm3 = _mm256_load_ps(pfProc + 24);
                my_ymm4 = _mm256_load_ps(pfProc + 32);
                my_ymm5 = _mm256_load_ps(pfProc + 40);
                my_ymm6 = _mm256_load_ps(pfProc + 48);

                my_ymm8 = _mm256_broadcast_ss(pfColStart2 + 0); // 1
                my_ymm9 = _mm256_broadcast_ss(pfColStart2 + 2); // 3
                my_ymm10 = _mm256_broadcast_ss(pfColStart2 + 4); // 5
                my_ymm11 = _mm256_broadcast_ss(pfColStart2 + 6); // 7 

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

                // 2nd row to not reload kernel again

                // even samples
                my_ymm8 = _mm256_broadcast_ss(pfColStart2 + 1); // 2
                my_ymm9 = _mm256_broadcast_ss(pfColStart2 + 3); // 4
                my_ymm10 = _mm256_broadcast_ss(pfColStart2 + 5); // 6
                my_ymm11 = _mm256_broadcast_ss(pfColStart2 + 7); // 8 

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
                _mm256_store_ps(pfProc + 28, my_ymm3);
                _mm256_store_ps(pfProc + 36, my_ymm4);
                _mm256_store_ps(pfProc + 44, my_ymm5);
                _mm256_store_ps(pfProc + 52, my_ymm6);

                pfCurrKernel_pos2 += 32/*iKernelSize*/;// point to next kernel row now
            } // k_row
            

        } // col
        /*
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
        */
        int iOutStartRow = (row - (32+4/*iTaps + iKernelSize*/)) * 4/*iMul*/;
        //iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
        if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight) * 4/*iMul*/)
        {
            ConvertNRowsToInt_avx2(vpfRowsPointers_nr, iInpWidth, iOutStartRow, dst, iDstStride, 8/*iMul * 2*/);
        }

        // circulate pointers to iMul * 2 rows upper
        std::rotate(vpfRowsPointers_nr.begin(), vpfRowsPointers_nr.begin() + 8/*iMul * 2*/, vpfRowsPointers_nr.end());

        // clear last iMul * 2 rows
        for (int i = 28/*(iKernelSize - iMul)*/; i < 36/*(iKernelSize + iMul)*/; i++)
        {
            memset(vpfRowsPointers_nr[i], 0, iWidthEl * 4/*iMul*/ * sizeof(float));
        }
    } // row
}

void JincResize::KernelRowAll_avx2_mul4_taps4_cb_mt(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride, const int ciMul, const int ciTaps, const int ciKS)
{

    // current input plane sizes
    iWidthEl = iInpWidth + 2 * ciKS;
    iHeightEl = iInpHeight + 2 * ciKS;

    const int k_col8 = ciKS - (ciKS % 8);
    const int col8 = iWidthEl - ciTaps - ((iWidthEl - ciTaps) % 8);
    float* pfCurrKernel = g_pfKernel;

    const int64_t iNumRowsPerThread = (iHeightEl - 2 * ciTaps) / threads_;

    // initial clearing
    for (int j = 0; j < threads_; j++)
    {
        for (int i = 0; i < iKernelSize; i++)
        {
            memset(vpvThreadsVectors[j][i], 0, iWidthEl * ciMul * sizeof(float));
        }
    }

#pragma omp parallel num_threads(threads_)
    {
        int64_t tidx = omp_get_thread_num(); // our thread id here

        std::vector<float*> vpfThreadVector = vpvThreadsVectors[tidx];
        // it looks vpfThreadVector uses copy of vector from vpvThreadsVectors and it may be slower, TO DO - remake to pointer to vector

        // calculate rows to process in this thread
        int64_t iStartRow = tidx * iNumRowsPerThread;
        int64_t iThreadSkipRows = 2 * ciTaps;//  iTaps * 2;
        // some check
        if (iStartRow < ciTaps) iStartRow = ciTaps;// iTaps;
        int64_t iEndRow = iStartRow + iNumRowsPerThread + 2 * ciTaps;// +2 * iTaps;
        if (iEndRow > iHeightEl - ciTaps) iEndRow = iHeightEl - ciTaps;//  -iTaps;
        // prepare float32 pre-converted row data for each threal separately

        for (int64_t row = iStartRow; row < iEndRow; row++)
        {

            float* pfInpRowSamplesFloatBufStart = pfInpFloatRow + tidx * iWidthEl;
            GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart, ciKS);

            int64_t col;

            for (col = ciTaps; col < col8; col += 8) // input cols counter iTaps
            {
                float* pfColStart = pfInpRowSamplesFloatBufStart + col;

                float* pfCurrKernel_pos = pfCurrKernel;

                for (int64_t k_row = 0; k_row < ciKS; k_row++) 
                {
                    float* pfProc = vpfThreadVector[k_row] + col * ciMul;// *iMul;

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

                    pfCurrKernel_pos += ciKS;// +=iKernelSize; // point to next kernel row now
                } // k_row

            } // col

            // need to process last up to 7 cols separately...
            for (col = col8 + ciTaps; col < iWidthEl - ciTaps; col++) // input cols counter +iT - iT
            {
                float* pfCurrKernel_pos = pfCurrKernel;
                float* pfProc;

                for (int64_t k_row = 0; k_row < ciKS; k_row++) // iKS
                {
                    pfProc = vpfThreadVector[k_row] + col * ciMul; // *iMul;
                    for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                    {
                        *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart[col]), *(__m256*)(pfProc + k_col));
                    }

                    // need to process last (up to 7) floats separately..
                    for (int64_t k_col = k_col8; k_col < ciKS; ++k_col) // iKS
                    {
                        pfProc[k_col] += (pfCurrKernel_pos[k_col] * pfInpRowSamplesFloatBufStart[col]);
                    }
                    pfCurrKernel_pos += ciKS;// +=iKernelSize; // point to next kernel row now
                } // k_row

            } // col

            int iOutStartRow = (row - (ciTaps + ciKS)) * ciMul;
            //iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
            if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight)*ciMul && iThreadSkipRows <= 0)
            {
                // it looks vpfThreadVector uses copy of vector and it may be slower, TO DO - remake to pointer to vector
             //   ConvertiMulRowsToInt_avx2(vpfThreadVector, iInpWidth, iOutStartRow, dst, iDstStride);
                ConvertNRowsToInt_avx2(vpfThreadVector, iInpWidth, iOutStartRow, dst, iDstStride, ciMul);
            }

            iThreadSkipRows--;

            // circulate pointers to iMul rows upper
            std::rotate(vpfThreadVector.begin(), vpfThreadVector.begin() + ciMul, vpfThreadVector.end());

            // clear last iMul rows
            for (int i = ciKS - ciMul; i < ciKS; i++)
            {
                memset(vpfThreadVector[i], 0, iWidthEl * ciMul * sizeof(float));
            }
        } // row
    } // parallel section
}

void JincResize::KernelRowAll_avx2_mul4_taps4_cb_mt_2r(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride, const int ciMul, const int ciTaps, const int ciKS)
{

    // current input plane sizes
    iWidthEl = iInpWidth + 2 * ciKS;
    iHeightEl = iInpHeight + 2 * ciKS;

    const int k_col8 = ciKS - (ciKS % 8);
    const int col8 = iWidthEl - ciTaps - ((iWidthEl - ciTaps) % 8);
    float* pfCurrKernel = g_pfKernel;

    const int64_t iNumRowsPerThread = (iHeightEl - 2 * ciTaps) / threads_;

    // initial clearing
    for (int j = 0; j < threads_; j++)
    {
        for (int i = 0; i < ciKS + ciMul; i++)
        {
            memset(vpvThreadsVectors[j][i], 0, iWidthEl * ciMul * sizeof(float));
        }
    }

#pragma omp parallel num_threads(threads_)
    {
        int64_t tidx = omp_get_thread_num(); // our thread id here

        std::vector<float*> vpfThreadVector = vpvThreadsVectors[tidx];
        // it looks vpfThreadVector uses copy of vector from vpvThreadsVectors and it may be slower, TO DO - remake to pointer to vector

        // calculate rows to process in this thread
        int64_t iStartRow = tidx * iNumRowsPerThread;
        int64_t iThreadSkipRows = 8 /*ciTaps * 2*/;
        // some check
        if (iStartRow < ciTaps) iStartRow = ciTaps;
        int64_t iEndRow = iStartRow + iNumRowsPerThread + 8/*2 * ciTaps*/;
        if (iEndRow > (iHeightEl - ciTaps - 8)) iEndRow = (iHeightEl - ciTaps - 8);  // !! temp untill align stripes
        // prepare float32 pre-converted row data for each threal separately

        for (int64_t row = iStartRow; row < iEndRow; row += 2)
        {

            float* pfInpRowSamplesFloatBufStart_r1 = pfInpFloatRow + tidx * (2 * iWidthEl);
            float* pfInpRowSamplesFloatBufStart_r2 = pfInpFloatRow + tidx * (2 * iWidthEl) + iWidthEl;
            GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart_r1, ciKS);
            GetInpElRowAsFloat_avx2(row + 1, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart_r2, ciKS);

            int64_t col;

            for (col = 4/*ciTaps*/; col < col8; col += 8) // input cols counter, iTaps=4
            {
                float* pfColStart1 = pfInpRowSamplesFloatBufStart_r1 + col;
                float* pfColStart2 = pfInpRowSamplesFloatBufStart_r2 + col;

                float* pfCurrKernel_pos1 = pfCurrKernel;
                float* pfCurrKernel_pos2 = pfCurrKernel;

                __m256 my_ymm0, my_ymm1, my_ymm2, my_ymm3, my_ymm4, my_ymm5, my_ymm6, my_ymm7; // out samples
                __m256 my_ymm8, my_ymm9, my_ymm10, my_ymm11; // inp samples
                __m256 my_ymm12, my_ymm13, my_ymm14, my_ymm15; // kernel samples

#pragma unroll (2)
                for (int64_t k_row = 0; k_row < 4/*ciMul*/; k_row++) // 1st row only proc, iMul=4
                {
                    float* pfProc = vpfThreadVector[k_row] + col * ciMul; // *iMul

                    my_ymm12 = _mm256_load_ps(pfCurrKernel_pos1);
                    my_ymm13 = _mm256_load_ps(pfCurrKernel_pos1 + 8);
                    my_ymm14 = _mm256_load_ps(pfCurrKernel_pos1 + 16);
                    my_ymm15 = _mm256_load_ps(pfCurrKernel_pos1 + 24);

                    my_ymm0 = _mm256_load_ps(pfProc);
                    my_ymm1 = _mm256_load_ps(pfProc + 8);
                    my_ymm2 = _mm256_load_ps(pfProc + 16);
                    my_ymm3 = _mm256_load_ps(pfProc + 24);
                    my_ymm4 = _mm256_load_ps(pfProc + 32);
                    my_ymm5 = _mm256_load_ps(pfProc + 40);
                    my_ymm6 = _mm256_load_ps(pfProc + 48);

                    my_ymm8 = _mm256_broadcast_ss(pfColStart1 + 0); // 1
                    my_ymm9 = _mm256_broadcast_ss(pfColStart1 + 2); // 3
                    my_ymm10 = _mm256_broadcast_ss(pfColStart1 + 4); // 5
                    my_ymm11 = _mm256_broadcast_ss(pfColStart1 + 6); // 7 

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
                    my_ymm8 = _mm256_broadcast_ss(pfColStart1 + 1); // 2
                    my_ymm9 = _mm256_broadcast_ss(pfColStart1 + 3); // 4
                    my_ymm10 = _mm256_broadcast_ss(pfColStart1 + 5); // 6
                    my_ymm11 = _mm256_broadcast_ss(pfColStart1 + 7); // 8 

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
                    _mm256_store_ps(pfProc + 28, my_ymm3);
                    _mm256_store_ps(pfProc + 36, my_ymm4);
                    _mm256_store_ps(pfProc + 44, my_ymm5);
                    _mm256_store_ps(pfProc + 52, my_ymm6);

                    pfCurrKernel_pos1 += 32/*ciKS*/;// point to next kernel row now, +iKS
                } // k_row

#pragma unroll (2)
                for (int64_t k_row = 4/*ciMul*/; k_row < 32/*ciKS*/; k_row++) // 1st and 2nd rows proc, iMul to iKS
                {
                    float* pfProc = vpfThreadVector[k_row] + col * 4/*ciMul*/; // *iMul

                    // 1st row

                    my_ymm12 = _mm256_load_ps(pfCurrKernel_pos1);
                    my_ymm13 = _mm256_load_ps(pfCurrKernel_pos1 + 8);
                    my_ymm14 = _mm256_load_ps(pfCurrKernel_pos1 + 16);
                    my_ymm15 = _mm256_load_ps(pfCurrKernel_pos1 + 24);

                    my_ymm0 = _mm256_load_ps(pfProc);
                    my_ymm1 = _mm256_load_ps(pfProc + 8);
                    my_ymm2 = _mm256_load_ps(pfProc + 16);
                    my_ymm3 = _mm256_load_ps(pfProc + 24);
                    my_ymm4 = _mm256_load_ps(pfProc + 32);
                    my_ymm5 = _mm256_load_ps(pfProc + 40);
                    my_ymm6 = _mm256_load_ps(pfProc + 48);

                    my_ymm8 = _mm256_broadcast_ss(pfColStart1 + 0); // 1
                    my_ymm9 = _mm256_broadcast_ss(pfColStart1 + 2); // 3
                    my_ymm10 = _mm256_broadcast_ss(pfColStart1 + 4); // 5
                    my_ymm11 = _mm256_broadcast_ss(pfColStart1 + 6); // 7 

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

                    // 2nd row

                    my_ymm12 = _mm256_load_ps(pfCurrKernel_pos2);
                    my_ymm13 = _mm256_load_ps(pfCurrKernel_pos2 + 8);
                    my_ymm14 = _mm256_load_ps(pfCurrKernel_pos2 + 16);
                    my_ymm15 = _mm256_load_ps(pfCurrKernel_pos2 + 24);

                    my_ymm8 = _mm256_broadcast_ss(pfColStart2 + 0); // 1
                    my_ymm9 = _mm256_broadcast_ss(pfColStart2 + 2); // 3
                    my_ymm10 = _mm256_broadcast_ss(pfColStart2 + 4); // 5
                    my_ymm11 = _mm256_broadcast_ss(pfColStart2 + 6); // 7 

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

                    // 2nd row to not reload kernel again

                    // even samples
                    my_ymm8 = _mm256_broadcast_ss(pfColStart2 + 1); // 2
                    my_ymm9 = _mm256_broadcast_ss(pfColStart2 + 3); // 4
                    my_ymm10 = _mm256_broadcast_ss(pfColStart2 + 5); // 6
                    my_ymm11 = _mm256_broadcast_ss(pfColStart2 + 7); // 8 

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

                    // 1st row reload 1st row kernel again

                    my_ymm12 = _mm256_load_ps(pfCurrKernel_pos1);
                    my_ymm13 = _mm256_load_ps(pfCurrKernel_pos1 + 8);
                    my_ymm14 = _mm256_load_ps(pfCurrKernel_pos1 + 16);
                    my_ymm15 = _mm256_load_ps(pfCurrKernel_pos1 + 24);

                    // even samples
                    my_ymm8 = _mm256_broadcast_ss(pfColStart1 + 1); // 2
                    my_ymm9 = _mm256_broadcast_ss(pfColStart1 + 3); // 4
                    my_ymm10 = _mm256_broadcast_ss(pfColStart1 + 5); // 6
                    my_ymm11 = _mm256_broadcast_ss(pfColStart1 + 7); // 8 

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
                    _mm256_store_ps(pfProc + 28, my_ymm3);
                    _mm256_store_ps(pfProc + 36, my_ymm4);
                    _mm256_store_ps(pfProc + 44, my_ymm5);
                    _mm256_store_ps(pfProc + 52, my_ymm6);

                    pfCurrKernel_pos1 += 32/*ciKS*/;// point to next kernel row now
                    pfCurrKernel_pos2 += 32/*ciKS*/;// point to next kernel row now
                } // k_row

#pragma unroll (2)
                for (int64_t k_row = 32/*ciKS*/; k_row < 36/*ciKS + ciMul*/; k_row++) // 2nd row only proc, iKS to iKS+iMul
                {
                    float* pfProc = vpfThreadVector[k_row] + col * 4/*ciMul*/; // *iMul

                    // 2nd row

                    my_ymm12 = _mm256_load_ps(pfCurrKernel_pos2);
                    my_ymm13 = _mm256_load_ps(pfCurrKernel_pos2 + 8);
                    my_ymm14 = _mm256_load_ps(pfCurrKernel_pos2 + 16);
                    my_ymm15 = _mm256_load_ps(pfCurrKernel_pos2 + 24);

                    my_ymm0 = _mm256_load_ps(pfProc);
                    my_ymm1 = _mm256_load_ps(pfProc + 8);
                    my_ymm2 = _mm256_load_ps(pfProc + 16);
                    my_ymm3 = _mm256_load_ps(pfProc + 24);
                    my_ymm4 = _mm256_load_ps(pfProc + 32);
                    my_ymm5 = _mm256_load_ps(pfProc + 40);
                    my_ymm6 = _mm256_load_ps(pfProc + 48);

                    my_ymm8 = _mm256_broadcast_ss(pfColStart2 + 0); // 1
                    my_ymm9 = _mm256_broadcast_ss(pfColStart2 + 2); // 3
                    my_ymm10 = _mm256_broadcast_ss(pfColStart2 + 4); // 5
                    my_ymm11 = _mm256_broadcast_ss(pfColStart2 + 6); // 7 

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

                    // 2nd row to not reload kernel again

                    // even samples
                    my_ymm8 = _mm256_broadcast_ss(pfColStart2 + 1); // 2
                    my_ymm9 = _mm256_broadcast_ss(pfColStart2 + 3); // 4
                    my_ymm10 = _mm256_broadcast_ss(pfColStart2 + 5); // 6
                    my_ymm11 = _mm256_broadcast_ss(pfColStart2 + 7); // 8 

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
                    _mm256_store_ps(pfProc + 28, my_ymm3);
                    _mm256_store_ps(pfProc + 36, my_ymm4);
                    _mm256_store_ps(pfProc + 44, my_ymm5);
                    _mm256_store_ps(pfProc + 52, my_ymm6);

                    pfCurrKernel_pos2 += 32/*ciKS*/;// point to next kernel row now, +iKS
                } // k_row

            } // col 
              /*
            // need to process last up to 7 cols separately...
            for (col = col8 + iTaps; col < iWidthEl - iTaps; col++) // input cols counter
            {
                float* pfCurrKernel_pos = pfCurrKernel;
                float* pfProc;

                for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
                {
                    pfProc = vpfThreadVector[k_row] + col * iMul;
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
            */
            int iOutStartRow = (row - (ciTaps + ciKS)) * ciMul;
            //iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
            if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight)*ciMul && iThreadSkipRows <= 0)
            {
                // it looks vpfThreadVector uses copy of vector and it may be slower, TO DO - remake to pointer to vector
             //   ConvertiMulRowsToInt_avx2(vpfThreadVector, iInpWidth, iOutStartRow, dst, iDstStride);
                ConvertNRowsToInt_avx2(vpfThreadVector, iInpWidth, iOutStartRow, dst, iDstStride, ciMul * 2);
            }

            iThreadSkipRows--;

            // circulate pointers to iMul rows upper
            std::rotate(vpfThreadVector.begin(), vpfThreadVector.begin() + ciMul * 2, vpfThreadVector.end());

            // clear last iMul * 2 rows
            for (int i = ciKS - ciMul; i < ciKS + ciMul; i++)
            {
                memset(vpfThreadVector[i], 0, iWidthEl * ciMul * sizeof(float));
            }
        } // row
    } // parallel section
}



void JincResize::KernelRowAll_avx2_mul2_taps4_cb(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride, const int ciMul, const int ciTaps, const int ciKS)
{
	// current input plane sizes
	iWidthEl = iInpWidth + 32/*2 * iKernelSize*/;
	iHeightEl = iInpHeight + 32/*2 * iKernelSize*/;

	const int k_col8 = 16/*iKernelSize*/ - (16/*iKernelSize*/ % 8);
    const int col20 = iWidthEl - 4/*iTaps*/ - ((iWidthEl - 4/*iTaps*/) % 20);
	float* pfCurrKernel = g_pfKernel;
    int64_t col;

	memset(pfFilteredCirculatingBuf, 0, iWidthEl * iKernelSize * 2/*iMul*/ * sizeof(float));

	// still no MT for now - to be done later 
	for (int64_t row = 4/*iTaps*/; row < iHeightEl - 4/*iTaps*/; row++) // input lines counter
	{
		// start all row-only dependent ptrs here 
		// prepare float32 pre-converted row data for each threal separately
		float* pfInpRowSamplesFloatBufStart = pfInpFloatRow; 
        GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart, 16/*iKernelSize*/);

		for (col = 4/*iTaps*/; col < col20; col += 20) // input cols counter
		{
            float* pfColStart = pfInpRowSamplesFloatBufStart + col;

			float* pfCurrKernel_pos = pfCurrKernel;

			for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
			{
                float* pfProc = vpfRowsPointers[k_row] + col * 2/*iMul*/;

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

				pfCurrKernel_pos += 16/*iKernelSize*/; // point to next kernel row now
			} // k_row

		} // col

        // need to process last up to 19 cols separately...
        for (col = col20 + 4/*iTaps*/; col < iWidthEl - 4/*iTaps*/; col++) // input cols counter
        {
            float* pfCurrKernel_pos = pfCurrKernel;
            float* pfProc;

            for (int64_t k_row = 0; k_row < 16/*iKernelSize*/; k_row++)
            {
                pfProc = vpfRowsPointers[k_row] + col * 2/*iMul*/;
                for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                {
                    *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart[col]), *(__m256*)(pfProc + k_col));
                }

                // need to process last (up to 7) floats separately..
                for (int64_t k_col = k_col8; k_col < 16/*iKernelSize*/; ++k_col)
                {
                    pfProc[k_col] += (pfCurrKernel_pos[k_col] * pfInpRowSamplesFloatBufStart[col]);
                }
                pfCurrKernel_pos += 16/*iKernelSize*/; // point to next kernel row now
            } // k_row

        } // col

		int iOutStartRow = (row - (4 + 16/*iTaps + iKernelSize*/)) * 2/*iMul*/;
		//iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
		if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight) * 2/*iMul*/)
		{
			ConvertiMulRowsToInt_avx2(vpfRowsPointers, iInpWidth, iOutStartRow, dst, iDstStride);
		}

		// circulate pointers to iMul rows upper
		std::rotate(vpfRowsPointers.begin(), vpfRowsPointers.begin() + 2/*iMul*/, vpfRowsPointers.end());

		// clear last iMul rows
		for (int i = 16-2/*iKernelSize - iMul*/; i < 16/*iKernelSize*/; i++)
		{
			memset(vpfRowsPointers[i], 0, iWidthEl * iMul * sizeof(float));
		}
	} // row
}

void JincResize::KernelRowAll_avx2_mul2_taps4_cb_mt(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride, const int ciMul, const int ciTaps, const int ciKS)
{
    // current input plane sizes
    iWidthEl = iInpWidth + 32/*2 * iKernelSize*/;
    iHeightEl = iInpHeight + 32/*2 * iKernelSize*/;

    const int k_col8 = 16/*iKernelSize*/ - (16/*iKernelSize*/ % 8);
    const int col20 = iWidthEl - 4/*iTaps*/ - ((iWidthEl - 4/*iTaps*/) % 20);
    const int64_t iNumRowsPerThread = (iHeightEl - 2 * 4/*iTaps*/) / threads_;

    // initial clearing
    for (int j = 0; j < threads_; j++)
    {
        for (int i = 0; i < 16/*iKernelSize*/; i++)
        {
            memset(vpvThreadsVectors[j][i], 0, iWidthEl * 2/*iMul*/ * sizeof(float));
        }
    }

#pragma omp parallel num_threads(threads_)
    {
        int64_t tidx = omp_get_thread_num(); // our thread id here

        std::vector<float*> vpfThreadVector = vpvThreadsVectors[tidx];
        // it looks vpfThreadVector uses copy of vector from vpvThreadsVectors and it may be slower, TO DO - remake to pointer to vector

        // calculate rows to process in this thread
        int64_t iStartRow = tidx * iNumRowsPerThread;
        int64_t iThreadSkipRows = 8/*iTaps * 2*/;
        // some check
        if (iStartRow < 4/*iTaps*/) iStartRow = 4/*iTaps*/;
        int64_t iEndRow = iStartRow + iNumRowsPerThread + 8/*2 * iTaps*/;
        if (iEndRow > iHeightEl - 4/*iTaps*/) iEndRow = iHeightEl - 4/*iTaps*/;
        // prepare float32 pre-converted row data for each thread separately

        for (int64_t row = iStartRow; row < iEndRow; row++)
        {
            // start all row-only dependent ptrs here 
            // prepare float32 pre-converted row data for each threal separately
            float* pfInpRowSamplesFloatBufStart = pfInpFloatRow + tidx * iWidthEl;
            GetInpElRowAsFloat_avx2(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart, 16/*iKernelSize*/);

            int64_t col;

            for (col = 4/*iTaps*/; col < col20; col += 20) // input cols counter
            {
                float* pfColStart = pfInpRowSamplesFloatBufStart + col;

                float* pfCurrKernel_pos = g_pfKernel;

                for (int64_t k_row = 0; k_row < 16/*iKernelSize*/; k_row++)
                {
                    float* pfProc = vpfThreadVector[k_row] + col * 2/*iMul*/;

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

                    pfCurrKernel_pos += 16/*iKernelSize*/; // point to next kernel row now
                } // k_row

            } // col

            // need to process last up to 19 cols separately...
            for (col = col20 + 4/*iTaps*/; col < iWidthEl - 4/*iTaps*/; col++) // input cols counter
            {
                float* pfCurrKernel_pos = g_pfKernel;
                float* pfProc;

                for (int64_t k_row = 0; k_row < 16/*KernelSize*/; k_row++)
                {
                    pfProc = vpfThreadVector[k_row] + col * 2/*iMul*/;
                    for (int64_t k_col = 0; k_col < k_col8; k_col += 8)
                    {
                        *(__m256*)(pfProc + k_col) = _mm256_fmadd_ps(*(__m256*)(pfCurrKernel_pos + k_col), _mm256_broadcast_ss(&pfInpRowSamplesFloatBufStart[col]), *(__m256*)(pfProc + k_col));
                    }

                    // need to process last (up to 7) floats separately..
                    for (int64_t k_col = k_col8; k_col < 16/*iKernelSize*/; ++k_col)
                    {
                        pfProc[k_col] += (pfCurrKernel_pos[k_col] * pfInpRowSamplesFloatBufStart[col]);
                    }
                    pfCurrKernel_pos += 16/*iKernelSize*/; // point to next kernel row now
                } // k_row

            } // col

            int iOutStartRow = (row - (4 + 16/*iTaps + iKernelSize*/)) * 2/*iMul*/;
            //iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
            if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight) * 2/*iMul*/ && iThreadSkipRows <= 0)
            {
                // it looks vpfThreadVector uses copy of vector and it may be slower, TO DO - remake to pointer to vector
                ConvertiMulRowsToInt_avx2(vpfThreadVector, iInpWidth, iOutStartRow, dst, iDstStride);
            }

            iThreadSkipRows--;

            // circulate pointers to iMul rows upper
            std::rotate(vpfThreadVector.begin(), vpfThreadVector.begin() + 2/*iMul*/, vpfThreadVector.end());

            // clear last iMul rows
            for (int i = (16 - 2)/*(iKernelSize - iMul)*/; i < 16/*iKernelSize*/; i++)
            {
                memset(vpfThreadVector[i], 0, iWidthEl * 2/*iMul*/ * sizeof(float));
            }
        } // row
    } // parallel section
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

__forceinline void JincResize::ConvertiMulRowsToInt_avx2(std::vector<float*>Vector, int iInpWidth, int iOutStartRow, unsigned char* dst, int iDstStride)
{
	const int col32 = (iInpWidth*iMul) - ((iInpWidth*iMul) % 32);

	__m256 my_zero_ymm2;

	int row_float_buf_index = 0;

	for (int64_t row = iOutStartRow; row < iOutStartRow + iMul; row++)
	{
		my_zero_ymm2 = _mm256_setzero_ps();

		int64_t col = 0;
//		float* pfProc = vpfRowsPointers[row_float_buf_index] + (iKernelSize + iTaps) * iMul;
        float* pfProc = Vector[row_float_buf_index] + (iKernelSize + iTaps) * iMul;
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

__forceinline void JincResize::ConvertNRowsToInt_avx2(std::vector<float*>const& Vector, int iInpWidth, int iOutStartRow, unsigned char* dst, int iDstStride, int iNumRows)
{
    const int col32 = (iInpWidth * iMul) - ((iInpWidth * iMul) % 32);

    __m256 my_zero_ymm2;

    int row_float_buf_index = 0;

    for (int64_t row = iOutStartRow; row < iOutStartRow + iNumRows; row++)
    {
        my_zero_ymm2 = _mm256_setzero_ps();

        int64_t col = 0;
        float* pfProc = Vector[row_float_buf_index] + (iKernelSize + iTaps) * iMul;
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

__forceinline void JincResize::GetInpElRowAsFloat_avx2(int iInpRow, int iCurrInpHeight, int iCurrInpWidth, unsigned char* pCurr_src, int iCurrSrcStride, float* dst, const int ciKS)
{
    int64_t col;

    const int64_t col8 = ciKS - ciKS % 8;
    // row range from iTaps to iHeightEl - iTaps - 1
    // use iWidthEl and iHeightEl set in KernelRow()

    if (iInpRow < ciKS) iInpRow = ciKS;
    if (iInpRow > (iCurrInpHeight + ciKS - 1)) iInpRow = iCurrInpHeight + ciKS - 1;

    unsigned char* pCurrRowSrc = pCurr_src + (iInpRow - ciKS) * iCurrSrcStride;

    // start cols
    __m256 my_ymm_start = _mm256_setzero_ps();
    my_ymm_start = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm_start), *pCurrRowSrc));

    for (col = ciKS - 8; col >= 0; col -= 8) // left
    {
        _mm256_store_ps(dst + col, my_ymm_start);
    }

    // mid cols
    unsigned char* pCurrColSrc = pCurrRowSrc;
    for (col = ciKS; col < ciKS + iCurrInpWidth; col += 8)
    {
        __m256 src_ps = _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(_mm_loadu_si128((__m128i*)pCurrColSrc)));
        _mm256_store_ps(dst + col, src_ps);
        pCurrColSrc += 8;
    }

    // end cols
    __m256 my_ymm_end = _mm256_setzero_ps();
    my_ymm_end = _mm256_broadcastss_ps(_mm_cvt_si2ss(_mm256_castps256_ps128(my_ymm_start), pCurrRowSrc[iCurrInpWidth - 1]));
    for (col = ciKS + iCurrInpWidth; col < (2 * ciKS + iCurrInpWidth); col += 8) // right
    {
        _mm256_store_ps(dst + col, my_ymm_end);
    }

}
