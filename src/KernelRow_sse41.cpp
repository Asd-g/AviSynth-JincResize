#include "JincRessize.h"

void JincResize::KernelRowAll_sse2_mul(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride)
{
	// current input plane sizes
	iWidthEl = iInpWidth + 2 * iKernelSize;
	iHeightEl = iInpHeight + 2 * iKernelSize;

	const int k_col4 = iKernelSize - (iKernelSize % 4);

	memset(g_pfFilteredImageBuffer, 0, iWidthEl * iHeightEl * iMul * iMul * sizeof(float));

#pragma omp parallel for num_threads(threads_) 
	for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
	{
		// start all row-only dependent ptrs here
		int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iWidthEl * iMul - (iKernelSize / 2);

		// prepare float32 pre-converted row data for each threal separately
		int64_t tidx = omp_get_thread_num();
		float* pfInpRowSamplesFloatBufStart = pfInpFloatRow + tidx * iWidthEl;
		(this->*GetInpElRowAsFloat)(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);

		float* pfProc;// = g_pfFilteredImageBuffer + iProcPtrRowStart;

		for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
		{
			float* pfCurrKernel_pos = g_pfKernel;

			pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

			for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
			{
				for (int64_t k_col = 0; k_col < k_col4; k_col += 4)
				{
					*(__m128*)(pfProc + k_col) = _mm_add_ps(_mm_mul_ps(*(__m128*)(pfCurrKernel_pos + k_col), _mm_load_ps1(pfInpRowSamplesFloatBufStart + col)), *(__m128*)(pfProc + k_col));
				}

				// need to process last up to 3 floats separately..
				for (int64_t k_col = k_col4; k_col < iKernelSize; ++k_col)
					pfProc[k_col] += pfCurrKernel_pos[k_col];

				pfProc += iWidthEl * iMul; // point to next start point in output buffer now
				pfCurrKernel_pos += iKernelSize; // point to next kernel row now
			} // k_row

		} // col
	}

	ConvertToInt_sse2(iInpWidth, iInpHeight, dst, iDstStride);

}

void JincResize::KernelRowAll_sse2_mul_cb(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride)
{
	// current input plane sizes
	iWidthEl = iInpWidth + 2 * iKernelSize;
	iHeightEl = iInpHeight + 2 * iKernelSize;

	// single threaded for now, small memory use so can be used frame-based MT in AvisynthMT
	const int k_col4 = iKernelSize - (iKernelSize % 4);

	memset(pfFilteredCirculatingBuf, 0, iWidthEl * iKernelSize * iMul * sizeof(float));

	for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
	{
		// start all row-only dependent ptrs here

		// prepare float32 pre-converted row data 
//		int64_t tidx = 1;// omp_get_thread_num();
		float* pfInpRowSamplesFloatBufStart = pfInpFloatRow;// +tidx * iWidthEl;
		(this->*GetInpElRowAsFloat)(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);

		float* pfProc;

		for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
		{
			float* pfCurrKernel_pos = g_pfKernel;

			for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
			{
				pfProc = vpfRowsPointers[k_row] + col * iMul;

				for (int64_t k_col = 0; k_col < k_col4; k_col += 4)
				{
					*(__m128*)(pfProc + k_col) = _mm_add_ps(_mm_mul_ps(*(__m128*)(pfCurrKernel_pos + k_col), _mm_load_ps1(pfInpRowSamplesFloatBufStart + col)), *(__m128*)(pfProc + k_col));
				}

				// need to process last up to 3 floats separately.. 
				for (int64_t k_col = k_col4; k_col < iKernelSize; ++k_col)
					pfProc[k_col] += pfCurrKernel_pos[k_col];

				pfCurrKernel_pos += iKernelSize; // point to next kernel row now
			} // k_row

		} // col

		int iOutStartRow = (row - (iTaps + iKernelSize))*iMul;
		//iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
		if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight)*iMul)
		{
			ConvertiMulRowsToInt_sse2(iInpWidth, iOutStartRow, dst, iDstStride);
		}

		// circulate pointers to iMul rows upper
		std::rotate(vpfRowsPointers.begin(), vpfRowsPointers.begin() + iMul, vpfRowsPointers.end());

		// clear last iMul rows
		for (int i = iKernelSize - iMul; i < iKernelSize; i++)
		{
			memset(vpfRowsPointers[i], 0, iWidthEl*iMul * sizeof(float));
		}
	}
}

void JincResize::KernelRowAll_sse2_mul_cb_frw(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride)
{
	// current input plane sizes
	iWidthEl = iInpWidth + 2 * iKernelSize;
	iHeightEl = iInpHeight + 2 * iKernelSize;

	// single threaded for now, small memory use so can be used frame-based MT in AvisynthMT
	const int k_col4 = iKernelSize - (iKernelSize % 4);

	memset(pfFilteredCirculatingBuf, 0, iWidthEl * iKernelSize * iMul * sizeof(float));

	for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
	{
		// start all row-only dependent ptrs here

		// prepare float32 pre-converted row data 
		//		int64_t tidx = 1;// omp_get_thread_num();
		float* pfInpRowSamplesFloatBufStart = pfInpFloatRow;// +tidx * iWidthEl;
		(this->*GetInpElRowAsFloat)(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);

		for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
		{
			float* pfCurrKernel_pos = g_pfKernel + k_row * iKernelSize;

			for (int64_t col = iTaps; col < iWidthEl - iTaps; col++)
			{
				float *pfProc = vpfRowsPointers[k_row] + col * iMul;
				float fInpSample = pfInpRowSamplesFloatBufStart[col];

				for (int64_t k_col = 0; k_col < k_col4; k_col += 4)
				{
					*(__m128*)(pfProc + k_col) = _mm_add_ps(_mm_mul_ps(*(__m128*)(pfCurrKernel_pos + k_col), _mm_load_ps1(pfInpRowSamplesFloatBufStart + col)), *(__m128*)(pfProc + k_col));
				}

				// need to process last up to 3 floats separately.. 
				for (int64_t k_col = k_col4; k_col < iKernelSize; ++k_col)
					pfProc[k_col] += pfCurrKernel_pos[k_col];

			} // col
		} // r_row 

		int iOutStartRow = (row - (iTaps + iKernelSize))*iMul;
		//iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
		if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight)*iMul)
		{
			ConvertiMulRowsToInt_sse2(iInpWidth, iOutStartRow, dst, iDstStride);
		}

		// circulate pointers to iMul rows upper
		std::rotate(vpfRowsPointers.begin(), vpfRowsPointers.begin() + iMul, vpfRowsPointers.end());

		// clear last iMul rows
		for (int i = iKernelSize - iMul; i < iKernelSize; i++)
		{
			memset(vpfRowsPointers[i], 0, iWidthEl*iMul * sizeof(float));
		}
	}
}


void JincResize::KernelRowAll_sse2_mul2_taps4_cb(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride)
{
	// current input plane sizes
	iWidthEl = iInpWidth + 2 * iKernelSize;
	iHeightEl = iInpHeight + 2 * iKernelSize;

	// single threaded for now, small memory use so can be used frame-based MT in AvisynthMT
	const int k_col4 = iKernelSize - (iKernelSize % 4);

	memset(pfFilteredCirculatingBuf, 0, iWidthEl * iKernelSize * iMul * sizeof(float));

	for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
	{
		// start all row-only dependent ptrs here

		// prepare float32 pre-converted row data 
		//		int64_t tidx = 1;// omp_get_thread_num();
		float* pfInpRowSamplesFloatBufStart = pfInpFloatRow;// +tidx * iWidthEl;
		(this->*GetInpElRowAsFloat)(row, iInpHeight, iInpWidth, src, iSrcStride, pfInpRowSamplesFloatBufStart);

		float* pfProc;

		for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
		{
			float* pfCurrKernel_pos = g_pfKernel;

			for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
			{
				pfProc = vpfRowsPointers[k_row] + col * iMul;

				float* pfColStart = pfInpRowSamplesFloatBufStart + col;
				
				__m128 my_xmm0, my_xmm1, my_xmm2, my_xmm3, my_xmm4; // out samples
				__m128 my_xmm5, my_xmm6, my_xmm7, my_xmm8; // krn samples
				__m128 my_xmm9, my_xmm10;  // inp samples

				my_xmm5 = _mm_load_ps(pfCurrKernel_pos);
				my_xmm6 = _mm_load_ps(pfCurrKernel_pos + 4);
				my_xmm7 = _mm_load_ps(pfCurrKernel_pos + 8);
				my_xmm8 = _mm_load_ps(pfCurrKernel_pos + 12);
				
				my_xmm0 = _mm_loadu_ps(pfProc);
				my_xmm1 = _mm_loadu_ps(pfProc + 4);
				my_xmm2 = _mm_loadu_ps(pfProc + 8);
				my_xmm3 = _mm_loadu_ps(pfProc + 12);
				my_xmm4 = _mm_loadu_ps(pfProc + 16); 
				
				my_xmm9 = _mm_load_ps1(pfColStart + 0); // 1
				my_xmm10 = _mm_load_ps1(pfColStart + 2); // 3
				
				// 1st sample
				my_xmm0 = _mm_add_ps(my_xmm0, _mm_mul_ps(my_xmm9, my_xmm5));
				my_xmm1 = _mm_add_ps(my_xmm1, _mm_mul_ps(my_xmm9, my_xmm6));
				my_xmm2 = _mm_add_ps(my_xmm2, _mm_mul_ps(my_xmm9, my_xmm7));
				my_xmm3 = _mm_add_ps(my_xmm3, _mm_mul_ps(my_xmm9, my_xmm8));

				// 3rd sample
				my_xmm1 = _mm_add_ps(my_xmm1, _mm_mul_ps(my_xmm10, my_xmm5));
				my_xmm2 = _mm_add_ps(my_xmm2, _mm_mul_ps(my_xmm10, my_xmm6));
				my_xmm3 = _mm_add_ps(my_xmm3, _mm_mul_ps(my_xmm10, my_xmm7));
				my_xmm4 = _mm_add_ps(my_xmm4, _mm_mul_ps(my_xmm10, my_xmm8));
				
				_mm_storel_pi((__m64*)pfProc, my_xmm0);
				my_xmm0 = _mm_shuffle_ps(my_xmm0, my_xmm1, 78);
				my_xmm1 = _mm_shuffle_ps(my_xmm1, my_xmm2, 78);
				my_xmm2 = _mm_shuffle_ps(my_xmm2, my_xmm3, 78);
				my_xmm3 = _mm_shuffle_ps(my_xmm3, my_xmm4, 78);
				my_xmm4 = _mm_shuffle_ps(my_xmm4, my_xmm4, 78);
				my_xmm4 = _mm_loadh_pi(my_xmm4, (__m64*)(pfProc + 20)); 
				
				// even samples
				my_xmm9 = _mm_load_ps1(pfColStart + 1); // 2
				my_xmm10 = _mm_load_ps1(pfColStart + 3); // 4

				// 2nd sample
				my_xmm0 = _mm_add_ps(my_xmm0, _mm_mul_ps(my_xmm9, my_xmm5));
				my_xmm1 = _mm_add_ps(my_xmm1, _mm_mul_ps(my_xmm9, my_xmm6));
				my_xmm2 = _mm_add_ps(my_xmm2, _mm_mul_ps(my_xmm9, my_xmm7));
				my_xmm3 = _mm_add_ps(my_xmm3, _mm_mul_ps(my_xmm9, my_xmm8));

				// 4th sample
				my_xmm1 = _mm_add_ps(my_xmm1, _mm_mul_ps(my_xmm10, my_xmm5));
				my_xmm2 = _mm_add_ps(my_xmm2, _mm_mul_ps(my_xmm10, my_xmm6));
				my_xmm3 = _mm_add_ps(my_xmm3, _mm_mul_ps(my_xmm10, my_xmm7));
				my_xmm4 = _mm_add_ps(my_xmm4, _mm_mul_ps(my_xmm10, my_xmm8));
				
				_mm_storeu_ps(pfProc + 2, my_xmm0);
				_mm_storeu_ps(pfProc + 6, my_xmm1);
				_mm_storeu_ps(pfProc + 10, my_xmm2);
				_mm_storeu_ps(pfProc + 14, my_xmm3);
				_mm_storeu_ps(pfProc + 18, my_xmm4);
			
				pfCurrKernel_pos += iKernelSize; // point to next kernel row now
			} // k_row

		} // col

		int iOutStartRow = (row - (iTaps + iKernelSize))*iMul;
		//iMul rows ready - output result, skip iKernelSize+iTaps rows from beginning
		if (iOutStartRow >= 0 && iOutStartRow < (iInpHeight)*iMul)
		{
			ConvertiMulRowsToInt_sse2(iInpWidth, iOutStartRow, dst, iDstStride);
		}

		// circulate pointers to iMul rows upper
		std::rotate(vpfRowsPointers.begin(), vpfRowsPointers.begin() + iMul, vpfRowsPointers.end());

		// clear last iMul rows
		for (int i = iKernelSize - iMul; i < iKernelSize; i++)
		{
			memset(vpfRowsPointers[i], 0, iWidthEl*iMul * sizeof(float));
		}
	}
}



void JincResize::ConvertiMulRowsToInt_sse2(int iInpWidth, int iOutStartRow, unsigned char* dst, int iDstStride)
{
	const int col16 = (iInpWidth*iMul) - ((iInpWidth*iMul) % 16);

	int row_float_buf_index = 0;

	for (int row = iOutStartRow; row < iOutStartRow + iMul; row++)
	{
		__m128 my_zero_xmm2 = _mm_setzero_ps();

		float* pfProc =  vpfRowsPointers[row_float_buf_index] + (iKernelSize + iTaps) * iMul;
		unsigned char* pucDst = dst + row * iDstStride;

		for (int col = 0; col < col16; col += 16)
		{
			__m128 my_Val_xmm1 = _mm_load_ps(pfProc);
			__m128 my_Val_xmm2 = _mm_load_ps(pfProc + 4);
			__m128 my_Val_xmm3 = _mm_load_ps(pfProc + 8);
			__m128 my_Val_xmm4 = _mm_load_ps(pfProc + 12);

			my_Val_xmm1 = _mm_max_ps(my_Val_xmm1, my_zero_xmm2);
			my_Val_xmm2 = _mm_max_ps(my_Val_xmm2, my_zero_xmm2);
			my_Val_xmm3 = _mm_max_ps(my_Val_xmm3, my_zero_xmm2);
			my_Val_xmm4 = _mm_max_ps(my_Val_xmm4, my_zero_xmm2);

			__m128i my_iVal_xmm1 = _mm_cvtps_epi32(my_Val_xmm1);
			__m128i my_iVal_xmm2 = _mm_cvtps_epi32(my_Val_xmm2);
			__m128i my_iVal_xmm3 = _mm_cvtps_epi32(my_Val_xmm3);
			__m128i my_iVal_xmm4 = _mm_cvtps_epi32(my_Val_xmm4);

			__m128i my_iVal_12 = _mm_packs_epi32(my_iVal_xmm1, my_iVal_xmm2);
			__m128i my_iVal_34 = _mm_packs_epi32(my_iVal_xmm3, my_iVal_xmm4);

			__m128i my_iVal_1234 = _mm_packus_epi16(my_iVal_12, my_iVal_34);

			_mm_stream_si128((__m128i*)(pucDst), my_iVal_1234); //  hope dst is 16-bytes aligned

			pucDst += 16;
			pfProc += 16;
		}
		// last up to 15..
		for (int64_t l_col = col16; l_col < iInpWidth * iMul; ++l_col)
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
		} // l_col

		row_float_buf_index++;
	} // row

}

void JincResize::ConvertToInt_sse2(int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{

	const int col16 = (iInpWidth*iMul) - ((iInpWidth*iMul) % 16);

	__m128 my_zero_xmm2 = _mm_setzero_ps();

#pragma omp parallel for num_threads(threads_) 
	for (int64_t row = 0; row < iInpHeight * iMul; row++)
	{
		float* pfProc = g_pfFilteredImageBuffer + (row + iKernelSize * iMul) * iWidthEl * iMul + iKernelSize * iMul;
		unsigned char* pucDst = dst + row * iDstStride;

		for (int col = 0; col < col16; col += 16)
		{
			__m128 my_Val_xmm1 = _mm_load_ps(pfProc);
			__m128 my_Val_xmm2 = _mm_load_ps(pfProc + 4);
			__m128 my_Val_xmm3 = _mm_load_ps(pfProc + 8);
			__m128 my_Val_xmm4 = _mm_load_ps(pfProc + 12);

			my_Val_xmm1 = _mm_max_ps(my_Val_xmm1, my_zero_xmm2);
			my_Val_xmm2 = _mm_max_ps(my_Val_xmm2, my_zero_xmm2);
			my_Val_xmm3 = _mm_max_ps(my_Val_xmm3, my_zero_xmm2);
			my_Val_xmm4 = _mm_max_ps(my_Val_xmm4, my_zero_xmm2);

			__m128i my_iVal_xmm1 = _mm_cvtps_epi32(my_Val_xmm1);
			__m128i my_iVal_xmm2 = _mm_cvtps_epi32(my_Val_xmm2);
			__m128i my_iVal_xmm3 = _mm_cvtps_epi32(my_Val_xmm3);
			__m128i my_iVal_xmm4 = _mm_cvtps_epi32(my_Val_xmm4);

			__m128i my_iVal_12 = _mm_packs_epi32(my_iVal_xmm1, my_iVal_xmm2);
			__m128i my_iVal_34 = _mm_packs_epi32(my_iVal_xmm3, my_iVal_xmm4);

			__m128i my_iVal_1234 = _mm_packus_epi16(my_iVal_12, my_iVal_34);

			_mm_stream_si128((__m128i*)(pucDst), my_iVal_1234); // hope dst is 16-bytes aligned

			pucDst += 16;
			pfProc += 16;
		}
		// last up to 15..
		for (int64_t l_col = col16; l_col < iInpWidth * iMul; ++l_col)
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
		} // l_col
	} // row
}

void JincResize::GetInpElRowAsFloat_sse2(int iInpRow, int iCurrInpHeight, int iCurrInpWidth, unsigned char* pCurr_src, int iCurrSrcStride, float* dst)
{
	int64_t col;

	const int64_t col4 = iCurrInpWidth - iCurrInpWidth % 4;
	// row range from iTaps to iHeightEl - iTaps - 1

	if (iInpRow < iKernelSize) iInpRow = iKernelSize;
	if (iInpRow >(iCurrInpHeight + iKernelSize - 1)) iInpRow = iCurrInpHeight + iKernelSize - 1;

	unsigned char* pCurrRowSrc = pCurr_src + (iInpRow - iKernelSize) * iCurrSrcStride;

	// start cols
	__m128 my_xmm_start = _mm_setzero_ps();
	my_xmm_start = _mm_shuffle_ps(_mm_cvt_si2ss(my_xmm_start, *pCurrRowSrc), _mm_cvt_si2ss(my_xmm_start, *pCurrRowSrc), 0);

	for (col = iKernelSize - 4; col >= 0; col -= 4) // left
	{
		_mm_storeu_ps(dst + col, my_xmm_start);
	}

	// mid cols
	__m128i my_xmm_zero = _mm_setzero_si128();
	unsigned char *pCurrColSrc = pCurrRowSrc;
	for (col = iKernelSize; col < iKernelSize + iCurrInpWidth; col += 4)
	{
		__m128 src_ps = _mm_cvtepi32_ps(_mm_unpacklo_epi16(_mm_unpacklo_epi8(_mm_cvtsi32_si128(*(int*)pCurrColSrc), my_xmm_zero), my_xmm_zero));
		_mm_store_ps(dst + col, src_ps);
		pCurrColSrc += 4;
	}
	// last up to 3 cols
	pCurrColSrc -= 4;
	for (col = col4 + iKernelSize; col < iKernelSize + iCurrInpWidth; ++col)
	{
		*(dst + col) = (float)(*pCurrColSrc);
		pCurrColSrc++;
	}
	// end cols
	__m128 my_xmm_end = _mm_setzero_ps();
	my_xmm_end = _mm_shuffle_ps(_mm_cvt_si2ss(my_xmm_start, *(pCurrRowSrc + iCurrInpWidth - 1)), _mm_cvt_si2ss(my_xmm_start, *(pCurrRowSrc + iCurrInpWidth - 1)), 0);
	for (col = iKernelSize + iCurrInpWidth; col < iWidthEl; col += 4) // right
	{
		_mm_store_ps(dst + col, my_xmm_end);
	}

}

