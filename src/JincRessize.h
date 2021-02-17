#ifndef __JINCRESIZE_H__
#define __JINCRESIZE_H__

#include <immintrin.h>
#include <string>
#include <vector>
#include <algorithm>

#include "math.h"
#include "avisynth.h"
#include "avs/minmax.h"
#include "omp.h"

struct EWAPixelCoeffMeta
{
    int start_x, start_y;
    int coeff_meta;
};

struct EWAPixelCoeff
{
    float* factor;
    EWAPixelCoeffMeta* meta;
    int* factor_map;
    int filter_size, quantize_x, quantize_y, coeff_stride;
};

enum Weighting
{
    JINC,
    TRAPEZOIDAL
};

class Lut
{
    int lut_size = 1024;

public:
    Lut();
    void InitLut(int lut_size, double radius, double blur, Weighting Weighting_type);
    float GetFactor(int index);

    double* lut;
};

class JincResize : public GenericVideoFilter
{
    int64_t iKernelSize; // reordered on suggestion from intel compiler
    int64_t iWidthEl;
    int64_t iMul;
    int64_t iTaps;

    Lut* init_lut;
    EWAPixelCoeff* out[3]{};
    int planecount;
    bool has_at_least_v8;
    float peak;
    int threads_;

    Weighting Weighting_type;

    bool bAddProc;
//    unsigned char *g_pElImageBuffer;
    float *g_pfImageBuffer = 0, *g_pfFilteredImageBuffer = 0, *pfInpFloatRow = 0;
    int64_t SzFilteredImageBuffer;
	float *pfFilteredCirculatingBuf = 0;
 //   float* pfEndOfFilteredImageBuffer;

    float *g_pfKernel = 0;
    float* g_pfKernelParProc = 0;
    int iParProc;
	std::vector<float*> vpfRowsPointers;

	// vector of vectors
	std::vector<std::vector<float*>> vpvThreadsVectors;

 //   int64_t iKernelStride; // Kernel stride > Kernel line size to have place reading zeroes at SIMD wide registers loading
    int64_t iKernelStridePP; // kernel stride for parallel samples in row processing, aligned to size of SIMD register

    int64_t iHeightEl;

    template<typename T>
    void resize_plane_c(EWAPixelCoeff* coeff[3], PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);
    template <typename T>
    void resize_plane_sse41(EWAPixelCoeff* coeff[3], PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);
    template <typename T>
    void resize_plane_avx2(EWAPixelCoeff* coeff[3], PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);
    template <typename T>
    void resize_plane_avx512(EWAPixelCoeff* coeff[3], PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env);

    void(JincResize::*process_frame)(EWAPixelCoeff**, PVideoFrame&, PVideoFrame&, IScriptEnvironment*);

    void fill2DKernel(void);

	void KernelRowAll_c_mul(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
	void KernelRowAll_c_mul_cb(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
	void KernelRowAll_c_mul_cb_mt(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
	void KernelRowAll_c_mul_cb_is(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
	void KernelRowAll_c_mul_cb_frw(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
	void KernelRowAll_c_mul_cb_nz(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);

	void KernelRowAll_sse2_mul(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
	void KernelRowAll_sse2_mul_cb(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
	void KernelRowAll_sse2_mul_cb_frw(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
	void KernelRowAll_sse2_mul_cb_frw_is(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
	void KernelRowAll_sse2_mul2_taps4_cb(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);

	void KernelRowAll_avx2_mul(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
	void KernelRowAll_avx2_mul_cb(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
    void KernelRowAll_avx2_mul_cb_mt(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);
    void KernelRowAll_avx2_mul4_taps4_cb(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);
    void KernelRowAll_avx2_mul4_taps4_cb_mt(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);
	void KernelRowAll_avx2_mul2_taps4_cb(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);
    void KernelRowAll_avx2_mul2_taps4_cb_mt(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);

    void KernelRowAll_avx512_mul(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);
    void KernelRowAll_avx512_mul_cb(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);
    void KernelRowAll_avx512_mul_cb_mt(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);

	void ConvertToInt_c(int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);
	void ConvertToInt_sse2(int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);
	void ConvertToInt_avx2(int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);

	void ConvertiMulRowsToInt_c(std::vector<float*>Vector, int  iInpWidth, int iOutStartRow, unsigned char* dst, int iDstStride);
	void ConvertiMulRowsToInt_sse2(std::vector<float*>Vector, int iInpWidth, int iOutStartRow, unsigned char* dst, int iDstStride);
	void ConvertiMulRowsToInt_avx2(std::vector<float*>Vector, int iInpWidth, int iOutStartRow, unsigned char* dst, int iDstStride);

	void GetInpElRowAsFloat_c(int iInpRow, int iCurrInpHeight, int iCurrInpWidth, unsigned char* pCurr_src, int iCurrSrcStride, float* dst);
    void GetInpElRowAsFloat_avx2(int iInpRow, int iCurrInpHeight, int iCurrInpWidth, unsigned char* pCurr_src, int iCurrSrcStride, float* dst);
	void GetInpElRowAsFloat_sse2(int iInpRow, int iCurrInpHeight, int iCurrInpWidth, unsigned char* pCurr_src, int iCurrSrcStride, float* dst);


	void(JincResize::* KernelProcAll)(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);
    void(JincResize::* ConvertToInt)(int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride);
    void(JincResize::* GetInpElRowAsFloat)(int iInpRow, int iCurrInpHeight, int iCurrInpWidth, unsigned char* pCurr_src, int iCurrSrcStride, float* dst); 


public:
    JincResize(PClip _child, int target_width, int target_height, double crop_left, double crop_top, double crop_width, double crop_height, int quant_x, int quant_y, int tap, double blur, int threads, int opt, int wt, int ap, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
    }
    ~JincResize();
    
};

class Arguments
{
    AVSValue _args[14];
    const char* _arg_names[14];
    int _idx;

public:
    Arguments() : _args{}, _arg_names{}, _idx{} {}

    void add(AVSValue arg, const char* arg_name = nullptr)
    {
        _args[_idx] = arg;
        _arg_names[_idx] = arg_name;
        ++_idx;
    }

    AVSValue args() const { return{ _args, _idx }; }

    const char* const* arg_names() const { return _arg_names; }
};

#endif
