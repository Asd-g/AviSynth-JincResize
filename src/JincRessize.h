#ifndef __JINCRESIZE_H__
#define __JINCRESIZE_H__

#include <immintrin.h>

#include "avisynth.h"
#include "avs/minmax.h"

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

class Lut
{
    int lut_size = 1024;

public:
    Lut();
    void InitLut(int lut_size, double radius, double blur);
    float GetFactor(int index);

    double* lut;
};

struct KrnRowUsefulRange
{
    int start_col, end_col;
};

class JincResize : public GenericVideoFilter
{
    Lut* init_lut;
    EWAPixelCoeff* out[3]{};
    int planecount;
    bool has_at_least_v8;
    float peak;
    int threads_;

    bool bAddProc;
    unsigned char *g_pElImageBuffer;
    float* g_pfImageBuffer = 0, * g_pfFilteredImageBuffer = 0;
    int64_t SzFilteredImageBuffer;
 //   float* pfEndOfFilteredImageBuffer;

    float *g_pfKernel = 0;
    float* g_pfKernelWeighted = 0;
    float* g_pfKernelParProc = 0;
    int iParProc;
    KrnRowUsefulRange* pKrnRUR;

    unsigned char ucFVal;

    int64_t iKernelSize;
 //   int64_t iKernelStride; // Kernel stride > Kernel line size to have place reading zeroes at SIMD wide registers loading
    int64_t iKernelStridePP; // kernel stride for parallel samples in row processing, aligned to size of SIMD register
    int64_t iMul;
    int64_t iTaps;
    int64_t iWidth, iHeight;
    int64_t iWidthEl, iHeightEl;
    
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
    void KernelProc(unsigned char *src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char *dst, int iDstStride);

    void KernelRow_c(int64_t iOutWidth);
    void KernelRow_c_mul(int64_t iOutWidth);
    void KernelRow_sse41(int64_t iOutWidth);
    void KernelRow_avx2(int64_t iOutWidth);
    void KernelRow_avx2_mul(int64_t iOutWidth);
    void KernelRow_avx2_mul2_taps8(int64_t iOutWidth);
    void KernelRow_avx2_mul8_taps2(int64_t iOutWidth);
    void KernelRow_avx2_mul8_taps3(int64_t iOutWidth);
    void KernelRow_avx2_mul4_taps4(int64_t iOutWidth);

    void KernelRow_avx512(int64_t iOutWidth);
    void(JincResize::* KernelRow)(int64_t iOutWidth);


public:
    JincResize(PClip _child, int target_width, int target_height, double crop_left, double crop_top, double crop_width, double crop_height, int quant_x, int quant_y, int tap, double blur, int threads, int opt, IScriptEnvironment* env);
    PVideoFrame __stdcall GetFrame(int n, IScriptEnvironment* env);
    int __stdcall SetCacheHints(int cachehints, int frame_range)
    {
        return cachehints == CACHE_GET_MTMODE ? MT_MULTI_INSTANCE : 0;
    }
    ~JincResize();
    
};

class Arguments
{
    AVSValue _args[12];
    const char* _arg_names[12];
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
