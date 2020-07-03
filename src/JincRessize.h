#ifndef __JINCRESIZE_H__
#define __JINCRESIZE_H__

#include <immintrin.h>

#include "avisynth.h"
#include "avs/minmax.h"

#define VS_RESTRICT __restrict

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

template <typename T>
void resize_plane_sse41(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch);
template <typename T>
void resize_plane_avx2(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch);
template <typename T>
void resize_plane_avx512(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch);

class JincResize : public GenericVideoFilter
{
    int w, h;
    int _opt;
    Lut* init_lut;
    EWAPixelCoeff* out[3]{};
    bool avx512, avx2, sse41;
    int planecount;
    bool has_at_least_v8;
    float peak;

    void(*process_frame)(EWAPixelCoeff* coeff, const void* src_, void* VS_RESTRICT dst_, int dst_width, int dst_height, int src_pitch, int dst_pitch);

public:
    JincResize(PClip _child, int target_width, int target_height, double crop_left, double crop_top, double crop_width, double crop_height, int quant_x, int quant_y, int tap, double blur, int opt, IScriptEnvironment* env);
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
