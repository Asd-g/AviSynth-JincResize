#ifndef __JINCRESIZE_H__
#define __JINCRESIZE_H__

#include <execution>
#include <string>
#include <vector>

#include "avisynth_c.h"
#include "avs/minmax.h"

struct EWAPixelCoeffMeta
{
    int start_x;
    int start_y;
    int coeff_meta;
};

struct EWAPixelCoeff
{
    float* factor;
    EWAPixelCoeffMeta* meta;
    int* factor_map;
    int filter_size;
    int coeff_stride;
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

struct JincResize
{
    std::string cplace;
    Lut* init_lut;
    std::vector<EWAPixelCoeff*> out;
    int planecount;
    bool has_at_least_v8;
    float peak;

    template<typename T, int thr, int subsampled>
    void resize_plane_c(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
    template <typename T, int thr, int subsampled>
    void resize_plane_sse41(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
    template <typename T, int thr, int subsampled>
    void resize_plane_avx2(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);
    template <typename T, int thr, int subsampled>
    void resize_plane_avx512(AVS_VideoFrame* src, AVS_VideoFrame* dst, AVS_VideoInfo* vi);

    void(JincResize::* process_frame)(AVS_VideoFrame*, AVS_VideoFrame*, AVS_VideoInfo*);
};

#endif
