#include <vector>
#include <cmath>
#include <thread>
#include <string>

#include "JincRessize.h"

#ifndef M_PI // GCC seems to have it
constexpr double M_PI = 3.14159265358979323846;
#endif

// Taylor series coefficients of 2*BesselJ1(pi*x)/(pi*x) as (x^2) -> 0
static double jinc_taylor_series[31] =
{
     1.0,
    -1.23370055013616982735431137,
     0.507339015802096027273126733,
    -0.104317403816764804365258186,
     0.0128696438477519721233840271,
    -0.00105848577966854543020422691,
     6.21835470803998638484476598e-05,
    -2.73985272294670461142756204e-06,
     9.38932725442064547796003405e-08,
    -2.57413737759717407304931036e-09,
     5.77402672521402031756429343e-11,
    -1.07930605263598241754572977e-12,
     1.70710316782347356046974552e-14,
    -2.31434518382749184406648762e-16,
     2.71924659665997312120515390e-18,
    -2.79561335187943028518083529e-20,
     2.53599244866299622352138464e-22,
    -2.04487273140961494085786452e-24,
     1.47529860450204338866792475e-26,
    -9.57935105257523453155043307e-29,
     5.62764317309979254140393917e-31,
    -3.00555258814860366342363867e-33,
     1.46559362903641161989338221e-35,
    -6.55110024064596600335624426e-38,
     2.69403199029404093412381643e-40,
    -1.02265499954159964097119923e-42,
     3.59444454568084324694180635e-45,
    -1.17313973900539982313119019e-47,
     3.56478606255557746426034301e-50,
    -1.01100655781438313239513538e-52,
     2.68232117541264485328658605e-55
};

static double jinc_zeros[16] =
{
    1.2196698912665045,
    2.2331305943815286,
    3.2383154841662362,
    4.2410628637960699,
    5.2427643768701817,
    6.2439216898644877,
    7.2447598687199570,
    8.2453949139520427,
    9.2458926849494673,
    10.246293348754916,
    11.246622794877883,
    12.246898461138105,
    13.247132522181061,
    14.247333735806849,
    15.247508563037300,
    16.247661874700962
};

//  Modified from boost package math/tools/`rational.hpp`
//
//  (C) Copyright John Maddock 2006.
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
static double evaluate_rational(const double* num, const double* denom, double z, int count)
{
    double s1, s2;
    if (z <= 1.0)
    {
        s1 = num[count - 1];
        s2 = denom[count - 1];
        for (auto i = count - 2; i >= 0; --i)
        {
            s1 *= z;
            s2 *= z;
            s1 += num[i];
            s2 += denom[i];
        }
    }
    else
    {
        z = 1.0f / z;
        s1 = num[0];
        s2 = denom[0];
        for (auto i = 1; i < count; ++i)
        {
            s1 *= z;
            s2 *= z;
            s1 += num[i];
            s2 += denom[i];
        }
    }

    return s1 / s2;
}

//  Modified from boost package `BesselJ1.hpp`
//
//  Copyright (c) 2006 Xiaogang Zhang
//  Use, modification and distribution are subject to the
//  Boost Software License, Version 1.0. (See accompanying file
//  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
static double jinc_sqr_boost_l(double x2)
{
    static const double bPC[7] =
    {
        -4.4357578167941278571e+06,
        -9.9422465050776411957e+06,
        -6.6033732483649391093e+06,
        -1.5235293511811373833e+06,
        -1.0982405543459346727e+05,
        -1.6116166443246101165e+03,
        0.0
    };
    static const double bQC[7] =
    {
        -4.4357578167941278568e+06,
        -9.9341243899345856590e+06,
        -6.5853394797230870728e+06,
        -1.5118095066341608816e+06,
        -1.0726385991103820119e+05,
        -1.4550094401904961825e+03,
        1.0
    };
    static const double bPS[7] =
    {
        3.3220913409857223519e+04,
        8.5145160675335701966e+04,
        6.6178836581270835179e+04,
        1.8494262873223866797e+04,
        1.7063754290207680021e+03,
        3.5265133846636032186e+01,
        0.0
    };
    static const double bQS[7] =
    {
        7.0871281941028743574e+05,
        1.8194580422439972989e+06,
        1.4194606696037208929e+06,
        4.0029443582266975117e+05,
        3.7890229745772202641e+04,
        8.6383677696049909675e+02,
        1.0
    };

    auto y2 = M_PI * M_PI * x2;
    auto xp = sqrt(y2);
    auto y2p = 64.0 / y2;
    auto yp = 8.0 / xp;
    auto factor = sqrt(xp / M_PI) * 2.0 / y2;
    auto rc = evaluate_rational(bPC, bQC, y2p, 7);
    auto rs = evaluate_rational(bPS, bQS, y2p, 7);
    auto sx = sin(xp);
    auto cx = cos(xp);

    return factor * (rc * (sx - cx) + yp * rs * (sx + cx));
}

// jinc(sqrt(x2))
static double jinc_sqr(double x2)
{
    if (x2 < 1.49)        // the 1-tap radius
    {
        double res = 0.0;
        for (auto j = 16; j > 0; --j)
            res = res * x2 + jinc_taylor_series[j - 1];
        return res;
    }
    else if (x2 < 4.97)   // the 2-tap radius
    {
        double res = 0.0;
        for (auto j = 21; j > 0; --j)
            res = res * x2 + jinc_taylor_series[j - 1];
        return res;
    }
    else if (x2 < 10.49)  // the 3-tap radius
    {
        double res = 0.0;
        for (auto j = 26; j > 0; --j)
            res = res * x2 + jinc_taylor_series[j - 1];
        return res;
    }
    else if (x2 < 17.99)  // the 4-tap radius
    {
        double res = 0.0;
        for (auto j = 31; j > 0; --j)
            res = res * x2 + jinc_taylor_series[j - 1];
        return res;
    }
    else if (x2 < 52.57)  // the 5~7-tap radius
    {
        auto x = M_PI * sqrt(x2);
#if defined(GCC) || defined(CLANG)
        return 2.0 * _j1(x) / x;
#else
        return 2.0 * std::cyl_bessel_j(1, x) / x;
#endif
    }
    else if (x2 < 68.07)  // the 8-tap radius // Modify from pull request #4
    {
        return jinc_sqr_boost_l(x2);
    }
    else                  // the 9~16-tap radius
    {
        auto x = M_PI * sqrt(x2);
#if defined(GCC) || defined(CLANG)
        return 2.0 * _j1(x) / x;
#else
        return 2.0 * std::cyl_bessel_j(1, x) / x;
#endif
    }
}

static double sample_sqr(double (*filter)(double), double x2, double blur2, double radius2)
{
    if (blur2 > 0.0)
        x2 /= blur2;

    if (x2 < radius2)
        return filter(x2);

    return 0.0;
}

constexpr double JINC_ZERO_SQR = 1.48759464366204680005356;

Lut::Lut()
{
    lut = new double[lut_size];
}

void Lut::InitLut(int lut_size, double radius, double blur, Weighting Weighting_type)
{
    auto radius2 = radius * radius;
    auto blur2 = blur * blur;

    for (auto i = 0; i < lut_size; ++i)
    {
        auto t2 = i / (lut_size - 1.0);
        double filter = sample_sqr(jinc_sqr, radius2 * t2, blur2, radius2);
        if (Weighting_type == JINC)
        {
            double window = sample_sqr(jinc_sqr, JINC_ZERO_SQR * t2, 1.0, radius2);
            lut[i] = filter * window;
        }
        else if (Weighting_type == TRAPEZOIDAL)
        {
            // trapezoidal window
            double window = (2 - (2 * t2));

            if (window < 1)
            {
                lut[i] = filter * window;
            }
            else
                lut[i] = filter;
        }
        // else - assert Wt == undefined ?

    }
}

float Lut::GetFactor(int index)
{
    if (index >= lut_size)
        return 0.f;
    return static_cast<float>(lut[index]);
}

constexpr double DOUBLE_ROUND_MAGIC_NUMBER = 6755399441055744.0;

static void init_coeff_table(EWAPixelCoeff* out, int quantize_x, int quantize_y,
    int filter_size, int dst_width, int dst_height)
{
    out->filter_size = filter_size;
    out->quantize_x = quantize_x;
    out->quantize_y = quantize_y;
    out->coeff_stride = (filter_size + 15) & ~15;

    // Allocate metadata
    out->meta = new EWAPixelCoeffMeta[static_cast<int64_t>(dst_width) * dst_height];

    // Alocate factor map
    if (quantize_x > 0 && quantize_y > 0)
        out->factor_map = new int[static_cast<int64_t>(quantize_x) * quantize_y];
    else
        out->factor_map = nullptr;

    // This will be reserved to exact size in coff generating procedure
    out->factor = nullptr;

    // Zeroed memory
    if (out->factor_map != nullptr)
        memset(out->factor_map, 0, static_cast<int64_t>(quantize_x) * quantize_y * sizeof(int));

    memset(out->meta, 0, static_cast<int64_t>(dst_width) * dst_height * sizeof(EWAPixelCoeffMeta));
}

static void delete_coeff_table(EWAPixelCoeff* out)
{
    if (out == nullptr)
        return;

    _aligned_free(out->factor);
    delete[] out->meta;
    delete[] out->factor_map;
}

/* Coefficient table generation */
static void generate_coeff_table_c(Lut* func, EWAPixelCoeff* out, int quantize_x, int quantize_y,
    int samples, int src_width, int src_height, int dst_width, int dst_height, double radius,
    double crop_left, double crop_top, double crop_width, double crop_height)
{
    const double filter_scale_x = static_cast<double>(dst_width) / crop_width;
    const double filter_scale_y = static_cast<double>(dst_height) / crop_height;

    const double filter_step_x = min(filter_scale_x, 1.0);
    const double filter_step_y = min(filter_scale_y, 1.0);

    const float filter_support_x = static_cast<float>(radius / filter_step_x);
    const float filter_support_y = static_cast<float>(radius / filter_step_y);

    const int filter_size_x = static_cast<int>(ceil(filter_support_x * 2.0));
    const int filter_size_y = static_cast<int>(ceil(filter_support_y * 2.0));

    const float filter_support = max(filter_support_x, filter_support_y);
    const int filter_size = max(filter_size_x, filter_size_y);

    const float start_x = static_cast<float>(crop_left + (crop_width - dst_width) / (dst_width * static_cast<int64_t>(2)));
    const float start_y = static_cast<float>(crop_top + (crop_height - dst_height) / (dst_height * static_cast<int64_t>(2)));

    const float x_step = static_cast<float>(crop_width / dst_width);
    const float y_step = static_cast<float>(crop_height / dst_height);

    float xpos = start_x;
    float ypos = start_y;

    // Initialize EWAPixelCoeff data structure
    init_coeff_table(out, quantize_x, quantize_y, filter_size, dst_width, dst_height);

    std::vector<float> tmp_array;
    int tmp_array_top = 0;

    // Use to advance the coeff pointer
    const int coeff_per_pixel = out->coeff_stride * filter_size;

    for (int y = 0; y < dst_height; ++y)
    {
        for (int x = 0; x < dst_width; ++x)
        {
            bool is_border = false;

            EWAPixelCoeffMeta* meta = &out->meta[y * dst_width + x];

            // Here, the window_*** variable specified a begin/size/end
            // of EWA window to process.
            int window_end_x = static_cast<int>(xpos + filter_support);
            int window_end_y = static_cast<int>(ypos + filter_support);

            if (window_end_x >= src_width)
            {
                window_end_x = src_width - 1;
                is_border = true;
            }
            if (window_end_y >= src_height)
            {
                window_end_y = src_height - 1;
                is_border = true;
            }

            int window_begin_x = window_end_x - filter_size + 1;
            int window_begin_y = window_end_y - filter_size + 1;

            if (window_begin_x < 0)
            {
                window_begin_x = 0;
                is_border = true;
            }
            if (window_begin_y < 0)
            {
                window_begin_y = 0;
                is_border = true;
            }

            meta->start_x = window_begin_x;
            meta->start_y = window_begin_y;

            // Quantize xpos and ypos
            const int quantized_x_int = lrint(static_cast<double>(xpos) * quantize_x);
            const int quantized_y_int = lrint(static_cast<double>(ypos) * quantize_y);
            const int quantized_x_value = quantized_x_int % quantize_x;
            const int quantized_y_value = quantized_y_int % quantize_y;
            const float quantized_xpos = static_cast<float>(quantized_x_int) / quantize_x;
            const float quantized_ypos = static_cast<float>(quantized_y_int) / quantize_y;

            if (!is_border && out->factor_map[quantized_y_value * quantize_x + quantized_x_value] != 0)
            {
                // Not border pixel and already have coefficient calculated at this quantized position
                meta->coeff_meta = out->factor_map[quantized_y_value * quantize_x + quantized_x_value] - 1;
            }
            else
            {
                // then need computation
                float divider = 0.f;

                // This is the location of current target pixel in source pixel
                // Quantized
                const float current_x = clamp(is_border ? xpos : quantized_xpos, 0.f, src_width - 1.f);
                const float current_y = clamp(is_border ? ypos : quantized_ypos, 0.f, src_height - 1.f);

                if (!is_border)
                {
                    // Change window position to quantized position
                    window_begin_x = static_cast<int>(quantized_xpos + filter_support) - filter_size + 1;
                    window_begin_y = static_cast<int>(quantized_ypos + filter_support) - filter_size + 1;
                }

                // Windowing positon
                int window_x = window_begin_x;
                int window_y = window_begin_y;

                // First loop calcuate coeff
                tmp_array.resize(tmp_array.size() + coeff_per_pixel, 0.f);
                int curr_factor_ptr = tmp_array_top;

                const double radius2 = radius * radius;
                for (int ly = 0; ly < filter_size; ++ly)
                {
                    for (int lx = 0; lx < filter_size; ++lx)
                    {
                        // Euclidean distance to sampling pixel
                        const float dx = static_cast<float>((current_x - window_x) * filter_step_x);
                        const float dy = static_cast<float>((current_y - window_y) * filter_step_y);
                        const float dist = dx * dx + dy * dy;
                        double index_d = lround(static_cast<double>((samples - static_cast<int64_t>(1))) * dist / radius2) + DOUBLE_ROUND_MAGIC_NUMBER;
                        int index = *reinterpret_cast<int*>(&index_d);

                        const float factor = func->GetFactor(index);

                        tmp_array[curr_factor_ptr + static_cast<int64_t>(lx)] = factor;
                        divider += factor;

                        ++window_x;
                    }

                    curr_factor_ptr += out->coeff_stride;

                    window_x = window_begin_x;
                    ++window_y;
                }

                // Second loop to divide the coeff
                curr_factor_ptr = tmp_array_top;
                for (int ly = 0; ly < filter_size; ++ly)
                {
                    for (int lx = 0; lx < filter_size; ++lx)
                    {
                        tmp_array[curr_factor_ptr + static_cast<int64_t>(lx)] /= divider;
                    }

                    curr_factor_ptr += out->coeff_stride;
                }

                // Save factor to table
                if (!is_border)
                    out->factor_map[quantized_y_value * quantize_x + quantized_x_value] = tmp_array_top + 1;

                meta->coeff_meta = tmp_array_top;
                tmp_array_top += coeff_per_pixel;
            }

            xpos += x_step;
        }

        ypos += y_step;
        xpos = start_x;
    }

    // Copy from tmp_array to real array
    const int tmp_array_size = tmp_array.size();
    out->factor = static_cast<float*>(_aligned_malloc(tmp_array_size * sizeof(float), 64)); // aligned to cache line
    if (out->factor)
        memcpy(out->factor, &tmp_array[0], tmp_array_size * sizeof(float));
}

/* Planar resampling with coeff table */
template<typename T>
void JincResize::resize_plane_c(EWAPixelCoeff* coeff[3], PVideoFrame& src, PVideoFrame& dst, IScriptEnvironment* env)
{
    const int pixel_size = sizeof(T);
    const int planes_y[4] = { PLANAR_Y, PLANAR_U, PLANAR_V, PLANAR_A };
    const int planes_r[4] = { PLANAR_G, PLANAR_B, PLANAR_R, PLANAR_A };
    const int* current_planes = (vi.IsYUV() || vi.IsYUVA()) ? planes_y : planes_r;
    for (int i = 0; i < planecount; ++i)
    {
        const int plane = current_planes[i];

        const int src_stride = src->GetPitch(plane) / pixel_size;
        const int dst_stride = dst->GetPitch(plane) / pixel_size;
        const int dst_width = dst->GetRowSize(plane) / pixel_size;
        const int dst_height = dst->GetHeight(plane);
        const T* srcp = reinterpret_cast<const T*>(src->GetReadPtr(plane));

        if (bAddProc)
        {
            const int src_width = src->GetRowSize(plane) / pixel_size;
            const int src_height = src->GetHeight(plane);
            unsigned char *dstp_add = reinterpret_cast<unsigned char *>(dst->GetWritePtr(plane));
            KernelProc((unsigned char*)srcp, src_stride, src_width, src_height, (unsigned char*)dstp_add, dst_stride);
            continue;
        }

#pragma omp parallel for num_threads(threads_)
        for (int y = 0; y < dst_height; ++y)
        {
            T* __restrict dstp = reinterpret_cast<T*>(dst->GetWritePtr(plane)) + static_cast<int64_t>(y) * dst_stride;
 
            for (int x = 0; x < dst_width; ++x)
            {
                EWAPixelCoeffMeta* meta = coeff[i]->meta + static_cast<int64_t>(y) * dst_width + x;
                const T* src_ptr = srcp + meta->start_y * static_cast<int64_t>(src_stride) + meta->start_x;
                const float* coeff_ptr = coeff[i]->factor + meta->coeff_meta;

                float result = 0.f;

                for (int ly = 0; ly < coeff[i]->filter_size; ++ly)
                {
                    for (int lx = 0; lx < coeff[i]->filter_size; ++lx)
                        result += src_ptr[lx] * coeff_ptr[lx];

                    coeff_ptr += coeff[i]->coeff_stride;
                    src_ptr += src_stride;
                }

                if constexpr (std::is_integral_v<T>)
                    dstp[x] = static_cast<T>(lrintf(clamp(result, 0.f, peak)));
                else
                    dstp[x] = result;

            }
        }
    }
}


JincResize::JincResize(PClip _child, int target_width, int target_height, double crop_left, double crop_top, double crop_width, double crop_height, int quant_x, int quant_y, int tap, double blur, int threads, int opt, int wt, int ap, IScriptEnvironment* env)
    : GenericVideoFilter(_child), threads_(threads)
{
    if (!vi.IsPlanar())
        env->ThrowError("JincResize: clip must be in planar format.");
    if (tap < 1 || tap > 16)
        env->ThrowError("JincResize: tap must be between 1..16.");
    if (quant_x < 1 || quant_x > 256)
        env->ThrowError("JincResize: quant_x must be between 1..256.");
    if (quant_y < 1 || quant_y > 256)
        env->ThrowError("JincResize: quant_y must be between 1..256.");
    if (opt > 3)
        env->ThrowError("JincResize: opt higher than 3 is not allowed.");
    if (blur < 0.0 || blur > 10.0)
        env->ThrowError("JincResize: blur must be between 0.0..10.0.");
    if (opt == 3 && !(env->GetCPUFlags() & CPUF_AVX512F))
        env->ThrowError("JincResize: opt=3 requires AVX-512F.");
    if (opt == 2 && !(env->GetCPUFlags() & CPUF_AVX2) && !(env->GetCPUFlags() & CPUF_FMA3))
        env->ThrowError("JincResize: opt=2 requires AVX2 and FMA3.");
    if (opt == 1 && !(env->GetCPUFlags() & CPUF_SSE4_1))
        env->ThrowError("JincResize: opt=1 requires SSE4.1.");
    if (wt < 0 || wt > 1)
        env->ThrowError("JincResize: Weighting type must be between 0..1.");
    if (ap < 0 || ap > 1)
        env->ThrowError("JincResize: Proc type must be between 0..1.");

    const int thr = std::thread::hardware_concurrency();
    if (threads_ == 0)
        threads_ = thr;
    else if (threads_ < 0 || threads_ > thr)
    {
        const std::string msg = "JincResize: threads must be between 0.." + std::to_string(thr) + ".";
        env->ThrowError(msg.c_str());
    }

    has_at_least_v8 = true;
    try { env->CheckVersion(8); }
    catch (const AvisynthError&) { has_at_least_v8 = false; };

    if (crop_width <= 0.0)
        crop_width = vi.width - crop_left + crop_width;

    if (crop_height <= 0.0)
        crop_height = vi.height - crop_top + crop_height;

    int src_width = vi.width;
    int src_height = vi.height;
    vi.width = target_width;
    vi.height = target_height;
    blur = 1.0 - blur / 100.0;
    peak = static_cast<float>((1 << vi.BitsPerComponent()) - 1);
    double radius = jinc_zeros[tap - 1];
    int samples = 1024;  // should be a multiple of 4
 
    if (wt == 0)
        Weighting_type = JINC;
    else if (wt == 1) Weighting_type = TRAPEZOIDAL;

    init_lut = new Lut();
    init_lut->InitLut(samples, radius, blur, Weighting_type);
    int sub_w, sub_h;
    double div_w, div_h;
    planecount = min(vi.NumComponents(), 3);

    if (planecount > 1 && (!vi.IsRGB() || !vi.Is444()))
    {
        sub_w = vi.GetPlaneWidthSubsampling(PLANAR_U);
        sub_h = vi.GetPlaneHeightSubsampling(PLANAR_U);
        div_w = static_cast<double>(static_cast<int64_t>(1) << sub_w);
        div_h = static_cast<double>(static_cast<int64_t>(1) << sub_h);
    }
    else
    {
        sub_w = sub_h = 0;
        div_w = div_h = 0.0;
    }

    bAddProc = false;

    // check if target widh and height are both integer and same multipliers
    float fMultH = (float)(target_width / src_width);
    float fMultV = (float)(target_height / src_height);
    if (fMultH == fMultV)
    {
        if ((fMultH - (long)fMultH) == 0 && (fMultV - (long)fMultV == 0)) // looks like both integer
        {
            if (ap == 1) bAddProc = true; // AddProc enabled now with ap arg = 1
            iMul = (int64_t)fMultH;
        }
    }

    if (bAddProc)
    {
        // init data arrays
//        iMul = static_cast<int64_t>(iMultH);
        iTaps = static_cast<int64_t>(tap);
        iKernelSize = (iMul * iTaps * 2);
        iParProc = 4; // number of parallel processed samples in 1 SIMD pass
        iKernelStridePP = iKernelSize + 8; // may be +16 for AVX512
        if ((iKernelStridePP % 32) != 0)
        {
            iKernelStridePP += 32 - (iKernelStridePP % 32);// num of Kernel line bytes must be mul of 32 bytes (64 for avx512 ?)
        }
        
        // start rows of kernel buf g_pfKernelParProc must be 32-bytes aligned 

        g_pfKernel = (float*)_mm_malloc(iKernelSize * iKernelSize * sizeof(float), 32);
        memset(g_pfKernel, 0, iKernelSize * iKernelSize * sizeof(float)); // pad kernel rows with zeroes for load to SIMD with shifts

        g_pfKernelParProc = (float*)_mm_malloc(iKernelStridePP * iKernelSize * iParProc * sizeof(float), 32);
        memset(g_pfKernelParProc, 0, iKernelStridePP * iKernelSize * iParProc * sizeof(float)); // pad kernel rows with zeroes for load to SIMD with shifts

        g_pfKernelWeighted = (float*)_mm_malloc(iKernelSize * iKernelSize * 256 * sizeof(float), 32);
        memset(g_pfKernelWeighted, 0, iKernelSize * iKernelSize * sizeof(float) * 256);

        pKrnRUR = new KrnRowUsefulRange[iKernelSize * sizeof(KrnRowUsefulRange)];
        memset(pKrnRUR, 0, iKernelSize * sizeof(KrnRowUsefulRange));

        fill2DKernel();

        iWidth = src_width;
        iHeight = src_height;

        iWidthEl = iWidth + 2 * iKernelSize; 
        iHeightEl = iHeight + 2 * iKernelSize;

        g_pElImageBuffer = (BYTE*)malloc(iWidthEl * iHeightEl);

        SzFilteredImageBuffer = iWidthEl * iHeightEl * iMul * iMul * sizeof(float); // assume largest size input plane
        g_pfFilteredImageBuffer = (float*)malloc(SzFilteredImageBuffer);

        // buffer to hold temp converted to float32 input line for each processing thread
        float* pfInpFloatRow = (float*)_mm_malloc(iWidthEl * iMul * threads_ * sizeof(float), 32);

 //       pfEndOfFilteredImageBuffer = g_pfFilteredImageBuffer + (iWidthEl * iHeightEl * iMul * iMul);
        
    }
    else
    {
        for (int i = 0; i < planecount; ++i)
        {
            out[i] = new EWAPixelCoeff();

            if (!vi.IsRGB() && !vi.Is444())
            {
                if (i == 0)
                    generate_coeff_table_c(init_lut, out[0], quant_x, quant_y, samples, src_width, src_height,
                        target_width, target_height, radius, crop_left, crop_top, crop_width, crop_height);
                else
                    generate_coeff_table_c(init_lut, out[i], quant_x, quant_y, samples, src_width >> sub_w, src_height >> sub_h,
                        target_width >> sub_w, target_height >> sub_h, radius, crop_left / div_w, crop_top / div_h, crop_width / div_w, crop_height / div_h);
            }
            else
                generate_coeff_table_c(init_lut, out[i], quant_x, quant_y, samples, src_width, src_height,
                    target_width, target_height, radius, crop_left, crop_top, crop_width, crop_height);
        }
    }

    const bool avx512 = (opt == 3);
    const bool avx2 = (!!(env->GetCPUFlags() & CPUF_AVX2) && opt < 0) || opt == 2;
    const bool sse41 = (!!(env->GetCPUFlags() & CPUF_SSE4_1) && opt < 0) || opt == 1;

    ConvertToInt = &JincResize::ConvertToInt_c; // default
    ConvertInpElRowToFloat = &JincResize::ConvertInpElRowToFloat_c; // default

    if (avx512)
    {
        if (bAddProc)
            KernelRow = &JincResize::KernelRow_avx512;

        switch (vi.ComponentSize())
        {
            case 1: process_frame = &JincResize::resize_plane_avx512<uint8_t>; break;
            case 2: process_frame = &JincResize::resize_plane_avx512<uint16_t>; break;
            default: process_frame = &JincResize::resize_plane_avx512<float>; break;
        }
    }
    else if (avx2)
    {
        if (bAddProc)
        {
          if (iMul == 4 && iTaps == 4)  KernelRow = &JincResize::KernelRow_avx2_mul4_taps4; // may be do AVS function JincUpsize64_x4(), still only avx2+fma for now
//          else if (iMul == 8 && iTaps == 3)  KernelRow = &JincResize::KernelRow_avx2_mul8_taps3; // looks still slower in compare with old, may be avx512 will be faster
//          else if (iMul == 2 && iTaps == 8)  KernelRow = &JincResize::KernelRow_avx2_mul2_taps8; - still unfinished
          else  KernelRow = &JincResize::KernelRow_avx2_mul;

          ConvertToInt = &JincResize::ConvertToInt_avx2;
//	  ConvertInpElRowToFloat = &JincResize::ConvertInpElRowToFloat_avx2; // TO DO and debug later

        }

        switch (vi.ComponentSize())
        {
            case 1: process_frame = &JincResize::resize_plane_avx2<uint8_t>; break;
            case 2: process_frame = &JincResize::resize_plane_avx2<uint16_t>; break;
            default: process_frame = &JincResize::resize_plane_avx2<float>; break;
        }
    }
    else if (sse41)
    {
        if (bAddProc)
            KernelRow = &JincResize::KernelRow_sse41;

        switch (vi.ComponentSize())
        {
            case 1: process_frame = &JincResize::resize_plane_sse41<uint8_t>; break;
            case 2: process_frame = &JincResize::resize_plane_sse41<uint16_t>; break;
            default: process_frame = &JincResize::resize_plane_sse41<float>; break;
        }
    }
    else
    {
        if (bAddProc)
            KernelRow = &JincResize::KernelRow_c_mul;

        switch (vi.ComponentSize())
        {
            case 1: process_frame = &JincResize::resize_plane_c<uint8_t>; break;
            case 2: process_frame = &JincResize::resize_plane_c<uint16_t>; break;
            default: process_frame = &JincResize::resize_plane_c<float>; break;
        }
    }
}

constexpr float fPi = 3.14159265358979f;

void JincResize::fill2DKernel(void)
{
    
    int i, j;

    // our 2D just iMul-finely sampled kernel defined in output size

    /* Sinc (SincLin2) kernel
    for (i = 0; i < iKernelSize; i++)
    {
        for (j = 0; j < iKernelSize; j++)
        {
            float fDist = sqrtf((float(iKernelSize / 2) - j)*(float(iKernelSize / 2) - j) + (float(iKernelSize / 2) - i)*(float(iKernelSize / 2) - i));

            // make kernel round in 2d
            if (fDist > iKernelSize / 2) continue;

            float fArg = (fPi*fDist/iMul);

            if (fDist <= (iKernelSize / 4))
            {
                if (fArg != 0)
                {
                    g_pfKernel[i*iKernelSize + j] = sinf(fArg) / fArg;
                }
                else
                    g_pfKernel[i*iKernelSize + j] = 1.0f;

            }
            else if (fDist > (iKernelSize / 4) && fDist <= (iKernelSize/2))
            {
                g_pfKernel[i*iKernelSize + j] = (sinf(fArg) / fArg) * ((2 - (4 * fDist / iKernelSize))); // SincLin2Resize trapezoidal weighting
            }
            else
                g_pfKernel[i*iKernelSize + j] = 0.0f;
        } //j
    } //i
    */

    /* Jinc weighted by Jinc - EWA Lanczos kernel */
    for (i = 0; i < iKernelSize; i++)
    {
        for (j = 0; j < iKernelSize; j++)
        {
            float fDist = sqrtf((float(iKernelSize / 2) - j) * (float(iKernelSize / 2) - j) + (float(iKernelSize / 2) - i) * (float(iKernelSize / 2) - i));

            // make kernel round in 2d
            if (fDist > iKernelSize / 2) continue;

            float fArg = (fPi * fDist / iMul);

            float fArg_w = fDist * 3.9f / (iKernelSize / 2);

            if (fArg != 0)
            {
                float fBess = 2.0f * std::cyl_bessel_jf(1, fArg) / fArg;
                float fW; // Jinc window
                if (Weighting_type == JINC)
                {
                     if (fArg_w != 0)
                     {
                         fW = 2.0f * std::cyl_bessel_jf(1, fArg_w) / fArg_w;
                     }
                     else
                     {
                         fW = 1.0f;
                     }
                }
                if (Weighting_type == TRAPEZOIDAL)
                {
                    fW = 1.0f;
                    if (fDist > (iKernelSize / 4))
                    {
                        fW = ((2 - (4 * fDist / iKernelSize))); // trapezoidal weighting
                    }
                }
                g_pfKernel[i * iKernelSize + j] = fBess * fW;
            }
            else
                g_pfKernel[i * iKernelSize + j] = 1.0f;


        } //j
    } //i


    // normalize to 1
    float fSum = 0.0f;
    for (i = 0; i < iKernelSize; i++)
    {
        for (j = 0; j < iKernelSize; j++)
        {
            fSum += g_pfKernel[i * iKernelSize + j];
        }
    }

    for (i = 0; i < iKernelSize; i++)
    {
        for (j = 0; j < iKernelSize; j++)
        {
            g_pfKernel[i * iKernelSize + j] /= fSum;
            g_pfKernel[i * iKernelSize + j] *= (iMul * iMul); // energy dissipated at iMul^2 output samples, so 1 norm * iMul^2
        }
    }

    // fill 256 kernel images weighted by 8bit unsigned int
    for (int64_t iSample = 1; iSample < 256; iSample++) // skip 0 - memory was memset to 0.
    {
        float* pfCurrKernel = g_pfKernelWeighted + iSample * iKernelSize * iKernelSize;
        for (i = 0; i < iKernelSize; i++)
        {
            for (j = 0; j < iKernelSize; j++)
            {
                pfCurrKernel[i * iKernelSize + j] = g_pfKernel[i * iKernelSize + j] * iSample;
            }
        }
    }

    // TO DO : 1. Make kernel half height. 2. Add run-length encoding for non-zero line length.

    // fill kernel row useful range table
    for (i = 0; i < iKernelSize; i++)
    {
        bool bStarted = false;
        for (j = 0; j < iKernelSize; j++)
        {
            float fDist2 = sqrtf((float(iKernelSize / 2) - j) * (float(iKernelSize / 2) - j) + (float(iKernelSize / 2) - i) * (float(iKernelSize / 2) - i));
            if ((fDist2 <= iKernelSize / 2) && !bStarted)
            {
                pKrnRUR[i].start_col = j;
                bStarted = true;
            }
            if ((fDist2 > iKernelSize / 2) && bStarted)
            {
                pKrnRUR[i].end_col = j - 1;
                bStarted = false;
            }
        }
        if (bStarted) pKrnRUR[i].end_col = iKernelSize;
    }

}

void JincResize::KernelProc(unsigned char* src, int iSrcStride, int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{
    int64_t row, col;

    // current input plane sizes
    iWidthEl = iInpWidth + 2 * iKernelSize;
    iHeightEl = iInpHeight + 2 * iKernelSize;

    memset(g_pElImageBuffer, 0, iWidthEl * iHeightEl);

    memset(g_pfFilteredImageBuffer, 0, (iWidthEl * iHeightEl * iMul * iMul * sizeof(float)));

    // fill center
    for (row = 0; row < iInpHeight; row++)
    {
        for (col = 0; col < iInpWidth; col++)
        {
            g_pElImageBuffer[(row + iKernelSize) * iWidthEl + (col + iKernelSize)] = src[(row * iSrcStride + col)];
        } 
//        memcpy(g_pElImageBuffer + (row + iKernelSize) * iWidthEl + iKernelSize, src + (row * iSrcStride), iInpWidth * sizeof(float)); - need debug

    }

    // fill upper strip - dup 1st line
    for (row = 0; row < iKernelSize; row++)
    {
        for (col = 0; col < iInpWidth; col++)
        {
            g_pElImageBuffer[row * iWidthEl + (col + iKernelSize)] = src[col]; 
        }
    }

    // fill left strip - dup first column, 
    // fill right strip - dup last column
    // combine left and right in 1 cycle for 1 memory pass.
    for (row = 0; row < iInpHeight; row++)
    {
        for (col = 0; col < iKernelSize; col++) // left
        {
            g_pElImageBuffer[(row + iKernelSize) * iWidthEl + col] = src[(row * iSrcStride) + 0];
        }

        for (col = iInpWidth; col < (iInpWidth + iKernelSize); col++) // right
        {
            g_pElImageBuffer[(row + iKernelSize) * iWidthEl + col + iKernelSize] = src[(row * iSrcStride) + (iInpWidth - 1)];
        }
    }
    
    // fill lower strip - dup last line
    for (row = iInpHeight + iKernelSize; row < iHeightEl; row++)
    {
        for (col = 0; col < iInpWidth; col++)
        {
            int64_t Ptr = static_cast<int64_t>((row) * iWidthEl + (col + iKernelSize));
            g_pElImageBuffer[Ptr] = src[((iInpHeight - 1) * iSrcStride + col)]; 
        }
    }
 
 
    // 2d convolution pass
    int64_t iOutWidth = iWidthEl * iMul;

    (this->*KernelRow)(iOutWidth);

    (this->*ConvertToInt)(iInpWidth, iInpHeight, dst, iDstStride);

}

void JincResize::ConvertToInt_c(int iInpWidth, int iInpHeight, unsigned char* dst, int iDstStride)
{
#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = 0; row < iInpHeight * iMul; row++)
    {
        for (int64_t col = 0; col < iInpWidth * iMul; col++)
        {
            unsigned char ucVal;
            float fVal = g_pfFilteredImageBuffer[(row + iKernelSize * iMul) * iWidthEl * iMul + col + iKernelSize * iMul];

            fVal += 0.5f;

            if (fVal > 255.0f)
            {
                fVal = 255.0f;
            }
            if (fVal < 0.0f)
            {
                fVal = 0.0f;
            }
            ucVal = (unsigned char)fVal;
            dst[(row * iDstStride + col)] = ucVal;
        }
    }
}

void JincResize::ConvertInpElRowToFloat_c(int64_t iWidth, unsigned char *src, float *dst)
{
    for (int64_t col=0; col < iWidth; col++)
	{
	  dst[col] = (float)src[col];
	}
}

void JincResize::KernelRow_c(int64_t iOutWidth)
{
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

            // fast skip zero
            if (ucInpSample == 0) continue;

            float* pfCurrKernel = g_pfKernelWeighted + static_cast<int64_t>(ucInpSample) * iKernelSize * iKernelSize;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                // add full kernel row to output - C
                for (int64_t k_col = pKrnRUR[k_row].start_col; k_col < pKrnRUR[k_row].end_col; k_col++)
                {
                    pfProc[k_col] += pfCurrKernel[k_col];
                } // k_col
                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel += iKernelSize; // point to next kernel row now
            } // k_row
        } // col
    }
}

void JincResize::KernelRow_c_mul(int64_t iOutWidth)
{
    float* pfCurrKernel = g_pfKernel; 

#pragma omp parallel for num_threads(threads_) 
    for (int64_t row = iTaps; row < iHeightEl - iTaps; row++) // input lines counter
    {
        // start all row-only dependent ptrs here
        int64_t iProcPtrRowStart = (row * iMul - (iKernelSize / 2)) * iOutWidth - (iKernelSize / 2);
        int64_t iInpPtrRowStart = row * iWidthEl;

        for (int64_t col = iTaps; col < iWidthEl - iTaps; col++) // input cols counter
        {
            unsigned char ucInpSample = g_pElImageBuffer[(iInpPtrRowStart + col)];

            float* pfCurrKernel_pos = pfCurrKernel;

            float* pfProc = g_pfFilteredImageBuffer + iProcPtrRowStart + col * iMul;

            for (int64_t k_row = 0; k_row < iKernelSize; k_row++)
            {
                for (int64_t k_col = pKrnRUR[k_row].start_col; k_col < pKrnRUR[k_row].end_col; k_col++)
                {
                    pfProc[k_col] += pfCurrKernel_pos[k_col] * ucInpSample;
                } // k_col 
                pfProc += iOutWidth; // point to next start point in output buffer now
                pfCurrKernel_pos += iKernelSize; // point to next kernel row now
            } // k_row
        } // col
    }
}

JincResize::~JincResize()
{
    delete[] init_lut->lut;
    delete init_lut;

    for (int i = 0; i < planecount; ++i)
    {
        delete_coeff_table(out[i]);
        delete out[i];
    }

    if (bAddProc)
    {
        _mm_free(g_pfKernel);
        _mm_free(g_pfKernelWeighted);
        _mm_free(g_pfKernelParProc);
        delete pKrnRUR;
        free(g_pElImageBuffer);
        free(g_pfFilteredImageBuffer);
	_mm_free(pfInpFloatRow);
    }
}

PVideoFrame JincResize::GetFrame(int n, IScriptEnvironment* env)
{
    PVideoFrame src = child->GetFrame(n, env);
    PVideoFrame dst = (has_at_least_v8) ? env->NewVideoFrameP(vi, &src) : env->NewVideoFrame(vi);

    (this->*process_frame)(out, src, dst, env);

    return dst;
}

AVSValue __cdecl Create_JincResize(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    const VideoInfo& vi = args[0].AsClip()->GetVideoInfo();

    return new JincResize(
        args[0].AsClip(),
        args[1].AsInt(),
        args[2].AsInt(),
        args[3].AsFloat(0),
        args[4].AsFloat(0),
        args[5].AsFloat(static_cast<float>(vi.width)),
        args[6].AsFloat(static_cast<float>(vi.height)),
        args[7].AsInt(256),
        args[8].AsInt(256),
        args[9].AsInt(3),
        args[10].AsFloat(0),
        args[11].AsInt(0),
        args[12].AsInt(-1),
        args[13].AsInt(0),
        args[14].AsInt(0),
        env);
}

static void resizer(const AVSValue& args, Arguments* out_args, int src_left_idx = 3)
{
    out_args->add(args[0]);
    out_args->add(args[1]);
    out_args->add(args[2]);

    if (args[src_left_idx + 0].Defined())
        out_args->add(args[src_left_idx + 0], "src_left");
    if (args[src_left_idx + 1].Defined())
        out_args->add(args[src_left_idx + 1], "src_top");
    if (args[src_left_idx + 2].Defined())
        out_args->add(args[src_left_idx + 2], "src_width");
    if (args[src_left_idx + 3].Defined())
        out_args->add(args[src_left_idx + 3], "src_height");
    if (args[src_left_idx + 4].Defined())
        out_args->add(args[src_left_idx + 4], "quant_x");
    if (args[src_left_idx + 5].Defined())
        out_args->add(args[src_left_idx + 5], "quant_y");
    if (args[src_left_idx + 6].Defined())
        out_args->add(args[src_left_idx + 6], "threads");
    if (args[src_left_idx + 7].Defined())
        out_args->add(args[src_left_idx + 7], "wt");
    if (args[src_left_idx + 8].Defined())
        out_args->add(args[src_left_idx + 8], "ap");
}

template <int taps>
AVSValue __cdecl resizer_jinc36resize(AVSValue args, void* user_data, IScriptEnvironment* env)
{
    Arguments mapped_args;

    resizer(args, &mapped_args);
    mapped_args.add(args[10].AsInt(taps), "tap");

    return env->Invoke("JincResize", mapped_args.args(), mapped_args.arg_names()).AsClip();
}

const AVS_Linkage* AVS_linkage;

extern "C" __declspec(dllexport)
const char* __stdcall AvisynthPluginInit3(IScriptEnvironment * env, const AVS_Linkage* const vectors)
{
    AVS_linkage = vectors;

//    env->AddFunction("JincResize", "cii[src_left]f[src_top]f[src_width]f[src_height]f[quant_x]i[quant_y]i[tap]i[blur]f[threads]i[opt]i", Create_JincResize, 0);
    env->AddFunction("JincResize", "cii[src_left]f[src_top]f[src_width]f[src_height]f[quant_x]i[quant_y]i[tap]i[blur]f[threads]i[opt]i[wt]i[ap]i", Create_JincResize, 0);

    env->AddFunction("Jinc36Resize", "cii[src_left]f[src_top]f[src_width]f[src_height]f[quant_x]i[quant_y]i[threads]i", resizer_jinc36resize<3>, 0);
    env->AddFunction("Jinc64Resize", "cii[src_left]f[src_top]f[src_width]f[src_height]f[quant_x]i[quant_y]i[threads]i", resizer_jinc36resize<4>, 0);
    env->AddFunction("Jinc144Resize", "cii[src_left]f[src_top]f[src_width]f[src_height]f[quant_x]i[quant_y]i[threads]i", resizer_jinc36resize<6>, 0);
    env->AddFunction("Jinc256Resize", "cii[src_left]f[src_top]f[src_width]f[src_height]f[quant_x]i[quant_y]i[threads]i", resizer_jinc36resize<8>, 0);

    return "JincResize";
}

