# Description

Jinc (EWA Lanczos) resampling plugin for AviSynth 2.6 / AviSynth+.

This is [a port of the VapourSynth plugin JincResize](https://github.com/Kiyamou/VapourSynth-JincResize).

SSE / AVX Intrinsics taken from [the other AviSynth plugin JincResize](https://github.com/AviSynth/jinc-resize).

# Usage

```
JincResize (clip, int target_width, int target_height, float "src_left", float "src_top", float "src_width", float "src_height", int "quant_x", int "quant_y", int "tap", float "blur", int "threads", int "opt", int "wt", int "ap")
```

##### There are 4 additional functions:
    Jinc36Resize is an alias for JincResize(tap=3).
    Jinc64Resize is an alias for JincResize(tap=4).
    Jinc144Resize is an alias for JincResize(tap=6).
    Jinc256Resize is an alias for JincResize(tap=8).
    
```
Jinc36Resize / Jinc64Resize / Jinc144Resize / Jinc256Resize (clip, int target_width, int target_height, float "src_left", float "src_top", float "src_width", float "src_height", int "quant_x", int "quant_y", int "threads")
```

## Parameters:

- clip\
    A clip to process. All planar formats are supported.
    
- target_width\
    The width of the output.
    
- target_height\
    The height of the output.
    
- src_left\
    Cropping of the left edge.\
    Default: 0.0.
    
- src_top\
    Cropping of the top edge.\
    Default: 0.0.
    
- src_width\
    If > 0.0 it sets the width of the clip before resizing.\
    If <= 0.0 it sets the cropping of the right edges before resizing.\
    Default: Source width.

- src_height\
    If > 0.0 it sets the height of the clip before resizing.\
    If <= 0.0 it sets the cropping of the bottom edges before resizing.\
    Default: Source height.
    
- quant_x, quant_y\
    Controls the sub-pixel quantization.\
    Must be between 1 and 256.\
    Default: 256.
    
- tap (JincResize only)\
    Corresponding to different zero points of Jinc function.\
    Must be between 1 and 16.\
    Default: 3.
    
- blur (JincResize only)\
    Blur processing, it can reduce side effects.\
    To achieve blur, the value should more than 0.0.\
    Must be between 0.0 and 10.0.
    Default: 0.0.
    
- threads\
    How many logical processors are used.\
    0: Maximum logical processors are used.\
    Default: 0.
    
- opt (JincResize only)\
    Sets which cpu optimizations to use.\
    -1: Auto-detect without AVX-512.\
    0: Use C++ code.\
    1: Use SSE4.1 code.\
    2: Use AVX2 code.\
    3: Use AVX-512 code.\
    4: Use SSE2 code. Only works for iMul 4 or 8.\
    Default: -1.

- wt (JincResize only)\
    Sets which weighting function to use.\
    0: Use Jinc weighting. I.e. Jinc weighted by Jinc = EWA Lanczos.\
    1: Use trapezoidal weighting. It allows for greater amplitude Jinc withing provided taps number \
    and makes result a bit sharper and closer to 'unweighted' Jinc. It provides at least taps/2 full-strike \
    Jinc function and linear decay to zero after taps/2. If input is not good conditioned for this kernel  \ 
    it also makes ringing and other distortions greater.
    Default: 0.
    
 - ap (JincResize only)\
    Sets which procesing optimizations to use.\
    0: Old 'large-kernel'. Can upsize and downsize with any ratio.\
    1: Alternative 1. Not for use - will be deleted soon. \
    2: Alternative 2. Uses even less memory. Can use multithreading with 'threads' parameter. Runs 1 thread or > 1 thread with different a bit internal resamplers. \
       Only upsize with integer ratio equal for width and height.
    Default: 0.
