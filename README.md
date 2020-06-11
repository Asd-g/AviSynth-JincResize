# Description

Jinc (EWA Lanczos) resampling plugin for AviSynth 2.6 / AviSynth+.

This is [a port of the VapourSynth plugin JincResize](https://github.com/Kiyamou/VapourSynth-JincResize).
SSE / AVX Intrinsics taken from [the other AviSynth plugin JincResize][https://github.com/AviSynth/jinc-resize].

# Usage

```
JincResize (clip, int target_width, int target_height, float "src_left", float "src_top", float "src_width", float "src_height", int "quant_x", int "quant_y", int "tap", float "blur", int "opt")
```

There are 4 additional functions:
    Jinc36Resize is an alias for JincResize(tap=3).
    Jinc64Resize is an alias for JincResize(tap=4).
    Jinc144Resize is an alias for JincResize(tap=6).
    Jinc256Resize is an alias for JincResize(tap=8).
    
```
Jinc36Resize / Jinc64Resize / Jinc144Resize / Jinc256Resize (clip, int target_width, int target_height, float "src_left", float "src_top", float "src_width", float "src_height", int "quant_x", int "quant_y")
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
    Must be between 1 and 256.
    Default: 256.
    
- tap (JincResize only)\
    Corresponding to different zero points of Jinc function.\
    Must be between 1 and 16.
    Default: 3.
    
- blur (JincResize only)\
    Blur processing, it can reduce side effects.\
    To achieve blur, the value should less than 1.\
    Default: 1.0.
    
- opt (JincResize only)\
    Controls the used CPU optimizations.\
    0: Use C++ routine.\
    1: Use SSE routine (SSE4.1 required).\
    2: Use AVX routine (AVX2 required).\
    Default: 2, if AVX not available 1, if SSE not available 0.  
