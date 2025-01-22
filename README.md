## Description

Jinc (EWA Lanczos) resampling plugin for AviSynth 2.6 / AviSynth+.

This is [a port of the VapourSynth plugin JincResize](https://github.com/Kiyamou/VapourSynth-JincResize).

SSE / AVX Intrinsics taken from [the other AviSynth plugin JincResize](https://github.com/AviSynth/jinc-resize).

NOTE: The 32-bit version is not supported. If you still want to use it keep in mind that the OS memory limit can be easily hit. (#10)

### Requirements:

- AviSynth 2.60 / AviSynth+ 3.4 or later

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Usage:

```
JincResize (clip, int target_width, int target_height, float "src_left", float "src_top", float "src_width", float "src_height", int "quant_x", int "quant_y", int "tap", float "blur", string "cplace", int "threads", int "opt")
```

##### There are 4 additional functions:
    Jinc36Resize is an alias for JincResize(tap=3).
    Jinc64Resize is an alias for JincResize(tap=4).
    Jinc144Resize is an alias for JincResize(tap=6).
    Jinc256Resize is an alias for JincResize(tap=8).

```
Jinc36Resize / Jinc64Resize / Jinc144Resize / Jinc256Resize (clip, int target_width, int target_height, float "src_left", float "src_top", float "src_width", float "src_height", int "quant_x", int "quant_y", string "cplace", int "threads")
```

### Parameters:

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
    To achieve blur, the value should be less than 1.0.\
    Default: 1.0.

- threads\
    Whether to use maximum logical processors.\
    0: Maximum logical processors are used.\
    1: Only one thread is used.\
    Default: 0.

- cplace\
    The location of the chroma samples.\
    "MPEG1": Chroma samples are located on the center of each group of 4 pixels.\
    "MPEG2": Chroma samples are located on the left pixel column of the group.\
    "topleft": Chroma samples are located on the left pixel column and the first row of the group.\
    Default: If frame properties are supported and frame property "_ChromaLocation" exists - "_ChromaLocation" value of the first frame is used.
    If frame properties aren't supported or there is no property "_ChromaLocation" - "MPEG2".

- opt (JincResize only)\
    Sets which cpu optimizations to use.\
    -1: Auto-detect without AVX-512.\
    0: Use C++ code.\
    1: Use SSE4.1 code.\
    2: Use AVX2 code.\
    3: Use AVX-512 code.\
    Default: -1.

### Building:

```
Requirements:
- Git
- C++17 compiler
- CMake >= 3.16
- Ninja
```

```
git clone https://github.com/Asd-g/AviSynth-JincResize && \
cd AviSynth-JincResize
cmake -B build -G Ninja
ninja -C build
```

Example of building on Windows:

1. Open x64 Native Tools Command Prompt for VS xxxx.
2. Type - `set LIB=%LIB%;path_to_avisynth.lib`
3. Navigate to the jincresize source folder.
4. Type - `cmake -B name_of_the_folder_containing_building_files -DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES=path_to_avisynth_c.h`

By default Visual Studio solution files will be created.
