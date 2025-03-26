## Description

Jinc (EWA Lanczos) resampling plugin for AviSynth 2.6 / AviSynth+.

This is [a port of the VapourSynth plugin JincResize](https://github.com/Kiyamou/VapourSynth-JincResize).

SSE / AVX Intrinsics taken from [the other AviSynth plugin JincResize](https://github.com/AviSynth/jinc-resize).

NOTE: The 32-bit version is not supported. If you still want to use it keep in mind that the OS memory limit can be easily hit. (#10)

### Requirements:

- AviSynth+ r3688 or later ([1](https://github.com/AviSynth/AviSynthPlus/releases) / [2](https://forum.doom9.org/showthread.php?t=181351) / [3](https://gitlab.com/uvz/AviSynthPlus-Builds))

- Microsoft VisualC++ Redistributable Package 2022 (can be downloaded from [here](https://github.com/abbodi1406/vcredist/releases))

### Usage:

```
JincResize (clip, int target_width, int target_height, float "src_left", float "src_top", float "src_width", float "src_height", int "quant_x", int "quant_y", int "tap", float "blur", string "cplace", int "threads", int "opt", int "initial_capacity", float "initial_factor")
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

- clip<br>
    A clip to process. All planar formats are supported.

- target_width<br>
    The width of the output.

- target_height<br>
    The height of the output.

- src_left<br>
    Cropping of the left edge.<br>
    Default: 0.0.

- src_top<br>
    Cropping of the top edge.<br>
    Default: 0.0.

- src_width<br>
    If > 0.0 it sets the width of the clip before resizing.<br>
    If <= 0.0 it sets the cropping of the right edges before resizing.<br>
    Default: Source width.

- src_height<br>
    If > 0.0 it sets the height of the clip before resizing.<br>
    If <= 0.0 it sets the cropping of the bottom edges before resizing.<br>
    Default: Source height.

- quant_x, quant_y<br>
    Controls the sub-pixel quantization.<br>
    Must be between 1 and 256.<br>
    Default: 256.

- tap (JincResize only)<br>
    Corresponding to different zero points of Jinc function.<br>
    Must be between 1 and 16.<br>
    Default: 3.

- blur (JincResize only)<br>
    Blur processing, it can reduce side effects.<br>
    To achieve blur, the value should be less than 1.0.<br>
    Default: 1.0.

- threads<br>
    Whether to use maximum logical processors.<br>
    0: Maximum logical processors are used.<br>
    1: Only one thread is used.<br>
    Default: 0.

- cplace<br>
    The location of the chroma samples.<br>
    "MPEG1": Chroma samples are located on the center of each group of 4 pixels.<br>
    "MPEG2": Chroma samples are located on the left pixel column of the group.<br>
    "topleft": Chroma samples are located on the left pixel column and the first row of the group.<br>
    Default: If frame properties are supported and frame property "_ChromaLocation" exists - "_ChromaLocation" value of the first frame is used.
    If frame properties aren't supported or there is no property "_ChromaLocation" - "MPEG2".

- opt (JincResize only)<br>
    Sets which cpu optimizations to use.<br>
    -1: Auto-detect without AVX-512.<br>
    0: Use C++ code.<br>
    1: Use SSE4.1 code.<br>
    2: Use AVX2 code.<br>
    3: Use AVX-512 code.<br>
    Default: -1.

- initial_capacity (JincResize only)<br>
    Initial memory allocation size.<br>
    Lower size forces more further memory reallocating that leads to initial slower startup but avoids excessive memory allocation.<br>
    Must be greater than 0.<br>
    Default: Max(target_width * target_height, src_width * src_height).

- initial_factor (JincResize only)<br>
    The initial factor used for the first memory reallocation.<br>
    After the first memory reallocation the factor starts to lower for the next reallocations.<br>
    `initial_factor=1` ensures that the next memory allocation is the minimal possible.<br>
    Must be equal to or greater than 1.0.<br>
    Default: 1.5.

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

Example of building on Windows with MSVC:

1. Open x64 Native Tools Command Prompt for VS xxxx.
2. Type - `set LIB=%LIB%;path_to_avisynth.lib`
3. Navigate to the jincresize source folder.
4. Type - `cmake -B name_of_the_folder_containing_building_files -DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES=path_to_avisynth_c.h`

By default Visual Studio solution files will be created.

Example of building on Windows with Mingw:

1. Open UCRT64/MINGW64 shell.
2. Type - `export LIBRARY_PATH=$LIBRARY_PATH:path_to_the_avs_lib`
3. Navigate to the jincresize source folder.
4. Type - `cmake -B name_of_the_folder_containing_building_files -DCMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES=path_to_avisynth_c.h`
