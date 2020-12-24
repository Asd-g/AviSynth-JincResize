##### 1.2.0:
    Used MSVC instead Intel C++ for faster binaries.

##### 2.0.0:
    Added OpenMP support to main processing loops. (DTL2020)
    Added parameter 'threads'.

##### 1.2.0:
    AVX-512 code is not used as default when AVX-512 CPU instructions are available.
    Fixed AVX-512 output.
    Prevent 'nan' values for the float input (SIMD).
    Fixed JincXXXResize parameters 'quant_x' and 'quant_y' when called by name.

##### 1.1.0:
    Added AVX-512 code.

##### 1.0.1:
    Fixed 8..16-bit processing when C++ routine is used.
    Changed blur parameter.
    Registered as MT_MULTI_INSTANCE.

##### 1.0.0:
    Port of the VapourSynth plugin JincResize r7.1.
    