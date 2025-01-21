cmake_minimum_required(VERSION 3.16)

if (NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    if (NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
    endif()
endif()

project(JincResize LANGUAGES CXX)

add_library(JincResize SHARED)

target_sources(JincResize PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src/JincResize.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/resize_plane_sse41.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/resize_plane_avx2.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/resize_plane_avx512.cpp
)

if (WIN32)
    target_sources(JincResize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/JincResize.rc)
endif()

target_include_directories(JincResize PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

if (UNIX)
    target_include_directories(JincResize PRIVATE
        /usr/local/include/avisynth
        /usr/local/include
    )
endif()

if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_link_libraries(JincResize PRIVATE libmmds)
endif()

if (NOT CMAKE_GENERATOR MATCHES "Visual Studio")
    string(TOLOWER ${CMAKE_BUILD_TYPE} build_type)
    if (build_type STREQUAL Debug)
        target_compile_definitions(JincResize PRIVATE DEBUG_BUILD)
    else (build_type STREQUAL Release)
        target_compile_definitions(JincResize PRIVATE RELEASE_BUILD)
    endif()

    if (NOT MSVC)
        target_compile_options(JincResize PRIVATE $<$<CONFIG:Release>:-s>)
    endif()

    message(STATUS "Build type - ${CMAKE_BUILD_TYPE}")
endif()

if (MSVC)
    set_source_files_properties(src/resize_plane_avx2.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX2")
    set_source_files_properties(src/resize_plane_avx512.cpp PROPERTIES COMPILE_OPTIONS "/arch:AVX512")
else()
    set_source_files_properties(src/resize_plane_sse41.cpp PROPERTIES COMPILE_OPTIONS "-mfpmath=sse;-msse4.1")
    set_source_files_properties(src/resize_plane_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(src/resize_plane_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512dq;-mavx512vl;-mfma")
endif()

target_link_libraries(JincResize PRIVATE avisynth)

target_compile_features(JincResize PRIVATE cxx_std_17)

if (CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(JincResize PRIVATE "/fp:precise")
endif()

if (UNIX)
    find_package (Git)

    if (GIT_FOUND)
        execute_process (COMMAND ${GIT_EXECUTABLE} describe --tags --abbrev=0
            OUTPUT_VARIABLE ver
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set_target_properties(JincResize PROPERTIES OUTPUT_NAME "jincresize.${ver}")
    else ()
        message (STATUS "GIT not found")
    endif ()

    include(GNUInstallDirs)

    INSTALL(TARGETS JincResize LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}/avisynth")

    # uninstall target
    if(NOT TARGET uninstall)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
        "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
        IMMEDIATE @ONLY)

    add_custom_target(uninstall
        COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
    endif()
endif()
