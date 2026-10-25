cmake_minimum_required(VERSION 3.16)
project(fast_yolov8_broadwell LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ON for every reportable run on ece007: enables AVX2/FMA (the fast kernels).
# OFF (default) builds scalar fallbacks so the repo also compiles on a Mac.
option(YOLO_ENABLE_BROADWELL "Build with Broadwell-specific x86 compiler options" OFF)

if(MSVC)
    add_compile_options(/O2)
else()
    add_compile_options(-O3 -Wall -Wextra -Wpedantic -Wno-unused-parameter)
endif()

if(YOLO_ENABLE_BROADWELL)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
        message(FATAL_ERROR "YOLO_ENABLE_BROADWELL requires an x86 CPU target")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        add_compile_options(-march=broadwell -mtune=broadwell)
    else()
        message(FATAL_ERROR "Broadwell flags are configured only for GCC/Clang")
    endif()
endif()

find_package(OpenMP)

add_library(yolov8_kernels
    src/layout.cpp
    src/reference_conv.cpp
    src/conv3x3_s1.cpp
    src/conv3x3_s2.cpp
    src/conv1x1.cpp
    src/sppf.cpp
)
target_include_directories(yolov8_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
if(OpenMP_CXX_FOUND)
    target_link_libraries(yolov8_kernels PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(test_kernels tests/test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE yolov8_kernels)

enable_testing()
add_test(NAME kernel_tests COMMAND test_kernels)

# Real-layer correctness + timing: ./build/bench_layer data/tensors/b1/<layer> s1_v1
add_executable(bench_layer benchmarks/bench_layer.cpp)
target_link_libraries(bench_layer PRIVATE yolov8_kernels)

# SPPF max-pool on the real model.9 tensors: ./build/bench_pool data/tensors/b1 v2
add_executable(bench_pool benchmarks/bench_pool.cpp)
target_link_libraries(bench_pool PRIVATE yolov8_kernels)

# Generated assembly of the stride-1 kernel: cmake --build build --target asm_s1
add_custom_target(asm_s1
    COMMAND ${CMAKE_CXX_COMPILER} -O3 -std=c++17 $<$<BOOL:${YOLO_ENABLE_BROADWELL}>:-march=broadwell>
            -I${CMAKE_CURRENT_SOURCE_DIR}/include -S -fverbose-asm
            ${CMAKE_CURRENT_SOURCE_DIR}/src/conv3x3_s1.cpp -o ${CMAKE_BINARY_DIR}/conv3x3_s1.s
    COMMENT "Writing build/conv3x3_s1.s")
