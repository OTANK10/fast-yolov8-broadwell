cmake_minimum_required(VERSION 3.16)
project(fast_yolov8_broadwell LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(YOLO_ENABLE_BROADWELL "Build with Broadwell-specific x86 compiler options" OFF)

add_library(yolov8_kernels
    src/reference_conv.cpp
    src/conv3x3_s1.cpp
    src/conv3x3_s2.cpp
    src/sppf.cpp
)
target_include_directories(yolov8_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

if(MSVC)
    target_compile_options(yolov8_kernels PRIVATE /O2)
else()
    target_compile_options(yolov8_kernels PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

if(YOLO_ENABLE_BROADWELL)
    if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64|i[3-6]86)$")
        message(FATAL_ERROR "YOLO_ENABLE_BROADWELL requires an x86 CPU target")
    endif()
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(yolov8_kernels PRIVATE -march=broadwell)
    else()
        message(FATAL_ERROR "Broadwell flags are configured only for GCC/Clang")
    endif()
endif()

add_executable(test_kernels tests/test_kernels.cpp)
target_link_libraries(test_kernels PRIVATE yolov8_kernels)

enable_testing()
add_test(NAME kernel_tests COMMAND test_kernels)

add_executable(bench_conv benchmarks/bench_conv.cpp)
target_link_libraries(bench_conv PRIVATE yolov8_kernels)
