cmake_minimum_required(VERSION 3.25)
project(cuda_tutorial VERSION 0.0.1 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# TODO: CMAKE_*_FLAGS, CMAKE_*_FLAGS_DEBUG

string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -lineinfo")
add_compile_options("--expt-relaxed-constexpr")

set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES "50;60;61;62;70;80;90")

file(GLOB all_cu ${CMAKE_SOURCE_DIR}/cu/*.cu)
foreach(cu_file IN LISTS all_cu)
    block()
        cmake_path(REMOVE_EXTENSION cu_file OUTPUT_VARIABLE cu_file_noext)
        cmake_path(GET cu_file_noext FILENAME filename)
        add_executable(${filename} ${cu_file})
        target_include_directories(${filename} PRIVATE ${CMAKE_SOURCE_DIR}/hpp)
    endblock()
endforeach()
