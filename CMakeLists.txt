cmake_minimum_required(VERSION 3.25)
project(cuda_tutorial VERSION 0.0.1 LANGUAGES CXX CUDA)

option(DISABLE_CUFFT "Do not build cufft-dependent executables" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Debug")
endif()

# TODO: CMAKE_*_FLAGS, CMAKE_*_FLAGS_DEBUG

string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr --extended-lambda")

string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -lineinfo")

set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES "60;61;62;70;80;90")

if(NOT DISABLE_CUFFT)
    find_package(CUDAToolkit REQUIRED)
    link_libraries(CUDA::cufft)
endif()

function(build_all_sources directory)
    if(NOT IS_DIRECTORY ${CMAKE_SOURCE_DIR}/${directory})
        message(FATAL_ERROR "Directory not found")
    endif()
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/${directory})

    file(GLOB sources ${CMAKE_SOURCE_DIR}/${directory}/*.cu)
    foreach(source IN LISTS sources)
        block()
            cmake_path(REMOVE_EXTENSION source OUTPUT_VARIABLE source_no_ext)
            cmake_path(GET source_no_ext FILENAME file_name)
            set(target_name "${directory}_${file_name}")
            add_executable(${target_name} ${source})
            set_property(
                TARGET ${target_name}
                PROPERTY OUTPUT_NAME ${file_name}
            )
            target_include_directories(
                ${target_name}
                PRIVATE ${CMAKE_SOURCE_DIR}/hpp
            )
        endblock()
    endforeach()
endfunction()

build_all_sources("cu")
if(NOT DISABLE_CUFFT)
    build_all_sources("cufft")
endif()
