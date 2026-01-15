cmake_minimum_required(VERSION 3.25)
project(
  cuda_tutorial
  VERSION 0.0.1
  LANGUAGES CXX CUDA)

option(DISABLE_CUFFT "Do not build cufft-dependent executables" OFF)

if(CMAKE_BUILD_TYPE STREQUAL "")
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES native)

string(
  APPEND
  CMAKE_CXX_FLAGS
  " -Wall -Wextra -Wpedantic -Wconversion -Wsign-conversion -Wcast-qual"
  " -Wformat=2 -Wundef -Wshadow -Wcast-align -Wunused -Wnull-dereference"
  " -Wdouble-promotion -Wimplicit-fallthrough -Wextra-semi -Woverloaded-virtual"
  " -Wnon-virtual-dtor -Wold-style-cast")
string(
  APPEND
  CMAKE_CXX_FLAGS_DEBUG
  # TODO: Understand why the address sanitizer does not behave well in the cnaf
  # machine -fsanitize=address,undefined
  " -O0 -fno-omit-frame-pointer -g -pg")
string(APPEND CMAKE_EXE_LINKER_FLAGS_DEBUG
       # TODO: Same as CMAKE_CXX_FLAGS -fsanitize=address,undefined
       " -fno-omit-frame-pointer")
string(APPEND CMAKE_CUDA_FLAGS " --expt-relaxed-constexpr --extended-lambda")
string(APPEND CMAKE_CUDA_FLAGS_DEBUG " -O0 -lineinfo -g -G")

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
    set(target_name ${directory}_${file_name})
    add_executable(${target_name} ${source})
    set_target_properties(${target_name} PROPERTIES OUTPUT_NAME ${file_name})
    target_include_directories(${target_name} PRIVATE ${CMAKE_SOURCE_DIR}/hpp)
    endblock()
  endforeach()
endfunction()

build_all_sources("cu")
if(NOT DISABLE_CUFFT)
  build_all_sources("cufft")
endif()
