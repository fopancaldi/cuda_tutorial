Code I have used to display some basic functionalities of [CUDA](https://developer.nvidia.com/cuda-toolkit) while learning it.

# Getting started

## Prerequisites

The code should be run on a CUDA-capable machine, i.e. one where the CUDA toolkit was installed.

There are no dependencies.
[cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) can optionally be installed to build some related examples.
It should automatically be installed with the CUDA toolkit.

## Installation

Clone the repo.

# Usage

    cmake -S . -B build
    cmake --build build

Then `build` will contain (inside the opportune subdirectories) one executable for each source file in `cu` and `cufft`.

Add `-D DISABLE_CUFFT=ON` to not build the sources in `cufft`.
