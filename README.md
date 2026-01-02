Code I have used to display some basic functionalities of [CUDA](https://developer.nvidia.com/cuda-toolkit) while learning it.

# Prerequisites

The code should be run on a CUDA-capable machine.

There are no dependencies.
[cuFFT](https://docs.nvidia.com/cuda/cufft/index.html) can optionally be installed to build some related examples.
It should automatically be installed with the CUDA toolkit.

# Usage

    cmake -S . -B build
    cmake --build build

Then `build` will contain (inside the opportune subdirectories) one executable for each source file in `cu` and `cufft`.

Add `-D DISABLE_CUFFT=ON` to omit building the sources in `cufft` and linking to the library.
