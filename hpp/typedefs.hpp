#pragma once

#include <cstddef>
#include <cuda.h>

namespace cuda_tutorial {

using size = std::size_t;
using ssize = std::ptrdiff_t;
using float_pt = double;

struct work_division {
    dim3 blocks;
    dim3 block_threads;
};

} // namespace cuda_tutorial
