#pragma once

#include <cuda.h>

namespace cuda_tutorial {

using float_pt = double;

struct work_division {
    dim3 blocks;
    dim3 block_threads;
};

} // namespace cuda_tutorial
