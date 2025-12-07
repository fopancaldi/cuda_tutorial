#pragma once

#include "concepts.hpp"
#include "work_division.hpp"

#include <cassert>
#include <cuda.h>

namespace cuda_tutorial {

namespace internal {

template <concepts::trivially_copyable T>
__global__ void fill_kernel(T* output, T value, std::size_t count) {
    assert(threadIdx.y == 0 and threadIdx.z == 0);
    assert(blockIdx.y == 0 and blockIdx.z == 0);
    unsigned int const global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_idx < count) {
        output[global_idx] = value;
    }
}

} // namespace internal

template <concepts::trivially_copyable T>
void fill(T* dev_ptr, T const& value, std::size_t count) {
    if (count == 0) {
        return;
    } else {
        work_division work_div = make_work_div(count);
        internal::fill_kernel<<<work_div.blocks, work_div.block_threads>>>(dev_ptr, value, count);
    }
}

} // namespace cuda_tutorial
