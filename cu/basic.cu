#include "cuda_tutorial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <span>

template <std::floating_point FP>
__global__ void multiply_kernel(FP* ptr, FP factor, std::size_t arr_len,
                                unsigned int block_threads) {
    const unsigned int i = threadIdx.x + blockIdx.x * block_threads;
    if (i < arr_len) {
        ptr[i] *= factor;
    }
}

int main() {
    using namespace cuda_tutorial;
    namespace c = constants;

    std::array<float_pt, c::array_len> arr_h;
    std::ranges::generate(arr_h, [i = 0]() mutable { return i++ / std::sqrt(float_pt{2}); });

    // Memory allocation
    const std::size_t mem_size = arr_h.size() * sizeof(float_pt);
    float_pt* arr_d = nullptr;
    check_err(cudaMalloc((void**)&arr_d, mem_size));

    // Memory copy
    check_err(cudaMemcpy(arr_d, arr_h.data(), mem_size, cudaMemcpyHostToDevice));

    // Kernel launch
    const work_division work_div = make_work_div(arr_h.size());
    multiply_kernel<float_pt><<<work_div.blocks, work_div.block_threads>>>(
        arr_d, 2, arr_h.size(), work_div.block_threads.x);

    check_err(cudaMemcpy(arr_h.data(), arr_d, mem_size, cudaMemcpyDeviceToHost));
    check(std::span(arr_h), [](int i) { return i * std::sqrt(float_pt{2}); });

    // Memory deallocation
    check_err(cudaFree(arr_d));
}
