#include "cuda_tutorial.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <span>

namespace ct = cuda_tutorial;

template <ct::concepts::arithmetic A>
__global__ void multiply_kernel(A* ptr, A factor, std::size_t arr_len, unsigned int block_threads) {
    assert(threadIdx.y == 0 and threadIdx.z == 0);
    assert(blockIdx.y == 0 and blockIdx.z == 0);
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

    // Streams
    cudaStream_t stream;
    check_err(cudaStreamCreate(&stream));

    // Allocation
    const std::size_t mem_size = arr_h.size() * sizeof(float_pt);
    float_pt* arr_d = nullptr;
    check_err(cudaMallocAsync((void**)&arr_d, mem_size, stream));

    // Copy
    check_err(cudaMemcpyAsync(arr_d, arr_h.data(), mem_size, cudaMemcpyHostToDevice, stream));

    // Kernels
    const work_division work_div = make_work_div(arr_h.size());
    multiply_kernel<float_pt><<<work_div.blocks, work_div.block_threads, 0, stream>>>(
        arr_d, 2, arr_h.size(), work_div.block_threads.x);

    check_err(cudaMemcpyAsync(arr_h.data(), arr_d, mem_size, cudaMemcpyDeviceToHost, stream));

    // Deallocation
    check_err(cudaFreeAsync(arr_d, stream));

    check_err(cudaStreamSynchronize(stream));
    check(std::span(arr_h), [](int i) { return i * std::sqrt(float_pt{2}); });
}
