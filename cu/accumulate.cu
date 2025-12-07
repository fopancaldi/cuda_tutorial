#include "cuda_tutorial.hpp"

#include <algorithm>
#include <array>
#include <ranges>
#include <span>

namespace ct = cuda_tutorial;

template <ct::concepts::arithmetic A>
__global__ void accumulate_kernel(A const* in, A* out, std::size_t arr_len,
                                  unsigned int block_threads) {
    assert(threadIdx.y == 0 and threadIdx.z == 0);
    assert(blockIdx.y == 0 and blockIdx.z == 0);
    assert(*out == 0);
    unsigned int const i = threadIdx.x + blockIdx.x * block_threads;
    __shared__ A partial_sum;
    if (threadIdx.x == 0) {
        partial_sum = 0;
    }
    __syncthreads();
    if (i < arr_len) {
        atomicAdd(&partial_sum, in[i]);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        atomicAdd(out, partial_sum);
    }
}

template <std::ranges::view View>
    requires std::ranges::contiguous_range<View> and
             ct::concepts::arithmetic<typename View::value_type>
View::value_type accumulate(View view_h) {
    using value = View::value_type;

    value *input_d = nullptr, *output_d = nullptr;
    std::size_t const bytes_in = view_h.size() * sizeof(value), bytes_out = sizeof(value);
    cudaMalloc(&input_d, bytes_in);
    cudaMalloc(&output_d, bytes_out);
    cudaMemcpy(input_d, view_h.data(), bytes_in, cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0x00, bytes_out);

    ct::work_division const work_div = ct::make_work_div(view_h.size());
    accumulate_kernel<value><<<work_div.blocks, work_div.block_threads>>>(
        input_d, output_d, view_h.size(), work_div.block_threads.x);

    value result;
    cudaMemcpy(&result, output_d, bytes_out, cudaMemcpyDeviceToHost);
    cudaFree(input_d);
    cudaFree(output_d);
    return result;
}

int main() {
    using namespace ct;
    namespace c = constants;

    std::array<float_pt, c::array_len> in_h;
    std::ranges::generate(in_h, [i = 0]() mutable { return i++ / c::sqrt3<float_pt>; });

    float_pt const out_h = accumulate(std::span(in_h));
    // TODO: Use a better comparison
    assert(
        internal::are_equal(out_h, c::array_len * (c::array_len - 1) / (2 * c::sqrt3<float_pt>)));
    check_never_err();
}
