#include "cuda_tutorial.hpp"

#include <algorithm>
#include <array>
#include <ranges>
#include <span>
#include <type_traits>

namespace ct = cuda_tutorial;

template <typename T>
struct gpu_array {
    T* data;
    std::size_t len;
};

template <typename T>
std::size_t to_bytes(std::size_t count) {
    return count * sizeof(T);
}

auto round_up_to_multiple(std::integral auto to_round, std::integral auto multiple_base) {
    return multiple_base * ct::util::ratio_rounded_up(to_round, multiple_base);
}

template <typename T, typename RedFunc>
    requires std::is_invocable_r_v<T, RedFunc, T const&, T const&>
__global__ void reduce_iter_kernel(T const* in_arr, T* out_arr, RedFunc red_func,
                                   unsigned int block_threads) {
    assert(threadIdx.y == 0 and threadIdx.z == 0);
    assert(blockIdx.y == 0 and blockIdx.z == 0);

    extern __shared__ T shared_arr[];
    unsigned int const global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    shared_arr[threadIdx.x] = in_arr[global_idx];
    __syncthreads();

    for (unsigned int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x % (2 * i) == 0) {
            shared_arr[threadIdx.x] += shared_arr[threadIdx.x + i];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        out_arr[blockIdx.x] = shared_arr[0];
    }
}

template <typename T, typename RedFunc>
    requires std::is_invocable_r_v<T, RedFunc, T const&, T const&>
void do_reduce_iter(gpu_array<T>& arr, RedFunc&& red_func, T const& neutral_element) {
    namespace ctc = ct::constants;
    namespace ctu = ct::util;

    std::size_t const ideal_reduced_len = ctu::ratio_rounded_up(arr.len, ctc::block_threads.x);
    gpu_array<T> reduced_arr{
        .data = nullptr,
        .len = ideal_reduced_len == 1
                   ? 1
                   : round_up_to_multiple(ideal_reduced_len, ctc::block_threads.x)};
    cudaMalloc(&(reduced_arr.data), to_bytes<T>(arr.len));
    ct::fill<T>(reduced_arr.data + ideal_reduced_len, neutral_element,
                reduced_arr.len - ideal_reduced_len);

    ct::work_division const work_div = ct::make_work_div(arr.len);
    reduce_iter_kernel<<<work_div.blocks, work_div.block_threads,
                         to_bytes<T>(work_div.block_threads.x)>>>(arr.data, reduced_arr.data,
                                                                  red_func, ctc::block_threads.x);

    cudaFree(arr.data);
    arr = reduced_arr;
}

template <std::ranges::view View, typename RedFunc>
    requires std::ranges::contiguous_range<View> and
             std::is_invocable_r_v<typename View::value_type, RedFunc,
                                   typename View::value_type const&,
                                   typename View::value_type const&>
View::value_type reduce(View view_h, RedFunc&& red_func,
                        typename View::value_type const& neutral_element) {
    namespace ctc = ct::constants;
    using value = View::value_type;

    gpu_array<value> arr{.data = nullptr,
                         .len = round_up_to_multiple(view_h.size(), ctc::block_threads.x)};
    cudaMalloc(&(arr.data), to_bytes<value>(arr.len));
    cudaMemcpy(arr.data, view_h.data(), to_bytes<value>(arr.len), cudaMemcpyHostToDevice);
    ct::fill<value>(arr.data + view_h.size(), neutral_element, arr.len - view_h.size());

    while (arr.len > 1) {
        do_reduce_iter(arr, red_func, neutral_element);
    }

    assert(arr.len == 1);
    value result;
    cudaMemcpy(&result, arr.data, to_bytes<value>(1), cudaMemcpyDeviceToHost);
    cudaFree(arr.data);
    return result;
}

int main() {
    using namespace ct;
    namespace c = constants;

    std::array<float_pt, c::array_len> in_h;
    std::ranges::generate(in_h, [i = 0]() mutable { return i++ / c::sqrt3<float_pt>; });

    float_pt const out_h = reduce(
        std::span(in_h),
        [] __host__ __device__(float_pt const& lhs, float_pt const& rhs) { return lhs + rhs; }, 0);
    // TODO: Use a better comparison
    assert(
        internal::are_equal(out_h, c::array_len * (c::array_len - 1) / (2 * c::sqrt3<float_pt>)));
    check_never_err();
}
