#include "cuda_tutorial.hpp"

#include <array>
#include <cstddef>
#include <numeric>
#include <ranges>
#include <span>
#include <vector>

namespace ct = cuda_tutorial;

template <typename T, typename CompFunc>
    requires std::is_invocable_r_v<bool, CompFunc, T const&, T const&>
__global__ void max_elem_reduce_kernel(T const* values_in, std::size_t const* indices_in,
                                       T* values_out, std::size_t* indices_out, std::size_t arr_len,
                                       CompFunc comp_func) {
    namespace ctu = ct::util;

    assert(threadIdx.y == 0 and threadIdx.z == 0);
    assert(blockIdx.y == 0 and blockIdx.z == 0);

    extern __shared__ std::byte shared_mem_arr[];
    T* shared_arr_v = reinterpret_cast<T*>(shared_mem_arr);
    std::size_t* shared_arr_i =
        reinterpret_cast<std::size_t*>(shared_mem_arr + ctu::to_bytes<T>(blockDim.x));
    unsigned int const global_idx = threadIdx.x + blockIdx.x * blockDim.x;
    bool const outside_arr = (global_idx >= arr_len);
    if (not outside_arr) {
        shared_arr_v[threadIdx.x] = values_in[global_idx];
        shared_arr_i[threadIdx.x] = indices_in[global_idx];
    }
    __syncthreads();

    for (unsigned int i = 1; i < blockDim.x; i *= 2) {
        if (threadIdx.x % (2 * i) == 0) {
            bool const right_greater =
                outside_arr ? false
                            : comp_func(shared_arr_v[threadIdx.x], shared_arr_v[threadIdx.x + i]);
            if (right_greater) {
                shared_arr_v[threadIdx.x] = shared_arr_v[threadIdx.x + i];
                shared_arr_i[threadIdx.x] = shared_arr_i[threadIdx.x + i];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        values_out[blockIdx.x] = shared_arr_v[0];
        indices_out[blockIdx.x] = shared_arr_i[0];
    }
}

template <typename T, typename CompFunc>
    requires std::is_invocable_r_v<bool, CompFunc, T const&, T const&>
std::pair<ct::gpu_buffer<T>, ct::gpu_buffer<std::size_t>>
max_elem_reduce_iter(ct::gpu_buffer<T> const& values, ct::gpu_buffer<std::size_t> const& indices,
                     CompFunc&& comp_func) {
    namespace ctc = ct::constants;
    namespace ctu = ct::util;

    std::size_t const orig_len = values.len();
    assert(orig_len == indices.len());
    std::size_t const result_len = ctu::ratio_rounded_up(orig_len, ctc::block_threads.x);
    ct::gpu_buffer<T> result_v(result_len);
    ct::gpu_buffer<std::size_t> result_i(result_len);

    ct::work_division const work_div = ct::make_work_div(orig_len);
    std::size_t const shared_bytes = ctu::to_bytes<T>(work_div.block_threads.x) +
                                     ctu::to_bytes<std::size_t>(work_div.block_threads.x);
    max_elem_reduce_kernel<T, CompFunc><<<work_div.blocks, work_div.block_threads, shared_bytes>>>(
        values.data(), indices.data(), result_v.data(), result_i.data(), orig_len, comp_func);

    return {result_v, result_i};
}

template <typename T, typename CompFunc>
    requires std::is_invocable_r_v<bool, CompFunc, T const&, T const&>
std::size_t max_element_idx(ct::gpu_buffer<T> const& orig_values, CompFunc&& comp_func) {
    std::vector<std::size_t> indices_h(orig_values.len());
    std::iota(indices_h.begin(), indices_h.end(), 0);

    auto [values, indices] = max_elem_reduce_iter(
        orig_values, ct::gpu_buffer<std::size_t>(std::span(indices_h)), comp_func);
    while (indices.len() > 1) {
        auto [values_, indices_] = max_elem_reduce_iter(values, indices, comp_func);
        assert(values_.len() == indices_.len());
        values = values_;
        indices = indices_;
    }

    assert(indices.len() == 1);
    std::size_t result;
    cudaMemcpy(&result, indices.data(), ct::util::to_bytes<std::size_t>(1), cudaMemcpyDeviceToHost);
    return result;
}

template <std::ranges::contiguous_range Range, typename CompFunc>
    requires std::is_invocable_r_v<bool, CompFunc, typename Range::value_type const&,
                                   typename Range::value_type const&>
auto max_element(Range const& range_h, CompFunc&& comp_func) {
    using value = Range::value_type;
    return range_h.begin() + max_element_idx(ct::gpu_buffer<value>(std::span(range_h)), comp_func);
}

int main() {
    using namespace ct;
    namespace c = constants;

    std::array<float_pt, c::array_len> arr_h;
    std::ranges::generate(arr_h, [i = 0]() mutable { return i++ / c::sqrt3<float_pt>; });

    auto max_element_ =
        max_element(arr_h, [] __host__ __device__(float_pt const& lhs, float_pt const& rhs) {
            return lhs < rhs;
        });
    assert(std::distance(arr_h.cbegin(), max_element_) == (arr_h.size() - 1));
    check_never_err();
}
