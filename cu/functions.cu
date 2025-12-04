#include "cuda_tutorial.hpp"

#include <functional>
#include <type_traits>

namespace ct = cuda_tutorial;

template <typename T, typename Func>
    requires std::is_invocable_r_v<T, Func, T const&>
__global__ void invoke_kernel(T const* input, T* output, Func&& func) {
    *output = func(*input);
}

template <typename T, typename Func>
    requires std::is_invocable_r_v<T, Func, T const&>
void check_invocation(T const& input, T const& expected_output, Func&& func) {
    using rw = std::reference_wrapper<T*>;

    constexpr std::size_t bytes = sizeof(T);
    T *input_d = nullptr, *output_d = nullptr;
    for (T*& ptr : {rw(input_d), rw(output_d)}) {
        cudaMalloc(&ptr, bytes);
    }

    cudaMemcpy(input_d, &input, bytes, cudaMemcpyHostToDevice);
    invoke_kernel<<<1, 1>>>(input_d, output_d, func);
    T output;
    cudaMemcpy(&output, output_d, bytes, cudaMemcpyDeviceToHost);

    for (T* ptr : {input_d, output_d}) {
        cudaFree(ptr);
    }
    assert(output == expected_output);
}

/* template <ct::concepts::arithmetic A>
__host__ __device__ A double_fn(A const& a) {
    return A{2} * a;
} */

struct double_str {
    template <ct::concepts::arithmetic A>
    __host__ __device__ A operator()(A const& a) const {
        return A{2} * a;
    }
};

template <ct::concepts::arithmetic auto Factor>
struct multiply_str {
    template <ct::concepts::arithmetic A>
        requires requires { typename std::common_type_t<A, decltype(Factor)>; }
    __host__ __device__ auto operator()(A const& a) const {
        return a * Factor;
    }
};

int main() {
    constexpr int input = 5, expected_output = 2 * input;

    check_invocation(input, expected_output, double_str{});
    check_invocation(input, expected_output, multiply_str<2>{});
    check_invocation(input, expected_output, [] __host__ __device__(int i) { return 2 * i; });
}
