#include "cuda_tutorial.hpp"

#include <functional>
#include <type_traits>

namespace ct = cuda_tutorial;

template <typename T, typename Func>
    requires std::is_invocable_r_v<T, Func, T const&>
__global__ void invoke_kernel(T const* input, T* output, Func func) {
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

struct double_str {
    template <ct::concepts::arithmetic A>
    __host__ __device__ A operator()(A const& a) const {
        return A{2} * a;
    }
};

template <ct::concepts::arithmetic auto Factor>
struct templ_multiply_str {
    template <ct::concepts::arithmetic A>
        requires ct::concepts::common_type<A, decltype(Factor)>
    __host__ __device__ auto operator()(A const& a) const {
        return a * Factor;
    }
};

template <ct::concepts::arithmetic A>
struct multiply_str {
    A value;
    template <ct::concepts::arithmetic A2>
        requires ct::concepts::common_type<A, A2>
    __host__ __device__ auto operator()(A2 const& a) const {
        return a * value;
    }
};

template <ct::concepts::arithmetic A>
__device__ A double_fn(A const& a) {
    return A{2} * a;
}

template <ct::concepts::arithmetic A>
__device__ A (*const double_fn_ptr)(A const&) = double_fn;

template <ct::concepts::pointer FnPtr, ct::concepts::arithmetic A>
    requires std::is_invocable_r_v<A, std::remove_pointer_t<FnPtr>, A const&>
class call_fn_cl {
    A (*fn_ptr)(A const&);

  public:
    call_fn_cl(A (*const& device_fn_ptr)(A const&)) {
        cudaMemcpyFromSymbol(&fn_ptr, device_fn_ptr, sizeof(decltype(device_fn_ptr)));
    }
    template <ct::concepts::arithmetic A2>
        requires ct::concepts::common_type<A, A2>
    __host__ __device__ auto operator()(A2 const& a) const {
        return (*fn_ptr)(a);
    }
};

int main() {
    constexpr int input = 5, expected_output = 2 * input;

    check_invocation(input, expected_output, double_str{});
    check_invocation(input, expected_output, templ_multiply_str<2>{});
    check_invocation(input, expected_output, multiply_str{2});
    check_invocation(input, expected_output, [] __host__ __device__(int i) { return 2 * i; });
    check_invocation(input, expected_output,
                     call_fn_cl<int (*)(int const&), int>(double_fn_ptr<int>));
    ct::check_never_err();
}
