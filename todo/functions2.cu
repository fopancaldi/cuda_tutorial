#include "cuda_tutorial.hpp"

#include <concepts>
#include <functional>

namespace cuda_tutorial::concepts {

template <typename T1, typename T2, typename Comp>
concept comparable = requires(T1 t1, T2 t2, Comp c) {
    { std::invoke(c, t1, t2) } -> std::convertible_to<bool>;
};

} // namespace cuda_tutorial::concepts

namespace ct = cuda_tutorial;

template <typename T, typename Comp = decltype(std::less())>
const T* max(const T* arr_begin, std::size_t arr_len) {
    return nullptr;
}

template <typename Comp>
    requires ct::concepts::comparable<int, int, Comp>
__global__ void compare_kernel(int* lhs, int* rhs, ct::gpu_bool* result, Comp comp) {
    printf("atsuya\n");
    *result = std::invoke(comp, *lhs, *rhs);
    printf("shirou\n");
}

struct less_str {
    template <typename T1, typename T2>
    __device__ bool operator()(T1 const& lhs, T2 const& rhs) {
        return lhs < rhs;
    }
};

int main() {
    using namespace ct;

    const int i1 = 3, i2 = 5;
    gpu_bool result = false;
    int *i1_d = nullptr, *i2_d = nullptr;
    gpu_bool* result_d = nullptr;
    cudaMalloc(&i1_d, sizeof(int));
    cudaMalloc(&i2_d, sizeof(int));
    cudaMalloc(&result_d, sizeof(gpu_bool));
    cudaMemcpy(i1_d, &i1, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(i2_d, &i2, sizeof(int), cudaMemcpyHostToDevice);

    const auto less_l = [] __host__ __device__(int const& lhs, int const& rhs) {
        return lhs < rhs;
    };

    compare_kernel<<<1, 1>>>(i1_d, i2_d, result_d, less_l);

    cudaMemcpy(&result, result_d, sizeof(gpu_bool), cudaMemcpyDeviceToHost);
    cudaFree(i1_d);
    cudaFree(i2_d);
    cudaFree(result_d);
    assert(result);
}
