// #include "cuda_tutorial.hpp"

#include <cuda.h>

#include <cassert>
#include <cstdio>
#include <cuda_runtime_api.h>

/* template <typename Func>
    requires std::is_invocable_r_v<void, Func>
__global__ void invoke_kernel(Func&& func) {
    func();
} */

int main() {
    // invoke_kernel<<<1, 16>>>([] __host__ __device__() { printf("shirou\n"); });
    cudaDeviceSynchronize();
}
