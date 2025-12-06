#include <cassert>
#include <cuda.h>
#include <iostream>

// TODO: namespace cuda_tutorial::error?
namespace cuda_tutorial {

inline void check_err(cudaError_t err) { assert(err == cudaError_t::cudaSuccess); }

inline void check_never_err() {
    cudaDeviceSynchronize();
    check_err(cudaPeekAtLastError());
}

inline void print_last_err() {
    cudaDeviceSynchronize();
    std::cout << cudaGetErrorName(cudaPeekAtLastError()) << '\n';
}

inline void print_last_err_code() {
    cudaDeviceSynchronize();
    std::cout << cudaPeekAtLastError() << '\n';
}

} // namespace cuda_tutorial
