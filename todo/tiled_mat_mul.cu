#include <array>
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <iostream>

#include <concepts>
constexpr unsigned int tileSize = 2;

__host__ __device__ unsigned int CeilFrac(unsigned int n, unsigned int d) {
    return (n + d - 1) / d;
}

template <std::floating_point T, std::size_t Length>
using Matrix = std::array<std::array<T, Length>, Length>;

template <std::floating_point T, std::size_t Length>
T* Ptr(Matrix<T, Length>& m) {
    return m.front().data();
}

template <std::floating_point T, std::size_t Length>
void Print(const Matrix<T, Length>& m) {
    for (const std::array<T, Length> row : m) {
        for (const T& t : row) {
            std::cout << t << ' ';
        }
        std::cout << '\n';
    }
}

template <std::floating_point T, std::size_t Length>
__global__ void Ker(T* lhs_d, T* rhs_d, T* result_d) {
    const unsigned int tix = threadIdx.x;
    const unsigned int tiy = threadIdx.y;
    const unsigned int bix = blockIdx.x;
    const unsigned int biy = blockIdx.y;
    __shared__ T lhsBlock[tileSize][tileSize];
    __shared__ T rhsBlock[tileSize][tileSize];
    T result = 0;
    for (unsigned int i = 0; i < CeilFrac(Length, tileSize); ++i) {
        if ((tileSize * i + tix < Length) && (tileSize * biy + tiy < Length)) {
            lhsBlock[tiy][tix] = lhs_d[Length * (tileSize * biy + tiy) + tileSize * i + tix];
        } else {
            lhsBlock[tiy][tix] = 0;
        }
        if ((tileSize * bix + tix < Length) && (tileSize * i + tiy < Length)) {
            rhsBlock[tiy][tix] = lhs_d[Length * (tileSize * i + tiy) + tileSize * bix + tix];
        } else {
            rhsBlock[tiy][tix] = 0;
        }
        __syncthreads();
        for (unsigned int j = 0; j != tileSize; ++j) {
            result += lhsBlock[tiy][j] * rhsBlock[j][tix];
        }
        __syncthreads();
    }
    if ((tileSize * bix + tix < Length) && (tileSize * biy + tiy < Length)) {
        result_d[Length * (tileSize * biy + tiy) + tileSize * bix + tix] = result;
    }
}

template <std::floating_point T, std::size_t Length>
void MatMul(T* lhs_h, T* rhs_h, T* result_h) {
    T *lhs_d, *rhs_d, *result_d;
    std::size_t memSize = sizeof(T) * Length * Length;
    cudaMalloc((void**)&lhs_d, memSize);
    cudaMalloc((void**)&rhs_d, memSize);
    cudaMalloc((void**)&result_d, memSize);
    cudaMemcpy(lhs_d, lhs_h, memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(rhs_d, rhs_h, memSize, cudaMemcpyHostToDevice);
    Ker<T, Length><<<dim3(CeilFrac(Length, tileSize), CeilFrac(Length, tileSize), 1),
                     dim3(tileSize, tileSize, 1)>>>(lhs_d, rhs_d, result_d);
    cudaMemcpy(result_h, result_d, memSize, cudaMemcpyDeviceToHost);
    cudaFree(lhs_d);
    cudaFree(rhs_d);
    cudaFree(result_d);
}

int main() {
    Matrix<float, 3> m1{std::array<float, 3>{0.5f, 0, 0}, std::array<float, 3>{0, 0.33f, 0},
                        std::array<float, 3>{0, 0, 1}};
    Matrix<float, 3> m2;
    MatMul<float, 3>(Ptr(m1), Ptr(m1), Ptr(m2));
    Print(m1);
    std::cout << "Square of the previous matrix:\n";
    Print(m2);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << '\n';
    }
    cudaDeviceSynchronize();
}
