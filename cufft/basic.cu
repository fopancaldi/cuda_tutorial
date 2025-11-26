#include "cufft.hpp"

#include <array>
#include <complex>
#include <functional>
#include <ranges>
#include <span>

namespace ct = cuda_tutorial;

template <typename T>
using rw = std::reference_wrapper<T>;

int main() {
    using namespace ct;
    namespace c = constants;

    constexpr float_pt frequency = c::sqrt2<float_pt> - 1;

    std::array<ct::cufftComplex, c::array_len> signal_h;
    std::ranges::generate(signal_h, [i = 0, frequency]() mutable {
        const std::complex c = std::polar<float_pt>(1, 2 * c::pi<float_pt> * frequency * i++);
        return ct::cufftComplex{c.real(), c.imag()};
    });

    cufftHandle plan;
    ct::cufftComplex *signal_d = nullptr, *dft_d = nullptr;
    cufftPlan1d(&plan, static_cast<int>(std::ssize(signal_h)), ct::cufftTypeC2C, 1);

    const std::size_t bytes = signal_h.size() * sizeof(ct::cufftComplex);
    for (ct::cufftComplex*& ptr : {rw(signal_d), rw(dft_d)}) {
        cudaMalloc(&ptr, bytes);
    }
    cudaMemcpy(signal_d, signal_h.data(), bytes, cudaMemcpyHostToDevice);

    ct::cufftExecC2C(plan, signal_d, dft_d, CUFFT_FORWARD);

    std::array<ct::cufftComplex, signal_h.size()> dft_h;
    cudaMemcpy(dft_h.data(), dft_d, bytes, cudaMemcpyDeviceToHost);
    for (ct::cufftComplex*& ptr : {rw(signal_d), rw(dft_d)}) {
        cudaFree(ptr);
    }

    check(std::span(dft_h) | std::views::transform([](ct::cufftComplex c) {
              return std::sqrt(c.x * c.x + c.y * c.y);
          }),
          [len = signal_h.size(), frequency](int i) {
              return std::abs(
                  std::sin(len * c::pi<float_pt> * frequency) /
                  std::sin(c::pi<float_pt> * (frequency - static_cast<float_pt>(i) / len)));
          });
}
