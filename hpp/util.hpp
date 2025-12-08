#pragma once

#include <concepts>
#include <cstddef>

namespace cuda_tutorial::util {

__host__ __device__ auto ratio_rounded_up(std::integral auto num, std::integral auto den) {
    return (num + den - 1) / den;
}

__host__ __device__ auto round_up_to_multiple(std::integral auto to_round,
                                              std::integral auto multiple_base) {
    return multiple_base * ratio_rounded_up(to_round, multiple_base);
}

template <typename T>
__host__ __device__ std::size_t to_bytes(std::size_t count) {
    return count * sizeof(T);
}

} // namespace cuda_tutorial::util
