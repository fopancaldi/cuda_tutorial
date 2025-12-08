#pragma once

#include <concepts>
#include <cstddef>

namespace cuda_tutorial::util {

auto ratio_rounded_up(std::integral auto num, std::integral auto den) {
    return (num + den - 1) / den;
}

template <typename T>
std::size_t to_bytes(std::size_t count) {
    return count * sizeof(T);
}

} // namespace cuda_tutorial::util
