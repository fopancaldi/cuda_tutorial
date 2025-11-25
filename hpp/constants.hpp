#pragma once

#include "typedefs.hpp"

#include <concepts>

namespace cuda_tutorial::constants {

constexpr dim3 block_threads{16, 1, 1};

constexpr size array_len = 1000;

template <std::floating_point FP>
constexpr FP max_relative_error = 0.001f;

template <std::floating_point FP>
constexpr FP max_machine_error = 1000 * std::numeric_limits<FP>::epsilon();

} // namespace cuda_tutorial::constants
