#pragma once

#include "concepts.hpp"
#include "constants.hpp"

#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <ranges>
#include <utility>

namespace cuda_tutorial {

inline void check_err(cudaError_t err) { assert(err == cudaError_t::cudaSuccess); }

namespace internal {

// TODO: This is a rather primitive way of doing the comparison. It performs poorly when exactly one
// of the two values is 0
template <std::floating_point FP>
bool are_equal(FP fp1, FP fp2) {
    if (fp1 == 0 and fp2 == 0) {
        return true;
    } else {
        const FP den = fp1 == 0 ? fp2 : fp1;
        return std::abs((fp1 - fp2) / den) < constants::max_relative_error<FP>;
    }
}

template <typename T>
    requires(not std::floating_point<T>)
bool are_equal(T const& t1, T const& t2) {
    return t1 == t2;
}

} // namespace internal

// TODO: Extend to ranges (to by taken by const& instead of value)
template <std::ranges::view View, concepts::checker<std::ranges::range_value_t<View>> Checker>
void check(View view, Checker&& checker) {
    using Elem = std::ranges::range_value_t<View>;

    // TODO: In c++23, use std::views::enumerate
    assert(std::ranges::all_of(view, [&c = std::as_const(checker), i = 0](Elem const& x) mutable {
        return internal::are_equal(x, static_cast<Elem>(c(i++)));
    }));
}

} // namespace cuda_tutorial
