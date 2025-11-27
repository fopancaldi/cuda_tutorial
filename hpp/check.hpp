#pragma once

#include "concepts.hpp"
#include "constants.hpp"

#include <algorithm>
#include <cassert>
#include <cuda.h>
#include <ranges>
#include <span>
#include <utility>
#include <vector>

namespace cuda_tutorial {

inline void check_err(cudaError_t err) { assert(err == cudaError_t::cudaSuccess); }

namespace internal {

template <std::floating_point FP>
bool are_equal(FP fp1, FP fp2) {
    namespace c = constants;

    if (std::abs(fp1) < c::max_machine_error<FP> or std::abs(fp2) < c::max_machine_error<FP>) {
        return std::abs(fp1) < c::max_machine_error<FP> and
               std::abs(fp2) < c::max_machine_error<FP>;
    } else {
        return std::abs((fp1 - fp2) / fp1) < c::max_relative_error<FP>;
    }
}

template <typename T>
    requires(not std::floating_point<T>)
bool are_equal(T const& t1, T const& t2) {
    return t1 == t2;
}

} // namespace internal

// TODO: Extend to ranges (to by taken by const& instead of value)
template <std::ranges::view View1, std::ranges::view View2>
    requires requires {
        typename std::common_type_t<std::ranges::range_value_t<View1>,
                                    std::ranges::range_value_t<View2>>;
    }
void check(View1 view1, View2 view2) {
    assert(view1.size() == view2.size());
    using Elem1 = std::ranges::range_value_t<View1>;
    using Elem2 = std::ranges::range_value_t<View2>;
    using Elem = std::common_type_t<Elem1, Elem2>;

    // TODO: In c++23, use std::ranges::all_of with std::views::zip
    std::vector<Elem> equalities(view1.size());
    std::ranges::transform(view1, view2, equalities.begin(), [](Elem1 const& e1, Elem2 const& e2) {
        return internal::are_equal(e1, e2);
    });
    assert(std::ranges::all_of(equalities, std::identity()));
}

template <std::ranges::view View, concepts::checker<std::ranges::range_value_t<View>> Checker>
void check(View view, Checker&& checker) {
    using Elem = std::ranges::range_value_t<View>;

    // TODO: In c++23, use std::views::enumerate
    std::vector<Elem> to_check(view.size());
    std::ranges::generate(to_check, [&c = std::as_const(checker), i = 0]() mutable {
        return static_cast<Elem>(c(i++));
    });
    check(view, std::span(to_check));
}

} // namespace cuda_tutorial
