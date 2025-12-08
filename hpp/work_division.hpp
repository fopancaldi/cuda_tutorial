#pragma once

#include "concepts.hpp"
#include "constants.hpp"
#include "typedefs.hpp"
#include "util.hpp"

#include <cassert>
#include <concepts>
#include <limits>
#include <utility>

namespace cuda_tutorial {

template <concepts::unsigned3 Unsigned3>
inline work_division make_work_div(Unsigned3 elements) {
    namespace c = constants;
    namespace u = util;
    using dim3_val = decltype(std::declval<dim3>().x);

    const auto cast = [](auto u) {
        assert(u <= std::numeric_limits<dim3_val>::max());
        return static_cast<dim3_val>(u);
    };
    const dim3 blocks{cast(u::ratio_rounded_up(elements.x, c::block_threads.x)),
                      cast(u::ratio_rounded_up(elements.y, c::block_threads.y)),
                      cast(u::ratio_rounded_up(elements.z, c::block_threads.z))};

    return {blocks, c::block_threads};
}

template <std::unsigned_integral U>
inline work_division make_work_div(U elements) {
    return make_work_div(ulonglong3{elements, 1, 1});
}

} // namespace cuda_tutorial
