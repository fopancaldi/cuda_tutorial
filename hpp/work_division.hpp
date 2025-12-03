#pragma once

#include "constants.hpp"
#include "typedefs.hpp"

#include <cassert>
#include <concepts>
#include <limits>

namespace cuda_tutorial {

namespace internal {

auto ratio_rounded_up(std::integral auto num, std::integral auto den) {
    return (num + den - 1) / den;
}

} // namespace internal

inline work_division make_work_div(ulonglong3 elements) {
    namespace c = constants;
    namespace i = internal;

    const auto cast = [](unsigned long long u) {
        assert(u <= std::numeric_limits<unsigned int>::max());
        return static_cast<unsigned int>(u);
    };
    const dim3 blocks{cast(i::ratio_rounded_up(elements.x, c::block_threads.x)),
                      cast(i::ratio_rounded_up(elements.y, c::block_threads.y)),
                      cast(i::ratio_rounded_up(elements.z, c::block_threads.z))};

    return {blocks, c::block_threads};
}

inline work_division make_work_div(std::size_t elements) {
    return make_work_div(ulonglong3{elements, 1, 1});
}

} // namespace cuda_tutorial
