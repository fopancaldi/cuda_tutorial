#pragma once

#include "constants.hpp"
#include "typedefs.hpp"

#include <concepts>

namespace cuda_tutorial {

namespace internal {

auto ratio_rounded_up(std::integral auto num, std::integral auto den) {
    return (num + den - 1) / den;
}

} // namespace internal

inline work_division make_work_div(dim3 const& elements) {
    namespace c = constants;
    namespace i = internal;

    const dim3 blocks{i::ratio_rounded_up(elements.x, c::block_threads.x),
                      i::ratio_rounded_up(elements.y, c::block_threads.y),
                      i::ratio_rounded_up(elements.z, c::block_threads.z)};

    return {blocks, c::block_threads};
}

} // namespace cuda_tutorial
