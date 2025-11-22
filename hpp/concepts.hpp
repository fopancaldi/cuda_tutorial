#pragma once

#include <type_traits>

namespace cuda_tutorial::concepts {

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template <typename Checker, typename T>
concept checker = requires(Checker c, int i) {
    { c(i) } -> std::convertible_to<T>;
};

} // namespace cuda_tutorial::concepts
