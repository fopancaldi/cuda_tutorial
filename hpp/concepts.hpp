#pragma once

#include <concepts>
#include <type_traits>

namespace cuda_tutorial::concepts {

template <typename T1, typename T2>
concept common_type = requires { typename std::common_type_t<T1, T2>; };

template <typename T>
concept arithmetic = std::is_arithmetic_v<T>;

template <typename T>
concept integral_or_int_ref = std::integral<std::remove_reference_t<T>>;

template <typename T>
concept unsigned3 = requires(T t) {
    { t.x } -> integral_or_int_ref;
    { t.y } -> integral_or_int_ref;
    { t.z } -> integral_or_int_ref;
};

template <typename Checker, typename T>
concept checker = requires(Checker c, int i) {
    { c(i) } -> std::convertible_to<T>;
};

} // namespace cuda_tutorial::concepts
