#pragma once

namespace cuda_tutorial::concepts {

template <typename Checker, typename T>
concept checker = requires(Checker c, int i) {
    { c(i) } -> std::convertible_to<T>;
};

} // namespace cuda_tutorial::concepts
