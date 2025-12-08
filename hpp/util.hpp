#pragma once

#include <cstddef>

namespace cuda_tutorial {

template <typename T>
std::size_t to_bytes(std::size_t count) {
    return count * sizeof(T);
}

} // namespace cuda_tutorial
