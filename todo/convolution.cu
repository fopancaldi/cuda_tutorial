#include "cufft.hpp"

#include <iostream>

#include <algorithm>
#include <iterator>
#include <numeric>

namespace ct = cuda_tutorial;

template <ct::concepts::arithmetic A1, ct::concepts::arithmetic A2, std::size_t N1, std::size_t N2>
    requires(N1 > 0 and N2 > 0)
auto convolve(std::array<A1, N1> const& arr1, std::array<A2, N2> const& arr2) {
    using A = std::common_type_t<A1, A2>;

    std::array<A, N1 + N2 - 1> result;
    for (std::size_t i = 0; i < result.size(); ++i) {
        const std::size_t start_idx = i > N2 - 1 ? i - N2 + 1 : 0;
        const std::size_t end_idx = std::min(N1 - 1, i) + 1;
        result[i] =
            std::inner_product(arr1.begin() + start_idx, arr1.begin() + end_idx,
                               std::reverse_iterator(arr2.begin() + i - start_idx) - 1, A{0});
    }
    return result;
}

int main() {}
