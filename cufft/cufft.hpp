#pragma once

#include "cuda_tutorial.hpp"

#include <cufft.h>

namespace cuda_tutorial {

namespace internal {

template <typename>
struct cufft_str {};

template <>
struct cufft_str<float> {
    using real_type = cufftReal;
    using complex_type = cufftComplex;
    static cufftResult cufft_exec_c2c(cufftHandle h, cufftComplex* i, cufftComplex* o, int d) {
        return cufftExecC2C(h, i, o, d);
    }
    static constexpr cufftType cufft_type_c2c = CUFFT_C2C;
};

template <>
struct cufft_str<double> {
    using real_type = cufftDoubleReal;
    using complex_type = cufftDoubleComplex;
    static cufftResult cufft_exec_c2c(cufftHandle h, cufftDoubleComplex* i, cufftDoubleComplex* o,
                                      int d) {
        return cufftExecZ2Z(h, i, o, d);
    }
    static constexpr cufftType cufft_type_c2c = CUFFT_Z2Z;
};

} // namespace internal

using cufftReal = internal::cufft_str<float_pt>::real_type;
using cufftComplex = internal::cufft_str<float_pt>::complex_type;
inline cufftResult cufftExecC2C(cufftHandle h, cuda_tutorial::cufftComplex* i,
                                cuda_tutorial::cufftComplex* o, int d) {
    return internal::cufft_str<float_pt>::cufft_exec_c2c(h, i, o, d);
}
constexpr cufftType cufftTypeC2C = internal::cufft_str<float_pt>::cufft_type_c2c;

} // namespace cuda_tutorial
