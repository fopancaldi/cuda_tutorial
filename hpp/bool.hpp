#include <cassert>
#include <cstdint>

namespace cuda_tutorial {

class gpu_bool {
    uint8_t m_value;

  public:
    __host__ __device__ gpu_bool(bool b) : m_value{b} {}
    __host__ __device__ operator bool() const {
        assert(m_value <= 1);
        return static_cast<bool>(m_value);
    }
};

} // namespace cuda_tutorial
