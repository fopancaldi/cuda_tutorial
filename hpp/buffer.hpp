#pragma once

#include "util.hpp"

#include <cassert>
#include <cuda.h>
#include <ranges>

namespace cuda_tutorial {

template <typename T>
class gpu_buffer {
    T* m_data;
    std::size_t m_len;

  public:
    gpu_buffer() : m_data{nullptr}, m_len{0} {}
    gpu_buffer(std::size_t len) : m_data{nullptr}, m_len{len} {
        cudaMalloc(&m_data, util::to_bytes<T>(m_len));
    }
    template <std::ranges::view View>
        requires std::ranges::contiguous_range<View> and
                     std::is_same_v<T, std::remove_cv_t<typename View::value_type>>
    gpu_buffer(View view) : m_data{nullptr}, m_len{view.size()} {
        cudaPointerAttributes view_data_attr;
        cudaPointerGetAttributes(&view_data_attr, view.data());
        assert(view_data_attr.type == cudaMemoryTypeHost);

        cudaMalloc(&m_data, util::to_bytes<T>(m_len));
        cudaMemcpy(m_data, view.data(), util::to_bytes<T>(m_len), cudaMemcpyHostToDevice);
    }
    gpu_buffer(gpu_buffer<T> const& other) : m_data{nullptr}, m_len{other.len()} {
        cudaMalloc(&m_data, util::to_bytes<T>(m_len));
        cudaMemcpy(m_data, other.data(), util::to_bytes<T>(m_len), cudaMemcpyDeviceToDevice);
    }
    gpu_buffer(gpu_buffer<T>&& other) : m_data{other.data()}, m_len{other.len()} {
        other.m_data = nullptr;
        other.m_len = 0;
    }
    gpu_buffer<T>& operator=(gpu_buffer<T> const& other) {
        cudaFree(m_data);
        m_len = other.m_len;
        cudaMalloc(&m_data, util::to_bytes<T>(m_len));
        cudaMemcpy(m_data, other.m_data, util::to_bytes<T>(m_len), cudaMemcpyDeviceToDevice);
        return *this;
    }
    gpu_buffer<T>& operator=(gpu_buffer<T>&& other) {
        m_data = other.m_data;
        other.m_data = nullptr;
        m_len = other.m_len;
        return *this;
    }
    ~gpu_buffer() { cudaFree(m_data); }

    T*& data() { return m_data; }
    T const* data() const { return m_data; }
    std::size_t len() const { return m_len; }
};

} // namespace cuda_tutorial
