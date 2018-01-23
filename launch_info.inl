#pragma once

#include <moderngpu/meta.hxx>
#include <cuda.h>
namespace cuda
{
    template <typename launch_t>
    __host__ __device__ __forceinline__
    constexpr int warps_per_block() {
        return launch_t::sm_ptx::nt / mgpu::warp_size;
    }

    template <typename launch_t, typename T>
    __host__ __device__ __forceinline__
    constexpr size_t blocks(const T worksize) {
        return mgpu::div_up(static_cast<size_t>(worksize), static_cast<size_t>(launch_t::sm_ptx::nt));
    }

    template <typename launch_t>
    __host__ __device__ __forceinline__
    constexpr size_t threads() {
        return static_cast<size_t>(launch_t::sm_ptx::nt);
    }

    template <typename launch_t>
    __host__ __device__ __forceinline__
    constexpr size_t shm_size(const size_t size) {
        return warps_per_block<launch_t>() * size;
    }

    __host__ __device__ __forceinline__
    std::uint8_t get_lane_id() {
        return (mgpu::warp_size - 1) & threadIdx.x;
    }

    __host__ __device__ __forceinline__
    std::uint8_t get_warp_id() {
        return threadIdx.x / mgpu::warp_size;
    }
}