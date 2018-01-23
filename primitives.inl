#pragma once

#include <cstdint>
#include <cmath>
#include <vector>
#include <type_traits>
#include <string>
#include <exception>
#include <thrust/device_ptr.h>
#include <thrust/iterator/iterator_traits.h>
#include <cub/warp/warp_reduce.cuh>
#include <cub/block/block_reduce.cuh>
#include <moderngpu/meta.hxx>
#include <cuda.h>
#include "arithmetic.inl"
#include "launch_info.inl"

namespace cuda
{
    // warp-wide cooperation: all threads do operation
    template <typename SizeT, typename Op, typename ...OutT>
    __device__ __forceinline__ void warp_for(SizeT length, Op operation = Op(), OutT... args)
    {
        const size_t iterations = mgpu::div_up(static_cast<size_t>(length), static_cast<size_t>(mgpu::warp_size));
        for (uint32_t c = 0; c < iterations; ++c) {
            const uint32_t dataSeq = c * mgpu::warp_size + cuda::get_lane_id();
            if (dataSeq < length){
                operation(dataSeq, args...);
            }
        }
    }

    struct warp_op
    {
        template <typename SizeT, typename Op, typename ...OutT>
        __device__ __forceinline__ void operator()(SizeT length, Op operation = Op(), OutT... args)
        {
            const size_t iterations = mgpu::div_up(static_cast<size_t>(length), static_cast<size_t>(mgpu::warp_size));
            for (uint32_t c = 0; c < iterations; ++c) {
                const uint32_t dataSeq = c * mgpu::warp_size + cuda::get_lane_id();
                if (dataSeq < length){
                    operation(dataSeq, args...);
                }
            }
        }
    };

    // block-wide cooperation: all threads do operation
    template <typename launch_t, typename SizeT, typename Op, typename ...OutT>
    __device__ __forceinline__ void block_for(SizeT length, Op operation = Op(), OutT... args)
    {
        __syncthreads();
        const size_t iterations = mgpu::div_up(static_cast<size_t>(length), static_cast<size_t>(launch_t::sm_ptx::nt));
        for (uint32_t c = 0; c < iterations; ++c) {
            const uint32_t dataSeq = c * launch_t::sm_ptx::nt + threadIdx.x;
            if (dataSeq < length) {
                operation(dataSeq, args...);
            }
        }
    }

    struct block_op
    {
        template <typename launch_t, typename SizeT, typename Op, typename ...OutT>
        __device__ __forceinline__ void operator()(SizeT length, Op operation = Op(), OutT... args) {
            const size_t iterations = mgpu::div_up(static_cast<size_t>(length),
                                                   static_cast<size_t>(launch_t::sm_ptx::nt));
            for (uint32_t c = 0; c < iterations; ++c) {
                const uint32_t dataSeq = c * launch_t::sm_ptx::nt + threadIdx.x;
                if (dataSeq < length) {
                    operation(dataSeq, args...);
                }
            }
        }
    };

    template <typename DeviceVectorT>
    __host__ __forceinline__
    typename DeviceVectorT::value_type * to_ptr(DeviceVectorT &vec)
    {
        return const_cast<typename DeviceVectorT::value_type *>(thrust::raw_pointer_cast(&vec[0]));
    }

    // pointer to thrust::device_ptr (essentially an iterator)
    template <typename T>
    __host__ __forceinline__
    thrust::device_ptr<T> ptoi(T *ptr)
    {
        return thrust::device_pointer_cast(ptr);
    }

    // (fake) iterator to pointer
    template <typename IterT>
    __host__ __forceinline__
    typename std::enable_if<std::is_pointer<IterT>::value, IterT>::type itop(IterT ptr)
    {
        return ptr;
    }

    // (real) iterator to pointer. need to verify
    template <typename IterT>
    __host__ __forceinline__
    typename std::enable_if<!std::is_pointer<IterT>::value,
            typename std::iterator_traits<IterT>::value_type>::type * itop(IterT ptr)
    {
        return const_cast<typename std::iterator_traits<IterT>::value_type *>(thrust::raw_pointer_cast(&(*ptr)));
    }

    template <typename T>
    __host__ __device__ __forceinline__
    constexpr T * ptr_at(T * ptr, const uint32_t row, const uint32_t col, const uint32_t ncols) {
        return ptr + row * ncols + col;
    }
}

