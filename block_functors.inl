#pragma once

#include "arithmetic.inl"
#include "primitives.inl"
#include "launch_info.inl"
#include <cub/block/block_reduce.cuh>
#include <cuda.h>

namespace cuda
{
    namespace block
    {
        template <typename launch_t, typename T>
        __device__ __forceinline__ T asum(const T *in, const uint32_t size) {
            typedef cub::BlockReduce<T, launch_t::sm_ptx::nt> BlockReduce;
            typedef typename BlockReduce::TempStorage TempStorage;
            __shared__ TempStorage temp_storage;
            __shared__ T accum_result;
            accum_result = 0;
            auto sum = []__device__(const uint32_t dataSeq, const T *in, TempStorage &temp_storage, T * output) {
                const T aggregate = BlockReduce(temp_storage).Reduce(in[dataSeq], cub::Sum());
                if (threadIdx.x == 0) {
                    *output += aggregate;
                }
                // sync just in case
                __syncthreads();
            };

            block_for(size, sum, temp_storage, &accum_result);
            return accum_result;
        }

        template <typename launch_t, typename T>
        __device__ __forceinline__ T max(const T *in, const uint32_t size)
        {
            typedef cub::BlockReduce<T, launch_t::sm_ptx::nt> BlockReduce;
            typedef typename BlockReduce::TempStorage TempStorage;
            __shared__ TempStorage temp_storage;
            __shared__ T max_result;
            max_result = 1e-318;
            auto max = []__device__(const uint32_t dataSeq, const T *in, TempStorage &temp_storage, T * output) {
                const T local_max = impl::max<BlockReduce>(in[dataSeq], temp_storage);
                if (threadIdx.x == 0) {
                    *output = *output > local_max ? *output : local_max;
                }
                __syncthreads();
            };
            block_for(size, max, in, temp_storage, &max_result);
            return max_result;
        }

        template <typename launch_t, typename T>
        __device__ __forceinline__ T dot(const T *v1, const T *v2, const uint32_t size)
        {
            typedef cub::BlockReduce<T, launch_t::sm_ptx::nt> BlockReduce;
            typedef typename BlockReduce::TempStorage TempStorage;
            __shared__ TempStorage temp_storage;
            __shared__ T result;
            result = 0;
            auto product = []__device__(const uint32_t dataSeq, TempStorage &temp_storage,
                                        const T *v1, const T *v2, T *output) {
                const T iteration_sum = BlockReduce(temp_storage).Sum(v1[dataSeq] * v2[dataSeq]);
                if (threadIdx.x == 0) {
                    *output += iteration_sum;
                }
            };

            block_for(size, dot, v1, v2, temp_storage, &result);
            return result;
        }

        template <typename T>
        __device__ __forceinline__ void scal(const T *in, const T a, const uint32_t size, T *out) {
            block_for(size, impl::cmult<T>, a, in, out);
        }

        template <typename T>
        __device__ __forceinline__ void add(const T *in, const T a, const uint32_t size, T *out) {
            block_for(size, impl::cadd<T>, a, in, out);
        }

        template <typename T>
        __device__ __forceinline__ void copy(const T *src, const uint32_t size, T *dst) {
            block_for(size, impl::copy<T>, src, dst);
        }

        template <typename T>
        __device__ __forceinline__ void zero(T *src, const uint32_t size) {
            block_for(size, impl::clear<T>, src);
        }

        template <typename T>
        __device__ __forceinline__ void axpy(const T a, const T *x, const T *y, const uint32_t size, T *out) {
            block_for(size, impl::fma<T>, a, x, y, out);
        }

        template <typename T>
        __device__ __forceinline__ void ewisemult(const T *a, const T *b, const uint32_t size, T *out) {
            block_for(size, impl::ewisemult<T>, a, b, out);
        }

        template <typename T>
        __device__ __forceinline__ void ewiseadd(const T *a, const T *b, const uint32_t size, T *out) {
            block_for(size, impl::ewiseadd<T>, a, b, out);
        }

        template <typename T>
        __device__ __forceinline__ void ewisefma(const T *x, const T *y, const T z, const uint32_t size, T *out) {
            block_for(size, impl::ewisefma<T>, x, y, z, out);
        }
    }
}
