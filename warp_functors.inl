#pragma once

/**
 * warp-for callable functions
 */

#include "primitives.inl"

namespace cuda
{
    namespace warp {

        template <typename launch_t, typename T, typename TempStorage=typename cub::WarpReduce<T>::TempStorage>
        __device__ __forceinline__ T transposed_dot(const T * const v1, const T * const v2,
                                                    const uint32_t v2_cols, const uint32_t size,
                                                    TempStorage &temp_storage)
        {
            enum { warps = warps_per_block<launch_t>() };
            __shared__ T output[warps];
            const std::uint8_t warpid = get_warp_id();
            output[warpid] = 0;

            auto product = []__device__(const uint32_t dataSeq, TempStorage &temp_storage,
                                        const T * const v1, const T * const v2, const uint32_t v2_cols, T *output) {
                const T iteration_sum = cub::WarpReduce<T>(temp_storage).Sum(v1[dataSeq] * v2[dataSeq * v2_cols]);
                if (get_lane_id() == 0) {
                    output[get_warp_id()] += iteration_sum;
                }
            };
            warp_for(size, product, temp_storage, v1, v2, v2_cols, output);
            return output[warpid];
        }

        template <typename launch_t, typename T, typename WarpStorage=typename cub::WarpReduce<T>::TempStorage>
        __device__ __forceinline__ T dot(const T *v1, const T *v2, const uint32_t size, WarpStorage &temp_storage)
        {
            enum { warps = warps_per_block<launch_t>() };
            __shared__ T output[warps];
            const std::uint8_t warpid = get_warp_id();
            output[warpid] = 0;

            auto product = []__device__(const uint32_t dataSeq, WarpStorage *temp_storage,
                                        const T *v1, const T *v2, T *output) {
                auto iteration_sum = cub::WarpReduce<T>(*temp_storage).Sum(v1[dataSeq] * v2[dataSeq]);
                if (get_lane_id() == 0) {
                    output[get_warp_id()] += iteration_sum;
                }
            };
            warp_for(size, product, &temp_storage, v1, v2, output);
            return output[warpid];
        }

        template <typename launch_t, typename T, typename WarpStorage=typename cub::WarpReduce<T>::TempStorage>
        __device__ __forceinline__ T asum(const T *in, const uint32_t size, WarpStorage &temp_storage) {
            enum { warps = warps_per_block<launch_t>() };
            __shared__ T output[warps];
            const std::uint8_t warpid = get_warp_id();
            output[warpid] = 0;

            auto sum = []__device__(const uint32_t dataSeq, WarpStorage &temp_storage, const T *in, T *output) {
                const T iteration_sum = cub::WarpReduce<T>(temp_storage).Sum(in[dataSeq]);
                if (get_lane_id() == 0) {
                    output[get_warp_id()] += iteration_sum;
                }
            };
            warp_for(size, sum, temp_storage, in, output);
            return output[warpid];
        }

        template <typename T>
        __device__ __forceinline__ void transposed_scal_accum(const T *in, const T a, const uint32_t in_cols,
                                                              const uint32_t size, T * const out) {

            auto cmult_T = []__device__(const uint32_t dataSeq,
                                        const T a, const uint32_t in_cols,
                                        const T *input, T * const output) {
                output[dataSeq] = cuda::fma(input[dataSeq * in_cols], a, output[dataSeq]);
            };
            warp_for(size, cmult_T, a, in_cols, in, out);
        }

        template <typename T>
        __device__ __forceinline__ void transpose_ewisemult(const T *a, const T *b, const uint32_t b_cols,
                                                            const uint32_t size, T *out) {
            auto ewiseT = []__device__(const uint32_t dataSeq, const T *input1, const T *input2,
                                       const uint32_t i2cols, T *output) {
                output[dataSeq] = input1[dataSeq] * input2[dataSeq * i2cols];
            };
            warp_for(size, ewiseT, a, b, b_cols, out);
        }

        template <typename launch_t, typename T, typename WarpStorage=typename cub::WarpReduce<T>::TempStorage>
        __device__ __forceinline__ T max(const T *a, const uint32_t size, WarpStorage &temp_storage)
        {
            enum { warps = warps_per_block<launch_t>() };
            __shared__ T shm[warps];
            const std::uint8_t warpid = get_warp_id();

            shm[warpid] = static_cast<T>(-1.0e318);

            auto max = []__device__(const uint32_t dataSeq, WarpStorage &temp_storage, const T *input, T *output) {
                auto iteration_max = impl::max<cub::WarpReduce<T> >(input[dataSeq], temp_storage);

                if (get_lane_id() == 0 && output[get_warp_id()] < iteration_max) {
                    output[get_warp_id()] = iteration_max;
                }
            };
            warp_for(size, max, temp_storage, a, shm);
            return shm[warpid];
        }

        template <typename T>
        __device__ __forceinline__ void scal(const T *in, const T a, const uint32_t size, T *out) {
            warp_for(size, impl::cmult<T>, a, in, out);
        }

        template <typename T>
        __device__ __forceinline__ void add(const T *in, const T a, const uint32_t size, T *out) {
            warp_for(size, impl::cadd<T>, a, in, out);
        }

        template <typename T>
        __device__ __forceinline__ void copy(const T *src, const uint32_t size, T *dst) {
            warp_for(size, impl::copy<T>, src, dst);
        }

        template <typename T>
        __device__ __forceinline__ void zero(T *src, const uint32_t size) {
            warp_for(size, impl::clear<T>, src);
        }

        template <typename T>
        __device__ __forceinline__ void axpy(const T a, const T *x, const T *y, const uint32_t size, T *out) {
            warp_for(size, impl::fma<T>, a, x, y, out);
        }

        template <typename T>
        __device__ __forceinline__ void ewisemult(const T *a, const T *b, const uint32_t size, T *out) {
            warp_for(size, impl::ewisemult<T>, a, b, out);
        }

        template <typename T>
        __device__ __forceinline__ void ewiseadd(const T *a, const T *b, const uint32_t size, T *out) {
            warp_for(size, impl::ewiseadd<T>, a, b, out);
        }

        template <typename T>
        __device__ __forceinline__ void ewisefma(const T *x, const T *y, const T z, const uint32_t size, T *out) {
            warp_for(size, impl::ewisefma<T>, x, y, z, out);
        }
    } //end namespace warp
}