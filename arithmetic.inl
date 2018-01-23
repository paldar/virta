#pragma once
#include <thrust/functional.h>
#include <host_defines.h>
#include <cub/thread/thread_operators.cuh>
#include <math.h>

//arithmetic functors
namespace cuda
{
    template <typename T>
    struct plus : thrust::binary_function<T, T, T>{
        const T c;
        plus(const T c) : c(c) {}

        __host__ __device__ __forceinline__
        T operator()(const T &lhs, const T &rhs) {
            return lhs + rhs + c;
        }
    };

    template <typename T>
    struct logarithm : thrust::unary_function<T, T>{
        __host__ __device__ __forceinline__
        T operator()(const T &v) {
            return log(v);
        }
    };

    template <typename T>
    struct exponential : thrust::unary_function<T, T>{
        __host__ __device__ __forceinline__
        T operator()(const T &v) {
            return exp(v);
        }
    };

    template <typename T>
    struct incr : thrust::unary_function<T, T>{
        const T c;
        incr(const T c) : c(c) {}

        __host__ __device__ __forceinline__
        T operator()(const T &lhs) {
            return lhs + c;
        }
    };

    // x * y + z
    template <typename T>
    __device__ __forceinline__
    typename std::enable_if<std::is_same<T, double>::value, double>::type fma(const T x, const T y, const T z) {
        return fma(x, y, z);
    }

    template <typename T>
    __device__ __forceinline__
    typename std::enable_if<std::is_same<T, float>::value, float>::type fma(const T x, const T y, const T z) {
        return fmaf(x, y, z);
    }

    template <typename T>
    __device__ __forceinline__
    typename std::enable_if<std::is_same<T, double>::value, double>::type abs(const T x) {
        return fabs(x);
    }

    template <typename T>
    __device__ __forceinline__
    typename std::enable_if<std::is_same<T, float>::value, float>::type abs(const T x) {
        return fabsf(x);
    }

    namespace impl {
        //fused multiply-add with a constant (saxpy)
        template <typename T>
        __device__ __forceinline__ void
        fma(const uint32_t dataSeq, const T a, const T *x, const T *y, T *output) {
            output[dataSeq] = cuda::fma(x[dataSeq], a, y[dataSeq]);
        }

        //constant multiplication
        template <typename T>
        __device__ __forceinline__ void
        cmult(const uint32_t dataSeq, const T a, const T *input, T *output) {
            output[dataSeq] = input[dataSeq] * a;
        }

        //const addition
        template <typename T>
        __device__ __forceinline__ void
        cadd(const uint32_t dataSeq, const T a, const T *input, T *output) {
            output[dataSeq] = input[dataSeq] + a;
        }

        template <typename T>
        __device__ __forceinline__ void
        copy(const uint32_t dataSeq, const T *src, T *dst) {
            dst[dataSeq] = src[dataSeq];
        }

        template <typename T>
        __device__ __forceinline__ void
        ewisemult(const uint32_t dataSeq, const T *input1, const T *input2, T *output) {
            output[dataSeq] = input1[dataSeq] * input2[dataSeq];
        }

        template <typename T>
        __device__ __forceinline__ void
        ewiseadd(const uint32_t dataSeq, const T *input1, const T *input2,
                 T *output) {
            output[dataSeq] = input1[dataSeq] + input2[dataSeq];
        }

        template <typename T>
        __device__ __forceinline__ void
        ewisefma(const uint32_t dataSeq, const T *input1, const T *input2, const T a, T *output) {
            output[dataSeq] = cuda::fma(input1[dataSeq], input2[dataSeq], a);
        }

        template <typename T>
        __device__ __forceinline__ void
        clear(const uint32_t dataSeq, T *data) {
            data[dataSeq] = static_cast<T>(0);
        }

        template <typename ReduceMethod, typename T, typename TempStorage=typename ReduceMethod::TempStorage>
        __device__ __forceinline__
        constexpr T max(const T value, TempStorage & temp_storage) {
            return ReduceMethod(temp_storage).Reduce(value, cub::Max());
        }
    } //end namespace impl
}
