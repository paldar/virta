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
#include "arithmetic.inl"
#include "launch_info.inl"
#include "primitives.inl"
#include "warp_functors.inl"
#include "block_functors.inl"


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        if (abort) {
            throw std::runtime_error("Error after kernel launch: \""
                    + std::string(cudaGetErrorString(code))
                    + "\" at"
                    + std::string(file) + ":" + std::to_string(line)
                    + "\nCode: " + std::to_string(code));
        } else {
            std::cerr << "GPUassert: " << cudaGetErrorString(code) << " "
                      << file << " " << line << std::endl;
        }
    }
}

#define check_error() { gpuAssert(cudaDeviceSynchronize(), __FILE__, __LINE__); \
                        gpuAssert(cudaPeekAtLastError(), __FILE__, __LINE__); }

#define checkCudaErrors(val)   check ( (val), #val, __FILE__, __LINE__ )
template< typename T >
void check(T result, char const *const func, const char *const file, int const line)
{
    if (result)
    {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",
                file, line, static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        cudaDeviceReset();
        // Make sure we call CUDA Device Reset before exiting
        exit(EXIT_FAILURE);
    }
}

namespace cuda
{

#if __CUDA_ARCH__ < 600
    //double atomic add support
    template <typename T>
    __device__ __forceinline__
    typename std::enable_if<std::is_same<T, double>::value, double>::type
    atomicAdd(T* address, T val)
    {
        unsigned long long int* address_as_ull =
                (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                                 __longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }

    template <typename T>
    __device__ __forceinline__
    typename std::enable_if<!std::is_same<T, double>::value, T>::type
    atomicAdd(T* address, T val)
    {
        return ::atomicAdd(address, val);
    }
#endif

    // execute when #cols <= 1024
    // reduce(sum) by row: collapse into one column
    template <typename launch_t, typename T>
    __global__ void row_reduce(T * source, const uint32_t rows, const uint32_t cols,
                               const T param, T * output)
    {
        typedef cub::WarpReduce<T> WarpReduce;
        enum { warps = warps_per_block<launch_t>() };
        __shared__ typename WarpReduce::TempStorage temp_storage[warps];
        __shared__ std::uint32_t work[warps];

        const std::uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
        bool has_work = false;

        const std::uint8_t warpid = get_warp_id();

        if (tid < rows) has_work = true;

        while(__any(has_work)) {
            if (has_work) work[warpid] = tid;
            if (work[warpid] == tid) has_work = false;
            const std::uint32_t row = work[warpid];

            const T sum = cuda::warp::asum<launch_t>(cuda::ptr_at(source, row, 0, cols), cols, temp_storage[warpid]);
            output[row] = (sum + param);
        }
    }

    template <typename T>
    struct broadcast_sum
    {
        const T * input;
        const T * params;
        const uint32_t cols;
        //input: number of columns
        //params: number of rows
        broadcast_sum(const T * input, const uint32_t cols, const T * params)
                : input(input), cols(cols), params(params) {}

        template <typename IndexT>
        __host__ __device__
        T operator()(const IndexT &idx) {
            auto row = idx / cols;
            auto col = idx % cols;
            return input[col] + params[row];
        }
    };

    template <typename launch_t, typename T, typename IndexT>
    __global__ void indexed_dot(T * a1, T * a2, IndexT * uu, IndexT *ii,
                                const uint32_t nz, const uint32_t ncomp, T * output)
    {
        extern __shared__ T shm[];
        typedef cub::WarpReduce<T> WarpReduce;
        enum { warps = warps_per_block<launch_t>() };
        __shared__ typename WarpReduce::TempStorage temp_storage[warps];
        __shared__ std::uint32_t work[warps];

        const std::uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
        bool has_work = false;

        const std::uint8_t laneid = get_lane_id();
        const std::uint8_t warpid = get_warp_id();

        if (tid < nz) {
            has_work = true;
        }

        #pragma unroll
        while (__any(has_work)) {
            //broadcast workid
            if (has_work) {
                work[warpid] = tid;
            }
            if (work[warpid] == tid) {
                has_work = false;
            }
            std::uint32_t chunk_idx = work[warpid];
            auto u = uu[chunk_idx] * ncomp;
            auto i = ii[chunk_idx] * ncomp;
            T reduce_result = 0.0f;

            for(std::uint8_t start = 0; start < ncomp; start += mgpu::warp_size) {
                auto valid_items = mgpu::min(static_cast<uint32_t>(mgpu::warp_size), ncomp - start);
                if (laneid < valid_items) {
                    auto seq = start + laneid;
                    auto seq_in_shm = seq + warpid * ncomp;
                    shm[seq_in_shm] = a1[u + seq] * a2[i + seq];
                    reduce_result += WarpReduce(temp_storage[warpid]).Reduce(shm[seq_in_shm], cub::Sum(), valid_items);
                }
            }
            if (laneid == 0) {
                *(output + chunk_idx) = reduce_result;
            }
        }
    }
} //end namespace cuda
