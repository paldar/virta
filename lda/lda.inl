#pragma once

#include "../psi.inl"
#include "../np.h"
#include "../util.h"
#include "../primitives.inl"
#include "../warp_functors.inl"


namespace lda {
    template <typename T>
    constexpr const __device__ __forceinline__
    typename std::enable_if<std::is_same<T, float>::value, float>::type EPS() {
        return FLT_EPSILON;
    }

    template <typename T>
    constexpr const __device__ __forceinline__
    typename std::enable_if<std::is_same<T, double>::value, double>::type EPS() {
        return DBL_EPSILON;
    }

    template <typename launch_t, typename T>
    __device__ __forceinline__
    void dirichlet_expectation_1d(const T *source, const uint32_t size, T *out,
                                  typename cub::WarpReduce<T>::TempStorage &ts) {
        const T sum = psi(cuda::warp::asum<launch_t>(source, size, ts));


        auto op = []__device__(const uint32_t dataSeq,
                               const T *input, const T sum, T *output) {
            output[dataSeq] = exp(psi(input[dataSeq]) - sum);
        };
        cuda::warp_for(size, op, source, sum, out);
    }

    template <typename launch_t, typename T>
    __device__ __forceinline__ T mean_change(const T *arr1, const T *arr2, const uint32_t size,
                                             typename cub::WarpReduce<T>::TempStorage &temp_storage)
    {
        enum { warps_per_block = cuda::warps_per_block<launch_t>() };
        __shared__ T total[warps_per_block];

        total[cuda::get_warp_id()] = 0;
        auto accum = []__device__(const uint32_t dataSeq,
                                  typename cub::WarpReduce<T>::TempStorage &temp_storage,
                                  const T *a1, const T *a2, T *total) {
            auto iteration_sum = cub::WarpReduce<T>(temp_storage).Sum(fabs(a1[dataSeq] - a2[dataSeq]));
            if (cuda::get_lane_id() == 0) {
                total[cuda::get_warp_id()] += iteration_sum;
            }
        };
        cuda::warp_for(size, accum, temp_storage, arr1, arr2, total);
        return total[cuda::get_warp_id()] / size;
    }

    template <typename launch_t, typename T, typename I, typename CountT>
    __global__ void bound_calc(const I *indptr, const I *rowptr, const CountT *counts,
                               const T *theta, const T *beta, const uint32_t ndocs,
                               const uint32_t ncomp, const uint32_t nfeat, T * const score)
    {
        using namespace cuda::warp;
        using cuda::ptr_at;
        enum { warps_per_block = cuda::warps_per_block<launch_t>() };
        typedef cub::BlockReduce<T, warps_per_block> BlockReduce;
        typedef cub::WarpReduce<T> WarpReduce;

        extern __shared__ T shared_memory[];
        __shared__ typename WarpReduce::TempStorage temp_storage[warps_per_block];
        __shared__ typename BlockReduce::TempStorage block_reduce_ts;
        __shared__ std::uint32_t work[warps_per_block];
        __shared__ T bcast[warps_per_block];

        T * const __restrict__ shm = shared_memory;

        const std::uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
        const std::uint8_t warpid = cuda::get_warp_id();
        bool has_work = false;

        if (tid < ndocs) has_work = true;

        #pragma unroll
        while (__any(has_work)) {
            if (has_work) work[warpid] = tid;
            if (work[warpid] == tid) has_work = false;

            const uint32_t row = work[warpid];
            const uint32_t row_length = indptr[row + 1] - indptr[row];
            const I * const __restrict__ ids = rowptr + indptr[row];
            const CountT * const __restrict__ cts = counts + indptr[row];

            #pragma unroll
            for (uint32_t id = 0; id < row_length; ++id) {
                transpose_ewisemult(ptr_at(theta, row, 0, ncomp), beta + ids[id], nfeat, ncomp,
                                    ptr_at(shm, warpid, 0, ncomp));
                const T tmax = max<launch_t>(ptr_at(shm, warpid, 0, ncomp), ncomp, temp_storage[warpid]);

                bcast[warpid] = 0;
                auto calc_localphi = []__device__(const uint32_t dataSeq,
                                                  typename WarpReduce::TempStorage &ts,
                                                  const float * const x, const float max, float * const output) {
                    auto iteration_sum = WarpReduce(ts).Sum(exp(x[dataSeq] - max));

                    if (cuda::get_lane_id() == 0) {
                        output[cuda::get_warp_id()] += iteration_sum;
                    }
                };

                cuda::warp_for(ncomp, calc_localphi, temp_storage[warpid], ptr_at(shm, warpid, 0, ncomp),
                               tmax, bcast);
                bcast[warpid] = (log(bcast[warpid]) + tmax) * cts[id];

                const T block_sum = BlockReduce(block_reduce_ts).Sum(bcast[warpid]);
                if (threadIdx.x == 0) {
                    atomicAdd(score, block_sum);
                }
            }
        }
    }


    template <typename launch_t, typename T, typename I, typename CountT>
    __global__ void gamma_loop(const I *indptr, const I *rowptr, const CountT *counts,
                               const T *E_theta, const T *E_beta,
                               const uint32_t ncomp, const uint32_t nfeat, const uint32_t ndocs,
                               const T alpha,
                               T *gamma, T *sstats, T *temp_gl)
    {
        using namespace cuda::warp;
        using cuda::ptr_at;
        typedef cub::WarpReduce<T> WarpReduce;
        enum { warps_per_block = cuda::warps_per_block<launch_t>() };

        extern __shared__ T gamma_block[];
        __shared__ typename WarpReduce::TempStorage temp_storage[warps_per_block];
        __shared__ std::uint32_t row[warps_per_block];

        __shared__ T *shm;
        shm = gamma_block + (ncomp * warps_per_block);

        //we dont use temp.
        T * const __restrict__ temp = temp_gl + (warps_per_block * ncomp * 3 * blockIdx.x);
        T * const __restrict__ last_gamma_block = temp + (ncomp * warps_per_block);
        T * const __restrict__ E_theta_block = last_gamma_block + (ncomp * warps_per_block);

        const uint32_t tid = threadIdx.x + blockDim.x * blockIdx.x;
        const std::uint8_t warpid = cuda::get_warp_id();
        bool has_work = false;

        if (tid < ndocs) has_work = true;

        while(__any(has_work)) {
            if (has_work) row[warpid] = tid;
            if (row[warpid] == tid) has_work = false;

            const uint32_t row_length = indptr[row[warpid] + 1] - indptr[row[warpid]];
            const I * const __restrict__ ids = rowptr + indptr[row[warpid]];
            const CountT * const __restrict__ cts = counts + indptr[row[warpid]];

            T * const __restrict__ E_theta_d = ptr_at(E_theta_block, warpid, 0, ncomp);
            T * const __restrict__ gamma_d = ptr_at(gamma_block, warpid, 0, ncomp);
            T * const __restrict__ last_gamma_d = ptr_at(last_gamma_block, warpid, 0, ncomp);
            T * const __restrict__ backup_E_theta_d = ptr_at(shm, warpid, 0, ncomp);


            //gammad = gamma[d, :]
            copy(ptr_at(gamma, row[warpid], 0, ncomp), ncomp, gamma_d);

            //E_theta_d = E_theta[d, :]
            copy(ptr_at(E_theta, row[warpid], 0, ncomp), ncomp, E_theta_d);

            #pragma unroll
            for (int iter = 0; iter < 100; ++iter) {
                copy(gamma_d, ncomp, last_gamma_d);
                zero(ptr_at(shm, warpid, 0, ncomp), ncomp);

                #pragma unroll
                for (uint32_t non_zero_col = 0; non_zero_col < row_length; ++non_zero_col) {

                    const T local_phi = transposed_dot<launch_t>(E_theta_d, E_beta + ids[non_zero_col],
                                                                        nfeat, ncomp, temp_storage[warpid]) + EPS<T>();

                    // np.dot(cts / phi_norm, E_beta_d.T) => shm
                    transposed_scal_accum(E_beta + ids[non_zero_col],
                                          static_cast<T>(cts[non_zero_col]) / local_phi, nfeat,
                                          ncomp, ptr_at(shm, warpid, 0, ncomp));
                }
                // alpha + (E_theta_d * shm[warpid]) => gamma_d
                ewisefma(ptr_at(shm, warpid, 0, ncomp), E_theta_d, alpha, ncomp, gamma_d);
                // backup E_theta_d. gets overwritten if not terminating.
                copy(E_theta_d, ncomp, backup_E_theta_d);

                dirichlet_expectation_1d<launch_t>(gamma_d, ncomp, E_theta_d, temp_storage[warpid]);

                if (mean_change<launch_t>(gamma_d, last_gamma_d, ncomp, temp_storage[warpid]) < 0.00001) {
                    break;
                }
            }
            // gamma[row, :] = gammad
            copy(gamma_d, ncomp, ptr_at(gamma, row[warpid], 0, ncomp));

            //sstats
            #pragma unroll
            for (uint32_t non_zero_col = 0; non_zero_col < row_length; ++non_zero_col) {
                const T local_phi = transposed_dot<launch_t>(backup_E_theta_d, E_beta + ids[non_zero_col],
                                                             nfeat, ncomp, temp_storage[warpid]) + EPS<T>();

                auto cmult_T = []__device__(const uint32_t dataSeq, const T a, const uint32_t out_cols,
                                            const T *input, T *output) {
                    output[dataSeq * out_cols] = cuda::fma(input[dataSeq], a, output[dataSeq * out_cols]);
                };
                cuda::warp_for(ncomp, cmult_T, static_cast<T>(cts[non_zero_col]) / local_phi,
                               nfeat, E_theta_d, sstats + ids[non_zero_col]);
            }
        }
    }
}
