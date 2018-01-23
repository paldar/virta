#pragma once
#include <stdexcept>
#include <thrust/reduce.h>
#include <cuda_runtime.h>
#include <moderngpu/transform.hxx>
#include <moderngpu/memory.hxx>
#include <moderngpu/meta.hxx>
#include <iomanip>
#include <ctime>
#include "cuda.inl"
#include "psi.inl"


namespace virta {
    using namespace mgpu;
    // standard launchbox arg
    typedef launch_box_t<
        arch_20_cta<128, 8>,    // Big Fermi GF100/GF110  eg GTX 580
        arch_21_cta<128, 4>,    // Lil Fermi GF10x/GF11x  eg GTX 550
        arch_30_cta<256, 4>,    // Lil Kepler GK10x       eg GTX 680
        arch_35_cta<256, 8>,    // Big Kepler GK110+      eg GTX 780 Ti
        arch_37_cta<256, 16>,   // Huge Kepler GK210      eg Tesla K80
        arch_50_cta<256, 8>,    // Lil Maxwell GM10x      eg GTX 750
        arch_52_cta<256, 16>    // Big Maxwell GM20x      eg GTX 980 Ti
    > launch_t;

    template<typename VectorT>
    void row_reduce_with_param(const VectorT &source, const uint32_t cols,
                               const typename VectorT::value_type param, VectorT &output) {
        if (!source.size()) {
            throw std::invalid_argument("source matrix is empty in util.h:row_reduce_with_param");
        }
        const uint32_t rows = source.size() / cols;
        thrust::fill(output.begin(), output.begin() + rows, 0);

        if (cols < 512 && rows > 256) {
            const auto s = cuda::to_ptr(source);
            auto o = cuda::to_ptr(output);
            cuda::row_reduce<launch_t><<< cuda::blocks<launch_t>(rows), cuda::threads<launch_t>() >>>(s, rows, cols, param, o);
            check_error();
        } else {
            for (uint32_t row = 0; row < rows; ++row) {
                output[row] = thrust::reduce(source.begin() + row * cols, source.begin() + (row + 1) * cols, param);
            }
        }
    }

    // "iterator" version
    template<typename InputIter, typename OutputIter>
    void col_reduce(const InputIter source, const uint32_t rows, const uint32_t cols, OutputIter output) {
        typedef typename std::iterator_traits<InputIter>::value_type InputT;
        typedef typename std::iterator_traits<OutputIter>::value_type OutputT;
        //clear out output
        thrust::fill(output, output + cols, 0);
        const auto s = cuda::itop(source);
        auto o = cuda::itop(output);

        standard_context_t context(false);
        auto col_reduce_kernel = []MGPU_DEVICE(const int threadId, const int cta, const InputT * source,
                                               const int rows, const int cols,  OutputT * output)
        {
            typedef typename launch_t::sm_ptx params_t;

            typedef cub::BlockReduce<InputT, params_t::nt> BlockReduce;
            __shared__ typename BlockReduce::TempStorage temp_storage;

            auto tid = threadId + cta * params_t::nt;

            if (tid < rows) {
                for (int i = 0; i < cols; ++i) {
                    OutputT aggregate = BlockReduce(temp_storage)
                            .Reduce(source[tid * cols + i], cub::Sum(),
                                    mgpu::min(static_cast<int>(params_t::nt), rows - params_t::nt * cta));
                    if (threadId == 0) {
                        atomicAdd(output + i, aggregate);
                    }
                }
            }
        };

        cta_launch<launch_t>(col_reduce_kernel, cuda::blocks<launch_t>(rows), context, s, rows, cols, o);
        context.synchronize();
    }

    template<typename VectorT>
    void col_reduce(const VectorT &source, const uint32_t cols, VectorT &output) {
        if (!source.size()) {
            throw std::invalid_argument("source matrix is empty in util.h:col_reduce");
        }
        const uint32_t rows = source.size() / cols;
        return col_reduce(source.begin(), rows, cols, output.begin());
    }

    template<typename ValueVectorT, typename IndexVectorT>
    void indexed_dot(const ValueVectorT &mat1, const ValueVectorT &mat2,
                     const IndexVectorT &u, const IndexVectorT &i, const uint32_t nz,
                     const uint32_t ncomp, ValueVectorT &output) {
        typedef typename ValueVectorT::value_type T;
        const auto a1 = cuda::to_ptr(mat1);
        const auto a2 = cuda::to_ptr(mat2);
        const auto uu = cuda::to_ptr(u);
        const auto ii = cuda::to_ptr(i);
        auto o = cuda::to_ptr(output);

        cuda::indexed_dot<launch_t><<< cuda::blocks<launch_t>(nz), cuda::threads<launch_t>(),
                                       cuda::shm_size<launch_t>(ncomp * sizeof(T)) >>>(a1, a2, uu, ii, nz, ncomp, o);
        check_error();
    }

    bool has_cuda_device() {
        int count = 0;
        const auto retVal = cudaGetDeviceCount(&count);
        return (!retVal && count != 0);
    }

    namespace util {
        std::ostream &time_str(std::ostream &out) {
            std::time_t t = std::time(nullptr);
            std::tm tme = *std::localtime(&t);
            out << std::put_time(&tme, "%F %T %Z\t");
            return out;
        }
    }
}
