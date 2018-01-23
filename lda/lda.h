#pragma once
#include <cstdint>
#include <algorithm>
#include <iomanip>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <cub/device/device_segmented_radix_sort.cuh>
#include "../psi.inl"
#include "../np.h"
#include "../util.h"
#include "lda.inl"


namespace lda {
    namespace t = thrust;
    using cuda::ptr_at;
    using namespace virta;

    namespace aux {
        template <typename ValueT, typename IndexT>
        struct dirichlet_2d_functor : public t::binary_function<ValueT, IndexT, ValueT> {
            const ValueT * psi_row_total;
            const IndexT cols;
            const bool calc_exp;
            dirichlet_2d_functor(const ValueT * psi_row, const IndexT cols, const bool calc_exp)
                    : psi_row_total(psi_row), cols(cols), calc_exp(calc_exp) {}
            __host__ __device__ __forceinline__
            ValueT operator()(const IndexT i, const ValueT value) {
                const auto row = i / cols;
                if (calc_exp) {
                    return exp(psi(value) - psi(psi_row_total[row]));
                } else {
                    return psi(value) - psi(psi_row_total[row]);
                }
            }
        };
    }

    template <typename VectorT>
    void dirichlet_expectation(const VectorT &alpha, const uint32_t n_cols, VectorT &out, VectorT &temp, const bool calc_exp = true)
    {
        typedef typename VectorT::value_type T;

        row_reduce_with_param(alpha, n_cols, 0, temp);

        aux::dirichlet_2d_functor <T, uint32_t> op(cuda::to_ptr(temp), n_cols, calc_exp);
        t::transform(t::make_counting_iterator(0U),
                     t::make_counting_iterator((uint32_t)alpha.size()),
                     alpha.begin(),
                     out.begin(), op);
    }

    template <typename T, typename VectorT=t::device_vector<T> >
    class LDA {
    public:
        LDA() = delete;

        LDA(const uint32_t ncomp, const uint32_t niter,
            const uint32_t ndocs, const uint32_t nfeat, const T alpha=0.1,
            const T eta=0.01)
                : ncomp(ncomp), niter(niter), ndocs(ndocs),
                  nfeat(nfeat), alpha(alpha), eta(eta), updatect(0),
                  lambda(ncomp * nfeat, 0), E_beta(ncomp * nfeat, 0),
                  beta(ncomp * nfeat, 0), gamma(ndocs * ncomp, 0), gamma_scratch(ndocs * ncomp, 0),
                  E_theta(ndocs * ncomp, 0), theta(ndocs * ncomp, 0), lambda_scratch(ncomp * nfeat, 0),
                  sstats(ncomp * nfeat, 0), temp(nullptr)
        {
            gpuErrchk(cudaMalloc(&score, sizeof(T)));
        }

        ~LDA() {
            if (temp != nullptr) {
                gpuErrchk(cudaFree(temp));
            }
            gpuErrchk(cudaFree(score));
        }

        template <typename MatrixT>
        void fit(const MatrixT &X) {
            np::populate_gamma(lambda.begin(), lambda.size(), 100, 1.0f / 100.0f);
            dirichlet_expectation(lambda, nfeat, beta, lambda_scratch, false);
            t::transform(beta.begin(), beta.end(), E_beta.begin(), cuda::exponential<T>());


            mgpu::standard_context_t context(false);
            std::cout << util::time_str << "Starting LDA. Components :" << ncomp << ", Iterations: " << niter << std::endl;
            for (uint32_t i = 0; i < niter; ++i) {
                context.timer_begin();
                const T score = update_lambda(X);
                std::cout << virta::util::time_str
                          << "Iteration: " << std::setfill('0') << std::setw(3) << i + 1
                          << " of max_iter: " << niter
                          << ". iteration time: " << std::setprecision(7) << context.timer_end() << "s "
                          << "score: " << std::setprecision(7) << score << std::endl;
            }
        }

        void print_n_top_words(std::vector<std::string> & feature_map, const uint32_t n) {
            const uint32_t comp_size = lambda.size();
            t::device_vector<int> indices(comp_size, 0);

            t::device_vector<T> component_out(comp_size, 0);
            t::device_vector<int> indices_out(comp_size, 0);
            t::device_vector<int> d_offsets(ncomp + 1, 0);

            const auto cols = comp_size / ncomp;

            t::transform(t::make_counting_iterator(0U),
                         t::make_counting_iterator(comp_size),
                         t::make_constant_iterator(cols),
                         indices.begin(),
                         t::modulus<uint32_t>());

            t::transform(t::make_counting_iterator(0U),
                         t::make_counting_iterator(ncomp + 1),
                         t::make_constant_iterator(cols),
                         d_offsets.begin(),
                         t::multiplies<uint32_t>());

            //segmented sort:
            void *d_temp_storage = nullptr;
            size_t temp_storage_bytes = 0;
            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    d_temp_storage, temp_storage_bytes,
                    cuda::to_ptr(lambda), cuda::to_ptr(component_out),
                    cuda::to_ptr(indices), cuda::to_ptr(indices_out),
                    comp_size, ncomp, cuda::to_ptr(d_offsets), cuda::to_ptr(d_offsets) + 1);

            gpuErrchk(cudaMalloc(&d_temp_storage, temp_storage_bytes));

            cub::DeviceSegmentedRadixSort::SortPairsDescending(
                    d_temp_storage, temp_storage_bytes,
                    cuda::to_ptr(lambda), cuda::to_ptr(component_out),
                    cuda::to_ptr(indices), cuda::to_ptr(indices_out),
                    comp_size, ncomp, cuda::to_ptr(d_offsets), cuda::to_ptr(d_offsets) + 1);

            t::host_vector<int> indices_h(indices_out);
            t::host_vector<T> component_vector_h(lambda);

            for (uint32_t i = 0; i < ncomp; ++i) {
                std::cout << "Topic #" << i <<std::endl;

                for (uint32_t j = i * cols; j < i * cols + n; ++j) {
                    std::cout << feature_map[indices_h[j]] << ", ";
                }
                std::cout << std::endl;
            }

            gpuErrchk(cudaFree(d_temp_storage));
        }

        std::vector<T> get_gamma() {
            thrust::host_vector<T> host_gamma(gamma);
            std::vector<T> retval(host_gamma.begin(), host_gamma.end());
            return retval;
        }

        std::vector<T> get_lambda() {
            thrust::host_vector<T> host_lambda(lambda);
            std::vector<T> retval(host_lambda.begin(), host_lambda.end());
            return retval;
        }

        template <typename MatrixT>
        T update_lambda(const MatrixT &X) {
            e_step(X);
            const auto bound = approx_bound(X);
            //m step
            t::transform(sstats.begin(), sstats.end(), lambda.begin(), cuda::incr<T>(eta));
            dirichlet_expectation(lambda, nfeat, beta, lambda_scratch, false);
            t::transform(beta.begin(), beta.end(), E_beta.begin(), cuda::exponential<T>());
            updatect++;
            return bound;
        }

        template <typename MatrixT>
        void e_step(const MatrixT &X) {
            using cuda::to_ptr;

            np::populate_gamma(gamma.begin(), ndocs * ncomp, 100, 1.0f / 100.0f);

            dirichlet_expectation(gamma, ncomp, theta, gamma_scratch, false);
            t::transform(theta.begin(), theta.end(), E_theta.begin(), cuda::exponential<T>());

            sstats.assign(lambda.size(), 0);

            //for each row in matrix X:
            const uint32_t blocks = mgpu::div_up(ndocs, launch_t::sm_ptx::nt);
            constexpr auto warps_per_block = launch_t::sm_ptx::nt / mgpu::warp_size;

            if (temp == nullptr) {
                const uint32_t temp_size = blocks * sizeof(T) * warps_per_block * ncomp * 3;
                gpuErrchk(cudaMalloc(&temp, temp_size));
            }

            const uint32_t shm_size = sizeof(T) * warps_per_block * ncomp * 2;

            gamma_loop<launch_t><<<blocks, launch_t::sm_ptx::nt, shm_size>>>(
                    to_ptr(X.ptrs), to_ptr(X.inds), to_ptr(X.vals),
                    to_ptr(E_theta), to_ptr(E_beta),
                    ncomp, nfeat, ndocs, alpha,
                    to_ptr(gamma), to_ptr(sstats), temp);

            check_error();

            t::transform(sstats.begin(), sstats.end(), E_beta.begin(), sstats.begin(), t::multiplies<T>());
        }

        template <typename MatrixT>
        T approx_bound(const MatrixT &X) {
            using cuda::to_ptr;

            T score_h = 0;

            gpuErrchk(cudaMemset(score, 0, sizeof(T)));

            dirichlet_expectation(gamma, ncomp, theta, gamma_scratch, false);

            constexpr auto warps_per_block = launch_t::sm_ptx::nt / mgpu::warp_size;

            const uint32_t shm_size = sizeof(T) * warps_per_block * ncomp;
            const uint32_t blocks = mgpu::div_up(ndocs, launch_t::sm_ptx::nt);

            bound_calc<launch_t><<<blocks, launch_t::sm_ptx::nt, shm_size>>>(
                to_ptr(X.ptrs), to_ptr(X.inds), to_ptr(X.vals),
                to_ptr(theta), to_ptr(beta), ndocs, ncomp, nfeat, score);

            check_error();

            gpuErrchk(cudaMemcpy(&score_h, score, sizeof(T), cudaMemcpyDeviceToHost));

            //score += n.sum((self._alpha - gamma) * Elogtheta)
            t::transform(gamma.begin(), gamma.end(),
                         t::make_zip_iterator(t::make_tuple(t::make_constant_iterator(alpha), theta.begin())),
                         gamma_scratch.begin(),
                         []__device__(const T v, const t::tuple<T, T> tup) -> T {
                             return (v - t::get<0>(tup)) * t::get<1>(tup);
                         });
            score_h += t::reduce(gamma_scratch.begin(), gamma_scratch.end());

            //score += n.sum(gammaln(gamma) - gammaln(self._alpha)
            t::transform(gamma.begin(), gamma.end(), t::make_constant_iterator(lgamma(alpha)), gamma_scratch.begin(),
                         []__device__(const T v, const T lgalpha) -> T {
                             return lgamma(v) - lgalpha;
                         });

            score_h += t::reduce(gamma_scratch.begin(), gamma_scratch.end());

            //score += sum(gammaln(self._alpha * self._K) - gammaln(n.sum(gamma, 1)))
            const auto lgamma_alpha_ncomp = lgamma(alpha * ncomp);
            row_reduce_with_param(gamma, ncomp, 0, gamma_scratch);
            t::transform(gamma_scratch.begin(), gamma_scratch.begin() + ndocs,
                         t::make_constant_iterator(lgamma_alpha_ncomp), gamma_scratch.begin(),
                         []__device__(const T v, const T lgan) -> T {
                             return lgan - lgamma(v);
                         });
            score_h += t::reduce(gamma_scratch.begin(), gamma_scratch.begin() + ndocs);

            //score = score + n.sum((self._eta - self._lambda) * self._Elogbeta
            t::transform(lambda.begin(), lambda.end(),
                         t::make_zip_iterator(t::make_tuple(t::make_constant_iterator(eta), beta.begin())),
                         lambda_scratch.begin(),
                         []__device__(const T v, const t::tuple<T, T> tup) -> T {
                             return (t::get<0>(tup) - v) * t::get<1>(tup);
                         });
            score_h += t::reduce(lambda_scratch.begin(), lambda_scratch.end());

            //score = score + n.sum(gammaln(self._lambda) - gammaln(self._eta)
            t::transform(lambda.begin(), lambda.end(), t::make_constant_iterator(lgamma(eta)), lambda_scratch.begin(),
                         []__device__(const T v, const T lgeta) -> T {
                             return lgamma(v) - lgeta;
                         });

            score_h += t::reduce(lambda_scratch.begin(), lambda_scratch.end());

            //score = score + n.sum(gammaln(self._eta * self._W) - gammaln(n.sum(self._lambda, 1))
            const auto lgamma_eta_nfeat = lgamma(eta * nfeat);
            row_reduce_with_param(lambda, ncomp, 0, lambda_scratch);
            t::transform(lambda_scratch.begin(), lambda_scratch.begin() + ncomp,
                         t::make_constant_iterator(lgamma_eta_nfeat), lambda_scratch.begin(),
                         []__device__(const T v, const T lgen) -> T {
                             return lgen - lgamma(v);
                         });
            score_h += t::reduce(lambda_scratch.begin(), lambda_scratch.begin() + ncomp);

            return score_h;
        }

        const uint32_t ncomp;
        const uint32_t niter;
        const uint32_t ndocs;
        const uint32_t nfeat;
        const T alpha;
        const T eta;
        T * score;

        VectorT lambda;
        VectorT lambda_scratch;
        VectorT E_beta;
        VectorT beta;

        VectorT sstats;

        VectorT gamma;
        VectorT E_theta;
        VectorT theta;
        VectorT gamma_scratch;

        uint32_t updatect;
        T * temp;
    };
} // end namespace lda
