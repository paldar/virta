#pragma once

#include <thrust/device_vector.h>
#include <thrust/iterator/discard_iterator.h>
#include <iostream>
#include <vector>
#include <utility>
#include <random>
#include <curand.h>
#include "primitives.inl"

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    std::cerr << "cuRAND Error at " << __FILE__ << ":" << __LINE__ << std::endl;}} while(0)

namespace np
{
namespace detail
{
template <typename OutputIterator, typename T>
void normalize_vector(OutputIterator output, const uint32_t size, const T hi, const T lo)
{
    if (hi != 1.0 || lo != 0.0) {
        auto normalizer = [=] __device__ (const T val) {return (val * (hi - lo)) + lo;};
        thrust::transform(output, output + size, output, normalizer);
    }
}
} // end namespace detail

template <typename OutputIterator, typename T>
typename std::enable_if<std::is_same<T, float>::value, void>::type
populate_uniform(OutputIterator output, const uint32_t size, const T hi=1.0,
                 const T lo=0.0)
{
    curandGenerator_t gen;
    std::random_device rd;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));
    CURAND_CALL(curandGenerateUniform(gen, thrust::raw_pointer_cast(&(*output)), size));
    detail::normalize_vector(output, size, hi, lo);
    CURAND_CALL(curandDestroyGenerator(gen));
}

template <typename OutputIterator, typename T>
typename std::enable_if<std::is_same<T, double>::value, void>::type
populate_uniform(OutputIterator output, const uint32_t size, const T hi=1.0, const T lo=0.0)
{
    curandGenerator_t gen;
    std::random_device rd;
    CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rd()));
    CURAND_CALL(curandGenerateUniformDouble(gen, thrust::raw_pointer_cast(&(*output)), size));
    detail::normalize_vector(output, size, hi, lo);
    CURAND_CALL(curandDestroyGenerator(gen));
}

// using std::gamma_distribution for now for simplicity
// TODO: use cuda version in the future if too slow
template <typename OutputIterator>
void populate_gamma(OutputIterator output, const uint32_t size, const float alpha=0.0,
                    const float beta=1.0)
{
    std::vector <typename std::iterator_traits<OutputIterator>::value_type> temp(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::gamma_distribution<> dis(alpha, beta);

    for (uint32_t n = 0; n < size; ++n) {
        temp[n] = dis(gen);
    }
    thrust::copy_n(temp.begin(), size, output);
}

template <typename IndexT, typename ValueT,
          typename InputIterator1=typename std::vector<IndexT>::iterator,
          typename InputIterator2=typename std::vector<ValueT>::iterator>
struct coo_matrix
{
    typedef ValueT value_type;
    typedef IndexT index_type;
    uint32_t n_rows;
    uint32_t n_cols;
    uint32_t nnz;

    thrust::device_vector<IndexT> rows;
    thrust::device_vector<IndexT> cols;
    thrust::device_vector<ValueT> vals;

    coo_matrix() = delete;

    coo_matrix(uint32_t n_rows, uint32_t n_cols,
               InputIterator1 rows_begin, InputIterator1 rows_end,
               InputIterator1 cols_begin, InputIterator2 vals_begin)
        : n_rows(n_rows), n_cols(n_cols), nnz(thrust::distance(rows_begin, rows_end)),
          rows(rows_begin, rows_end), cols(cols_begin, cols_begin + thrust::distance(rows_begin, rows_end)),
          vals(vals_begin, vals_begin + thrust::distance(rows_begin, rows_end)) {}
};

template <typename IndexT, typename ValueT,
          typename InputIterator1=typename std::vector<IndexT>::iterator,
          typename InputIterator2=typename std::vector<ValueT>::iterator>
class csr_matrix
{
public:
    typedef ValueT value_type;
    typedef IndexT index_type;
    const uint32_t n_rows;
    const uint32_t n_cols;
    const uint32_t nnz;

    thrust::device_vector<IndexT> ptrs;
    thrust::device_vector<IndexT> inds;
    thrust::device_vector<ValueT> vals;

    csr_matrix() = delete;

    csr_matrix(uint32_t n_rows, uint32_t n_cols,
               InputIterator1 ptrs_begin, InputIterator1 ptrs_end,
               InputIterator1 inds_begin, InputIterator1 inds_end,
               InputIterator2 vals_begin)
        : n_rows(n_rows), n_cols(n_cols), nnz(thrust::distance(inds_begin, inds_end)),
          ptrs(ptrs_begin, ptrs_end), inds(inds_begin, inds_end), vals(vals_begin, vals_begin + thrust::distance(inds_begin, inds_end))
    {}

    csr_matrix(const coo_matrix<IndexT, ValueT, InputIterator1, InputIterator2> &rhs)
        : n_rows(rhs.n_rows), n_cols(rhs.n_cols), ptrs(rhs.n_rows + 1), inds(rhs.cols), vals(rhs.vals), nnz(rhs.nnz) {
        thrust::reduce_by_key(rhs.rows.begin(), rhs.rows.end(), thrust::make_constant_iterator(1),
                              thrust::make_discard_iterator(), ptrs.begin());
    }
};

// TODO: add generic transpose function for matrices.

template <typename ValueT>
class dense_matrix
{
public:
    typedef ValueT value_type;
    const uint32_t n_rows;
    const uint32_t n_cols;
    thrust::device_vector<ValueT> vals;

    dense_matrix() = delete;

    dense_matrix(const thrust::host_vector<ValueT> &rval, const uint32_t n_rows, const uint32_t n_cols)
            : vals(rval), n_rows(n_rows), n_cols(n_cols) {}

    dense_matrix(const thrust::device_vector<ValueT> &rval, const uint32_t n_rows, const uint32_t n_cols)
            : vals(rval), n_rows(n_rows), n_cols(n_cols) {}

    dense_matrix(const std::vector<ValueT> &rval, const uint32_t n_rows, const uint32_t n_cols)
            : vals(rval), n_rows(n_rows), n_cols(n_cols) {}

    dense_matrix(const uint32_t n_rows, const uint32_t n_cols)
            : n_rows(n_rows), n_cols(n_cols),
              vals(n_rows * n_cols) {}

    // copy construction
    dense_matrix(const dense_matrix<ValueT> & rhs)
            : n_rows(rhs.n_rows), n_cols(rhs.n_cols),
              vals(rhs.vals) {}

    __host__
    std::pair<uint32_t, uint32_t> shape() {
        return std::make_pair(n_rows, n_cols);
    }

    __host__
    thrust::device_vector<ValueT> & get_vector_ref() {
        return this->vals;
    }

    __host__
    ValueT * data() {
        return cuda::to_ptr(vals);
    }

    __host__
    uint32_t size() {
        return vals.size();
    }

    __host__
    std::vector<ValueT> to_host_vector() {
        std::vector<ValueT> temp;
        temp.reserve(vals.size());
        thrust::copy(vals.cbegin(), vals.cend(), temp.begin());
        return temp;
    }
};
} //end namespace np
