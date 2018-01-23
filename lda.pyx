from libcpp.vector cimport vector
from libcpp.string cimport string
from libcpp cimport bool
from libc.stdint cimport uint32_t
from cython.operator cimport dereference as deref
import numpy as np
import cython
import sklearn.decomposition
import logging
import cuda
logger = logging.getLogger(__name__)


cdef extern from "lda/lda.h" namespace "lda":
    cdef cppclass LDA[T]:
        LDA(const uint32_t ncomp, const uint32_t niter,
            const uint32_t ndocs, const uint32_t nfeat, const T alpha,
            const T eta) except +

        void fit[MatrixT] (const MatrixT &X)
        void print_n_top_words(vector[string] &feature_map, const uint32_t n)
        vector[T] get_gamma()
        vector[T] get_lambda()


cdef extern from "np.h" namespace "np":
    cdef cppclass csr_matrix[I, T]:
        const uint32_t nnz, n_rows, n_cols
        csr_matrix(uint32_t n_rows, uint32_t n_cols,
                   vector[I].iterator ptrs_begin, vector[I].iterator ptrs_end,
                   vector[I].iterator inds_begin, vector[I].iterator inds_end,
                   vector[T].iterator vals_begin) except +

    cdef cppclass dense_matrix[T]:
        const uint32_t n_rows, n_cols
        dense_matrix(vector[T] val, uint32_t rows, uint32_t cols) except +
        vector[T] to_host_vector()


cdef class FloatCsrMatrix:
    cdef csr_matrix[uint32_t, float] *mat

    def __cinit__(self, scipy_csr_matrix):
        rows, cols = scipy_csr_matrix.shape
        cdef vector[uint32_t] cindptr = scipy_csr_matrix.indptr.astype(np.uint32).tolist()
        cdef vector[uint32_t] cindices = scipy_csr_matrix.indices.astype(np.uint32).tolist()
        # TODO: default to float, deal with double later
        cdef vector[float] cdata = scipy_csr_matrix.data.astype(np.float32).tolist()
        self.mat = new csr_matrix[uint32_t, float](
            rows, cols, cindptr.begin(), cindptr.end(),
            cindices.begin(), cindices.end(),
            cdata.begin())

    def get_nnz(self):
        return self.mat.nnz

    def get_shape(self):
        return (self.mat.n_rows, self.mat.n_cols)

    cdef inline csr_matrix[uint32_t, float] * get_ptr(self):
        return <csr_matrix[uint32_t, float]*> self.mat

    def __dealloc__(self):
        del self.mat


cdef class DenseMatrix:
    cdef dense_matrix[float] *mat
    def __cinit__(self, np_matrix):
        rows, cols = np_matrix.shape
        cdef vector[float] cdata = np_matrix.astype(np.float32).flatten().tolist()
        self.mat = new dense_matrix[float](cdata, rows, cols)

    def get_shape(self):
        return (self.mat.n_rows, self.mat.n_cols)

    def to_numpy_array(self):
        return np.array(self.mat.to_host_vector()).reshape(self.get_shape())

    cdef inline dense_matrix[float] * get_ptr(self):
        return <dense_matrix[float]*> self.mat

    def __dealloc__(self):
        del self.mat


def run_lda(W, wordmap, ncomp=60, niter=10):

    cdef vector[string] lda_word_map = [word.encode('utf-8') for word in wordmap]
    cdef uint32_t ndocs, nfeat
    cdef float alpha, eta

    ndocs, nfeat = W.shape
    alpha = 0.1
    eta = 0.01

    cdef LDA[float] *lda = new LDA[float](ncomp, niter, ndocs, nfeat, alpha, eta)
    content = FloatCsrMatrix(W)

    lda.fit(deref(content.get_ptr()))
    lda.print_n_top_words(lda_word_map, 10)

    item_factors = np.array(lda.get_gamma()).reshape((ndocs, ncomp))

    del lda
    del content

    # normalize item factors
    return item_factors / np.sum(item_factors, axis=1)[:, np.newaxis]
