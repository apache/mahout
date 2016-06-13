/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

// include necessary system headers
#include <iostream>

#include "viennacl.hpp"
#include "viennacl_private.hpp"

#include "blas3.hpp"

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/prod.hpp"


#ifdef VIENNACL_WITH_CUDA



//
// xGEMV
//

namespace detail
{
  template <typename NumericT>
  ViennaCLStatus ViennaCLCUDAgemm_impl(ViennaCLBackend /*backend*/,
                                       ViennaCLOrder orderA, ViennaCLTranspose transA,
                                       ViennaCLOrder orderB, ViennaCLTranspose transB,
                                       ViennaCLOrder orderC,
                                       ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                       NumericT alpha,
                                       NumericT *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                       NumericT *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                       NumericT beta,
                                       NumericT *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc)
  {
    ViennaCLInt A_size1 = (transA == ViennaCLTrans) ? k : m;
    ViennaCLInt A_size2 = (transA == ViennaCLTrans) ? m : k;

    ViennaCLInt B_size1 = (transB == ViennaCLTrans) ? n : k;
    ViennaCLInt B_size2 = (transB == ViennaCLTrans) ? k : n;

    bool A_row_major = (orderA == ViennaCLRowMajor);
    bool B_row_major = (orderB == ViennaCLRowMajor);
    bool C_row_major = (orderC == ViennaCLRowMajor);

    viennacl::matrix_base<NumericT> matA(A, viennacl::CUDA_MEMORY,
                                         A_size1, offA_row, incA_row, A_row_major ? m : lda,
                                         A_size2, offA_col, incA_col, A_row_major ? lda : k, A_row_major);

    viennacl::matrix_base<NumericT> matB(B, viennacl::CUDA_MEMORY,
                                         B_size1, offB_row, incB_row, B_row_major ? k : ldb,
                                         B_size2, offB_col, incB_col, B_row_major ? ldb : n, B_row_major);

    viennacl::matrix_base<NumericT> matC(C, viennacl::CUDA_MEMORY,
                                         m, offC_row, incC_row, C_row_major ? m : ldc,
                                         n, offC_col, incC_col, C_row_major ? ldc : n, C_row_major);

    detail::gemm_dispatch(alpha, matA, transA, matB, transB, beta, matC);

    return ViennaCLSuccess;
  }

}


VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASgemm(ViennaCLBackend backend,
                                                            ViennaCLOrder orderA, ViennaCLTranspose transA,
                                                            ViennaCLOrder orderB, ViennaCLTranspose transB,
                                                            ViennaCLOrder orderC,
                                                            ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                                            float alpha,
                                                            float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            float *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                                            float beta,
                                                            float *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc)
{
  return detail::ViennaCLCUDAgemm_impl<float>(backend,
                                              orderA, transA,
                                              orderB, transB,
                                              orderC,
                                              m, n, k,
                                              alpha,
                                              A, offA_row, offA_col, incA_row, incA_col, lda,
                                              B, offB_row, offB_col, incB_row, incB_col, ldb,
                                              beta,
                                              C, offC_row, offC_col, incC_row, incC_col, ldc);
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADgemm(ViennaCLBackend backend,
                                                            ViennaCLOrder orderA, ViennaCLTranspose transA,
                                                            ViennaCLOrder orderB, ViennaCLTranspose transB,
                                                            ViennaCLOrder orderC,
                                                            ViennaCLInt m, ViennaCLInt n, ViennaCLInt k,
                                                            double alpha,
                                                            double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            double *B, ViennaCLInt offB_row, ViennaCLInt offB_col, ViennaCLInt incB_row, ViennaCLInt incB_col, ViennaCLInt ldb,
                                                            double beta,
                                                            double *C, ViennaCLInt offC_row, ViennaCLInt offC_col, ViennaCLInt incC_row, ViennaCLInt incC_col, ViennaCLInt ldc)
{
  return detail::ViennaCLCUDAgemm_impl<double>(backend,
                                               orderA, transA,
                                               orderB, transB,
                                               orderC,
                                               m, n, k,
                                               alpha,
                                               A, offA_row, offA_col, incA_row, incA_col, lda,
                                               B, offB_row, offB_col, incB_row, incB_col, ldb,
                                               beta,
                                               C, offC_row, offC_col, incC_row, incC_col, ldc);
}


#endif
