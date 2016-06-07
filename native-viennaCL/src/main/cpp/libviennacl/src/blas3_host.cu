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


//
// xGEMV
//

namespace detail
{
  template <typename NumericT>
  ViennaCLStatus ViennaCLHostgemm_impl(ViennaCLBackend /*backend*/,
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
    typedef typename viennacl::matrix_base<NumericT>::size_type           size_type;
    typedef typename viennacl::matrix_base<NumericT>::size_type           difference_type;

    size_type A_size1 = static_cast<size_type>((transA == ViennaCLTrans) ? k : m);
    size_type A_size2 = static_cast<size_type>((transA == ViennaCLTrans) ? m : k);

    size_type B_size1 = static_cast<size_type>((transB == ViennaCLTrans) ? n : k);
    size_type B_size2 = static_cast<size_type>((transB == ViennaCLTrans) ? k : n);

    bool A_row_major = (orderA == ViennaCLRowMajor);
    bool B_row_major = (orderB == ViennaCLRowMajor);
    bool C_row_major = (orderC == ViennaCLRowMajor);

    viennacl::matrix_base<NumericT> matA(A, viennacl::MAIN_MEMORY,
                                         A_size1, size_type(offA_row), difference_type(incA_row), size_type(A_row_major ? m : lda),
                                         A_size2, size_type(offA_col), difference_type(incA_col), size_type(A_row_major ? lda : k), A_row_major);

    viennacl::matrix_base<NumericT> matB(B, viennacl::MAIN_MEMORY,
                                         B_size1, size_type(offB_row), difference_type(incB_row), size_type(B_row_major ? k : ldb),
                                         B_size2, size_type(offB_col), difference_type(incB_col), size_type(B_row_major ? ldb : n), B_row_major);

    viennacl::matrix_base<NumericT> matC(C, viennacl::MAIN_MEMORY,
                                         size_type(m), size_type(offC_row), difference_type(incC_row), size_type(C_row_major ? m : ldc),
                                         size_type(n), size_type(offC_col), difference_type(incC_col), size_type(C_row_major ? ldc : n), C_row_major);

    detail::gemm_dispatch(alpha, matA, transA, matB, transB, beta, matC);

    return ViennaCLSuccess;
  }

}


VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostSgemm(ViennaCLBackend backend,
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
  return detail::ViennaCLHostgemm_impl<float>(backend,
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

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLHostDgemm(ViennaCLBackend backend,
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
  return detail::ViennaCLHostgemm_impl<double>(backend,
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


