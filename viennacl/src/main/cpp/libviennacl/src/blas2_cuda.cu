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

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/prod.hpp"


#ifdef VIENNACL_WITH_CUDA

// xGEMV

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASgemv(ViennaCLBackend /*backend*/,
                                                            ViennaCLOrder order, ViennaCLTranspose transA,
                                                            ViennaCLInt m, ViennaCLInt n, float alpha, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            float beta,
                                                            float *y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<float> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::CUDA_MEMORY, m, offy, incy);
  viennacl::matrix_base<float> mat(A, viennacl::CUDA_MEMORY,
                                   m, offA_row, incA_row, m,
                                   n, offA_col, incA_col, lda, order == ViennaCLRowMajor);
  v2 *= beta;
  if (transA == ViennaCLTrans)
    v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
  else
    v2 += alpha * viennacl::linalg::prod(mat, v1);

  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADgemv(ViennaCLBackend /*backend*/,
                                                            ViennaCLOrder order, ViennaCLTranspose transA,
                                                            ViennaCLInt m, ViennaCLInt n, double alpha, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                            double beta,
                                                            double *y, ViennaCLInt offy, ViennaCLInt incy)
{
  viennacl::vector_base<double> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::CUDA_MEMORY, m, offy, incy);
  viennacl::matrix_base<double> mat(A, viennacl::CUDA_MEMORY,
                                    m, offA_row, incA_row, m,
                                    n, offA_col, incA_col, lda, order == ViennaCLRowMajor);
  v2 *= beta;
  if (transA == ViennaCLTrans)
    v2 += alpha * viennacl::linalg::prod(viennacl::trans(mat), v1);
  else
    v2 += alpha * viennacl::linalg::prod(mat, v1);

  return ViennaCLSuccess;
}



// xTRSV

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDAStrsv(ViennaCLBackend /*backend*/,
                                                            ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                                            ViennaCLInt n, float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            float *x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<float> v(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::matrix_base<float> mat(A, viennacl::CUDA_MEMORY,
                                   n, offA_row, incA_row, n,
                                   n, offA_col, incA_col, lda, order == ViennaCLRowMajor);
  if (transA == ViennaCLTrans)
  {
    if (uplo == ViennaCLUpper)
      if (diag == ViennaCLUnit)
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::unit_upper_tag());
      else
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::upper_tag());
    else
      if (diag == ViennaCLUnit)
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::unit_lower_tag());
      else
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::lower_tag());
  }
  else
  {
    if (uplo == ViennaCLUpper)
      if (diag == ViennaCLUnit)
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::unit_upper_tag());
      else
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::upper_tag());
    else
      if (diag == ViennaCLUnit)
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::unit_lower_tag());
      else
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::lower_tag());
  }

  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADtrsv(ViennaCLBackend /*backend*/,
                                                            ViennaCLUplo uplo, ViennaCLOrder order, ViennaCLTranspose transA, ViennaCLDiag diag,
                                                            ViennaCLInt n, double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda,
                                                            double *x, ViennaCLInt offx, ViennaCLInt incx)
{
  viennacl::vector_base<double> v(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::matrix_base<double> mat(A, viennacl::CUDA_MEMORY,
                                    n, offA_row, incA_row, n,
                                    n, offA_col, incA_col, lda, order == ViennaCLRowMajor);
  if (transA == ViennaCLTrans)
  {
    if (uplo == ViennaCLUpper)
      if (diag == ViennaCLUnit)
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::unit_upper_tag());
      else
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::upper_tag());
    else
      if (diag == ViennaCLUnit)
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::unit_lower_tag());
      else
        viennacl::linalg::inplace_solve(viennacl::trans(mat), v, viennacl::linalg::lower_tag());
  }
  else
  {
    if (uplo == ViennaCLUpper)
      if (diag == ViennaCLUnit)
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::unit_upper_tag());
      else
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::upper_tag());
    else
      if (diag == ViennaCLUnit)
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::unit_lower_tag());
      else
        viennacl::linalg::inplace_solve(mat, v, viennacl::linalg::lower_tag());
  }

  return ViennaCLSuccess;
}



// xGER

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDASger(ViennaCLBackend /*backend*/,
                                                           ViennaCLOrder order,
                                                           ViennaCLInt m, ViennaCLInt n,
                                                           float alpha,
                                                           float *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           float *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           float *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda)
{
  viennacl::vector_base<float> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<float> v2(y, viennacl::CUDA_MEMORY, m, offy, incy);
  viennacl::matrix_base<float> mat(A, viennacl::CUDA_MEMORY,
                                   m, offA_row, incA_row, m,
                                   n, offA_col, incA_col, lda, order == ViennaCLRowMajor);

  mat += alpha * viennacl::linalg::outer_prod(v1, v2);

  return ViennaCLSuccess;
}

VIENNACL_EXPORTED_FUNCTION ViennaCLStatus ViennaCLCUDADger(ViennaCLBackend /*backend*/,
                                                           ViennaCLOrder order,
                                                           ViennaCLInt m,  ViennaCLInt n,
                                                           double alpha,
                                                           double *x, ViennaCLInt offx, ViennaCLInt incx,
                                                           double *y, ViennaCLInt offy, ViennaCLInt incy,
                                                           double *A, ViennaCLInt offA_row, ViennaCLInt offA_col, ViennaCLInt incA_row, ViennaCLInt incA_col, ViennaCLInt lda)
{
  viennacl::vector_base<double> v1(x, viennacl::CUDA_MEMORY, n, offx, incx);
  viennacl::vector_base<double> v2(y, viennacl::CUDA_MEMORY, m, offy, incy);
  viennacl::matrix_base<double> mat(A, viennacl::CUDA_MEMORY,
                                    m, offA_row, incA_row, m,
                                    n, offA_col, incA_col, lda, order == ViennaCLRowMajor);

  mat += alpha * viennacl::linalg::outer_prod(v1, v2);

  return ViennaCLSuccess;
}

#endif
