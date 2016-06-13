#ifndef VIENNACL_LINALG_QR_METHOD_COMMON_HPP
#define VIENNACL_LINALG_QR_METHOD_COMMON_HPP

/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

#include <cmath>

#ifdef VIENNACL_WITH_OPENCL
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/linalg/opencl/kernels/svd.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
#include "viennacl/linalg/cuda/matrix_operations.hpp"
#endif
#include "viennacl/meta/result_of.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

//#include <boost/numeric/ublas/vector.hpp>
//#include <boost/numeric/ublas/io.hpp>

/** @file viennacl/linalg/qr-method-common.hpp
    @brief Common routines used for the QR method and SVD. Experimental.
*/

namespace viennacl
{
namespace linalg
{

const std::string SVD_HOUSEHOLDER_UPDATE_QR_KERNEL = "house_update_QR";
const std::string SVD_MATRIX_TRANSPOSE_KERNEL = "transpose_inplace";
const std::string SVD_INVERSE_SIGNS_KERNEL = "inverse_signs";
const std::string SVD_GIVENS_PREV_KERNEL = "givens_prev";
const std::string SVD_FINAL_ITER_UPDATE_KERNEL = "final_iter_update";
const std::string SVD_UPDATE_QR_COLUMN_KERNEL = "update_qr_column";
const std::string SVD_HOUSEHOLDER_UPDATE_A_LEFT_KERNEL = "house_update_A_left";
const std::string SVD_HOUSEHOLDER_UPDATE_A_RIGHT_KERNEL = "house_update_A_right";
const std::string SVD_HOUSEHOLDER_UPDATE_QL_KERNEL = "house_update_QL";

namespace detail
{
static const double EPS = 1e-10;
static const vcl_size_t ITER_MAX = 50;

template <typename SCALARTYPE>
SCALARTYPE pythag(SCALARTYPE a, SCALARTYPE b)
{
  return std::sqrt(a*a + b*b);
}

template <typename SCALARTYPE>
SCALARTYPE sign(SCALARTYPE val)
{
    return (val >= 0) ? SCALARTYPE(1) : SCALARTYPE(-1);
}

// DEPRECATED: Replace with viennacl::linalg::norm_2
template <typename VectorType>
typename VectorType::value_type norm_lcl(VectorType const & x, vcl_size_t size)
{
  typename VectorType::value_type x_norm = 0.0;
  for(vcl_size_t i = 0; i < size; i++)
    x_norm += std::pow(x[i], 2);
  return std::sqrt(x_norm);
}

template <typename VectorType>
void normalize(VectorType & x, vcl_size_t size)
{
  typename VectorType::value_type x_norm = norm_lcl(x, size);
  for(vcl_size_t i = 0; i < size; i++)
      x[i] /= x_norm;
}



template <typename VectorType>
void householder_vector(VectorType & v, vcl_size_t start)
{
  typedef typename VectorType::value_type    ScalarType;
  ScalarType x_norm = norm_lcl(v, v.size());
  ScalarType alpha = -sign(v[start]) * x_norm;
  v[start] += alpha;
  normalize(v, v.size());
}

template <typename SCALARTYPE>
void transpose(matrix_base<SCALARTYPE> & A)
{
  (void)A;
#ifdef VIENNACL_WITH_OPENCL

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  if(A.row_major())
  {
      viennacl::linalg::opencl::kernels::svd<SCALARTYPE, row_major>::init(ctx);
      viennacl::ocl::kernel & kernel = viennacl::ocl::get_kernel(viennacl::linalg::opencl::kernels::svd<SCALARTYPE, row_major>::program_name(), SVD_MATRIX_TRANSPOSE_KERNEL);

      viennacl::ocl::enqueue(kernel(A,
                                    static_cast<cl_uint>(A.internal_size1()),
                                    static_cast<cl_uint>(A.internal_size2())
                                   )
                            );
  }
  else
  {
      viennacl::linalg::opencl::kernels::svd<SCALARTYPE, row_major>::init(ctx);
      viennacl::ocl::kernel & kernel = viennacl::ocl::get_kernel(viennacl::linalg::opencl::kernels::svd<SCALARTYPE, column_major>::program_name(), SVD_MATRIX_TRANSPOSE_KERNEL);

      viennacl::ocl::enqueue(kernel(A,
                                    static_cast<cl_uint>(A.internal_size1()),
                                    static_cast<cl_uint>(A.internal_size2())
                                   )
                            );
  }

#endif
}



template <typename T>
void cdiv(T xr, T xi, T yr, T yi, T& cdivr, T& cdivi)
{
    // Complex scalar division.
    T r;
    T d;
    if (std::fabs(yr) > std::fabs(yi))
    {
        r = yi / yr;
        d = yr + r * yi;
        cdivr = (xr + r * xi) / d;
        cdivi = (xi - r * xr) / d;
    }
    else
    {
        r = yr / yi;
        d = yi + r * yr;
        cdivr = (r * xr + xi) / d;
        cdivi = (r * xi - xr) / d;
    }
}


template<typename SCALARTYPE>
void prepare_householder_vector(
                              matrix_base<SCALARTYPE>& A,
                              vector_base<SCALARTYPE>& D,
                              vcl_size_t size,
                              vcl_size_t row_start,
                              vcl_size_t col_start,
                              vcl_size_t start,
                              bool is_column
                              )
{
  //boost::numeric::ublas::vector<SCALARTYPE> tmp = boost::numeric::ublas::scalar_vector<SCALARTYPE>(size, 0);
  std::vector<SCALARTYPE> tmp(size);
  copy_vec(A, D, row_start, col_start, is_column);
  fast_copy(D.begin(), D.begin() + vcl_ptrdiff_t(size - start), tmp.begin() + vcl_ptrdiff_t(start));

  detail::householder_vector(tmp, start);
  fast_copy(tmp, D);
}

} //detail
}
}

#endif
