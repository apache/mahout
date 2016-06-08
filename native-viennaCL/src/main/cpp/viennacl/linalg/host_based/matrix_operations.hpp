#ifndef VIENNACL_LINALG_HOST_BASED_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_MATRIX_OPERATIONS_HPP_

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

/** @file  viennacl/linalg/host_based/matrix_operations.hpp
    @brief Implementations of dense matrix related operations, including matrix-vector products, using a plain single-threaded or OpenMP-enabled execution on CPU.
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/detail/op_applier.hpp"
#include "viennacl/linalg/host_based/common.hpp"
#include "viennacl/linalg/prod.hpp"

// Minimum Matrix size(size1*size2) for using OpenMP on matrix operations:
#ifndef VIENNACL_OPENMP_MATRIX_MIN_SIZE
  #define VIENNACL_OPENMP_MATRIX_MIN_SIZE  5000
#endif

namespace viennacl
{
namespace linalg
{
namespace host_based
{

//
// Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
//

template<typename DestNumericT, typename SrcNumericT>
void convert(matrix_base<DestNumericT> & mat1, matrix_base<SrcNumericT> const & mat2)
{
  assert(mat1.row_major() == mat2.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  DestNumericT      * data_A = detail::extract_raw_pointer<DestNumericT>(mat1);
  SrcNumericT const * data_B = detail::extract_raw_pointer<SrcNumericT>(mat2);

  vcl_size_t A_start1 = viennacl::traits::start1(mat1);
  vcl_size_t A_start2 = viennacl::traits::start2(mat1);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat1);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat1);
  vcl_size_t A_size1  = viennacl::traits::size1(mat1);
  vcl_size_t A_size2  = viennacl::traits::size2(mat1);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);

  vcl_size_t B_start1 = viennacl::traits::start1(mat2);
  vcl_size_t B_start2 = viennacl::traits::start2(mat2);
  vcl_size_t B_inc1   = viennacl::traits::stride1(mat2);
  vcl_size_t B_inc2   = viennacl::traits::stride2(mat2);
  vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(mat2);
  vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(mat2);

  if (mat1.row_major())
  {
    detail::matrix_array_wrapper<DestNumericT,      row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<SrcNumericT const, row_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for (long row = 0; row < static_cast<long>(A_size1); ++row)
      for (vcl_size_t col = 0; col < A_size2; ++col)
        wrapper_A(row, col) = static_cast<DestNumericT>(wrapper_B(row, col));
  }
  else
  {
    detail::matrix_array_wrapper<DestNumericT,      column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<SrcNumericT const, column_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for (long col = 0; col < static_cast<long>(A_size2); ++col)
      for (vcl_size_t row = 0; row < A_size1; ++row)
        wrapper_A(row, col) = static_cast<DestNumericT>(wrapper_B(row, col));
  }
}



template<typename NumericT,
         typename SizeT, typename DistanceT>
void trans(const matrix_expression<const matrix_base<NumericT, SizeT, DistanceT>,
           const matrix_base<NumericT, SizeT, DistanceT>, op_trans> & proxy, matrix_base<NumericT> & temp_trans)
{
  typedef NumericT        value_type;
  const value_type * data_A = detail::extract_raw_pointer<value_type>(proxy.lhs());
  value_type * data_B       = detail::extract_raw_pointer<value_type>(temp_trans);

  vcl_size_t A_start1         = viennacl::traits::start1(proxy.lhs());
  vcl_size_t A_start2         = viennacl::traits::start2(proxy.lhs());
  vcl_size_t A_internal_size1 = viennacl::traits::internal_size1(proxy.lhs());
  vcl_size_t A_internal_size2 = viennacl::traits::internal_size2(proxy.lhs());
  vcl_size_t A_inc1           = viennacl::traits::stride1(proxy.lhs());
  vcl_size_t A_inc2           = viennacl::traits::stride2(proxy.lhs());
  vcl_size_t A_size1          = viennacl::traits::size1(proxy.lhs());
  vcl_size_t A_size2          = viennacl::traits::size2(proxy.lhs());

  vcl_size_t B_start1         = viennacl::traits::start1(temp_trans);
  vcl_size_t B_start2         = viennacl::traits::start2(temp_trans);
  vcl_size_t B_internal_size1 = viennacl::traits::internal_size1(temp_trans);
  vcl_size_t B_internal_size2 = viennacl::traits::internal_size2(temp_trans);
  vcl_size_t B_inc1           = viennacl::traits::stride1(temp_trans);
  vcl_size_t B_inc2           = viennacl::traits::stride2(temp_trans);

  const vcl_size_t sub_mat_size = 64; //The matrix will be divided into sub-matrices for better storage access.

  vcl_size_t row_count = A_size1 / sub_mat_size;
  vcl_size_t col_count = A_size2 / sub_mat_size;

  vcl_size_t row_count_remainder = A_size1 % sub_mat_size;
  vcl_size_t col_count_remainder = A_size2 % sub_mat_size;

  if (proxy.lhs().row_major())
  {
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for(long i = 0; i < static_cast<long>(row_count*col_count); ++i)//This is the main part of the transposition
    {
      vcl_size_t row = vcl_size_t(i) / col_count;
      vcl_size_t col = vcl_size_t(i) % col_count;

      detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_A(data_A, A_start1 + A_inc1 * (row * sub_mat_size)
                                                                               , A_start2 + A_inc2 * (col * sub_mat_size), A_inc1
                                                                               , A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type      , row_major, false> wrapper_B(data_B, B_start1 + B_inc1 * (col * sub_mat_size)
                                                                               , B_start2 + B_inc2 * (row * sub_mat_size), B_inc1
                                                                               , B_inc2, B_internal_size1, B_internal_size2);
      for(vcl_size_t j = 0; j < (sub_mat_size); ++j)
        for(vcl_size_t k = 0; k < (sub_mat_size); ++k)
          wrapper_B(j, k) = wrapper_A(k, j);
    }
    { //This is the transposition of the remainder on the right side of the matrix
      detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_A(data_A, A_start1
                                                                               , A_start2 + A_inc2 * (col_count * sub_mat_size), A_inc1
                                                                               , A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type      , row_major, false> wrapper_B(data_B, B_start1 + B_inc1 * (col_count * sub_mat_size)
                                                                               , B_start2, B_inc1
                                                                               , B_inc2, B_internal_size1, B_internal_size2);
      for(vcl_size_t j = 0; j < col_count_remainder; ++j)
        for(vcl_size_t k = 0 ; k < A_size1; ++k)
          wrapper_B(j, k) = wrapper_A(k, j);
    }
    { //This is the transposition of the remainder on the bottom side of the matrix
      detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_A(data_A, A_start1 + A_inc1 * (row_count * sub_mat_size)
                                                                               , A_start2, A_inc1
                                                                               , A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type      , row_major, false> wrapper_B(data_B,B_start1
                                                                               , B_start2  + B_inc2 * (row_count * sub_mat_size), B_inc1
                                                                               , B_inc2, B_internal_size1, B_internal_size2);
      for(vcl_size_t j = 0; j < row_count_remainder; ++j)
        for(vcl_size_t k = 0; k < (A_size2 - col_count_remainder); ++k)
          wrapper_B(k, j) = wrapper_A(j, k);
    }
  }
  else
  {
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for(long i = 0; i < static_cast<long>(row_count*col_count); ++i)//This is the main part of the transposition
    {
      vcl_size_t row = vcl_size_t(i) / col_count;
      vcl_size_t col = vcl_size_t(i) % col_count;

      detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_A(data_A, A_start1 + A_inc1 * (row * sub_mat_size)
                                                                                  , A_start2 + A_inc2 * (col * sub_mat_size), A_inc1
                                                                                  , A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type      , column_major, false> wrapper_B(data_B, B_start1 + B_inc1 * (col * sub_mat_size)
                                                                                  , B_start2 + B_inc2 * (row * sub_mat_size), B_inc1
                                                                                  , B_inc2, B_internal_size1, B_internal_size2);
      for(vcl_size_t j = 0; j < (sub_mat_size); ++j)
        for(vcl_size_t k = 0; k < (sub_mat_size); ++k)
          wrapper_B(k, j)=wrapper_A(j, k);
    }
    { //This is the transposition of the remainder on the right side of the matrix
      detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_A(data_A, A_start1
                                                                                  , A_start2 + A_inc2 * (col_count * sub_mat_size), A_inc1
                                                                                  , A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type      , column_major, false> wrapper_B(data_B,B_start1 + B_inc1 * (col_count * sub_mat_size)
                                                                                  , B_start2, B_inc1
                                                                                  , B_inc2, B_internal_size1, B_internal_size2);
      for(vcl_size_t j = 0; j < col_count_remainder; ++j)
        for(vcl_size_t k = 0; k < A_size1; ++k)
          wrapper_B(j, k)=wrapper_A(k, j);
    }
    { //This is the transposition of the remainder on the bottom side of the matrix
      detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_A(data_A, A_start1 + A_inc1 * (row_count * sub_mat_size)
                                                                                  , A_start2, A_inc1
                                                                                  , A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type      , column_major, false> wrapper_B(data_B, B_start1
                                                                                  , B_start2  + B_inc2 * (row_count * sub_mat_size), B_inc1
                                                                                  , B_inc2, B_internal_size1, B_internal_size2);
      for(vcl_size_t j = 0; j < row_count_remainder; ++j)
        for(vcl_size_t k = 0; k < (A_size2 - col_count_remainder); ++k)
          wrapper_B(k, j)=wrapper_A(j, k);
    }
  }
}

template<typename NumericT, typename ScalarT1>
void am(matrix_base<NumericT> & mat1,
        matrix_base<NumericT> const & mat2, ScalarT1 const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha)
{
  assert(mat1.row_major() == mat2.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  typedef NumericT        value_type;

  value_type       * data_A = detail::extract_raw_pointer<value_type>(mat1);
  value_type const * data_B = detail::extract_raw_pointer<value_type>(mat2);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;

  vcl_size_t A_start1 = viennacl::traits::start1(mat1);
  vcl_size_t A_start2 = viennacl::traits::start2(mat1);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat1);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat1);
  vcl_size_t A_size1  = viennacl::traits::size1(mat1);
  vcl_size_t A_size2  = viennacl::traits::size2(mat1);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);

  vcl_size_t B_start1 = viennacl::traits::start1(mat2);
  vcl_size_t B_start2 = viennacl::traits::start2(mat2);
  vcl_size_t B_inc1   = viennacl::traits::stride1(mat2);
  vcl_size_t B_inc2   = viennacl::traits::stride2(mat2);
  vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(mat2);
  vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(mat2);

  if (mat1.row_major())
  {
    detail::matrix_array_wrapper<value_type,       row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);

    if (reciprocal_alpha)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
        for (vcl_size_t col = 0; col < A_size2; ++col)
          wrapper_A(row, col) = wrapper_B(row, col) / data_alpha;
    }
    else
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
        for (vcl_size_t col = 0; col < A_size2; ++col)
          wrapper_A(row, col) = wrapper_B(row, col) * data_alpha;
    }
  }
  else
  {
    detail::matrix_array_wrapper<value_type,       column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);

    if (reciprocal_alpha)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long col = 0; col < static_cast<long>(A_size2); ++col)
        for (vcl_size_t row = 0; row < A_size1; ++row)
          wrapper_A(row, col) = wrapper_B(row, col) / data_alpha;
    }
    else
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long col = 0; col < static_cast<long>(A_size2); ++col)
        for (vcl_size_t row = 0; row < A_size1; ++row)
          wrapper_A(row, col) = wrapper_B(row, col) * data_alpha;
    }
  }
}


template<typename NumericT,
         typename ScalarT1, typename ScalarT2>
void ambm(matrix_base<NumericT> & mat1,
          matrix_base<NumericT> const & mat2, ScalarT1 const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
          matrix_base<NumericT> const & mat3, ScalarT2 const & beta,  vcl_size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(mat1.row_major() == mat2.row_major() && mat1.row_major() == mat3.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  typedef NumericT        value_type;

  value_type       * data_A = detail::extract_raw_pointer<value_type>(mat1);
  value_type const * data_B = detail::extract_raw_pointer<value_type>(mat2);
  value_type const * data_C = detail::extract_raw_pointer<value_type>(mat3);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;

  value_type data_beta = beta;
  if (flip_sign_beta)
    data_beta = -data_beta;

  vcl_size_t A_start1 = viennacl::traits::start1(mat1);
  vcl_size_t A_start2 = viennacl::traits::start2(mat1);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat1);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat1);
  vcl_size_t A_size1  = viennacl::traits::size1(mat1);
  vcl_size_t A_size2  = viennacl::traits::size2(mat1);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);

  vcl_size_t B_start1 = viennacl::traits::start1(mat2);
  vcl_size_t B_start2 = viennacl::traits::start2(mat2);
  vcl_size_t B_inc1   = viennacl::traits::stride1(mat2);
  vcl_size_t B_inc2   = viennacl::traits::stride2(mat2);
  vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(mat2);
  vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(mat2);

  vcl_size_t C_start1 = viennacl::traits::start1(mat3);
  vcl_size_t C_start2 = viennacl::traits::start2(mat3);
  vcl_size_t C_inc1   = viennacl::traits::stride1(mat3);
  vcl_size_t C_inc2   = viennacl::traits::stride2(mat3);
  vcl_size_t C_internal_size1  = viennacl::traits::internal_size1(mat3);
  vcl_size_t C_internal_size2  = viennacl::traits::internal_size2(mat3);

  if (mat1.row_major())
  {
    detail::matrix_array_wrapper<value_type,       row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
    detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

    if (reciprocal_alpha && reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
        for (vcl_size_t col = 0; col < A_size2; ++col)
          wrapper_A(row, col) = wrapper_B(row, col) / data_alpha + wrapper_C(row, col) / data_beta;
    }
    else if (reciprocal_alpha && !reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
        for (vcl_size_t col = 0; col < A_size2; ++col)
          wrapper_A(row, col) = wrapper_B(row, col) / data_alpha + wrapper_C(row, col) * data_beta;
    }
    else if (!reciprocal_alpha && reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
        for (vcl_size_t col = 0; col < A_size2; ++col)
          wrapper_A(row, col) = wrapper_B(row, col) * data_alpha + wrapper_C(row, col) / data_beta;
    }
    else if (!reciprocal_alpha && !reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
        for (vcl_size_t col = 0; col < A_size2; ++col)
          wrapper_A(row, col) = wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
    }
  }
  else
  {
    detail::matrix_array_wrapper<value_type,       column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
    detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

    if (reciprocal_alpha && reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long col = 0; col < static_cast<long>(A_size2); ++col)
        for (vcl_size_t row = 0; row < A_size1; ++row)
          wrapper_A(row, col) = wrapper_B(row, col) / data_alpha + wrapper_C(row, col) / data_beta;
    }
    else if (reciprocal_alpha && !reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long col = 0; col < static_cast<long>(A_size2); ++col)
        for (vcl_size_t row = 0; row < A_size1; ++row)
          wrapper_A(row, col) = wrapper_B(row, col) / data_alpha + wrapper_C(row, col) * data_beta;
    }
    else if (!reciprocal_alpha && reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long col = 0; col < static_cast<long>(A_size2); ++col)
        for (vcl_size_t row = 0; row < A_size1; ++row)
          wrapper_A(row, col) = wrapper_B(row, col) * data_alpha + wrapper_C(row, col) / data_beta;
    }
    else if (!reciprocal_alpha && !reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long col = 0; col < static_cast<long>(A_size2); ++col)
        for (vcl_size_t row = 0; row < A_size1; ++row)
          wrapper_A(row, col) = wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
    }
  }

}


template<typename NumericT,
         typename ScalarT1, typename ScalarT2>
void ambm_m(matrix_base<NumericT> & mat1,
            matrix_base<NumericT> const & mat2, ScalarT1 const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
            matrix_base<NumericT> const & mat3, ScalarT2 const & beta,  vcl_size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(mat1.row_major() == mat2.row_major() && mat1.row_major() == mat3.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  typedef NumericT        value_type;

  value_type       * data_A = detail::extract_raw_pointer<value_type>(mat1);
  value_type const * data_B = detail::extract_raw_pointer<value_type>(mat2);
  value_type const * data_C = detail::extract_raw_pointer<value_type>(mat3);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;

  value_type data_beta = beta;
  if (flip_sign_beta)
    data_beta = -data_beta;

  vcl_size_t A_start1 = viennacl::traits::start1(mat1);
  vcl_size_t A_start2 = viennacl::traits::start2(mat1);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat1);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat1);
  vcl_size_t A_size1  = viennacl::traits::size1(mat1);
  vcl_size_t A_size2  = viennacl::traits::size2(mat1);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);

  vcl_size_t B_start1 = viennacl::traits::start1(mat2);
  vcl_size_t B_start2 = viennacl::traits::start2(mat2);
  vcl_size_t B_inc1   = viennacl::traits::stride1(mat2);
  vcl_size_t B_inc2   = viennacl::traits::stride2(mat2);
  vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(mat2);
  vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(mat2);

  vcl_size_t C_start1 = viennacl::traits::start1(mat3);
  vcl_size_t C_start2 = viennacl::traits::start2(mat3);
  vcl_size_t C_inc1   = viennacl::traits::stride1(mat3);
  vcl_size_t C_inc2   = viennacl::traits::stride2(mat3);
  vcl_size_t C_internal_size1  = viennacl::traits::internal_size1(mat3);
  vcl_size_t C_internal_size2  = viennacl::traits::internal_size2(mat3);

  if (mat1.row_major())
  {
    detail::matrix_array_wrapper<value_type,       row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
    detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

    if (reciprocal_alpha && reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
        for (vcl_size_t col = 0; col < A_size2; ++col)
          wrapper_A(row, col) += wrapper_B(row, col) / data_alpha + wrapper_C(row, col) / data_beta;
    }
    else if (reciprocal_alpha && !reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
        for (vcl_size_t col = 0; col < A_size2; ++col)
          wrapper_A(row, col) += wrapper_B(row, col) / data_alpha + wrapper_C(row, col) * data_beta;
    }
    else if (!reciprocal_alpha && reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
        for (vcl_size_t col = 0; col < A_size2; ++col)
          wrapper_A(row, col) += wrapper_B(row, col) * data_alpha + wrapper_C(row, col) / data_beta;
    }
    else if (!reciprocal_alpha && !reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
        for (vcl_size_t col = 0; col < A_size2; ++col)
          wrapper_A(row, col) += wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
    }
  }
  else
  {
    detail::matrix_array_wrapper<value_type,       column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
    detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

    if (reciprocal_alpha && reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long col = 0; col < static_cast<long>(A_size2); ++col)
        for (vcl_size_t row = 0; row < A_size1; ++row)
          wrapper_A(row, col) += wrapper_B(row, col) / data_alpha + wrapper_C(row, col) / data_beta;
    }
    else if (reciprocal_alpha && !reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long col = 0; col < static_cast<long>(A_size2); ++col)
        for (vcl_size_t row = 0; row < A_size1; ++row)
          wrapper_A(row, col) += wrapper_B(row, col) / data_alpha + wrapper_C(row, col) * data_beta;
    }
    else if (!reciprocal_alpha && reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long col = 0; col < static_cast<long>(A_size2); ++col)
        for (vcl_size_t row = 0; row < A_size1; ++row)
          wrapper_A(row, col) += wrapper_B(row, col) * data_alpha + wrapper_C(row, col) / data_beta;
    }
    else if (!reciprocal_alpha && !reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long col = 0; col < static_cast<long>(A_size2); ++col)
        for (vcl_size_t row = 0; row < A_size1; ++row)
          wrapper_A(row, col) += wrapper_B(row, col) * data_alpha + wrapper_C(row, col) * data_beta;
    }
  }

}




template<typename NumericT>
void matrix_assign(matrix_base<NumericT> & mat, NumericT s, bool clear = false)
{
  typedef NumericT        value_type;

  value_type * data_A = detail::extract_raw_pointer<value_type>(mat);
  value_type    alpha = static_cast<value_type>(s);

  vcl_size_t A_start1 = viennacl::traits::start1(mat);
  vcl_size_t A_start2 = viennacl::traits::start2(mat);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
  vcl_size_t A_size1  = clear ? viennacl::traits::internal_size1(mat) : viennacl::traits::size1(mat);
  vcl_size_t A_size2  = clear ? viennacl::traits::internal_size2(mat) : viennacl::traits::size2(mat);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

  if (mat.row_major())
  {
    detail::matrix_array_wrapper<value_type, row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for (long row = 0; row < static_cast<long>(A_size1); ++row)
      for (vcl_size_t col = 0; col < A_size2; ++col)
        wrapper_A(static_cast<vcl_size_t>(row), col) = alpha;
        //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]
        // = data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha;
  }
  else
  {
    detail::matrix_array_wrapper<value_type, column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for (long col = 0; col < static_cast<long>(A_size2); ++col)
      for (vcl_size_t row = 0; row < A_size1; ++row)
        wrapper_A(row, static_cast<vcl_size_t>(col)) = alpha;
        //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]
        // = data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha;
  }
}



template<typename NumericT>
void matrix_diagonal_assign(matrix_base<NumericT> & mat, NumericT s)
{
  typedef NumericT        value_type;

  value_type * data_A = detail::extract_raw_pointer<value_type>(mat);
  value_type    alpha = static_cast<value_type>(s);

  vcl_size_t A_start1 = viennacl::traits::start1(mat);
  vcl_size_t A_start2 = viennacl::traits::start2(mat);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
  vcl_size_t A_size1  = viennacl::traits::size1(mat);
  //vcl_size_t A_size2  = viennacl::traits::size2(mat);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

  if (mat.row_major())
  {
    detail::matrix_array_wrapper<value_type, row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size1) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for (long row = 0; row < static_cast<long>(A_size1); ++row)
      wrapper_A(row, row) = alpha;
  }
  else
  {
    detail::matrix_array_wrapper<value_type, column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size1) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for (long row = 0; row < static_cast<long>(A_size1); ++row)
      wrapper_A(row, row) = alpha;
  }
}

template<typename NumericT>
void matrix_diag_from_vector(const vector_base<NumericT> & vec, int k, matrix_base<NumericT> & mat)
{
  typedef NumericT        value_type;

  value_type       *data_A   = detail::extract_raw_pointer<value_type>(mat);
  value_type const *data_vec = detail::extract_raw_pointer<value_type>(vec);

  vcl_size_t A_start1 = viennacl::traits::start1(mat);
  vcl_size_t A_start2 = viennacl::traits::start2(mat);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
  //vcl_size_t A_size1  = viennacl::traits::size1(mat);
  //vcl_size_t A_size2  = viennacl::traits::size2(mat);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

  vcl_size_t v_start = viennacl::traits::start(vec);
  vcl_size_t v_inc   = viennacl::traits::stride(vec);
  vcl_size_t v_size  = viennacl::traits::size(vec);

  vcl_size_t row_start = 0;
  vcl_size_t col_start = 0;

  if (k >= 0)
    col_start = static_cast<vcl_size_t>(k);
  else
    row_start = static_cast<vcl_size_t>(-k);

  matrix_assign(mat, NumericT(0));

  if (mat.row_major())
  {
    detail::matrix_array_wrapper<value_type, row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

    for (vcl_size_t i = 0; i < v_size; ++i)
      wrapper_A(row_start + i, col_start + i) = data_vec[v_start + i * v_inc];
  }
  else
  {
    detail::matrix_array_wrapper<value_type, column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

    for (vcl_size_t i = 0; i < v_size; ++i)
      wrapper_A(row_start + i, col_start + i) = data_vec[v_start + i * v_inc];
  }
}

template<typename NumericT>
void matrix_diag_to_vector(const matrix_base<NumericT> & mat, int k, vector_base<NumericT> & vec)
{
  typedef NumericT        value_type;

  value_type const * data_A   = detail::extract_raw_pointer<value_type>(mat);
  value_type       * data_vec = detail::extract_raw_pointer<value_type>(vec);

  vcl_size_t A_start1 = viennacl::traits::start1(mat);
  vcl_size_t A_start2 = viennacl::traits::start2(mat);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
  //vcl_size_t A_size1  = viennacl::traits::size1(mat);
  //vcl_size_t A_size2  = viennacl::traits::size2(mat);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

  vcl_size_t v_start = viennacl::traits::start(vec);
  vcl_size_t v_inc   = viennacl::traits::stride(vec);
  vcl_size_t v_size  = viennacl::traits::size(vec);

  vcl_size_t row_start = 0;
  vcl_size_t col_start = 0;

  if (k >= 0)
    col_start = static_cast<vcl_size_t>(k);
  else
    row_start = static_cast<vcl_size_t>(-k);

  if (mat.row_major())
  {
    detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

    for (vcl_size_t i = 0; i < v_size; ++i)
      data_vec[v_start + i * v_inc] = wrapper_A(row_start + i, col_start + i);
  }
  else
  {
    detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

    for (vcl_size_t i = 0; i < v_size; ++i)
      data_vec[v_start + i * v_inc] = wrapper_A(row_start + i, col_start + i);
  }
}

template<typename NumericT>
void matrix_row(const matrix_base<NumericT> & mat, unsigned int i, vector_base<NumericT> & vec)
{
  typedef NumericT        value_type;

  value_type const * data_A   = detail::extract_raw_pointer<value_type>(mat);
  value_type       * data_vec = detail::extract_raw_pointer<value_type>(vec);

  vcl_size_t A_start1 = viennacl::traits::start1(mat);
  vcl_size_t A_start2 = viennacl::traits::start2(mat);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
  //vcl_size_t A_size1  = viennacl::traits::size1(mat);
  //vcl_size_t A_size2  = viennacl::traits::size2(mat);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

  vcl_size_t v_start = viennacl::traits::start(vec);
  vcl_size_t v_inc   = viennacl::traits::stride(vec);
  vcl_size_t v_size  = viennacl::traits::size(vec);

  if (mat.row_major())
  {
    detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

    for (vcl_size_t j = 0; j < v_size; ++j)
      data_vec[v_start + j * v_inc] = wrapper_A(static_cast<vcl_size_t>(i), j);
  }
  else
  {
    detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

    for (vcl_size_t j = 0; j < v_size; ++j)
      data_vec[v_start + j * v_inc] = wrapper_A(static_cast<vcl_size_t>(i), j);
  }
}

template<typename NumericT>
void matrix_column(const matrix_base<NumericT> & mat, unsigned int j, vector_base<NumericT> & vec)
{
  typedef NumericT        value_type;

  value_type const * data_A   = detail::extract_raw_pointer<value_type>(mat);
  value_type       * data_vec = detail::extract_raw_pointer<value_type>(vec);

  vcl_size_t A_start1 = viennacl::traits::start1(mat);
  vcl_size_t A_start2 = viennacl::traits::start2(mat);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
  //vcl_size_t A_size1  = viennacl::traits::size1(mat);
  //vcl_size_t A_size2  = viennacl::traits::size2(mat);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

  vcl_size_t v_start = viennacl::traits::start(vec);
  vcl_size_t v_inc   = viennacl::traits::stride(vec);
  vcl_size_t v_size  = viennacl::traits::size(vec);

  if (mat.row_major())
  {
    detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

    for (vcl_size_t i = 0; i < v_size; ++i)
      data_vec[v_start + i * v_inc] = wrapper_A(i, static_cast<vcl_size_t>(j));
  }
  else
  {
    detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);

    for (vcl_size_t i = 0; i < v_size; ++i)
      data_vec[v_start + i * v_inc] = wrapper_A(i, static_cast<vcl_size_t>(j));
  }
}

//
///////////////////////// Element-wise operation //////////////////////////////////
//

// Binary operations A = B .* C and A = B ./ C

/** @brief Implementation of the element-wise operations A = B .* C and A = B ./ C    (using MATLAB syntax)
*
* @param A      The result matrix (or -range, or -slice)
* @param proxy  The proxy object holding B, C, and the operation
*/
template<typename NumericT, typename OpT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_binary<OpT> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT        value_type;
  typedef viennacl::linalg::detail::op_applier<op_element_binary<OpT> >    OpFunctor;

  value_type       * data_A = detail::extract_raw_pointer<value_type>(A);
  value_type const * data_B = detail::extract_raw_pointer<value_type>(proxy.lhs());
  value_type const * data_C = detail::extract_raw_pointer<value_type>(proxy.rhs());

  vcl_size_t A_start1 = viennacl::traits::start1(A);
  vcl_size_t A_start2 = viennacl::traits::start2(A);
  vcl_size_t A_inc1   = viennacl::traits::stride1(A);
  vcl_size_t A_inc2   = viennacl::traits::stride2(A);
  vcl_size_t A_size1  = viennacl::traits::size1(A);
  vcl_size_t A_size2  = viennacl::traits::size2(A);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

  vcl_size_t B_start1 = viennacl::traits::start1(proxy.lhs());
  vcl_size_t B_start2 = viennacl::traits::start2(proxy.lhs());
  vcl_size_t B_inc1   = viennacl::traits::stride1(proxy.lhs());
  vcl_size_t B_inc2   = viennacl::traits::stride2(proxy.lhs());
  vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(proxy.lhs());
  vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(proxy.lhs());

  vcl_size_t C_start1 = viennacl::traits::start1(proxy.rhs());
  vcl_size_t C_start2 = viennacl::traits::start2(proxy.rhs());
  vcl_size_t C_inc1   = viennacl::traits::stride1(proxy.rhs());
  vcl_size_t C_inc2   = viennacl::traits::stride2(proxy.rhs());
  vcl_size_t C_internal_size1  = viennacl::traits::internal_size1(proxy.rhs());
  vcl_size_t C_internal_size2  = viennacl::traits::internal_size2(proxy.rhs());

  if (A.row_major())
  {
    detail::matrix_array_wrapper<value_type,       row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
    detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for (long row = 0; row < static_cast<long>(A_size1); ++row)
      for (vcl_size_t col = 0; col < A_size2; ++col)
        OpFunctor::apply(wrapper_A(row, col), wrapper_B(row, col), wrapper_C(row, col));
        //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]
        // =   data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha
        //   + data_C[index_generator_C::mem_index(row * C_inc1 + C_start1, col * C_inc2 + C_start2, C_internal_size1, C_internal_size2)] * beta;
  }
  else
  {
    detail::matrix_array_wrapper<value_type,       column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
    detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for (long col = 0; col < static_cast<long>(A_size2); ++col)
      for (vcl_size_t row = 0; row < A_size1; ++row)
        OpFunctor::apply(wrapper_A(row, col), wrapper_B(row, col), wrapper_C(row, col));

        //data_A[index_generator_A::mem_index(row * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]
        // =   data_B[index_generator_B::mem_index(row * B_inc1 + B_start1, col * B_inc2 + B_start2, B_internal_size1, B_internal_size2)] * alpha
        //   + data_C[index_generator_C::mem_index(row * C_inc1 + C_start1, col * C_inc2 + C_start2, C_internal_size1, C_internal_size2)] * beta;
  }
}

// Unary operations

// A = op(B)
template<typename NumericT, typename OpT>
void element_op(matrix_base<NumericT> & A,
                matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_element_unary<OpT> > const & proxy)
{
  assert(A.row_major() == proxy.lhs().row_major() && A.row_major() == proxy.rhs().row_major() && bool("Element-wise operations on mixed matrix layouts not supported yet!"));

  typedef NumericT        value_type;
  typedef viennacl::linalg::detail::op_applier<op_element_unary<OpT> >    OpFunctor;

  value_type       * data_A = detail::extract_raw_pointer<value_type>(A);
  value_type const * data_B = detail::extract_raw_pointer<value_type>(proxy.lhs());

  vcl_size_t A_start1 = viennacl::traits::start1(A);
  vcl_size_t A_start2 = viennacl::traits::start2(A);
  vcl_size_t A_inc1   = viennacl::traits::stride1(A);
  vcl_size_t A_inc2   = viennacl::traits::stride2(A);
  vcl_size_t A_size1  = viennacl::traits::size1(A);
  vcl_size_t A_size2  = viennacl::traits::size2(A);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

  vcl_size_t B_start1 = viennacl::traits::start1(proxy.lhs());
  vcl_size_t B_start2 = viennacl::traits::start2(proxy.lhs());
  vcl_size_t B_inc1   = viennacl::traits::stride1(proxy.lhs());
  vcl_size_t B_inc2   = viennacl::traits::stride2(proxy.lhs());
  vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(proxy.lhs());
  vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(proxy.lhs());

  if (A.row_major())
  {
    detail::matrix_array_wrapper<value_type,       row_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for (long row = 0; row < static_cast<long>(A_size1); ++row)
      for (vcl_size_t col = 0; col < A_size2; ++col)
        OpFunctor::apply(wrapper_A(row, col), wrapper_B(row, col));
  }
  else
  {
    detail::matrix_array_wrapper<value_type,       column_major, false> wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
    detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for (long col = 0; col < static_cast<long>(A_size2); ++col)
      for (vcl_size_t row = 0; row < A_size1; ++row)
        OpFunctor::apply(wrapper_A(row, col), wrapper_B(row, col));
  }
}



//
/////////////////////////   matrix-vector products /////////////////////////////////
//

// A * x

/** @brief Carries out matrix-vector multiplication
*
* Implementation of the convenience expression result = prod(mat, vec);
*
* @param mat    The matrix
* @param trans  Flag whether mat is to be transposed
* @param vec    The vector
* @param result The result vector
*/
template<typename NumericT>
void prod_impl(const matrix_base<NumericT> & mat, bool trans,
               const vector_base<NumericT> & vec,
                     vector_base<NumericT> & result)
{
  typedef NumericT        value_type;

  value_type const * data_A = detail::extract_raw_pointer<value_type>(mat);
  value_type const * data_x = detail::extract_raw_pointer<value_type>(vec);
  value_type       * data_result = detail::extract_raw_pointer<value_type>(result);

  vcl_size_t A_start1 = viennacl::traits::start1(mat);
  vcl_size_t A_start2 = viennacl::traits::start2(mat);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat);
  vcl_size_t A_size1  = viennacl::traits::size1(mat);
  vcl_size_t A_size2  = viennacl::traits::size2(mat);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat);

  vcl_size_t start1 = viennacl::traits::start(vec);
  vcl_size_t inc1   = viennacl::traits::stride(vec);

  vcl_size_t start2 = viennacl::traits::start(result);
  vcl_size_t inc2   = viennacl::traits::stride(result);

  if (mat.row_major())
  {
    if (trans)
    {
      vcl_size_t thread_count = 1;
#ifdef VIENNACL_WITH_OPENMP
      if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
        thread_count = omp_get_max_threads();
#endif
      std::vector<value_type> temp_array(A_size2*thread_count, 0);
      detail::vector_array_wrapper<value_type> wrapper_res(data_result, start2, inc2);

      for (vcl_size_t col = 0; col < A_size2; ++col)
        wrapper_res(col) = 0;

#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      {
        vcl_size_t id = 0;
#ifdef VIENNACL_WITH_OPENMP
        if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
          id = omp_get_thread_num();
 #endif
        vcl_size_t begin = (A_size1 * id) / thread_count;
        vcl_size_t end   = (A_size1 * (id + 1)) / thread_count;

        detail::matrix_array_wrapper<value_type const, row_major, false> wrapper_mat(data_A, A_start1 + A_inc1 * begin, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::vector_array_wrapper<value_type const> wrapper_vec(data_x, start1 + inc1 * begin, inc1);

        for (vcl_size_t row = 0; row < (end - begin); ++row)  //run through matrix sequentially
        {
          value_type temp = wrapper_vec(row);
          for (vcl_size_t col = 0; col < A_size2; ++col)
            temp_array[A_size2 * id + col] += wrapper_mat(row , col) * temp;
        }
      }
      for (vcl_size_t id = 0; id < thread_count; ++id)
        for (vcl_size_t col = 0; col < A_size2; ++col)
          wrapper_res(col) += temp_array[A_size2 * id + col];
    }

    else
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
      {
        value_type temp = 0;
        for (vcl_size_t col = 0; col < A_size2; ++col)
          temp += data_A[viennacl::row_major::mem_index(static_cast<vcl_size_t>(row) * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] * data_x[col * inc1 + start1];

        data_result[static_cast<vcl_size_t>(row) * inc2 + start2] = temp;
      }
    }
  }
  else
  {
    if (!trans)
    {
      vcl_size_t thread_count = 1;
#ifdef VIENNACL_WITH_OPENMP
      if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
        thread_count = omp_get_max_threads();
#endif
      std::vector<value_type> temp_array(A_size1*thread_count, 0);
      detail::vector_array_wrapper<value_type> wrapper_res(data_result, start2, inc2);

      for (vcl_size_t row = 0; row < A_size1; ++row)
        wrapper_res(row) = 0;

#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      {
        vcl_size_t id = 0;
#ifdef VIENNACL_WITH_OPENMP
        if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
          id = omp_get_thread_num();
 #endif
        vcl_size_t begin = (A_size2 * id) / thread_count;
        vcl_size_t end   = (A_size2 * (id + 1)) / thread_count;

        detail::matrix_array_wrapper<value_type const, column_major, false> wrapper_mat(data_A, A_start1, A_start2 + A_inc2 * begin, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
        detail::vector_array_wrapper<value_type const> wrapper_vec(data_x, start1 + inc1 * begin, inc1);

        for (vcl_size_t col = 0; col < (end - begin); ++col)  //run through matrix sequentially
        {
          value_type temp = wrapper_vec(col);
          for (vcl_size_t row = 0; row < A_size1; ++row)
            temp_array[A_size1 * id + row] += wrapper_mat(row , col) * temp;
        }
      }
      for (vcl_size_t id = 0; id < thread_count; ++id)
        for (vcl_size_t row = 0; row < A_size1; ++row)
          wrapper_res(row) += temp_array[A_size1 * id + row];
    }
    else
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size2); ++row)
      {
        value_type temp = 0;
        for (vcl_size_t col = 0; col < A_size1; ++col)
          temp += data_A[viennacl::column_major::mem_index(col * A_inc1 + A_start1, static_cast<vcl_size_t>(row) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] * data_x[col * inc1 + start1];

        data_result[static_cast<vcl_size_t>(row) * inc2 + start2] = temp;
      }
    }
  }
}



//
/////////////////////////   matrix-matrix products /////////////////////////////////
//

namespace detail
{
  template<typename MatrixAccT1, typename MatrixAccT2, typename MatrixAccT3, typename NumericT>
  void prod(MatrixAccT1 & A, MatrixAccT2 & B, MatrixAccT3 & C,
            vcl_size_t C_size1, vcl_size_t C_size2, vcl_size_t A_size2,
            NumericT alpha, NumericT beta)
  {
    if (C_size1 == 0 || C_size2 == 0 || A_size2 == 0)
      return;

    static const vcl_size_t blocksize = 64;

    vcl_size_t num_blocks_C1 = (C_size1 - 1) / blocksize + 1;
    vcl_size_t num_blocks_C2 = (C_size2 - 1) / blocksize + 1;
    vcl_size_t num_blocks_A2 = (A_size2 - 1) / blocksize + 1;

    //
    // outer loop pair: Run over all blocks with indices (block_idx_i, block_idx_j) of the result matrix C:
    //
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((C_size1*C_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
    for (long block_idx_i2=0; block_idx_i2<static_cast<long>(num_blocks_C1); ++block_idx_i2)
    {
      // thread-local auxiliary buffers
      std::vector<NumericT> buffer_A(blocksize * blocksize); // row-major
      std::vector<NumericT> buffer_B(blocksize * blocksize); // column-major
      std::vector<NumericT> buffer_C(blocksize * blocksize); // row-major

      vcl_size_t block_idx_i = static_cast<vcl_size_t>(block_idx_i2);
      for (vcl_size_t block_idx_j=0; block_idx_j<num_blocks_C2; ++block_idx_j)
      {
        // Reset block matrix:
        std::fill(buffer_C.begin(), buffer_C.end(), NumericT(0));

        vcl_size_t offset_i = block_idx_i*blocksize;
        vcl_size_t offset_j = block_idx_j*blocksize;

        //  C(block_idx_i, block_idx_i) += A(block_idx_i, block_idx_k) * B(block_idx_k, block_idx_j)
        for (vcl_size_t block_idx_k=0; block_idx_k<num_blocks_A2; ++block_idx_k)
        {
          // flush buffers:
          std::fill(buffer_A.begin(), buffer_A.end(), NumericT(0));
          std::fill(buffer_B.begin(), buffer_B.end(), NumericT(0));

          vcl_size_t offset_k = block_idx_k*blocksize;

          // load current data:
          for (vcl_size_t i = offset_i; i < std::min(offset_i + blocksize, C_size1); ++i)
            for (vcl_size_t k = offset_k; k < std::min(offset_k + blocksize, A_size2); ++k)
              buffer_A[(i - offset_i) * blocksize + (k - offset_k)] = A(i, k);

          for (vcl_size_t j = offset_j; j < std::min(offset_j + blocksize, C_size2); ++j)
            for (vcl_size_t k = offset_k; k < std::min(offset_k + blocksize, A_size2); ++k)
              buffer_B[(k - offset_k) + (j - offset_j) * blocksize] = B(k, j);

          // multiply (this is the hot spot in terms of flops)
          for (vcl_size_t i = 0; i < blocksize; ++i)
          {
            NumericT const * ptrA = &(buffer_A[i*blocksize]);
            for (vcl_size_t j = 0; j < blocksize; ++j)
            {
              NumericT const * ptrB = &(buffer_B[j*blocksize]);

              NumericT temp = NumericT(0);
              for (vcl_size_t k = 0; k < blocksize; ++k)
                temp += ptrA[k] * ptrB[k];  // buffer_A[i*blocksize + k] * buffer_B[k + j*blocksize];

              buffer_C[i*blocksize + j] += temp;
            }
          }
        }

        // write result:
        if (beta > 0 || beta < 0)
        {
          for (vcl_size_t i = offset_i; i < std::min(offset_i + blocksize, C_size1); ++i)
            for (vcl_size_t j = offset_j; j < std::min(offset_j + blocksize, C_size2); ++j)
              C(i,j) = beta * C(i,j) + alpha * buffer_C[(i - offset_i) * blocksize + (j - offset_j)];
        }
        else
        {
          for (vcl_size_t i = offset_i; i < std::min(offset_i + blocksize, C_size1); ++i)
            for (vcl_size_t j = offset_j; j < std::min(offset_j + blocksize, C_size2); ++j)
              C(i,j) =                 alpha * buffer_C[(i - offset_i) * blocksize + (j - offset_j)];
        }

      } // for block j
    } // for block i

  } // prod()

} // namespace detail

/** @brief Carries out matrix-matrix multiplication
*
* Implementation of C = prod(A, B);
*
*/
template<typename NumericT, typename ScalarT1, typename ScalarT2 >
void prod_impl(const matrix_base<NumericT> & A, bool trans_A,
               const matrix_base<NumericT> & B, bool trans_B,
                     matrix_base<NumericT> & C,
               ScalarT1 alpha,
               ScalarT2 beta)
{
  typedef NumericT        value_type;

  value_type const * data_A = detail::extract_raw_pointer<value_type>(A);
  value_type const * data_B = detail::extract_raw_pointer<value_type>(B);
  value_type       * data_C = detail::extract_raw_pointer<value_type>(C);

  vcl_size_t A_start1 = viennacl::traits::start1(A);
  vcl_size_t A_start2 = viennacl::traits::start2(A);
  vcl_size_t A_inc1   = viennacl::traits::stride1(A);
  vcl_size_t A_inc2   = viennacl::traits::stride2(A);
  vcl_size_t A_size1  = viennacl::traits::size1(A);
  vcl_size_t A_size2  = viennacl::traits::size2(A);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

  vcl_size_t B_start1 = viennacl::traits::start1(B);
  vcl_size_t B_start2 = viennacl::traits::start2(B);
  vcl_size_t B_inc1   = viennacl::traits::stride1(B);
  vcl_size_t B_inc2   = viennacl::traits::stride2(B);
  vcl_size_t B_internal_size1  = viennacl::traits::internal_size1(B);
  vcl_size_t B_internal_size2  = viennacl::traits::internal_size2(B);

  vcl_size_t C_start1 = viennacl::traits::start1(C);
  vcl_size_t C_start2 = viennacl::traits::start2(C);
  vcl_size_t C_inc1   = viennacl::traits::stride1(C);
  vcl_size_t C_inc2   = viennacl::traits::stride2(C);
  vcl_size_t C_size1  = viennacl::traits::size1(C);
  vcl_size_t C_size2  = viennacl::traits::size2(C);
  vcl_size_t C_internal_size1  = viennacl::traits::internal_size1(C);
  vcl_size_t C_internal_size2  = viennacl::traits::internal_size2(C);

  if (!trans_A && !trans_B)
  {
    if (A.row_major() && B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && !B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && !B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && !B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else
    {
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
  }
  else if (!trans_A && trans_B)
  {
    if (A.row_major() && B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && !B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && !B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && !B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else
    {
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size2, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
  }
  else if (trans_A && !trans_B)
  {
    if (A.row_major() && B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && !B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && !B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && !B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else
    {
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, false>   wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
  }
  else if (trans_A && trans_B)
  {
    if (A.row_major() && B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, row_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, row_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const,    row_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const,    row_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && !B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const,    row_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,          row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (A.row_major() && !B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const,    row_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const,    row_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,          row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && B.row_major() && !C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const,    row_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else if (!A.row_major() && !B.row_major() && C.row_major())
    {
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,          row_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
    else
    {
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_A(data_A, A_start1, A_start2, A_inc1, A_inc2, A_internal_size1, A_internal_size2);
      detail::matrix_array_wrapper<value_type const, column_major, true>    wrapper_B(data_B, B_start1, B_start2, B_inc1, B_inc2, B_internal_size1, B_internal_size2);
      detail::matrix_array_wrapper<value_type,       column_major, false>   wrapper_C(data_C, C_start1, C_start2, C_inc1, C_inc2, C_internal_size1, C_internal_size2);

      detail::prod(wrapper_A, wrapper_B, wrapper_C, C_size1, C_size2, A_size1, static_cast<value_type>(alpha), static_cast<value_type>(beta));
    }
  }
}




//
/////////////////////////   miscellaneous operations /////////////////////////////////
//


/** @brief The implementation of the operation mat += alpha * vec1 * vec2^T, i.e. a scaled rank 1 update
*
* Implementation of the convenience expression result += alpha * outer_prod(vec1, vec2);
*
* @param mat1    The matrix to be updated
* @param alpha            The scaling factor (either a viennacl::scalar<>, float, or double)
* @param reciprocal_alpha Use 1/alpha instead of alpha
* @param flip_sign_alpha  Use -alpha instead of alpha
* @param vec1    The first vector
* @param vec2    The second vector
*/
template<typename NumericT, typename ScalarT>
void scaled_rank_1_update(matrix_base<NumericT> & mat1,
                          ScalarT const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
                          const vector_base<NumericT> & vec1,
                          const vector_base<NumericT> & vec2)
{
  typedef NumericT        value_type;

  value_type       * data_A  = detail::extract_raw_pointer<value_type>(mat1);
  value_type const * data_v1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_v2 = detail::extract_raw_pointer<value_type>(vec2);

  vcl_size_t A_start1 = viennacl::traits::start1(mat1);
  vcl_size_t A_start2 = viennacl::traits::start2(mat1);
  vcl_size_t A_inc1   = viennacl::traits::stride1(mat1);
  vcl_size_t A_inc2   = viennacl::traits::stride2(mat1);
  vcl_size_t A_size1  = viennacl::traits::size1(mat1);
  vcl_size_t A_size2  = viennacl::traits::size2(mat1);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(mat1);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(mat1);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);

  vcl_size_t start2 = viennacl::traits::start(vec2);
  vcl_size_t inc2   = viennacl::traits::stride(vec2);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;

  if (mat1.row_major())
  {
    if(reciprocal_alpha)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
      {
        value_type value_v1 = data_v1[static_cast<vcl_size_t>(row) * inc1 + start1] / data_alpha;
        for (vcl_size_t col = 0; col < A_size2; ++col)
          data_A[viennacl::row_major::mem_index(static_cast<vcl_size_t>(row) * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] += value_v1 * data_v2[col * inc2 + start2];
      }
    }
    else
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for (long row = 0; row < static_cast<long>(A_size1); ++row)
      {
        value_type value_v1 = data_v1[static_cast<vcl_size_t>(row) * inc1 + start1] * data_alpha;
        for (vcl_size_t col = 0; col < A_size2; ++col)
          data_A[viennacl::row_major::mem_index(static_cast<vcl_size_t>(row) * A_inc1 + A_start1, col * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] += value_v1 * data_v2[col * inc2 + start2];
      }
    }
  }
  else
  {
      if(reciprocal_alpha)
      {
#ifdef VIENNACL_WITH_OPENMP
        #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
        for (long col = 0; col < static_cast<long>(A_size2); ++col)  //run through matrix sequentially
        {
          value_type value_v2 = data_v2[static_cast<vcl_size_t>(col) * inc2 + start2] / data_alpha;
          for (vcl_size_t row = 0; row < A_size1; ++row)
            data_A[viennacl::column_major::mem_index(row * A_inc1 + A_start1, static_cast<vcl_size_t>(col) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] += data_v1[row * inc1 + start1] * value_v2;
        }
      }
      else
      {
#ifdef VIENNACL_WITH_OPENMP
        #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
        for (long col = 0; col < static_cast<long>(A_size2); ++col)  //run through matrix sequentially
        {
          value_type value_v2 = data_v2[static_cast<vcl_size_t>(col) * inc2 + start2] * data_alpha;
          for (vcl_size_t row = 0; row < A_size1; ++row)
            data_A[viennacl::column_major::mem_index(row * A_inc1 + A_start1, static_cast<vcl_size_t>(col) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] += data_v1[row * inc1 + start1] * value_v2;
        }
      }
  }
}


/** @brief This function stores the diagonal and the superdiagonal of a matrix in two vectors.
*
*
* @param A    The matrix from which the vectors will be extracted of.
* @param D    The vector in which the diagonal of the matrix will be stored in.
* @param S    The vector in which the superdiagonal of the matrix will be stored in.
*/
template <typename NumericT, typename S1>
 void bidiag_pack_impl(matrix_base<NumericT> & A,
                  vector_base<S1> & D,
                  vector_base<S1> & S
                  )

 {
   typedef NumericT        value_type;

   value_type * data_A  = detail::extract_raw_pointer<value_type>(A);
   value_type * data_D = detail::extract_raw_pointer<value_type>(D);
   value_type * data_S = detail::extract_raw_pointer<value_type>(S);

   vcl_size_t A_start1 = viennacl::traits::start1(A);
   vcl_size_t A_start2 = viennacl::traits::start2(A);
   vcl_size_t A_inc1   = viennacl::traits::stride1(A);
   vcl_size_t A_inc2   = viennacl::traits::stride2(A);
   vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
   vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

   vcl_size_t start1 = viennacl::traits::start(D);
   vcl_size_t inc1   = viennacl::traits::stride(D);
   vcl_size_t size1  = viennacl::traits::size(D);

   vcl_size_t start2 = viennacl::traits::start(S);
   vcl_size_t inc2   = viennacl::traits::stride(S);
   vcl_size_t size2  = viennacl::traits::size(S);

   vcl_size_t size = std::min(size1, size2);
   if (A.row_major())
   {
#ifdef VIENNACL_WITH_OPENMP
     #pragma omp parallel for if ((size1*size1) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
     for(long i2 = 0;  i2 < long(size) - 1; i2++)
     {
       vcl_size_t i = vcl_size_t(i2);
       data_D[start1 + inc1 * i] =        data_A[viennacl::row_major::mem_index(i * A_inc1 + A_start1, i * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
       data_S[start2 + inc2 * (i + 1)] =  data_A[viennacl::row_major::mem_index(i * A_inc1 + A_start1, (i + 1) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
     }
     data_D[start1 + inc1 * (size-1)] = data_A[viennacl::row_major::mem_index((size-1) * A_inc1 + A_start1, (size-1) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];

    }
   else
   {
#ifdef VIENNACL_WITH_OPENMP
     #pragma omp parallel for if ((size1*size1) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
     for(long i2 = 0;  i2 < long(size) - 1; i2++)
     {
       vcl_size_t i = vcl_size_t(i2);
       data_D[start1 + inc1 * i] =        data_A[viennacl::column_major::mem_index(i * A_inc1 + A_start1, i * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
       data_S[start2 + inc2 * (i + 1)] =  data_A[viennacl::column_major::mem_index(i * A_inc1 + A_start1, (i + 1) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
     }
     data_D[start1 + inc1 * (size-1)] = data_A[viennacl::column_major::mem_index((size-1) * A_inc1 + A_start1, (size-1) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
   }

 }



 template <typename NumericT, typename VectorType>
  void bidiag_pack(matrix_base<NumericT> & A,
                   VectorType & dh,
                   VectorType & sh
                  )
  {

    viennacl::linalg::host_based::bidiag_pack_impl(A, dh, sh);

  }

  /** @brief This function applies a householder transformation to a matrix. A <- P * A with a householder reflection P
  *
  * @param A       The matrix to be updated.
  * @param D       The normalized householder vector.
  * @param start   The repetition counter.
  */
 template <typename NumericT>
 void house_update_A_left(matrix_base<NumericT>& A,
                          vector_base<NumericT> & D,
                          vcl_size_t start)
 {
   typedef NumericT        value_type;
   NumericT ss = 0;
   vcl_size_t row_start = start + 1;

   value_type * data_A  = detail::extract_raw_pointer<value_type>(A);
   value_type * data_D = detail::extract_raw_pointer<value_type>(D);

   vcl_size_t A_start1 = viennacl::traits::start1(A);
   vcl_size_t A_start2 = viennacl::traits::start2(A);
   vcl_size_t A_inc1   = viennacl::traits::stride1(A);
   vcl_size_t A_inc2   = viennacl::traits::stride2(A);
   vcl_size_t A_size1  = viennacl::traits::size1(A);
   vcl_size_t A_size2  = viennacl::traits::size2(A);
   vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
   vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

   vcl_size_t start1 = viennacl::traits::start(D);
   vcl_size_t inc1   = viennacl::traits::stride(D);

   if (A.row_major())
     {
       for(vcl_size_t i = 0; i < A_size2; i++)
         {
           ss = 0;
           for(vcl_size_t j = row_start; j < A_size1; j++)
               ss = ss + data_D[start1 + inc1 * j] * data_A[viennacl::row_major::mem_index((j) * A_inc1 + A_start1, (i) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
#ifdef VIENNACL_WITH_OPENMP
           #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
           for(long j = static_cast<long>(row_start); j < static_cast<long>(A_size1); j++)
               data_A[viennacl::row_major::mem_index(static_cast<vcl_size_t>(j) * A_inc1 + A_start1, (i) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] =
                   data_A[viennacl::row_major::mem_index(static_cast<vcl_size_t>(j) * A_inc1 + A_start1, (i) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] -
                   (2 * data_D[start1 + inc1 * static_cast<vcl_size_t>(j)]* ss);
         }
     }
   else
     {
       for(vcl_size_t i = 0; i < A_size2; i++)
         {
           ss = 0;
           for(vcl_size_t j = row_start; j < A_size1; j++)
               ss = ss + data_D[start1 + inc1 * j] * data_A[viennacl::column_major::mem_index((j) * A_inc1 + A_start1, (i) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
#ifdef VIENNACL_WITH_OPENMP
           #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
           for(long j = static_cast<long>(row_start); j < static_cast<long>(A_size1); j++)
               data_A[viennacl::column_major::mem_index(static_cast<vcl_size_t>(j) * A_inc1 + A_start1, (i) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]=
                   data_A[viennacl::column_major::mem_index(static_cast<vcl_size_t>(j) * A_inc1 + A_start1, (i) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] -
                   (2 * data_D[start1 + inc1 * static_cast<vcl_size_t>(j)]* ss);
         }
     }

 }

 /** @brief This function applies a householder transformation to a matrix: A <- A * P with a householder reflection P
 *
 *
 * @param A        The matrix to be updated.
 * @param D        The normalized householder vector.
 */
 template <typename NumericT>
 void house_update_A_right(matrix_base<NumericT> & A,
                           vector_base<NumericT> & D)
 {
   typedef NumericT        value_type;
   NumericT ss = 0;

   value_type * data_A  = detail::extract_raw_pointer<value_type>(A);
   value_type * data_D = detail::extract_raw_pointer<value_type>(D);

   vcl_size_t A_start1 = viennacl::traits::start1(A);
   vcl_size_t A_start2 = viennacl::traits::start2(A);
   vcl_size_t A_inc1   = viennacl::traits::stride1(A);
   vcl_size_t A_inc2   = viennacl::traits::stride2(A);
   vcl_size_t A_size1  = viennacl::traits::size1(A);
   vcl_size_t A_size2  = viennacl::traits::size2(A);
   vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
   vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

   vcl_size_t start1 = viennacl::traits::start(D);
   vcl_size_t inc1   = viennacl::traits::stride(D);

   if (A.row_major())
     {
       for(vcl_size_t i = 0; i < A_size1; i++)
         {
           ss = 0;
           for(vcl_size_t j = 0; j < A_size2; j++) // ss = ss + D[j] * A(i, j)
               ss = ss + (data_D[start1 + inc1 * j] * data_A[viennacl::row_major::mem_index((i) * A_inc1 + A_start1, (j) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]);

           NumericT sum_Av = ss;
#ifdef VIENNACL_WITH_OPENMP
           #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
           for(long j = 0; j < static_cast<long>(A_size2); j++) // A(i, j) = A(i, j) - 2 * D[j] * sum_Av
               data_A[viennacl::row_major::mem_index((i) * A_inc1 + A_start1, static_cast<vcl_size_t>(j) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]  =
                   data_A[viennacl::row_major::mem_index((i) * A_inc1 + A_start1, static_cast<vcl_size_t>(j) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] - (2 * data_D[start1 + inc1 * static_cast<vcl_size_t>(j)] * sum_Av);
         }
     }
   else
     {
       for(vcl_size_t i = 0; i < A_size1; i++)
         {
           ss = 0;
           for(vcl_size_t j = 0; j < A_size2; j++) // ss = ss + D[j] * A(i, j)
               ss = ss + (data_D[start1 + inc1 * j] * data_A[viennacl::column_major::mem_index((i) * A_inc1 + A_start1, (j) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]);

           NumericT sum_Av = ss;
#ifdef VIENNACL_WITH_OPENMP
           #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
           for(long j = 0; j < static_cast<long>(A_size2); j++) // A(i, j) = A(i, j) - 2 * D[j] * sum_Av
               data_A[viennacl::column_major::mem_index((i) * A_inc1 + A_start1, static_cast<vcl_size_t>(j) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)]  =
                   data_A[viennacl::column_major::mem_index((i) * A_inc1 + A_start1, static_cast<vcl_size_t>(j) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)] - (2 * data_D[start1 + inc1 * static_cast<vcl_size_t>(j)] * sum_Av);
         }
     }


 }

 /** @brief This function updates the matrix Q, which is needed for the computation of the eigenvectors.
 *
 * @param Q        The matrix to be updated.
 * @param D        The householder vector.
 * @param A_size1  size1 of matrix A
 */
 template <typename NumericT>
 void house_update_QL(matrix_base<NumericT>& Q,
                      vector_base<NumericT> & D,
                      vcl_size_t A_size1)

 {
   NumericT beta = 2;
   viennacl::matrix<NumericT> vcl_P = viennacl::identity_matrix<NumericT>(A_size1);
   viennacl::matrix<NumericT> Q_temp = Q;
   viennacl::vector<NumericT> vcl_D = D;


   viennacl::linalg::host_based::scaled_rank_1_update(vcl_P, beta, 1, 0, 1, vcl_D, vcl_D);
   Q = viennacl::linalg::prod(Q_temp, vcl_P);

 }

 /** @brief This function updates the matrix Q. It is part of the tql2 algorithm.
 *
 *
 * @param Q       The matrix to be updated.
 * @param tmp1    Vector with data from the tql2 algorithm.
 * @param tmp2    Vector with data from the tql2 algorithm.
 * @param l       Data from the tql2 algorithm.
 * @param m       Data from the tql2 algorithm.
 */
 template<typename NumericT>
   void givens_next(matrix_base<NumericT>& Q,
                   vector_base<NumericT> & tmp1,
                   vector_base<NumericT> & tmp2,
                   int l,
                   int m
                 )
   {
     typedef NumericT        value_type;

     value_type * data_Q  = detail::extract_raw_pointer<value_type>(Q);
     value_type * data_tmp1 = detail::extract_raw_pointer<value_type>(tmp1);
     value_type * data_tmp2 = detail::extract_raw_pointer<value_type>(tmp2);

     vcl_size_t Q_start1 = viennacl::traits::start1(Q);
     vcl_size_t Q_start2 = viennacl::traits::start2(Q);
     vcl_size_t Q_inc1   = viennacl::traits::stride1(Q);
     vcl_size_t Q_inc2   = viennacl::traits::stride2(Q);
     vcl_size_t Q_size1  = viennacl::traits::size1(Q);
     vcl_size_t Q_internal_size1  = viennacl::traits::internal_size1(Q);
     vcl_size_t Q_internal_size2  = viennacl::traits::internal_size2(Q);

     vcl_size_t start1 = viennacl::traits::start(tmp1);
     vcl_size_t inc1   = viennacl::traits::stride(tmp1);

     vcl_size_t start2 = viennacl::traits::start(tmp2);
     vcl_size_t inc2   = viennacl::traits::stride(tmp2);

     if (Q.row_major())
     {
         for( int i = m - 1; i >= l; i--)
           {
#ifdef VIENNACL_WITH_OPENMP
             #pragma omp parallel for if ((Q_size1*Q_size1) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
             for(long k = 0; k < static_cast<long>(Q_size1); k++)
               {

                 // h = data_Q(k, i+1);
                 NumericT h = data_Q[viennacl::row_major::mem_index(static_cast<vcl_size_t>(k) * Q_inc1 + Q_start1, vcl_size_t(i + 1) * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)];

                 // Q(k, i+1) = tmp2[i] * Q(k, i) + tmp1[i]*h;
                 data_Q[viennacl::row_major::mem_index(static_cast<vcl_size_t>(k) * Q_inc1 + Q_start1, vcl_size_t(i + 1) * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] = data_tmp2[start2 + inc2 * vcl_size_t(i)] *
                     data_Q[viennacl::row_major::mem_index(static_cast<vcl_size_t>(k)  * Q_inc1 + Q_start1, vcl_size_t(i) * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] + data_tmp1[start1 + inc1 * vcl_size_t(i)] * h;

                 // Q(k,   i) = tmp1[i] * Q(k, i) - tmp2[i]*h;
                 data_Q[viennacl::row_major::mem_index(static_cast<vcl_size_t>(k)  * Q_inc1 + Q_start1, vcl_size_t(i) * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] = data_tmp1[start1 + inc1 * vcl_size_t(i)] *
                     data_Q[viennacl::row_major::mem_index(static_cast<vcl_size_t>(k)  * Q_inc1 + Q_start1, vcl_size_t(i) * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] - data_tmp2[start2 + inc2 * vcl_size_t(i)]*h;
               }
           }
     }
     else       // column_major
       {
         for( int i = m - 1; i >= l; i--)
           {
#ifdef VIENNACL_WITH_OPENMP
             #pragma omp parallel for if ((Q_size1*Q_size1) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
             for(long k = 0; k < static_cast<long>(Q_size1); k++)
               {

                  // h = data_Q(k, i+1);
                  NumericT h = data_Q[viennacl::column_major::mem_index(static_cast<vcl_size_t>(k) * Q_inc1 + Q_start1, vcl_size_t(i + 1) * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)];

                   // Q(k, i+1) = tmp2[i] * Q(k, i) + tmp1[i]*h;
                  data_Q[viennacl::column_major::mem_index(static_cast<vcl_size_t>(k) * Q_inc1 + Q_start1, vcl_size_t(i + 1) * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] = data_tmp2[start2 + inc2 * vcl_size_t(i)] *
                      data_Q[viennacl::column_major::mem_index(static_cast<vcl_size_t>(k)  * Q_inc1 + Q_start1, vcl_size_t(i) * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] + data_tmp1[start1 + inc1 * vcl_size_t(i)] * h;

                  // Q(k,   i) = tmp1[i] * Q(k, i) - tmp2[i]*h;
                  data_Q[viennacl::column_major::mem_index(static_cast<vcl_size_t>(k)  * Q_inc1 + Q_start1, vcl_size_t(i) * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] = data_tmp1[start1 + inc1 * vcl_size_t(i)] *
                      data_Q[viennacl::column_major::mem_index(static_cast<vcl_size_t>(k)  * Q_inc1 + Q_start1, vcl_size_t(i) * Q_inc2 + Q_start2, Q_internal_size1, Q_internal_size2)] - data_tmp2[start2 + inc2 * vcl_size_t(i)]*h;
               }
           }
       }

   }


/** @brief This function copies a row or a column from a matrix to a vector.
*
*
* @param A          The matrix where to copy from.
* @param V          The vector to fill with data.
* @param row_start  The number of the first row to copy.
* @param col_start  The number of the first column to copy.
* @param copy_col   Set to TRUE to copy a column, FALSE to copy a row.
*/
template <typename NumericT, typename S1>
void copy_vec(matrix_base<NumericT>& A, vector_base<S1> & V,
              vcl_size_t row_start, vcl_size_t col_start, bool copy_col)
{
  typedef NumericT        value_type;

  value_type * data_A  = detail::extract_raw_pointer<value_type>(A);
  value_type * data_V = detail::extract_raw_pointer<value_type>(V);

  vcl_size_t A_start1 = viennacl::traits::start1(A);
  vcl_size_t A_start2 = viennacl::traits::start2(A);
  vcl_size_t A_inc1   = viennacl::traits::stride1(A);
  vcl_size_t A_inc2   = viennacl::traits::stride2(A);
  vcl_size_t A_size1  = viennacl::traits::size1(A);
  vcl_size_t A_size2  = viennacl::traits::size1(A);
  vcl_size_t A_internal_size1  = viennacl::traits::internal_size1(A);
  vcl_size_t A_internal_size2  = viennacl::traits::internal_size2(A);

  if(copy_col)
  {
    if (A.row_major())
    {
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for(long i = static_cast<long>(row_start); i < static_cast<long>(A_size1); i++)
      {
        data_V[i - static_cast<long>(row_start)] = data_A[viennacl::row_major::mem_index(static_cast<vcl_size_t>(i) * A_inc1 + A_start1, col_start * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
      }
    }
    else
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for(long i = static_cast<long>(row_start); i < static_cast<long>(A_size1); i++)
      {
        data_V[i - static_cast<long>(row_start)] = data_A[viennacl::column_major::mem_index(static_cast<vcl_size_t>(i) * A_inc1 + A_start1, col_start * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
      }
    }
  }
  else
  {
    if (A.row_major())
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for(long i = static_cast<long>(col_start); i < static_cast<long>(A_size2); i++)
      {
        data_V[i - static_cast<long>(col_start)] = data_A[viennacl::row_major::mem_index(row_start * A_inc1 + A_start1, static_cast<vcl_size_t>(i) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
      }
    }
    else
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if ((A_size1*A_size2) > VIENNACL_OPENMP_MATRIX_MIN_SIZE)
#endif
      for(long i = static_cast<long>(col_start); i < static_cast<long>(A_size2); i++)
      {
        data_V[i - static_cast<long>(col_start)] = data_A[viennacl::column_major::mem_index(row_start * A_inc1 + A_start1, static_cast<vcl_size_t>(i) * A_inc2 + A_start2, A_internal_size1, A_internal_size2)];
      }
    }
  }
}

} // namespace host_based
} //namespace linalg
} //namespace viennacl

#endif
