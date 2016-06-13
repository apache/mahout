#ifndef VIENNACL_LINALG_HOST_BASED_SPARSE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_SPARSE_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/host_based/sparse_matrix_operations.hpp
    @brief Implementations of operations using sparse matrices on the CPU using a single thread or OpenMP.
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/host_based/common.hpp"
#include "viennacl/linalg/host_based/vector_operations.hpp"

#include "viennacl/linalg/host_based/spgemm_vector.hpp"

#include <vector>

#ifdef VIENNACL_WITH_OPENMP
#include <omp.h>
#endif

namespace viennacl
{
namespace linalg
{
namespace host_based
{
//
// Compressed matrix
//

namespace detail
{
  template<typename NumericT, unsigned int AlignmentV>
  void row_info(compressed_matrix<NumericT, AlignmentV> const & mat,
                vector_base<NumericT> & vec,
                viennacl::linalg::detail::row_info_types info_selector)
  {
    NumericT         * result_buf = detail::extract_raw_pointer<NumericT>(vec.handle());
    NumericT   const * elements   = detail::extract_raw_pointer<NumericT>(mat.handle());
    unsigned int const * row_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle1());
    unsigned int const * col_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle2());

    for (vcl_size_t row = 0; row < mat.size1(); ++row)
    {
      NumericT value = 0;
      unsigned int row_end = row_buffer[row+1];

      switch (info_selector)
      {
        case viennacl::linalg::detail::SPARSE_ROW_NORM_INF: //inf-norm
          for (unsigned int i = row_buffer[row]; i < row_end; ++i)
            value = std::max<NumericT>(value, std::fabs(elements[i]));
          break;

        case viennacl::linalg::detail::SPARSE_ROW_NORM_1: //1-norm
          for (unsigned int i = row_buffer[row]; i < row_end; ++i)
            value += std::fabs(elements[i]);
          break;

        case viennacl::linalg::detail::SPARSE_ROW_NORM_2: //2-norm
          for (unsigned int i = row_buffer[row]; i < row_end; ++i)
            value += elements[i] * elements[i];
          value = std::sqrt(value);
          break;

        case viennacl::linalg::detail::SPARSE_ROW_DIAGONAL: //diagonal entry
          for (unsigned int i = row_buffer[row]; i < row_end; ++i)
          {
            if (col_buffer[i] == row)
            {
              value = elements[i];
              break;
            }
          }
          break;
      }
      result_buf[row] = value;
    }
  }
}


/** @brief Carries out matrix-vector multiplication with a compressed_matrix
*
* Implementation of the convenience expression result = prod(mat, vec);
*
* @param mat    The matrix
* @param vec    The vector
* @param result The result vector
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::compressed_matrix<NumericT, AlignmentV> & mat,
               const viennacl::vector_base<NumericT> & vec,
               NumericT alpha,
               viennacl::vector_base<NumericT> & result,
               NumericT beta)
{
  NumericT           * result_buf = detail::extract_raw_pointer<NumericT>(result.handle());
  NumericT     const * vec_buf    = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements   = detail::extract_raw_pointer<NumericT>(mat.handle());
  unsigned int const * row_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle1());
  unsigned int const * col_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle2());

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long row = 0; row < static_cast<long>(mat.size1()); ++row)
  {
    NumericT dot_prod = 0;
    vcl_size_t row_end = row_buffer[row+1];
    for (vcl_size_t i = row_buffer[row]; i < row_end; ++i)
      dot_prod += elements[i] * vec_buf[col_buffer[i] * vec.stride() + vec.start()];

    if (beta < 0 || beta > 0)
    {
      vcl_size_t index = static_cast<vcl_size_t>(row) * result.stride() + result.start();
      result_buf[index] = alpha * dot_prod + beta * result_buf[index];
    }
    else
      result_buf[static_cast<vcl_size_t>(row) * result.stride() + result.start()] = alpha * dot_prod;
  }

}

/** @brief Carries out sparse_matrix-matrix multiplication first matrix being compressed
*
* Implementation of the convenience expression result = prod(sp_mat, d_mat);
*
* @param sp_mat     The sparse matrix
* @param d_mat      The dense matrix
* @param result     The result matrix
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::compressed_matrix<NumericT, AlignmentV> & sp_mat,
               const viennacl::matrix_base<NumericT> & d_mat,
                     viennacl::matrix_base<NumericT> & result) {

  NumericT     const * sp_mat_elements   = detail::extract_raw_pointer<NumericT>(sp_mat.handle());
  unsigned int const * sp_mat_row_buffer = detail::extract_raw_pointer<unsigned int>(sp_mat.handle1());
  unsigned int const * sp_mat_col_buffer = detail::extract_raw_pointer<unsigned int>(sp_mat.handle2());

  NumericT const * d_mat_data  = detail::extract_raw_pointer<NumericT>(d_mat);
  NumericT       * result_data = detail::extract_raw_pointer<NumericT>(result);

  vcl_size_t d_mat_start1 = viennacl::traits::start1(d_mat);
  vcl_size_t d_mat_start2 = viennacl::traits::start2(d_mat);
  vcl_size_t d_mat_inc1   = viennacl::traits::stride1(d_mat);
  vcl_size_t d_mat_inc2   = viennacl::traits::stride2(d_mat);
  vcl_size_t d_mat_internal_size1  = viennacl::traits::internal_size1(d_mat);
  vcl_size_t d_mat_internal_size2  = viennacl::traits::internal_size2(d_mat);

  vcl_size_t result_start1 = viennacl::traits::start1(result);
  vcl_size_t result_start2 = viennacl::traits::start2(result);
  vcl_size_t result_inc1   = viennacl::traits::stride1(result);
  vcl_size_t result_inc2   = viennacl::traits::stride2(result);
  vcl_size_t result_internal_size1  = viennacl::traits::internal_size1(result);
  vcl_size_t result_internal_size2  = viennacl::traits::internal_size2(result);

  detail::matrix_array_wrapper<NumericT const, row_major, false>
      d_mat_wrapper_row(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);
  detail::matrix_array_wrapper<NumericT const, column_major, false>
      d_mat_wrapper_col(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);

  detail::matrix_array_wrapper<NumericT, row_major, false>
      result_wrapper_row(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);
  detail::matrix_array_wrapper<NumericT, column_major, false>
      result_wrapper_col(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);

  if ( d_mat.row_major() ) {
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long row = 0; row < static_cast<long>(sp_mat.size1()); ++row) {
      vcl_size_t row_start = sp_mat_row_buffer[row];
      vcl_size_t row_end = sp_mat_row_buffer[row+1];
      for (vcl_size_t col = 0; col < d_mat.size2(); ++col) {
        NumericT temp = 0;
        for (vcl_size_t k = row_start; k < row_end; ++k) {
          temp += sp_mat_elements[k] * d_mat_wrapper_row(static_cast<vcl_size_t>(sp_mat_col_buffer[k]), col);
        }
        if (result.row_major())
          result_wrapper_row(row, col) = temp;
        else
          result_wrapper_col(row, col) = temp;
      }
    }
  }
  else {
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long col = 0; col < static_cast<long>(d_mat.size2()); ++col) {
      for (long row = 0; row < static_cast<long>(sp_mat.size1()); ++row) {
        vcl_size_t row_start = sp_mat_row_buffer[row];
        vcl_size_t row_end = sp_mat_row_buffer[row+1];
        NumericT temp = 0;
        for (vcl_size_t k = row_start; k < row_end; ++k) {
          temp += sp_mat_elements[k] * d_mat_wrapper_col(static_cast<vcl_size_t>(sp_mat_col_buffer[k]), static_cast<vcl_size_t>(col));
        }
        if (result.row_major())
          result_wrapper_row(row, col) = temp;
        else
          result_wrapper_col(row, col) = temp;
      }
    }
  }

}

/** @brief Carries out matrix-trans(matrix) multiplication first matrix being compressed
*          and the second transposed
*
* Implementation of the convenience expression result = prod(sp_mat, trans(d_mat));
*
* @param sp_mat             The sparse matrix
* @param d_mat              The transposed dense matrix
* @param result             The result matrix
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::compressed_matrix<NumericT, AlignmentV> & sp_mat,
               const viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                                  const viennacl::matrix_base<NumericT>,
                                                  viennacl::op_trans > & d_mat,
                viennacl::matrix_base<NumericT> & result) {

  NumericT     const * sp_mat_elements   = detail::extract_raw_pointer<NumericT>(sp_mat.handle());
  unsigned int const * sp_mat_row_buffer = detail::extract_raw_pointer<unsigned int>(sp_mat.handle1());
  unsigned int const * sp_mat_col_buffer = detail::extract_raw_pointer<unsigned int>(sp_mat.handle2());

  NumericT const *  d_mat_data = detail::extract_raw_pointer<NumericT>(d_mat.lhs());
  NumericT       * result_data = detail::extract_raw_pointer<NumericT>(result);

  vcl_size_t d_mat_start1 = viennacl::traits::start1(d_mat.lhs());
  vcl_size_t d_mat_start2 = viennacl::traits::start2(d_mat.lhs());
  vcl_size_t d_mat_inc1   = viennacl::traits::stride1(d_mat.lhs());
  vcl_size_t d_mat_inc2   = viennacl::traits::stride2(d_mat.lhs());
  vcl_size_t d_mat_internal_size1  = viennacl::traits::internal_size1(d_mat.lhs());
  vcl_size_t d_mat_internal_size2  = viennacl::traits::internal_size2(d_mat.lhs());

  vcl_size_t result_start1 = viennacl::traits::start1(result);
  vcl_size_t result_start2 = viennacl::traits::start2(result);
  vcl_size_t result_inc1   = viennacl::traits::stride1(result);
  vcl_size_t result_inc2   = viennacl::traits::stride2(result);
  vcl_size_t result_internal_size1  = viennacl::traits::internal_size1(result);
  vcl_size_t result_internal_size2  = viennacl::traits::internal_size2(result);

  detail::matrix_array_wrapper<NumericT const, row_major, false>
      d_mat_wrapper_row(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);
  detail::matrix_array_wrapper<NumericT const, column_major, false>
      d_mat_wrapper_col(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);

  detail::matrix_array_wrapper<NumericT, row_major, false>
      result_wrapper_row(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);
  detail::matrix_array_wrapper<NumericT, column_major, false>
      result_wrapper_col(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);

  if ( d_mat.lhs().row_major() ) {
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long row = 0; row < static_cast<long>(sp_mat.size1()); ++row) {
      vcl_size_t row_start = sp_mat_row_buffer[row];
      vcl_size_t row_end = sp_mat_row_buffer[row+1];
      for (vcl_size_t col = 0; col < d_mat.size2(); ++col) {
        NumericT temp = 0;
        for (vcl_size_t k = row_start; k < row_end; ++k) {
          temp += sp_mat_elements[k] * d_mat_wrapper_row(col, static_cast<vcl_size_t>(sp_mat_col_buffer[k]));
        }
        if (result.row_major())
          result_wrapper_row(row, col) = temp;
        else
          result_wrapper_col(row, col) = temp;
      }
    }
  }
  else {
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long col = 0; col < static_cast<long>(d_mat.size2()); ++col) {
      for (vcl_size_t row = 0; row < sp_mat.size1(); ++row) {
        vcl_size_t row_start = sp_mat_row_buffer[row];
        vcl_size_t row_end = sp_mat_row_buffer[row+1];
        NumericT temp = 0;
        for (vcl_size_t k = row_start; k < row_end; ++k) {
          temp += sp_mat_elements[k] * d_mat_wrapper_col(col, static_cast<vcl_size_t>(sp_mat_col_buffer[k]));
        }
        if (result.row_major())
          result_wrapper_row(row, col) = temp;
        else
          result_wrapper_col(row, col) = temp;
      }
    }
  }

}


/** @brief Carries out sparse_matrix-sparse_matrix multiplication for CSR matrices
*
* Implementation of the convenience expression C = prod(A, B);
* Based on computing C(i, :) = A(i, :) * B via merging the respective rows of B
*
* @param A     Left factor
* @param B     Right factor
* @param C     Result matrix
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::compressed_matrix<NumericT, AlignmentV> const & A,
               viennacl::compressed_matrix<NumericT, AlignmentV> const & B,
               viennacl::compressed_matrix<NumericT, AlignmentV> & C)
{

  NumericT     const * A_elements   = detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = detail::extract_raw_pointer<unsigned int>(A.handle2());

  NumericT     const * B_elements   = detail::extract_raw_pointer<NumericT>(B.handle());
  unsigned int const * B_row_buffer = detail::extract_raw_pointer<unsigned int>(B.handle1());
  unsigned int const * B_col_buffer = detail::extract_raw_pointer<unsigned int>(B.handle2());

  C.resize(A.size1(), B.size2(), false);
  unsigned int * C_row_buffer = detail::extract_raw_pointer<unsigned int>(C.handle1());

#if defined(VIENNACL_WITH_OPENMP)
  unsigned int block_factor = 10;
  unsigned int max_threads = omp_get_max_threads();
  long chunk_size = long(A.size1()) / long(block_factor * max_threads) + 1;
#else
  unsigned int max_threads = 1;
#endif
  std::vector<unsigned int> max_length_row_C(max_threads);
  std::vector<unsigned int *> row_C_temp_index_buffers(max_threads);
  std::vector<NumericT *>     row_C_temp_value_buffers(max_threads);


  /*
   * Stage 1: Determine maximum length of work buffers:
   */

#if defined(VIENNACL_WITH_OPENMP)
  #pragma omp parallel for schedule(dynamic, chunk_size)
#endif
  for (long i=0; i<long(A.size1()); ++i)
  {
    unsigned int row_start_A = A_row_buffer[i];
    unsigned int row_end_A   = A_row_buffer[i+1];

    unsigned int row_C_upper_bound_row = 0;
    for (unsigned int j = row_start_A; j<row_end_A; ++j)
    {
      unsigned int row_B = A_col_buffer[j];

      unsigned int entries_in_row = B_row_buffer[row_B+1] - B_row_buffer[row_B];
      row_C_upper_bound_row += entries_in_row;
    }

#ifdef VIENNACL_WITH_OPENMP
    unsigned int thread_id = omp_get_thread_num();
#else
    unsigned int thread_id = 0;
#endif

    max_length_row_C[thread_id] = std::max(max_length_row_C[thread_id], std::min(row_C_upper_bound_row, static_cast<unsigned int>(B.size2())));
  }

  // determine global maximum row length
  for (std::size_t i=1; i<max_length_row_C.size(); ++i)
    max_length_row_C[0] = std::max(max_length_row_C[0], max_length_row_C[i]);

  // allocate work vectors:
  for (unsigned int i=0; i<max_threads; ++i)
    row_C_temp_index_buffers[i] = (unsigned int *)malloc(sizeof(unsigned int)*3*max_length_row_C[0]);


  /*
   * Stage 2: Determine sparsity pattern of C
   */

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for schedule(dynamic, chunk_size)
#endif
  for (long i=0; i<long(A.size1()); ++i)
  {
    unsigned int thread_id = 0;
  #ifdef VIENNACL_WITH_OPENMP
    thread_id = omp_get_thread_num();
  #endif
    unsigned int buffer_len = max_length_row_C[0];

    unsigned int *row_C_vector_1 = row_C_temp_index_buffers[thread_id];
    unsigned int *row_C_vector_2 = row_C_vector_1 + buffer_len;
    unsigned int *row_C_vector_3 = row_C_vector_2 + buffer_len;

    unsigned int row_start_A = A_row_buffer[i];
    unsigned int row_end_A   = A_row_buffer[i+1];

    C_row_buffer[i] = row_C_scan_symbolic_vector(row_start_A, row_end_A, A_col_buffer,
                                                 B_row_buffer, B_col_buffer, static_cast<unsigned int>(B.size2()),
                                                 row_C_vector_1, row_C_vector_2, row_C_vector_3);
  }

  // exclusive scan to obtain row start indices:
  unsigned int current_offset = 0;
  for (std::size_t i=0; i<C.size1(); ++i)
  {
    unsigned int tmp = C_row_buffer[i];
    C_row_buffer[i] = current_offset;
    current_offset += tmp;
  }
  C_row_buffer[C.size1()] = current_offset;
  C.reserve(current_offset, false);

  // allocate work vectors:
  for (unsigned int i=0; i<max_threads; ++i)
    row_C_temp_value_buffers[i] = (NumericT *)malloc(sizeof(NumericT)*3*max_length_row_C[0]);

  /*
   * Stage 3: Compute product (code similar, maybe pull out into a separate function to avoid code duplication?)
   */
  NumericT     * C_elements   = detail::extract_raw_pointer<NumericT>(C.handle());
  unsigned int * C_col_buffer = detail::extract_raw_pointer<unsigned int>(C.handle2());

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for schedule(dynamic, chunk_size)
#endif
  for (long i = 0; i < long(A.size1()); ++i)
  {
    unsigned int row_start_A  = A_row_buffer[i];
    unsigned int row_end_A    = A_row_buffer[i+1];

    unsigned int row_C_buffer_start = C_row_buffer[i];
    unsigned int row_C_buffer_end   = C_row_buffer[i+1];

#ifdef VIENNACL_WITH_OPENMP
    unsigned int thread_id = omp_get_thread_num();
#else
    unsigned int thread_id = 0;
#endif

    unsigned int *row_C_vector_1 = row_C_temp_index_buffers[thread_id];
    unsigned int *row_C_vector_2 = row_C_vector_1 + max_length_row_C[0];
    unsigned int *row_C_vector_3 = row_C_vector_2 + max_length_row_C[0];

    NumericT *row_C_vector_1_values = row_C_temp_value_buffers[thread_id];
    NumericT *row_C_vector_2_values = row_C_vector_1_values + max_length_row_C[0];
    NumericT *row_C_vector_3_values = row_C_vector_2_values + max_length_row_C[0];

    row_C_scan_numeric_vector(row_start_A, row_end_A, A_col_buffer, A_elements,
                              B_row_buffer, B_col_buffer, B_elements, static_cast<unsigned int>(B.size2()),
                              row_C_buffer_start, row_C_buffer_end, C_col_buffer, C_elements,
                              row_C_vector_1, row_C_vector_1_values,
                              row_C_vector_2, row_C_vector_2_values,
                              row_C_vector_3, row_C_vector_3_values);
  }

  // clean up at the end:
  for (unsigned int i=0; i<max_threads; ++i)
  {
    free(row_C_temp_index_buffers[i]);
    free(row_C_temp_value_buffers[i]);
  }

}




//
// Triangular solve for compressed_matrix, A \ b
//
namespace detail
{
  template<typename NumericT, typename ConstScalarArrayT, typename ScalarArrayT, typename IndexArrayT>
  void csr_inplace_solve(IndexArrayT const & row_buffer,
                         IndexArrayT const & col_buffer,
                         ConstScalarArrayT const & element_buffer,
                         ScalarArrayT & vec_buffer,
                         vcl_size_t num_cols,
                         viennacl::linalg::unit_lower_tag)
  {
    vcl_size_t row_begin = row_buffer[1];
    for (vcl_size_t row = 1; row < num_cols; ++row)
    {
      NumericT vec_entry = vec_buffer[row];
      vcl_size_t row_end = row_buffer[row+1];
      for (vcl_size_t i = row_begin; i < row_end; ++i)
      {
        vcl_size_t col_index = col_buffer[i];
        if (col_index < row)
          vec_entry -= vec_buffer[col_index] * element_buffer[i];
      }
      vec_buffer[row] = vec_entry;
      row_begin = row_end;
    }
  }

  template<typename NumericT, typename ConstScalarArrayT, typename ScalarArrayT, typename IndexArrayT>
  void csr_inplace_solve(IndexArrayT const & row_buffer,
                         IndexArrayT const & col_buffer,
                         ConstScalarArrayT const & element_buffer,
                         ScalarArrayT & vec_buffer,
                         vcl_size_t num_cols,
                         viennacl::linalg::lower_tag)
  {
    vcl_size_t row_begin = row_buffer[0];
    for (vcl_size_t row = 0; row < num_cols; ++row)
    {
      NumericT vec_entry = vec_buffer[row];

      // substitute and remember diagonal entry
      vcl_size_t row_end = row_buffer[row+1];
      NumericT diagonal_entry = 0;
      for (vcl_size_t i = row_begin; i < row_end; ++i)
      {
        vcl_size_t col_index = col_buffer[i];
        if (col_index < row)
          vec_entry -= vec_buffer[col_index] * element_buffer[i];
        else if (col_index == row)
          diagonal_entry = element_buffer[i];
      }

      vec_buffer[row] = vec_entry / diagonal_entry;
      row_begin = row_end;
    }
  }


  template<typename NumericT, typename ConstScalarArrayT, typename ScalarArrayT, typename IndexArrayT>
  void csr_inplace_solve(IndexArrayT const & row_buffer,
                         IndexArrayT const & col_buffer,
                         ConstScalarArrayT const & element_buffer,
                         ScalarArrayT & vec_buffer,
                         vcl_size_t num_cols,
                         viennacl::linalg::unit_upper_tag)
  {
    for (vcl_size_t row2 = 1; row2 < num_cols; ++row2)
    {
      vcl_size_t row = (num_cols - row2) - 1;
      NumericT vec_entry = vec_buffer[row];
      vcl_size_t row_begin = row_buffer[row];
      vcl_size_t row_end   = row_buffer[row+1];
      for (vcl_size_t i = row_begin; i < row_end; ++i)
      {
        vcl_size_t col_index = col_buffer[i];
        if (col_index > row)
          vec_entry -= vec_buffer[col_index] * element_buffer[i];
      }
      vec_buffer[row] = vec_entry;
    }
  }

  template<typename NumericT, typename ConstScalarArrayT, typename ScalarArrayT, typename IndexArrayT>
  void csr_inplace_solve(IndexArrayT const & row_buffer,
                         IndexArrayT const & col_buffer,
                         ConstScalarArrayT const & element_buffer,
                         ScalarArrayT & vec_buffer,
                         vcl_size_t num_cols,
                         viennacl::linalg::upper_tag)
  {
    for (vcl_size_t row2 = 0; row2 < num_cols; ++row2)
    {
      vcl_size_t row = (num_cols - row2) - 1;
      NumericT vec_entry = vec_buffer[row];

      // substitute and remember diagonal entry
      vcl_size_t row_begin = row_buffer[row];
      vcl_size_t row_end   = row_buffer[row+1];
      NumericT diagonal_entry = 0;
      for (vcl_size_t i = row_begin; i < row_end; ++i)
      {
        vcl_size_t col_index = col_buffer[i];
        if (col_index > row)
          vec_entry -= vec_buffer[col_index] * element_buffer[i];
        else if (col_index == row)
          diagonal_entry = element_buffer[i];
      }

      vec_buffer[row] = vec_entry / diagonal_entry;
    }
  }

} //namespace detail



/** @brief Inplace solution of a lower triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
*
* @param L    The matrix
* @param vec  The vector holding the right hand side. Is overwritten by the solution.
* @param tag  The solver tag identifying the respective triangular solver
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(compressed_matrix<NumericT, AlignmentV> const & L,
                   vector_base<NumericT> & vec,
                   viennacl::linalg::unit_lower_tag tag)
{
  NumericT           * vec_buf    = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements   = detail::extract_raw_pointer<NumericT>(L.handle());
  unsigned int const * row_buffer = detail::extract_raw_pointer<unsigned int>(L.handle1());
  unsigned int const * col_buffer = detail::extract_raw_pointer<unsigned int>(L.handle2());

  detail::csr_inplace_solve<NumericT>(row_buffer, col_buffer, elements, vec_buf, L.size2(), tag);
}

/** @brief Inplace solution of a lower triangular compressed_matrix. Typically used for LU substitutions
*
* @param L    The matrix
* @param vec  The vector holding the right hand side. Is overwritten by the solution.
* @param tag  The solver tag identifying the respective triangular solver
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(compressed_matrix<NumericT, AlignmentV> const & L,
                   vector_base<NumericT> & vec,
                   viennacl::linalg::lower_tag tag)
{
  NumericT           * vec_buf    = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements   = detail::extract_raw_pointer<NumericT>(L.handle());
  unsigned int const * row_buffer = detail::extract_raw_pointer<unsigned int>(L.handle1());
  unsigned int const * col_buffer = detail::extract_raw_pointer<unsigned int>(L.handle2());

  detail::csr_inplace_solve<NumericT>(row_buffer, col_buffer, elements, vec_buf, L.size2(), tag);
}


/** @brief Inplace solution of a upper triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
*
* @param U    The matrix
* @param vec  The vector holding the right hand side. Is overwritten by the solution.
* @param tag  The solver tag identifying the respective triangular solver
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(compressed_matrix<NumericT, AlignmentV> const & U,
                   vector_base<NumericT> & vec,
                   viennacl::linalg::unit_upper_tag tag)
{
  NumericT           * vec_buf    = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements   = detail::extract_raw_pointer<NumericT>(U.handle());
  unsigned int const * row_buffer = detail::extract_raw_pointer<unsigned int>(U.handle1());
  unsigned int const * col_buffer = detail::extract_raw_pointer<unsigned int>(U.handle2());

  detail::csr_inplace_solve<NumericT>(row_buffer, col_buffer, elements, vec_buf, U.size2(), tag);
}

/** @brief Inplace solution of a upper triangular compressed_matrix. Typically used for LU substitutions
*
* @param U    The matrix
* @param vec  The vector holding the right hand side. Is overwritten by the solution.
* @param tag  The solver tag identifying the respective triangular solver
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(compressed_matrix<NumericT, AlignmentV> const & U,
                   vector_base<NumericT> & vec,
                   viennacl::linalg::upper_tag tag)
{
  NumericT           * vec_buf    = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements   = detail::extract_raw_pointer<NumericT>(U.handle());
  unsigned int const * row_buffer = detail::extract_raw_pointer<unsigned int>(U.handle1());
  unsigned int const * col_buffer = detail::extract_raw_pointer<unsigned int>(U.handle2());

  detail::csr_inplace_solve<NumericT>(row_buffer, col_buffer, elements, vec_buf, U.size2(), tag);
}







//
// Triangular solve for compressed_matrix, A^T \ b
//

namespace detail
{
  template<typename NumericT, typename ConstScalarArrayT, typename ScalarArrayT, typename IndexArrayT>
  void csr_trans_inplace_solve(IndexArrayT const & row_buffer,
                               IndexArrayT const & col_buffer,
                               ConstScalarArrayT const & element_buffer,
                               ScalarArrayT & vec_buffer,
                               vcl_size_t num_cols,
                               viennacl::linalg::unit_lower_tag)
  {
    vcl_size_t col_begin = row_buffer[0];
    for (vcl_size_t col = 0; col < num_cols; ++col)
    {
      NumericT vec_entry = vec_buffer[col];
      vcl_size_t col_end = row_buffer[col+1];
      for (vcl_size_t i = col_begin; i < col_end; ++i)
      {
        unsigned int row_index = col_buffer[i];
        if (row_index > col)
          vec_buffer[row_index] -= vec_entry * element_buffer[i];
      }
      col_begin = col_end;
    }
  }

  template<typename NumericT, typename ConstScalarArrayT, typename ScalarArrayT, typename IndexArrayT>
  void csr_trans_inplace_solve(IndexArrayT const & row_buffer,
                               IndexArrayT const & col_buffer,
                               ConstScalarArrayT const & element_buffer,
                               ScalarArrayT & vec_buffer,
                               vcl_size_t num_cols,
                               viennacl::linalg::lower_tag)
  {
    vcl_size_t col_begin = row_buffer[0];
    for (vcl_size_t col = 0; col < num_cols; ++col)
    {
      vcl_size_t col_end = row_buffer[col+1];

      // Stage 1: Find diagonal entry:
      NumericT diagonal_entry = 0;
      for (vcl_size_t i = col_begin; i < col_end; ++i)
      {
        vcl_size_t row_index = col_buffer[i];
        if (row_index == col)
        {
          diagonal_entry = element_buffer[i];
          break;
        }
      }

      // Stage 2: Substitute
      NumericT vec_entry = vec_buffer[col] / diagonal_entry;
      vec_buffer[col] = vec_entry;
      for (vcl_size_t i = col_begin; i < col_end; ++i)
      {
        vcl_size_t row_index = col_buffer[i];
        if (row_index > col)
          vec_buffer[row_index] -= vec_entry * element_buffer[i];
      }
      col_begin = col_end;
    }
  }

  template<typename NumericT, typename ConstScalarArrayT, typename ScalarArrayT, typename IndexArrayT>
  void csr_trans_inplace_solve(IndexArrayT const & row_buffer,
                               IndexArrayT const & col_buffer,
                               ConstScalarArrayT const & element_buffer,
                               ScalarArrayT & vec_buffer,
                               vcl_size_t num_cols,
                               viennacl::linalg::unit_upper_tag)
  {
    for (vcl_size_t col2 = 0; col2 < num_cols; ++col2)
    {
      vcl_size_t col = (num_cols - col2) - 1;

      NumericT vec_entry = vec_buffer[col];
      vcl_size_t col_begin = row_buffer[col];
      vcl_size_t col_end = row_buffer[col+1];
      for (vcl_size_t i = col_begin; i < col_end; ++i)
      {
        vcl_size_t row_index = col_buffer[i];
        if (row_index < col)
          vec_buffer[row_index] -= vec_entry * element_buffer[i];
      }

    }
  }

  template<typename NumericT, typename ConstScalarArrayT, typename ScalarArrayT, typename IndexArrayT>
  void csr_trans_inplace_solve(IndexArrayT const & row_buffer,
                               IndexArrayT const & col_buffer,
                               ConstScalarArrayT const & element_buffer,
                               ScalarArrayT & vec_buffer,
                               vcl_size_t num_cols,
                               viennacl::linalg::upper_tag)
  {
    for (vcl_size_t col2 = 0; col2 < num_cols; ++col2)
    {
      vcl_size_t col = (num_cols - col2) - 1;
      vcl_size_t col_begin = row_buffer[col];
      vcl_size_t col_end = row_buffer[col+1];

      // Stage 1: Find diagonal entry:
      NumericT diagonal_entry = 0;
      for (vcl_size_t i = col_begin; i < col_end; ++i)
      {
        vcl_size_t row_index = col_buffer[i];
        if (row_index == col)
        {
          diagonal_entry = element_buffer[i];
          break;
        }
      }

      // Stage 2: Substitute
      NumericT vec_entry = vec_buffer[col] / diagonal_entry;
      vec_buffer[col] = vec_entry;
      for (vcl_size_t i = col_begin; i < col_end; ++i)
      {
        vcl_size_t row_index = col_buffer[i];
        if (row_index < col)
          vec_buffer[row_index] -= vec_entry * element_buffer[i];
      }
    }
  }


  //
  // block solves
  //
  template<typename NumericT, unsigned int AlignmentV>
  void block_inplace_solve(const matrix_expression<const compressed_matrix<NumericT, AlignmentV>,
                                                   const compressed_matrix<NumericT, AlignmentV>,
                                                   op_trans> & L,
                           viennacl::backend::mem_handle const & /* block_indices */, vcl_size_t /* num_blocks */,
                           vector_base<NumericT> const & /* L_diagonal */,  //ignored
                           vector_base<NumericT> & vec,
                           viennacl::linalg::unit_lower_tag)
  {
    // Note: The following could be implemented more efficiently using the block structure and possibly OpenMP.

    unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L.lhs().handle1());
    unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L.lhs().handle2());
    NumericT     const * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(L.lhs().handle());
    NumericT           * vec_buffer = detail::extract_raw_pointer<NumericT>(vec.handle());

    vcl_size_t col_begin = row_buffer[0];
    for (vcl_size_t col = 0; col < L.lhs().size1(); ++col)
    {
      NumericT vec_entry = vec_buffer[col];
      vcl_size_t col_end = row_buffer[col+1];
      for (vcl_size_t i = col_begin; i < col_end; ++i)
      {
        unsigned int row_index = col_buffer[i];
        if (row_index > col)
          vec_buffer[row_index] -= vec_entry * elements[i];
      }
      col_begin = col_end;
    }
  }

  template<typename NumericT, unsigned int AlignmentV>
  void block_inplace_solve(const matrix_expression<const compressed_matrix<NumericT, AlignmentV>,
                                                   const compressed_matrix<NumericT, AlignmentV>,
                                                   op_trans> & L,
                           viennacl::backend::mem_handle const & /*block_indices*/, vcl_size_t /* num_blocks */,
                           vector_base<NumericT> const & L_diagonal,
                           vector_base<NumericT> & vec,
                           viennacl::linalg::lower_tag)
  {
    // Note: The following could be implemented more efficiently using the block structure and possibly OpenMP.

    unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L.lhs().handle1());
    unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L.lhs().handle2());
    NumericT     const * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(L.lhs().handle());
    NumericT     const * diagonal_buffer = detail::extract_raw_pointer<NumericT>(L_diagonal.handle());
    NumericT           * vec_buffer = detail::extract_raw_pointer<NumericT>(vec.handle());

    vcl_size_t col_begin = row_buffer[0];
    for (vcl_size_t col = 0; col < L.lhs().size1(); ++col)
    {
      vcl_size_t col_end = row_buffer[col+1];

      NumericT vec_entry = vec_buffer[col] / diagonal_buffer[col];
      vec_buffer[col] = vec_entry;
      for (vcl_size_t i = col_begin; i < col_end; ++i)
      {
        vcl_size_t row_index = col_buffer[i];
        if (row_index > col)
          vec_buffer[row_index] -= vec_entry * elements[i];
      }
      col_begin = col_end;
    }
  }



  template<typename NumericT, unsigned int AlignmentV>
  void block_inplace_solve(const matrix_expression<const compressed_matrix<NumericT, AlignmentV>,
                                                   const compressed_matrix<NumericT, AlignmentV>,
                                                   op_trans> & U,
                           viennacl::backend::mem_handle const & /*block_indices*/, vcl_size_t /* num_blocks */,
                           vector_base<NumericT> const & /* U_diagonal */, //ignored
                           vector_base<NumericT> & vec,
                           viennacl::linalg::unit_upper_tag)
  {
    // Note: The following could be implemented more efficiently using the block structure and possibly OpenMP.

    unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(U.lhs().handle1());
    unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(U.lhs().handle2());
    NumericT     const * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(U.lhs().handle());
    NumericT           * vec_buffer = detail::extract_raw_pointer<NumericT>(vec.handle());

    for (vcl_size_t col2 = 0; col2 < U.lhs().size1(); ++col2)
    {
      vcl_size_t col = (U.lhs().size1() - col2) - 1;

      NumericT vec_entry = vec_buffer[col];
      vcl_size_t col_begin = row_buffer[col];
      vcl_size_t col_end = row_buffer[col+1];
      for (vcl_size_t i = col_begin; i < col_end; ++i)
      {
        vcl_size_t row_index = col_buffer[i];
        if (row_index < col)
          vec_buffer[row_index] -= vec_entry * elements[i];
      }

    }
  }

  template<typename NumericT, unsigned int AlignmentV>
  void block_inplace_solve(const matrix_expression<const compressed_matrix<NumericT, AlignmentV>,
                                                   const compressed_matrix<NumericT, AlignmentV>,
                                                   op_trans> & U,
                           viennacl::backend::mem_handle const & /* block_indices */, vcl_size_t /* num_blocks */,
                           vector_base<NumericT> const & U_diagonal,
                           vector_base<NumericT> & vec,
                           viennacl::linalg::upper_tag)
  {
    // Note: The following could be implemented more efficiently using the block structure and possibly OpenMP.

    unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(U.lhs().handle1());
    unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(U.lhs().handle2());
    NumericT     const * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(U.lhs().handle());
    NumericT     const * diagonal_buffer = detail::extract_raw_pointer<NumericT>(U_diagonal.handle());
    NumericT           * vec_buffer = detail::extract_raw_pointer<NumericT>(vec.handle());

    for (vcl_size_t col2 = 0; col2 < U.lhs().size1(); ++col2)
    {
      vcl_size_t col = (U.lhs().size1() - col2) - 1;
      vcl_size_t col_begin = row_buffer[col];
      vcl_size_t col_end = row_buffer[col+1];

      // Stage 2: Substitute
      NumericT vec_entry = vec_buffer[col] / diagonal_buffer[col];
      vec_buffer[col] = vec_entry;
      for (vcl_size_t i = col_begin; i < col_end; ++i)
      {
        vcl_size_t row_index = col_buffer[i];
        if (row_index < col)
          vec_buffer[row_index] -= vec_entry * elements[i];
      }
    }
  }


} //namespace detail

/** @brief Inplace solution of a lower triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
*
* @param proxy  Proxy object for a transposed CSR-matrix
* @param vec    The right hand side vector
* @param tag    The solver tag identifying the respective triangular solver
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(matrix_expression< const compressed_matrix<NumericT, AlignmentV>,
                                      const compressed_matrix<NumericT, AlignmentV>,
                                      op_trans> const & proxy,
                   vector_base<NumericT> & vec,
                   viennacl::linalg::unit_lower_tag tag)
{
  NumericT           * vec_buf    = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements   = detail::extract_raw_pointer<NumericT>(proxy.lhs().handle());
  unsigned int const * row_buffer = detail::extract_raw_pointer<unsigned int>(proxy.lhs().handle1());
  unsigned int const * col_buffer = detail::extract_raw_pointer<unsigned int>(proxy.lhs().handle2());

  detail::csr_trans_inplace_solve<NumericT>(row_buffer, col_buffer, elements, vec_buf, proxy.lhs().size1(), tag);
}

/** @brief Inplace solution of a lower triangular compressed_matrix. Typically used for LU substitutions
*
* @param proxy  Proxy object for a transposed CSR-matrix
* @param vec    The right hand side vector
* @param tag    The solver tag identifying the respective triangular solver
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(matrix_expression< const compressed_matrix<NumericT, AlignmentV>,
                                      const compressed_matrix<NumericT, AlignmentV>,
                                      op_trans> const & proxy,
                   vector_base<NumericT> & vec,
                   viennacl::linalg::lower_tag tag)
{
  NumericT           * vec_buf    = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements   = detail::extract_raw_pointer<NumericT>(proxy.lhs().handle());
  unsigned int const * row_buffer = detail::extract_raw_pointer<unsigned int>(proxy.lhs().handle1());
  unsigned int const * col_buffer = detail::extract_raw_pointer<unsigned int>(proxy.lhs().handle2());

  detail::csr_trans_inplace_solve<NumericT>(row_buffer, col_buffer, elements, vec_buf, proxy.lhs().size1(), tag);
}


/** @brief Inplace solution of a upper triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
*
* @param proxy  Proxy object for a transposed CSR-matrix
* @param vec    The right hand side vector
* @param tag    The solver tag identifying the respective triangular solver
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(matrix_expression< const compressed_matrix<NumericT, AlignmentV>,
                                      const compressed_matrix<NumericT, AlignmentV>,
                                      op_trans> const & proxy,
                   vector_base<NumericT> & vec,
                   viennacl::linalg::unit_upper_tag tag)
{
  NumericT           * vec_buf    = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements   = detail::extract_raw_pointer<NumericT>(proxy.lhs().handle());
  unsigned int const * row_buffer = detail::extract_raw_pointer<unsigned int>(proxy.lhs().handle1());
  unsigned int const * col_buffer = detail::extract_raw_pointer<unsigned int>(proxy.lhs().handle2());

  detail::csr_trans_inplace_solve<NumericT>(row_buffer, col_buffer, elements, vec_buf, proxy.lhs().size1(), tag);
}


/** @brief Inplace solution of a upper triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
*
* @param proxy  Proxy object for a transposed CSR-matrix
* @param vec    The right hand side vector
* @param tag    The solver tag identifying the respective triangular solver
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(matrix_expression< const compressed_matrix<NumericT, AlignmentV>,
                                      const compressed_matrix<NumericT, AlignmentV>,
                                      op_trans> const & proxy,
                   vector_base<NumericT> & vec,
                   viennacl::linalg::upper_tag tag)
{
  NumericT           * vec_buf    = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements   = detail::extract_raw_pointer<NumericT>(proxy.lhs().handle());
  unsigned int const * row_buffer = detail::extract_raw_pointer<unsigned int>(proxy.lhs().handle1());
  unsigned int const * col_buffer = detail::extract_raw_pointer<unsigned int>(proxy.lhs().handle2());

  detail::csr_trans_inplace_solve<NumericT>(row_buffer, col_buffer, elements, vec_buf, proxy.lhs().size1(), tag);
}



//
// Compressed Compressed Matrix
//

/** @brief Carries out matrix-vector multiplication with a compressed_matrix
*
* Implementation of the convenience expression result = prod(mat, vec);
*
* @param mat    The matrix
* @param vec    The vector
* @param result The result vector
*/
template<typename NumericT>
void prod_impl(const viennacl::compressed_compressed_matrix<NumericT> & mat,
               const viennacl::vector_base<NumericT> & vec,
               NumericT alpha,
                     viennacl::vector_base<NumericT> & result,
               NumericT beta)
{
  NumericT           * result_buf  = detail::extract_raw_pointer<NumericT>(result.handle());
  NumericT     const * vec_buf     = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements    = detail::extract_raw_pointer<NumericT>(mat.handle());
  unsigned int const * row_buffer  = detail::extract_raw_pointer<unsigned int>(mat.handle1());
  unsigned int const * row_indices = detail::extract_raw_pointer<unsigned int>(mat.handle3());
  unsigned int const * col_buffer  = detail::extract_raw_pointer<unsigned int>(mat.handle2());

  if (beta < 0 || beta > 0)
  {
    for (vcl_size_t i = 0; i< result.size(); ++i)
      result_buf[i * result.stride() + result.start()] *= beta;
  }
  else // flush
  {
    for (vcl_size_t i = 0; i< result.size(); ++i)
      result_buf[i * result.stride() + result.start()] = 0;
  }

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i = 0; i < static_cast<long>(mat.nnz1()); ++i)
  {
    NumericT dot_prod = 0;
    vcl_size_t row_end = row_buffer[i+1];
    for (vcl_size_t j = row_buffer[i]; j < row_end; ++j)
      dot_prod += elements[j] * vec_buf[col_buffer[j] * vec.stride() + vec.start()];

    if (beta > 0 || beta < 0)
      result_buf[vcl_size_t(row_indices[i]) * result.stride() + result.start()] += alpha * dot_prod;
    else
      result_buf[vcl_size_t(row_indices[i]) * result.stride() + result.start()]  = alpha * dot_prod;
  }

}



//
// Coordinate Matrix
//

namespace detail
{
  template<typename NumericT, unsigned int AlignmentV>
  void row_info(coordinate_matrix<NumericT, AlignmentV> const & mat,
                vector_base<NumericT> & vec,
                viennacl::linalg::detail::row_info_types info_selector)
  {
    NumericT           * result_buf   = detail::extract_raw_pointer<NumericT>(vec.handle());
    NumericT     const * elements     = detail::extract_raw_pointer<NumericT>(mat.handle());
    unsigned int const * coord_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle12());

    NumericT value = 0;
    unsigned int last_row = 0;

    for (vcl_size_t i = 0; i < mat.nnz(); ++i)
    {
      unsigned int current_row = coord_buffer[2*i];

      if (current_row != last_row)
      {
        if (info_selector == viennacl::linalg::detail::SPARSE_ROW_NORM_2)
          value = std::sqrt(value);

        result_buf[last_row] = value;
        value = 0;
        last_row = current_row;
      }

      switch (info_selector)
      {
        case viennacl::linalg::detail::SPARSE_ROW_NORM_INF: //inf-norm
          value = std::max<NumericT>(value, std::fabs(elements[i]));
          break;

        case viennacl::linalg::detail::SPARSE_ROW_NORM_1: //1-norm
          value += std::fabs(elements[i]);
          break;

        case viennacl::linalg::detail::SPARSE_ROW_NORM_2: //2-norm
          value += elements[i] * elements[i];
          break;

        case viennacl::linalg::detail::SPARSE_ROW_DIAGONAL: //diagonal entry
          if (coord_buffer[2*i+1] == current_row)
            value = elements[i];
          break;

        //default:
        //  break;
      }
    }

    if (info_selector == viennacl::linalg::detail::SPARSE_ROW_NORM_2)
      value = std::sqrt(value);

    result_buf[last_row] = value;
  }
}

/** @brief Carries out matrix-vector multiplication with a coordinate_matrix
*
* Implementation of the convenience expression result = prod(mat, vec);
*
* @param mat    The matrix
* @param vec    The vector
* @param result The result vector
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::coordinate_matrix<NumericT, AlignmentV> & mat,
               const viennacl::vector_base<NumericT> & vec,
               NumericT alpha,
                     viennacl::vector_base<NumericT> & result,
               NumericT beta)
{
  NumericT           * result_buf   = detail::extract_raw_pointer<NumericT>(result.handle());
  NumericT     const * vec_buf      = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements     = detail::extract_raw_pointer<NumericT>(mat.handle());
  unsigned int const * coord_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle12());

  if (beta < 0 || beta > 0)
  {
    for (vcl_size_t i = 0; i< result.size(); ++i)
      result_buf[i * result.stride() + result.start()] *= beta;
  }
  else // flush
  {
    for (vcl_size_t i = 0; i< result.size(); ++i)
      result_buf[i * result.stride() + result.start()] = 0;
  }

  for (vcl_size_t i = 0; i < mat.nnz(); ++i)
    result_buf[coord_buffer[2*i] * result.stride() + result.start()]
      += alpha * elements[i] * vec_buf[coord_buffer[2*i+1] * vec.stride() + vec.start()];
}

/** @brief Carries out Compressed Matrix(COO)-Dense Matrix multiplication
*
* Implementation of the convenience expression result = prod(sp_mat, d_mat);
*
* @param sp_mat     The Sparse Matrix (Coordinate format)
* @param d_mat      The Dense Matrix
* @param result     The Result Matrix
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::coordinate_matrix<NumericT, AlignmentV> & sp_mat,
               const viennacl::matrix_base<NumericT> & d_mat,
                     viennacl::matrix_base<NumericT> & result) {

  NumericT     const * sp_mat_elements = detail::extract_raw_pointer<NumericT>(sp_mat.handle());
  unsigned int const * sp_mat_coords   = detail::extract_raw_pointer<unsigned int>(sp_mat.handle12());

  NumericT const * d_mat_data  = detail::extract_raw_pointer<NumericT>(d_mat);
  NumericT       * result_data = detail::extract_raw_pointer<NumericT>(result);

  vcl_size_t d_mat_start1 = viennacl::traits::start1(d_mat);
  vcl_size_t d_mat_start2 = viennacl::traits::start2(d_mat);
  vcl_size_t d_mat_inc1   = viennacl::traits::stride1(d_mat);
  vcl_size_t d_mat_inc2   = viennacl::traits::stride2(d_mat);
  vcl_size_t d_mat_internal_size1  = viennacl::traits::internal_size1(d_mat);
  vcl_size_t d_mat_internal_size2  = viennacl::traits::internal_size2(d_mat);

  vcl_size_t result_start1 = viennacl::traits::start1(result);
  vcl_size_t result_start2 = viennacl::traits::start2(result);
  vcl_size_t result_inc1   = viennacl::traits::stride1(result);
  vcl_size_t result_inc2   = viennacl::traits::stride2(result);
  vcl_size_t result_internal_size1  = viennacl::traits::internal_size1(result);
  vcl_size_t result_internal_size2  = viennacl::traits::internal_size2(result);

  detail::matrix_array_wrapper<NumericT const, row_major, false>
      d_mat_wrapper_row(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);
  detail::matrix_array_wrapper<NumericT const, column_major, false>
      d_mat_wrapper_col(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);

  detail::matrix_array_wrapper<NumericT, row_major, false>
      result_wrapper_row(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);
  detail::matrix_array_wrapper<NumericT, column_major, false>
      result_wrapper_col(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);

  if ( d_mat.row_major() ) {

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long row = 0; row < static_cast<long>(sp_mat.size1()); ++row)
    {
      if (result.row_major())
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
          result_wrapper_row(row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
      else
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
          result_wrapper_col(row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
    }

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long i = 0; i < static_cast<long>(sp_mat.nnz()); ++i) {
      NumericT x = static_cast<NumericT>(sp_mat_elements[i]);
      vcl_size_t r = static_cast<vcl_size_t>(sp_mat_coords[2*i]);
      vcl_size_t c = static_cast<vcl_size_t>(sp_mat_coords[2*i+1]);
      for (vcl_size_t col = 0; col < d_mat.size2(); ++col) {
        NumericT y = d_mat_wrapper_row( c, col);
        if (result.row_major())
          result_wrapper_row(r, col) += x * y;
        else
          result_wrapper_col(r, col) += x * y;
      }
    }
  }

  else {

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long col = 0; col < static_cast<long>(d_mat.size2()); ++col)
    {
      if (result.row_major())
        for (vcl_size_t row = 0; row < sp_mat.size1(); ++row)
          result_wrapper_row( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
      else
        for (vcl_size_t row = 0; row < sp_mat.size1(); ++row)
          result_wrapper_col( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
    }

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long col = 0; col < static_cast<long>(d_mat.size2()); ++col) {

      for (vcl_size_t i = 0; i < sp_mat.nnz(); ++i) {

        NumericT x = static_cast<NumericT>(sp_mat_elements[i]);
        vcl_size_t r = static_cast<vcl_size_t>(sp_mat_coords[2*i]);
        vcl_size_t c = static_cast<vcl_size_t>(sp_mat_coords[2*i+1]);
        NumericT y = d_mat_wrapper_col( c, col);

        if (result.row_major())
          result_wrapper_row( r, col) += x*y;
        else
          result_wrapper_col( r, col) += x*y;
      }

    }
  }

}


/** @brief Carries out Compressed Matrix(COO)-Dense Transposed Matrix multiplication
*
* Implementation of the convenience expression result = prod(sp_mat, trans(d_mat));
*
* @param sp_mat     The Sparse Matrix (Coordinate format)
* @param d_mat      The Dense Transposed Matrix
* @param result     The Result Matrix
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::coordinate_matrix<NumericT, AlignmentV> & sp_mat,
               const viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                                  const viennacl::matrix_base<NumericT>,
                                                  viennacl::op_trans > & d_mat,
                     viennacl::matrix_base<NumericT> & result) {

  NumericT     const * sp_mat_elements     = detail::extract_raw_pointer<NumericT>(sp_mat.handle());
  unsigned int const * sp_mat_coords       = detail::extract_raw_pointer<unsigned int>(sp_mat.handle12());

  NumericT const * d_mat_data = detail::extract_raw_pointer<NumericT>(d_mat.lhs());
  NumericT       * result_data = detail::extract_raw_pointer<NumericT>(result);

  vcl_size_t d_mat_start1 = viennacl::traits::start1(d_mat.lhs());
  vcl_size_t d_mat_start2 = viennacl::traits::start2(d_mat.lhs());
  vcl_size_t d_mat_inc1   = viennacl::traits::stride1(d_mat.lhs());
  vcl_size_t d_mat_inc2   = viennacl::traits::stride2(d_mat.lhs());
  vcl_size_t d_mat_internal_size1  = viennacl::traits::internal_size1(d_mat.lhs());
  vcl_size_t d_mat_internal_size2  = viennacl::traits::internal_size2(d_mat.lhs());

  vcl_size_t result_start1 = viennacl::traits::start1(result);
  vcl_size_t result_start2 = viennacl::traits::start2(result);
  vcl_size_t result_inc1   = viennacl::traits::stride1(result);
  vcl_size_t result_inc2   = viennacl::traits::stride2(result);
  vcl_size_t result_internal_size1  = viennacl::traits::internal_size1(result);
  vcl_size_t result_internal_size2  = viennacl::traits::internal_size2(result);

  detail::matrix_array_wrapper<NumericT const, row_major, false>
      d_mat_wrapper_row(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);
  detail::matrix_array_wrapper<NumericT const, column_major, false>
      d_mat_wrapper_col(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);

  detail::matrix_array_wrapper<NumericT, row_major, false>
      result_wrapper_row(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);
  detail::matrix_array_wrapper<NumericT, column_major, false>
      result_wrapper_col(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);

  if ( d_mat.lhs().row_major() )
  {
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long row = 0; row < static_cast<long>(sp_mat.size1()); ++row)
    {
      if (result.row_major())
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
          result_wrapper_row( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
      else
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
          result_wrapper_col( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
    }

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
    for (long i = 0; i < static_cast<long>(sp_mat.nnz()); ++i) {
      NumericT x = static_cast<NumericT>(sp_mat_elements[i]);
      vcl_size_t r = static_cast<vcl_size_t>(sp_mat_coords[2*i]);
      vcl_size_t c = static_cast<vcl_size_t>(sp_mat_coords[2*i+1]);
      if (result.row_major())
      {
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col) {
          NumericT y = d_mat_wrapper_row( col, c);
          result_wrapper_row(r, col) += x * y;
        }
      }
      else
      {
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col) {
          NumericT y = d_mat_wrapper_row( col, c);
          result_wrapper_col(r, col) += x * y;
        }
      }
    }


  }
  else
  {
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long col = 0; col < static_cast<long>(d_mat.size2()); ++col)
    {
      if (result.row_major())
        for (vcl_size_t row = 0; row < sp_mat.size1(); ++row)
          result_wrapper_row( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
      else
        for (vcl_size_t row = 0; row < sp_mat.size1(); ++row)
          result_wrapper_col( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
    }

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long i = 0; i < static_cast<long>(sp_mat.nnz()); ++i) {
      NumericT x = static_cast<NumericT>(sp_mat_elements[i]);
      vcl_size_t r = static_cast<vcl_size_t>(sp_mat_coords[2*i]);
      vcl_size_t c = static_cast<vcl_size_t>(sp_mat_coords[2*i+1]);
      if (result.row_major())
      {
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col) {
          NumericT y = d_mat_wrapper_col( col, c);
          result_wrapper_row(r, col) += x * y;
        }
      }
      else
      {
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col) {
          NumericT y = d_mat_wrapper_col( col, c);
          result_wrapper_col(r, col) += x * y;
        }
      }
    }
  }

}



//
// ELL Matrix
//
/** @brief Carries out matrix-vector multiplication with a ell_matrix
*
* Implementation of the convenience expression result = prod(mat, vec);
*
* @param mat    The matrix
* @param vec    The vector
* @param result The result vector
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::ell_matrix<NumericT, AlignmentV> & mat,
               const viennacl::vector_base<NumericT> & vec,
               NumericT alpha,
                     viennacl::vector_base<NumericT> & result,
               NumericT beta)
{
  NumericT           * result_buf   = detail::extract_raw_pointer<NumericT>(result.handle());
  NumericT     const * vec_buf      = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements     = detail::extract_raw_pointer<NumericT>(mat.handle());
  unsigned int const * coords       = detail::extract_raw_pointer<unsigned int>(mat.handle2());

  for (vcl_size_t row = 0; row < mat.size1(); ++row)
  {
    NumericT sum = 0;

    for (unsigned int item_id = 0; item_id < mat.internal_maxnnz(); ++item_id)
    {
      vcl_size_t offset = row + item_id * mat.internal_size1();
      NumericT val = elements[offset];

      if (val > 0 || val < 0)
      {
        unsigned int col = coords[offset];
        sum += (vec_buf[col * vec.stride() + vec.start()] * val);
      }
    }

    if (beta < 0 || beta > 0)
    {
      vcl_size_t index = row * result.stride() + result.start();
      result_buf[index] = alpha * sum + beta * result_buf[index];
    }
    else
      result_buf[row * result.stride() + result.start()] = alpha * sum;
  }
}

/** @brief Carries out ell_matrix-d_matrix multiplication
*
* Implementation of the convenience expression result = prod(sp_mat, d_mat);
*
* @param sp_mat     The sparse(ELL) matrix
* @param d_mat      The dense matrix
* @param result     The result dense matrix
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::ell_matrix<NumericT, AlignmentV> & sp_mat,
               const viennacl::matrix_base<NumericT> & d_mat,
                     viennacl::matrix_base<NumericT> & result)
{
  NumericT     const * sp_mat_elements     = detail::extract_raw_pointer<NumericT>(sp_mat.handle());
  unsigned int const * sp_mat_coords       = detail::extract_raw_pointer<unsigned int>(sp_mat.handle2());

  NumericT const * d_mat_data = detail::extract_raw_pointer<NumericT>(d_mat);
  NumericT       * result_data = detail::extract_raw_pointer<NumericT>(result);

  vcl_size_t d_mat_start1 = viennacl::traits::start1(d_mat);
  vcl_size_t d_mat_start2 = viennacl::traits::start2(d_mat);
  vcl_size_t d_mat_inc1   = viennacl::traits::stride1(d_mat);
  vcl_size_t d_mat_inc2   = viennacl::traits::stride2(d_mat);
  vcl_size_t d_mat_internal_size1  = viennacl::traits::internal_size1(d_mat);
  vcl_size_t d_mat_internal_size2  = viennacl::traits::internal_size2(d_mat);

  vcl_size_t result_start1 = viennacl::traits::start1(result);
  vcl_size_t result_start2 = viennacl::traits::start2(result);
  vcl_size_t result_inc1   = viennacl::traits::stride1(result);
  vcl_size_t result_inc2   = viennacl::traits::stride2(result);
  vcl_size_t result_internal_size1  = viennacl::traits::internal_size1(result);
  vcl_size_t result_internal_size2  = viennacl::traits::internal_size2(result);

  detail::matrix_array_wrapper<NumericT const, row_major, false>
      d_mat_wrapper_row(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);
  detail::matrix_array_wrapper<NumericT const, column_major, false>
      d_mat_wrapper_col(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);

  detail::matrix_array_wrapper<NumericT, row_major, false>
      result_wrapper_row(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);
  detail::matrix_array_wrapper<NumericT, column_major, false>
      result_wrapper_col(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);

  if ( d_mat.row_major() ) {
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long row = 0; row < static_cast<long>(sp_mat.size1()); ++row)
    {
      if (result.row_major())
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
          result_wrapper_row( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
      else
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
          result_wrapper_col( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
    }

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long row = 0; row < static_cast<long>(sp_mat.size1()); ++row)
    {
      for (long item_id = 0; item_id < static_cast<long>(sp_mat.maxnnz()); ++item_id)
      {
        vcl_size_t offset = static_cast<vcl_size_t>(row) + static_cast<vcl_size_t>(item_id) * sp_mat.internal_size1();
        NumericT sp_mat_val = static_cast<NumericT>(sp_mat_elements[offset]);
        vcl_size_t sp_mat_col = static_cast<vcl_size_t>(sp_mat_coords[offset]);

        if (sp_mat_val < 0 || sp_mat_val > 0) // sp_mat_val != 0 without compiler warnings
        {
          if (result.row_major())
            for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
              result_wrapper_row(static_cast<vcl_size_t>(row), col) += sp_mat_val * d_mat_wrapper_row( sp_mat_col, col);
          else
            for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
              result_wrapper_col(static_cast<vcl_size_t>(row), col) += sp_mat_val * d_mat_wrapper_row( sp_mat_col, col);
        }
      }
    }
  }
  else {
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long col = 0; col < static_cast<long>(d_mat.size2()); ++col)
    {
      if (result.row_major())
        for (long row = 0; row < static_cast<long>(sp_mat.size1()); ++row)
          result_wrapper_row( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
      else
        for (long row = 0; row < static_cast<long>(sp_mat.size1()); ++row)
          result_wrapper_col( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
    }

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long col = 0; col < static_cast<long>(d_mat.size2()); ++col) {

      for (unsigned int item_id = 0; item_id < sp_mat.maxnnz(); ++item_id) {

        for (vcl_size_t row = 0; row < sp_mat.size1(); ++row) {

          vcl_size_t offset = row + item_id * sp_mat.internal_size1();
          NumericT sp_mat_val = static_cast<NumericT>(sp_mat_elements[offset]);
          vcl_size_t sp_mat_col = static_cast<vcl_size_t>(sp_mat_coords[offset]);

          if (sp_mat_val < 0 || sp_mat_val > 0)  // sp_mat_val != 0 without compiler warnings
          {
            if (result.row_major())
              result_wrapper_row( row, col) += sp_mat_val * d_mat_wrapper_col( sp_mat_col, col);
            else
              result_wrapper_col( row, col) += sp_mat_val * d_mat_wrapper_col( sp_mat_col, col);
          }
        }
      }
    }
  }

}

/** @brief Carries out matrix-trans(matrix) multiplication first matrix being sparse ell
*          and the second dense transposed
*
* Implementation of the convenience expression result = prod(sp_mat, trans(d_mat));
*
* @param sp_mat             The sparse matrix
* @param d_mat              The transposed dense matrix
* @param result             The result matrix
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::ell_matrix<NumericT, AlignmentV> & sp_mat,
               const viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                                  const viennacl::matrix_base<NumericT>,
                                                  viennacl::op_trans > & d_mat,
                     viennacl::matrix_base<NumericT> & result) {

  NumericT     const * sp_mat_elements     = detail::extract_raw_pointer<NumericT>(sp_mat.handle());
  unsigned int const * sp_mat_coords       = detail::extract_raw_pointer<unsigned int>(sp_mat.handle2());

  NumericT const * d_mat_data  = detail::extract_raw_pointer<NumericT>(d_mat.lhs());
  NumericT       * result_data = detail::extract_raw_pointer<NumericT>(result);

  vcl_size_t d_mat_start1 = viennacl::traits::start1(d_mat.lhs());
  vcl_size_t d_mat_start2 = viennacl::traits::start2(d_mat.lhs());
  vcl_size_t d_mat_inc1   = viennacl::traits::stride1(d_mat.lhs());
  vcl_size_t d_mat_inc2   = viennacl::traits::stride2(d_mat.lhs());
  vcl_size_t d_mat_internal_size1  = viennacl::traits::internal_size1(d_mat.lhs());
  vcl_size_t d_mat_internal_size2  = viennacl::traits::internal_size2(d_mat.lhs());

  vcl_size_t result_start1 = viennacl::traits::start1(result);
  vcl_size_t result_start2 = viennacl::traits::start2(result);
  vcl_size_t result_inc1   = viennacl::traits::stride1(result);
  vcl_size_t result_inc2   = viennacl::traits::stride2(result);
  vcl_size_t result_internal_size1  = viennacl::traits::internal_size1(result);
  vcl_size_t result_internal_size2  = viennacl::traits::internal_size2(result);

  detail::matrix_array_wrapper<NumericT const, row_major, false>
      d_mat_wrapper_row(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);
  detail::matrix_array_wrapper<NumericT const, column_major, false>
      d_mat_wrapper_col(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);

  detail::matrix_array_wrapper<NumericT, row_major, false>
      result_wrapper_row(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);
  detail::matrix_array_wrapper<NumericT, column_major, false>
      result_wrapper_col(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);

  if ( d_mat.lhs().row_major() )
  {
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
    for (long row = 0; row < static_cast<long>(sp_mat.size1()); ++row)
    {
      if (result.row_major())
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
          result_wrapper_row( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
      else
        for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
          result_wrapper_col( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
    }

    for (vcl_size_t col = 0; col < d_mat.size2(); ++col) {

      for (unsigned int item_id = 0; item_id < sp_mat.maxnnz(); ++item_id) {

        for (vcl_size_t row = 0; row < sp_mat.size1(); ++row) {

          vcl_size_t offset = row + item_id * sp_mat.internal_size1();
          NumericT sp_mat_val = static_cast<NumericT>(sp_mat_elements[offset]);
          vcl_size_t sp_mat_col = static_cast<vcl_size_t>(sp_mat_coords[offset]);

          if (sp_mat_val < 0 || sp_mat_val > 0) // sp_mat_val != 0 without compiler warnings
          {
            if (result.row_major())
              result_wrapper_row( row, col) += sp_mat_val * d_mat_wrapper_row( col, sp_mat_col);
            else
              result_wrapper_col( row, col) += sp_mat_val * d_mat_wrapper_row( col, sp_mat_col);
          }
        }
      }
    }
  }
  else
  {
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (long col = 0; col < static_cast<long>(d_mat.size2()); ++col)
    {
      if (result.row_major())
        for (vcl_size_t row = 0; row < sp_mat.size1(); ++row)
          result_wrapper_row( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
      else
        for (vcl_size_t row = 0; row < sp_mat.size1(); ++row)
          result_wrapper_col( row, col) = (NumericT)0; /* filling result with zeros, as the product loops are reordered */
    }

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
    for (vcl_size_t row = 0; row < sp_mat.size1(); ++row) {

      for (long item_id = 0; item_id < static_cast<long>(sp_mat.maxnnz()); ++item_id) {

        vcl_size_t offset = row + static_cast<vcl_size_t>(item_id) * sp_mat.internal_size1();
        NumericT sp_mat_val = static_cast<NumericT>(sp_mat_elements[offset]);
        vcl_size_t sp_mat_col = static_cast<vcl_size_t>(sp_mat_coords[offset]);

        if (sp_mat_val < 0 || sp_mat_val > 0)  // sp_mat_val != 0 without compiler warnings
        {
          if (result.row_major())
            for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
              result_wrapper_row( row, col) += sp_mat_val * d_mat_wrapper_col( col, sp_mat_col);
          else
            for (vcl_size_t col = 0; col < d_mat.size2(); ++col)
              result_wrapper_col( row, col) += sp_mat_val * d_mat_wrapper_col( col, sp_mat_col);
        }
      }
    }
  }

}


//
// SELL-C-\sigma Matrix
//
/** @brief Carries out matrix-vector multiplication with a sliced_ell_matrix
*
* Implementation of the convenience expression result = prod(mat, vec);
*
* @param mat    The matrix
* @param vec    The vector
* @param result The result vector
*/
template<typename NumericT, typename IndexT>
void prod_impl(const viennacl::sliced_ell_matrix<NumericT, IndexT> & mat,
               const viennacl::vector_base<NumericT> & vec,
               NumericT alpha,
                     viennacl::vector_base<NumericT> & result,
               NumericT beta)
{
  NumericT       * result_buf        = detail::extract_raw_pointer<NumericT>(result.handle());
  NumericT const * vec_buf           = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT const * elements          = detail::extract_raw_pointer<NumericT>(mat.handle());
  IndexT   const * columns_per_block = detail::extract_raw_pointer<IndexT>(mat.handle1());
  IndexT   const * column_indices    = detail::extract_raw_pointer<IndexT>(mat.handle2());
  IndexT   const * block_start       = detail::extract_raw_pointer<IndexT>(mat.handle3());

  vcl_size_t num_blocks = mat.size1() / mat.rows_per_block() + 1;

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long block_idx2 = 0; block_idx2 < static_cast<long>(num_blocks); ++block_idx2)
  {
    vcl_size_t block_idx = static_cast<vcl_size_t>(block_idx2);
    vcl_size_t current_columns_per_block = columns_per_block[block_idx];

    std::vector<NumericT> result_values(mat.rows_per_block());

    for (IndexT column_entry_index = 0;
                column_entry_index < current_columns_per_block;
              ++column_entry_index)
    {
      vcl_size_t stride_start = block_start[block_idx] + column_entry_index * mat.rows_per_block();
      // Note: This for-loop may be unrolled by hand for exploiting vectorization
      //       Careful benchmarking recommended first, memory channels may be saturated already!
      for (IndexT row_in_block = 0; row_in_block < mat.rows_per_block(); ++row_in_block)
      {
        NumericT val = elements[stride_start + row_in_block];

        result_values[row_in_block] += (val > 0 || val < 0) ? vec_buf[column_indices[stride_start + row_in_block] * vec.stride() + vec.start()] * val : 0;
      }
    }

    vcl_size_t first_row_in_matrix = block_idx * mat.rows_per_block();
    if (beta < 0 || beta > 0)
    {
      for (IndexT row_in_block = 0; row_in_block < mat.rows_per_block(); ++row_in_block)
      {
        if (first_row_in_matrix + row_in_block < result.size())
        {
          vcl_size_t index = (first_row_in_matrix + row_in_block) * result.stride() + result.start();
          result_buf[index] = alpha * result_values[row_in_block] + beta * result_buf[index];
        }
      }
    }
    else
    {
      for (IndexT row_in_block = 0; row_in_block < mat.rows_per_block(); ++row_in_block)
      {
        if (first_row_in_matrix + row_in_block < result.size())
          result_buf[(first_row_in_matrix + row_in_block) * result.stride() + result.start()] = alpha * result_values[row_in_block];
      }
    }
  }
}


//
// Hybrid Matrix
//
/** @brief Carries out matrix-vector multiplication with a hyb_matrix
*
* Implementation of the convenience expression result = prod(mat, vec);
*
* @param mat    The matrix
* @param vec    The vector
* @param result The result vector
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::hyb_matrix<NumericT, AlignmentV> & mat,
               const viennacl::vector_base<NumericT> & vec,
               NumericT alpha,
                     viennacl::vector_base<NumericT> & result,
               NumericT beta)
{
  NumericT           * result_buf     = detail::extract_raw_pointer<NumericT>(result.handle());
  NumericT     const * vec_buf        = detail::extract_raw_pointer<NumericT>(vec.handle());
  NumericT     const * elements       = detail::extract_raw_pointer<NumericT>(mat.handle());
  unsigned int const * coords         = detail::extract_raw_pointer<unsigned int>(mat.handle2());
  NumericT     const * csr_elements   = detail::extract_raw_pointer<NumericT>(mat.handle5());
  unsigned int const * csr_row_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle3());
  unsigned int const * csr_col_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle4());


  for (vcl_size_t row = 0; row < mat.size1(); ++row)
  {
    NumericT sum = 0;

    //
    // Part 1: Process ELL part
    //
    for (unsigned int item_id = 0; item_id < mat.internal_ellnnz(); ++item_id)
    {
      vcl_size_t offset = row + item_id * mat.internal_size1();
      NumericT val = elements[offset];

      if (val > 0 || val < 0)
      {
        unsigned int col = coords[offset];
        sum += (vec_buf[col * vec.stride() + vec.start()] * val);
      }
    }

    //
    // Part 2: Process HYB part
    //
    vcl_size_t col_begin = csr_row_buffer[row];
    vcl_size_t col_end   = csr_row_buffer[row + 1];

    for (vcl_size_t item_id = col_begin; item_id < col_end; item_id++)
    {
        sum += (vec_buf[csr_col_buffer[item_id] * vec.stride() + vec.start()] * csr_elements[item_id]);
    }

    if (beta < 0 || beta > 0)
    {
      vcl_size_t index = row * result.stride() + result.start();
      result_buf[index] = alpha * sum + beta * result_buf[index];
    }
    else
      result_buf[row * result.stride() + result.start()] = alpha * sum;
  }

}

//
// Hybrid Matrix
//
/** @brief Carries out sparse-matrix-dense-matrix multiplication with a hyb_matrix
*
* Implementation of the convenience expression C = prod(A, B);
*
* @param mat    The sparse matrix A
* @param d_mat  The dense matrix B
* @param result The dense result matrix C
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::hyb_matrix<NumericT, AlignmentV> & mat,
               const viennacl::matrix_base<NumericT> & d_mat,
                     viennacl::matrix_base<NumericT> & result)
{
  NumericT const * d_mat_data = detail::extract_raw_pointer<NumericT>(d_mat);
  NumericT       * result_data = detail::extract_raw_pointer<NumericT>(result);

  vcl_size_t d_mat_start1 = viennacl::traits::start1(d_mat);
  vcl_size_t d_mat_start2 = viennacl::traits::start2(d_mat);
  vcl_size_t d_mat_inc1   = viennacl::traits::stride1(d_mat);
  vcl_size_t d_mat_inc2   = viennacl::traits::stride2(d_mat);
  vcl_size_t d_mat_internal_size1  = viennacl::traits::internal_size1(d_mat);
  vcl_size_t d_mat_internal_size2  = viennacl::traits::internal_size2(d_mat);

  vcl_size_t result_start1 = viennacl::traits::start1(result);
  vcl_size_t result_start2 = viennacl::traits::start2(result);
  vcl_size_t result_inc1   = viennacl::traits::stride1(result);
  vcl_size_t result_inc2   = viennacl::traits::stride2(result);
  vcl_size_t result_internal_size1  = viennacl::traits::internal_size1(result);
  vcl_size_t result_internal_size2  = viennacl::traits::internal_size2(result);

  detail::matrix_array_wrapper<NumericT const, row_major, false>
      d_mat_wrapper_row(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);
  detail::matrix_array_wrapper<NumericT const, column_major, false>
      d_mat_wrapper_col(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);

  detail::matrix_array_wrapper<NumericT, row_major, false>
      result_wrapper_row(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);
  detail::matrix_array_wrapper<NumericT, column_major, false>
      result_wrapper_col(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);

  NumericT     const * elements       = detail::extract_raw_pointer<NumericT>(mat.handle());
  unsigned int const * coords         = detail::extract_raw_pointer<unsigned int>(mat.handle2());
  NumericT     const * csr_elements   = detail::extract_raw_pointer<NumericT>(mat.handle5());
  unsigned int const * csr_row_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle3());
  unsigned int const * csr_col_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle4());


  for (vcl_size_t result_col = 0; result_col < result.size2(); ++result_col)
  {
    for (vcl_size_t row = 0; row < mat.size1(); ++row)
    {
      NumericT sum = 0;

      //
      // Part 1: Process ELL part
      //
      for (unsigned int item_id = 0; item_id < mat.internal_ellnnz(); ++item_id)
      {
        vcl_size_t offset = row + item_id * mat.internal_size1();
        NumericT val = elements[offset];

        if (val < 0 || val > 0)  // val != 0 without compiler warnings
        {
          vcl_size_t col = static_cast<vcl_size_t>(coords[offset]);
          if (d_mat.row_major())
            sum += d_mat_wrapper_row(col, result_col) * val;
          else
            sum += d_mat_wrapper_col(col, result_col) * val;
        }
      }

      //
      // Part 2: Process HYB/CSR part
      //
      vcl_size_t col_begin = csr_row_buffer[row];
      vcl_size_t col_end   = csr_row_buffer[row + 1];

      if (d_mat.row_major())
        for (vcl_size_t item_id = col_begin; item_id < col_end; item_id++)
          sum += d_mat_wrapper_row(static_cast<vcl_size_t>(csr_col_buffer[item_id]), result_col) * csr_elements[item_id];
      else
        for (vcl_size_t item_id = col_begin; item_id < col_end; item_id++)
          sum += d_mat_wrapper_col(static_cast<vcl_size_t>(csr_col_buffer[item_id]), result_col) * csr_elements[item_id];

      if (result.row_major())
        result_wrapper_row(row, result_col) = sum;
      else
        result_wrapper_col(row, result_col) = sum;
    }
  } // for result_col
}


/** @brief Carries out sparse-matrix-transposed-dense-matrix multiplication with a hyb_matrix
*
* Implementation of the convenience expression C = prod(A, trans(B));
*
* @param mat    The sparse matrix A
* @param d_mat  The dense matrix B
* @param result The dense result matrix C
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::hyb_matrix<NumericT, AlignmentV> & mat,
               const viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                                  const viennacl::matrix_base<NumericT>,
                                                  viennacl::op_trans > & d_mat,
                     viennacl::matrix_base<NumericT> & result)
{
  NumericT const * d_mat_data  = detail::extract_raw_pointer<NumericT>(d_mat);
  NumericT       * result_data = detail::extract_raw_pointer<NumericT>(result);

  vcl_size_t d_mat_start1 = viennacl::traits::start1(d_mat.lhs());
  vcl_size_t d_mat_start2 = viennacl::traits::start2(d_mat.lhs());
  vcl_size_t d_mat_inc1   = viennacl::traits::stride1(d_mat.lhs());
  vcl_size_t d_mat_inc2   = viennacl::traits::stride2(d_mat.lhs());
  vcl_size_t d_mat_internal_size1  = viennacl::traits::internal_size1(d_mat.lhs());
  vcl_size_t d_mat_internal_size2  = viennacl::traits::internal_size2(d_mat.lhs());

  vcl_size_t result_start1 = viennacl::traits::start1(result);
  vcl_size_t result_start2 = viennacl::traits::start2(result);
  vcl_size_t result_inc1   = viennacl::traits::stride1(result);
  vcl_size_t result_inc2   = viennacl::traits::stride2(result);
  vcl_size_t result_internal_size1  = viennacl::traits::internal_size1(result);
  vcl_size_t result_internal_size2  = viennacl::traits::internal_size2(result);

  detail::matrix_array_wrapper<NumericT const, row_major, false>
      d_mat_wrapper_row(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);
  detail::matrix_array_wrapper<NumericT const, column_major, false>
      d_mat_wrapper_col(d_mat_data, d_mat_start1, d_mat_start2, d_mat_inc1, d_mat_inc2, d_mat_internal_size1, d_mat_internal_size2);

  detail::matrix_array_wrapper<NumericT, row_major, false>
      result_wrapper_row(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);
  detail::matrix_array_wrapper<NumericT, column_major, false>
      result_wrapper_col(result_data, result_start1, result_start2, result_inc1, result_inc2, result_internal_size1, result_internal_size2);

  NumericT     const * elements       = detail::extract_raw_pointer<NumericT>(mat.handle());
  unsigned int const * coords         = detail::extract_raw_pointer<unsigned int>(mat.handle2());
  NumericT     const * csr_elements   = detail::extract_raw_pointer<NumericT>(mat.handle5());
  unsigned int const * csr_row_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle3());
  unsigned int const * csr_col_buffer = detail::extract_raw_pointer<unsigned int>(mat.handle4());


  for (vcl_size_t result_col = 0; result_col < result.size2(); ++result_col)
  {
    for (vcl_size_t row = 0; row < mat.size1(); ++row)
    {
      NumericT sum = 0;

      //
      // Part 1: Process ELL part
      //
      for (unsigned int item_id = 0; item_id < mat.internal_ellnnz(); ++item_id)
      {
        vcl_size_t offset = row + item_id * mat.internal_size1();
        NumericT val = elements[offset];

        if (val < 0 || val > 0)  // val != 0 without compiler warnings
        {
          vcl_size_t col = static_cast<vcl_size_t>(coords[offset]);
          if (d_mat.lhs().row_major())
            sum += d_mat_wrapper_row(result_col, col) * val;
          else
            sum += d_mat_wrapper_col(result_col, col) * val;
        }
      }

      //
      // Part 2: Process HYB/CSR part
      //
      vcl_size_t col_begin = csr_row_buffer[row];
      vcl_size_t col_end   = csr_row_buffer[row + 1];

      if (d_mat.lhs().row_major())
        for (vcl_size_t item_id = col_begin; item_id < col_end; item_id++)
          sum += d_mat_wrapper_row(result_col, static_cast<vcl_size_t>(csr_col_buffer[item_id])) * csr_elements[item_id];
      else
        for (vcl_size_t item_id = col_begin; item_id < col_end; item_id++)
          sum += d_mat_wrapper_col(result_col, static_cast<vcl_size_t>(csr_col_buffer[item_id])) * csr_elements[item_id];

      if (result.row_major())
        result_wrapper_row(row, result_col) = sum;
      else
        result_wrapper_col(row, result_col) = sum;
    }
  } // for result_col
}


} // namespace host_based
} //namespace linalg
} //namespace viennacl


#endif
