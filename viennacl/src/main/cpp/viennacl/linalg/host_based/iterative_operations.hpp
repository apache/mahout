#ifndef VIENNACL_LINALG_HOST_BASED_ITERATIVE_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_ITERATIVE_OPERATIONS_HPP_

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

/** @file viennacl/linalg/host_based/iterative_operations.hpp
    @brief Implementations of specialized kernels for fast iterative solvers using OpenMP on the CPU
*/

#include <cmath>
#include <algorithm>  //for std::max and std::min

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/linalg/host_based/common.hpp"
#include "viennacl/linalg/detail/op_applier.hpp"
#include "viennacl/traits/stride.hpp"


// Minimum vector size for using OpenMP on vector operations:
#ifndef VIENNACL_OPENMP_VECTOR_MIN_SIZE
  #define VIENNACL_OPENMP_VECTOR_MIN_SIZE  5000
#endif

namespace viennacl
{
namespace linalg
{
namespace host_based
{

namespace detail
{
  /** @brief Implementation of a fused matrix-vector product with a compressed_matrix for an efficient pipelined CG algorithm.
    *
    * This routines computes for a matrix A and vectors 'p', 'Ap', and 'r0':
    *   Ap = prod(A, p);
    * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap), inner_prod(Ap, r0)
    */
  template<typename NumericT>
  void pipelined_prod_impl(compressed_matrix<NumericT> const & A,
                           vector_base<NumericT> const & p,
                           vector_base<NumericT> & Ap,
                           NumericT const * r0star,
                           vector_base<NumericT> & inner_prod_buffer,
                           vcl_size_t buffer_chunk_size,
                           vcl_size_t buffer_chunk_offset)
  {
    typedef NumericT        value_type;

    value_type         * Ap_buf      = detail::extract_raw_pointer<value_type>(Ap.handle()) + viennacl::traits::start(Ap);
    value_type   const *  p_buf      = detail::extract_raw_pointer<value_type>(p.handle()) + viennacl::traits::start(p);
    value_type   const * elements    = detail::extract_raw_pointer<value_type>(A.handle());
    unsigned int const *  row_buffer = detail::extract_raw_pointer<unsigned int>(A.handle1());
    unsigned int const *  col_buffer = detail::extract_raw_pointer<unsigned int>(A.handle2());
    value_type         * data_buffer = detail::extract_raw_pointer<value_type>(inner_prod_buffer);

    value_type inner_prod_ApAp = 0;
    value_type inner_prod_pAp = 0;
    value_type inner_prod_Ap_r0star = 0;
    for (long row = 0; row < static_cast<long>(A.size1()); ++row)
    {
      value_type dot_prod = 0;
      value_type val_p_diag = p_buf[static_cast<vcl_size_t>(row)]; //likely to be loaded from cache if required again in this row

      vcl_size_t row_end = row_buffer[row+1];
      for (vcl_size_t i = row_buffer[row]; i < row_end; ++i)
        dot_prod += elements[i] * p_buf[col_buffer[i]];

      // update contributions for the inner products (Ap, Ap) and (p, Ap)
      Ap_buf[static_cast<vcl_size_t>(row)] = dot_prod;
      inner_prod_ApAp += dot_prod * dot_prod;
      inner_prod_pAp  += val_p_diag * dot_prod;
      inner_prod_Ap_r0star += r0star ? dot_prod * r0star[static_cast<vcl_size_t>(row)] : value_type(0);
    }

    data_buffer[    buffer_chunk_size] = inner_prod_ApAp;
    data_buffer[2 * buffer_chunk_size] = inner_prod_pAp;
    if (r0star)
      data_buffer[buffer_chunk_offset] = inner_prod_Ap_r0star;
  }



  /** @brief Implementation of a fused matrix-vector product with a coordinate_matrix for an efficient pipelined CG algorithm.
    *
    * This routines computes for a matrix A and vectors 'p', 'Ap', and 'r0':
    *   Ap = prod(A, p);
    * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap), inner_prod(Ap, r0)
    */
  template<typename NumericT>
  void pipelined_prod_impl(coordinate_matrix<NumericT> const & A,
                           vector_base<NumericT> const & p,
                           vector_base<NumericT> & Ap,
                           NumericT const * r0star,
                           vector_base<NumericT> & inner_prod_buffer,
                           vcl_size_t buffer_chunk_size,
                           vcl_size_t buffer_chunk_offset)
  {
    typedef NumericT        value_type;

    value_type         * Ap_buf       = detail::extract_raw_pointer<value_type>(Ap.handle()) + viennacl::traits::start(Ap);;
    value_type   const *  p_buf       = detail::extract_raw_pointer<value_type>(p.handle()) + viennacl::traits::start(p);;
    value_type   const * elements     = detail::extract_raw_pointer<value_type>(A.handle());
    unsigned int const * coord_buffer = detail::extract_raw_pointer<unsigned int>(A.handle12());
    value_type         * data_buffer  = detail::extract_raw_pointer<value_type>(inner_prod_buffer);

    // flush result buffer (cannot be expected to be zero)
    for (vcl_size_t i = 0; i< Ap.size(); ++i)
      Ap_buf[i] = 0;

    // matrix-vector product with a general COO format
    for (vcl_size_t i = 0; i < A.nnz(); ++i)
      Ap_buf[coord_buffer[2*i]] += elements[i] * p_buf[coord_buffer[2*i+1]];

    // computing the inner products (Ap, Ap) and (p, Ap):
    // Note: The COO format does not allow to inject the subsequent operations into the matrix-vector product, because row and column ordering assumptions are too weak
    value_type inner_prod_ApAp = 0;
    value_type inner_prod_pAp = 0;
    value_type inner_prod_Ap_r0star = 0;
    for (vcl_size_t i = 0; i<Ap.size(); ++i)
    {
      NumericT value_Ap = Ap_buf[i];
      NumericT value_p  =  p_buf[i];

      inner_prod_ApAp += value_Ap * value_Ap;
      inner_prod_pAp  += value_Ap * value_p;
      inner_prod_Ap_r0star += r0star ? value_Ap * r0star[i] : value_type(0);
    }

    data_buffer[    buffer_chunk_size] = inner_prod_ApAp;
    data_buffer[2 * buffer_chunk_size] = inner_prod_pAp;
    if (r0star)
      data_buffer[buffer_chunk_offset] = inner_prod_Ap_r0star;
  }


  /** @brief Implementation of a fused matrix-vector product with an ell_matrix for an efficient pipelined CG algorithm.
    *
    * This routines computes for a matrix A and vectors 'p', 'Ap', and 'r0':
    *   Ap = prod(A, p);
    * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap), inner_prod(Ap, r0)
    */
  template<typename NumericT>
  void pipelined_prod_impl(ell_matrix<NumericT> const & A,
                           vector_base<NumericT> const & p,
                           vector_base<NumericT> & Ap,
                           NumericT const * r0star,
                           vector_base<NumericT> & inner_prod_buffer,
                           vcl_size_t buffer_chunk_size,
                           vcl_size_t buffer_chunk_offset)
  {
    typedef NumericT     value_type;

    value_type         * Ap_buf       = detail::extract_raw_pointer<value_type>(Ap.handle()) + viennacl::traits::start(Ap);;
    value_type   const *  p_buf       = detail::extract_raw_pointer<value_type>(p.handle()) + viennacl::traits::start(p);;
    value_type   const * elements     = detail::extract_raw_pointer<value_type>(A.handle());
    unsigned int const * coords       = detail::extract_raw_pointer<unsigned int>(A.handle2());
    value_type         * data_buffer  = detail::extract_raw_pointer<value_type>(inner_prod_buffer);

    value_type inner_prod_ApAp = 0;
    value_type inner_prod_pAp = 0;
    value_type inner_prod_Ap_r0star = 0;
    for (vcl_size_t row = 0; row < A.size1(); ++row)
    {
      value_type sum = 0;
      value_type val_p_diag = p_buf[static_cast<vcl_size_t>(row)]; //likely to be loaded from cache if required again in this row

      for (unsigned int item_id = 0; item_id < A.internal_maxnnz(); ++item_id)
      {
        vcl_size_t offset = row + item_id * A.internal_size1();
        value_type val = elements[offset];

        if (val)
          sum += (p_buf[coords[offset]] * val);
      }

      Ap_buf[row] = sum;
      inner_prod_ApAp += sum * sum;
      inner_prod_pAp  += val_p_diag * sum;
      inner_prod_Ap_r0star += r0star ? sum * r0star[row] : value_type(0);
    }

    data_buffer[    buffer_chunk_size] = inner_prod_ApAp;
    data_buffer[2 * buffer_chunk_size] = inner_prod_pAp;
    if (r0star)
      data_buffer[buffer_chunk_offset] = inner_prod_Ap_r0star;
  }


  /** @brief Implementation of a fused matrix-vector product with an sliced_ell_matrix for an efficient pipelined CG algorithm.
    *
    * This routines computes for a matrix A and vectors 'p', 'Ap', and 'r0':
    *   Ap = prod(A, p);
    * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap), inner_prod(Ap, r0)
    */
  template<typename NumericT, typename IndexT>
  void pipelined_prod_impl(sliced_ell_matrix<NumericT, IndexT> const & A,
                           vector_base<NumericT> const & p,
                           vector_base<NumericT> & Ap,
                           NumericT const * r0star,
                           vector_base<NumericT> & inner_prod_buffer,
                           vcl_size_t buffer_chunk_size,
                           vcl_size_t buffer_chunk_offset)
  {
    typedef NumericT     value_type;

    value_type       * Ap_buf            = detail::extract_raw_pointer<value_type>(Ap.handle()) + viennacl::traits::start(Ap);;
    value_type const *  p_buf            = detail::extract_raw_pointer<value_type>(p.handle()) + viennacl::traits::start(p);;
    value_type const * elements          = detail::extract_raw_pointer<value_type>(A.handle());
    IndexT     const * columns_per_block = detail::extract_raw_pointer<IndexT>(A.handle1());
    IndexT     const * column_indices    = detail::extract_raw_pointer<IndexT>(A.handle2());
    IndexT     const * block_start       = detail::extract_raw_pointer<IndexT>(A.handle3());
    value_type         * data_buffer     = detail::extract_raw_pointer<value_type>(inner_prod_buffer);

    vcl_size_t num_blocks = A.size1() / A.rows_per_block() + 1;
    std::vector<value_type> result_values(A.rows_per_block());

    value_type inner_prod_ApAp = 0;
    value_type inner_prod_pAp = 0;
    value_type inner_prod_Ap_r0star = 0;
    for (vcl_size_t block_idx = 0; block_idx < num_blocks; ++block_idx)
    {
      vcl_size_t current_columns_per_block = columns_per_block[block_idx];

      for (vcl_size_t i=0; i<result_values.size(); ++i)
        result_values[i] = 0;

      for (IndexT column_entry_index = 0;
                  column_entry_index < current_columns_per_block;
                ++column_entry_index)
      {
        vcl_size_t stride_start = block_start[block_idx] + column_entry_index * A.rows_per_block();
        // Note: This for-loop may be unrolled by hand for exploiting vectorization
        //       Careful benchmarking recommended first, memory channels may be saturated already!
        for (IndexT row_in_block = 0; row_in_block < A.rows_per_block(); ++row_in_block)
        {
          value_type val = elements[stride_start + row_in_block];

          result_values[row_in_block] += val ? p_buf[column_indices[stride_start + row_in_block]] * val : 0;
        }
      }

      vcl_size_t first_row_in_matrix = block_idx * A.rows_per_block();
      for (IndexT row_in_block = 0; row_in_block < A.rows_per_block(); ++row_in_block)
      {
        vcl_size_t row = first_row_in_matrix + row_in_block;
        if (row < Ap.size())
        {
          value_type row_result = result_values[row_in_block];

          Ap_buf[row] = row_result;
          inner_prod_ApAp += row_result * row_result;
          inner_prod_pAp  += p_buf[row] * row_result;
          inner_prod_Ap_r0star += r0star ? row_result * r0star[row] : value_type(0);
        }
      }
    }

    data_buffer[    buffer_chunk_size] = inner_prod_ApAp;
    data_buffer[2 * buffer_chunk_size] = inner_prod_pAp;
    if (r0star)
      data_buffer[buffer_chunk_offset] = inner_prod_Ap_r0star;
  }


  /** @brief Implementation of a fused matrix-vector product with an hyb_matrix for an efficient pipelined CG algorithm.
    *
    * This routines computes for a matrix A and vectors 'p', 'Ap', and 'r0':
    *   Ap = prod(A, p);
    * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap), inner_prod(Ap, r0)
    */
  template<typename NumericT>
  void pipelined_prod_impl(hyb_matrix<NumericT> const & A,
                           vector_base<NumericT> const & p,
                           vector_base<NumericT> & Ap,
                           NumericT const * r0star,
                           vector_base<NumericT> & inner_prod_buffer,
                           vcl_size_t buffer_chunk_size,
                           vcl_size_t buffer_chunk_offset)
  {
    typedef NumericT     value_type;
    typedef unsigned int index_type;

    value_type       * Ap_buf            = detail::extract_raw_pointer<value_type>(Ap.handle()) + viennacl::traits::start(Ap);;
    value_type const *  p_buf            = detail::extract_raw_pointer<value_type>(p.handle()) + viennacl::traits::start(p);;
    value_type const * elements          = detail::extract_raw_pointer<value_type>(A.handle());
    index_type const * coords            = detail::extract_raw_pointer<index_type>(A.handle2());
    value_type const * csr_elements      = detail::extract_raw_pointer<value_type>(A.handle5());
    index_type const * csr_row_buffer    = detail::extract_raw_pointer<index_type>(A.handle3());
    index_type const * csr_col_buffer    = detail::extract_raw_pointer<index_type>(A.handle4());
    value_type         * data_buffer     = detail::extract_raw_pointer<value_type>(inner_prod_buffer);

    value_type inner_prod_ApAp = 0;
    value_type inner_prod_pAp = 0;
    value_type inner_prod_Ap_r0star = 0;
    for (vcl_size_t row = 0; row < A.size1(); ++row)
    {
      value_type val_p_diag = p_buf[static_cast<vcl_size_t>(row)]; //likely to be loaded from cache if required again in this row
      value_type sum = 0;

      //
      // Part 1: Process ELL part
      //
      for (index_type item_id = 0; item_id < A.internal_ellnnz(); ++item_id)
      {
        vcl_size_t offset = row + item_id * A.internal_size1();
        value_type val = elements[offset];

        if (val)
          sum += p_buf[coords[offset]] * val;
      }

      //
      // Part 2: Process HYB part
      //
      vcl_size_t col_begin = csr_row_buffer[row];
      vcl_size_t col_end   = csr_row_buffer[row + 1];

      for (vcl_size_t item_id = col_begin; item_id < col_end; item_id++)
        sum += p_buf[csr_col_buffer[item_id]] * csr_elements[item_id];

      Ap_buf[row] = sum;
      inner_prod_ApAp += sum * sum;
      inner_prod_pAp  += val_p_diag * sum;
      inner_prod_Ap_r0star += r0star ? sum * r0star[row] : value_type(0);
    }

    data_buffer[    buffer_chunk_size] = inner_prod_ApAp;
    data_buffer[2 * buffer_chunk_size] = inner_prod_pAp;
    if (r0star)
      data_buffer[buffer_chunk_offset] = inner_prod_Ap_r0star;
  }

} // namespace detail


/** @brief Performs a joint vector update operation needed for an efficient pipelined CG algorithm.
  *
  * This routines computes for vectors 'result', 'p', 'r', 'Ap':
  *   result += alpha * p;
  *   r      -= alpha * Ap;
  *   p       = r + beta * p;
  * and runs the parallel reduction stage for computing inner_prod(r,r)
  */
template<typename NumericT>
void pipelined_cg_vector_update(vector_base<NumericT> & result,
                                NumericT alpha,
                                vector_base<NumericT> & p,
                                vector_base<NumericT> & r,
                                vector_base<NumericT> const & Ap,
                                NumericT beta,
                                vector_base<NumericT> & inner_prod_buffer)
{
  typedef NumericT       value_type;

  value_type       * data_result = detail::extract_raw_pointer<value_type>(result);
  value_type       * data_p      = detail::extract_raw_pointer<value_type>(p);
  value_type       * data_r      = detail::extract_raw_pointer<value_type>(r);
  value_type const * data_Ap     = detail::extract_raw_pointer<value_type>(Ap);
  value_type       * data_buffer = detail::extract_raw_pointer<value_type>(inner_prod_buffer);

  // Note: Due to the special setting in CG, there is no need to check for sizes and strides
  vcl_size_t size  = viennacl::traits::size(result);

  value_type inner_prod_r = 0;
  for (long i = 0; i < static_cast<long>(size); ++i)
  {
    value_type value_p = data_p[static_cast<vcl_size_t>(i)];
    value_type value_r = data_r[static_cast<vcl_size_t>(i)];


    data_result[static_cast<vcl_size_t>(i)] += alpha * value_p;
    value_r -= alpha * data_Ap[static_cast<vcl_size_t>(i)];
    value_p  = value_r + beta * value_p;
    inner_prod_r += value_r * value_r;

    data_p[static_cast<vcl_size_t>(i)] = value_p;
    data_r[static_cast<vcl_size_t>(i)] = value_r;
  }

  data_buffer[0] = inner_prod_r;
}


/** @brief Performs a fused matrix-vector product with a compressed_matrix for an efficient pipelined CG algorithm.
  *
  * This routines computes for a matrix A and vectors 'p' and 'Ap':
  *   Ap = prod(A, p);
  * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap)
  */
template<typename NumericT>
void pipelined_cg_prod(compressed_matrix<NumericT> const & A,
                       vector_base<NumericT> const & p,
                       vector_base<NumericT> & Ap,
                       vector_base<NumericT> & inner_prod_buffer)
{
  typedef NumericT const *    PtrType;
  viennacl::linalg::host_based::detail::pipelined_prod_impl(A, p, Ap, PtrType(NULL), inner_prod_buffer, inner_prod_buffer.size() / 3, 0);
}



/** @brief Performs a fused matrix-vector product with a coordinate_matrix for an efficient pipelined CG algorithm.
  *
  * This routines computes for a matrix A and vectors 'p' and 'Ap':
  *   Ap = prod(A, p);
  * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap)
  */
template<typename NumericT>
void pipelined_cg_prod(coordinate_matrix<NumericT> const & A,
                       vector_base<NumericT> const & p,
                       vector_base<NumericT> & Ap,
                       vector_base<NumericT> & inner_prod_buffer)
{
  typedef NumericT const *    PtrType;
  viennacl::linalg::host_based::detail::pipelined_prod_impl(A, p, Ap, PtrType(NULL), inner_prod_buffer, inner_prod_buffer.size() / 3, 0);
}


/** @brief Performs a fused matrix-vector product with an ell_matrix for an efficient pipelined CG algorithm.
  *
  * This routines computes for a matrix A and vectors 'p' and 'Ap':
  *   Ap = prod(A, p);
  * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap)
  */
template<typename NumericT>
void pipelined_cg_prod(ell_matrix<NumericT> const & A,
                       vector_base<NumericT> const & p,
                       vector_base<NumericT> & Ap,
                       vector_base<NumericT> & inner_prod_buffer)
{
  typedef NumericT const *    PtrType;
  viennacl::linalg::host_based::detail::pipelined_prod_impl(A, p, Ap, PtrType(NULL), inner_prod_buffer, inner_prod_buffer.size() / 3, 0);
}


/** @brief Performs a fused matrix-vector product with an sliced_ell_matrix for an efficient pipelined CG algorithm.
  *
  * This routines computes for a matrix A and vectors 'p' and 'Ap':
  *   Ap = prod(A, p);
  * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap)
  */
template<typename NumericT, typename IndexT>
void pipelined_cg_prod(sliced_ell_matrix<NumericT, IndexT> const & A,
                       vector_base<NumericT> const & p,
                       vector_base<NumericT> & Ap,
                       vector_base<NumericT> & inner_prod_buffer)
{
  typedef NumericT const *    PtrType;
  viennacl::linalg::host_based::detail::pipelined_prod_impl(A, p, Ap, PtrType(NULL), inner_prod_buffer, inner_prod_buffer.size() / 3, 0);
}




/** @brief Performs a fused matrix-vector product with an hyb_matrix for an efficient pipelined CG algorithm.
  *
  * This routines computes for a matrix A and vectors 'p' and 'Ap':
  *   Ap = prod(A, p);
  * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap)
  */
template<typename NumericT>
void pipelined_cg_prod(hyb_matrix<NumericT> const & A,
                       vector_base<NumericT> const & p,
                       vector_base<NumericT> & Ap,
                       vector_base<NumericT> & inner_prod_buffer)
{
  typedef NumericT const *    PtrType;
  viennacl::linalg::host_based::detail::pipelined_prod_impl(A, p, Ap, PtrType(NULL), inner_prod_buffer, inner_prod_buffer.size() / 3, 0);
}

//////////////////////////


/** @brief Performs a joint vector update operation needed for an efficient pipelined BiCGStab algorithm.
  *
  * This routines computes for vectors 's', 'r', 'Ap':
  *   s = r - alpha * Ap
  * with alpha obtained from a reduction step on the 0th and the 3rd out of 6 chunks in inner_prod_buffer
  * and runs the parallel reduction stage for computing inner_prod(s,s)
  */
template<typename NumericT>
void pipelined_bicgstab_update_s(vector_base<NumericT> & s,
                                 vector_base<NumericT> & r,
                                 vector_base<NumericT> const & Ap,
                                 vector_base<NumericT> & inner_prod_buffer,
                                 vcl_size_t buffer_chunk_size,
                                 vcl_size_t buffer_chunk_offset)
{
  typedef NumericT      value_type;

  value_type       * data_s      = detail::extract_raw_pointer<value_type>(s);
  value_type       * data_r      = detail::extract_raw_pointer<value_type>(r);
  value_type const * data_Ap     = detail::extract_raw_pointer<value_type>(Ap);
  value_type       * data_buffer = detail::extract_raw_pointer<value_type>(inner_prod_buffer);

  // Note: Due to the special setting in CG, there is no need to check for sizes and strides
  vcl_size_t size  = viennacl::traits::size(s);

  // part 1: compute alpha:
  value_type r_in_r0 = 0;
  value_type Ap_in_r0 = 0;
  for (vcl_size_t i=0; i<buffer_chunk_size; ++i)
  {
     r_in_r0 += data_buffer[i];
    Ap_in_r0 += data_buffer[i + 3 * buffer_chunk_size];
  }
  value_type alpha = r_in_r0 / Ap_in_r0;

  // part 2: s = r - alpha * Ap  and first step in reduction for s:
  value_type inner_prod_s = 0;
  for (long i = 0; i < static_cast<long>(size); ++i)
  {
    value_type value_s  = data_s[static_cast<vcl_size_t>(i)];

    value_s = data_r[static_cast<vcl_size_t>(i)] - alpha * data_Ap[static_cast<vcl_size_t>(i)];
    inner_prod_s += value_s * value_s;

    data_s[static_cast<vcl_size_t>(i)] = value_s;
  }

  data_buffer[buffer_chunk_offset] = inner_prod_s;
}

/** @brief Performs a joint vector update operation needed for an efficient pipelined BiCGStab algorithm.
  *
  * x_{j+1} = x_j + alpha * p_j + omega * s_j
  * r_{j+1} = s_j - omega * t_j
  * p_{j+1} = r_{j+1} + beta * (p_j - omega * q_j)
  * and compute first stage of r_dot_r0 = <r_{j+1}, r_o^*> for use in next iteration
  */
 template<typename NumericT>
 void pipelined_bicgstab_vector_update(vector_base<NumericT> & result, NumericT alpha, vector_base<NumericT> & p, NumericT omega, vector_base<NumericT> const & s,
                                       vector_base<NumericT> & residual, vector_base<NumericT> const & As,
                                       NumericT beta, vector_base<NumericT> const & Ap,
                                       vector_base<NumericT> const & r0star,
                                       vector_base<NumericT>       & inner_prod_buffer,
                                       vcl_size_t buffer_chunk_size)
 {
   typedef NumericT    value_type;

   value_type       * data_result   = detail::extract_raw_pointer<value_type>(result);
   value_type       * data_p        = detail::extract_raw_pointer<value_type>(p);
   value_type const * data_s        = detail::extract_raw_pointer<value_type>(s);
   value_type       * data_residual = detail::extract_raw_pointer<value_type>(residual);
   value_type const * data_As       = detail::extract_raw_pointer<value_type>(As);
   value_type const * data_Ap       = detail::extract_raw_pointer<value_type>(Ap);
   value_type const * data_r0star   = detail::extract_raw_pointer<value_type>(r0star);
   value_type       * data_buffer   = detail::extract_raw_pointer<value_type>(inner_prod_buffer);

   vcl_size_t size = viennacl::traits::size(result);

   value_type inner_prod_r_r0star = 0;
   for (long i = 0; i < static_cast<long>(size); ++i)
   {
     vcl_size_t index = static_cast<vcl_size_t>(i);
     value_type value_result   = data_result[index];
     value_type value_p        = data_p[index];
     value_type value_s        = data_s[index];
     value_type value_residual = data_residual[index];
     value_type value_As       = data_As[index];
     value_type value_Ap       = data_Ap[index];
     value_type value_r0star   = data_r0star[index];

     value_result   += alpha * value_p + omega * value_s;
     value_residual  = value_s - omega * value_As;
     value_p         = value_residual + beta * (value_p - omega * value_Ap);
     inner_prod_r_r0star += value_residual * value_r0star;

     data_result[index]   = value_result;
     data_residual[index] = value_residual;
     data_p[index]        = value_p;
   }

   (void)buffer_chunk_size; // not needed here, just silence compiler warning (unused variable)
   data_buffer[0] = inner_prod_r_r0star;
 }

 /** @brief Performs a fused matrix-vector product with a compressed_matrix for an efficient pipelined BiCGStab algorithm.
   *
   * This routines computes for a matrix A and vectors 'p', 'Ap', and 'r0':
   *   Ap = prod(A, p);
   * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap), inner_prod(Ap, r0)
   */
 template<typename NumericT>
 void pipelined_bicgstab_prod(compressed_matrix<NumericT> const & A,
                              vector_base<NumericT> const & p,
                              vector_base<NumericT> & Ap,
                              vector_base<NumericT> const & r0star,
                              vector_base<NumericT> & inner_prod_buffer,
                              vcl_size_t buffer_chunk_size,
                              vcl_size_t buffer_chunk_offset)
 {
   NumericT const * data_r0star   = detail::extract_raw_pointer<NumericT>(r0star);

   viennacl::linalg::host_based::detail::pipelined_prod_impl(A, p, Ap, data_r0star, inner_prod_buffer, buffer_chunk_size, buffer_chunk_offset);
 }

 /** @brief Performs a fused matrix-vector product with a coordinate_matrix for an efficient pipelined BiCGStab algorithm.
   *
   * This routines computes for a matrix A and vectors 'p', 'Ap', and 'r0':
   *   Ap = prod(A, p);
   * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap), inner_prod(Ap, r0)
   */
 template<typename NumericT>
 void pipelined_bicgstab_prod(coordinate_matrix<NumericT> const & A,
                              vector_base<NumericT> const & p,
                              vector_base<NumericT> & Ap,
                              vector_base<NumericT> const & r0star,
                              vector_base<NumericT> & inner_prod_buffer,
                              vcl_size_t buffer_chunk_size,
                              vcl_size_t buffer_chunk_offset)
 {
   NumericT const * data_r0star   = detail::extract_raw_pointer<NumericT>(r0star);

   viennacl::linalg::host_based::detail::pipelined_prod_impl(A, p, Ap, data_r0star, inner_prod_buffer, buffer_chunk_size, buffer_chunk_offset);
 }

 /** @brief Performs a fused matrix-vector product with an ell_matrix for an efficient pipelined BiCGStab algorithm.
   *
   * This routines computes for a matrix A and vectors 'p', 'Ap', and 'r0':
   *   Ap = prod(A, p);
   * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap), inner_prod(Ap, r0)
   */
 template<typename NumericT>
 void pipelined_bicgstab_prod(ell_matrix<NumericT> const & A,
                              vector_base<NumericT> const & p,
                              vector_base<NumericT> & Ap,
                              vector_base<NumericT> const & r0star,
                              vector_base<NumericT> & inner_prod_buffer,
                              vcl_size_t buffer_chunk_size,
                              vcl_size_t buffer_chunk_offset)
 {
   NumericT const * data_r0star   = detail::extract_raw_pointer<NumericT>(r0star);

   viennacl::linalg::host_based::detail::pipelined_prod_impl(A, p, Ap, data_r0star, inner_prod_buffer, buffer_chunk_size, buffer_chunk_offset);
 }

 /** @brief Performs a fused matrix-vector product with a sliced_ell_matrix for an efficient pipelined BiCGStab algorithm.
   *
   * This routines computes for a matrix A and vectors 'p', 'Ap', and 'r0':
   *   Ap = prod(A, p);
   * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap), inner_prod(Ap, r0)
   */
 template<typename NumericT, typename IndexT>
 void pipelined_bicgstab_prod(sliced_ell_matrix<NumericT, IndexT> const & A,
                              vector_base<NumericT> const & p,
                              vector_base<NumericT> & Ap,
                              vector_base<NumericT> const & r0star,
                              vector_base<NumericT> & inner_prod_buffer,
                              vcl_size_t buffer_chunk_size,
                              vcl_size_t buffer_chunk_offset)
 {
   NumericT const * data_r0star   = detail::extract_raw_pointer<NumericT>(r0star);

   viennacl::linalg::host_based::detail::pipelined_prod_impl(A, p, Ap, data_r0star, inner_prod_buffer, buffer_chunk_size, buffer_chunk_offset);
 }

 /** @brief Performs a fused matrix-vector product with a hyb_matrix for an efficient pipelined BiCGStab algorithm.
   *
   * This routines computes for a matrix A and vectors 'p', 'Ap', and 'r0':
   *   Ap = prod(A, p);
   * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap), inner_prod(Ap, r0)
   */
 template<typename NumericT>
 void pipelined_bicgstab_prod(hyb_matrix<NumericT> const & A,
                              vector_base<NumericT> const & p,
                              vector_base<NumericT> & Ap,
                              vector_base<NumericT> const & r0star,
                              vector_base<NumericT> & inner_prod_buffer,
                              vcl_size_t buffer_chunk_size,
                              vcl_size_t buffer_chunk_offset)
 {
   NumericT const * data_r0star   = detail::extract_raw_pointer<NumericT>(r0star);

   viennacl::linalg::host_based::detail::pipelined_prod_impl(A, p, Ap, data_r0star, inner_prod_buffer, buffer_chunk_size, buffer_chunk_offset);
 }


/////////////////////////////////////////////////////////////

/** @brief Performs a vector normalization needed for an efficient pipelined GMRES algorithm.
 *
 * This routines computes for vectors 'r', 'v_k':
 *   Second reduction step for ||v_k||
 *   v_k /= ||v_k||
 *   First reduction step for <r, v_k>
 */
template <typename T>
void pipelined_gmres_normalize_vk(vector_base<T> & v_k,
                                 vector_base<T> const & residual,
                                 vector_base<T> & R_buffer,
                                 vcl_size_t offset_in_R,
                                 vector_base<T> const & inner_prod_buffer,
                                 vector_base<T> & r_dot_vk_buffer,
                                 vcl_size_t buffer_chunk_size,
                                 vcl_size_t buffer_chunk_offset)
{
  typedef T        value_type;

  value_type       * data_v_k      = detail::extract_raw_pointer<value_type>(v_k);
  value_type const * data_residual = detail::extract_raw_pointer<value_type>(residual);
  value_type       * data_R        = detail::extract_raw_pointer<value_type>(R_buffer);
  value_type const * data_buffer   = detail::extract_raw_pointer<value_type>(inner_prod_buffer);
  value_type       * data_r_dot_vk = detail::extract_raw_pointer<value_type>(r_dot_vk_buffer);

  // Note: Due to the special setting in GMRES, there is no need to check for sizes and strides
  vcl_size_t size     = viennacl::traits::size(v_k);
  vcl_size_t vk_start = viennacl::traits::start(v_k);

  // part 1: compute alpha:
  value_type norm_vk = 0;
  for (vcl_size_t i=0; i<buffer_chunk_size; ++i)
   norm_vk += data_buffer[i + buffer_chunk_size];
  norm_vk = std::sqrt(norm_vk);
  data_R[offset_in_R] = norm_vk;

  // Compute <r, v_k> after normalization of v_k:
  value_type inner_prod_r_dot_vk = 0;
  for (long i = 0; i < static_cast<long>(size); ++i)
  {
    value_type value_vk = data_v_k[static_cast<vcl_size_t>(i) + vk_start] / norm_vk;

    inner_prod_r_dot_vk += data_residual[static_cast<vcl_size_t>(i)] * value_vk;

    data_v_k[static_cast<vcl_size_t>(i) + vk_start] = value_vk;
  }

  data_r_dot_vk[buffer_chunk_offset] = inner_prod_r_dot_vk;
}



/** @brief Computes first reduction stage for multiple inner products <v_i, v_k>, i=0..k-1
 *
 *  All vectors v_i are stored column-major in the array 'device_krylov_basis', where each vector has an actual length 'v_k_size', but might be padded to have 'v_k_internal_size'
 */
template <typename T>
void pipelined_gmres_gram_schmidt_stage1(vector_base<T> const & device_krylov_basis,
                                        vcl_size_t v_k_size,
                                        vcl_size_t v_k_internal_size,
                                        vcl_size_t k,
                                        vector_base<T> & vi_in_vk_buffer,
                                        vcl_size_t buffer_chunk_size)
{
  typedef T        value_type;

  value_type const * data_krylov_basis = detail::extract_raw_pointer<value_type>(device_krylov_basis);
  value_type       * data_inner_prod   = detail::extract_raw_pointer<value_type>(vi_in_vk_buffer);

  // reset buffer:
  for (vcl_size_t j = 0; j < k; ++j)
    data_inner_prod[j*buffer_chunk_size] = value_type(0);

  // compute inner products:
  for (vcl_size_t i = 0; i < v_k_size; ++i)
  {
    value_type value_vk = data_krylov_basis[static_cast<vcl_size_t>(i) + k * v_k_internal_size];

    for (vcl_size_t j = 0; j < k; ++j)
      data_inner_prod[j*buffer_chunk_size] += data_krylov_basis[static_cast<vcl_size_t>(i) + j * v_k_internal_size] * value_vk;
  }
}


/** @brief Computes the second reduction stage for multiple inner products <v_i, v_k>, i=0..k-1, then updates v_k -= <v_i, v_k> v_i and computes the first reduction stage for ||v_k||
 *
 *  All vectors v_i are stored column-major in the array 'device_krylov_basis', where each vector has an actual length 'v_k_size', but might be padded to have 'v_k_internal_size'
 */
template <typename T>
void pipelined_gmres_gram_schmidt_stage2(vector_base<T> & device_krylov_basis,
                                        vcl_size_t v_k_size,
                                        vcl_size_t v_k_internal_size,
                                        vcl_size_t k,
                                        vector_base<T> const & vi_in_vk_buffer,
                                        vector_base<T> & R_buffer,
                                        vcl_size_t krylov_dim,
                                        vector_base<T> & inner_prod_buffer,
                                        vcl_size_t buffer_chunk_size)
{
  typedef T        value_type;

  value_type * data_krylov_basis = detail::extract_raw_pointer<value_type>(device_krylov_basis);

  std::vector<T> values_vi_in_vk(k);

  // Step 1: Finish reduction of <v_i, v_k> to obtain scalars:
  for (std::size_t i=0; i<k; ++i)
    for (vcl_size_t j=0; j<buffer_chunk_size; ++j)
      values_vi_in_vk[i] += vi_in_vk_buffer[i*buffer_chunk_size + j];


  // Step 2: Compute v_k -= <v_i, v_k> v_i and reduction on ||v_k||:
  value_type norm_vk = 0;
  for (vcl_size_t i = 0; i < v_k_size; ++i)
  {
    value_type value_vk = data_krylov_basis[static_cast<vcl_size_t>(i) + k * v_k_internal_size];

    for (vcl_size_t j = 0; j < k; ++j)
      value_vk -= values_vi_in_vk[j] * data_krylov_basis[static_cast<vcl_size_t>(i) + j * v_k_internal_size];

    norm_vk += value_vk * value_vk;
    data_krylov_basis[static_cast<vcl_size_t>(i) + k * v_k_internal_size] = value_vk;
  }

  // Step 3: Write values to R_buffer:
  for (std::size_t i=0; i<k; ++i)
    R_buffer[i + k * krylov_dim] = values_vi_in_vk[i];

  inner_prod_buffer[buffer_chunk_size] = norm_vk;
}

/** @brief Computes x += eta_0 r + sum_{i=1}^{k-1} eta_i v_{i-1} */
template <typename T>
void pipelined_gmres_update_result(vector_base<T> & result,
                                  vector_base<T> const & residual,
                                  vector_base<T> const & krylov_basis,
                                  vcl_size_t v_k_size,
                                  vcl_size_t v_k_internal_size,
                                  vector_base<T> const & coefficients,
                                  vcl_size_t k)
{
  typedef T        value_type;

  value_type       * data_result       = detail::extract_raw_pointer<value_type>(result);
  value_type const * data_residual     = detail::extract_raw_pointer<value_type>(residual);
  value_type const * data_krylov_basis = detail::extract_raw_pointer<value_type>(krylov_basis);
  value_type const * data_coefficients = detail::extract_raw_pointer<value_type>(coefficients);

  for (vcl_size_t i = 0; i < v_k_size; ++i)
  {
    value_type value_result = data_result[i];

    value_result += data_coefficients[0] * data_residual[i];
    for (vcl_size_t j = 1; j<k; ++j)
      value_result += data_coefficients[j] * data_krylov_basis[i + (j-1) * v_k_internal_size];

    data_result[i] = value_result;
  }

}

// Reuse implementation from CG:
template <typename MatrixType, typename T>
void pipelined_gmres_prod(MatrixType const & A,
                      vector_base<T> const & p,
                      vector_base<T> & Ap,
                      vector_base<T> & inner_prod_buffer)
{
  pipelined_cg_prod(A, p, Ap, inner_prod_buffer);
}


} //namespace host_based
} //namespace linalg
} //namespace viennacl


#endif
