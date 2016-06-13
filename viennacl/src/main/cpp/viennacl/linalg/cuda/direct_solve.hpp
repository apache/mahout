#ifndef VIENNACL_LINALG_CUDA_DIRECT_SOLVE_HPP
#define VIENNACL_LINALG_CUDA_DIRECT_SOLVE_HPP

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

/** @file viennacl/linalg/cuda/direct_solve.hpp
    @brief Implementations of dense direct solvers using CUDA are found here.
*/

#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"


#include "viennacl/linalg/cuda/common.hpp"


namespace viennacl
{
namespace linalg
{
namespace cuda
{

template<typename NumericT>
__global__ void matrix_matrix_upper_solve_kernel(
          const NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1, unsigned int A_internal_size2,
          bool row_major_A,

          NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_size1,  unsigned int B_size2,
          unsigned int B_internal_size1, unsigned int B_internal_size2,
          bool row_major_B,

          bool unit_diagonal)
{
  NumericT temp;
  NumericT entry_A;

  for (unsigned int row_cnt = 0; row_cnt < A_size1; ++row_cnt)
  {
    unsigned int row = A_size1 - 1 - row_cnt;

    if (!unit_diagonal)
    {
      __syncthreads();

      if (threadIdx.x == 0)
      {
        if (row_major_B)
          B[(row * B_inc1 + B_start1) * B_internal_size2 + (blockIdx.x * B_inc2 + B_start2)] /= (row_major_A) ? A[(row * A_inc1 + A_start1) * A_internal_size2 + (row * A_inc2 + A_start2)]
                                                                                                              : A[(row * A_inc1 + A_start1) + (row * A_inc2 + A_start2)*A_internal_size1];
        else //if (!row_major_B)
          B[(row * B_inc1 + B_start1) + (blockIdx.x * B_inc2 + B_start2) * B_internal_size1] /= (row_major_A) ? A[(row * A_inc1 + A_start1) * A_internal_size2 + (row * A_inc2 + A_start2)]
                                                                                                              : A[(row * A_inc1 + A_start1) + (row * A_inc2 + A_start2)*A_internal_size1];
      }
    }

    __syncthreads();

    if (row_major_B)
      temp = B[(row * B_inc1 + B_start1) * B_internal_size2 + (blockIdx.x * B_inc2 + B_start2)];
    else //if (!row_major_B)
      temp = B[(row * B_inc1 + B_start1) + (blockIdx.x * B_inc2 + B_start2) * B_internal_size1];

    //eliminate column of op(A) with index 'row' in parallel: " << std::endl;
    for  (unsigned int elim = threadIdx.x; elim < row; elim += blockDim.x)
    {
      if (row_major_A)
        entry_A = A[(elim * A_inc1 + A_start1) * A_internal_size2 + (row * A_inc2 + A_start2)];
      else //if (!row_major_A)
        entry_A = A[(elim * A_inc1 + A_start1) + (row * A_inc2 + A_start2) * A_internal_size1];

      if (row_major_B)
        B[(elim * B_inc1 + B_start1) * B_internal_size2 + (blockIdx.x * B_inc2 + B_start2)] -= temp * entry_A;
      else //if (!row_major_B)
        B[(elim * B_inc1 + B_start1) + (blockIdx.x * B_inc2 + B_start2) * B_internal_size1] -= temp * entry_A;

    }
  }
}



template<typename NumericT>
__global__ void matrix_matrix_lower_solve_kernel(
          const NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1, unsigned int A_internal_size2,
          bool row_major_A,

          NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_size1,  unsigned int B_size2,
          unsigned int B_internal_size1, unsigned int B_internal_size2,
          bool row_major_B,

          bool unit_diagonal)
{
  NumericT temp;
  NumericT entry_A;

  for (unsigned int row = 0; row < A_size1; ++row)
  {

    if (!unit_diagonal)
    {
      __syncthreads();

      if (threadIdx.x == 0)
      {
        if (row_major_B)
          B[(row * B_inc1 + B_start1) * B_internal_size2 + (blockIdx.x * B_inc2 + B_start2)] /= (row_major_A) ? A[(row * A_inc1 + A_start1) * A_internal_size2 + (row * A_inc2 + A_start2)]
                                                                                                              : A[(row * A_inc1 + A_start1) + (row * A_inc2 + A_start2)*A_internal_size1];
        else //if (!row_major_B)
          B[(row * B_inc1 + B_start1) + (blockIdx.x * B_inc2 + B_start2) * B_internal_size1] /= (row_major_A) ? A[(row * A_inc1 + A_start1) * A_internal_size2 + (row * A_inc2 + A_start2)]
                                                                                                              : A[(row * A_inc1 + A_start1) + (row * A_inc2 + A_start2)*A_internal_size1];
      }
    }

    __syncthreads();

    if (row_major_B)
      temp = B[(row * B_inc1 + B_start1) * B_internal_size2 + (blockIdx.x * B_inc2 + B_start2)];
    else //if (!row_major_B)
      temp = B[(row * B_inc1 + B_start1) + (blockIdx.x * B_inc2 + B_start2) * B_internal_size1];

    //eliminate column of op(A) with index 'row' in parallel: " << std::endl;
    for  (unsigned int elim = row + threadIdx.x + 1; elim < A_size1; elim += blockDim.x)
    {
      if (row_major_A)
        entry_A = A[(elim * A_inc1 + A_start1) * A_internal_size2 + (row * A_inc2 + A_start2)];
      else //if (!row_major_A)
        entry_A = A[(elim * A_inc1 + A_start1) + (row * A_inc2 + A_start2) * A_internal_size1];

      if (row_major_B)
        B[(elim * B_inc1 + B_start1) * B_internal_size2 + (blockIdx.x * B_inc2 + B_start2)] -= temp * entry_A;
      else //if (!row_major_B)
        B[(elim * B_inc1 + B_start1) + (blockIdx.x * B_inc2 + B_start2) * B_internal_size1] -= temp * entry_A;

    }
  }
}






namespace detail
{
  template<typename TagT>
  bool is_unit_solve(TagT const & tag) { return false; }

  inline bool is_unit_solve(viennacl::linalg::unit_lower_tag) { return true; }
  inline bool is_unit_solve(viennacl::linalg::unit_upper_tag) { return true; }

  template<typename TagT>
  bool is_upper_solve(TagT const & tag) { return false; }

  inline bool is_upper_solve(viennacl::linalg::upper_tag) { return true; }
  inline bool is_upper_solve(viennacl::linalg::unit_upper_tag) { return true; }

  template<typename Matrix1T, typename Matrix2T, typename SolverTagT>
  void inplace_solve_impl(Matrix1T const & A,
                          Matrix2T & B,
                          SolverTagT const & tag)
  {
    typedef typename viennacl::result_of::cpu_value_type<Matrix1T>::type        value_type;

    dim3 threads(128);
    dim3 grid(B.size2());

    if (is_upper_solve(tag))
    {
      matrix_matrix_upper_solve_kernel<<<grid,threads>>>(viennacl::cuda_arg(A),
                                                         static_cast<unsigned int>(viennacl::traits::start1(A)),         static_cast<unsigned int>(viennacl::traits::start2(A)),
                                                         static_cast<unsigned int>(viennacl::traits::stride1(A)),        static_cast<unsigned int>(viennacl::traits::stride2(A)),
                                                         static_cast<unsigned int>(viennacl::traits::size1(A)),          static_cast<unsigned int>(viennacl::traits::size2(A)),
                                                         static_cast<unsigned int>(viennacl::traits::internal_size1(A)), static_cast<unsigned int>(viennacl::traits::internal_size2(A)),
                                                         bool(A.row_major()),

                                                         viennacl::cuda_arg(B),
                                                         static_cast<unsigned int>(viennacl::traits::start1(B)),         static_cast<unsigned int>(viennacl::traits::start2(B)),
                                                         static_cast<unsigned int>(viennacl::traits::stride1(B)),        static_cast<unsigned int>(viennacl::traits::stride2(B)),
                                                         static_cast<unsigned int>(viennacl::traits::size1(B)),          static_cast<unsigned int>(viennacl::traits::size2(B)),
                                                         static_cast<unsigned int>(viennacl::traits::internal_size1(B)), static_cast<unsigned int>(viennacl::traits::internal_size2(B)),
                                                         bool(B.row_major()),

                                                         is_unit_solve(tag)
                                                        );
    }
    else
    {
      matrix_matrix_lower_solve_kernel<<<grid,threads>>>(viennacl::cuda_arg(A),
                                                         static_cast<unsigned int>(viennacl::traits::start1(A)),         static_cast<unsigned int>(viennacl::traits::start2(A)),
                                                         static_cast<unsigned int>(viennacl::traits::stride1(A)),        static_cast<unsigned int>(viennacl::traits::stride2(A)),
                                                         static_cast<unsigned int>(viennacl::traits::size1(A)),          static_cast<unsigned int>(viennacl::traits::size2(A)),
                                                         static_cast<unsigned int>(viennacl::traits::internal_size1(A)), static_cast<unsigned int>(viennacl::traits::internal_size2(A)),
                                                         bool(A.row_major()),

                                                         viennacl::cuda_arg(B),
                                                         static_cast<unsigned int>(viennacl::traits::start1(B)),         static_cast<unsigned int>(viennacl::traits::start2(B)),
                                                         static_cast<unsigned int>(viennacl::traits::stride1(B)),        static_cast<unsigned int>(viennacl::traits::stride2(B)),
                                                         static_cast<unsigned int>(viennacl::traits::size1(B)),          static_cast<unsigned int>(viennacl::traits::size2(B)),
                                                         static_cast<unsigned int>(viennacl::traits::internal_size1(B)), static_cast<unsigned int>(viennacl::traits::internal_size2(B)),
                                                         bool(B.row_major()),

                                                         is_unit_solve(tag)
                                                        );
    }

  }
}


//
// Note: By convention, all size checks are performed in the calling frontend. No need to double-check here.
//

////////////////// triangular solver //////////////////////////////////////
/** @brief Direct inplace solver for triangular systems with multiple right hand sides, i.e. A \ B   (MATLAB notation).
*
* @param A         The system matrix
* @param B         The matrix of row vectors, where the solution is directly written to
* @param tag       Solver tag for identifying the respective triangular solver
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(matrix_base<NumericT> const & A,
                   matrix_base<NumericT> & B,
                   SolverTagT tag)
{
  detail::inplace_solve_impl(A, B, tag);
}


//
//  Solve on vector
//

template<typename NumericT>
__global__ void triangular_substitute_inplace_row_kernel(
          NumericT const * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,
          NumericT * v,
          unsigned int v_start,
          unsigned int v_inc,
          unsigned int v_size,

          unsigned int options)
{
  NumericT temp;
  unsigned int unit_diagonal_flag  = (options & (1 << 0));

  unsigned int is_lower_solve      = (options & (1 << 2));
  unsigned int row;
  for (unsigned int rows_processed = 0; rows_processed < A_size1; ++rows_processed)    //Note: A required to be square
  {
    row = is_lower_solve ? rows_processed : ((A_size1 - rows_processed) - 1);
    if (!unit_diagonal_flag)
    {
      __syncthreads();
      if (threadIdx.x == 0)
        v[row * v_inc + v_start] /= A[(row * A_inc1 + A_start1) * A_internal_size2 + (row * A_inc2 + A_start2)];
    }

    __syncthreads();

    temp = v[row * v_inc + v_start];

    for (int elim = (is_lower_solve ? (row + threadIdx.x + 1) : threadIdx.x);
            elim < (is_lower_solve ? A_size1 : row);
            elim += blockDim.x)
      v[elim * v_inc + v_start] -= temp * A[(elim * A_inc1 + A_start1) * A_internal_size2 + (row  * A_inc2 + A_start2)];
  }
}


template<typename NumericT>
__global__ void triangular_substitute_inplace_col_kernel(
          NumericT const * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,
          NumericT * v,
          unsigned int v_start,
          unsigned int v_inc,
          unsigned int v_size,
          unsigned int options)
{
  NumericT temp;
  unsigned int unit_diagonal_flag  = (options & (1 << 0));

  unsigned int is_lower_solve      = (options & (1 << 2));
  unsigned int row;
  for (unsigned int rows_processed = 0; rows_processed < A_size1; ++rows_processed)    //Note: A required to be square
  {
    row = is_lower_solve ? rows_processed : ((A_size1 - rows_processed) - 1);
    if (!unit_diagonal_flag)
    {
      __syncthreads();
      if (threadIdx.x == 0)
        v[row * v_inc + v_start] /= A[(row * A_inc1 + A_start1) + (row * A_inc2 + A_start2) * A_internal_size1];
    }

    __syncthreads();

    temp = v[row * v_inc + v_start];

    for (int elim = (is_lower_solve ? (row + threadIdx.x + 1) : threadIdx.x);
            elim < (is_lower_solve ? A_size1 : row);
            elim += blockDim.x)
      v[elim * v_inc + v_start] -= temp * A[(elim * A_inc1 + A_start1) + (row  * A_inc2 + A_start2) * A_internal_size1];
  }
}


namespace detail
{
  inline unsigned int get_option_for_solver_tag(viennacl::linalg::upper_tag)      { return 0; }
  inline unsigned int get_option_for_solver_tag(viennacl::linalg::unit_upper_tag) { return (1 << 0); }
  inline unsigned int get_option_for_solver_tag(viennacl::linalg::lower_tag)      { return (1 << 2); }
  inline unsigned int get_option_for_solver_tag(viennacl::linalg::unit_lower_tag) { return (1 << 2) | (1 << 0); }

  template<typename MatrixT, typename VectorT>
  void inplace_solve_vector_impl(MatrixT const & mat,
                                 VectorT & vec,
                                 unsigned int options)
  {
    typedef typename viennacl::result_of::cpu_value_type<MatrixT>::type        value_type;

    if (mat.row_major())
    {
      triangular_substitute_inplace_row_kernel<<<1, 128>>>(viennacl::cuda_arg(mat),
                                                           static_cast<unsigned int>(viennacl::traits::start1(mat)),         static_cast<unsigned int>(viennacl::traits::start2(mat)),
                                                           static_cast<unsigned int>(viennacl::traits::stride1(mat)),        static_cast<unsigned int>(viennacl::traits::stride2(mat)),
                                                           static_cast<unsigned int>(viennacl::traits::size1(mat)),          static_cast<unsigned int>(viennacl::traits::size2(mat)),
                                                           static_cast<unsigned int>(viennacl::traits::internal_size1(mat)), static_cast<unsigned int>(viennacl::traits::internal_size2(mat)),
                                                           viennacl::cuda_arg(vec),
                                                           static_cast<unsigned int>(viennacl::traits::start(vec)),
                                                           static_cast<unsigned int>(viennacl::traits::stride(vec)),
                                                           static_cast<unsigned int>(viennacl::traits::size(vec)),
                                                           options
                                                          );
    }
    else
    {
      triangular_substitute_inplace_col_kernel<<<1, 128>>>(viennacl::cuda_arg(mat),
                                                           static_cast<unsigned int>(viennacl::traits::start1(mat)),         static_cast<unsigned int>(viennacl::traits::start2(mat)),
                                                           static_cast<unsigned int>(viennacl::traits::stride1(mat)),        static_cast<unsigned int>(viennacl::traits::stride2(mat)),
                                                           static_cast<unsigned int>(viennacl::traits::size1(mat)),          static_cast<unsigned int>(viennacl::traits::size2(mat)),
                                                           static_cast<unsigned int>(viennacl::traits::internal_size1(mat)), static_cast<unsigned int>(viennacl::traits::internal_size2(mat)),
                                                           viennacl::cuda_arg(vec),
                                                           static_cast<unsigned int>(viennacl::traits::start(vec)),
                                                           static_cast<unsigned int>(viennacl::traits::stride(vec)),
                                                           static_cast<unsigned int>(viennacl::traits::size(vec)),
                                                           options
                                                          );
    }
  }

}

/** @brief Direct inplace solver for dense triangular systems (non-transposed version)
*
* @param mat       The system matrix proxy
* @param vec       The load vector, where the solution is directly written to
*/
template<typename NumericT, typename SolverTagT>
void inplace_solve(matrix_base<NumericT> const & mat,
                   vector_base<NumericT> & vec,
                   SolverTagT)
{
  unsigned int options = detail::get_option_for_solver_tag(SolverTagT());

  detail::inplace_solve_vector_impl(mat, vec, options);
}


}
}
}

#endif
