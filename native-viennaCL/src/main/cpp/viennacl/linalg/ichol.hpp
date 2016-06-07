#ifndef VIENNACL_LINALG_ICHOL_HPP_
#define VIENNACL_LINALG_ICHOL_HPP_

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

/** @file viennacl/linalg/ichol.hpp
  @brief Implementations of incomplete Cholesky factorization preconditioners with static nonzero pattern.
*/

#include <vector>
#include <cmath>
#include <iostream>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/compressed_matrix.hpp"

#include "viennacl/linalg/host_based/common.hpp"

#include <map>

namespace viennacl
{
namespace linalg
{

/** @brief A tag for incomplete Cholesky factorization with static pattern (ILU0)
*/
class ichol0_tag {};


/** @brief Implementation of a ILU-preconditioner with static pattern. Optimized version for CSR matrices.
  *
  *  Refer to Chih-Jen Lin and Jorge J. Moré, Incomplete Cholesky Factorizations with Limited Memory, SIAM J. Sci. Comput., 21(1), 24–45
  *  for one of many descriptions of incomplete Cholesky Factorizations
  *
  *  @param A       The input matrix in CSR format
  *  // param tag     An ichol0_tag in order to dispatch among several other preconditioners.
  */
template<typename NumericT>
void precondition(viennacl::compressed_matrix<NumericT> & A, ichol0_tag const & /* tag */)
{
  assert( (viennacl::traits::context(A).memory_type() == viennacl::MAIN_MEMORY) && bool("System matrix must reside in main memory for ICHOL0") );

  NumericT           * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  //std::cout << A.size1() << std::endl;
  for (vcl_size_t i=0; i<A.size1(); ++i)
  {
    unsigned int row_i_begin = row_buffer[i];
    unsigned int row_i_end   = row_buffer[i+1];

    // get a_ii:
    NumericT a_ii = 0;
    for (unsigned int buf_index_aii = row_i_begin; buf_index_aii < row_i_end; ++buf_index_aii)
    {
      if (col_buffer[buf_index_aii] == i)
      {
        a_ii = std::sqrt(elements[buf_index_aii]);
        elements[buf_index_aii] = a_ii;
        break;
      }
    }

    // Now scale column/row i, i.e. A(k, i) /= A(i, i)
    for (unsigned int buf_index_aii = row_i_begin; buf_index_aii < row_i_end; ++buf_index_aii)
    {
      if (col_buffer[buf_index_aii] > i)
        elements[buf_index_aii] /= a_ii;
    }

    // Now compute A(k, j) -= A(k, i) * A(j, i) for all nonzero k, j in column i:
    for (unsigned int buf_index_j = row_i_begin; buf_index_j < row_i_end; ++buf_index_j)
    {
      unsigned int j = col_buffer[buf_index_j];
      if (j <= i)
        continue;

      NumericT a_ji = elements[buf_index_j];

      for (unsigned int buf_index_k = row_i_begin; buf_index_k < row_i_end; ++buf_index_k)
      {
        unsigned int k = col_buffer[buf_index_k];
        if (k < j)
          continue;

        NumericT a_ki = elements[buf_index_k];

        //Now check whether A(k, j) is in nonzero pattern:
        unsigned int row_j_begin = row_buffer[j];
        unsigned int row_j_end   = row_buffer[j+1];
        for (unsigned int buf_index_kj = row_j_begin; buf_index_kj < row_j_end; ++buf_index_kj)
        {
          if (col_buffer[buf_index_kj] == k)
          {
            elements[buf_index_kj] -= a_ki * a_ji;
            break;
          }
        }
      }
    }

  }

}


/** @brief Incomplete Cholesky preconditioner class with static pattern (ICHOL0), can be supplied to solve()-routines
*/
template<typename MatrixT>
class ichol0_precond
{
  typedef typename MatrixT::value_type      NumericType;

public:
  ichol0_precond(MatrixT const & mat, ichol0_tag const & tag) : tag_(tag), LLT(mat.size1(), mat.size2(), viennacl::context(viennacl::MAIN_MEMORY))
  {
      //initialize preconditioner:
      //std::cout << "Start CPU precond" << std::endl;
      init(mat);
      //std::cout << "End CPU precond" << std::endl;
  }

  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(LLT.handle1());
    unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(LLT.handle2());
    NumericType  const * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericType>(LLT.handle());

    // Note: L is stored in a column-oriented fashion, i.e. transposed w.r.t. the row-oriented layout. Thus, the factorization A = L L^T holds L in the upper triangular part of A.
    viennacl::linalg::host_based::detail::csr_trans_inplace_solve<NumericType>(row_buffer, col_buffer, elements, vec, LLT.size2(), lower_tag());
    viennacl::linalg::host_based::detail::csr_inplace_solve<NumericType>(row_buffer, col_buffer, elements, vec, LLT.size2(), upper_tag());
  }

private:
  void init(MatrixT const & mat)
  {
    viennacl::context host_ctx(viennacl::MAIN_MEMORY);
    viennacl::switch_memory_context(LLT, host_ctx);

    viennacl::copy(mat, LLT);
    viennacl::linalg::precondition(LLT, tag_);
  }

  ichol0_tag const & tag_;
  viennacl::compressed_matrix<NumericType> LLT;
};


/** @brief ILU0 preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV>
class ichol0_precond< compressed_matrix<NumericT, AlignmentV> >
{
  typedef compressed_matrix<NumericT, AlignmentV>   MatrixType;

public:
  ichol0_precond(MatrixType const & mat, ichol0_tag const & tag) : tag_(tag), LLT(mat.size1(), mat.size2(), viennacl::traits::context(mat))
  {
    //initialize preconditioner:
    //std::cout << "Start GPU precond" << std::endl;
    init(mat);
    //std::cout << "End GPU precond" << std::endl;
  }

  void apply(vector<NumericT> & vec) const
  {
    if (viennacl::traits::context(vec).memory_type() != viennacl::MAIN_MEMORY)
    {
      viennacl::context host_ctx(viennacl::MAIN_MEMORY);
      viennacl::context old_ctx = viennacl::traits::context(vec);

      viennacl::switch_memory_context(vec, host_ctx);
      viennacl::linalg::inplace_solve(trans(LLT), vec, lower_tag());
      viennacl::linalg::inplace_solve(      LLT , vec, upper_tag());
      viennacl::switch_memory_context(vec, old_ctx);
    }
    else //apply ILU0 directly:
    {
      // Note: L is stored in a column-oriented fashion, i.e. transposed w.r.t. the row-oriented layout. Thus, the factorization A = L L^T holds L in the upper triangular part of A.
      viennacl::linalg::inplace_solve(trans(LLT), vec, lower_tag());
      viennacl::linalg::inplace_solve(      LLT , vec, upper_tag());
    }
  }

private:
  void init(MatrixType const & mat)
  {
    viennacl::context host_ctx(viennacl::MAIN_MEMORY);
    viennacl::switch_memory_context(LLT, host_ctx);
    LLT = mat;

    viennacl::linalg::precondition(LLT, tag_);
  }

  ichol0_tag const & tag_;
  viennacl::compressed_matrix<NumericT> LLT;
};

}
}




#endif



