#ifndef VIENNACL_LINALG_DETAIL_CHOW_PATEL_ILU_HPP_
#define VIENNACL_LINALG_DETAIL_CHOW_PATEL_ILU_HPP_

/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
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

/** @file viennacl/linalg/detail/ilu/chow_patel_ilu.hpp
  @brief Implementations of incomplete factorization preconditioners with fine-grained parallelism.

  Based on "Fine-Grained Parallel Incomplete LU Factorization" by Chow and Patel, SIAM J. Sci. Comput., vol. 37, no. 2, pp. C169-C193
*/

#include <vector>
#include <cmath>
#include <iostream>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/detail/ilu/common.hpp"
#include "viennacl/linalg/ilu_operations.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/backend/memory.hpp"

namespace viennacl
{
namespace linalg
{

/** @brief A tag for incomplete LU and incomplete Cholesky factorization with static pattern (Parallel-ILU0, Parallel ICC0)
*/
class chow_patel_tag
{
public:
  /** @brief Constructor allowing to set the number of sweeps and Jacobi iterations.
    *
    * @param num_sweeps        Number of sweeps in setup phase
    * @param num_jacobi_iters  Number of Jacobi iterations for each triangular 'solve' when applying the preconditioner to a vector
    */
  chow_patel_tag(vcl_size_t num_sweeps = 3, vcl_size_t num_jacobi_iters = 2) : sweeps_(num_sweeps), jacobi_iters_(num_jacobi_iters) {}

  /** @brief Returns the number of sweeps (i.e. number of nonlinear iterations) in the solver setup stage */
  vcl_size_t sweeps() const { return sweeps_; }
  /** @brief Sets the number of sweeps (i.e. number of nonlinear iterations) in the solver setup stage */
  void       sweeps(vcl_size_t num) { sweeps_ = num; }

  /** @brief Returns the number of Jacobi iterations (i.e. applications of x_{k+1} = (I - D^{-1}R)x_k + D^{-1} b) for each of the solves y = U^{-1} x and z = L^{-1} y) for each preconditioner application. */
  vcl_size_t jacobi_iters() const { return jacobi_iters_; }
  /** @brief Sets the number of Jacobi iterations for each triangular 'solve' when applying the preconditioner to a vector. */
  void       jacobi_iters(vcl_size_t num) { jacobi_iters_ = num; }

private:
  vcl_size_t sweeps_;
  vcl_size_t jacobi_iters_;
};

namespace detail
{
  /** @brief Implementation of the parallel ICC0 factorization, Algorithm 3 in Chow-Patel paper.
   *
   *  Rather than dealing with a column-major upper triangular matrix U, we use the lower-triangular matrix L such that A is approximately given by LL^T.
   *  The advantage is that L is readily available in row-major format.
   */
  template<typename NumericT>
  void precondition(viennacl::compressed_matrix<NumericT> const & A,
                    viennacl::compressed_matrix<NumericT>       & L,
                    viennacl::vector<NumericT>                  & diag_L,
                    viennacl::compressed_matrix<NumericT>       & L_trans,
                    chow_patel_tag const & tag)
  {
    // make sure L and U have correct dimensions:
    L.resize(A.size1(), A.size2(), false);

    // initialize L and U from values in A:
    viennacl::linalg::extract_L(A, L);

    // diagonally scale values from A in L:
    viennacl::linalg::icc_scale(A, L);

    viennacl::vector<NumericT> aij_L(L.nnz(), viennacl::traits::context(A));
    viennacl::backend::memory_copy(L.handle(), aij_L.handle(), 0, 0, sizeof(NumericT) * L.nnz());

    // run sweeps:
    for (vcl_size_t i=0; i<tag.sweeps(); ++i)
      viennacl::linalg::icc_chow_patel_sweep(L, aij_L);

    // transpose L to obtain L_trans:
    viennacl::linalg::ilu_transpose(L, L_trans);

    // form (I - D_L^{-1}L) and (I - D_U^{-1} U), with U := L_trans
    viennacl::linalg::ilu_form_neumann_matrix(L,       diag_L);
    viennacl::linalg::ilu_form_neumann_matrix(L_trans, diag_L);
  }


  /** @brief Implementation of the parallel ILU0 factorization, Algorithm 2 in Chow-Patel paper. */
  template<typename NumericT>
  void precondition(viennacl::compressed_matrix<NumericT> const & A,
                    viennacl::compressed_matrix<NumericT>       & L,
                    viennacl::vector<NumericT>                  & diag_L,
                    viennacl::compressed_matrix<NumericT>       & U,
                    viennacl::vector<NumericT>                  & diag_U,
                    chow_patel_tag const & tag)
  {
    // make sure L and U have correct dimensions:
    L.resize(A.size1(), A.size2(), false);
    U.resize(A.size1(), A.size2(), false);

    // initialize L and U from values in A:
    viennacl::linalg::extract_LU(A, L, U);

    // diagonally scale values from A in L and U:
    viennacl::linalg::ilu_scale(A, L, U);

    // transpose storage layout of U from CSR to CSC via transposition
    viennacl::compressed_matrix<NumericT> U_trans;
    viennacl::linalg::ilu_transpose(U, U_trans);

    // keep entries of a_ij for the sweeps
    viennacl::vector<NumericT> aij_L      (L.nnz(),       viennacl::traits::context(A));
    viennacl::vector<NumericT> aij_U_trans(U_trans.nnz(), viennacl::traits::context(A));

    viennacl::backend::memory_copy(      L.handle(), aij_L.handle(),       0, 0, sizeof(NumericT) * L.nnz());
    viennacl::backend::memory_copy(U_trans.handle(), aij_U_trans.handle(), 0, 0, sizeof(NumericT) * U_trans.nnz());

    // run sweeps:
    for (vcl_size_t i=0; i<tag.sweeps(); ++i)
      viennacl::linalg::ilu_chow_patel_sweep(L, aij_L, U_trans, aij_U_trans);

    // transpose U_trans back:
    viennacl::linalg::ilu_transpose(U_trans, U);

    // form (I - D_L^{-1}L) and (I - D_U^{-1} U)
    viennacl::linalg::ilu_form_neumann_matrix(L, diag_L);
    viennacl::linalg::ilu_form_neumann_matrix(U, diag_U);
  }

}




/** @brief Parallel Chow-Patel ILU preconditioner class, can be supplied to solve()-routines
*/
template<typename MatrixT>
class chow_patel_icc_precond
{
  // only works with compressed_matrix!
  typedef typename MatrixT::CHOW_PATEL_ICC_ONLY_WORKS_WITH_COMPRESSED_MATRIX  error_type;
};


/** @brief Parallel Chow-Patel ILU preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV>
class chow_patel_icc_precond< viennacl::compressed_matrix<NumericT, AlignmentV> >
{

public:
  chow_patel_icc_precond(viennacl::compressed_matrix<NumericT, AlignmentV> const & A, chow_patel_tag const & tag)
    : tag_(tag),
      L_(0, 0, 0, viennacl::traits::context(A)),
      diag_L_(A.size1(), viennacl::traits::context(A)),
      L_trans_(0, 0, 0, viennacl::traits::context(A)),
      x_k_(A.size1(), viennacl::traits::context(A)),
      b_(A.size1(), viennacl::traits::context(A))
  {
    viennacl::linalg::detail::precondition(A, L_, diag_L_, L_trans_, tag_);
  }

  /** @brief Preconditioner application: LL^Tx = b, computed via Ly = b, L^Tx = y using Jacobi iterations.
    *
    * L contains (I - D_L^{-1}L), L_trans contains (I - D_L^{-1}L^T) where D denotes the respective diagonal matrix
    */
  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    //
    // y = L^{-1} b through Jacobi iteration y_{k+1} = (I - D^{-1}L)y_k + D^{-1}x
    //
    b_ = viennacl::linalg::element_div(vec, diag_L_);
    x_k_ = b_;
    for (unsigned int i=0; i<tag_.jacobi_iters(); ++i)
    {
      vec = viennacl::linalg::prod(L_, x_k_);
      x_k_ = vec + b_;
    }

    //
    // x = U^{-1} y through Jacobi iteration x_{k+1} = (I - D^{-1}L^T)x_k + D^{-1}b
    //
    b_ = viennacl::linalg::element_div(x_k_, diag_L_);
    x_k_ = b_; // x_1 if x_0 \equiv 0
    for (unsigned int i=0; i<tag_.jacobi_iters(); ++i)
    {
      vec = viennacl::linalg::prod(L_trans_, x_k_);
      x_k_ = vec + b_;
    }

    // return result:
    vec = x_k_;
  }

private:
  chow_patel_tag                          tag_;
  viennacl::compressed_matrix<NumericT>   L_;
  viennacl::vector<NumericT>              diag_L_;
  viennacl::compressed_matrix<NumericT>   L_trans_;

  mutable viennacl::vector<NumericT>      x_k_;
  mutable viennacl::vector<NumericT>      b_;
};






/** @brief Parallel Chow-Patel ILU preconditioner class, can be supplied to solve()-routines
*/
template<typename MatrixT>
class chow_patel_ilu_precond
{
  // only works with compressed_matrix!
  typedef typename MatrixT::CHOW_PATEL_ILU_ONLY_WORKS_WITH_COMPRESSED_MATRIX  error_type;
};


/** @brief Parallel Chow-Patel ILU preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV>
class chow_patel_ilu_precond< viennacl::compressed_matrix<NumericT, AlignmentV> >
{

public:
  chow_patel_ilu_precond(viennacl::compressed_matrix<NumericT, AlignmentV> const & A, chow_patel_tag const & tag)
    : tag_(tag),
      L_(0, 0, 0, viennacl::traits::context(A)),
      diag_L_(A.size1(), viennacl::traits::context(A)),
      U_(0, 0, 0, viennacl::traits::context(A)),
      diag_U_(A.size1(), viennacl::traits::context(A)),
      x_k_(A.size1(), viennacl::traits::context(A)),
      b_(A.size1(), viennacl::traits::context(A))
  {
    viennacl::linalg::detail::precondition(A, L_, diag_L_, U_, diag_U_, tag_);
  }

  /** @brief Preconditioner application: LUx = b, computed via Ly = b, Ux = y using Jacobi iterations.
    *
    * L_ contains (I - D_L^{-1}L), U_ contains (I - D_U^{-1}U) where D denotes the respective diagonal matrix
    */
  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    //
    // y = L^{-1} b through Jacobi iteration y_{k+1} = (I - D^{-1}L)y_k + D^{-1}x
    //
    b_ = viennacl::linalg::element_div(vec, diag_L_);
    x_k_ = b_;
    for (unsigned int i=0; i<tag_.jacobi_iters(); ++i)
    {
      vec = viennacl::linalg::prod(L_, x_k_);
      x_k_ = vec + b_;
    }

    //
    // x = U^{-1} y through Jacobi iteration x_{k+1} = (I - D^{-1}U)x_k + D^{-1}b
    //
    b_ = viennacl::linalg::element_div(x_k_, diag_U_);
    x_k_ = b_; // x_1 if x_0 \equiv 0
    for (unsigned int i=0; i<tag_.jacobi_iters(); ++i)
    {
      vec = viennacl::linalg::prod(U_, x_k_);
      x_k_ = vec + b_;
    }

    // return result:
    vec = x_k_;
  }

private:
  chow_patel_tag                          tag_;
  viennacl::compressed_matrix<NumericT>   L_;
  viennacl::vector<NumericT>              diag_L_;
  viennacl::compressed_matrix<NumericT>   U_;
  viennacl::vector<NumericT>              diag_U_;

  mutable viennacl::vector<NumericT>      x_k_;
  mutable viennacl::vector<NumericT>      b_;
};


} // namespace linalg
} // namespace viennacl


#endif



