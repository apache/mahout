#ifndef VIENNACL_LINALG_ILU_OPERATIONS_HPP_
#define VIENNACL_LINALG_ILU_OPERATIONS_HPP_

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

/** @file viennacl/linalg/ilu_operations.hpp
    @brief Implementations of specialized routines for the Chow-Patel parallel ILU preconditioner
*/

#include "viennacl/forwards.h"
#include "viennacl/range.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/host_based/ilu_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/ilu_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/ilu_operations.hpp"
#endif

namespace viennacl
{
namespace linalg
{

/** @brief Extracts the lower triangular part L from A.
  *
  * Diagonal of L is stored explicitly in order to enable better code reuse.
  *
  */
template<typename NumericT>
void extract_L(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L)
{
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::extract_L(A, L);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::extract_L(A, L);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::extract_L(A, L);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L accordingly.
  *
  * Since A should not be modified (const-correctness), updates are in L.
  *
  */
template<typename NumericT>
void icc_scale(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L)
{
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::icc_scale(A, L);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::icc_scale(A, L);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::icc_scale(A, L);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ICC (cf. Algorithm 3 in paper, but for L rather than U)
  *
  * We use a fully synchronous (Jacobi-like) variant, because asynchronous methods as described in the paper are a nightmare to debug
  * (and particularly funny if they sometimes fail, sometimes not)
  *
  * @param L       Factor L to be updated for the incomplete Cholesky factorization
  * @param aij_L   Lower triangular potion from system matrix
  */
template<typename NumericT>
void icc_chow_patel_sweep(compressed_matrix<NumericT>       & L,
                          vector<NumericT>                  & aij_L)
{
  switch (viennacl::traits::handle(L).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::icc_chow_patel_sweep(L, aij_L);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::icc_chow_patel_sweep(L, aij_L);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::icc_chow_patel_sweep(L, aij_L);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}



//////////////////////// ILU ////////////////////

/** @brief Extracts the lower triangular part L and the upper triangular part U from A.
  *
  * Diagonals of L and U are stored explicitly in order to enable better code reuse.
  *
  */
template<typename NumericT>
void extract_LU(compressed_matrix<NumericT> const & A,
                compressed_matrix<NumericT>       & L,
                compressed_matrix<NumericT>       & U)
{
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::extract_LU(A, L, U);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::extract_LU(A, L, U);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::extract_LU(A, L, U);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L and U accordingly.
  *
  * Since A should not be modified (const-correctness), updates are in L and U.
  *
  */
template<typename NumericT>
void ilu_scale(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L,
               compressed_matrix<NumericT>       & U)
{
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::ilu_scale(A, L, U);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::ilu_scale(A, L, U);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::ilu_scale(A, L, U);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/** @brief Transposition B <- A^T, where the aij-vector is permuted in the same way as the value array in A when assigned to B
  *
  * @param A     Input matrix to be transposed
  * @param B     Output matrix containing the transposed matrix
  */
template<typename NumericT>
void ilu_transpose(compressed_matrix<NumericT> const & A,
                   compressed_matrix<NumericT>       & B)
{
  viennacl::context orig_ctx = viennacl::traits::context(A);
  viennacl::context cpu_ctx(viennacl::MAIN_MEMORY);
  (void)orig_ctx;
  (void)cpu_ctx;

  viennacl::compressed_matrix<NumericT> A_host(0, 0, 0, cpu_ctx);
  (void)A_host;

  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::ilu_transpose(A, B);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    A_host = A;
    B.switch_memory_context(cpu_ctx);
    viennacl::linalg::host_based::ilu_transpose(A_host, B);
    B.switch_memory_context(orig_ctx);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    A_host = A;
    B.switch_memory_context(cpu_ctx);
    viennacl::linalg::host_based::ilu_transpose(A_host, B);
    B.switch_memory_context(orig_ctx);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}



/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ILU (cf. Algorithm 2 in paper)
  *
  * We use a fully synchronous (Jacobi-like) variant, because asynchronous methods as described in the paper are a nightmare to debug
  * (and particularly funny if they sometimes fail, sometimes not)
  *
  * @param L            Lower-triangular matrix L in LU factorization
  * @param aij_L        Lower-triangular matrix L from A
  * @param U_trans      Upper-triangular matrix U in CSC-storage, which is the same as U^trans in CSR-storage
  * @param aij_U_trans  Upper-triangular matrix from A in CSC-storage, which is the same as U^trans in CSR-storage
  */
template<typename NumericT>
void ilu_chow_patel_sweep(compressed_matrix<NumericT>       & L,
                          vector<NumericT>            const & aij_L,
                          compressed_matrix<NumericT>       & U_trans,
                          vector<NumericT>            const & aij_U_trans)
{
  switch (viennacl::traits::handle(L).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::ilu_chow_patel_sweep(L, aij_L, U_trans, aij_U_trans);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::ilu_chow_patel_sweep(L, aij_L, U_trans, aij_U_trans);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::ilu_chow_patel_sweep(L, aij_L, U_trans, aij_U_trans);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/** @brief Extracts the lower triangular part L and the upper triangular part U from A.
  *
  * Diagonals of L and U are stored explicitly in order to enable better code reuse.
  *
  */
template<typename NumericT>
void ilu_form_neumann_matrix(compressed_matrix<NumericT> & R,
                             vector<NumericT> & diag_R)
{
  switch (viennacl::traits::handle(R).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::ilu_form_neumann_matrix(R, diag_R);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::ilu_form_neumann_matrix(R, diag_R);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::ilu_form_neumann_matrix(R, diag_R);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

} //namespace linalg
} //namespace viennacl


#endif
