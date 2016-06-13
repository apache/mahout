#ifndef VIENNACL_LINALG_ITERATIVE_OPERATIONS_HPP_
#define VIENNACL_LINALG_ITERATIVE_OPERATIONS_HPP_

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

/** @file viennacl/linalg/iterative_operations.hpp
    @brief Implementations of specialized routines for the iterative solvers.
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
#include "viennacl/linalg/host_based/iterative_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/iterative_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/iterative_operations.hpp"
#endif

namespace viennacl
{
namespace linalg
{

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
  switch (viennacl::traits::handle(result).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::pipelined_cg_vector_update(result, alpha, p, r, Ap, beta, inner_prod_buffer);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::pipelined_cg_vector_update(result, alpha, p, r, Ap, beta, inner_prod_buffer);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::pipelined_cg_vector_update(result, alpha, p, r, Ap, beta, inner_prod_buffer);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}


/** @brief Performs a joint vector update operation needed for an efficient pipelined CG algorithm.
  *
  * This routines computes for a matrix A and vectors 'p' and 'Ap':
  *   Ap = prod(A, p);
  * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap)
  */
template<typename MatrixT, typename NumericT>
void pipelined_cg_prod(MatrixT const & A,
                       vector_base<NumericT> const & p,
                       vector_base<NumericT> & Ap,
                       vector_base<NumericT> & inner_prod_buffer)
{
  switch (viennacl::traits::handle(p).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::pipelined_cg_prod(A, p, Ap, inner_prod_buffer);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::pipelined_cg_prod(A, p, Ap, inner_prod_buffer);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::pipelined_cg_prod(A, p, Ap, inner_prod_buffer);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

////////////////////////////////////////////

/** @brief Performs a joint vector update operation needed for an efficient pipelined CG algorithm.
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
  switch (viennacl::traits::handle(s).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::pipelined_bicgstab_update_s(s, r, Ap, inner_prod_buffer, buffer_chunk_size, buffer_chunk_offset);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::pipelined_bicgstab_update_s(s, r, Ap, inner_prod_buffer, buffer_chunk_size, buffer_chunk_offset);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::pipelined_bicgstab_update_s(s, r, Ap, inner_prod_buffer, buffer_chunk_size, buffer_chunk_offset);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
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
                                      vector_base<NumericT> & inner_prod_buffer,
                                      vcl_size_t buffer_chunk_size)
{
  switch (viennacl::traits::handle(s).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::pipelined_bicgstab_vector_update(result, alpha, p, omega, s, residual, As, beta, Ap, r0star, inner_prod_buffer, buffer_chunk_size);
    break;
  #ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::pipelined_bicgstab_vector_update(result, alpha, p, omega, s, residual, As, beta, Ap, r0star, inner_prod_buffer, buffer_chunk_size);
    break;
  #endif
  #ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::pipelined_bicgstab_vector_update(result, alpha, p, omega, s, residual, As, beta, Ap, r0star, inner_prod_buffer, buffer_chunk_size);
    break;
  #endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}


/** @brief Performs a joint vector update operation needed for an efficient pipelined CG algorithm.
  *
  * This routines computes for a matrix A and vectors 'p' and 'Ap':
  *   Ap = prod(A, p);
  * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap)
  */
template<typename MatrixT, typename NumericT>
void pipelined_bicgstab_prod(MatrixT const & A,
                             vector_base<NumericT> const & p,
                             vector_base<NumericT> & Ap,
                             vector_base<NumericT> const & r0star,
                             vector_base<NumericT> & inner_prod_buffer,
                             vcl_size_t buffer_chunk_size,
                             vcl_size_t buffer_chunk_offset)
{
  switch (viennacl::traits::handle(p).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::pipelined_bicgstab_prod(A, p, Ap, r0star, inner_prod_buffer, buffer_chunk_size, buffer_chunk_offset);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::pipelined_bicgstab_prod(A, p, Ap, r0star, inner_prod_buffer, buffer_chunk_size, buffer_chunk_offset);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::pipelined_bicgstab_prod(A, p, Ap, r0star, inner_prod_buffer, buffer_chunk_size, buffer_chunk_offset);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

////////////////////////////////////////////

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
  switch (viennacl::traits::handle(v_k).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::pipelined_gmres_normalize_vk(v_k, residual, R_buffer, offset_in_R, inner_prod_buffer, r_dot_vk_buffer, buffer_chunk_size, buffer_chunk_offset);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::pipelined_gmres_normalize_vk(v_k, residual, R_buffer, offset_in_R, inner_prod_buffer, r_dot_vk_buffer, buffer_chunk_size, buffer_chunk_offset);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::pipelined_gmres_normalize_vk(v_k, residual, R_buffer, offset_in_R, inner_prod_buffer, r_dot_vk_buffer, buffer_chunk_size, buffer_chunk_offset);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}



/** @brief Computes the first reduction stage for multiple inner products <v_i, v_k>, i=0..k-1
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
  switch (viennacl::traits::handle(device_krylov_basis).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::pipelined_gmres_gram_schmidt_stage1(device_krylov_basis, v_k_size, v_k_internal_size, k, vi_in_vk_buffer, buffer_chunk_size);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::pipelined_gmres_gram_schmidt_stage1(device_krylov_basis, v_k_size, v_k_internal_size, k, vi_in_vk_buffer, buffer_chunk_size);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::pipelined_gmres_gram_schmidt_stage1(device_krylov_basis, v_k_size, v_k_internal_size, k, vi_in_vk_buffer, buffer_chunk_size);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
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
  switch (viennacl::traits::handle(device_krylov_basis).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::pipelined_gmres_gram_schmidt_stage2(device_krylov_basis, v_k_size, v_k_internal_size, k, vi_in_vk_buffer, R_buffer, krylov_dim, inner_prod_buffer, buffer_chunk_size);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::pipelined_gmres_gram_schmidt_stage2(device_krylov_basis, v_k_size, v_k_internal_size, k, vi_in_vk_buffer, R_buffer, krylov_dim, inner_prod_buffer, buffer_chunk_size);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::pipelined_gmres_gram_schmidt_stage2(device_krylov_basis, v_k_size, v_k_internal_size, k, vi_in_vk_buffer, R_buffer, krylov_dim, inner_prod_buffer, buffer_chunk_size);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
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
  switch (viennacl::traits::handle(result).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::pipelined_gmres_update_result(result, residual, krylov_basis, v_k_size, v_k_internal_size, coefficients, k);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::pipelined_gmres_update_result(result, residual, krylov_basis, v_k_size, v_k_internal_size, coefficients, k);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::pipelined_gmres_update_result(result, residual, krylov_basis, v_k_size, v_k_internal_size, coefficients, k);
    break;
#endif
  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/** @brief Performs a joint vector update operation needed for an efficient pipelined GMRES algorithm.
  *
  * This routines computes for a matrix A and vectors 'p' and 'Ap':
  *   Ap = prod(A, p);
  * and computes the two reduction stages for computing inner_prod(p,Ap), inner_prod(Ap,Ap)
  */
template <typename MatrixType, typename T>
void pipelined_gmres_prod(MatrixType const & A,
                       vector_base<T> const & p,
                       vector_base<T> & Ap,
                       vector_base<T> & inner_prod_buffer)
{
  switch (viennacl::traits::handle(p).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::pipelined_gmres_prod(A, p, Ap, inner_prod_buffer);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::pipelined_gmres_prod(A, p, Ap, inner_prod_buffer);
    break;
#endif
#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::pipelined_gmres_prod(A, p, Ap, inner_prod_buffer);
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
