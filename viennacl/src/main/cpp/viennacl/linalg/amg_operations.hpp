#ifndef VIENNACL_LINALG_AMG_OPERATIONS_HPP_
#define VIENNACL_LINALG_AMG_OPERATIONS_HPP_

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

/** @file viennacl/linalg/amg_operations.hpp
    @brief Implementations of operations for algebraic multigrid
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/detail/amg/amg_base.hpp"
#include "viennacl/linalg/host_based/amg_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/amg_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/amg_operations.hpp"
#endif

namespace viennacl
{
namespace linalg
{
namespace detail
{
namespace amg
{

template<typename NumericT, typename AMGContextT>
void amg_influence(compressed_matrix<NumericT> const & A, AMGContextT & amg_context, amg_tag & tag)
{
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::amg::amg_influence(A, amg_context, tag);
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::amg::amg_influence(A, amg_context, tag);
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::amg::amg_influence(A, amg_context, tag);
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}


template<typename NumericT, typename AMGContextT>
void amg_coarse(compressed_matrix<NumericT> const & A, AMGContextT & amg_context, amg_tag & tag)
{
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::amg::amg_coarse(A, amg_context, tag);
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::amg::amg_coarse(A, amg_context, tag);
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::amg::amg_coarse(A, amg_context, tag);
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}


template<typename NumericT, typename AMGContextT>
void amg_interpol(compressed_matrix<NumericT> const & A,
                  compressed_matrix<NumericT>       & P,
                  AMGContextT & amg_context,
                  amg_tag & tag)
{
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::amg::amg_interpol(A, P, amg_context, tag);
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::amg::amg_interpol(A, P, amg_context, tag);
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::amg::amg_interpol(A, P, amg_context, tag);
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}


template<typename NumericT>
void amg_transpose(compressed_matrix<NumericT> & A,
                   compressed_matrix<NumericT> & B)
{
  viennacl::context orig_ctx = viennacl::traits::context(A);
  viennacl::context cpu_ctx(viennacl::MAIN_MEMORY);
  (void)orig_ctx;
  (void)cpu_ctx;

  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::amg::amg_transpose(A, B);
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      A.switch_memory_context(cpu_ctx);
      B.switch_memory_context(cpu_ctx);
      viennacl::linalg::host_based::amg::amg_transpose(A, B);
      A.switch_memory_context(orig_ctx);
      B.switch_memory_context(orig_ctx);
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      A.switch_memory_context(cpu_ctx);
      B.switch_memory_context(cpu_ctx);
      viennacl::linalg::host_based::amg::amg_transpose(A, B);
      A.switch_memory_context(orig_ctx);
      B.switch_memory_context(orig_ctx);
      //viennacl::linalg::cuda::amg_transpose(A, B);
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}

/** Assign sparse matrix A to dense matrix B */
template<typename SparseMatrixType, typename NumericT>
typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value>::type
assign_to_dense(SparseMatrixType const & A,
                viennacl::matrix_base<NumericT> & B)
{
  assert( (A.size1() == B.size1()) && bool("Size check failed for assignment to dense matrix: size1(A) != size1(B)"));
  assert( (A.size2() == B.size1()) && bool("Size check failed for assignment to dense matrix: size2(A) != size2(B)"));

  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::amg::assign_to_dense(A, B);
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::amg::assign_to_dense(A, B);
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::amg::assign_to_dense(A, B);
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}

template<typename NumericT>
void smooth_jacobi(unsigned int iterations,
                   compressed_matrix<NumericT> const & A,
                   vector<NumericT> & x,
                   vector<NumericT> & x_backup,
                   vector<NumericT> const & rhs_smooth,
                   NumericT weight)
{
  switch (viennacl::traits::handle(A).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::amg::smooth_jacobi(iterations, A, x, x_backup, rhs_smooth, weight);
      break;
#ifdef VIENNACL_WITH_OPENCL
    case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::amg::smooth_jacobi(iterations, A, x, x_backup, rhs_smooth, weight);
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::amg::smooth_jacobi(iterations, A, x, x_backup, rhs_smooth, weight);
      break;
#endif
    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}

} //namespace amg
} //namespace detail
} //namespace linalg
} //namespace viennacl


#endif
