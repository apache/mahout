#ifndef VIENNACL_LINALG_OPENCL_SPARSE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_SPARSE_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/sparse_matrix_operations.hpp
    @brief Implementations of operations using sparse matrices and OpenCL
*/

#include "viennacl/forwards.h"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/host_based/common.hpp"
#include "viennacl/linalg/opencl/kernels/compressed_matrix.hpp"
#include "viennacl/linalg/opencl/kernels/coordinate_matrix.hpp"
#include "viennacl/linalg/opencl/kernels/ell_matrix.hpp"
#include "viennacl/linalg/opencl/kernels/sliced_ell_matrix.hpp"
#include "viennacl/linalg/opencl/kernels/hyb_matrix.hpp"
#include "viennacl/linalg/opencl/kernels/compressed_compressed_matrix.hpp"
#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/linalg/opencl/vector_operations.hpp"

namespace viennacl
{
namespace linalg
{
namespace opencl
{

//
// Compressed matrix
//

namespace detail
{
  template<typename NumericT, unsigned int AlignmentV>
  void row_info(compressed_matrix<NumericT, AlignmentV> const & A,
                vector_base<NumericT> & x,
                viennacl::linalg::detail::row_info_types info_selector)
  {
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
    viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::init(ctx);
    viennacl::ocl::kernel & row_info_kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(), "row_info_extractor");

    viennacl::ocl::enqueue(row_info_kernel(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(),
                                           viennacl::traits::opencl_handle(x),
                                           cl_uint(A.size1()),
                                           cl_uint(info_selector)
                                          )
                          );
  }
}

/** @brief Carries out matrix-vector multiplication with a compressed_matrix
*
* Implementation of the convenience expression y = prod(A, x);
*
* @param A    The matrix
* @param x    The vector
* @param y the result vector
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::compressed_matrix<NumericT, AlignmentV> & A,
               const viennacl::vector_base<NumericT> & x,
               NumericT alpha,
                     viennacl::vector_base<NumericT> & y,
               NumericT beta)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::init(ctx);
  bool use_nvidia_specific = AlignmentV == 1 && ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id && (double(A.nnz()) / double(A.size1()) > 12.0);
  bool with_alpha_beta = (alpha < NumericT(1) || alpha > NumericT(1)) || (beta < 0 || beta > 0);


  std::stringstream ss;
  ss << "vec_mul";
  unsigned int alignment = AlignmentV; //prevent unreachable code warnings below
  if (use_nvidia_specific)
    ss << "_nvidia";
  else
  {
    if (alignment == 4)
      ss << "4";
    if (alignment == 8)
      ss << "8";
  }

  if (with_alpha_beta)
    ss << "_alpha_beta";

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(), ss.str());

  viennacl::ocl::packed_cl_uint layout_x;
  layout_x.start  = cl_uint(viennacl::traits::start(x));
  layout_x.stride = cl_uint(viennacl::traits::stride(x));
  layout_x.size   = cl_uint(viennacl::traits::size(x));
  layout_x.internal_size   = cl_uint(viennacl::traits::internal_size(x));

  viennacl::ocl::packed_cl_uint layout_y;
  layout_y.start  = cl_uint(viennacl::traits::start(y));
  layout_y.stride = cl_uint(viennacl::traits::stride(y));
  layout_y.size   = cl_uint(viennacl::traits::size(y));
  layout_y.internal_size   = cl_uint(viennacl::traits::internal_size(y));

  if (alignment == 4 || alignment == 8)
  {
    if (with_alpha_beta)
      viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(),
                               x, layout_x,
                               alpha,
                               y, layout_y,
                               beta
                              ));
    else
      viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(),
                               x, layout_x,
                               y, layout_y
                              ));
  }
  else
  {
    if (ctx.current_device().max_work_group_size() >= 256)
      k.local_work_size(0, 256);

    if (use_nvidia_specific)
    {
      k.global_work_size(0, 512 * k.local_work_size(0));

      if (with_alpha_beta)
        viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle3().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.blocks1()),
                                 x, layout_x,
                                 alpha,
                                 y, layout_y,
                                 beta
                                ));
      else
        viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle3().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.blocks1()),
                                 x, layout_x,
                                 y, layout_y
                                ));
    }
    else // use CSR adaptive:
    {
      k.global_work_size(0, A.blocks1() * k.local_work_size(0));

      if (with_alpha_beta)
        viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle3().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.blocks1()),
                                 x, layout_x,
                                 alpha,
                                 y, layout_y,
                                 beta
                                ));
      else
        viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle3().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.blocks1()),
                                 x, layout_x,
                                 y, layout_y
                                ));
    }
  }
}


/** @brief Carries out sparse_matrix-matrix multiplication first matrix being compressed
*
* Implementation of the convenience expression y = prod(sp_A, d_A);
*
* @param sp_A     The sparse matrix
* @param d_A      The dense matrix
* @param y        The y matrix
*/
template< typename NumericT, unsigned int AlignmentV>
void prod_impl(const viennacl::compressed_matrix<NumericT, AlignmentV> & sp_A,
               const viennacl::matrix_base<NumericT> & d_A,
                     viennacl::matrix_base<NumericT> & y) {

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(sp_A).context());
  viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(),
                                             detail::sparse_dense_matmult_kernel_name(false, d_A.row_major(), y.row_major()));

  viennacl::ocl::enqueue(k(sp_A.handle1().opencl_handle(), sp_A.handle2().opencl_handle(), sp_A.handle().opencl_handle(),
                           viennacl::traits::opencl_handle(d_A),
                           cl_uint(viennacl::traits::start1(d_A)),          cl_uint(viennacl::traits::start2(d_A)),
                           cl_uint(viennacl::traits::stride1(d_A)),         cl_uint(viennacl::traits::stride2(d_A)),
                           cl_uint(viennacl::traits::size1(d_A)),           cl_uint(viennacl::traits::size2(d_A)),
                           cl_uint(viennacl::traits::internal_size1(d_A)),  cl_uint(viennacl::traits::internal_size2(d_A)),
                           viennacl::traits::opencl_handle(y),
                           cl_uint(viennacl::traits::start1(y)),         cl_uint(viennacl::traits::start2(y)),
                           cl_uint(viennacl::traits::stride1(y)),        cl_uint(viennacl::traits::stride2(y)),
                           cl_uint(viennacl::traits::size1(y)),          cl_uint(viennacl::traits::size2(y)),
                           cl_uint(viennacl::traits::internal_size1(y)), cl_uint(viennacl::traits::internal_size2(y)) ));
}

/** @brief Carries out matrix-trans(matrix) multiplication first matrix being compressed
*          and the second transposed
*
* Implementation of the convenience expression y = prod(sp_A, d_A);
*
* @param sp_A             The sparse matrix
* @param d_A              The transposed dense matrix
* @param y                The y matrix
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::compressed_matrix<NumericT, AlignmentV> const & sp_A,
               viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                            const viennacl::matrix_base<NumericT>,
                                            viennacl::op_trans > const & d_A,
               viennacl::matrix_base<NumericT> & y) {

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(sp_A).context());
  viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(),
                                             detail::sparse_dense_matmult_kernel_name(true, d_A.lhs().row_major(), y.row_major()));

  viennacl::ocl::enqueue(k(sp_A.handle1().opencl_handle(), sp_A.handle2().opencl_handle(), sp_A.handle().opencl_handle(),
                           viennacl::traits::opencl_handle(d_A.lhs()),
                           cl_uint(viennacl::traits::start1(d_A.lhs())),          cl_uint(viennacl::traits::start2(d_A.lhs())),
                           cl_uint(viennacl::traits::stride1(d_A.lhs())),         cl_uint(viennacl::traits::stride2(d_A.lhs())),
                           cl_uint(viennacl::traits::size1(d_A.lhs())),           cl_uint(viennacl::traits::size2(d_A.lhs())),
                           cl_uint(viennacl::traits::internal_size1(d_A.lhs())),  cl_uint(viennacl::traits::internal_size2(d_A.lhs())),
                           viennacl::traits::opencl_handle(y),
                           cl_uint(viennacl::traits::start1(y)),         cl_uint(viennacl::traits::start2(y)),
                           cl_uint(viennacl::traits::stride1(y)),        cl_uint(viennacl::traits::stride2(y)),
                           cl_uint(viennacl::traits::size1(y)),          cl_uint(viennacl::traits::size2(y)),
                           cl_uint(viennacl::traits::internal_size1(y)), cl_uint(viennacl::traits::internal_size2(y)) ) );
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

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::init(ctx);

  /*
   * Stage 1: Analyze sparsity pattern in order to properly allocate temporary arrays
   *
   * - Upper bound for the row lengths in C
   */
  viennacl::vector<unsigned int> upper_bound_nonzeros_per_row_A(256, ctx); // upper bound for the nonzeros per row encountered for each work group

  viennacl::ocl::kernel & k1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(), "spgemm_stage1");
  viennacl::ocl::enqueue(k1(A.handle1().opencl_handle(), A.handle2().opencl_handle(), cl_uint(A.size1()),
                            viennacl::traits::opencl_handle(upper_bound_nonzeros_per_row_A)
                        )  );

  upper_bound_nonzeros_per_row_A.switch_memory_context(viennacl::context(MAIN_MEMORY));
  unsigned int * upper_bound_nonzeros_per_row_A_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(upper_bound_nonzeros_per_row_A.handle());

  unsigned int max_nnz_per_row_A = 0;
  for (std::size_t i=0; i<upper_bound_nonzeros_per_row_A.size(); ++i)
    max_nnz_per_row_A = std::max(max_nnz_per_row_A, upper_bound_nonzeros_per_row_A_ptr[i]);

  if (max_nnz_per_row_A > 32)
  {
    // determine augmented size:
    unsigned int max_entries_in_G = 32;
    if (max_nnz_per_row_A <= 256)
      max_entries_in_G = 16;
    if (max_nnz_per_row_A <= 64)
      max_entries_in_G = 8;

    viennacl::vector<unsigned int> exclusive_scan_helper(A.size1() + 1, viennacl::traits::context(A));
    viennacl::ocl::kernel & k_decompose_1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(), "spgemm_decompose_1");
    viennacl::ocl::enqueue(k_decompose_1(A.handle1().opencl_handle(), cl_uint(A.size1()),
                                         cl_uint(max_entries_in_G),
                                         viennacl::traits::opencl_handle(exclusive_scan_helper)
                          )             );

    // exclusive scan of helper array to find new size:
    viennacl::linalg::exclusive_scan(exclusive_scan_helper);
    unsigned int augmented_size = exclusive_scan_helper[A.size1()];

    // split A = A2 * G1
    viennacl::compressed_matrix<NumericT, AlignmentV> A2(A.size1(), augmented_size, augmented_size, viennacl::traits::context(A));
    viennacl::compressed_matrix<NumericT, AlignmentV> G1(augmented_size, A.size2(),        A.nnz(), viennacl::traits::context(A));

    // fill A2:
    viennacl::ocl::kernel & k_fill_A2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(), "spgemm_A2");
    viennacl::ocl::enqueue(k_fill_A2(A2.handle1().opencl_handle(), A2.handle2().opencl_handle(), A2.handle().opencl_handle(), cl_uint(A2.size1()),
                                     viennacl::traits::opencl_handle(exclusive_scan_helper)
                          )         );

    // fill G1:
    viennacl::ocl::kernel & k_fill_G1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(), "spgemm_G1");
    viennacl::ocl::enqueue(k_fill_G1(G1.handle1().opencl_handle(), G1.handle2().opencl_handle(), G1.handle().opencl_handle(), cl_uint(G1.size1()),
                                     A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.size1()), cl_uint(A.nnz()),
                                     cl_uint(max_entries_in_G),
                                     viennacl::traits::opencl_handle(exclusive_scan_helper)
                          )         );

    // compute tmp = G1 * B;
    // C = A2 * tmp;
    viennacl::compressed_matrix<NumericT, AlignmentV> tmp(G1.size1(), B.size2(), 0, viennacl::traits::context(A));
    prod_impl(G1, B, tmp); // this runs a standard RMerge without decomposition of G1
    prod_impl(A2, tmp, C); // this may split A2 again
    return;
  }


  /*
   * Stage 2: Determine sparsity pattern of C
   */
  C.resize(A.size1(), B.size2(), false);

  viennacl::ocl::kernel & k2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(), "spgemm_stage2");
  k2.local_work_size(0, 32); // run with one warp/wavefront
  k2.global_work_size(0, 256*256*32); // make sure enough warps/wavefronts are in flight
  viennacl::ocl::enqueue(k2(A.handle1().opencl_handle(), A.handle2().opencl_handle(), cl_uint(A.size1()),
                            B.handle1().opencl_handle(), B.handle2().opencl_handle(), cl_uint(B.size2()),
                            C.handle1().opencl_handle()
                        )  );

  // exclusive scan on host to obtain row start indices:
  viennacl::backend::typesafe_host_array<unsigned int> row_buffer(C.handle1(), C.size1() + 1);
  viennacl::backend::memory_read(C.handle1(), 0, row_buffer.raw_size(), row_buffer.get());
  unsigned int current_offset = 0;
  for (std::size_t i=0; i<C.size1(); ++i)
  {
    unsigned int tmp = row_buffer[i];
    row_buffer.set(i, current_offset);
    current_offset += tmp;
  }
  row_buffer.set(C.size1(), current_offset);
  viennacl::backend::memory_write(C.handle1(), 0, row_buffer.raw_size(), row_buffer.get());


  /*
   * Stage 3: Compute entries in C
   */

  C.reserve(current_offset, false);

  viennacl::ocl::kernel & k3 = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(), "spgemm_stage3");
  k3.local_work_size(0, 32); // run with one warp/wavefront
  k3.global_work_size(0, 256*256*32); // make sure enough warps/wavefronts are in flight
  viennacl::ocl::enqueue(k3(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.size1()),
                            B.handle1().opencl_handle(), B.handle2().opencl_handle(), B.handle().opencl_handle(), cl_uint(B.size2()),
                            C.handle1().opencl_handle(), C.handle2().opencl_handle(), C.handle().opencl_handle()
                        )  );

}

// triangular solvers

/** @brief Inplace solution of a lower triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
*
* @param L    The matrix
* @param x  The vector holding the right hand side. Is overwritten by the solution.
*/
template<typename NumericT, unsigned int MAT_AlignmentV>
void inplace_solve(compressed_matrix<NumericT, MAT_AlignmentV> const & L,
                   vector_base<NumericT> & x,
                   viennacl::linalg::unit_lower_tag)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(L).context());
  viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::program_name(), "unit_lu_forward");

  k.local_work_size(0, 128);
  k.global_work_size(0, k.local_work_size());
  viennacl::ocl::enqueue(k(L.handle1().opencl_handle(), L.handle2().opencl_handle(), L.handle().opencl_handle(),
                           viennacl::traits::opencl_handle(x),
                           cl_uint(L.size1())
                          )
                        );
}

/** @brief Inplace solution of a lower triangular compressed_matrix. Typically used for LU substitutions
*
* @param L    The matrix
* @param x  The vector holding the right hand side. Is overwritten by the solution.
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(compressed_matrix<NumericT, AlignmentV> const & L,
                   vector_base<NumericT> & x,
                   viennacl::linalg::lower_tag)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(L).context());
  viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::program_name(), "lu_forward");

  k.local_work_size(0, 128);
  k.global_work_size(0, k.local_work_size());
  viennacl::ocl::enqueue(k(L.handle1().opencl_handle(), L.handle2().opencl_handle(), L.handle().opencl_handle(),
                           viennacl::traits::opencl_handle(x),
                           cl_uint(L.size1())
                          )
                        );
}


/** @brief Inplace solution of an upper triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
*
* @param U    The matrix
* @param x  The vector holding the right hand side. Is overwritten by the solution.
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(compressed_matrix<NumericT, AlignmentV> const & U,
                   vector_base<NumericT> & x,
                   viennacl::linalg::unit_upper_tag)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(U).context());
  viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::program_name(), "unit_lu_backward");

  k.local_work_size(0, 128);
  k.global_work_size(0, k.local_work_size());
  viennacl::ocl::enqueue(k(U.handle1().opencl_handle(), U.handle2().opencl_handle(), U.handle().opencl_handle(),
                           viennacl::traits::opencl_handle(x),
                           cl_uint(U.size1())
                          )
                        );
}

/** @brief Inplace solution of an upper triangular compressed_matrix. Typically used for LU substitutions
*
* @param U    The matrix
* @param x  The vector holding the right hand side. Is overwritten by the solution.
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(compressed_matrix<NumericT, AlignmentV> const & U,
                   vector_base<NumericT> & x,
                   viennacl::linalg::upper_tag)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(U).context());
  viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::program_name(), "lu_backward");

  k.local_work_size(0, 128);
  k.global_work_size(0, k.local_work_size());
  viennacl::ocl::enqueue(k(U.handle1().opencl_handle(), U.handle2().opencl_handle(), U.handle().opencl_handle(),
                           viennacl::traits::opencl_handle(x),
                           cl_uint(U.size1())
                          )
                        );
}





// transposed triangular solvers

namespace detail
{
  //
  // block solves
  //
  template<typename NumericT, unsigned int AlignmentV>
  void block_inplace_solve(const matrix_expression<const compressed_matrix<NumericT, AlignmentV>,
                                                   const compressed_matrix<NumericT, AlignmentV>,
                                                   op_trans> & L,
                           viennacl::backend::mem_handle const & block_indices, vcl_size_t num_blocks,
                           vector_base<NumericT> const & /* L_diagonal */,  //ignored
                           vector_base<NumericT> & x,
                           viennacl::linalg::unit_lower_tag)
  {
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(L.lhs()).context());
    viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::init(ctx);
    viennacl::ocl::kernel & block_solve_kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::program_name(), "block_trans_unit_lu_forward");
    block_solve_kernel.global_work_size(0, num_blocks * block_solve_kernel.local_work_size(0));

    viennacl::ocl::enqueue(block_solve_kernel(L.lhs().handle1().opencl_handle(),
                                              L.lhs().handle2().opencl_handle(),
                                              L.lhs().handle().opencl_handle(),
                                              block_indices.opencl_handle(),
                                              x,
                                              static_cast<cl_uint>(x.size())));
  }


  template<typename NumericT, unsigned int AlignmentV>
  void block_inplace_solve(matrix_expression<const compressed_matrix<NumericT, AlignmentV>,
                                             const compressed_matrix<NumericT, AlignmentV>,
                                             op_trans> const & U,
                           viennacl::backend::mem_handle const & block_indices, vcl_size_t num_blocks,
                           vector_base<NumericT> const & U_diagonal,
                           vector_base<NumericT>       & x,
                           viennacl::linalg::upper_tag)
  {
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(U.lhs()).context());
    viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::init(ctx);
    viennacl::ocl::kernel & block_solve_kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::program_name(), "block_trans_lu_backward");
    block_solve_kernel.global_work_size(0, num_blocks * block_solve_kernel.local_work_size(0));

    viennacl::ocl::enqueue(block_solve_kernel(U.lhs().handle1().opencl_handle(),
                                              U.lhs().handle2().opencl_handle(),
                                              U.lhs().handle().opencl_handle(),
                                              U_diagonal,
                                              block_indices.opencl_handle(),
                                              x,
                                              static_cast<cl_uint>(x.size())));
  }


}


/** @brief Inplace solution of a lower triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
*
* @param proxy_L  The transposed matrix proxy
* @param x      The vector
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(matrix_expression< const compressed_matrix<NumericT, AlignmentV>,
                                      const compressed_matrix<NumericT, AlignmentV>,
                                      op_trans> const & proxy_L,
                   vector_base<NumericT> & x,
                   viennacl::linalg::unit_lower_tag)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(proxy_L.lhs()).context());
  viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::program_name(), "trans_unit_lu_forward");

  k.local_work_size(0, 128);
  k.global_work_size(0, k.local_work_size());
  viennacl::ocl::enqueue(k(proxy_L.lhs().handle1().opencl_handle(), proxy_L.lhs().handle2().opencl_handle(), proxy_L.lhs().handle().opencl_handle(),
                           viennacl::traits::opencl_handle(x),
                           cl_uint(proxy_L.lhs().size1())
                          )
                        );
}


/** @brief Inplace solution of a lower triangular compressed_matrix. Typically used for LU substitutions
*
* @param proxy_L  The transposed matrix proxy
* @param x      The vector
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(matrix_expression< const compressed_matrix<NumericT, AlignmentV>,
                                      const compressed_matrix<NumericT, AlignmentV>,
                                      op_trans> const & proxy_L,
                   vector_base<NumericT> & x,
                   viennacl::linalg::lower_tag)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(proxy_L.lhs()).context());
  viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::init(ctx);

  viennacl::vector<NumericT> diagonal(x.size());
  detail::row_info(proxy_L.lhs(), diagonal, viennacl::linalg::detail::SPARSE_ROW_DIAGONAL);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::program_name(), "trans_lu_forward");

  k.local_work_size(0, 128);
  k.global_work_size(0, k.local_work_size());
  viennacl::ocl::enqueue(k(proxy_L.lhs().handle1().opencl_handle(), proxy_L.lhs().handle2().opencl_handle(), proxy_L.lhs().handle().opencl_handle(),
                           viennacl::traits::opencl_handle(diagonal),
                           viennacl::traits::opencl_handle(x),
                           cl_uint(proxy_L.lhs().size1())
                          )
                        );
}

/** @brief Inplace solution of a lower triangular compressed_matrix with unit diagonal. Typically used for LU substitutions
*
* @param proxy_U  The transposed matrix proxy
* @param x      The vector
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(matrix_expression< const compressed_matrix<NumericT, AlignmentV>,
                                      const compressed_matrix<NumericT, AlignmentV>,
                                      op_trans> const & proxy_U,
                   vector_base<NumericT> & x,
                   viennacl::linalg::unit_upper_tag)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(proxy_U.lhs()).context());
  viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::program_name(), "trans_unit_lu_backward");

  k.local_work_size(0, 128);
  k.global_work_size(0, k.local_work_size());
  viennacl::ocl::enqueue(k(proxy_U.lhs().handle1().opencl_handle(), proxy_U.lhs().handle2().opencl_handle(), proxy_U.lhs().handle().opencl_handle(),
                           viennacl::traits::opencl_handle(x),
                           cl_uint(proxy_U.lhs().size1())
                          )
                        );
}


/** @brief Inplace solution of a lower triangular compressed_matrix. Typically used for LU substitutions
*
* @param proxy_U  The transposed matrix proxy
* @param x      The vector
*/
template<typename NumericT, unsigned int AlignmentV>
void inplace_solve(matrix_expression< const compressed_matrix<NumericT, AlignmentV>,
                                      const compressed_matrix<NumericT, AlignmentV>,
                                      op_trans> const & proxy_U,
                   vector_base<NumericT> & x,
                   viennacl::linalg::upper_tag)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(proxy_U.lhs()).context());
  viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::init(ctx);

  viennacl::vector<NumericT> diagonal(x.size());
  detail::row_info(proxy_U.lhs(), diagonal, viennacl::linalg::detail::SPARSE_ROW_DIAGONAL);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix_solve<NumericT>::program_name(), "trans_lu_backward");

  k.local_work_size(0, 128);
  k.global_work_size(0, k.local_work_size());
  viennacl::ocl::enqueue(k(proxy_U.lhs().handle1().opencl_handle(), proxy_U.lhs().handle2().opencl_handle(), proxy_U.lhs().handle().opencl_handle(),
                           viennacl::traits::opencl_handle(diagonal),
                           viennacl::traits::opencl_handle(x),
                           cl_uint(proxy_U.lhs().size1())
                          )
                        );
}


//
// Compressed Compressed matrix
//

/** @brief Carries out matrix-vector multiplication with a compressed_compressed_matrix
*
* Implementation of the convenience expression y = prod(A, x);
*
* @param A    The matrix
* @param x    The vector
* @param y the result vector
*/
template<typename NumericT>
void prod_impl(viennacl::compressed_compressed_matrix<NumericT> const & A,
               viennacl::vector_base<NumericT> const & x,
               NumericT alpha,
               viennacl::vector_base<NumericT>       & y,
               NumericT beta)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::compressed_compressed_matrix<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_compressed_matrix<NumericT>::program_name(), "vec_mul");

  if (beta < 0 || beta > 0) // multiply by beta
    viennacl::linalg::opencl::av(y, y, beta, 1, false, false);
  else
    y.clear();

  viennacl::ocl::packed_cl_uint layout_x;
  layout_x.start  = cl_uint(viennacl::traits::start(x));
  layout_x.stride = cl_uint(viennacl::traits::stride(x));
  layout_x.size   = cl_uint(viennacl::traits::size(x));
  layout_x.internal_size   = cl_uint(viennacl::traits::internal_size(x));

  viennacl::ocl::packed_cl_uint layout_y;
  layout_y.start  = cl_uint(viennacl::traits::start(y));
  layout_y.stride = cl_uint(viennacl::traits::stride(y));
  layout_y.size   = cl_uint(viennacl::traits::size(y));
  layout_y.internal_size   = cl_uint(viennacl::traits::internal_size(y));

  viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle3().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.nnz1()),
                           x, layout_x,
                           alpha,
                           y, layout_y,
                           beta
                          ));
}


//
// Coordinate matrix
//

namespace detail
{
  template<typename NumericT, unsigned int AlignmentV>
  void row_info(coordinate_matrix<NumericT, AlignmentV> const & A,
                vector_base<NumericT> & x,
                viennacl::linalg::detail::row_info_types info_selector)
  {
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
    viennacl::linalg::opencl::kernels::coordinate_matrix<NumericT>::init(ctx);
    viennacl::ocl::kernel & row_info_kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::coordinate_matrix<NumericT>::program_name(), "row_info_extractor");
    unsigned int thread_num = 128; //k.local_work_size(0);

    row_info_kernel.local_work_size(0, thread_num);

    row_info_kernel.global_work_size(0, 64 * thread_num);  //64 work groups are hard-coded for now. Gives reasonable performance in most cases
    viennacl::ocl::enqueue(row_info_kernel(A.handle12().opencl_handle(), A.handle().opencl_handle(), A.handle3().opencl_handle(),
                                           viennacl::traits::opencl_handle(x),
                                           cl_uint(info_selector),
                                           viennacl::ocl::local_mem(sizeof(cl_uint)*thread_num),
                                           viennacl::ocl::local_mem(sizeof(NumericT)*thread_num)) );
  }
}

/** @brief Carries out matrix-vector multiplication with a coordinate_matrix
*
* Implementation of the convenience expression y = prod(A, x);
*
* @param A    The matrix
* @param x    The vector
* @param y the result vector
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::coordinate_matrix<NumericT, AlignmentV> const & A,
               viennacl::vector_base<NumericT> const & x,
               NumericT alpha,
               viennacl::vector_base<NumericT>       & y,
               NumericT beta)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::coordinate_matrix<NumericT>::init(ctx);

  if (beta < 0 || beta > 0) // multiply by beta
    viennacl::linalg::opencl::av(y, y, beta, 1, false, false);
  else
    y.clear();

  viennacl::ocl::packed_cl_uint layout_x;
  layout_x.start  = cl_uint(viennacl::traits::start(x));
  layout_x.stride = cl_uint(viennacl::traits::stride(x));
  layout_x.size   = cl_uint(viennacl::traits::size(x));
  layout_x.internal_size   = cl_uint(viennacl::traits::internal_size(x));

  viennacl::ocl::packed_cl_uint layout_y;
  layout_y.start  = cl_uint(viennacl::traits::start(y));
  layout_y.stride = cl_uint(viennacl::traits::stride(y));
  layout_y.size   = cl_uint(viennacl::traits::size(y));
  layout_y.internal_size   = cl_uint(viennacl::traits::internal_size(y));

  //std::cout << "prod(coordinate_matrix" << AlignmentV << ", vector) called with internal_nnz=" << A.internal_nnz() << std::endl;

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::coordinate_matrix<NumericT>::program_name(), "vec_mul");
  unsigned int thread_num = 128; //k.local_work_size(0);

  k.local_work_size(0, thread_num);

  k.global_work_size(0, 64 * thread_num);  //64 work groups are hard-coded for now. Gives reasonable performance in most cases
  //k.global_work_size(0, thread_num);  //Only one work group
  viennacl::ocl::enqueue(k(A.handle12().opencl_handle(), A.handle().opencl_handle(), A.handle3().opencl_handle(),
                           viennacl::traits::opencl_handle(x),
                           layout_x,
                           alpha,
                           viennacl::traits::opencl_handle(y),
                           layout_y,
                           beta,
                           viennacl::ocl::local_mem(sizeof(cl_uint)*thread_num),
                           viennacl::ocl::local_mem(sizeof(NumericT)*thread_num)) );

}


/** @brief Carries out sparse-matrix-dense-matrix multiplication, where the sparse matrix is a coordinate_matrix
*
* Implementation of the convenience expression y = prod(A, B); with A being sparse (COO) and B being dense
*
* @param A    The sparse matrix (COO forA)
* @param d_A  The dense matrix
* @param y the result vector
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::coordinate_matrix<NumericT, AlignmentV> const & A,
               viennacl::matrix_base<NumericT> const & d_A,
               viennacl::matrix_base<NumericT>       & y)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::coordinate_matrix<NumericT>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::coordinate_matrix<NumericT>::program_name(),
                                             detail::sparse_dense_matmult_kernel_name(false, d_A.row_major(), y.row_major()));

  y.clear();

  unsigned int thread_num = 128; //k.local_work_size(0);
  k.local_work_size(0, thread_num);
  k.global_work_size(0, 64 * thread_num);  //64 work groups are hard-coded for now. Gives reasonable performance in most cases

  viennacl::ocl::enqueue(k(A.handle12().opencl_handle(), A.handle().opencl_handle(), A.handle3().opencl_handle(),
                           viennacl::traits::opencl_handle(d_A),
                           cl_uint(viennacl::traits::start1(d_A)),          cl_uint(viennacl::traits::start2(d_A)),
                           cl_uint(viennacl::traits::stride1(d_A)),         cl_uint(viennacl::traits::stride2(d_A)),
                           cl_uint(viennacl::traits::size1(d_A)),           cl_uint(viennacl::traits::size2(d_A)),
                           cl_uint(viennacl::traits::internal_size1(d_A)),  cl_uint(viennacl::traits::internal_size2(d_A)),
                           viennacl::traits::opencl_handle(y),
                           cl_uint(viennacl::traits::start1(y)),         cl_uint(viennacl::traits::start2(y)),
                           cl_uint(viennacl::traits::stride1(y)),        cl_uint(viennacl::traits::stride2(y)),
                           cl_uint(viennacl::traits::size1(y)),          cl_uint(viennacl::traits::size2(y)),
                           cl_uint(viennacl::traits::internal_size1(y)), cl_uint(viennacl::traits::internal_size2(y)),
                           viennacl::ocl::local_mem(sizeof(cl_uint)*k.local_work_size(0)),
                           viennacl::ocl::local_mem(sizeof(NumericT)*k.local_work_size(0))) );

}

/** @brief Carries out sparse-matrix-dense-matrix multiplication, where the sparse matrix is a coordinate_matrix
*
* Implementation of the convenience expression y = prod(A, trans(B)); with A being sparse (COO) and B being dense
*
* @param A    The sparse matrix (COO forA)
* @param d_A  The dense matrix
* @param y the result vector
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::coordinate_matrix<NumericT, AlignmentV> const & A,
               viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                            const viennacl::matrix_base<NumericT>,
                                            viennacl::op_trans > const & d_A,
               viennacl::matrix_base<NumericT> & y)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::coordinate_matrix<NumericT>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::coordinate_matrix<NumericT>::program_name(),
                                             detail::sparse_dense_matmult_kernel_name(true, d_A.lhs().row_major(), y.row_major()));

  y.clear();

  unsigned int thread_num = 128; //k.local_work_size(0);
  k.local_work_size(0, thread_num);
  k.global_work_size(0, 64 * thread_num);  //64 work groups are hard-coded for now. Gives reasonable performance in most cases

  viennacl::ocl::enqueue(k(A.handle12().opencl_handle(), A.handle().opencl_handle(), A.handle3().opencl_handle(),
                           viennacl::traits::opencl_handle(d_A),
                           cl_uint(viennacl::traits::start1(d_A.lhs())),          cl_uint(viennacl::traits::start2(d_A.lhs())),
                           cl_uint(viennacl::traits::stride1(d_A.lhs())),         cl_uint(viennacl::traits::stride2(d_A.lhs())),
                           cl_uint(viennacl::traits::size1(d_A.lhs())),           cl_uint(viennacl::traits::size2(d_A.lhs())),
                           cl_uint(viennacl::traits::internal_size1(d_A.lhs())),  cl_uint(viennacl::traits::internal_size2(d_A.lhs())),
                           viennacl::traits::opencl_handle(y),
                           cl_uint(viennacl::traits::start1(y)),         cl_uint(viennacl::traits::start2(y)),
                           cl_uint(viennacl::traits::stride1(y)),        cl_uint(viennacl::traits::stride2(y)),
                           cl_uint(viennacl::traits::size1(y)),          cl_uint(viennacl::traits::size2(y)),
                           cl_uint(viennacl::traits::internal_size1(y)), cl_uint(viennacl::traits::internal_size2(y)),
                           viennacl::ocl::local_mem(sizeof(cl_uint)*k.local_work_size(0)),
                           viennacl::ocl::local_mem(sizeof(NumericT)*k.local_work_size(0))) );

}


//
// ELL Matrix
//

template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::ell_matrix<NumericT, AlignmentV> const & A,
               viennacl::vector_base<NumericT> const & x,
               NumericT alpha,
               viennacl::vector_base<NumericT>       & y,
               NumericT beta)
{
  assert(A.size1() == y.size());
  assert(A.size2() == x.size());

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::ell_matrix<NumericT>::init(ctx);

  bool with_alpha_beta = (alpha < NumericT(1) || alpha > NumericT(1)) || (beta < 0 || beta > 0);

  viennacl::ocl::packed_cl_uint layout_x;
  layout_x.start  = cl_uint(viennacl::traits::start(x));
  layout_x.stride = cl_uint(viennacl::traits::stride(x));
  layout_x.size   = cl_uint(viennacl::traits::size(x));
  layout_x.internal_size   = cl_uint(viennacl::traits::internal_size(x));

  viennacl::ocl::packed_cl_uint layout_y;
  layout_y.start  = cl_uint(viennacl::traits::start(y));
  layout_y.stride = cl_uint(viennacl::traits::stride(y));
  layout_y.size   = cl_uint(viennacl::traits::size(y));
  layout_y.internal_size   = cl_uint(viennacl::traits::internal_size(y));

  std::stringstream ss;
  ss << "vec_mul_" << 1;//(AlignmentV != 1?4:1);
  viennacl::ocl::kernel& k = ctx.get_kernel(viennacl::linalg::opencl::kernels::ell_matrix<NumericT>::program_name(), with_alpha_beta ? "vec_mul_alpha_beta" : "vec_mul");

  unsigned int thread_num = 128;
  unsigned int group_num = 256;

  k.local_work_size(0, thread_num);
  k.global_work_size(0, thread_num * group_num);

  if (with_alpha_beta)
    viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                             A.handle().opencl_handle(),
                             viennacl::traits::opencl_handle(x),
                             layout_x,
                             alpha,
                             viennacl::traits::opencl_handle(y),
                             layout_y,
                             beta,
                             cl_uint(A.size1()),
                             cl_uint(A.size2()),
                             cl_uint(A.internal_size1()),
                             cl_uint(A.maxnnz()),
                             cl_uint(A.internal_maxnnz())
                            )
    );
  else
    viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                             A.handle().opencl_handle(),
                             viennacl::traits::opencl_handle(x),
                             layout_x,
                             viennacl::traits::opencl_handle(y),
                             layout_y,
                             cl_uint(A.size1()),
                             cl_uint(A.size2()),
                             cl_uint(A.internal_size1()),
                             cl_uint(A.maxnnz()),
                             cl_uint(A.internal_maxnnz())
                            )
    );


}

/** @brief Carries out Sparse Matrix(ELL)-Dense Matrix multiplication
*
* Implementation of the convenience expression y = prod(sp_A, d_A);
* sp_mat being in ELL format
*
* @param sp_A     The sparse matrix (ELL)
* @param d_A      The dense matrix
* @param y        The y matrix
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::ell_matrix<NumericT, AlignmentV> const & sp_A,
               viennacl::matrix_base<NumericT> const & d_A,
               viennacl::matrix_base<NumericT>       & y) {

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(sp_A).context());
  viennacl::linalg::opencl::kernels::ell_matrix<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::ell_matrix<NumericT>::program_name(),
                                             detail::sparse_dense_matmult_kernel_name(false, d_A.row_major(), y.row_major()));

  //unsigned int thread_num = 128;
  //unsigned int group_num = 256;
  //
  //k.local_work_size(0, thread_num);
  //k.global_work_size(0, thread_num * group_num);

  viennacl::ocl::enqueue(k(sp_A.handle2().opencl_handle(), sp_A.handle().opencl_handle(),
                           cl_uint(sp_A.size1()),
                           cl_uint(sp_A.size2()),
                           cl_uint(sp_A.internal_size1()),
                           cl_uint(sp_A.maxnnz()),
                           cl_uint(sp_A.internal_maxnnz()),
                           viennacl::traits::opencl_handle(d_A),
                           cl_uint(viennacl::traits::start1(d_A)),          cl_uint(viennacl::traits::start2(d_A)),
                           cl_uint(viennacl::traits::stride1(d_A)),         cl_uint(viennacl::traits::stride2(d_A)),
                           cl_uint(viennacl::traits::size1(d_A)),           cl_uint(viennacl::traits::size2(d_A)),
                           cl_uint(viennacl::traits::internal_size1(d_A)),  cl_uint(viennacl::traits::internal_size2(d_A)),
                           viennacl::traits::opencl_handle(y),
                           cl_uint(viennacl::traits::start1(y)),         cl_uint(viennacl::traits::start2(y)),
                           cl_uint(viennacl::traits::stride1(y)),        cl_uint(viennacl::traits::stride2(y)),
                           cl_uint(viennacl::traits::size1(y)),          cl_uint(viennacl::traits::size2(y)),
                           cl_uint(viennacl::traits::internal_size1(y)), cl_uint(viennacl::traits::internal_size2(y))
                          )
                        );
}

/** @brief Carries out Sparse Matrix(ELL)-Dense Transposed Matrix multiplication
*
* Implementation of the convenience expression y = prod(sp_A, trans(d_A));
* sp_mat being in ELL format
*
* @param sp_A     The sparse matrix (ELL)
* @param d_A      The dense transposed matrix
* @param y        The y matrix
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::ell_matrix<NumericT, AlignmentV> const & sp_A,
               viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                            const viennacl::matrix_base<NumericT>,
                                            viennacl::op_trans > const & d_A,
               viennacl::matrix_base<NumericT> & y) {

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(sp_A).context());
  viennacl::linalg::opencl::kernels::ell_matrix<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::ell_matrix<NumericT>::program_name(),
                                             detail::sparse_dense_matmult_kernel_name(true, d_A.lhs().row_major(), y.row_major()));

  //unsigned int thread_num = 128;
  //unsigned int group_num = 256;
  //
  //k.local_work_size(0, thread_num);
  //k.global_work_size(0, thread_num * group_num);

  viennacl::ocl::enqueue(k(sp_A.handle2().opencl_handle(), sp_A.handle().opencl_handle(),
                           cl_uint(sp_A.size1()),
                           cl_uint(sp_A.size2()),
                           cl_uint(sp_A.internal_size1()),
                           cl_uint(sp_A.maxnnz()),
                           cl_uint(sp_A.internal_maxnnz()),
                           viennacl::traits::opencl_handle(d_A.lhs()),
                           cl_uint(viennacl::traits::start1(d_A.lhs())),          cl_uint(viennacl::traits::start2(d_A.lhs())),
                           cl_uint(viennacl::traits::stride1(d_A.lhs())),         cl_uint(viennacl::traits::stride2(d_A.lhs())),
                           cl_uint(viennacl::traits::size1(d_A.lhs())),           cl_uint(viennacl::traits::size2(d_A.lhs())),
                           cl_uint(viennacl::traits::internal_size1(d_A.lhs())),  cl_uint(viennacl::traits::internal_size2(d_A.lhs())),
                           viennacl::traits::opencl_handle(y),
                           cl_uint(viennacl::traits::start1(y)),         cl_uint(viennacl::traits::start2(y)),
                           cl_uint(viennacl::traits::stride1(y)),        cl_uint(viennacl::traits::stride2(y)),
                           cl_uint(viennacl::traits::size1(y)),          cl_uint(viennacl::traits::size2(y)),
                           cl_uint(viennacl::traits::internal_size1(y)), cl_uint(viennacl::traits::internal_size2(y))
                          )
                        );
}

//
// SELL-C-\sigma Matrix
//

template<typename ScalarT, typename IndexT>
void prod_impl(viennacl::sliced_ell_matrix<ScalarT, IndexT> const & A,
               viennacl::vector_base<ScalarT> const & x,
               ScalarT alpha,
               viennacl::vector_base<ScalarT>       & y,
               ScalarT beta)
{
  assert(A.size1() == y.size());
  assert(A.size2() == x.size());

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::sliced_ell_matrix<ScalarT, unsigned int>::init(ctx);

  bool with_alpha_beta = (alpha < ScalarT(1) || alpha > ScalarT(1)) || (beta < 0 || beta > 0);

  viennacl::ocl::packed_cl_uint layout_x;
  layout_x.start  = cl_uint(viennacl::traits::start(x));
  layout_x.stride = cl_uint(viennacl::traits::stride(x));
  layout_x.size   = cl_uint(viennacl::traits::size(x));
  layout_x.internal_size   = cl_uint(viennacl::traits::internal_size(x));

  viennacl::ocl::packed_cl_uint layout_y;
  layout_y.start  = cl_uint(viennacl::traits::start(y));
  layout_y.stride = cl_uint(viennacl::traits::stride(y));
  layout_y.size   = cl_uint(viennacl::traits::size(y));
  layout_y.internal_size   = cl_uint(viennacl::traits::internal_size(y));

  std::stringstream ss;
  ss << "vec_mul_" << 1;//(AlignmentV != 1?4:1);
  viennacl::ocl::kernel& k = ctx.get_kernel(viennacl::linalg::opencl::kernels::sliced_ell_matrix<ScalarT, IndexT>::program_name(), with_alpha_beta ? "vec_mul_alpha_beta" : "vec_mul");

  vcl_size_t thread_num = std::max(A.rows_per_block(), static_cast<vcl_size_t>(128));
  unsigned int group_num = 256;

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
    thread_num = 256;

  k.local_work_size(0, thread_num);
  k.global_work_size(0, thread_num * group_num);

  if (with_alpha_beta)
    viennacl::ocl::enqueue(k(A.handle1().opencl_handle(),
                             A.handle2().opencl_handle(),
                             A.handle3().opencl_handle(),
                             A.handle().opencl_handle(),
                             viennacl::traits::opencl_handle(x),
                             layout_x,
                             alpha,
                             viennacl::traits::opencl_handle(y),
                             layout_y,
                             beta,
                             cl_uint(A.rows_per_block()))
    );
  else
    viennacl::ocl::enqueue(k(A.handle1().opencl_handle(),
                             A.handle2().opencl_handle(),
                             A.handle3().opencl_handle(),
                             A.handle().opencl_handle(),
                             viennacl::traits::opencl_handle(x),
                             layout_x,
                             viennacl::traits::opencl_handle(y),
                             layout_y,
                             cl_uint(A.rows_per_block()))
    );
}


//
// Hybrid Matrix
//

template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::hyb_matrix<NumericT, AlignmentV> const & A,
               viennacl::vector_base<NumericT> const & x,
               NumericT alpha,
               viennacl::vector_base<NumericT>       & y,
               NumericT beta)
{
  assert(A.size1() == y.size());
  assert(A.size2() == x.size());

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::hyb_matrix<NumericT>::init(ctx);

  bool with_alpha_beta = (alpha < NumericT(1) || alpha > NumericT(1)) || (beta < 0 || beta > 0);

  viennacl::ocl::packed_cl_uint layout_x;
  layout_x.start  = cl_uint(viennacl::traits::start(x));
  layout_x.stride = cl_uint(viennacl::traits::stride(x));
  layout_x.size   = cl_uint(viennacl::traits::size(x));
  layout_x.internal_size   = cl_uint(viennacl::traits::internal_size(x));

  viennacl::ocl::packed_cl_uint layout_y;
  layout_y.start  = cl_uint(viennacl::traits::start(y));
  layout_y.stride = cl_uint(viennacl::traits::stride(y));
  layout_y.size   = cl_uint(viennacl::traits::size(y));
  layout_y.internal_size   = cl_uint(viennacl::traits::internal_size(y));

  viennacl::ocl::kernel& k = ctx.get_kernel(viennacl::linalg::opencl::kernels::hyb_matrix<NumericT>::program_name(), with_alpha_beta ? "vec_mul_alpha_beta" : "vec_mul");

  if (with_alpha_beta)
    viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                             A.handle().opencl_handle(),
                             A.handle3().opencl_handle(),
                             A.handle4().opencl_handle(),
                             A.handle5().opencl_handle(),
                             viennacl::traits::opencl_handle(x),
                             layout_x,
                             alpha,
                             viennacl::traits::opencl_handle(y),
                             layout_y,
                             beta,
                             cl_uint(A.size1()),
                             cl_uint(A.internal_size1()),
                             cl_uint(A.ell_nnz()),
                             cl_uint(A.internal_ellnnz())
                            )
    );
  else
    viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                             A.handle().opencl_handle(),
                             A.handle3().opencl_handle(),
                             A.handle4().opencl_handle(),
                             A.handle5().opencl_handle(),
                             viennacl::traits::opencl_handle(x),
                             layout_x,
                             viennacl::traits::opencl_handle(y),
                             layout_y,
                             cl_uint(A.size1()),
                             cl_uint(A.internal_size1()),
                             cl_uint(A.ell_nnz()),
                             cl_uint(A.internal_ellnnz())
                            )
    );
}

template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::hyb_matrix<NumericT, AlignmentV> const & A,
               viennacl::matrix_base<NumericT> const & d_A,
               viennacl::matrix_base<NumericT>       & y)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::hyb_matrix<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::hyb_matrix<NumericT>::program_name(),
                                             detail::sparse_dense_matmult_kernel_name(false, d_A.row_major(), y.row_major()));

  viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                           A.handle().opencl_handle(),
                           A.handle3().opencl_handle(),
                           A.handle4().opencl_handle(),
                           A.handle5().opencl_handle(),
                           cl_uint(A.size1()),
                           cl_uint(A.internal_size1()),
                           cl_uint(A.ell_nnz()),
                           cl_uint(A.internal_ellnnz()),
                           viennacl::traits::opencl_handle(d_A),
                           cl_uint(viennacl::traits::start1(d_A)),          cl_uint(viennacl::traits::start2(d_A)),
                           cl_uint(viennacl::traits::stride1(d_A)),         cl_uint(viennacl::traits::stride2(d_A)),
                           cl_uint(viennacl::traits::size1(d_A)),           cl_uint(viennacl::traits::size2(d_A)),
                           cl_uint(viennacl::traits::internal_size1(d_A)),  cl_uint(viennacl::traits::internal_size2(d_A)),
                           viennacl::traits::opencl_handle(y),
                           cl_uint(viennacl::traits::start1(y)),         cl_uint(viennacl::traits::start2(y)),
                           cl_uint(viennacl::traits::stride1(y)),        cl_uint(viennacl::traits::stride2(y)),
                           cl_uint(viennacl::traits::size1(y)),          cl_uint(viennacl::traits::size2(y)),
                           cl_uint(viennacl::traits::internal_size1(y)), cl_uint(viennacl::traits::internal_size2(y))
                          )
  );
}

template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::hyb_matrix<NumericT, AlignmentV> const & A,
               viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                            const viennacl::matrix_base<NumericT>,
                                            viennacl::op_trans > const & d_A,
               viennacl::matrix_base<NumericT> & y)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::hyb_matrix<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::hyb_matrix<NumericT>::program_name(),
                                             detail::sparse_dense_matmult_kernel_name(true, d_A.lhs().row_major(), y.row_major()));

  viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                           A.handle().opencl_handle(),
                           A.handle3().opencl_handle(),
                           A.handle4().opencl_handle(),
                           A.handle5().opencl_handle(),
                           cl_uint(A.size1()),
                           cl_uint(A.internal_size1()),
                           cl_uint(A.ell_nnz()),
                           cl_uint(A.internal_ellnnz()),
                           viennacl::traits::opencl_handle(d_A.lhs()),
                           cl_uint(viennacl::traits::start1(d_A.lhs())),          cl_uint(viennacl::traits::start2(d_A.lhs())),
                           cl_uint(viennacl::traits::stride1(d_A.lhs())),         cl_uint(viennacl::traits::stride2(d_A.lhs())),
                           cl_uint(viennacl::traits::size1(d_A.lhs())),           cl_uint(viennacl::traits::size2(d_A.lhs())),
                           cl_uint(viennacl::traits::internal_size1(d_A.lhs())),  cl_uint(viennacl::traits::internal_size2(d_A.lhs())),
                           viennacl::traits::opencl_handle(y),
                           cl_uint(viennacl::traits::start1(y)),         cl_uint(viennacl::traits::start2(y)),
                           cl_uint(viennacl::traits::stride1(y)),        cl_uint(viennacl::traits::stride2(y)),
                           cl_uint(viennacl::traits::size1(y)),          cl_uint(viennacl::traits::size2(y)),
                           cl_uint(viennacl::traits::internal_size1(y)), cl_uint(viennacl::traits::internal_size2(y))
                          )
  );
}


} // namespace opencl
} //namespace linalg
} //namespace viennacl


#endif
