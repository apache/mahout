#ifndef VIENNACL_LINALG_OPENCL_ILU_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_ILU_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/ilu_operations.hpp
    @brief Implementations of specialized routines for the Chow-Patel parallel ILU preconditioner using OpenCL
*/

#include <cmath>
#include <algorithm>  //for std::max and std::min

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/linalg/opencl/kernels/ilu.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/vector_operations.hpp"


namespace viennacl
{
namespace linalg
{
namespace opencl
{

/////////////////////// ICC /////////////////////

template<typename NumericT>
void extract_L(compressed_matrix<NumericT> const & A,
                compressed_matrix<NumericT>       & L)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::ilu<NumericT>::init(ctx);

  //
  // Step 1: Count elements in L:
  //
  viennacl::ocl::kernel & k1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "extract_L_1");

  viennacl::ocl::enqueue(k1(A.handle1().opencl_handle(), A.handle2().opencl_handle(), cl_uint(A.size1()),
                            L.handle1().opencl_handle())
                        );

  //
  // Step 2: Exclusive scan on row_buffers:
  //
  viennacl::vector_base<unsigned int> wrapped_L_row_buffer(L.handle1(), A.size1() + 1, 0, 1);
  viennacl::linalg::exclusive_scan(wrapped_L_row_buffer, wrapped_L_row_buffer);
  L.reserve(wrapped_L_row_buffer[L.size1()], false);


  //
  // Step 3: Write entries
  //
  viennacl::ocl::kernel & k2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "extract_L_2");

  viennacl::ocl::enqueue(k2(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.size1()),
                            L.handle1().opencl_handle(), L.handle2().opencl_handle(), L.handle().opencl_handle())
                        );

  L.generate_row_block_information();

} // extract_LU

///////////////////////////////////////////////



/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L and U accordingly. */
template<typename NumericT>
void icc_scale(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L)
{
  viennacl::vector<NumericT> D(A.size1(), viennacl::traits::context(A));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::ilu<NumericT>::init(ctx);

  // fill D:
  viennacl::ocl::kernel & k1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_scale_kernel_1");
  viennacl::ocl::enqueue(k1(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.size1()), D) );

  // scale L:
  viennacl::ocl::kernel & k2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_scale_kernel_2");
  viennacl::ocl::enqueue(k2(L.handle1().opencl_handle(), L.handle2().opencl_handle(), L.handle().opencl_handle(), cl_uint(A.size1()), D) );

}

/////////////////////////////////////


/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ILU using OpenCL (cf. Algorithm 2 in paper) */
template<typename NumericT>
void icc_chow_patel_sweep(compressed_matrix<NumericT>       & L,
                          vector<NumericT>            const & aij_L)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(L).context());
  viennacl::linalg::opencl::kernels::ilu<NumericT>::init(ctx);

  viennacl::backend::mem_handle L_backup;
  viennacl::backend::memory_create(L_backup, L.handle().raw_size(), viennacl::traits::context(L));
  viennacl::backend::memory_copy(L.handle(), L_backup, 0, 0, L.handle().raw_size());

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "icc_chow_patel_sweep_kernel");
  viennacl::ocl::enqueue(k(L.handle1().opencl_handle(), L.handle2().opencl_handle(), L.handle().opencl_handle(), L_backup.opencl_handle(), cl_uint(L.size1()),
                           aij_L)
                        );

}


/////////////////////// ILU /////////////////////

template<typename NumericT>
void extract_LU(compressed_matrix<NumericT> const & A,
                compressed_matrix<NumericT>       & L,
                compressed_matrix<NumericT>       & U)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::ilu<NumericT>::init(ctx);

  //
  // Step 1: Count elements in L and U:
  //
  viennacl::ocl::kernel & k1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "extract_LU_1");

  viennacl::ocl::enqueue(k1(A.handle1().opencl_handle(), A.handle2().opencl_handle(), cl_uint(A.size1()),
                            L.handle1().opencl_handle(),
                            U.handle1().opencl_handle())
                        );

  //
  // Step 2: Exclusive scan on row_buffers:
  //
  viennacl::vector_base<unsigned int> wrapped_L_row_buffer(L.handle1(), A.size1() + 1, 0, 1);
  viennacl::linalg::exclusive_scan(wrapped_L_row_buffer, wrapped_L_row_buffer);
  L.reserve(wrapped_L_row_buffer[L.size1()], false);

  viennacl::vector_base<unsigned int> wrapped_U_row_buffer(U.handle1(), A.size1() + 1, 0, 1);
  viennacl::linalg::exclusive_scan(wrapped_U_row_buffer, wrapped_U_row_buffer);
  U.reserve(wrapped_U_row_buffer[U.size1()], false);

  //
  // Step 3: Write entries
  //
  viennacl::ocl::kernel & k2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "extract_LU_2");

  viennacl::ocl::enqueue(k2(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.size1()),
                            L.handle1().opencl_handle(), L.handle2().opencl_handle(), L.handle().opencl_handle(),
                            U.handle1().opencl_handle(), U.handle2().opencl_handle(), U.handle().opencl_handle())
                        );

  L.generate_row_block_information();
  // Note: block information for U will be generated after transposition

} // extract_LU

///////////////////////////////////////////////



/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L and U accordingly. */
template<typename NumericT>
void ilu_scale(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L,
               compressed_matrix<NumericT>       & U)
{
  viennacl::vector<NumericT> D(A.size1(), viennacl::traits::context(A));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::ilu<NumericT>::init(ctx);

  // fill D:
  viennacl::ocl::kernel & k1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_scale_kernel_1");
  viennacl::ocl::enqueue(k1(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.size1()), D) );

  // scale L:
  viennacl::ocl::kernel & k2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_scale_kernel_2");
  viennacl::ocl::enqueue(k2(L.handle1().opencl_handle(), L.handle2().opencl_handle(), L.handle().opencl_handle(), cl_uint(A.size1()), D) );

  // scale U:
  viennacl::ocl::enqueue(k2(U.handle1().opencl_handle(), U.handle2().opencl_handle(), U.handle().opencl_handle(), cl_uint(A.size1()), D) );

}

/////////////////////////////////////


/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ILU using OpenCL (cf. Algorithm 2 in paper) */
template<typename NumericT>
void ilu_chow_patel_sweep(compressed_matrix<NumericT>       & L,
                          vector<NumericT>            const & aij_L,
                          compressed_matrix<NumericT>       & U_trans,
                          vector<NumericT>            const & aij_U_trans)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(L).context());
  viennacl::linalg::opencl::kernels::ilu<NumericT>::init(ctx);

  viennacl::backend::mem_handle L_backup;
  viennacl::backend::memory_create(L_backup, L.handle().raw_size(), viennacl::traits::context(L));
  viennacl::backend::memory_copy(L.handle(), L_backup, 0, 0, L.handle().raw_size());

  viennacl::backend::mem_handle U_backup;
  viennacl::backend::memory_create(U_backup, U_trans.handle().raw_size(), viennacl::traits::context(U_trans));
  viennacl::backend::memory_copy(U_trans.handle(), U_backup, 0, 0, U_trans.handle().raw_size());

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_chow_patel_sweep_kernel");
  viennacl::ocl::enqueue(k(L.handle1().opencl_handle(), L.handle2().opencl_handle(), L.handle().opencl_handle(), L_backup.opencl_handle(), cl_uint(L.size1()),
                           aij_L,
                           U_trans.handle1().opencl_handle(), U_trans.handle2().opencl_handle(), U_trans.handle().opencl_handle(), U_backup.opencl_handle(),
                           aij_U_trans)
                        );

}

//////////////////////////////////////



template<typename NumericT>
void ilu_form_neumann_matrix(compressed_matrix<NumericT> & R,
                             vector<NumericT> & diag_R)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(R).context());
  viennacl::linalg::opencl::kernels::ilu<NumericT>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::ilu<NumericT>::program_name(), "ilu_form_neumann_matrix_kernel");
  viennacl::ocl::enqueue(k(R.handle1().opencl_handle(), R.handle2().opencl_handle(), R.handle().opencl_handle(), cl_uint(R.size1()),
                           diag_R)
                        );
}

} //namespace opencl
} //namespace linalg
} //namespace viennacl


#endif
