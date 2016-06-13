#ifndef VIENNACL_LINALG_OPENCL_ITERATIVE_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_ITERATIVE_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/iterative_operations.hpp
    @brief  Implementations of specialized kernels for fast iterative solvers using OpenCL
*/

#include <cmath>

#include "viennacl/forwards.h"
#include "viennacl/detail/vector_def.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/linalg/opencl/kernels/iterative.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"

namespace viennacl
{
namespace linalg
{
namespace opencl
{

template<typename NumericT>
void pipelined_cg_vector_update(vector_base<NumericT> & result,
                                NumericT alpha,
                                vector_base<NumericT> & p,
                                vector_base<NumericT> & r,
                                vector_base<NumericT> const & Ap,
                                NumericT beta,
                                vector_base<NumericT> & inner_prod_buffer)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(result).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), "cg_vector_update");
  cl_uint    vec_size = cl_uint(viennacl::traits::size(result));

  k.local_work_size(0, 128);
  k.global_work_size(0, 128*128);

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
  {
    k.local_work_size(0, 256);
    k.global_work_size(0, 256*256);
  }

  viennacl::ocl::enqueue(k(result, alpha, p, r, Ap, beta, inner_prod_buffer, vec_size, viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))));
}

template<typename NumericT>
void pipelined_cg_prod(compressed_matrix<NumericT> const & A,
                       vector_base<NumericT> const & p,
                       vector_base<NumericT> & Ap,
                       vector_base<NumericT> & inner_prod_buffer)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  bool use_nvidia_blocked = (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id && (double(A.nnz()) / double(A.size1()) > 12.0));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), use_nvidia_blocked ? "cg_csr_blocked_prod" : "cg_csr_prod");

  cl_uint vec_size               = cl_uint(viennacl::traits::size(p));
  cl_uint buffer_size_per_vector = cl_uint(inner_prod_buffer.size()) / cl_uint(3);

  k.local_work_size(0, 128);
  k.global_work_size(0, 128*128);

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
  {
    k.local_work_size(0, 256);
    k.global_work_size(0, 256*256);
  }

  if (use_nvidia_blocked)
  {
    viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(),
                             p,
                             Ap,
                             vec_size,
                             inner_prod_buffer,
                             buffer_size_per_vector,
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                            ));
  }
  else
  {
    viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle3().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.blocks1()),
                             p,
                             Ap,
                             vec_size,
                             inner_prod_buffer,
                             buffer_size_per_vector,
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                             viennacl::ocl::local_mem(1024 * sizeof(NumericT))
                            ));
  }

}

template<typename NumericT>
void pipelined_cg_prod(coordinate_matrix<NumericT> const & A,
                       vector_base<NumericT> const & p,
                       vector_base<NumericT> & Ap,
                       vector_base<NumericT> & inner_prod_buffer)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  cl_uint vec_size               = cl_uint(viennacl::traits::size(p));
  cl_uint buffer_size_per_vector = cl_uint(inner_prod_buffer.size()) / cl_uint(3);

  Ap.clear();

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), "cg_coo_prod");
  unsigned int thread_num = 256; //k.local_work_size(0);

  k.local_work_size(0, thread_num);

  k.global_work_size(0, 64 * thread_num);  //64 work groups are hard-coded for now. Gives reasonable performance in most cases

  viennacl::ocl::enqueue(k(A.handle12().opencl_handle(), A.handle().opencl_handle(), A.handle3().opencl_handle(),
                           p,
                           Ap,
                           vec_size,
                           viennacl::ocl::local_mem(sizeof(cl_uint)*thread_num),
                           viennacl::ocl::local_mem(sizeof(NumericT)*thread_num),
                           inner_prod_buffer,
                           buffer_size_per_vector,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                          ));
}

template<typename NumericT>
void pipelined_cg_prod(ell_matrix<NumericT> const & A,
                       vector_base<NumericT> const & p,
                       vector_base<NumericT> & Ap,
                       vector_base<NumericT> & inner_prod_buffer)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  cl_uint vec_size               = cl_uint(viennacl::traits::size(p));
  cl_uint buffer_size_per_vector = cl_uint(inner_prod_buffer.size()) / cl_uint(3);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), "cg_ell_prod");

  unsigned int thread_num = 128;
  unsigned int group_num = 256;

  k.local_work_size(0, thread_num);
  k.global_work_size(0, thread_num * group_num);

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
  {
    k.local_work_size(0, 256);
    k.global_work_size(0, 256*256);
  }

  viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                           A.handle().opencl_handle(),
                           cl_uint(A.internal_size1()),
                           cl_uint(A.maxnnz()),
                           cl_uint(A.internal_maxnnz()),
                           viennacl::traits::opencl_handle(p),
                           viennacl::traits::opencl_handle(Ap),
                           vec_size,
                           inner_prod_buffer,
                           buffer_size_per_vector,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                          )
                         );
}

template<typename NumericT>
void pipelined_cg_prod(sliced_ell_matrix<NumericT> const & A,
                       vector_base<NumericT> const & p,
                       vector_base<NumericT> & Ap,
                       vector_base<NumericT> & inner_prod_buffer)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  cl_uint vec_size               = cl_uint(viennacl::traits::size(p));
  cl_uint buffer_size_per_vector = cl_uint(inner_prod_buffer.size()) / cl_uint(3);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), "cg_sliced_ell_prod");

  vcl_size_t thread_num = std::max(A.rows_per_block(), static_cast<vcl_size_t>(128));
  unsigned int group_num = 256;

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
    thread_num = 256;

  k.local_work_size(0, thread_num);
  k.global_work_size(0, thread_num * group_num);

  viennacl::ocl::enqueue(k(A.handle1().opencl_handle(),
                           A.handle2().opencl_handle(),
                           A.handle3().opencl_handle(),
                           A.handle().opencl_handle(),
                           viennacl::traits::opencl_handle(p),
                           viennacl::traits::opencl_handle(Ap),
                           vec_size,
                           cl_uint(A.rows_per_block()),
                           inner_prod_buffer,
                           buffer_size_per_vector,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                          )
                        );
}


template<typename NumericT>
void pipelined_cg_prod(hyb_matrix<NumericT> const & A,
                       vector_base<NumericT> const & p,
                       vector_base<NumericT> & Ap,
                       vector_base<NumericT> & inner_prod_buffer)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  cl_uint vec_size               = cl_uint(viennacl::traits::size(p));
  cl_uint buffer_size_per_vector = cl_uint(inner_prod_buffer.size()) / cl_uint(3);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), "cg_hyb_prod");

  unsigned int thread_num = 128;
  unsigned int group_num = 128;

  k.local_work_size(0, thread_num);
  k.global_work_size(0, thread_num * group_num);

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
  {
    k.local_work_size(0, 256);
    k.global_work_size(0, 256*256);
  }

  viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                           A.handle().opencl_handle(),
                           A.handle3().opencl_handle(),
                           A.handle4().opencl_handle(),
                           A.handle5().opencl_handle(),
                           cl_uint(A.internal_size1()),
                           cl_uint(A.ell_nnz()),
                           cl_uint(A.internal_ellnnz()),
                           viennacl::traits::opencl_handle(p),
                           viennacl::traits::opencl_handle(Ap),
                           vec_size,
                           inner_prod_buffer,
                           buffer_size_per_vector,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                          )
                        );
}


//////////////////////////// BiCGStab ////////////////////////

template<typename NumericT>
void pipelined_bicgstab_update_s(vector_base<NumericT> & s,
                                 vector_base<NumericT> & r,
                                 vector_base<NumericT> const & Ap,
                                 vector_base<NumericT> & inner_prod_buffer,
                                 vcl_size_t buffer_chunk_size,
                                 vcl_size_t buffer_chunk_offset)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(s).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), "bicgstab_update_s");
  cl_uint    vec_size = cl_uint(viennacl::traits::size(s));

  k.local_work_size(0, 128);
  k.global_work_size(0, 128*128);

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
  {
    k.local_work_size(0, 256);
    k.global_work_size(0, 256*256);
  }

  cl_uint chunk_size   = cl_uint(buffer_chunk_size);
  cl_uint chunk_offset = cl_uint(buffer_chunk_offset);
  viennacl::ocl::enqueue(k(s, r, Ap,
                           inner_prod_buffer, chunk_size, chunk_offset, vec_size,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))));
}

template<typename NumericT>
void pipelined_bicgstab_vector_update(vector_base<NumericT> & result, NumericT alpha, vector_base<NumericT> & p, NumericT omega, vector_base<NumericT> const & s,
                                      vector_base<NumericT> & residual, vector_base<NumericT> const & As,
                                      NumericT beta, vector_base<NumericT> const & Ap,
                                      vector_base<NumericT> const & r0star,
                                      vector_base<NumericT> & inner_prod_buffer, vcl_size_t buffer_chunk_size)
{
  (void)buffer_chunk_size;

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(s).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), "bicgstab_vector_update");
  cl_uint    vec_size = cl_uint(viennacl::traits::size(result));

  k.local_work_size(0, 128);
  k.global_work_size(0, 128*128);

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
  {
    k.local_work_size(0, 256);
    k.global_work_size(0, 256*256);
  }

  viennacl::ocl::enqueue(k(result, alpha, p, omega, s,
                           residual, As,
                           beta, Ap,
                           r0star,
                           inner_prod_buffer,
                           vec_size, viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                           )
                         );
}

template<typename NumericT>
void pipelined_bicgstab_prod(compressed_matrix<NumericT> const & A,
                             vector_base<NumericT> const & p,
                             vector_base<NumericT> & Ap,
                             vector_base<NumericT> const & r0star,
                             vector_base<NumericT> & inner_prod_buffer,
                             vcl_size_t buffer_chunk_size,
                             vcl_size_t buffer_chunk_offset)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  bool use_nvidia_blocked = (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id && (double(A.nnz()) / double(A.size1()) > 12.0));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), use_nvidia_blocked ? "bicgstab_csr_blocked_prod" : "bicgstab_csr_prod");

  cl_uint vec_size     = cl_uint(viennacl::traits::size(p));
  cl_uint chunk_size   = cl_uint(buffer_chunk_size);
  cl_uint chunk_offset = cl_uint(buffer_chunk_offset);

  k.local_work_size(0, 128);
  k.global_work_size(0, 128*128);

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
  {
    k.local_work_size(0, 256);
    k.global_work_size(0, 256*256);
  }

  if (use_nvidia_blocked)
  {
    viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(),
                             p,
                             Ap,
                             r0star,
                             vec_size,
                             inner_prod_buffer, chunk_size, chunk_offset,
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                            ));
  }
  else
  {
    viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle3().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.blocks1()),
                             p,
                             Ap,
                             r0star,
                             vec_size,
                             inner_prod_buffer, chunk_size, chunk_offset,
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                            ));
  }

}


template<typename NumericT>
void pipelined_bicgstab_prod(coordinate_matrix<NumericT> const & A,
                             vector_base<NumericT> const & p,
                             vector_base<NumericT> & Ap,
                             vector_base<NumericT> const & r0star,
                             vector_base<NumericT> & inner_prod_buffer,
                             vcl_size_t buffer_chunk_size,
                             vcl_size_t buffer_chunk_offset)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  cl_uint vec_size     = cl_uint(viennacl::traits::size(p));
  cl_uint chunk_size   = cl_uint(buffer_chunk_size);
  cl_uint chunk_offset = cl_uint(buffer_chunk_offset);

  Ap.clear();

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), "bicgstab_coo_prod");
  unsigned int thread_num = 256; //k.local_work_size(0);

  k.local_work_size(0, thread_num);

  k.global_work_size(0, 64 * thread_num);  //64 work groups are hard-coded for now. Gives reasonable performance in most cases

  viennacl::ocl::enqueue(k(A.handle12().opencl_handle(), A.handle().opencl_handle(), A.handle3().opencl_handle(),
                           p,
                           Ap,
                           r0star,
                           vec_size,
                           viennacl::ocl::local_mem(sizeof(cl_uint)*thread_num),
                           viennacl::ocl::local_mem(sizeof(NumericT)*thread_num),
                           inner_prod_buffer, chunk_size, chunk_offset,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                          ));
}

template<typename NumericT>
void pipelined_bicgstab_prod(ell_matrix<NumericT> const & A,
                             vector_base<NumericT> const & p,
                             vector_base<NumericT> & Ap,
                             vector_base<NumericT> const & r0star,
                             vector_base<NumericT> & inner_prod_buffer,
                             vcl_size_t buffer_chunk_size,
                             vcl_size_t buffer_chunk_offset)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  cl_uint vec_size     = cl_uint(viennacl::traits::size(p));
  cl_uint chunk_size   = cl_uint(buffer_chunk_size);
  cl_uint chunk_offset = cl_uint(buffer_chunk_offset);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), "bicgstab_ell_prod");

  unsigned int thread_num = 128;
  unsigned int group_num = 128;

  k.local_work_size(0, thread_num);
  k.global_work_size(0, thread_num * group_num);

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
  {
    k.local_work_size(0, 256);
    k.global_work_size(0, 256*256);
  }

  viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                           A.handle().opencl_handle(),
                           cl_uint(A.internal_size1()),
                           cl_uint(A.maxnnz()),
                           cl_uint(A.internal_maxnnz()),
                           viennacl::traits::opencl_handle(p),
                           viennacl::traits::opencl_handle(Ap),
                           r0star,
                           vec_size,
                           inner_prod_buffer, chunk_size, chunk_offset,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                          )
                         );
}

template<typename NumericT>
void pipelined_bicgstab_prod(sliced_ell_matrix<NumericT> const & A,
                             vector_base<NumericT> const & p,
                             vector_base<NumericT> & Ap,
                             vector_base<NumericT> const & r0star,
                             vector_base<NumericT> & inner_prod_buffer,
                             vcl_size_t buffer_chunk_size,
                             vcl_size_t buffer_chunk_offset)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  cl_uint vec_size     = cl_uint(viennacl::traits::size(p));
  cl_uint chunk_size   = cl_uint(buffer_chunk_size);
  cl_uint chunk_offset = cl_uint(buffer_chunk_offset);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), "bicgstab_sliced_ell_prod");

  vcl_size_t thread_num = std::max(A.rows_per_block(), static_cast<vcl_size_t>(128));
  unsigned int group_num = 256;

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
    thread_num = 256;

  k.local_work_size(0, thread_num);
  k.global_work_size(0, thread_num * group_num);

  viennacl::ocl::enqueue(k(A.handle1().opencl_handle(),
                           A.handle2().opencl_handle(),
                           A.handle3().opencl_handle(),
                           A.handle().opencl_handle(),
                           viennacl::traits::opencl_handle(p),
                           viennacl::traits::opencl_handle(Ap),
                           r0star,
                           vec_size,
                           cl_uint(A.rows_per_block()),
                           inner_prod_buffer, chunk_size, chunk_offset,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                          )
                        );
}


template<typename NumericT>
void pipelined_bicgstab_prod(hyb_matrix<NumericT> const & A,
                             vector_base<NumericT> const & p,
                             vector_base<NumericT> & Ap,
                             vector_base<NumericT> const & r0star,
                             vector_base<NumericT> & inner_prod_buffer,
                             vcl_size_t buffer_chunk_size,
                             vcl_size_t buffer_chunk_offset)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<NumericT>::init(ctx);

  cl_uint vec_size     = cl_uint(viennacl::traits::size(p));
  cl_uint chunk_size   = cl_uint(buffer_chunk_size);
  cl_uint chunk_offset = cl_uint(buffer_chunk_offset);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<NumericT>::program_name(), "bicgstab_hyb_prod");

  unsigned int thread_num = 256;
  unsigned int group_num = 128;

  k.local_work_size(0, thread_num);
  k.global_work_size(0, thread_num * group_num);

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
  {
    k.local_work_size(0, 256);
    k.global_work_size(0, 256*256);
  }

  viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                           A.handle().opencl_handle(),
                           A.handle3().opencl_handle(),
                           A.handle4().opencl_handle(),
                           A.handle5().opencl_handle(),
                           cl_uint(A.internal_size1()),
                           cl_uint(A.ell_nnz()),
                           cl_uint(A.internal_ellnnz()),
                           viennacl::traits::opencl_handle(p),
                           viennacl::traits::opencl_handle(Ap),
                           r0star,
                           vec_size,
                           inner_prod_buffer, chunk_size, chunk_offset,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(NumericT))
                          )
                        );
}

///////////////////////////////////

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
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(v_k).context());
  viennacl::linalg::opencl::kernels::iterative<T>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<T>::program_name(), "gmres_normalize_vk");

  k.local_work_size(0, 128);
  k.global_work_size(0, 128*128);

  cl_uint size_vk      = cl_uint(v_k.size());
  cl_uint vk_offset    = cl_uint(viennacl::traits::start(v_k));
  cl_uint R_offset     = cl_uint(offset_in_R);
  cl_uint chunk_size   = cl_uint(buffer_chunk_size);
  cl_uint chunk_offset = cl_uint(buffer_chunk_offset);
  viennacl::ocl::enqueue(k(v_k, vk_offset,
                           residual,
                           R_buffer, R_offset,
                           inner_prod_buffer, chunk_size,
                           r_dot_vk_buffer, chunk_offset,
                           size_vk,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(T))
                           ));
}

template <typename T>
void pipelined_gmres_gram_schmidt_stage1(vector_base<T> const & device_krylov_basis,
                                         vcl_size_t v_k_size,
                                         vcl_size_t v_k_internal_size,
                                         vcl_size_t param_k,
                                         vector_base<T> & vi_in_vk_buffer,
                                         vcl_size_t buffer_chunk_size)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(device_krylov_basis).context());
  viennacl::linalg::opencl::kernels::iterative<T>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<T>::program_name(), "gmres_gram_schmidt_1");

  k.local_work_size(0, 128);
  k.global_work_size(0, 128*128);

  cl_uint size_vk          = cl_uint(v_k_size);
  cl_uint internal_size_vk = cl_uint(v_k_internal_size);
  cl_uint ocl_k            = cl_uint(param_k);
  cl_uint chunk_size = cl_uint(buffer_chunk_size);
  viennacl::ocl::enqueue(k(device_krylov_basis, size_vk, internal_size_vk, ocl_k,
                           vi_in_vk_buffer, chunk_size
                           ));
}

template <typename T>
void pipelined_gmres_gram_schmidt_stage2(vector_base<T> & device_krylov_basis,
                                         vcl_size_t v_k_size,
                                         vcl_size_t v_k_internal_size,
                                         vcl_size_t param_k,
                                         vector_base<T> const & vi_in_vk_buffer,
                                         vector_base<T> & R_buffer,
                                         vcl_size_t krylov_dim,
                                         vector_base<T> & inner_prod_buffer,
                                         vcl_size_t buffer_chunk_size)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(device_krylov_basis).context());
  viennacl::linalg::opencl::kernels::iterative<T>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<T>::program_name(), "gmres_gram_schmidt_2");

  k.local_work_size(0, 128);
  k.global_work_size(0, 128*128);

  cl_uint size_vk          = cl_uint(v_k_size);
  cl_uint internal_size_vk = cl_uint(v_k_internal_size);
  cl_uint ocl_k            = cl_uint(param_k);
  cl_uint chunk_size       = cl_uint(buffer_chunk_size);
  cl_uint ocl_krylov_dim   = cl_uint(krylov_dim);
  viennacl::ocl::enqueue(k(device_krylov_basis, size_vk, internal_size_vk, ocl_k,
                           vi_in_vk_buffer, chunk_size,
                           R_buffer, ocl_krylov_dim,
                           inner_prod_buffer,
                           viennacl::ocl::local_mem(7 * k.local_work_size() * sizeof(T))
                           ));
}

template <typename T>
void pipelined_gmres_update_result(vector_base<T> & result,
                                   vector_base<T> const & residual,
                                   vector_base<T> const & krylov_basis,
                                   vcl_size_t v_k_size,
                                   vcl_size_t v_k_internal_size,
                                   vector_base<T> const & coefficients,
                                   vcl_size_t param_k)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(result).context());
  viennacl::linalg::opencl::kernels::iterative<T>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<T>::program_name(), "gmres_update_result");

  k.local_work_size(0, 128);
  k.global_work_size(0, 128*128);

  cl_uint size_vk          = cl_uint(v_k_size);
  cl_uint internal_size_vk = cl_uint(v_k_internal_size);
  cl_uint ocl_k            = cl_uint(param_k);
  viennacl::ocl::enqueue(k(result,
                           residual,
                           krylov_basis, size_vk, internal_size_vk,
                           coefficients, ocl_k
                           ));
}


template <typename T>
void pipelined_gmres_prod(compressed_matrix<T> const & A,
                          vector_base<T> const & p,
                          vector_base<T> & Ap,
                          vector_base<T> & inner_prod_buffer)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<T>::init(ctx);

  bool use_nvidia_blocked = (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id && (double(A.nnz()) / double(A.size1()) > 12.0));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<T>::program_name(), use_nvidia_blocked ? "gmres_csr_blocked_prod" : "gmres_csr_prod");

  cl_uint vec_size               = cl_uint(viennacl::traits::size(p));
  cl_uint buffer_size_per_vector = cl_uint(inner_prod_buffer.size()) / cl_uint(3);
  cl_uint start_p                = cl_uint(viennacl::traits::start(p));
  cl_uint start_Ap               = cl_uint(viennacl::traits::start(Ap));

  k.local_work_size(0, 128);
  k.global_work_size(0, 128*128);

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
  {
    k.local_work_size(0, 256);
    k.global_work_size(0, 256*128);
  }

  if (use_nvidia_blocked)
  {
    viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(),
                             p, start_p,
                             Ap, start_Ap,
                             vec_size,
                             inner_prod_buffer,
                             buffer_size_per_vector,
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(T)),
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(T))
                            ));
  }
  else
  {
    viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle3().opencl_handle(), A.handle().opencl_handle(), cl_uint(A.blocks1()),
                             p, start_p,
                             Ap, start_Ap,
                             vec_size,
                             inner_prod_buffer,
                             buffer_size_per_vector,
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(T)),
                             viennacl::ocl::local_mem(k.local_work_size() * sizeof(T)),
                             viennacl::ocl::local_mem(1024 * sizeof(T))
                            ));
  }
}

template <typename T>
void pipelined_gmres_prod(coordinate_matrix<T> const & A,
                          vector_base<T> const & p,
                          vector_base<T> & Ap,
                          vector_base<T> & inner_prod_buffer)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<T>::init(ctx);

  cl_uint vec_size               = cl_uint(viennacl::traits::size(p));
  cl_uint buffer_size_per_vector = cl_uint(inner_prod_buffer.size()) / cl_uint(3);
  cl_uint start_p                = cl_uint(viennacl::traits::start(p));
  cl_uint start_Ap               = cl_uint(viennacl::traits::start(Ap));

  Ap.clear();
  inner_prod_buffer.clear();

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<T>::program_name(), "gmres_coo_prod");
  unsigned int thread_num = 128; //k.local_work_size(0);

  k.local_work_size(0, thread_num);

  k.global_work_size(0, 64 * thread_num);  //64 work groups are hard-coded for now. Gives reasonable performance in most cases

  viennacl::ocl::enqueue(k(A.handle12().opencl_handle(), A.handle().opencl_handle(), A.handle3().opencl_handle(),
                           p, start_p,
                           Ap, start_Ap,
                           vec_size,
                           viennacl::ocl::local_mem(sizeof(cl_uint)*thread_num),
                           viennacl::ocl::local_mem(sizeof(T)*thread_num),
                           inner_prod_buffer,
                           buffer_size_per_vector,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(T)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(T))
                          ));
}

template <typename T>
void pipelined_gmres_prod(ell_matrix<T> const & A,
                          vector_base<T> const & p,
                          vector_base<T> & Ap,
                          vector_base<T> & inner_prod_buffer)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<T>::init(ctx);

  cl_uint vec_size               = cl_uint(viennacl::traits::size(p));
  cl_uint buffer_size_per_vector = cl_uint(inner_prod_buffer.size()) / cl_uint(3);
  cl_uint start_p                = cl_uint(viennacl::traits::start(p));
  cl_uint start_Ap               = cl_uint(viennacl::traits::start(Ap));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<T>::program_name(), "gmres_ell_prod");

  unsigned int thread_num = (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id) ? 256 : 128;
  unsigned int group_num = 128;

  k.local_work_size(0, thread_num);
  k.global_work_size(0, thread_num * group_num);

  viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                           A.handle().opencl_handle(),
                           cl_uint(A.internal_size1()),
                           cl_uint(A.maxnnz()),
                           cl_uint(A.internal_maxnnz()),
                           viennacl::traits::opencl_handle(p), start_p,
                           viennacl::traits::opencl_handle(Ap), start_Ap,
                           vec_size,
                           inner_prod_buffer,
                           buffer_size_per_vector,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(T)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(T))
                          )
                         );
}

template <typename T>
void pipelined_gmres_prod(sliced_ell_matrix<T> const & A,
                          vector_base<T> const & p,
                          vector_base<T> & Ap,
                          vector_base<T> & inner_prod_buffer)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<T>::init(ctx);

  cl_uint vec_size               = cl_uint(viennacl::traits::size(p));
  cl_uint buffer_size_per_vector = cl_uint(inner_prod_buffer.size()) / cl_uint(3);
  cl_uint start_p                = cl_uint(viennacl::traits::start(p));
  cl_uint start_Ap               = cl_uint(viennacl::traits::start(Ap));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<T>::program_name(), "gmres_sliced_ell_prod");

  vcl_size_t thread_num = std::max(A.rows_per_block(), static_cast<vcl_size_t>(128));
  unsigned int group_num = 128;

  if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
    thread_num = 256;

  k.local_work_size(0, thread_num);
  k.global_work_size(0, thread_num * group_num);

  viennacl::ocl::enqueue(k(A.handle1().opencl_handle(),
                           A.handle2().opencl_handle(),
                           A.handle3().opencl_handle(),
                           A.handle().opencl_handle(),
                           viennacl::traits::opencl_handle(p), start_p,
                           viennacl::traits::opencl_handle(Ap), start_Ap,
                           vec_size,
                           cl_uint(A.rows_per_block()),
                           inner_prod_buffer,
                           buffer_size_per_vector,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(T)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(T))
                          )
                        );
}


template <typename T>
void pipelined_gmres_prod(hyb_matrix<T> const & A,
                          vector_base<T> const & p,
                          vector_base<T> & Ap,
                          vector_base<T> & inner_prod_buffer)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::iterative<T>::init(ctx);

  cl_uint vec_size               = cl_uint(viennacl::traits::size(p));
  cl_uint buffer_size_per_vector = cl_uint(inner_prod_buffer.size()) / cl_uint(3);
  cl_uint start_p                = cl_uint(viennacl::traits::start(p));
  cl_uint start_Ap               = cl_uint(viennacl::traits::start(Ap));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::iterative<T>::program_name(), "gmres_hyb_prod");

  unsigned int thread_num = (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id) ? 256 : 128;
  unsigned int group_num = 128;

  k.local_work_size(0, thread_num);
  k.global_work_size(0, thread_num * group_num);


  viennacl::ocl::enqueue(k(A.handle2().opencl_handle(),
                           A.handle().opencl_handle(),
                           A.handle3().opencl_handle(),
                           A.handle4().opencl_handle(),
                           A.handle5().opencl_handle(),
                           cl_uint(A.internal_size1()),
                           cl_uint(A.ell_nnz()),
                           cl_uint(A.internal_ellnnz()),
                           viennacl::traits::opencl_handle(p), start_p,
                           viennacl::traits::opencl_handle(Ap), start_Ap,
                           vec_size,
                           inner_prod_buffer,
                           buffer_size_per_vector,
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(T)),
                           viennacl::ocl::local_mem(k.local_work_size() * sizeof(T))
                          )
                        );
}


} //namespace opencl
} //namespace linalg
} //namespace viennacl


#endif
