#ifndef VIENNACL_LINALG_OPENCL_VECTOR_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_VECTOR_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/vector_operations.hpp
    @brief Implementations of vector operations using OpenCL
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
#include "viennacl/linalg/opencl/kernels/vector.hpp"
#include "viennacl/linalg/opencl/kernels/vector_element.hpp"
#include "viennacl/linalg/opencl/kernels/scan.hpp"
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

//
// Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
//
template<typename DestNumericT, typename SrcNumericT>
void convert(vector_base<DestNumericT> & dest, vector_base<SrcNumericT> const & src)
{
  assert(viennacl::traits::opencl_handle(dest).context() == viennacl::traits::opencl_handle(src).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  std::string kernel_name("convert_");
  kernel_name += viennacl::ocl::type_to_string<DestNumericT>::apply();
  kernel_name += "_";
  kernel_name += viennacl::ocl::type_to_string<SrcNumericT>::apply();

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(dest).context());
  viennacl::linalg::opencl::kernels::vector_convert::init(ctx);
  viennacl::ocl::kernel& k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_convert::program_name(), kernel_name);

  viennacl::ocl::enqueue(k( dest, cl_uint(dest.start()), cl_uint(dest.stride()), cl_uint(dest.size()),
                            src,  cl_uint( src.start()), cl_uint( src.stride())
                        ) );

}

template <typename T, typename ScalarType1>
void av(vector_base<T> & vec1,
        vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
  viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

  cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(),
                                             (viennacl::is_cpu_scalar<ScalarType1>::value ? "av_cpu" : "av_gpu"));
  k.global_work_size(0, std::min<vcl_size_t>(128 * k.local_work_size(),
                                              viennacl::tools::align_to_multiple<vcl_size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );

  viennacl::ocl::packed_cl_uint size_vec1;
  size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
  size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
  size_vec1.size   = cl_uint(viennacl::traits::size(vec1));
  size_vec1.internal_size   = cl_uint(viennacl::traits::internal_size(vec1));

  viennacl::ocl::packed_cl_uint size_vec2;
  size_vec2.start  = cl_uint(viennacl::traits::start(vec2));
  size_vec2.stride = cl_uint(viennacl::traits::stride(vec2));
  size_vec2.size   = cl_uint(viennacl::traits::size(vec2));
  size_vec2.internal_size   = cl_uint(viennacl::traits::internal_size(vec2));


  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                           size_vec1,

                           viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<T>(alpha)),
                           options_alpha,
                           viennacl::traits::opencl_handle(vec2),
                           size_vec2 )
                        );
}


template <typename T, typename ScalarType1, typename ScalarType2>
void avbv(vector_base<T> & vec1,
          vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          vector_base<T> const & vec3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(vec2).context() == viennacl::traits::opencl_handle(vec3).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
  viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

  std::string kernel_name;
  if (viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
    kernel_name = "avbv_cpu_cpu";
  else if (viennacl::is_cpu_scalar<ScalarType1>::value && !viennacl::is_cpu_scalar<ScalarType2>::value)
    kernel_name = "avbv_cpu_gpu";
  else if (!viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
    kernel_name = "avbv_gpu_cpu";
  else
    kernel_name = "avbv_gpu_gpu";

  cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
  cl_uint options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), kernel_name);
  k.global_work_size(0, std::min<vcl_size_t>(128 * k.local_work_size(),
                                              viennacl::tools::align_to_multiple<vcl_size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );

  viennacl::ocl::packed_cl_uint size_vec1;
  size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
  size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
  size_vec1.size   = cl_uint(viennacl::traits::size(vec1));
  size_vec1.internal_size   = cl_uint(viennacl::traits::internal_size(vec1));

  viennacl::ocl::packed_cl_uint size_vec2;
  size_vec2.start  = cl_uint(viennacl::traits::start(vec2));
  size_vec2.stride = cl_uint(viennacl::traits::stride(vec2));
  size_vec2.size   = cl_uint(viennacl::traits::size(vec2));
  size_vec2.internal_size   = cl_uint(viennacl::traits::internal_size(vec2));

  viennacl::ocl::packed_cl_uint size_vec3;
  size_vec3.start  = cl_uint(viennacl::traits::start(vec3));
  size_vec3.stride = cl_uint(viennacl::traits::stride(vec3));
  size_vec3.size   = cl_uint(viennacl::traits::size(vec3));
  size_vec3.internal_size   = cl_uint(viennacl::traits::internal_size(vec3));

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                           size_vec1,

                           viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<T>(alpha)),
                           options_alpha,
                           viennacl::traits::opencl_handle(vec2),
                           size_vec2,

                           viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<T>(beta)),
                           options_beta,
                           viennacl::traits::opencl_handle(vec3),
                           size_vec3 )
                        );
}


template <typename T, typename ScalarType1, typename ScalarType2>
void avbv_v(vector_base<T> & vec1,
            vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
            vector_base<T> const & vec3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(vec2).context() == viennacl::traits::opencl_handle(vec3).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
  viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

  std::string kernel_name;
  if (viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
    kernel_name = "avbv_v_cpu_cpu";
  else if (viennacl::is_cpu_scalar<ScalarType1>::value && !viennacl::is_cpu_scalar<ScalarType2>::value)
    kernel_name = "avbv_v_cpu_gpu";
  else if (!viennacl::is_cpu_scalar<ScalarType1>::value && viennacl::is_cpu_scalar<ScalarType2>::value)
    kernel_name = "avbv_v_gpu_cpu";
  else
    kernel_name = "avbv_v_gpu_gpu";

  cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
  cl_uint options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), kernel_name);
  k.global_work_size(0, std::min<vcl_size_t>(128 * k.local_work_size(),
                                              viennacl::tools::align_to_multiple<vcl_size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );

  viennacl::ocl::packed_cl_uint size_vec1;
  size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
  size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
  size_vec1.size   = cl_uint(viennacl::traits::size(vec1));
  size_vec1.internal_size   = cl_uint(viennacl::traits::internal_size(vec1));

  viennacl::ocl::packed_cl_uint size_vec2;
  size_vec2.start  = cl_uint(viennacl::traits::start(vec2));
  size_vec2.stride = cl_uint(viennacl::traits::stride(vec2));
  size_vec2.size   = cl_uint(viennacl::traits::size(vec2));
  size_vec2.internal_size   = cl_uint(viennacl::traits::internal_size(vec2));

  viennacl::ocl::packed_cl_uint size_vec3;
  size_vec3.start  = cl_uint(viennacl::traits::start(vec3));
  size_vec3.stride = cl_uint(viennacl::traits::stride(vec3));
  size_vec3.size   = cl_uint(viennacl::traits::size(vec3));
  size_vec3.internal_size   = cl_uint(viennacl::traits::internal_size(vec3));

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                           size_vec1,

                           viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<T>(alpha)),
                           options_alpha,
                           viennacl::traits::opencl_handle(vec2),
                           size_vec2,

                           viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<T>(beta)),
                           options_beta,
                           viennacl::traits::opencl_handle(vec3),
                           size_vec3 )
                        );
}


/** @brief Assign a constant value to a vector (-range/-slice)
*
* @param vec1   The vector to which the value should be assigned
* @param alpha  The value to be assigned
* @param up_to_internal_size  Specifies whether alpha should also be written to padded memory (mostly used for clearing the whole buffer).
*/
template <typename T>
void vector_assign(vector_base<T> & vec1, const T & alpha, bool up_to_internal_size = false)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
  viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "assign_cpu");
  k.global_work_size(0, std::min<vcl_size_t>(128 * k.local_work_size(),
                                              viennacl::tools::align_to_multiple<vcl_size_t>(viennacl::traits::size(vec1), k.local_work_size()) ) );

  cl_uint size = up_to_internal_size ? cl_uint(vec1.internal_size()) : cl_uint(viennacl::traits::size(vec1));
  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                           cl_uint(viennacl::traits::start(vec1)),
                           cl_uint(viennacl::traits::stride(vec1)),
                           size,
                           cl_uint(vec1.internal_size()),     //Note: Do NOT use traits::internal_size() here, because vector proxies don't require padding.
                           viennacl::traits::opencl_handle(T(alpha)) )
                        );
}


/** @brief Swaps the contents of two vectors, data is copied
*
* @param vec1   The first vector (or -range, or -slice)
* @param vec2   The second vector (or -range, or -slice)
*/
template <typename T>
void vector_swap(vector_base<T> & vec1, vector_base<T> & vec2)
{
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
  viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "swap");

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                           cl_uint(viennacl::traits::start(vec1)),
                           cl_uint(viennacl::traits::stride(vec1)),
                           cl_uint(viennacl::traits::size(vec1)),
                           viennacl::traits::opencl_handle(vec2),
                           cl_uint(viennacl::traits::start(vec2)),
                           cl_uint(viennacl::traits::stride(vec2)),
                           cl_uint(viennacl::traits::size(vec2)))
                        );
}

///////////////////////// Binary Elementwise operations /////////////

/** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
*
* @param vec1   The result vector (or -range, or -slice)
* @param proxy  The proxy object holding v2, v3 and the operation
*/
template <typename T, typename OP>
void element_op(vector_base<T> & vec1,
                vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> > const & proxy)
{
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
  viennacl::linalg::opencl::kernels::vector_element<T>::init(ctx);

  std::string kernel_name = "element_pow";
  cl_uint op_type = 2; //0: product, 1: division, 2: power
  if (viennacl::is_division<OP>::value)
  {
    op_type = 1;
    kernel_name = "element_div";
  }
  else if (viennacl::is_product<OP>::value)
  {
    op_type = 0;
    kernel_name = "element_prod";
  }

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_element<T>::program_name(), kernel_name);

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                           cl_uint(viennacl::traits::start(vec1)),
                           cl_uint(viennacl::traits::stride(vec1)),
                           cl_uint(viennacl::traits::size(vec1)),

                           viennacl::traits::opencl_handle(proxy.lhs()),
                           cl_uint(viennacl::traits::start(proxy.lhs())),
                           cl_uint(viennacl::traits::stride(proxy.lhs())),

                           viennacl::traits::opencl_handle(proxy.rhs()),
                           cl_uint(viennacl::traits::start(proxy.rhs())),
                           cl_uint(viennacl::traits::stride(proxy.rhs())),

                           op_type)
                        );
}

///////////////////////// Unary Elementwise operations /////////////

/** @brief Implementation of unary element-wise operations v1 = OP(v2)
*
* @param vec1   The result vector (or -range, or -slice)
* @param proxy  The proxy object holding v2 and the operation
*/
template <typename T, typename OP>
void element_op(vector_base<T> & vec1,
                vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<OP> > const & proxy)
{
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
  viennacl::linalg::opencl::kernels::vector_element<T>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_element<T>::program_name(), detail::op_to_string(OP()) + "_assign");

  viennacl::ocl::packed_cl_uint size_vec1;
  size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
  size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
  size_vec1.size   = cl_uint(viennacl::traits::size(vec1));
  size_vec1.internal_size   = cl_uint(viennacl::traits::internal_size(vec1));

  viennacl::ocl::packed_cl_uint size_vec2;
  size_vec2.start  = cl_uint(viennacl::traits::start(proxy.lhs()));
  size_vec2.stride = cl_uint(viennacl::traits::stride(proxy.lhs()));
  size_vec2.size   = cl_uint(viennacl::traits::size(proxy.lhs()));
  size_vec2.internal_size   = cl_uint(viennacl::traits::internal_size(proxy.lhs()));

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                           size_vec1,
                           viennacl::traits::opencl_handle(proxy.lhs()),
                           size_vec2)
                        );
}

///////////////////////// Norms and inner product ///////////////////

/** @brief Computes the partial inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
*
* @param vec1 The first vector
* @param vec2 The second vector
* @param partial_result The results of each group
*/
template <typename T>
void inner_prod_impl(vector_base<T> const & vec1,
                     vector_base<T> const & vec2,
                     vector_base<T> & partial_result)
{
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(vec2).context() == viennacl::traits::opencl_handle(partial_result).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
  viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

  assert( (viennacl::traits::size(vec1) == viennacl::traits::size(vec2))
        && bool("Incompatible vector sizes in inner_prod_impl()!"));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "inner_prod1");

  assert( (k.global_work_size() / k.local_work_size() <= partial_result.size()) && bool("Size mismatch for partial reduction in inner_prod_impl()") );

  viennacl::ocl::packed_cl_uint size_vec1;
  size_vec1.start  = cl_uint(viennacl::traits::start(vec1));
  size_vec1.stride = cl_uint(viennacl::traits::stride(vec1));
  size_vec1.size   = cl_uint(viennacl::traits::size(vec1));
  size_vec1.internal_size   = cl_uint(viennacl::traits::internal_size(vec1));

  viennacl::ocl::packed_cl_uint size_vec2;
  size_vec2.start  = cl_uint(viennacl::traits::start(vec2));
  size_vec2.stride = cl_uint(viennacl::traits::stride(vec2));
  size_vec2.size   = cl_uint(viennacl::traits::size(vec2));
  size_vec2.internal_size   = cl_uint(viennacl::traits::internal_size(vec2));

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                           size_vec1,
                           viennacl::traits::opencl_handle(vec2),
                           size_vec2,
                           viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * k.local_work_size()),
                           viennacl::traits::opencl_handle(partial_result)
                          )
                        );
}


//implementation of inner product:
//namespace {
/** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
*
* @param vec1 The first vector
* @param vec2 The second vector
* @param result The result scalar (on the gpu)
*/
template <typename T>
void inner_prod_impl(vector_base<T> const & vec1,
                     vector_base<T> const & vec2,
                     scalar<T> & result)
{
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());

  vcl_size_t work_groups = 128;
  viennacl::vector<T> temp(work_groups, viennacl::traits::context(vec1));
  temp.resize(work_groups, ctx); // bring default-constructed vectors to the correct size:

  // Step 1: Compute partial inner products for each work group:
  inner_prod_impl(vec1, vec2, temp);

  // Step 2: Sum partial results:
  viennacl::ocl::kernel & ksum = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "sum");

  ksum.global_work_size(0, ksum.local_work_size(0));
  viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                              cl_uint(viennacl::traits::start(temp)),
                              cl_uint(viennacl::traits::stride(temp)),
                              cl_uint(viennacl::traits::size(temp)),
                              cl_uint(1),
                              viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * ksum.local_work_size()),
                              viennacl::traits::opencl_handle(result) )
                        );
}

namespace detail
{
  template<typename NumericT>
  viennacl::ocl::packed_cl_uint make_layout(vector_base<NumericT> const & vec)
  {
    viennacl::ocl::packed_cl_uint ret;
    ret.start           = cl_uint(viennacl::traits::start(vec));
    ret.stride          = cl_uint(viennacl::traits::stride(vec));
    ret.size            = cl_uint(viennacl::traits::size(vec));
    ret.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
    return ret;
  }
}

/** @brief Computes multiple inner products where one argument is common to all inner products. <x, y1>, <x, y2>, ..., <x, yN>
*
* @param x          The common vector
* @param vec_tuple  The tuple of vectors y1, y2, ..., yN
* @param result     The result vector
*/
template <typename NumericT>
void inner_prod_impl(vector_base<NumericT> const & x,
                     vector_tuple<NumericT> const & vec_tuple,
                     vector_base<NumericT> & result)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(x).context());
  viennacl::linalg::opencl::kernels::vector<NumericT>::init(ctx);
  viennacl::linalg::opencl::kernels::vector_multi_inner_prod<NumericT>::init(ctx);

  viennacl::ocl::packed_cl_uint layout_x = detail::make_layout(x);

  viennacl::ocl::kernel & ksum = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_multi_inner_prod<NumericT>::program_name(), "sum_inner_prod");
  viennacl::ocl::kernel & inner_prod_kernel_1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<NumericT>::program_name(), "inner_prod1");
  viennacl::ocl::kernel & inner_prod_kernel_2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_multi_inner_prod<NumericT>::program_name(), "inner_prod2");
  viennacl::ocl::kernel & inner_prod_kernel_3 = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_multi_inner_prod<NumericT>::program_name(), "inner_prod3");
  viennacl::ocl::kernel & inner_prod_kernel_4 = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_multi_inner_prod<NumericT>::program_name(), "inner_prod4");
  viennacl::ocl::kernel & inner_prod_kernel_8 = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector_multi_inner_prod<NumericT>::program_name(), "inner_prod8");

  vcl_size_t work_groups = inner_prod_kernel_8.global_work_size(0) / inner_prod_kernel_8.local_work_size(0);
  viennacl::vector<NumericT> temp(8 * work_groups, viennacl::traits::context(x));

  vcl_size_t current_index = 0;
  while (current_index < vec_tuple.const_size())
  {
    switch (vec_tuple.const_size() - current_index)
    {
      case 7:
      case 6:
      case 5:
      case 4:
      {
        vector_base<NumericT> const & y0 = vec_tuple.const_at(current_index    );
        vector_base<NumericT> const & y1 = vec_tuple.const_at(current_index + 1);
        vector_base<NumericT> const & y2 = vec_tuple.const_at(current_index + 2);
        vector_base<NumericT> const & y3 = vec_tuple.const_at(current_index + 3);
        viennacl::ocl::enqueue(inner_prod_kernel_4( viennacl::traits::opencl_handle(x), layout_x,
                                                   viennacl::traits::opencl_handle(y0), detail::make_layout(y0),
                                                   viennacl::traits::opencl_handle(y1), detail::make_layout(y1),
                                                   viennacl::traits::opencl_handle(y2), detail::make_layout(y2),
                                                   viennacl::traits::opencl_handle(y3), detail::make_layout(y3),
                                                   viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * 4 * inner_prod_kernel_4.local_work_size()),
                                                   viennacl::traits::opencl_handle(temp)
                                                  ) );

        ksum.global_work_size(0, 4 * ksum.local_work_size(0));
        viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                    cl_uint(work_groups),
                                    viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * 4 * ksum.local_work_size()),
                                    viennacl::traits::opencl_handle(result),
                                    cl_uint(viennacl::traits::start(result) + current_index * viennacl::traits::stride(result)),
                                    cl_uint(viennacl::traits::stride(result))
                                    )
                              );
      }
        current_index += 4;
        break;

      case 3:
      {
        vector_base<NumericT> const & y0 = vec_tuple.const_at(current_index    );
        vector_base<NumericT> const & y1 = vec_tuple.const_at(current_index + 1);
        vector_base<NumericT> const & y2 = vec_tuple.const_at(current_index + 2);
        viennacl::ocl::enqueue(inner_prod_kernel_3( viennacl::traits::opencl_handle(x), layout_x,
                                                    viennacl::traits::opencl_handle(y0), detail::make_layout(y0),
                                                    viennacl::traits::opencl_handle(y1), detail::make_layout(y1),
                                                    viennacl::traits::opencl_handle(y2), detail::make_layout(y2),
                                                    viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * 3 * inner_prod_kernel_3.local_work_size()),
                                                    viennacl::traits::opencl_handle(temp)
                                                   ) );

        ksum.global_work_size(0, 3 * ksum.local_work_size(0));
        viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                    cl_uint(work_groups),
                                    viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * 3 * ksum.local_work_size()),
                                    viennacl::traits::opencl_handle(result),
                                    cl_uint(viennacl::traits::start(result) + current_index * viennacl::traits::stride(result)),
                                    cl_uint(viennacl::traits::stride(result))
                                    )
                              );
      }
        current_index += 3;
        break;

      case 2:
      {
        vector_base<NumericT> const & y0 = vec_tuple.const_at(current_index    );
        vector_base<NumericT> const & y1 = vec_tuple.const_at(current_index + 1);
        viennacl::ocl::enqueue(inner_prod_kernel_2( viennacl::traits::opencl_handle(x), layout_x,
                                                    viennacl::traits::opencl_handle(y0), detail::make_layout(y0),
                                                    viennacl::traits::opencl_handle(y1), detail::make_layout(y1),
                                                    viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * 2 * inner_prod_kernel_2.local_work_size()),
                                                    viennacl::traits::opencl_handle(temp)
                                                  ) );

        ksum.global_work_size(0, 2 * ksum.local_work_size(0));
        viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                    cl_uint(work_groups),
                                    viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * 2 * ksum.local_work_size()),
                                    viennacl::traits::opencl_handle(result),
                                    cl_uint(viennacl::traits::start(result) + current_index * viennacl::traits::stride(result)),
                                    cl_uint(viennacl::traits::stride(result))
                                    )
                              );
      }
        current_index += 2;
        break;

      case 1:
      {
        vector_base<NumericT> const & y0 = vec_tuple.const_at(current_index    );
        viennacl::ocl::enqueue(inner_prod_kernel_1( viennacl::traits::opencl_handle(x), layout_x,
                                                    viennacl::traits::opencl_handle(y0), detail::make_layout(y0),
                                                    viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * 1 * inner_prod_kernel_1.local_work_size()),
                                                    viennacl::traits::opencl_handle(temp)
                                                  ) );

        ksum.global_work_size(0, 1 * ksum.local_work_size(0));
        viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                    cl_uint(work_groups),
                                    viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * 1 * ksum.local_work_size()),
                                    viennacl::traits::opencl_handle(result),
                                    cl_uint(viennacl::traits::start(result) + current_index * viennacl::traits::stride(result)),
                                    cl_uint(viennacl::traits::stride(result))
                                    )
                              );
      }
        current_index += 1;
        break;

      default: //8 or more vectors
      {
        vector_base<NumericT> const & y0 = vec_tuple.const_at(current_index    );
        vector_base<NumericT> const & y1 = vec_tuple.const_at(current_index + 1);
        vector_base<NumericT> const & y2 = vec_tuple.const_at(current_index + 2);
        vector_base<NumericT> const & y3 = vec_tuple.const_at(current_index + 3);
        vector_base<NumericT> const & y4 = vec_tuple.const_at(current_index + 4);
        vector_base<NumericT> const & y5 = vec_tuple.const_at(current_index + 5);
        vector_base<NumericT> const & y6 = vec_tuple.const_at(current_index + 6);
        vector_base<NumericT> const & y7 = vec_tuple.const_at(current_index + 7);
        viennacl::ocl::enqueue(inner_prod_kernel_8( viennacl::traits::opencl_handle(x), layout_x,
                                                    viennacl::traits::opencl_handle(y0), detail::make_layout(y0),
                                                    viennacl::traits::opencl_handle(y1), detail::make_layout(y1),
                                                    viennacl::traits::opencl_handle(y2), detail::make_layout(y2),
                                                    viennacl::traits::opencl_handle(y3), detail::make_layout(y3),
                                                    viennacl::traits::opencl_handle(y4), detail::make_layout(y4),
                                                    viennacl::traits::opencl_handle(y5), detail::make_layout(y5),
                                                    viennacl::traits::opencl_handle(y6), detail::make_layout(y6),
                                                    viennacl::traits::opencl_handle(y7), detail::make_layout(y7),
                                                    viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * 8 * inner_prod_kernel_8.local_work_size()),
                                                    viennacl::traits::opencl_handle(temp)
                                                  ) );

        ksum.global_work_size(0, 8 * ksum.local_work_size(0));
        viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                                    cl_uint(work_groups),
                                    viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * 8 * ksum.local_work_size()),
                                    viennacl::traits::opencl_handle(result),
                                    cl_uint(viennacl::traits::start(result) + current_index * viennacl::traits::stride(result)),
                                    cl_uint(viennacl::traits::stride(result))
                                    )
                              );
      }
        current_index += 8;
        break;
    }
  }

}



//implementation of inner product:
//namespace {
/** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
*
* @param vec1 The first vector
* @param vec2 The second vector
* @param result The result scalar (on the gpu)
*/
template <typename T>
void inner_prod_cpu(vector_base<T> const & vec1,
                    vector_base<T> const & vec2,
                    T & result)
{
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Vectors do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());

  vcl_size_t work_groups = 128;
  viennacl::vector<T> temp(work_groups, viennacl::traits::context(vec1));
  temp.resize(work_groups, ctx); // bring default-constructed vectors to the correct size:

  // Step 1: Compute partial inner products for each work group:
  inner_prod_impl(vec1, vec2, temp);

  // Step 2: Sum partial results:

  // Now copy partial results from GPU back to CPU and run reduction there:
  std::vector<T> temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = 0;
  for (typename std::vector<T>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result += *it;
}


//////////// Helper for norms

/** @brief Computes the partial work group results for vector norms
*
* @param vec The vector
* @param partial_result The result scalar
* @param norm_id        Norm selector. 0: norm_inf, 1: norm_1, 2: norm_2
*/
template <typename T>
void norm_reduction_impl(vector_base<T> const & vec,
                         vector_base<T> & partial_result,
                          cl_uint norm_id)
{
  assert(viennacl::traits::opencl_handle(vec).context() == viennacl::traits::opencl_handle(partial_result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());
  viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "norm");

  assert( (k.global_work_size() / k.local_work_size() <= partial_result.size()) && bool("Size mismatch for partial reduction in norm_reduction_impl()") );

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec),
                           cl_uint(viennacl::traits::start(vec)),
                           cl_uint(viennacl::traits::stride(vec)),
                           cl_uint(viennacl::traits::size(vec)),
                           cl_uint(norm_id),
                           viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * k.local_work_size()),
                           viennacl::traits::opencl_handle(partial_result) )
                        );
}


//////////// Norm 1

/** @brief Computes the l^1-norm of a vector
*
* @param vec The vector
* @param result The result scalar
*/
template <typename T>
void norm_1_impl(vector_base<T> const & vec,
                 scalar<T> & result)
{
  assert(viennacl::traits::opencl_handle(vec).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());

  vcl_size_t work_groups = 128;
  viennacl::vector<T> temp(work_groups, viennacl::traits::context(vec));

  // Step 1: Compute the partial work group results
  norm_reduction_impl(vec, temp, 1);

  // Step 2: Compute the partial reduction using OpenCL
  viennacl::ocl::kernel & ksum = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "sum");

  ksum.global_work_size(0, ksum.local_work_size(0));
  viennacl::ocl::enqueue(ksum(viennacl::traits::opencl_handle(temp),
                              cl_uint(viennacl::traits::start(temp)),
                              cl_uint(viennacl::traits::stride(temp)),
                              cl_uint(viennacl::traits::size(temp)),
                              cl_uint(1),
                              viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * ksum.local_work_size()),
                              result)
                        );
}

/** @brief Computes the l^1-norm of a vector with final reduction on CPU
*
* @param vec The vector
* @param result The result scalar
*/
template <typename T>
void norm_1_cpu(vector_base<T> const & vec,
                T & result)
{
  vcl_size_t work_groups = 128;
  viennacl::vector<T> temp(work_groups, viennacl::traits::context(vec));

  // Step 1: Compute the partial work group results
  norm_reduction_impl(vec, temp, 1);

  // Step 2: Now copy partial results from GPU back to CPU and run reduction there:
  typedef std::vector<typename viennacl::result_of::cl_type<T>::type>  CPUVectorType;

  CPUVectorType temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = 0;
  for (typename CPUVectorType::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result += static_cast<T>(*it);
}



//////// Norm 2


/** @brief Computes the l^2-norm of a vector - implementation using OpenCL summation at second step
*
* @param vec The vector
* @param result The result scalar
*/
template <typename T>
void norm_2_impl(vector_base<T> const & vec,
                 scalar<T> & result)
{
  assert(viennacl::traits::opencl_handle(vec).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());

  vcl_size_t work_groups = 128;
  viennacl::vector<T> temp(work_groups, viennacl::traits::context(vec));

  // Step 1: Compute the partial work group results
  norm_reduction_impl(vec, temp, 2);

  // Step 2: Reduction via OpenCL
  viennacl::ocl::kernel & ksum = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "sum");

  ksum.global_work_size(0, ksum.local_work_size(0));
  viennacl::ocl::enqueue( ksum(viennacl::traits::opencl_handle(temp),
                                cl_uint(viennacl::traits::start(temp)),
                                cl_uint(viennacl::traits::stride(temp)),
                                cl_uint(viennacl::traits::size(temp)),
                                cl_uint(2),
                                viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * ksum.local_work_size()),
                                result)
                        );
}

/** @brief Computes the l^1-norm of a vector with final reduction on CPU
*
* @param vec The vector
* @param result The result scalar
*/
template <typename T>
void norm_2_cpu(vector_base<T> const & vec,
                T & result)
{
  vcl_size_t work_groups = 128;
  viennacl::vector<T> temp(work_groups, viennacl::traits::context(vec));

  // Step 1: Compute the partial work group results
  norm_reduction_impl(vec, temp, 2);

  // Step 2: Now copy partial results from GPU back to CPU and run reduction there:
  typedef std::vector<typename viennacl::result_of::cl_type<T>::type>  CPUVectorType;

  CPUVectorType temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = 0;
  for (typename CPUVectorType::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result += static_cast<T>(*it);
  result = std::sqrt(result);
}



////////// Norm inf

/** @brief Computes the supremum-norm of a vector
*
* @param vec The vector
* @param result The result scalar
*/
template <typename T>
void norm_inf_impl(vector_base<T> const & vec,
                   scalar<T> & result)
{
  assert(viennacl::traits::opencl_handle(vec).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());

  vcl_size_t work_groups = 128;
  viennacl::vector<T> temp(work_groups, viennacl::traits::context(vec));

  // Step 1: Compute the partial work group results
  norm_reduction_impl(vec, temp, 0);

  //part 2: parallel reduction of reduced kernel:
  viennacl::ocl::kernel & ksum = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "sum");

  ksum.global_work_size(0, ksum.local_work_size(0));
  viennacl::ocl::enqueue( ksum(viennacl::traits::opencl_handle(temp),
                               cl_uint(viennacl::traits::start(temp)),
                               cl_uint(viennacl::traits::stride(temp)),
                               cl_uint(viennacl::traits::size(temp)),
                               cl_uint(0),
                               viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * ksum.local_work_size()),
                               result)
                        );
}

/** @brief Computes the supremum-norm of a vector
*
* @param vec The vector
* @param result The result scalar
*/
template <typename T>
void norm_inf_cpu(vector_base<T> const & vec,
                  T & result)
{
  vcl_size_t work_groups = 128;
  viennacl::vector<T> temp(work_groups, viennacl::traits::context(vec));

  // Step 1: Compute the partial work group results
  norm_reduction_impl(vec, temp, 0);

  // Step 2: Now copy partial results from GPU back to CPU and run reduction there:
  typedef std::vector<typename viennacl::result_of::cl_type<T>::type>  CPUVectorType;

  CPUVectorType temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = 0;
  for (typename CPUVectorType::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result = std::max(result, static_cast<T>(*it));
}


/////////// index norm_inf

//This function should return a CPU scalar, otherwise statements like
// vcl_rhs[index_norm_inf(vcl_rhs)]
// are ambiguous
/** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
*
* @param vec The vector
* @return The result. Note that the result must be a CPU scalar (unsigned int), since gpu scalars are floating point types.
*/
template <typename T>
cl_uint index_norm_inf(vector_base<T> const & vec)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec).context());
  viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

  viennacl::ocl::handle<cl_mem> h = ctx.create_memory(CL_MEM_READ_WRITE, sizeof(cl_uint));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "index_norm_inf");
  //cl_uint size = static_cast<cl_uint>(vcl_vec.internal_size());

  //TODO: Use multi-group kernel for large vector sizes

  k.global_work_size(0, k.local_work_size());
  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec),
                           cl_uint(viennacl::traits::start(vec)),
                           cl_uint(viennacl::traits::stride(vec)),
                           cl_uint(viennacl::traits::size(vec)),
                           viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<T>::type) * k.local_work_size()),
                           viennacl::ocl::local_mem(sizeof(cl_uint) * k.local_work_size()), h));

  //read value:
  cl_uint result;
  cl_int err = clEnqueueReadBuffer(ctx.get_queue().handle().get(), h.get(), CL_TRUE, 0, sizeof(cl_uint), &result, 0, NULL, NULL);
  VIENNACL_ERR_CHECK(err);
  return result;
}


////////// max

/** @brief Computes the maximum value of a vector, where the result is stored in an OpenCL buffer.
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void max_impl(vector_base<NumericT> const & x,
                   scalar<NumericT> & result)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(x).context());
  viennacl::linalg::opencl::kernels::vector<NumericT>::init(ctx);

  vcl_size_t work_groups = 128;
  viennacl::vector<NumericT> temp(work_groups, viennacl::traits::context(x));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<NumericT>::program_name(), "max_kernel");

  k.global_work_size(0, work_groups * k.local_work_size(0));
  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(x),
                           cl_uint(viennacl::traits::start(x)),
                           cl_uint(viennacl::traits::stride(x)),
                           cl_uint(viennacl::traits::size(x)),
                           viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * k.local_work_size()),
                           viennacl::traits::opencl_handle(temp)
                         ));

  k.global_work_size(0, k.local_work_size());
  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(temp),
                           cl_uint(viennacl::traits::start(temp)),
                           cl_uint(viennacl::traits::stride(temp)),
                           cl_uint(viennacl::traits::size(temp)),
                           viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * k.local_work_size()),
                           viennacl::traits::opencl_handle(result)
                         ));
}

/** @brief Computes the maximum value of a vector, where the value is stored in a host value.
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void max_cpu(vector_base<NumericT> const & x,
             NumericT & result)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(x).context());
  viennacl::linalg::opencl::kernels::vector<NumericT>::init(ctx);

  vcl_size_t work_groups = 128;
  viennacl::vector<NumericT> temp(work_groups, viennacl::traits::context(x));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<NumericT>::program_name(), "max_kernel");

  k.global_work_size(0, work_groups * k.local_work_size(0));
  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(x),
                           cl_uint(viennacl::traits::start(x)),
                           cl_uint(viennacl::traits::stride(x)),
                           cl_uint(viennacl::traits::size(x)),
                           viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * k.local_work_size()),
                           viennacl::traits::opencl_handle(temp)
                         ));

  // Step 2: Now copy partial results from GPU back to CPU and run reduction there:
  typedef std::vector<typename viennacl::result_of::cl_type<NumericT>::type>  CPUVectorType;

  CPUVectorType temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = static_cast<NumericT>(temp_cpu[0]);
  for (typename CPUVectorType::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result = std::max(result, static_cast<NumericT>(*it));

}


////////// min

/** @brief Computes the minimum of a vector, where the result is stored in an OpenCL buffer.
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void min_impl(vector_base<NumericT> const & x,
                   scalar<NumericT> & result)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(x).context());
  viennacl::linalg::opencl::kernels::vector<NumericT>::init(ctx);

  vcl_size_t work_groups = 128;
  viennacl::vector<NumericT> temp(work_groups, viennacl::traits::context(x));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<NumericT>::program_name(), "min_kernel");

  k.global_work_size(0, work_groups * k.local_work_size(0));
  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(x),
                           cl_uint(viennacl::traits::start(x)),
                           cl_uint(viennacl::traits::stride(x)),
                           cl_uint(viennacl::traits::size(x)),
                           viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * k.local_work_size()),
                           viennacl::traits::opencl_handle(temp)
                         ));

  k.global_work_size(0, k.local_work_size());
  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(temp),
                           cl_uint(viennacl::traits::start(temp)),
                           cl_uint(viennacl::traits::stride(temp)),
                           cl_uint(viennacl::traits::size(temp)),
                           viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * k.local_work_size()),
                           viennacl::traits::opencl_handle(result)
                         ));
}

/** @brief Computes the minimum of a vector, where the result is stored on a CPU scalar.
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void min_cpu(vector_base<NumericT> const & x,
                  NumericT & result)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(x).context());
  viennacl::linalg::opencl::kernels::vector<NumericT>::init(ctx);

  vcl_size_t work_groups = 128;
  viennacl::vector<NumericT> temp(work_groups, viennacl::traits::context(x));

  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<NumericT>::program_name(), "min_kernel");

  k.global_work_size(0, work_groups * k.local_work_size(0));
  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(x),
                           cl_uint(viennacl::traits::start(x)),
                           cl_uint(viennacl::traits::stride(x)),
                           cl_uint(viennacl::traits::size(x)),
                           viennacl::ocl::local_mem(sizeof(typename viennacl::result_of::cl_type<NumericT>::type) * k.local_work_size()),
                           viennacl::traits::opencl_handle(temp)
                         ));

  // Step 2: Now copy partial results from GPU back to CPU and run reduction there:
  typedef std::vector<typename viennacl::result_of::cl_type<NumericT>::type>  CPUVectorType;

  CPUVectorType temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = static_cast<NumericT>(temp_cpu[0]);
  for (typename CPUVectorType::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result = std::min(result, static_cast<NumericT>(*it));
}

////////// sum

/** @brief Computes the sum over all entries of a vector
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void sum_impl(vector_base<NumericT> const & x,
                   scalar<NumericT> & result)
{
  assert(viennacl::traits::opencl_handle(x).context() == viennacl::traits::opencl_handle(result).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::vector<NumericT> all_ones = viennacl::scalar_vector<NumericT>(x.size(), NumericT(1), viennacl::traits::context(x));
  viennacl::linalg::opencl::inner_prod_impl(x, all_ones, result);
}

/** @brief Computes the sum over all entries of a vector.
*
* @param x      The vector
* @param result The result scalar
*/
template<typename NumericT>
void sum_cpu(vector_base<NumericT> const & x, NumericT & result)
{
  scalar<NumericT> tmp(0, viennacl::traits::context(x));
  sum_impl(x, tmp);
  result = tmp;
}


//TODO: Special case vec1 == vec2 allows improvement!!
/** @brief Computes a plane rotation of two vectors.
*
* Computes (x,y) <- (alpha * x + beta * y, -beta * x + alpha * y)
*
* @param vec1   The first vector
* @param vec2   The second vector
* @param alpha  The first transformation coefficient
* @param beta   The second transformation coefficient
*/
template <typename T>
void plane_rotation(vector_base<T> & vec1,
                    vector_base<T> & vec2,
                    T alpha, T beta)
{
  assert(viennacl::traits::opencl_handle(vec1).context() == viennacl::traits::opencl_handle(vec2).context() && bool("Operands do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(vec1).context());
  viennacl::linalg::opencl::kernels::vector<T>::init(ctx);

  assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2));
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::vector<T>::program_name(), "plane_rotation");

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(vec1),
                           cl_uint(viennacl::traits::start(vec1)),
                           cl_uint(viennacl::traits::stride(vec1)),
                           cl_uint(viennacl::traits::size(vec1)),
                           viennacl::traits::opencl_handle(vec2),
                           cl_uint(viennacl::traits::start(vec2)),
                           cl_uint(viennacl::traits::stride(vec2)),
                           cl_uint(viennacl::traits::size(vec2)),
                           viennacl::traits::opencl_handle(alpha),
                           viennacl::traits::opencl_handle(beta))
                        );
}


//////////////////////////


namespace detail
{
  /** @brief Worker routine for scan routines using OpenCL
   *
   * Note on performance: For non-in-place scans one could optimize away the temporary 'opencl_carries'-array.
   * This, however, only provides small savings in the latency-dominated regime, yet would effectively double the amount of code to maintain.
   */
  template<typename NumericT>
  void scan_impl(vector_base<NumericT> const & input,
                 vector_base<NumericT>       & output,
                 bool is_inclusive)
  {
    vcl_size_t local_worksize = 128;
    vcl_size_t workgroups = 128;

    viennacl::backend::mem_handle opencl_carries;
    viennacl::backend::memory_create(opencl_carries, sizeof(NumericT)*workgroups, viennacl::traits::context(input));

    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input).context());
    viennacl::linalg::opencl::kernels::scan<NumericT>::init(ctx);
    viennacl::ocl::kernel& k1 = ctx.get_kernel(viennacl::linalg::opencl::kernels::scan<NumericT>::program_name(), "scan_1");
    viennacl::ocl::kernel& k2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::scan<NumericT>::program_name(), "scan_2");
    viennacl::ocl::kernel& k3 = ctx.get_kernel(viennacl::linalg::opencl::kernels::scan<NumericT>::program_name(), "scan_3");

    // First step: Scan within each thread group and write carries
    k1.local_work_size(0, local_worksize);
    k1.global_work_size(0, workgroups * local_worksize);
    viennacl::ocl::enqueue(k1( input, cl_uint( input.start()), cl_uint( input.stride()), cl_uint(input.size()),
                              output, cl_uint(output.start()), cl_uint(output.stride()),
                              cl_uint(is_inclusive ? 0 : 1), opencl_carries.opencl_handle())
                          );

    // Second step: Compute offset for each thread group (exclusive scan for each thread group)
    k2.local_work_size(0, workgroups);
    k2.global_work_size(0, workgroups);
    viennacl::ocl::enqueue(k2(opencl_carries.opencl_handle()));

    // Third step: Offset each thread group accordingly
    k3.local_work_size(0, local_worksize);
    k3.global_work_size(0, workgroups * local_worksize);
    viennacl::ocl::enqueue(k3(output, cl_uint(output.start()), cl_uint(output.stride()), cl_uint(output.size()),
                              opencl_carries.opencl_handle())
                          );
  }
}


/** @brief This function implements an inclusive scan using CUDA.
*
* @param input       Input vector.
* @param output      The output vector. Either idential to input or non-overlapping.
*/
template<typename NumericT>
void inclusive_scan(vector_base<NumericT> const & input,
                    vector_base<NumericT>       & output)
{
  detail::scan_impl(input, output, true);
}


/** @brief This function implements an exclusive scan using CUDA.
*
* @param input       Input vector
* @param output      The output vector. Either idential to input or non-overlapping.
*/
template<typename NumericT>
void exclusive_scan(vector_base<NumericT> const & input,
                    vector_base<NumericT>       & output)
{
  detail::scan_impl(input, output, false);
}


} //namespace opencl
} //namespace linalg
} //namespace viennacl


#endif
