#ifndef VIENNACL_LINALG_OPENCL_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_MATRIX_OPERATIONS_HPP_

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

/** @file  viennacl/linalg/opencl/matrix_operations.hpp
    @brief Implementations of dense matrix related operations, including matrix-vector products, using OpenCL.
*/

#include "viennacl/forwards.h"

#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/result_of.hpp"

#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"

#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/linalg/opencl/kernels/svd.hpp"
#include "viennacl/linalg/opencl/kernels/vector.hpp"
#include "viennacl/linalg/opencl/kernels/matrix.hpp"
#include "viennacl/linalg/opencl/kernels/matrix_element.hpp"

namespace viennacl
{
namespace linalg
{
namespace opencl
{

namespace detail
{

  template<typename NumericT>
  viennacl::ocl::kernel & kernel_for_matrix(matrix_base<NumericT> const & M, std::string const & kernel_name)
  {
    viennacl::ocl::context & ctx = traits::opencl_context(M);
    viennacl::ocl::program * program;
    if (M.row_major())
    {
      typedef viennacl::linalg::opencl::kernels::matrix<NumericT, row_major>  KernelClass;
      KernelClass::init(ctx);
      program = &ctx.get_program(KernelClass::program_name());
    }
    else
    {
      typedef viennacl::linalg::opencl::kernels::matrix<NumericT, column_major>  KernelClass;
      KernelClass::init(ctx);
      program = &ctx.get_program(KernelClass::program_name());
    }
    return program->get_kernel(kernel_name);
  }

  template<typename NumericT>
  viennacl::ocl::kernel & element_kernel_for_matrix(matrix_base<NumericT> const & M, std::string const & kernel_name)
  {
    viennacl::ocl::context & ctx = traits::opencl_context(M);
    viennacl::ocl::program * program;
    if (M.row_major())
    {
      typedef viennacl::linalg::opencl::kernels::matrix_element<NumericT, row_major>  KernelClass;
      KernelClass::init(ctx);
      program = &ctx.get_program(KernelClass::program_name());
    }
    else
    {
      typedef viennacl::linalg::opencl::kernels::matrix_element<NumericT, column_major>  KernelClass;
      KernelClass::init(ctx);
      program = &ctx.get_program(KernelClass::program_name());
    }
    return program->get_kernel(kernel_name);
  }

  template<typename NumericT>
  viennacl::ocl::kernel & legacy_kernel_for_matrix(matrix_base<NumericT> const & M, std::string const & kernel_name)
  {
    viennacl::ocl::context & ctx = traits::opencl_context(M);
    viennacl::ocl::program * program;
    if (M.row_major())
    {
      typedef viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, row_major>  KernelClass;
      KernelClass::init(ctx);
      program = &ctx.get_program(KernelClass::program_name());
    }
    else
    {
      typedef viennacl::linalg::opencl::kernels::matrix_legacy<NumericT, column_major>  KernelClass;
      KernelClass::init(ctx);
      program = &ctx.get_program(KernelClass::program_name());
    }
    return program->get_kernel(kernel_name);
  }

}

//
// Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
//

const std::string SVD_BIDIAG_PACK_KERNEL = "bidiag_pack";
const std::string SVD_HOUSEHOLDER_UPDATE_A_LEFT_KERNEL = "house_update_A_left";
const std::string SVD_HOUSEHOLDER_UPDATE_A_RIGHT_KERNEL = "house_update_A_right";
const std::string SVD_HOUSEHOLDER_UPDATE_QL_KERNEL = "house_update_QL";
const std::string SVD_GIVENS_NEXT_KERNEL = "givens_next";
const std::string SVD_COPY_COL_KERNEL = "copy_col";
const std::string SVD_COPY_ROW_KERNEL = "copy_row";

template<typename DestNumericT, typename SrcNumericT>
void convert(matrix_base<DestNumericT> & dest, matrix_base<SrcNumericT> const & src)
{
  assert(dest.row_major() == src.row_major() && bool("Addition/subtraction on mixed matrix layouts not supported yet!"));

  assert(viennacl::traits::opencl_handle(dest).context() == viennacl::traits::opencl_handle(src).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  std::string kernel_name("convert_");
  kernel_name += dest.row_major() ? "row_" : "col_";
  kernel_name += viennacl::ocl::type_to_string<DestNumericT>::apply();
  kernel_name += "_";
  kernel_name += viennacl::ocl::type_to_string<SrcNumericT>::apply();

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(dest).context());
  viennacl::linalg::opencl::kernels::matrix_convert::init(ctx);
  viennacl::ocl::kernel& k = ctx.get_kernel(viennacl::linalg::opencl::kernels::matrix_convert::program_name(), kernel_name);

  viennacl::ocl::enqueue(k( dest, cl_uint(dest.start1()), cl_uint(dest.stride1()), cl_uint(dest.size1()), cl_uint(dest.internal_size1()), cl_uint(dest.start2()), cl_uint(dest.stride2()), cl_uint(dest.size2()), cl_uint(dest.internal_size2()),
                            src,  cl_uint( src.start1()), cl_uint( src.stride1()), cl_uint( src.size1()), cl_uint( src.internal_size1()), cl_uint( src.start2()), cl_uint( src.stride2()), cl_uint( src.size2()), cl_uint( src.internal_size2())
                        ) );
}

//
// Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
//

template <typename NumericT,
          typename ScalarT1>
void am(matrix_base<NumericT> & mat1,
        matrix_base<NumericT> const & mat2, ScalarT1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
  viennacl::ocl::kernel & k= detail::kernel_for_matrix(mat1, (viennacl::is_cpu_scalar<ScalarT1>::value ? "am_cpu" : "am_gpu"));

  cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat1),
                          cl_uint(viennacl::traits::start1(mat1)),           cl_uint(viennacl::traits::start2(mat1)),
                          cl_uint(viennacl::traits::stride1(mat1)),          cl_uint(viennacl::traits::stride2(mat1)),
                          cl_uint(viennacl::traits::size1(mat1)),            cl_uint(viennacl::traits::size2(mat1)),
                          cl_uint(viennacl::traits::internal_size1(mat1)),   cl_uint(viennacl::traits::internal_size2(mat1)),

                          viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<NumericT>(alpha)),
                          options_alpha,
                          viennacl::traits::opencl_handle(mat2),
                          cl_uint(viennacl::traits::start1(mat2)),           cl_uint(viennacl::traits::start2(mat2)),
                          cl_uint(viennacl::traits::stride1(mat2)),          cl_uint(viennacl::traits::stride2(mat2)),
                          cl_uint(viennacl::traits::internal_size1(mat2)),   cl_uint(viennacl::traits::internal_size2(mat2))
                          )
                        );
}


template <typename NumericT,
          typename ScalarT1, typename ScalarT2>
void ambm(matrix_base<NumericT> & mat1,
          matrix_base<NumericT> const & mat2, ScalarT1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          matrix_base<NumericT> const & mat3, ScalarT2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  std::string kernel_name;
  if      ( viennacl::is_cpu_scalar<ScalarT1>::value &&  viennacl::is_cpu_scalar<ScalarT2>::value)
    kernel_name = "ambm_cpu_cpu";
  else if ( viennacl::is_cpu_scalar<ScalarT1>::value && !viennacl::is_cpu_scalar<ScalarT2>::value)
    kernel_name = "ambm_cpu_gpu";
  else if (!viennacl::is_cpu_scalar<ScalarT1>::value &&  viennacl::is_cpu_scalar<ScalarT2>::value)
    kernel_name = "ambm_gpu_cpu";
  else
    kernel_name = "ambm_gpu_gpu";

  viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat1, kernel_name);

  cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
  cl_uint options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat1),
                          cl_uint(viennacl::traits::start1(mat1)),           cl_uint(viennacl::traits::start2(mat1)),
                          cl_uint(viennacl::traits::stride1(mat1)),          cl_uint(viennacl::traits::stride2(mat1)),
                          cl_uint(viennacl::traits::size1(mat1)),            cl_uint(viennacl::traits::size2(mat1)),
                          cl_uint(viennacl::traits::internal_size1(mat1)),   cl_uint(viennacl::traits::internal_size2(mat1)),

                          viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<NumericT>(alpha)),
                          options_alpha,
                          viennacl::traits::opencl_handle(mat2),
                          cl_uint(viennacl::traits::start1(mat2)),           cl_uint(viennacl::traits::start2(mat2)),
                          cl_uint(viennacl::traits::stride1(mat2)),          cl_uint(viennacl::traits::stride2(mat2)),
                          cl_uint(viennacl::traits::internal_size1(mat2)),   cl_uint(viennacl::traits::internal_size2(mat2)),

                          viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<NumericT>(beta)),
                          options_beta,
                          viennacl::traits::opencl_handle(mat3),
                          cl_uint(viennacl::traits::start1(mat3)),           cl_uint(viennacl::traits::start2(mat3)),
                          cl_uint(viennacl::traits::stride1(mat3)),          cl_uint(viennacl::traits::stride2(mat3)),
                          cl_uint(viennacl::traits::internal_size1(mat3)),   cl_uint(viennacl::traits::internal_size2(mat3))
                          )
                        );
}


template <typename NumericT,
          typename ScalarT1, typename ScalarT2>
void ambm_m(matrix_base<NumericT> & mat1,
            matrix_base<NumericT> const & mat2, ScalarT1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
            matrix_base<NumericT> const & mat3, ScalarT2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  std::string kernel_name;
  if      ( viennacl::is_cpu_scalar<ScalarT1>::value &&  viennacl::is_cpu_scalar<ScalarT2>::value)
    kernel_name = "ambm_m_cpu_cpu";
  else if ( viennacl::is_cpu_scalar<ScalarT1>::value && !viennacl::is_cpu_scalar<ScalarT2>::value)
    kernel_name = "ambm_m_cpu_gpu";
  else if (!viennacl::is_cpu_scalar<ScalarT1>::value &&  viennacl::is_cpu_scalar<ScalarT2>::value)
    kernel_name = "ambm_m_gpu_cpu";
  else
    kernel_name = "ambm_m_gpu_gpu";

  viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat1, kernel_name);

  cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
  cl_uint options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat1),
                          cl_uint(viennacl::traits::start1(mat1)),           cl_uint(viennacl::traits::start2(mat1)),
                          cl_uint(viennacl::traits::stride1(mat1)),          cl_uint(viennacl::traits::stride2(mat1)),
                          cl_uint(viennacl::traits::size1(mat1)),            cl_uint(viennacl::traits::size2(mat1)),
                          cl_uint(viennacl::traits::internal_size1(mat1)),   cl_uint(viennacl::traits::internal_size2(mat1)),

                          viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<NumericT>(alpha)),
                          options_alpha,
                          viennacl::traits::opencl_handle(mat2),
                          cl_uint(viennacl::traits::start1(mat2)),           cl_uint(viennacl::traits::start2(mat2)),
                          cl_uint(viennacl::traits::stride1(mat2)),          cl_uint(viennacl::traits::stride2(mat2)),
                          cl_uint(viennacl::traits::internal_size1(mat2)),   cl_uint(viennacl::traits::internal_size2(mat2)),

                          viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<NumericT>(beta)),
                          options_beta,
                          viennacl::traits::opencl_handle(mat3),
                          cl_uint(viennacl::traits::start1(mat3)),           cl_uint(viennacl::traits::start2(mat3)),
                          cl_uint(viennacl::traits::stride1(mat3)),          cl_uint(viennacl::traits::stride2(mat3)),
                          cl_uint(viennacl::traits::internal_size1(mat3)),   cl_uint(viennacl::traits::internal_size2(mat3))
                          )
                        );
}

template<typename NumericT,
          typename SizeT, typename DistanceT>
void trans(const matrix_expression<const matrix_base<NumericT, SizeT, DistanceT>,const matrix_base<NumericT, SizeT, DistanceT>, op_trans> & proxy,
           matrix_base<NumericT> & temp_trans)
{
  std::string kernel_name("trans_kernel");
  viennacl::ocl::kernel& kernel = detail::legacy_kernel_for_matrix(proxy.lhs(),kernel_name);
  viennacl::ocl::enqueue(kernel(proxy.lhs(),
                                static_cast<cl_uint>(proxy.lhs().start1()),         static_cast<cl_uint>(proxy.lhs().start2()),
                                static_cast<cl_uint>(proxy.lhs().internal_size1()), static_cast<cl_uint>(proxy.lhs().internal_size2()),
                                static_cast<cl_uint>(proxy.lhs().size1()),          static_cast<cl_uint>(proxy.lhs().size2()),
                                static_cast<cl_uint>(proxy.lhs().stride1()),        static_cast<cl_uint>(proxy.lhs().stride2()),

                                temp_trans,
                                static_cast<cl_uint>(temp_trans.start1()),         static_cast<cl_uint>(temp_trans.start2()),
                                static_cast<cl_uint>(temp_trans.internal_size1()), static_cast<cl_uint>(temp_trans.internal_size2()),
                                static_cast<cl_uint>(temp_trans.stride1()),        static_cast<cl_uint>(temp_trans.stride2())));
}

template <typename NumericT>
void matrix_assign(matrix_base<NumericT> & mat, NumericT s, bool clear = false)
{
  cl_uint s1 = clear ? cl_uint(viennacl::traits::internal_size1(mat)) : cl_uint(viennacl::traits::size1(mat));
  cl_uint s2 = clear ? cl_uint(viennacl::traits::internal_size2(mat)) : cl_uint(viennacl::traits::size2(mat));

  viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat, "assign_cpu");
  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                           cl_uint(viennacl::traits::start1(mat)),           cl_uint(viennacl::traits::start2(mat)),
                           cl_uint(viennacl::traits::stride1(mat)),          cl_uint(viennacl::traits::stride2(mat)),
                           s1,                                               s2,
                           cl_uint(viennacl::traits::internal_size1(mat)),   cl_uint(viennacl::traits::internal_size2(mat)),
                           viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<NumericT>(s))
                          )
                        );
}

template <typename NumericT>
void matrix_diagonal_assign(matrix_base<NumericT> & mat, NumericT s)
{
  viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat, "diagonal_assign_cpu");
  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                           cl_uint(viennacl::traits::start1(mat)),           cl_uint(viennacl::traits::start2(mat)),
                           cl_uint(viennacl::traits::stride1(mat)),          cl_uint(viennacl::traits::stride2(mat)),
                           cl_uint(viennacl::traits::size1(mat)),            cl_uint(viennacl::traits::size2(mat)),
                           cl_uint(viennacl::traits::internal_size1(mat)),   cl_uint(viennacl::traits::internal_size2(mat)),
                           viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<NumericT>(s))
                          )
                        );
}

template <typename NumericT>
void matrix_diag_from_vector(const vector_base<NumericT> & vec, int k, matrix_base<NumericT> & mat)
{
  // Step 1: set everything to zero
  matrix_assign(mat, NumericT(0));

  // Step 2: set the diagonal:

  // reuse vector ambm kernel for assigning the elements:
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(mat).context());
  typedef viennacl::linalg::opencl::kernels::vector<NumericT>  KernelClass;
  KernelClass::init(ctx);

  cl_uint options_alpha = 0;
  viennacl::ocl::packed_cl_uint size_mat;
  if (mat.row_major())
  {
    vcl_size_t first_row_index = 0;
    vcl_size_t first_col_index = 0;
    if (k < 0)
      first_row_index = vcl_size_t(-k);
    else
      first_col_index = vcl_size_t(k);
    size_mat.start  = cl_uint( (viennacl::traits::start1(mat) + first_row_index * viennacl::traits::stride1(mat)) * viennacl::traits::internal_size2(mat)
                              + viennacl::traits::start2(mat) + first_col_index * viennacl::traits::stride2(mat));
    size_mat.stride = cl_uint(viennacl::traits::stride1(mat) * viennacl::traits::internal_size2(mat) + viennacl::traits::stride2(mat));
    size_mat.size   = cl_uint(viennacl::traits::size(vec));
    size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
  }
  else
  {
    vcl_size_t first_row_index = 0;
    vcl_size_t first_col_index = 0;
    if (k < 0)
      first_row_index = vcl_size_t(-k);
    else
      first_col_index = vcl_size_t(k);
    size_mat.start  = cl_uint(   viennacl::traits::start1(mat) + first_row_index * viennacl::traits::stride1(mat)
                              + (viennacl::traits::start2(mat) + first_col_index * viennacl::traits::stride2(mat)) * viennacl::traits::internal_size1(mat));
    size_mat.stride = cl_uint(viennacl::traits::stride2(mat) * viennacl::traits::internal_size1(mat) + viennacl::traits::stride1(mat));
    size_mat.size   = cl_uint(viennacl::traits::size(vec));
    size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
  }

  viennacl::ocl::packed_cl_uint size_vec;
  size_vec.start  = cl_uint(viennacl::traits::start(vec));
  size_vec.stride = cl_uint(viennacl::traits::stride(vec));
  size_vec.size   = cl_uint(viennacl::traits::size(vec));
  size_vec.internal_size   = cl_uint(viennacl::traits::internal_size(vec));

  viennacl::ocl::kernel & kern = ctx.get_kernel(KernelClass::program_name(), "av_cpu");
  viennacl::ocl::enqueue(kern(viennacl::traits::opencl_handle(mat),
                              size_mat,

                              viennacl::traits::opencl_handle(NumericT(1)),
                              options_alpha,
                              viennacl::traits::opencl_handle(vec),
                              size_vec)
                        );
}

template <typename NumericT>
void matrix_diag_to_vector(const matrix_base<NumericT> & mat, int k, vector_base<NumericT> & vec)
{
  // reuse vector ambm kernel for assigning the elements:
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(mat).context());
  typedef viennacl::linalg::opencl::kernels::vector<NumericT>  KernelClass;
  KernelClass::init(ctx);

  cl_uint options_alpha = 0;
  viennacl::ocl::packed_cl_uint size_mat;
  if (mat.row_major())
  {
    vcl_size_t first_row_index = 0;
    vcl_size_t first_col_index = 0;
    if (k < 0)
      first_row_index = vcl_size_t(-k);
    else
      first_col_index = vcl_size_t(k);
    size_mat.start  = cl_uint( (viennacl::traits::start1(mat) + first_row_index * viennacl::traits::stride1(mat)) * viennacl::traits::internal_size2(mat)
                              + viennacl::traits::start2(mat) + first_col_index * viennacl::traits::stride2(mat));
    size_mat.stride = cl_uint(viennacl::traits::stride1(mat) * viennacl::traits::internal_size2(mat) + viennacl::traits::stride2(mat));
    size_mat.size   = cl_uint(viennacl::traits::size(vec));
    size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
  }
  else
  {
    vcl_size_t first_row_index = 0;
    vcl_size_t first_col_index = 0;
    if (k < 0)
      first_row_index = vcl_size_t(-k);
    else
      first_col_index = vcl_size_t(k);
    size_mat.start  = cl_uint(   viennacl::traits::start1(mat) + first_row_index * viennacl::traits::stride1(mat)
                              + (viennacl::traits::start2(mat) + first_col_index * viennacl::traits::stride2(mat)) * viennacl::traits::internal_size1(mat));
    size_mat.stride = cl_uint(viennacl::traits::stride2(mat) * viennacl::traits::internal_size1(mat) + viennacl::traits::stride1(mat));
    size_mat.size   = cl_uint(viennacl::traits::size(vec));
    size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
  }

  viennacl::ocl::packed_cl_uint size_vec;
  size_vec.start  = cl_uint(viennacl::traits::start(vec));
  size_vec.stride = cl_uint(viennacl::traits::stride(vec));
  size_vec.size   = cl_uint(viennacl::traits::size(vec));
  size_vec.internal_size   = cl_uint(viennacl::traits::internal_size(vec));


  viennacl::ocl::kernel & kern = ctx.get_kernel(KernelClass::program_name(), "av_cpu");
  viennacl::ocl::enqueue(kern(viennacl::traits::opencl_handle(vec),
                              size_vec,

                              viennacl::traits::opencl_handle(NumericT(1)),
                              options_alpha,
                              viennacl::traits::opencl_handle(mat),
                              size_mat)
                        );
}

template <typename NumericT>
void matrix_row(matrix_base<NumericT> const & mat, unsigned int i, vector_base<NumericT> & vec)
{
  // reuse vector ambm kernel for assigning the elements:
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(mat).context());
  typedef viennacl::linalg::opencl::kernels::vector<NumericT>  KernelClass;
  KernelClass::init(ctx);

  cl_uint options_alpha = 0;
  viennacl::ocl::packed_cl_uint size_mat;
  if (mat.row_major())
  {
    size_mat.start  = cl_uint((viennacl::traits::start1(mat) + i * viennacl::traits::stride1(mat)) * viennacl::traits::internal_size2(mat) + viennacl::traits::start2(mat));
    size_mat.stride = cl_uint(viennacl::traits::stride2(mat));
    size_mat.size   = cl_uint(viennacl::traits::size(vec));
    size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
  }
  else
  {
    size_mat.start  = cl_uint((viennacl::traits::start1(mat) + i * viennacl::traits::stride1(mat)) + viennacl::traits::start2(mat) * viennacl::traits::internal_size1(mat));
    size_mat.stride = cl_uint(viennacl::traits::stride2(mat) * viennacl::traits::internal_size1(mat));
    size_mat.size   = cl_uint(viennacl::traits::size(vec));
    size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
  }

  viennacl::ocl::packed_cl_uint size_vec;
  size_vec.start  = cl_uint(viennacl::traits::start(vec));
  size_vec.stride = cl_uint(viennacl::traits::stride(vec));
  size_vec.size   = cl_uint(viennacl::traits::size(vec));
  size_vec.internal_size   = cl_uint(viennacl::traits::internal_size(vec));


  viennacl::ocl::kernel & kern = ctx.get_kernel(KernelClass::program_name(), "av_cpu");
  viennacl::ocl::enqueue(kern(viennacl::traits::opencl_handle(vec),
                              size_vec,

                              viennacl::traits::opencl_handle(NumericT(1)),
                              options_alpha,
                              viennacl::traits::opencl_handle(mat),
                              size_mat)
                        );
}

template <typename NumericT>
void matrix_column(const matrix_base<NumericT> & mat, unsigned int j, vector_base<NumericT> & vec)
{
  // reuse vector ambm kernel for assigning the elements:
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(mat).context());
  typedef viennacl::linalg::opencl::kernels::vector<NumericT>  KernelClass;
  KernelClass::init(ctx);

  cl_uint options_alpha = 0;
  viennacl::ocl::packed_cl_uint size_mat;
  if (mat.row_major())
  {
    size_mat.start  = cl_uint(viennacl::traits::start1(mat) * viennacl::traits::internal_size2(mat) + viennacl::traits::start2(mat) + j * viennacl::traits::stride2(mat));
    size_mat.stride = cl_uint(viennacl::traits::stride2(mat) * viennacl::traits::internal_size2(mat));
    size_mat.size   = cl_uint(viennacl::traits::size(vec));
    size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
  }
  else
  {
    size_mat.start  = cl_uint(viennacl::traits::start1(mat) + (viennacl::traits::start2(mat) + j * viennacl::traits::stride2(mat)) * viennacl::traits::internal_size1(mat));
    size_mat.stride = cl_uint(viennacl::traits::stride2(mat));
    size_mat.size   = cl_uint(viennacl::traits::size(vec));
    size_mat.internal_size   = cl_uint(viennacl::traits::internal_size(vec));
  }

  viennacl::ocl::packed_cl_uint size_vec;
  size_vec.start  = cl_uint(viennacl::traits::start(vec));
  size_vec.stride = cl_uint(viennacl::traits::stride(vec));
  size_vec.size   = cl_uint(viennacl::traits::size(vec));
  size_vec.internal_size   = cl_uint(viennacl::traits::internal_size(vec));


  viennacl::ocl::kernel & kern = ctx.get_kernel(KernelClass::program_name(), "av_cpu");
  viennacl::ocl::enqueue(kern(viennacl::traits::opencl_handle(vec),
                              size_vec,

                              viennacl::traits::opencl_handle(NumericT(1)),
                              options_alpha,
                              viennacl::traits::opencl_handle(mat),
                              size_mat)
                        );
}


//
///////////////////////// Element-wise operation //////////////////////////////////
//

// Binary operations A = B .* C and A = B ./ C
/** @brief Implementation of binary element-wise operations A = OP(B,C)
*
* @param A      The result matrix (or -range, or -slice)
* @param proxy  The proxy object holding B, C, and the operation
*/
template <typename T, typename OP>
void element_op(matrix_base<T> & A,
                matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> > const & proxy)
{
  assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::kernel & k = detail::kernel_for_matrix(A, "element_op");

  cl_uint op_type = 2; //0: product, 1: division, 2: power
  if (viennacl::is_division<OP>::value)
    op_type = 1;
  else if (viennacl::is_product<OP>::value)
    op_type = 0;

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(A),
                          cl_uint(viennacl::traits::start1(A)),           cl_uint(viennacl::traits::start2(A)),
                          cl_uint(viennacl::traits::stride1(A)),          cl_uint(viennacl::traits::stride2(A)),
                          cl_uint(viennacl::traits::size1(A)),            cl_uint(viennacl::traits::size2(A)),
                          cl_uint(viennacl::traits::internal_size1(A)),   cl_uint(viennacl::traits::internal_size2(A)),

                          viennacl::traits::opencl_handle(proxy.lhs()),
                          cl_uint(viennacl::traits::start1(proxy.lhs())),           cl_uint(viennacl::traits::start2(proxy.lhs())),
                          cl_uint(viennacl::traits::stride1(proxy.lhs())),          cl_uint(viennacl::traits::stride2(proxy.lhs())),
                          cl_uint(viennacl::traits::internal_size1(proxy.lhs())),   cl_uint(viennacl::traits::internal_size2(proxy.lhs())),

                          viennacl::traits::opencl_handle(proxy.rhs()),
                          cl_uint(viennacl::traits::start1(proxy.rhs())),           cl_uint(viennacl::traits::start2(proxy.rhs())),
                          cl_uint(viennacl::traits::stride1(proxy.rhs())),          cl_uint(viennacl::traits::stride2(proxy.rhs())),
                          cl_uint(viennacl::traits::internal_size1(proxy.rhs())),   cl_uint(viennacl::traits::internal_size2(proxy.rhs())),

                          op_type)
                        );
}


// Unary operations

/** @brief Implementation of unary element-wise operations A = OP(B)
*
* @param A      The result matrix (or -range, or -slice)
* @param proxy  The proxy object holding B and the operation
*/
template <typename T, typename OP>
void element_op(matrix_base<T> & A,
                matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<OP> > const & proxy)
{
  assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.lhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));
  assert(viennacl::traits::opencl_handle(A).context() == viennacl::traits::opencl_handle(proxy.rhs()).context() && bool("Matrices do not reside in the same OpenCL context. Automatic migration not yet supported!"));

  viennacl::ocl::kernel & k = detail::element_kernel_for_matrix(A, detail::op_to_string(OP()) + "_assign");

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(A),
                           cl_uint(viennacl::traits::start1(A)),           cl_uint(viennacl::traits::start2(A)),
                           cl_uint(viennacl::traits::stride1(A)),          cl_uint(viennacl::traits::stride2(A)),
                           cl_uint(viennacl::traits::size1(A)),            cl_uint(viennacl::traits::size2(A)),
                           cl_uint(viennacl::traits::internal_size1(A)),   cl_uint(viennacl::traits::internal_size2(A)),

                           viennacl::traits::opencl_handle(proxy.lhs()),
                           cl_uint(viennacl::traits::start1(proxy.lhs())),           cl_uint(viennacl::traits::start2(proxy.lhs())),
                           cl_uint(viennacl::traits::stride1(proxy.lhs())),          cl_uint(viennacl::traits::stride2(proxy.lhs())),
                           cl_uint(viennacl::traits::internal_size1(proxy.lhs())),   cl_uint(viennacl::traits::internal_size2(proxy.lhs())))
                        );
}


//
/////////////////////////   matrix-vector products /////////////////////////////////
//

// A * x

/** @brief Carries out matrix-vector multiplication
*
* Implementation of the convenience expression result = prod(mat, vec);
*
* @param mat    The matrix
* @param vec    The vector
* @param result The result vector
*/
template <typename NumericT>
void prod_impl(const matrix_base<NumericT> & mat, bool trans_A,
               const vector_base<NumericT> & vec,
                     vector_base<NumericT> & result)
{
  assert(viennacl::traits::handle(vec) != viennacl::traits::handle(result) && bool("No direct inplace transposed matrix-vector product possible. Introduce a temporary!"));

  viennacl::ocl::kernel & k = detail::kernel_for_matrix(mat, trans_A ? "trans_vec_mul" : "vec_mul");

  viennacl::ocl::enqueue(k(viennacl::traits::opencl_handle(mat),
                          cl_uint(viennacl::traits::start1(mat)),         cl_uint(viennacl::traits::start2(mat)),
                          cl_uint(viennacl::traits::stride1(mat)),        cl_uint(viennacl::traits::stride2(mat)),
                          cl_uint(viennacl::traits::size1(mat)),          cl_uint(viennacl::traits::size2(mat)),
                          cl_uint(viennacl::traits::internal_size1(mat)), cl_uint(viennacl::traits::internal_size2(mat)),

                          viennacl::traits::opencl_handle(vec),
                          cl_uint(viennacl::traits::start(vec)),
                          cl_uint(viennacl::traits::stride(vec)),
                          cl_uint(viennacl::traits::size(vec)),

                          viennacl::traits::opencl_handle(result),
                          cl_uint(viennacl::traits::start(result)),
                          cl_uint(viennacl::traits::stride(result)),
                          cl_uint(viennacl::traits::size(result)),

                          viennacl::ocl::local_mem(sizeof(NumericT) * k.local_work_size())
                        ) );
}


//


/** @brief Carries out matrix-matrix multiplication
*
* Implementation of C = prod(A, B);
*
*/
template<typename NumericT, typename ScalarType >
void prod_impl(matrix_base<NumericT> const & A, bool A_trans,
               matrix_base<NumericT> const & B, bool B_trans,
               matrix_base<NumericT>       & C,
               ScalarType alpha,
               ScalarType beta)
{
    bool effective_A_trans = A_trans ^ A.row_major();
    bool effective_B_trans = B_trans ^ B.row_major();

    char cAt = effective_A_trans ? 'T' : 'N';
    char cBt = effective_B_trans ? 'T' : 'N';

    std::string kernel_prefix("prod_");
    kernel_prefix+=cAt;
    kernel_prefix+=cBt;

    scheduler::statement statement = scheduler::preset::mat_mat_prod(alpha, &A, effective_A_trans, &B, effective_B_trans, beta, &C);
    kernels::matrix_prod<NumericT>::execution_handler(C.row_major(), viennacl::traits::opencl_context(C)).execute(kernel_prefix, statement);
}

//
/////////////////////////   miscellaneous operations /////////////////////////////////
//


/** @brief The implementation of the operation mat += alpha * vec1 * vec2^T, i.e. a scaled rank 1 update
*
* Implementation of the convenience expression result += alpha * outer_prod(vec1, vec2);
*
* @param A    The matrix to be updated
* @param alpha            The scaling factor (either a viennacl::scalar<>, float, or double)
* @param len_alpha        Length of the buffer for an eventual final reduction step (currently always '1')
* @param reciprocal_alpha Use 1/alpha instead of alpha
* @param flip_sign_alpha  Use -alpha instead of alpha
* @param vec1    The first vector
* @param vec2    The second vector
*/
template<typename NumericT, typename ScalarT1>
void scaled_rank_1_update(matrix_base<NumericT> & A,
                          ScalarT1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                          const vector_base<NumericT> & vec1,
                          const vector_base<NumericT> & vec2)
{
  assert( (viennacl::traits::size1(A) == viennacl::traits::size(vec1)) && bool("Size mismatch in scaled_rank_1_update: size1(A) != size(v1)"));
  assert( (viennacl::traits::size2(A) == viennacl::traits::size(vec2)) && bool("Size mismatch in scaled_rank_1_update: size2(A) != size(v2)"));

  cl_uint options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);
  bool is_cpu = viennacl::is_cpu_scalar<ScalarT1>::value;
  viennacl::ocl::kernel& kernel= detail::legacy_kernel_for_matrix(A, is_cpu ? "scaled_rank1_update_cpu" : "scaled_rank1_update_gpu");

  viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(A),
                           cl_uint(viennacl::traits::start1(A)),           cl_uint(viennacl::traits::start2(A)),
                           cl_uint(viennacl::traits::stride1(A)),          cl_uint(viennacl::traits::stride2(A)),
                           cl_uint(viennacl::traits::size1(A)),            cl_uint(viennacl::traits::size2(A)),
                           cl_uint(viennacl::traits::internal_size1(A)),   cl_uint(viennacl::traits::internal_size2(A)),

                           viennacl::traits::opencl_handle(viennacl::tools::promote_if_host_scalar<NumericT>(alpha)),
                           options_alpha,

                           viennacl::traits::opencl_handle(vec1),
                           cl_uint(viennacl::traits::start(vec1)),
                           cl_uint(viennacl::traits::stride(vec1)),
                           cl_uint(viennacl::traits::size(vec1)),

                           viennacl::traits::opencl_handle(vec2),
                           cl_uint(viennacl::traits::start(vec2)),
                           cl_uint(viennacl::traits::stride(vec2)),
                           cl_uint(viennacl::traits::size(vec2))
                          )
                        );
}

//
template <typename SCALARTYPE, typename VectorType>
void bidiag_pack_svd(viennacl::matrix<SCALARTYPE>& A,
                 VectorType & dh,
                 VectorType & sh
                )
{
  viennacl::vector<SCALARTYPE> D(dh.size());
  viennacl::vector<SCALARTYPE> S(sh.size());

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<SCALARTYPE>::program_name(), SVD_BIDIAG_PACK_KERNEL);

  viennacl::ocl::enqueue(kernel(
                                A,
                                D,
                                S,
                                static_cast<cl_uint>(A.size1()),
                                static_cast<cl_uint>(A.size2()),
                                static_cast<cl_uint>(A.internal_size2())
                              ));

  fast_copy(D, dh);
  fast_copy(S, sh);
}


template <typename NumericT>
void bidiag_pack(matrix_base<NumericT> & A,
                 viennacl::vector<NumericT> & dh,
                 viennacl::vector<NumericT> & sh
                )
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

  if(A.row_major())
  {
      viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
      viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), SVD_BIDIAG_PACK_KERNEL);

      viennacl::ocl::enqueue(kernel(
                                    A,
                                    dh,
                                    sh,
                                    cl_uint(viennacl::traits::size1(A)),
                                    cl_uint(viennacl::traits::size2(A)),
                                    cl_uint(viennacl::traits::internal_size2(A))
                                  ));
  }
  else
  {
      viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
      viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), SVD_BIDIAG_PACK_KERNEL);

      viennacl::ocl::enqueue(kernel(
                                    A,
                                    dh,
                                    sh,
                                    cl_uint(viennacl::traits::size1(A)),
                                    cl_uint(viennacl::traits::size2(A)),
                                    cl_uint(viennacl::traits::internal_size2(A))
                                  ));
  }
}


template <typename NumericT>
void house_update_A_left(matrix_base<NumericT> & A,
                         vector_base<NumericT> & D,
                         vcl_size_t start)
{

    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
    if(A.row_major())
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_A_LEFT_KERNEL);
        viennacl::ocl::enqueue(kernel(
                                      A,
                                      D,
                                      static_cast<cl_uint>(start + 1),
                                      static_cast<cl_uint>(start),
                                      cl_uint(viennacl::traits::size1(A)),
                                      cl_uint(viennacl::traits::size2(A)),
                                      cl_uint(viennacl::traits::internal_size2(A)),
                                      viennacl::ocl::local_mem(static_cast<cl_uint>(128 * 4))
                              ));
    }
    else
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_A_LEFT_KERNEL);
        viennacl::ocl::enqueue(kernel(
                                      A,
                                      D,
                                      static_cast<cl_uint>(start + 1),
                                      static_cast<cl_uint>(start),
                                      cl_uint(viennacl::traits::size1(A)),
                                      cl_uint(viennacl::traits::size2(A)),
                                      cl_uint(viennacl::traits::internal_size2(A)),
                                      viennacl::ocl::local_mem(static_cast<cl_uint>(128 * 4))
                              ));
    }




}

template <typename NumericT>
void house_update_A_right(matrix_base<NumericT> & A,
                          vector_base<NumericT> & D)
{
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

    if(A.row_major())
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_A_RIGHT_KERNEL);

        viennacl::ocl::enqueue(kernel(
                                      A,
                                      D,
                                      static_cast<cl_uint>(0),
                                      static_cast<cl_uint>(0),
                                      cl_uint(viennacl::traits::size1(A)),
                                      cl_uint(viennacl::traits::size2(A)),
                                      cl_uint(viennacl::traits::internal_size2(A)),
                                      viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(NumericT)))
                              ));
    }
    else
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_A_RIGHT_KERNEL);

        viennacl::ocl::enqueue(kernel(
                                      A,
                                      D,
                                      static_cast<cl_uint>(0),
                                      static_cast<cl_uint>(0),
                                      cl_uint(viennacl::traits::size1(A)),
                                      cl_uint(viennacl::traits::size2(A)),
                                      cl_uint(viennacl::traits::internal_size2(A)),
                                      viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(NumericT)))
                              ));
    }


}



template <typename NumericT>
void house_update_QL(matrix_base<NumericT> & Q,
                     vector_base<NumericT> & D,
                     vcl_size_t A_size1)

{
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(Q).context());

    if(Q.row_major())
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_QL_KERNEL);

        viennacl::ocl::enqueue(kernel(
                                        Q,
                                        D,
                                        cl_uint(A_size1),
                                        cl_uint(viennacl::traits::internal_size2(Q)),
                                        viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(NumericT)))
                                    ));
    }
    else
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), SVD_HOUSEHOLDER_UPDATE_QL_KERNEL);

        viennacl::ocl::enqueue(kernel(
                                        Q,
                                        D,
                                        cl_uint(A_size1),
                                        cl_uint(viennacl::traits::internal_size2(Q)),
                                        viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(NumericT)))
                                    ));
    }

}


template<typename NumericT>
  void givens_next(matrix_base<NumericT> & matrix,
                  vector_base<NumericT>& tmp1,
                  vector_base<NumericT>& tmp2,
                  int l,
                  int m
                )
  {
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(matrix).context());

    if(matrix.row_major())
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), SVD_GIVENS_NEXT_KERNEL);
        kernel.global_work_size(0, viennacl::tools::align_to_multiple<cl_uint>(cl_uint(viennacl::traits::size1(matrix)), 256));
        kernel.local_work_size(0, 256);

        viennacl::ocl::enqueue(kernel(
                                      matrix,
                                      tmp1,
                                      tmp2,
                                      cl_uint(viennacl::traits::size1(matrix)),
                                      cl_uint(viennacl::traits::internal_size2(matrix)),
                                      static_cast<cl_uint>(l),
                                      static_cast<cl_uint>(m - 1)
                              ));
    }
    else
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), SVD_GIVENS_NEXT_KERNEL);
        kernel.global_work_size(0, viennacl::tools::align_to_multiple<cl_uint>(cl_uint(viennacl::traits::size1(matrix)), 256));
        kernel.local_work_size(0, 256);

        viennacl::ocl::enqueue(kernel(
                                      matrix,
                                      tmp1,
                                      tmp2,
                                      cl_uint(viennacl::traits::size1(matrix)),
                                      cl_uint(viennacl::traits::internal_size2(matrix)),
                                      static_cast<cl_uint>(l),
                                      static_cast<cl_uint>(m - 1)
                              ));
    }


  }

  template <typename NumericT>
  void copy_vec(matrix_base<NumericT>& A,
                vector_base<NumericT> & V,
                vcl_size_t row_start,
                vcl_size_t col_start,
                bool copy_col
  )
  {
    std::string kernel_name = copy_col ? SVD_COPY_COL_KERNEL : SVD_COPY_ROW_KERNEL;
    viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

    if(A.row_major())
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, row_major>::program_name(), kernel_name);

        viennacl::ocl::enqueue(kernel(
                                      A,
                                      V,
                                      static_cast<cl_uint>(row_start),
                                      static_cast<cl_uint>(col_start),
                                      copy_col ? cl_uint(viennacl::traits::size1(A))
                                               : cl_uint(viennacl::traits::size2(A)),
                                      static_cast<cl_uint>(A.internal_size2())
                              ));
    }
    else
    {
        viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::init(ctx);
        viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<NumericT, column_major>::program_name(), kernel_name);

        viennacl::ocl::enqueue(kernel(
                                      A,
                                      V,
                                      static_cast<cl_uint>(row_start),
                                      static_cast<cl_uint>(col_start),
                                      copy_col ? cl_uint(viennacl::traits::size1(A))
                                               : cl_uint(viennacl::traits::size2(A)),
                                      static_cast<cl_uint>(A.internal_size2())
                              ));
    }


  }

} // namespace opencl
} //namespace linalg
} //namespace viennacl


#endif
