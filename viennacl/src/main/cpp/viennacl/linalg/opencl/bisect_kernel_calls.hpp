#ifndef VIENNACL_LINALG_OPENCL_BISECT_KERNEL_CALLS_HPP_
#define VIENNACL_LINALG_OPENCL_BISECT_KERNEL_CALLS_HPP_


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


/** @file viennacl/linalg/opencl/bisect_kernel_calls.hpp
    @brief OpenCL kernel calls for the bisection algorithm

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/

// includes, project
#include "viennacl/linalg/opencl/kernels/bisect.hpp"
#include "viennacl/linalg/detail/bisect/structs.hpp"
#include "viennacl/linalg/detail/bisect/config.hpp"
#include "viennacl/linalg/detail/bisect/util.hpp"

namespace viennacl
{
namespace linalg
{
namespace opencl
{
const std::string BISECT_KERNEL_SMALL = "bisectKernelSmall";
const std::string BISECT_KERNEL_LARGE = "bisectKernelLarge";
const std::string BISECT_KERNEL_LARGE_ONE_INTERVALS  = "bisectKernelLarge_OneIntervals";
const std::string BISECT_KERNEL_LARGE_MULT_INTERVALS = "bisectKernelLarge_MultIntervals";

template<typename NumericT>
void bisectSmall(const viennacl::linalg::detail::InputData<NumericT> &input,
                         viennacl::linalg::detail::ResultDataSmall<NumericT> &result,
                         const unsigned int mat_size,
                         const NumericT lg, const NumericT ug,
                         const NumericT precision)
    {
      viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.g_a).context());
      viennacl::linalg::opencl::kernels::bisect_kernel<NumericT>::init(ctx);

      viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel<NumericT>::program_name(), BISECT_KERNEL_SMALL);
      kernel.global_work_size(0, 1 * VIENNACL_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX);
      kernel.local_work_size(0, VIENNACL_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX);

      viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(input.g_a),
                                    viennacl::traits::opencl_handle(input.g_b),
                                    static_cast<cl_uint>(mat_size),
                                    viennacl::traits::opencl_handle(result.vcl_g_left),
                                    viennacl::traits::opencl_handle(result.vcl_g_right),
                                    viennacl::traits::opencl_handle(result.vcl_g_left_count),
                                    viennacl::traits::opencl_handle(result.vcl_g_right_count),
                                    static_cast<NumericT>(lg),
                                    static_cast<NumericT>(ug),
                                    static_cast<cl_uint>(0),
                                    static_cast<cl_uint>(mat_size),
                                    static_cast<NumericT>(precision)
                            ));

    }

template<typename NumericT>
void bisectLarge(const viennacl::linalg::detail::InputData<NumericT> &input,
                 viennacl::linalg::detail::ResultDataLarge<NumericT> &result,
                 const unsigned int mat_size,
                 const NumericT lg, const NumericT ug,
                 const NumericT precision)
    {
      viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.g_a).context());
      viennacl::linalg::opencl::kernels::bisect_kernel<NumericT>::init(ctx);

      viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel<NumericT>::program_name(), BISECT_KERNEL_LARGE);
      kernel.global_work_size(0, mat_size > 512 ? VIENNACL_BISECT_MAX_THREADS_BLOCK : VIENNACL_BISECT_MAX_THREADS_BLOCK / 2);     // Use only 128 threads for 256 < n <= 512, this
      kernel.local_work_size(0,  mat_size > 512 ? VIENNACL_BISECT_MAX_THREADS_BLOCK : VIENNACL_BISECT_MAX_THREADS_BLOCK / 2);     // is reasoned

      viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(input.g_a),
                                    viennacl::traits::opencl_handle(input.g_b),
                                    static_cast<cl_uint>(mat_size),
                                    static_cast<NumericT>(lg),
                                    static_cast<NumericT>(ug),
                                    static_cast<cl_uint>(0),
                                    static_cast<cl_uint>(mat_size),
                                    static_cast<NumericT>(precision),
                                    viennacl::traits::opencl_handle(result.g_num_one),
                                    viennacl::traits::opencl_handle(result.g_num_blocks_mult),
                                    viennacl::traits::opencl_handle(result.g_left_one),
                                    viennacl::traits::opencl_handle(result.g_right_one),
                                    viennacl::traits::opencl_handle(result.g_pos_one),
                                    viennacl::traits::opencl_handle(result.g_left_mult),
                                    viennacl::traits::opencl_handle(result.g_right_mult),
                                    viennacl::traits::opencl_handle(result.g_left_count_mult),
                                    viennacl::traits::opencl_handle(result.g_right_count_mult),
                                    viennacl::traits::opencl_handle(result.g_blocks_mult),
                                    viennacl::traits::opencl_handle(result.g_blocks_mult_sum)
                            ));

    }

template<typename NumericT>
void bisectLargeOneIntervals(const viennacl::linalg::detail::InputData<NumericT> &input,
                             viennacl::linalg::detail::ResultDataLarge<NumericT> &result,
                             const unsigned int mat_size,
                             const NumericT precision)
    {
      unsigned int num_one_intervals = result.g_num_one;
      unsigned int num_blocks = viennacl::linalg::detail::getNumBlocksLinear(num_one_intervals,
                                                                             mat_size > 512 ? VIENNACL_BISECT_MAX_THREADS_BLOCK: VIENNACL_BISECT_MAX_THREADS_BLOCK / 2);

      viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.g_a).context());
      viennacl::linalg::opencl::kernels::bisect_kernel<NumericT>::init(ctx);

      viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel<NumericT>::program_name(), BISECT_KERNEL_LARGE_ONE_INTERVALS);
      kernel.global_work_size(0, num_blocks * (mat_size > 512 ? VIENNACL_BISECT_MAX_THREADS_BLOCK : VIENNACL_BISECT_MAX_THREADS_BLOCK / 2));
      kernel.local_work_size(0, mat_size > 512 ? VIENNACL_BISECT_MAX_THREADS_BLOCK : VIENNACL_BISECT_MAX_THREADS_BLOCK / 2);

      viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(input.g_a),
                                    viennacl::traits::opencl_handle(input.g_b),
                                    static_cast<cl_uint>(mat_size),
                                    static_cast<cl_uint>(num_one_intervals),
                                    viennacl::traits::opencl_handle(result.g_left_one),
                                    viennacl::traits::opencl_handle(result.g_right_one),
                                    viennacl::traits::opencl_handle(result.g_pos_one),
                                    static_cast<NumericT>(precision)
                            ));
    }


template<typename NumericT>
void bisectLargeMultIntervals(const viennacl::linalg::detail::InputData<NumericT> &input,
                              viennacl::linalg::detail::ResultDataLarge<NumericT> &result,
                              const unsigned int mat_size,
                              const NumericT precision)
    {
      unsigned int  num_blocks_mult = result.g_num_blocks_mult;

      viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(input.g_a).context());
      viennacl::linalg::opencl::kernels::bisect_kernel<NumericT>::init(ctx);

      viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::bisect_kernel<NumericT>::program_name(), BISECT_KERNEL_LARGE_MULT_INTERVALS);
      kernel.global_work_size(0, num_blocks_mult * (mat_size > 512 ? VIENNACL_BISECT_MAX_THREADS_BLOCK : VIENNACL_BISECT_MAX_THREADS_BLOCK / 2));
      kernel.local_work_size(0,                     mat_size > 512 ? VIENNACL_BISECT_MAX_THREADS_BLOCK : VIENNACL_BISECT_MAX_THREADS_BLOCK / 2);

      viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(input.g_a),
                                    viennacl::traits::opencl_handle(input.g_b),
                                    static_cast<cl_uint>(mat_size),
                                    viennacl::traits::opencl_handle(result.g_blocks_mult),
                                    viennacl::traits::opencl_handle(result.g_blocks_mult_sum),
                                    viennacl::traits::opencl_handle(result.g_left_mult),
                                    viennacl::traits::opencl_handle(result.g_right_mult),
                                    viennacl::traits::opencl_handle(result.g_left_count_mult),
                                    viennacl::traits::opencl_handle(result.g_right_count_mult),
                                    viennacl::traits::opencl_handle(result.g_lambda_mult),
                                    viennacl::traits::opencl_handle(result.g_pos_mult),
                                    static_cast<NumericT>(precision)
                            ));
    }
} // namespace opencl
} // namespace linalg
} // namespace viennacl

#endif
