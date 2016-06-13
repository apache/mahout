#ifndef VIENNACL_LINALG_DETAIL_BISECT_KERNEL_CALLS_HPP_
#define VIENNACL_LINALG_DETAIL_BISECT_KERNEL_CALLS_HPP_

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

/** @file viennacl/linalg/detail/bisect/bisect_kernel_calls.hpp
    @brief Kernel calls for the bisection algorithm

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/


#include "viennacl/forwards.h"
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

#include "viennacl/linalg/detail/bisect/structs.hpp"
#ifdef VIENNACL_WITH_OPENCL
   #include "viennacl/linalg/opencl/bisect_kernel_calls.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/bisect_kernel_calls.hpp"
#endif

namespace viennacl
{
namespace linalg
{
namespace detail
{
 template<typename NumericT>
 void bisectSmall(const InputData<NumericT> &input, ResultDataSmall<NumericT> &result,
                  const unsigned int mat_size,
                  const NumericT lg, const NumericT ug,
                  const NumericT precision)
  {
    switch (viennacl::traits::handle(input.g_a).get_active_handle_id())
    {
#ifdef VIENNACL_WITH_OPENCL
      case viennacl::OPENCL_MEMORY:
        viennacl::linalg::opencl::bisectSmall(input, result,
                                             mat_size,
                                             lg,ug,
                                             precision);
        break;
#endif
#ifdef VIENNACL_WITH_CUDA
      case viennacl::CUDA_MEMORY:
        viennacl::linalg::cuda::bisectSmall(input, result,
                                             mat_size,
                                             lg,ug,
                                             precision);
        break;
#endif
      case viennacl::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }




 template<typename NumericT>
 void bisectLarge(const InputData<NumericT> &input, ResultDataLarge<NumericT> &result,
                    const unsigned int mat_size,
                    const NumericT lg, const NumericT ug,
                    const NumericT precision)
  {
    switch (viennacl::traits::handle(input.g_a).get_active_handle_id())
    {
#ifdef VIENNACL_WITH_OPENCL
      case viennacl::OPENCL_MEMORY:
        viennacl::linalg::opencl::bisectLarge(input, result,
                                             mat_size,
                                             lg,ug,
                                             precision);
        break;
#endif
#ifdef VIENNACL_WITH_CUDA
      case viennacl::CUDA_MEMORY:
        viennacl::linalg::cuda::bisectLarge(input, result,
                                             mat_size,
                                             lg,ug,
                                             precision);
        break;
#endif
      case viennacl::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }





 template<typename NumericT>
 void bisectLarge_OneIntervals(const InputData<NumericT> &input, ResultDataLarge<NumericT> &result,
                    const unsigned int mat_size,
                    const NumericT precision)
  {
    switch (viennacl::traits::handle(input.g_a).get_active_handle_id())
    {
#ifdef VIENNACL_WITH_OPENCL
      case viennacl::OPENCL_MEMORY:
        viennacl::linalg::opencl::bisectLargeOneIntervals(input, result,
                                             mat_size,
                                             precision);
        break;
#endif
#ifdef VIENNACL_WITH_CUDA
      case viennacl::CUDA_MEMORY:
        viennacl::linalg::cuda::bisectLarge_OneIntervals(input, result,
                                             mat_size,
                                             precision);

        break;
#endif
      case viennacl::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }




 template<typename NumericT>
 void bisectLarge_MultIntervals(const InputData<NumericT> &input, ResultDataLarge<NumericT> &result,
                    const unsigned int mat_size,
                    const NumericT precision)
  {
    switch (viennacl::traits::handle(input.g_a).get_active_handle_id())
    {
#ifdef VIENNACL_WITH_OPENCL
      case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::bisectLargeMultIntervals(input, result,
                                           mat_size,
                                           precision);
        break;
#endif
#ifdef VIENNACL_WITH_CUDA
      case viennacl::CUDA_MEMORY:
        viennacl::linalg::cuda::bisectLarge_MultIntervals(input, result,
                                             mat_size,
                                             precision);
        break;
#endif
      case viennacl::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }
} // namespace detail
} // namespace linalg
} //namespace viennacl


#endif
