#ifndef VIENNACL_LINALG_NMF_HPP
#define VIENNACL_LINALG_NMF_HPP

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

/** @file viennacl/linalg/nmf.hpp
 @brief Provides a nonnegative matrix factorization implementation.  Experimental.


 */

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_frobenius.hpp"

#include "viennacl/linalg/host_based/nmf_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
#include "viennacl/linalg/opencl/kernels/nmf.hpp"
#include "viennacl/linalg/opencl/nmf_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
#include "viennacl/linalg/cuda/nmf_operations.hpp"
#endif

namespace viennacl
{
  namespace linalg
  {

    /** @brief The nonnegative matrix factorization (approximation) algorithm as suggested by Lee and Seung. Factorizes a matrix V with nonnegative entries into matrices W and H such that ||V - W*H|| is minimized.
     *
     * @param V     Input matrix
     * @param W     First factor
     * @param H     Second factor
     * @param conf  A configuration object holding tolerances and the like
     */
    template<typename ScalarType>
    void nmf(viennacl::matrix_base<ScalarType> const & V, viennacl::matrix_base<ScalarType> & W,
        viennacl::matrix_base<ScalarType> & H, viennacl::linalg::nmf_config const & conf)
    {
      assert(V.size1() == W.size1() && V.size2() == H.size2() && bool("Dimensions of W and H don't allow for V = W * H"));
      assert(W.size2() == H.size1() && bool("Dimensions of W and H don't match, prod(W, H) impossible"));

      switch (viennacl::traits::handle(V).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::nmf(V, W, H, conf);
          break;
#ifdef VIENNACL_WITH_OPENCL
          case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::nmf(V,W,H,conf);
          break;
#endif

#ifdef VIENNACL_WITH_CUDA
          case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::nmf(V,W,H,conf);
          break;
#endif

        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");

      }

    }
  }
}

#endif
