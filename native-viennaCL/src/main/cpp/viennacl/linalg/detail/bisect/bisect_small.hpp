#ifndef VIENNACL_LINALG_DETAIL_BISECT_SMALL_HPP_
#define VIENNACL_LINALG_DETAIL_BISECT_SMALL_HPP_

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


/** @file viennacl/linalg/detail//bisect/bisect_small.hpp
    @brief Computation of eigenvalues of a small symmetric, tridiagonal matrix

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>

// includes, project

#include "viennacl/linalg/detail/bisect/structs.hpp"

// includes, kernels
#include "viennacl/linalg/detail/bisect/bisect_kernel_calls.hpp"

namespace viennacl
{
namespace linalg
{
namespace detail
{
////////////////////////////////////////////////////////////////////////////////
//! Determine eigenvalues for matrices smaller than MAX_SMALL_MATRIX
//! @param  input  handles to input data of kernel
//! @param  result handles to result of kernel
//! @param  mat_size  matrix size
//! @param  lg  lower limit of Gerschgorin interval
//! @param  ug  upper limit of Gerschgorin interval
//! @param  precision  desired precision of eigenvalues
////////////////////////////////////////////////////////////////////////////////
template<typename NumericT>
void
computeEigenvaluesSmallMatrix(const InputData<NumericT> &input, ResultDataSmall<NumericT> &result,
                              const unsigned int mat_size,
                              const NumericT lg, const NumericT ug,
                              const NumericT precision)
{
  viennacl::linalg::detail::bisectSmall( input, result, mat_size, lg, ug, precision);
}


////////////////////////////////////////////////////////////////////////////////
//! Process the result obtained on the device, that is transfer to host and
//! perform basic sanity checking
//! @param  result  handles to result data
//! @param  mat_size   matrix size
////////////////////////////////////////////////////////////////////////////////
template<typename NumericT>
void
processResultSmallMatrix(ResultDataSmall<NumericT> &result,
                         const unsigned int mat_size)
{
  // copy data back to host
  std::vector<NumericT> left(mat_size);
  std::vector<unsigned int> left_count(mat_size);

  viennacl::copy(result.vcl_g_left, left);
  viennacl::copy(result.vcl_g_left_count, left_count);

  for (unsigned int i = 0; i < mat_size; ++i)
  {
      result.std_eigenvalues[left_count[i]] = left[i];
  }
}
}  // namespace detail
}  // namespace linalg
} // namespace viennacl
#endif
