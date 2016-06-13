#ifndef _VIENNACL_LINALG_DETAIL_BISECT_GERSCHORIN_HPP_
#define _VIENNACL_LINALG_DETAIL_BISECT_GERSCHORIN_HPP_


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


/** @file viennacl/linalg/detail//bisect/gerschgorin.hpp
    @brief  Computation of Gerschgorin interval for symmetric, tridiagonal matrix

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

#include "viennacl/linalg/detail/bisect/util.hpp"
#include "viennacl/vector.hpp"

namespace viennacl
{
namespace linalg
{
namespace detail
{
  ////////////////////////////////////////////////////////////////////////////////
  //! Compute Gerschgorin interval for symmetric, tridiagonal matrix
  //! @param  d  diagonal elements
  //! @param  s  superdiagonal elements
  //! @param  n  size of matrix
  //! @param  lg  lower limit of Gerschgorin interval
  //! @param  ug  upper limit of Gerschgorin interval
  ////////////////////////////////////////////////////////////////////////////////
  template<typename NumericT>
  void
  computeGerschgorin(std::vector<NumericT> & d, std::vector<NumericT> & s, unsigned int n, NumericT &lg, NumericT &ug)
  {
      // compute bounds
      for (unsigned int i = 1; i < (n - 1); ++i)
      {

          // sum over the absolute values of all elements of row i
          NumericT sum_abs_ni = fabsf(s[i]) + fabsf(s[i + 1]);

          lg = min(lg, d[i] - sum_abs_ni);
          ug = max(ug, d[i] + sum_abs_ni);
      }

      // first and last row, only one superdiagonal element

      // first row
      lg = min(lg, d[0] - fabsf(s[1]));
      ug = max(ug, d[0] + fabsf(s[1]));

      // last row
      lg = min(lg, d[n-1] - fabsf(s[n-1]));
      ug = max(ug, d[n-1] + fabsf(s[n-1]));

      // increase interval to avoid side effects of fp arithmetic
      NumericT bnorm = max(fabsf(ug), fabsf(lg));

      // these values depend on the implmentation of floating count that is
      // employed in the following
      NumericT psi_0 = 11 * FLT_EPSILON * bnorm;
      NumericT psi_n = 11 * FLT_EPSILON * bnorm;

      lg = lg - bnorm * 2 * static_cast<NumericT>(n) * FLT_EPSILON - psi_0;
      ug = ug + bnorm * 2 * static_cast<NumericT>(n) * FLT_EPSILON + psi_n;

      ug = max(lg, ug);
  }
}  // namespace detail
}  // namespace linalg
} // namespace viennacl
#endif  // _VIENNACL_LINALG_DETAIL_GERSCHORIN_H_
