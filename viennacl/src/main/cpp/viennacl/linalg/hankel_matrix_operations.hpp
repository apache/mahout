#ifndef VIENNACL_LINALG_HANKEL_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_HANKEL_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/hankel_matrix_operations.hpp
    @brief Implementations of operations using hankel_matrix. Experimental.
*/

#include "viennacl/forwards.h"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/fft.hpp"
#include "viennacl/linalg/toeplitz_matrix_operations.hpp"

namespace viennacl
{
namespace linalg
{

// A * x

/** @brief Carries out matrix-vector multiplication with a hankel_matrix
*
* Implementation of the convenience expression result = prod(mat, vec);
*
* @param A      The matrix
* @param vec    The vector
* @param result The result vector
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::hankel_matrix<NumericT, AlignmentV> const & A,
               viennacl::vector_base<NumericT> const & vec,
               viennacl::vector_base<NumericT>       & result)
{
  assert(A.size1() == result.size() && bool("Dimension mismatch"));
  assert(A.size2() == vec.size()    && bool("Dimension mismatch"));

  prod_impl(A.elements(), vec, result);
  viennacl::linalg::reverse(result);
}

} //namespace linalg


} //namespace viennacl


#endif
