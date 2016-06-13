#ifndef VIENNACL_LINALG_TOEPLITZ_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_TOEPLITZ_MATRIX_OPERATIONS_HPP_

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

/** @file toeplitz_matrix_operations.hpp
    @brief Implementations of operations using toeplitz_matrix. Experimental.
*/

#include "viennacl/forwards.h"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/fft.hpp"

namespace viennacl
{
  namespace linalg
  {


    // A * x

    /** @brief Carries out matrix-vector multiplication with a toeplitz_matrix
    *
    * Implementation of the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param result The result vector
    */
    template<class SCALARTYPE, unsigned int ALIGNMENT>
    void prod_impl(const viennacl::toeplitz_matrix<SCALARTYPE, ALIGNMENT> & mat,
                   const viennacl::vector_base<SCALARTYPE> & vec,
                         viennacl::vector_base<SCALARTYPE> & result)
    {
      assert(mat.size1() == result.size());
      assert(mat.size2() == vec.size());

      viennacl::vector<SCALARTYPE> tmp(vec.size() * 4); tmp.clear();
      viennacl::vector<SCALARTYPE> tmp2(vec.size() * 4);

      viennacl::vector<SCALARTYPE> tep(mat.elements().size() * 2);
      viennacl::linalg::real_to_complex(mat.elements(), tep, mat.elements().size());



      viennacl::copy(vec.begin(), vec.end(), tmp.begin());
      viennacl::linalg::real_to_complex(tmp, tmp2, vec.size() * 2);
      viennacl::linalg::convolve(tep, tmp2, tmp);
      viennacl::linalg::complex_to_real(tmp, tmp2, vec.size() * 2);
      viennacl::copy(tmp2.begin(), tmp2.begin() + static_cast<vcl_ptrdiff_t>(vec.size()), result.begin());
    }

  } //namespace linalg



} //namespace viennacl


#endif
