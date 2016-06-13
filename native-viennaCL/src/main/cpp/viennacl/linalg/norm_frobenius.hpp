#ifndef VIENNACL_LINALG_NORM_FROBENIUS_HPP_
#define VIENNACL_LINALG_NORM_FROBENIUS_HPP_

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

/** @file viennacl/linalg/norm_frobenius.hpp
    @brief Generic interface for the Frobenius norm.
*/

#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/tag_of.hpp"

namespace viennacl
{
  //
  // generic norm_frobenius function
  //   uses tag dispatch to identify which algorithm
  //   should be called
  //
  namespace linalg
  {

    #ifdef VIENNACL_WITH_UBLAS
    // ----------------------------------------------------
    // UBLAS
    //
    template< typename VectorT >
    typename viennacl::enable_if< viennacl::is_ublas< typename viennacl::traits::tag_of< VectorT >::type >::value,
                                  typename VectorT::value_type
                                >::type
    norm_frobenius(VectorT const& v1)
    {
      return boost::numeric::ublas::norm_frobenius(v1);
    }
    #endif


    // ----------------------------------------------------
    // VIENNACL
    //
    template<typename NumericT>
    scalar_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_norm_frobenius>
    norm_frobenius(const matrix_base<NumericT> & A)
    {
      return scalar_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_norm_frobenius>(A, A);
    }

  } // end namespace linalg
} // end namespace viennacl
#endif





