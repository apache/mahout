#ifndef VIENNACL_LINALG_NORM_INF_HPP_
#define VIENNACL_LINALG_NORM_INF_HPP_

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

/** @file norm_inf.hpp
    @brief Generic interface for the l^infty-norm. See viennacl/linalg/vector_operations.hpp for implementations.
*/

#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/tag_of.hpp"

namespace viennacl
{
  //
  // generic norm_inf function
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
    norm_inf(VectorT const& v1)
    {
      return boost::numeric::ublas::norm_inf(v1);
    }
    #endif


    // ----------------------------------------------------
    // STL
    //
    template< typename T, typename A >
    T norm_inf(std::vector<T, A> const & v1)
    {
      //std::cout << "stl .. " << std::endl;
      T result = 0;
      for (typename std::vector<T, A>::size_type i=0; i<v1.size(); ++i)
      {
        if (std::fabs(v1[i]) > result)
          result = std::fabs(v1[i]);
      }

      return result;
    }

    // ----------------------------------------------------
    // VIENNACL
    //
    template< typename ScalarType>
    viennacl::scalar_expression< const viennacl::vector_base<ScalarType>,
                                 const viennacl::vector_base<ScalarType>,
                                 viennacl::op_norm_inf >
    norm_inf(viennacl::vector_base<ScalarType> const & v1)
    {
       //std::cout << "viennacl .. " << std::endl;
      return viennacl::scalar_expression< const viennacl::vector_base<ScalarType>,
                                          const viennacl::vector_base<ScalarType>,
                                          viennacl::op_norm_inf >(v1, v1);
    }

    // with vector expression:
    template<typename LHS, typename RHS, typename OP>
    viennacl::scalar_expression<const viennacl::vector_expression<const LHS, const RHS, OP>,
                                const viennacl::vector_expression<const LHS, const RHS, OP>,
                                viennacl::op_norm_inf>
    norm_inf(viennacl::vector_expression<const LHS, const RHS, OP> const & vector)
    {
      return viennacl::scalar_expression< const viennacl::vector_expression<const LHS, const RHS, OP>,
                                          const viennacl::vector_expression<const LHS, const RHS, OP>,
                                          viennacl::op_norm_inf >(vector, vector);
    }


  } // end namespace linalg
} // end namespace viennacl
#endif





