#ifndef VIENNACL_LINALG_NORM_2_HPP_
#define VIENNACL_LINALG_NORM_2_HPP_

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

/** @file norm_2.hpp
    @brief Generic interface for the l^2-norm. See viennacl/linalg/vector_operations.hpp for implementations.
*/

#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/tag_of.hpp"

namespace viennacl
{
  //
  // generic norm_2 function
  //   uses tag dispatch to identify which algorithm
  //   should be called
  //
  namespace linalg
  {
    #ifdef VIENNACL_WITH_MTL4
    // ----------------------------------------------------
    // MTL4
    //
    template< typename VectorT >
    typename viennacl::enable_if< viennacl::is_mtl4< typename viennacl::traits::tag_of< VectorT >::type >::value,
                                  typename VectorT::value_type>::type
    norm_2(VectorT const & v)
    {
      return mtl::two_norm(v);
    }
    #endif

    #ifdef VIENNACL_WITH_ARMADILLO
    // ----------------------------------------------------
    // Armadillo
    //
    template<typename NumericT>
    NumericT norm_2(arma::Col<NumericT> const& v)
    {
      return norm(v);
    }
    #endif

    #ifdef VIENNACL_WITH_EIGEN
    // ----------------------------------------------------
    // EIGEN
    //
    template< typename VectorT >
    typename viennacl::enable_if< viennacl::is_eigen< typename viennacl::traits::tag_of< VectorT >::type >::value,
                                  typename VectorT::RealScalar>::type
    norm_2(VectorT const & v)
    {
      return v.norm();
    }
    #endif


    #ifdef VIENNACL_WITH_UBLAS
    // ----------------------------------------------------
    // UBLAS
    //
    template< typename VectorT >
    typename viennacl::enable_if< viennacl::is_ublas< typename viennacl::traits::tag_of< VectorT >::type >::value,
                                  typename VectorT::value_type>::type
    norm_2(VectorT const & v)
    {
      return boost::numeric::ublas::norm_2(v);
    }
    #endif


    // ----------------------------------------------------
    // STL
    //
    template< typename T, typename A >
    T norm_2(std::vector<T, A> const & v1)
    {
      T result = 0;
      for (typename std::vector<T, A>::size_type i=0; i<v1.size(); ++i)
        result += v1[i] * v1[i];

      return std::sqrt(result);
    }

    // ----------------------------------------------------
    // VIENNACL
    //
    template< typename ScalarType>
    viennacl::scalar_expression< const viennacl::vector_base<ScalarType>,
                                 const viennacl::vector_base<ScalarType>,
                                 viennacl::op_norm_2 >
    norm_2(viennacl::vector_base<ScalarType> const & v)
    {
       //std::cout << "viennacl .. " << std::endl;
      return viennacl::scalar_expression< const viennacl::vector_base<ScalarType>,
                                          const viennacl::vector_base<ScalarType>,
                                          viennacl::op_norm_2 >(v, v);
    }

    // with vector expression:
    template<typename LHS, typename RHS, typename OP>
    viennacl::scalar_expression<const viennacl::vector_expression<const LHS, const RHS, OP>,
                                const viennacl::vector_expression<const LHS, const RHS, OP>,
                                viennacl::op_norm_2>
    norm_2(viennacl::vector_expression<const LHS, const RHS, OP> const & vector)
    {
      return viennacl::scalar_expression< const viennacl::vector_expression<const LHS, const RHS, OP>,
                                          const viennacl::vector_expression<const LHS, const RHS, OP>,
                                          viennacl::op_norm_2>(vector, vector);
    }


  } // end namespace linalg
} // end namespace viennacl
#endif





