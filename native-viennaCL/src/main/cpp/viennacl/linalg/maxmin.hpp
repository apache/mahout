#ifndef VIENNACL_LINALG_MAXMIN_HPP_
#define VIENNACL_LINALG_MAXMIN_HPP_

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
#include "viennacl/meta/result_of.hpp"

namespace viennacl
{
  //
  // generic norm_inf function
  //   uses tag dispatch to identify which algorithm
  //   should be called
  //
  namespace linalg
  {


    // ----------------------------------------------------
    // STL
    //
    template< typename NumericT >
    NumericT max(std::vector<NumericT> const & v1)
    {
      //std::cout << "stl .. " << std::endl;
      NumericT result = v1[0];
      for (vcl_size_t i=1; i<v1.size(); ++i)
      {
        if (v1[i] > result)
          result = v1[i];
      }

      return result;
    }

    // ----------------------------------------------------
    // VIENNACL
    //
    template< typename ScalarType>
    viennacl::scalar_expression< const viennacl::vector_base<ScalarType>,
                                 const viennacl::vector_base<ScalarType>,
                                 viennacl::op_max >
    max(viennacl::vector_base<ScalarType> const & v1)
    {
       //std::cout << "viennacl .. " << std::endl;
      return viennacl::scalar_expression< const viennacl::vector_base<ScalarType>,
                                          const viennacl::vector_base<ScalarType>,
                                          viennacl::op_max >(v1, v1);
    }

    // with vector expression:
    template<typename LHS, typename RHS, typename OP>
    viennacl::scalar_expression<const viennacl::vector_expression<const LHS, const RHS, OP>,
                                const viennacl::vector_expression<const LHS, const RHS, OP>,
                                viennacl::op_max>
    max(viennacl::vector_expression<const LHS, const RHS, OP> const & vector)
    {
      return viennacl::scalar_expression< const viennacl::vector_expression<const LHS, const RHS, OP>,
                                          const viennacl::vector_expression<const LHS, const RHS, OP>,
                                          viennacl::op_max >(vector, vector);
    }

    // ----------------------------------------------------
    // STL
    //
    template< typename NumericT >
    NumericT min(std::vector<NumericT> const & v1)
    {
      //std::cout << "stl .. " << std::endl;
      NumericT result = v1[0];
      for (vcl_size_t i=1; i<v1.size(); ++i)
      {
        if (v1[i] < result)
          result = v1[i];
      }

      return result;
    }

    // ----------------------------------------------------
    // VIENNACL
    //
    template< typename ScalarType>
    viennacl::scalar_expression< const viennacl::vector_base<ScalarType>,
                                 const viennacl::vector_base<ScalarType>,
                                 viennacl::op_min >
    min(viennacl::vector_base<ScalarType> const & v1)
    {
       //std::cout << "viennacl .. " << std::endl;
      return viennacl::scalar_expression< const viennacl::vector_base<ScalarType>,
                                          const viennacl::vector_base<ScalarType>,
                                          viennacl::op_min >(v1, v1);
    }

    template< typename ScalarType>
    viennacl::scalar_expression< const viennacl::vector_base<ScalarType>,
                                 const viennacl::vector_base<ScalarType>,
                                 viennacl::op_min >
    min(viennacl::vector<ScalarType> const & v1)
    {
       //std::cout << "viennacl .. " << std::endl;
      return viennacl::scalar_expression< const viennacl::vector_base<ScalarType>,
                                          const viennacl::vector_base<ScalarType>,
                                          viennacl::op_min >(v1, v1);
    }

    // with vector expression:
    template<typename LHS, typename RHS, typename OP>
    viennacl::scalar_expression<const viennacl::vector_expression<const LHS, const RHS, OP>,
                                const viennacl::vector_expression<const LHS, const RHS, OP>,
                                viennacl::op_min>
    min(viennacl::vector_expression<const LHS, const RHS, OP> const & vector)
    {
      return viennacl::scalar_expression< const viennacl::vector_expression<const LHS, const RHS, OP>,
                                          const viennacl::vector_expression<const LHS, const RHS, OP>,
                                          viennacl::op_min >(vector, vector);
    }



  } // end namespace linalg
} // end namespace viennacl
#endif





