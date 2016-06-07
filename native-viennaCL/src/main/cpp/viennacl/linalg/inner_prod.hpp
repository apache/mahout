#ifndef VIENNACL_LINALG_INNER_PROD_HPP_
#define VIENNACL_LINALG_INNER_PROD_HPP_

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

/** @file viennacl/linalg/inner_prod.hpp
    @brief Generic interface for the computation of inner products. See viennacl/linalg/vector_operations.hpp for implementations.
*/

#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/tag_of.hpp"
#include "viennacl/meta/result_of.hpp"

namespace viennacl
{
//
// generic inner_prod function
//   uses tag dispatch to identify which algorithm
//   should be called
//
namespace linalg
{

#ifdef VIENNACL_WITH_ARMADILLO
// ----------------------------------------------------
// Armadillo
//
template<typename NumericT>
NumericT inner_prod(arma::Col<NumericT> const& v1, arma::Col<NumericT> const& v2)
{
  return dot(v1, v2);
}
#endif

#ifdef VIENNACL_WITH_EIGEN
// ----------------------------------------------------
// EIGEN
//
template<typename VectorT1, typename VectorT2>
typename viennacl::enable_if< viennacl::is_eigen< typename viennacl::traits::tag_of< VectorT1 >::type >::value,
                              typename VectorT1::RealScalar>::type
inner_prod(VectorT1 const & v1, VectorT2 const & v2)
{
  //std::cout << "eigen .. " << std::endl;
  return v1.dot(v2);
}
#endif

#ifdef VIENNACL_WITH_MTL4
// ----------------------------------------------------
// MTL4
//
template<typename VectorT1, typename VectorT2>
typename viennacl::enable_if< viennacl::is_mtl4< typename viennacl::traits::tag_of< VectorT1 >::type >::value,
                              typename VectorT1::value_type>::type
inner_prod(VectorT1 const & v1, VectorT2 const & v2)
{
  //std::cout << "mtl4 .. " << std::endl;
  return mtl::dot(v1, v2);
}
#endif

#ifdef VIENNACL_WITH_UBLAS
// ----------------------------------------------------
// UBLAS
//
template<typename VectorT1, typename VectorT2>
typename viennacl::enable_if< viennacl::is_ublas< typename viennacl::traits::tag_of< VectorT1 >::type >::value,
                              typename VectorT1::value_type>::type
inner_prod(VectorT1 const & v1, VectorT2 const & v2)
{
  //std::cout << "ublas .. " << std::endl;
  return boost::numeric::ublas::inner_prod(v1, v2);
}
#endif

// ----------------------------------------------------
// STL
//
template<typename VectorT1, typename VectorT2>
typename viennacl::enable_if< viennacl::is_stl< typename viennacl::traits::tag_of< VectorT1 >::type >::value,
                              typename VectorT1::value_type>::type
inner_prod(VectorT1 const & v1, VectorT2 const & v2)
{
  assert(v1.size() == v2.size() && bool("Vector sizes mismatch"));
  //std::cout << "stl .. " << std::endl;
  typename VectorT1::value_type result = 0;
  for (typename VectorT1::size_type i=0; i<v1.size(); ++i)
    result += v1[i] * v2[i];

  return result;
}

// ----------------------------------------------------
// VIENNACL
//
template<typename NumericT>
viennacl::scalar_expression< const vector_base<NumericT>, const vector_base<NumericT>, viennacl::op_inner_prod >
inner_prod(vector_base<NumericT> const & vector1,
           vector_base<NumericT> const & vector2)
{
  //std::cout << "viennacl .. " << std::endl;
  return viennacl::scalar_expression< const vector_base<NumericT>,
                                      const vector_base<NumericT>,
                                      viennacl::op_inner_prod >(vector1, vector2);
}


// expression on lhs:
template< typename LHS, typename RHS, typename OP, typename NumericT>
viennacl::scalar_expression< const viennacl::vector_expression<LHS, RHS, OP>,
                             const vector_base<NumericT>,
                             viennacl::op_inner_prod >
inner_prod(viennacl::vector_expression<LHS, RHS, OP> const & vector1,
           vector_base<NumericT> const & vector2)
{
  //std::cout << "viennacl .. " << std::endl;
  return viennacl::scalar_expression< const viennacl::vector_expression<LHS, RHS, OP>,
                                      const vector_base<NumericT>,
                                      viennacl::op_inner_prod >(vector1, vector2);
}

// expression on rhs:
template<typename NumericT, typename LHS, typename RHS, typename OP>
viennacl::scalar_expression< const vector_base<NumericT>,
                             const viennacl::vector_expression<LHS, RHS, OP>,
                             viennacl::op_inner_prod >
inner_prod(vector_base<NumericT> const & vector1,
           viennacl::vector_expression<LHS, RHS, OP> const & vector2)
{
  //std::cout << "viennacl .. " << std::endl;
  return viennacl::scalar_expression< const vector_base<NumericT>,
                                      const viennacl::vector_expression<LHS, RHS, OP>,
                                      viennacl::op_inner_prod >(vector1, vector2);
}

// expression on lhs and rhs:
template<typename LHS1, typename RHS1, typename OP1,
         typename LHS2, typename RHS2, typename OP2>
viennacl::scalar_expression< const viennacl::vector_expression<LHS1, RHS1, OP1>,
                             const viennacl::vector_expression<LHS2, RHS2, OP2>,
                             viennacl::op_inner_prod >
inner_prod(viennacl::vector_expression<LHS1, RHS1, OP1> const & vector1,
           viennacl::vector_expression<LHS2, RHS2, OP2> const & vector2)
{
  //std::cout << "viennacl .. " << std::endl;
  return viennacl::scalar_expression< const viennacl::vector_expression<LHS1, RHS1, OP1>,
                                      const viennacl::vector_expression<LHS2, RHS2, OP2>,
                                      viennacl::op_inner_prod >(vector1, vector2);
}


// Multiple inner products:
template<typename NumericT>
viennacl::vector_expression< const vector_base<NumericT>, const vector_tuple<NumericT>, viennacl::op_inner_prod >
inner_prod(vector_base<NumericT> const & x,
           vector_tuple<NumericT> const & y_tuple)
{
  return viennacl::vector_expression< const vector_base<NumericT>,
                                      const vector_tuple<NumericT>,
                                      viennacl::op_inner_prod >(x, y_tuple);
}


} // end namespace linalg
} // end namespace viennacl
#endif


