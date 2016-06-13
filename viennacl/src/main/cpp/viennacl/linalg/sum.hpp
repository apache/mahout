#ifndef VIENNACL_LINALG_REDUCE_HPP_
#define VIENNACL_LINALG_REDUCE_HPP_

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

/** @file viennacl/linalg/sum.hpp
    @brief Stub routines for the summation of elements in a vector, or all elements in either a row or column of a dense matrix.
*/

#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/tag_of.hpp"
#include "viennacl/meta/result_of.hpp"

namespace viennacl
{
namespace linalg
{

//
// Sum of vector entries
//

/** @brief User interface function for computing the sum of all elements of a vector */
template<typename NumericT>
viennacl::scalar_expression< const viennacl::vector_base<NumericT>,
                             const viennacl::vector_base<NumericT>,
                             viennacl::op_sum >
sum(viennacl::vector_base<NumericT> const & x)
{
  return viennacl::scalar_expression< const viennacl::vector_base<NumericT>,
                                      const viennacl::vector_base<NumericT>,
                                      viennacl::op_sum >(x, x);
}

/** @brief User interface function for computing the sum of all elements of a vector specified by a vector operation.
 *
 *  Typical use case:   double my_sum = viennacl::linalg::sum(x + y);
 */
template<typename LHS, typename RHS, typename OP>
viennacl::scalar_expression<const viennacl::vector_expression<const LHS, const RHS, OP>,
                            const viennacl::vector_expression<const LHS, const RHS, OP>,
                            viennacl::op_sum>
sum(viennacl::vector_expression<const LHS, const RHS, OP> const & x)
{
  return viennacl::scalar_expression< const viennacl::vector_expression<const LHS, const RHS, OP>,
                                      const viennacl::vector_expression<const LHS, const RHS, OP>,
                                      viennacl::op_sum >(x, x);
}


//
// Sum of entries in rows of a matrix
//

/** @brief User interface function for computing the sum of all elements of each row of a matrix. */
template<typename NumericT>
viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                             const viennacl::matrix_base<NumericT>,
                             viennacl::op_row_sum >
row_sum(viennacl::matrix_base<NumericT> const & A)
{
  return viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                      const viennacl::matrix_base<NumericT>,
                                      viennacl::op_row_sum >(A, A);
}

/** @brief User interface function for computing the sum of all elements of each row of a matrix specified by a matrix operation.
 *
 *  Typical use case:   vector<double> my_sums = viennacl::linalg::row_sum(A + B);
 */
template<typename LHS, typename RHS, typename OP>
viennacl::vector_expression<const viennacl::matrix_expression<const LHS, const RHS, OP>,
                            const viennacl::matrix_expression<const LHS, const RHS, OP>,
                            viennacl::op_row_sum>
row_sum(viennacl::matrix_expression<const LHS, const RHS, OP> const & A)
{
  return viennacl::vector_expression< const viennacl::matrix_expression<const LHS, const RHS, OP>,
                                      const viennacl::matrix_expression<const LHS, const RHS, OP>,
                                      viennacl::op_row_sum >(A, A);
}


//
// Sum of entries in columns of a matrix
//

/** @brief User interface function for computing the sum of all elements of each column of a matrix. */
template<typename NumericT>
viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                             const viennacl::matrix_base<NumericT>,
                             viennacl::op_col_sum >
column_sum(viennacl::matrix_base<NumericT> const & A)
{
  return viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                      const viennacl::matrix_base<NumericT>,
                                      viennacl::op_col_sum >(A, A);
}

/** @brief User interface function for computing the sum of all elements of each column of a matrix specified by a matrix operation.
 *
 *  Typical use case:   vector<double> my_sums = viennacl::linalg::column_sum(A + B);
 */
template<typename LHS, typename RHS, typename OP>
viennacl::vector_expression<const viennacl::matrix_expression<const LHS, const RHS, OP>,
                            const viennacl::matrix_expression<const LHS, const RHS, OP>,
                            viennacl::op_col_sum>
column_sum(viennacl::matrix_expression<const LHS, const RHS, OP> const & A)
{
  return viennacl::vector_expression< const viennacl::matrix_expression<const LHS, const RHS, OP>,
                                      const viennacl::matrix_expression<const LHS, const RHS, OP>,
                                      viennacl::op_col_sum >(A, A);
}


} // end namespace linalg
} // end namespace viennacl
#endif


