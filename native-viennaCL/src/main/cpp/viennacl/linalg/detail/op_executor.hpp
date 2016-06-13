#ifndef VIENNACL_LINALG_DETAIL_OP_EXECUTOR_HPP
#define VIENNACL_LINALG_DETAIL_OP_EXECUTOR_HPP

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

/** @file viennacl/linalg/detail/op_executor.hpp
 *
 * @brief Defines the worker class for decomposing an expression tree into small chunks, which can be processed by the predefined operations in ViennaCL.
*/

#include "viennacl/forwards.h"

namespace viennacl
{
namespace linalg
{
namespace detail
{

template<typename NumericT, typename B>
bool op_aliasing(vector_base<NumericT> const & /*lhs*/, B const & /*b*/)
{
  return false;
}

template<typename NumericT>
bool op_aliasing(vector_base<NumericT> const & lhs, vector_base<NumericT> const & b)
{
  return lhs.handle() == b.handle();
}

template<typename NumericT, typename LhsT, typename RhsT, typename OpT>
bool op_aliasing(vector_base<NumericT> const & lhs, vector_expression<const LhsT, const RhsT, OpT> const & rhs)
{
  return op_aliasing(lhs, rhs.lhs()) || op_aliasing(lhs, rhs.rhs());
}


template<typename NumericT, typename B>
bool op_aliasing(matrix_base<NumericT> const & /*lhs*/, B const & /*b*/)
{
  return false;
}

template<typename NumericT>
bool op_aliasing(matrix_base<NumericT> const & lhs, matrix_base<NumericT> const & b)
{
  return lhs.handle() == b.handle();
}

template<typename NumericT, typename LhsT, typename RhsT, typename OpT>
bool op_aliasing(matrix_base<NumericT> const & lhs, matrix_expression<const LhsT, const RhsT, OpT> const & rhs)
{
  return op_aliasing(lhs, rhs.lhs()) || op_aliasing(lhs, rhs.rhs());
}


/** @brief Worker class for decomposing expression templates.
  *
  * @tparam A    Type to which is assigned to
  * @tparam OP   One out of {op_assign, op_inplace_add, op_inplace_sub}
  @ @tparam T    Right hand side of the assignment
*/
template<typename A, typename OP, typename T>
struct op_executor {};

}
}
}

#endif // VIENNACL_LINALG_DETAIL_OP_EXECUTOR_HPP
