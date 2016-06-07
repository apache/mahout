#ifndef VIENNACL_SCHEDULER_EXECUTE_HPP
#define VIENNACL_SCHEDULER_EXECUTE_HPP

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


/** @file viennacl/scheduler/execute.hpp
    @brief Provides the datastructures for dealing with a single statement such as 'x = y + z;'
*/

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"

#include "viennacl/scheduler/execute_scalar_assign.hpp"
#include "viennacl/scheduler/execute_axbx.hpp"
#include "viennacl/scheduler/execute_elementwise.hpp"
#include "viennacl/scheduler/execute_matrix_prod.hpp"
#include "viennacl/scheduler/execute_util.hpp"

namespace viennacl
{
namespace scheduler
{
namespace detail
{
  /** @brief Deals with x = RHS where RHS is an expression and x is either a scalar, a vector, or a matrix */
  void execute_composite(statement const & s, statement_node const & root_node)
  {
    statement::container_type const & expr = s.array();
    viennacl::context ctx = extract_context(root_node);

    statement_node const & leaf = expr[root_node.rhs.node_index];

    if (leaf.op.type  == OPERATION_BINARY_ADD_TYPE || leaf.op.type  == OPERATION_BINARY_SUB_TYPE) // x = (y) +- (z)  where y and z are either data objects or expressions
      execute_axbx(s, root_node);
    else if (leaf.op.type == OPERATION_BINARY_MULT_TYPE || leaf.op.type == OPERATION_BINARY_DIV_TYPE) // x = (y) * / alpha;
    {
      bool scalar_is_temporary = (leaf.rhs.type_family != SCALAR_TYPE_FAMILY);

      statement_node scalar_temp_node;
      if (scalar_is_temporary)
      {
        lhs_rhs_element temp;
        temp.type_family  = SCALAR_TYPE_FAMILY;
        temp.subtype      = DEVICE_SCALAR_TYPE;
        temp.numeric_type = root_node.lhs.numeric_type;
        detail::new_element(scalar_temp_node.lhs, temp, ctx);

        scalar_temp_node.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        scalar_temp_node.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

        scalar_temp_node.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
        scalar_temp_node.rhs.subtype      = INVALID_SUBTYPE;
        scalar_temp_node.rhs.numeric_type = INVALID_NUMERIC_TYPE;
        scalar_temp_node.rhs.node_index   = leaf.rhs.node_index;

        // work on subexpression:
        // TODO: Catch exception, free temporary, then rethrow
        execute_composite(s, scalar_temp_node);
      }

      if (leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY)  //(y) is an expression, so introduce a temporary z = (y):
      {
        statement_node new_root_y;

        new_root_y.lhs.type_family  = root_node.lhs.type_family;
        new_root_y.lhs.subtype      = root_node.lhs.subtype;
        new_root_y.lhs.numeric_type = root_node.lhs.numeric_type;
        detail::new_element(new_root_y.lhs, root_node.lhs, ctx);

        new_root_y.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        new_root_y.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

        new_root_y.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
        new_root_y.rhs.subtype      = INVALID_SUBTYPE;
        new_root_y.rhs.numeric_type = INVALID_NUMERIC_TYPE;
        new_root_y.rhs.node_index   = leaf.lhs.node_index;

        // work on subexpression:
        // TODO: Catch exception, free temporary, then rethrow
        execute_composite(s, new_root_y);

        // now compute x = z * / alpha:
        lhs_rhs_element u = root_node.lhs;
        lhs_rhs_element v = new_root_y.lhs;
        lhs_rhs_element alpha = scalar_is_temporary ? scalar_temp_node.lhs : leaf.rhs;

        bool is_division = (leaf.op.type  == OPERATION_BINARY_DIV_TYPE);
        switch (root_node.op.type)
        {
        case OPERATION_BINARY_ASSIGN_TYPE:
          detail::ax(u,
                     v, alpha, 1, is_division, false);
          break;
        case OPERATION_BINARY_INPLACE_ADD_TYPE:
          detail::axbx(u,
                       u,   1.0, 1, false,       false,
                       v, alpha, 1, is_division, false);
          break;
        case OPERATION_BINARY_INPLACE_SUB_TYPE:
          detail::axbx(u,
                       u,   1.0, 1, false,       false,
                       v, alpha, 1, is_division, true);
          break;
        default:
          throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
        }

        detail::delete_element(new_root_y.lhs);
      }
      else if (leaf.lhs.type_family != COMPOSITE_OPERATION_FAMILY)
      {
        lhs_rhs_element u = root_node.lhs;
        lhs_rhs_element v = leaf.lhs;
        lhs_rhs_element alpha = scalar_is_temporary ? scalar_temp_node.lhs : leaf.rhs;

        bool is_division = (leaf.op.type  == OPERATION_BINARY_DIV_TYPE);
        switch (root_node.op.type)
        {
        case OPERATION_BINARY_ASSIGN_TYPE:
          detail::ax(u,
                     v, alpha, 1, is_division, false);
          break;
        case OPERATION_BINARY_INPLACE_ADD_TYPE:
          detail::axbx(u,
                       u,   1.0, 1, false,       false,
                       v, alpha, 1, is_division, false);
          break;
        case OPERATION_BINARY_INPLACE_SUB_TYPE:
          detail::axbx(u,
                       u,   1.0, 1, false,       false,
                       v, alpha, 1, is_division, true);
          break;
        default:
          throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
        }
      }
      else
        throw statement_not_supported_exception("Unsupported binary operator for OPERATION_BINARY_MULT_TYPE || OPERATION_BINARY_DIV_TYPE on leaf node.");

      // clean up
      if (scalar_is_temporary)
        detail::delete_element(scalar_temp_node.lhs);
    }
    else if (   leaf.op.type == OPERATION_BINARY_INNER_PROD_TYPE
                || leaf.op.type == OPERATION_UNARY_NORM_1_TYPE
                || leaf.op.type == OPERATION_UNARY_NORM_2_TYPE
                || leaf.op.type == OPERATION_UNARY_NORM_INF_TYPE
                || leaf.op.type == OPERATION_UNARY_MAX_TYPE
                || leaf.op.type == OPERATION_UNARY_MIN_TYPE)
      execute_scalar_assign_composite(s, root_node);
    else if (   (leaf.op.type_family == OPERATION_UNARY_TYPE_FAMILY && leaf.op.type != OPERATION_UNARY_TRANS_TYPE)
                || leaf.op.type == OPERATION_BINARY_ELEMENT_PROD_TYPE
                || leaf.op.type == OPERATION_BINARY_ELEMENT_DIV_TYPE
                || leaf.op.type == OPERATION_BINARY_ELEMENT_POW_TYPE) // element-wise operations
      execute_element_composite(s, root_node);
    else if (   leaf.op.type == OPERATION_BINARY_MAT_VEC_PROD_TYPE
                || leaf.op.type == OPERATION_BINARY_MAT_MAT_PROD_TYPE)
      execute_matrix_prod(s, root_node);
    else if (   leaf.op.type == OPERATION_UNARY_TRANS_TYPE)
    {
      if (root_node.op.type == OPERATION_BINARY_ASSIGN_TYPE)
        assign_trans(root_node.lhs, leaf.lhs);
      else // use temporary object:
      {
        statement_node new_root_y;

        new_root_y.lhs.type_family  = root_node.lhs.type_family;
        new_root_y.lhs.subtype      = root_node.lhs.subtype;
        new_root_y.lhs.numeric_type = root_node.lhs.numeric_type;
        detail::new_element(new_root_y.lhs, root_node.lhs, ctx);

        new_root_y.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
        new_root_y.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

        new_root_y.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
        new_root_y.rhs.subtype      = INVALID_SUBTYPE;
        new_root_y.rhs.numeric_type = INVALID_NUMERIC_TYPE;
        new_root_y.rhs.node_index   = root_node.rhs.node_index;

        // work on subexpression:
        // TODO: Catch exception, free temporary, then rethrow
        execute_composite(s, new_root_y);

        // now compute x += temp or x -= temp:
        lhs_rhs_element u = root_node.lhs;
        lhs_rhs_element v = new_root_y.lhs;

        if (root_node.op.type == OPERATION_BINARY_INPLACE_ADD_TYPE)
        {
          detail::axbx(u,
                       u,   1.0, 1, false, false,
                       v,   1.0, 1, false, false);
        }
        else if (root_node.op.type == OPERATION_BINARY_INPLACE_SUB_TYPE)
        {
          detail::axbx(u,
                       u,   1.0, 1, false, false,
                       v,   1.0, 1, false, true);
        }
        else
          throw statement_not_supported_exception("Unsupported binary operator for operation in root node (should be =, +=, or -=)");

        detail::delete_element(new_root_y.lhs);
      }
    }
    else
      throw statement_not_supported_exception("Unsupported binary operator");
  }


  /** @brief Deals with x = y  for a scalar/vector/matrix x, y */
  inline void execute_single(statement const &, statement_node const & root_node)
  {
    lhs_rhs_element u = root_node.lhs;
    lhs_rhs_element v = root_node.rhs;
    switch (root_node.op.type)
    {
    case OPERATION_BINARY_ASSIGN_TYPE:
      detail::ax(u,
                 v, 1.0, 1, false, false);
      break;
    case OPERATION_BINARY_INPLACE_ADD_TYPE:
      detail::axbx(u,
                   u, 1.0, 1, false, false,
                   v, 1.0, 1, false, false);
      break;
    case OPERATION_BINARY_INPLACE_SUB_TYPE:
      detail::axbx(u,
                   u, 1.0, 1, false, false,
                   v, 1.0, 1, false, true);
      break;
    default:
      throw statement_not_supported_exception("Unsupported binary operator for operation in root note (should be =, +=, or -=)");
    }

  }


  inline void execute_impl(statement const & s, statement_node const & root_node)
  {
    if (   root_node.lhs.type_family != SCALAR_TYPE_FAMILY
           && root_node.lhs.type_family != VECTOR_TYPE_FAMILY
           && root_node.lhs.type_family != MATRIX_TYPE_FAMILY)
      throw statement_not_supported_exception("Unsupported lvalue encountered in head node.");

    switch (root_node.rhs.type_family)
    {
    case COMPOSITE_OPERATION_FAMILY:
      execute_composite(s, root_node);
      break;
    case SCALAR_TYPE_FAMILY:
    case VECTOR_TYPE_FAMILY:
    case MATRIX_TYPE_FAMILY:
      execute_single(s, root_node);
      break;
    default:
      throw statement_not_supported_exception("Invalid rvalue encountered in vector assignment");
    }

  }
}

inline void execute(statement const & s)
{
  // simply start execution from the root node:
  detail::execute_impl(s, s.array()[s.root()]);
}


}
} //namespace viennacl

#endif

