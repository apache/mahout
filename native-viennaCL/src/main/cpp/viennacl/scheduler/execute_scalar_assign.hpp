#ifndef VIENNACL_SCHEDULER_EXECUTE_SCALAR_ASSIGN_HPP
#define VIENNACL_SCHEDULER_EXECUTE_SCALAR_ASSIGN_HPP

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


/** @file viennacl/scheduler/execute_scalar_assign.hpp
    @brief Deals with the execution of x = RHS; for a vector x and any compatible right hand side expression RHS.
*/

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/execute_vector_dispatcher.hpp"
#include "viennacl/scheduler/execute_util.hpp"

namespace viennacl
{
namespace scheduler
{

/** @brief Deals with x = RHS where RHS is a vector expression */
inline void execute_scalar_assign_composite(statement const & s, statement_node const & root_node)
{
  statement_node const & leaf = s.array()[root_node.rhs.node_index];
  viennacl::context ctx = detail::extract_context(root_node);

  if (leaf.op.type  == OPERATION_BINARY_INNER_PROD_TYPE) // alpha = inner_prod( (x), (y) ) with x, y being either vectors or expressions
  {
    assert(root_node.lhs.type_family == SCALAR_TYPE_FAMILY && bool("Inner product requires assignment to scalar type!"));

    if (   leaf.lhs.type_family == VECTOR_TYPE_FAMILY
           && leaf.rhs.type_family == VECTOR_TYPE_FAMILY)

    {
      detail::inner_prod_impl(leaf.lhs, leaf.rhs, root_node.lhs);
    }
    else if (   leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY  // temporary for (x)
                && leaf.rhs.type_family == VECTOR_TYPE_FAMILY)
    {
      statement_node new_root_x;

      detail::new_element(new_root_x.lhs, leaf.rhs, ctx);

      new_root_x.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
      new_root_x.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

      new_root_x.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
      new_root_x.rhs.subtype      = INVALID_SUBTYPE;
      new_root_x.rhs.numeric_type = INVALID_NUMERIC_TYPE;
      new_root_x.rhs.node_index   = leaf.lhs.node_index;

      // work on subexpression:
      // TODO: Catch exception, free temporary, then rethrow
      detail::execute_composite(s, new_root_x);

      detail::inner_prod_impl(new_root_x.lhs, leaf.rhs, root_node.lhs);

      detail::delete_element(new_root_x.lhs);
    }
    else if (   leaf.lhs.type_family == VECTOR_TYPE_FAMILY
                && leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY) // temporary for (y)
    {
      statement_node new_root_y;

      detail::new_element(new_root_y.lhs, leaf.lhs, ctx);

      new_root_y.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
      new_root_y.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

      new_root_y.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
      new_root_y.rhs.subtype      = INVALID_SUBTYPE;
      new_root_y.rhs.numeric_type = INVALID_NUMERIC_TYPE;
      new_root_y.rhs.node_index   = leaf.rhs.node_index;

      // work on subexpression:
      // TODO: Catch exception, free temporary, then rethrow
      detail::execute_composite(s, new_root_y);

      detail::inner_prod_impl(leaf.lhs, new_root_y.lhs, root_node.lhs);

      detail::delete_element(new_root_y.lhs);
    }
    else if (   leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY   // temporary for (x)
                && leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY)  // temporary for (y)
    {
      // extract size information from vectors:
      lhs_rhs_element const & temp_node = detail::extract_representative_vector(s, leaf.lhs);

      // temporary for (x)
      statement_node new_root_x;
      detail::new_element(new_root_x.lhs, temp_node, ctx);

      new_root_x.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
      new_root_x.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

      new_root_x.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
      new_root_x.rhs.subtype      = INVALID_SUBTYPE;
      new_root_x.rhs.numeric_type = INVALID_NUMERIC_TYPE;
      new_root_x.rhs.node_index   = leaf.lhs.node_index;

      // work on subexpression:
      // TODO: Catch exception, free temporary, then rethrow
      detail::execute_composite(s, new_root_x);

      // temporary for (y)
      statement_node new_root_y;
      detail::new_element(new_root_y.lhs, temp_node, ctx);

      new_root_y.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
      new_root_y.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

      new_root_y.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
      new_root_y.rhs.subtype      = INVALID_SUBTYPE;
      new_root_y.rhs.numeric_type = INVALID_NUMERIC_TYPE;
      new_root_y.rhs.node_index   = leaf.rhs.node_index;

      // work on subexpression:
      // TODO: Catch exception, free temporary, then rethrow
      detail::execute_composite(s, new_root_y);

      // compute inner product:
      detail::inner_prod_impl(new_root_x.lhs, new_root_y.lhs, root_node.lhs);

      detail::delete_element(new_root_x.lhs);
      detail::delete_element(new_root_y.lhs);
    }
    else
      throw statement_not_supported_exception("Cannot deal with inner product of the provided arguments");
  }
  else if (   leaf.op.type  == OPERATION_UNARY_NORM_1_TYPE
              || leaf.op.type  == OPERATION_UNARY_NORM_2_TYPE
              || leaf.op.type  == OPERATION_UNARY_NORM_INF_TYPE
              || leaf.op.type  == OPERATION_UNARY_MAX_TYPE
              || leaf.op.type  == OPERATION_UNARY_MIN_TYPE)
  {
    assert(root_node.lhs.type_family == SCALAR_TYPE_FAMILY && bool("Inner product requires assignment to scalar type!"));

    if (leaf.lhs.type_family == VECTOR_TYPE_FAMILY)
    {
      detail::norm_impl(leaf.lhs, root_node.lhs, leaf.op.type);
    }
    else if (leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY) //introduce temporary:
    {
      lhs_rhs_element const & temp_node = detail::extract_representative_vector(s, leaf.lhs);

      statement_node new_root_y;

      detail::new_element(new_root_y.lhs, temp_node, ctx);

      new_root_y.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
      new_root_y.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

      new_root_y.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
      new_root_y.rhs.subtype      = INVALID_SUBTYPE;
      new_root_y.rhs.numeric_type = INVALID_NUMERIC_TYPE;
      new_root_y.rhs.node_index   = leaf.lhs.node_index;

      // work on subexpression:
      // TODO: Catch exception, free temporary, then rethrow
      detail::execute_composite(s, new_root_y);

      detail::norm_impl(new_root_y.lhs, root_node.lhs, leaf.op.type);

      detail::delete_element(new_root_y.lhs);
    }
    else
      throw statement_not_supported_exception("Cannot deal with norm_inf of the provided arguments");
  }
  else
    throw statement_not_supported_exception("Unsupported operation for scalar.");
}

}
} //namespace viennacl

#endif

