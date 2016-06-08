#ifndef VIENNACL_SCHEDULER_EXECUTE_AXBX_HPP
#define VIENNACL_SCHEDULER_EXECUTE_AXBX_HPP

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


/** @file viennacl/scheduler/execute_axbx.hpp
    @brief Provides the datastructures for dealing with statements of the type x = (y) +- (z)
*/

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"

#include "viennacl/scheduler/execute_scalar_assign.hpp"
#include "viennacl/scheduler/execute_generic_dispatcher.hpp"

namespace viennacl
{
namespace scheduler
{
namespace detail
{

/** @brief Deals with x = (y) +- (z)  where y and z are either data objects or expressions */
inline void execute_axbx(statement const & s, statement_node const & root_node)
{
  statement::container_type const & expr = s.array();
  viennacl::context ctx = detail::extract_context(root_node);

  statement_node const & leaf = expr[root_node.rhs.node_index];

  if (leaf.op.type  == OPERATION_BINARY_ADD_TYPE || leaf.op.type  == OPERATION_BINARY_SUB_TYPE) // x = (y) +- (z)  where y and z are either data objects or expressions
  {
    bool flip_sign_z = (leaf.op.type  == OPERATION_BINARY_SUB_TYPE);

    if (   leaf.lhs.type_family != COMPOSITE_OPERATION_FAMILY
           && leaf.rhs.type_family != COMPOSITE_OPERATION_FAMILY)
    {
      lhs_rhs_element u = root_node.lhs;
      lhs_rhs_element v = leaf.lhs;
      lhs_rhs_element w = leaf.rhs;
      switch (root_node.op.type)
      {
      case OPERATION_BINARY_ASSIGN_TYPE:
        detail::axbx(u,
                     v, 1.0, 1, false, false,
                     w, 1.0, 1, false, flip_sign_z);
        break;
      case OPERATION_BINARY_INPLACE_ADD_TYPE:
        detail::axbx_x(u,
                       v, 1.0, 1, false, false,
                       w, 1.0, 1, false, flip_sign_z);
        break;
      case OPERATION_BINARY_INPLACE_SUB_TYPE:
        detail::axbx_x(u,
                       v, 1.0, 1, false, true,
                       w, 1.0, 1, false, !flip_sign_z);
        break;
      default:
        throw statement_not_supported_exception("Unsupported binary operator for operation in root note (should be =, +=, or -=)");
      }
    }
    else if (  leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY
               && leaf.rhs.type_family != COMPOSITE_OPERATION_FAMILY) // x = (y) + z, y being a subtree itself, z being a scalar, vector, or matrix
    {
      statement_node const & y = expr[leaf.lhs.node_index];

      if (y.op.type_family == OPERATION_BINARY_TYPE_FAMILY || y.op.type_family == OPERATION_UNARY_TYPE_FAMILY)
      {
        // y might be  'v * alpha' or 'v / alpha' with {scalar|vector|matrix} v
        if (   (y.op.type == OPERATION_BINARY_MULT_TYPE || y.op.type == OPERATION_BINARY_DIV_TYPE)
               &&  y.lhs.type_family != COMPOSITE_OPERATION_FAMILY
               &&  y.rhs.type_family == SCALAR_TYPE_FAMILY)
        {
          lhs_rhs_element u = root_node.lhs;
          lhs_rhs_element v = y.lhs;
          lhs_rhs_element w = leaf.rhs;
          lhs_rhs_element alpha = y.rhs;

          bool is_division = (y.op.type == OPERATION_BINARY_DIV_TYPE);
          switch (root_node.op.type)
          {
          case OPERATION_BINARY_ASSIGN_TYPE:
            detail::axbx(u,
                         v, alpha, 1, is_division, false,
                         w,   1.0, 1, false,       flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_ADD_TYPE:
            detail::axbx_x(u,
                           v, alpha, 1, is_division, false,
                           w,   1.0, 1, false,       flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_SUB_TYPE:
            detail::axbx_x(u,
                           v, alpha, 1, is_division, true,
                           w,   1.0, 1, false,       !flip_sign_z);
            break;
          default:
            throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
          }
        }
        else // no built-in kernel, we use a temporary.
        {
          statement_node new_root_y;

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

          // now add:
          lhs_rhs_element u = root_node.lhs;
          lhs_rhs_element v = new_root_y.lhs;
          lhs_rhs_element w = leaf.rhs;
          switch (root_node.op.type)
          {
          case OPERATION_BINARY_ASSIGN_TYPE:
            detail::axbx(u,
                         v, 1.0, 1, false, false,
                         w, 1.0, 1, false, flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_ADD_TYPE:
            detail::axbx_x(u,
                           v, 1.0, 1, false, false,
                           w, 1.0, 1, false, flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_SUB_TYPE:
            detail::axbx_x(u,
                           v, 1.0, 1, false, true,
                           w, 1.0, 1, false, !flip_sign_z);
            break;
          default:
            throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
          }

          detail::delete_element(new_root_y.lhs);
        }
      }
      else
        throw statement_not_supported_exception("Cannot deal with unknown non-unary and non-binary operations on vectors");

    }
    else if (  leaf.lhs.type_family != COMPOSITE_OPERATION_FAMILY
               && leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY) // x = y + (z), y being vector, z being a subtree itself
    {
      statement_node const & z = expr[leaf.rhs.node_index];

      if (z.op.type_family == OPERATION_BINARY_TYPE_FAMILY || z.op.type_family == OPERATION_UNARY_TYPE_FAMILY)
      {
        // z might be  'v * alpha' or 'v / alpha' with vector v
        if (   (z.op.type == OPERATION_BINARY_MULT_TYPE || z.op.type == OPERATION_BINARY_DIV_TYPE)
               &&  z.lhs.type_family != COMPOSITE_OPERATION_FAMILY
               &&  z.rhs.type_family == SCALAR_TYPE_FAMILY)
        {
          lhs_rhs_element u = root_node.lhs;
          lhs_rhs_element v = leaf.lhs;
          lhs_rhs_element w = z.lhs;
          lhs_rhs_element beta = z.rhs;

          bool is_division = (z.op.type == OPERATION_BINARY_DIV_TYPE);
          switch (root_node.op.type)
          {
          case OPERATION_BINARY_ASSIGN_TYPE:
            detail::axbx(u,
                         v,  1.0, 1, false, false,
                         w, beta, 1, is_division, flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_ADD_TYPE:
            detail::axbx_x(u,
                           v,  1.0, 1, false, false,
                           w, beta, 1, is_division, flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_SUB_TYPE:
            detail::axbx_x(u,
                           v,  1.0, 1, false, true,
                           w, beta, 1, is_division, !flip_sign_z);
            break;
          default:
            throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
          }
        }
        else // no built-in kernel, we use a temporary.
        {
          statement_node new_root_z;

          detail::new_element(new_root_z.lhs, root_node.lhs, ctx);

          new_root_z.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
          new_root_z.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

          new_root_z.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
          new_root_z.rhs.subtype      = INVALID_SUBTYPE;
          new_root_z.rhs.numeric_type = INVALID_NUMERIC_TYPE;
          new_root_z.rhs.node_index   = leaf.rhs.node_index;

          // work on subexpression:
          // TODO: Catch exception, free temporary, then rethrow
          execute_composite(s, new_root_z);

          // now add:
          lhs_rhs_element u = root_node.lhs;
          lhs_rhs_element v = leaf.lhs;
          lhs_rhs_element w = new_root_z.lhs;
          switch (root_node.op.type)
          {
          case OPERATION_BINARY_ASSIGN_TYPE:
            detail::axbx(u,
                         v, 1.0, 1, false, false,
                         w, 1.0, 1, false, flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_ADD_TYPE:
            detail::axbx_x(u,
                           v, 1.0, 1, false, false,
                           w, 1.0, 1, false, flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_SUB_TYPE:
            detail::axbx_x(u,
                           v, 1.0, 1, false, true,
                           w, 1.0, 1, false, !flip_sign_z);
            break;
          default:
            throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
          }

          detail::delete_element(new_root_z.lhs);
        }
      }
      else
        throw statement_not_supported_exception("Cannot deal with unknown non-unary and non-binary operations on vectors");

    }
    else if (  leaf.lhs.type_family == COMPOSITE_OPERATION_FAMILY
               && leaf.rhs.type_family == COMPOSITE_OPERATION_FAMILY) // x = (y) + (z), y and z being subtrees
    {
      statement_node const & y = expr[leaf.lhs.node_index];
      statement_node const & z = expr[leaf.rhs.node_index];

      if (   (y.op.type_family == OPERATION_BINARY_TYPE_FAMILY || y.op.type_family == OPERATION_UNARY_TYPE_FAMILY)
          && (z.op.type_family == OPERATION_BINARY_TYPE_FAMILY || z.op.type_family == OPERATION_UNARY_TYPE_FAMILY))
      {
        // z might be  'v * alpha' or 'v / alpha' with vector v
        if (   (y.op.type == OPERATION_BINARY_MULT_TYPE || y.op.type == OPERATION_BINARY_DIV_TYPE)
               &&  y.lhs.type_family != COMPOSITE_OPERATION_FAMILY
               &&  y.rhs.type_family == SCALAR_TYPE_FAMILY
               && (z.op.type == OPERATION_BINARY_MULT_TYPE || z.op.type == OPERATION_BINARY_DIV_TYPE)
               &&  z.lhs.type_family != COMPOSITE_OPERATION_FAMILY
               &&  z.rhs.type_family == SCALAR_TYPE_FAMILY)
        {
          lhs_rhs_element u = root_node.lhs;
          lhs_rhs_element v = y.lhs;
          lhs_rhs_element w = z.lhs;
          lhs_rhs_element alpha = y.rhs;
          lhs_rhs_element beta  = z.rhs;

          bool is_division_y = (y.op.type == OPERATION_BINARY_DIV_TYPE);
          bool is_division_z = (z.op.type == OPERATION_BINARY_DIV_TYPE);
          switch (root_node.op.type)
          {
          case OPERATION_BINARY_ASSIGN_TYPE:
            detail::axbx(u,
                         v, alpha, 1, is_division_y, false,
                         w,  beta, 1, is_division_z, flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_ADD_TYPE:
            detail::axbx_x(u,
                           v, alpha, 1, is_division_y, false,
                           w,  beta, 1, is_division_z, flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_SUB_TYPE:
            detail::axbx_x(u,
                           v, alpha, 1, is_division_y, true,
                           w,  beta, 1, is_division_z, !flip_sign_z);
            break;
          default:
            throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
          }
        }
        else // no built-in kernel, we use a temporary.
        {
          statement_node new_root_y;

          detail::new_element(new_root_y.lhs, root_node.lhs, ctx);

          new_root_y.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
          new_root_y.op.type   = OPERATION_BINARY_ASSIGN_TYPE;

          new_root_y.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
          new_root_y.rhs.subtype      = INVALID_SUBTYPE;
          new_root_y.rhs.numeric_type = INVALID_NUMERIC_TYPE;
          new_root_y.rhs.node_index   = leaf.lhs.node_index;

          // work on subexpression:
          // TODO: Catch exception, free temporary, then rethrow
          execute_composite(s, new_root_y);

          statement_node new_root_z;

          detail::new_element(new_root_z.lhs, root_node.lhs, ctx);

          new_root_z.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
          new_root_z.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

          new_root_z.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
          new_root_z.rhs.subtype      = INVALID_SUBTYPE;
          new_root_z.rhs.numeric_type = INVALID_NUMERIC_TYPE;
          new_root_z.rhs.node_index   = leaf.rhs.node_index;

          // work on subexpression:
          // TODO: Catch exception, free temporaries, then rethrow
          execute_composite(s, new_root_z);

          // now add:
          lhs_rhs_element u = root_node.lhs;
          lhs_rhs_element v = new_root_y.lhs;
          lhs_rhs_element w = new_root_z.lhs;

          switch (root_node.op.type)
          {
          case OPERATION_BINARY_ASSIGN_TYPE:
            detail::axbx(u,
                         v, 1.0, 1, false, false,
                         w, 1.0, 1, false, flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_ADD_TYPE:
            detail::axbx_x(u,
                           v, 1.0, 1, false, false,
                           w, 1.0, 1, false, flip_sign_z);
            break;
          case OPERATION_BINARY_INPLACE_SUB_TYPE:
            detail::axbx_x(u,
                           v, 1.0, 1, false, true,
                           w, 1.0, 1, false, !flip_sign_z);
            break;
          default:
            throw statement_not_supported_exception("Unsupported binary operator for vector operation in root note (should be =, +=, or -=)");
          }

          detail::delete_element(new_root_y.lhs);
          detail::delete_element(new_root_z.lhs);
        }
      }
      else
        throw statement_not_supported_exception("Cannot deal with unknown non-unary and non-binary operations on vectors");
    }
    else
      throw statement_not_supported_exception("Cannot deal with addition of vectors");
  }
  else
    throw statement_not_supported_exception("Unsupported binary operator for vector operations");
}

} // namespace detail
} // namespace scheduler
} // namespace viennacl

#endif

