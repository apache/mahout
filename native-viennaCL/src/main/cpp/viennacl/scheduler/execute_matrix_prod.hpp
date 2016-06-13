#ifndef VIENNACL_SCHEDULER_EXECUTE_MATRIX_PROD_HPP
#define VIENNACL_SCHEDULER_EXECUTE_MATRIX_PROD_HPP

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


/** @file viennacl/scheduler/execute_matrix_prod.hpp
    @brief Deals with matrix-vector and matrix-matrix products.
*/

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/execute_util.hpp"
#include "viennacl/scheduler/execute_generic_dispatcher.hpp"
#include "viennacl/linalg/vector_operations.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include "viennacl/linalg/sparse_matrix_operations.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/ell_matrix.hpp"
#include "viennacl/hyb_matrix.hpp"

namespace viennacl
{
namespace scheduler
{
namespace detail
{
  inline bool matrix_prod_temporary_required(statement const & s, lhs_rhs_element const & elem)
  {
    if (elem.type_family != COMPOSITE_OPERATION_FAMILY)
      return false;

    // check composite node for being a transposed matrix proxy:
    statement_node const & leaf = s.array()[elem.node_index];
    if (   leaf.op.type == OPERATION_UNARY_TRANS_TYPE && leaf.lhs.type_family == MATRIX_TYPE_FAMILY)
      return false;

    return true;
  }

  inline void matrix_matrix_prod(statement const & s,
                                 lhs_rhs_element result,
                                 lhs_rhs_element const & A,
                                 lhs_rhs_element const & B,
                                 double alpha,
                                 double beta)
  {
    if (A.type_family == MATRIX_TYPE_FAMILY && B.type_family == MATRIX_TYPE_FAMILY)        // C = A * B
    {
      assert(      A.numeric_type == B.numeric_type && bool("Numeric type not the same!"));
      assert( result.numeric_type == B.numeric_type && bool("Numeric type not the same!"));

      if (A.subtype == DENSE_MATRIX_TYPE && B.subtype == DENSE_MATRIX_TYPE && result.subtype == DENSE_MATRIX_TYPE)
      {
        switch (result.numeric_type)
        {
        case FLOAT_TYPE:
          viennacl::linalg::prod_impl(*A.matrix_float, *B.matrix_float, *result.matrix_float, static_cast<float>(alpha), static_cast<float>(beta)); break;
        case DOUBLE_TYPE:
          viennacl::linalg::prod_impl(*A.matrix_double, *B.matrix_double, *result.matrix_double, alpha, beta); break;
        default:
          throw statement_not_supported_exception("Invalid numeric type in matrix-matrix multiplication");
        }
      }

    }
    else if (A.type_family == MATRIX_TYPE_FAMILY && B.type_family == COMPOSITE_OPERATION_FAMILY)        // C = A * B^T
    {
      statement_node const & leaf = s.array()[B.node_index];

      assert(leaf.lhs.type_family  == MATRIX_TYPE_FAMILY && leaf.op.type == OPERATION_UNARY_TRANS_TYPE && bool("Logic error: Argument not a matrix transpose!"));
      assert(leaf.lhs.numeric_type == result.numeric_type && bool("Numeric type not the same!"));
      assert(result.numeric_type == A.numeric_type && bool("Numeric type not the same!"));

      if (A.subtype == DENSE_MATRIX_TYPE && leaf.lhs.subtype == DENSE_MATRIX_TYPE && result.subtype == DENSE_MATRIX_TYPE)
      {
        switch (result.numeric_type)
        {
        case FLOAT_TYPE:
          viennacl::linalg::prod_impl(*A.matrix_float,
                                      viennacl::matrix_expression< const matrix_base<float>,
                                      const matrix_base<float>,
                                      op_trans> (*(leaf.lhs.matrix_float), *(leaf.lhs.matrix_float)),
                                      *result.matrix_float, static_cast<float>(alpha), static_cast<float>(beta)); break;
        case DOUBLE_TYPE:
          viennacl::linalg::prod_impl(*A.matrix_double,
                                      viennacl::matrix_expression< const matrix_base<double>,
                                      const matrix_base<double>,
                                      op_trans>(*(leaf.lhs.matrix_double), *(leaf.lhs.matrix_double)),
                                      *result.matrix_double, alpha, beta); break;
        default:
          throw statement_not_supported_exception("Invalid numeric type in matrix-matrix multiplication");
        }
      }
    }
    else if (A.type_family == COMPOSITE_OPERATION_FAMILY && B.type_family == MATRIX_TYPE_FAMILY)        // C = A^T * B
    {
      statement_node const & leaf = s.array()[A.node_index];

      assert(leaf.lhs.type_family  == MATRIX_TYPE_FAMILY && leaf.op.type == OPERATION_UNARY_TRANS_TYPE && bool("Logic error: Argument not a matrix transpose!"));
      assert(leaf.lhs.numeric_type == result.numeric_type && bool("Numeric type not the same!"));
      assert(result.numeric_type == B.numeric_type && bool("Numeric type not the same!"));

      if (leaf.lhs.subtype == DENSE_MATRIX_TYPE && B.subtype == DENSE_MATRIX_TYPE && result.subtype == DENSE_MATRIX_TYPE)
      {
        switch (result.numeric_type)
        {
        case FLOAT_TYPE:
          viennacl::linalg::prod_impl(viennacl::matrix_expression< const matrix_base<float>,
                                      const matrix_base<float>,
                                      op_trans>(*leaf.lhs.matrix_float, *leaf.lhs.matrix_float),
                                      *B.matrix_float,
                                      *result.matrix_float, static_cast<float>(alpha), static_cast<float>(beta)); break;
        case DOUBLE_TYPE:
          viennacl::linalg::prod_impl(viennacl::matrix_expression< const matrix_base<double>,
                                      const matrix_base<double>,
                                      op_trans>(*leaf.lhs.matrix_double, *leaf.lhs.matrix_double),
                                      *B.matrix_double,
                                      *result.matrix_double, alpha, beta); break;
        default:
          throw statement_not_supported_exception("Invalid numeric type in matrix-matrix multiplication");
        }
      }
    }
    else if (A.type_family == COMPOSITE_OPERATION_FAMILY && B.type_family == COMPOSITE_OPERATION_FAMILY)        // C = A^T * B^T
    {
      statement_node const & leafA = s.array()[A.node_index];
      statement_node const & leafB = s.array()[B.node_index];

      assert(leafA.lhs.type_family  == MATRIX_TYPE_FAMILY && leafA.op.type == OPERATION_UNARY_TRANS_TYPE && bool("Logic error: Argument not a matrix transpose!"));
      assert(leafB.lhs.type_family  == MATRIX_TYPE_FAMILY && leafB.op.type == OPERATION_UNARY_TRANS_TYPE && bool("Logic error: Argument not a matrix transpose!"));
      assert(leafA.lhs.numeric_type == result.numeric_type && bool("Numeric type not the same!"));
      assert(leafB.lhs.numeric_type == result.numeric_type && bool("Numeric type not the same!"));

      if (leafA.lhs.subtype == DENSE_MATRIX_TYPE && leafB.lhs.subtype == DENSE_MATRIX_TYPE && result.subtype == DENSE_MATRIX_TYPE)
      {
        switch (result.numeric_type)
        {
        case FLOAT_TYPE:
          viennacl::linalg::prod_impl(viennacl::matrix_expression< const matrix_base<float>,
                                      const matrix_base<float>,
                                      op_trans>(*leafA.lhs.matrix_float, *leafA.lhs.matrix_float),
                                      viennacl::matrix_expression< const matrix_base<float>,
                                      const matrix_base<float>,
                                      op_trans>(*leafB.lhs.matrix_float, *leafB.lhs.matrix_float),
                                      *result.matrix_float, static_cast<float>(alpha), static_cast<float>(beta)); break;
        case DOUBLE_TYPE:
          viennacl::linalg::prod_impl(viennacl::matrix_expression< const matrix_base<double>,
                                      const matrix_base<double>,
                                      op_trans>(*leafA.lhs.matrix_double, *leafA.lhs.matrix_double),
                                      viennacl::matrix_expression< const matrix_base<double>,
                                      const matrix_base<double>,
                                      op_trans>(*leafB.lhs.matrix_double, *leafB.lhs.matrix_double),
                                      *result.matrix_double, alpha, beta); break;
        default:
          throw statement_not_supported_exception("Invalid numeric type in matrix-matrix multiplication");
        }
      }
    }
    else
      throw statement_not_supported_exception("Matrix-matrix multiplication encountered operands being neither dense matrices nor transposed dense matrices");
  }

  inline void matrix_vector_prod(statement const & s,
                                 lhs_rhs_element result,
                                 lhs_rhs_element const & A,
                                 lhs_rhs_element const & x)
  {
    assert( result.numeric_type == x.numeric_type && bool("Numeric type not the same!"));
    assert( result.type_family == x.type_family && bool("Subtype not the same!"));
    assert( result.subtype == DENSE_VECTOR_TYPE && bool("Result node for matrix-vector product not a vector type!"));

    // deal with transposed product first:
    // switch: trans for A
    if (A.type_family == COMPOSITE_OPERATION_FAMILY) // prod(trans(A), x)
    {
      statement_node const & leaf = s.array()[A.node_index];

      assert(leaf.lhs.type_family  == MATRIX_TYPE_FAMILY && leaf.op.type == OPERATION_UNARY_TRANS_TYPE && bool("Logic error: Argument not a matrix transpose!"));
      assert(leaf.lhs.numeric_type == x.numeric_type && bool("Numeric type not the same!"));

      if (leaf.lhs.subtype == DENSE_MATRIX_TYPE)
      {
        switch (leaf.lhs.numeric_type)
        {
        case FLOAT_TYPE:
          viennacl::linalg::prod_impl(viennacl::matrix_expression< const matrix_base<float>,
                                      const matrix_base<float>,
                                      op_trans>(*leaf.lhs.matrix_float, *leaf.lhs.matrix_float),
                                      *x.vector_float,
                                      *result.vector_float); break;
        case DOUBLE_TYPE:
          viennacl::linalg::prod_impl(viennacl::matrix_expression< const matrix_base<double>,
                                      const matrix_base<double>,
                                      op_trans>(*leaf.lhs.matrix_double, *leaf.lhs.matrix_double),
                                      *x.vector_double,
                                      *result.vector_double); break;
        default:
          throw statement_not_supported_exception("Invalid numeric type in matrix-{matrix,vector} multiplication");
        }
      }
      else
        throw statement_not_supported_exception("Invalid matrix type for transposed matrix-vector product");
    }
    else if (A.subtype == DENSE_MATRIX_TYPE)
    {
      switch (A.numeric_type)
      {
      case FLOAT_TYPE:
        viennacl::linalg::prod_impl(*A.matrix_float, *x.vector_float, *result.vector_float);
        break;
      case DOUBLE_TYPE:
        viennacl::linalg::prod_impl(*A.matrix_double, *x.vector_double, *result.vector_double);
        break;
      default:
        throw statement_not_supported_exception("Invalid numeric type in matrix-{matrix,vector} multiplication");
      }
    }
    else if (A.subtype == COMPRESSED_MATRIX_TYPE)
    {
      switch (A.numeric_type)
      {
      case FLOAT_TYPE:
        viennacl::linalg::prod_impl(*A.compressed_matrix_float, *x.vector_float, float(1), *result.vector_float, float(0));
        break;
      case DOUBLE_TYPE:
        viennacl::linalg::prod_impl(*A.compressed_matrix_double, *x.vector_double, double(1),  *result.vector_double, double(0));
        break;
      default:
        throw statement_not_supported_exception("Invalid numeric type in matrix-{matrix,vector} multiplication");
      }
    }
    else if (A.subtype == COORDINATE_MATRIX_TYPE)
    {
      switch (A.numeric_type)
      {
      case FLOAT_TYPE:
        viennacl::linalg::prod_impl(*A.coordinate_matrix_float, *x.vector_float, float(1), *result.vector_float, float(0));
        break;
      case DOUBLE_TYPE:
        viennacl::linalg::prod_impl(*A.coordinate_matrix_double, *x.vector_double, double(1), *result.vector_double, double(0));
        break;
      default:
        throw statement_not_supported_exception("Invalid numeric type in matrix-{matrix,vector} multiplication");
      }
    }
    else if (A.subtype == ELL_MATRIX_TYPE)
    {
      switch (A.numeric_type)
      {
      case FLOAT_TYPE:
        viennacl::linalg::prod_impl(*A.ell_matrix_float, *x.vector_float, float(1), *result.vector_float, float(0));
        break;
      case DOUBLE_TYPE:
        viennacl::linalg::prod_impl(*A.ell_matrix_double, *x.vector_double, double(1), *result.vector_double, double(0));
        break;
      default:
        throw statement_not_supported_exception("Invalid numeric type in matrix-{matrix,vector} multiplication");
      }
    }
    else if (A.subtype == HYB_MATRIX_TYPE)
    {
      switch (A.numeric_type)
      {
      case FLOAT_TYPE:
        viennacl::linalg::prod_impl(*A.hyb_matrix_float, *x.vector_float, float(1), *result.vector_float, float(0));
        break;
      case DOUBLE_TYPE:
        viennacl::linalg::prod_impl(*A.hyb_matrix_double, *x.vector_double, double(1), *result.vector_double, double(0));
        break;
      default:
        throw statement_not_supported_exception("Invalid numeric type in matrix-{matrix,vector} multiplication");
      }
    }
    else
    {
      std::cout << "A.subtype: " << A.subtype << std::endl;
      throw statement_not_supported_exception("Invalid matrix type for matrix-vector product");
    }
  }

} // namespace detail

inline void execute_matrix_prod(statement const & s, statement_node const & root_node)
{
  statement_node const & leaf = s.array()[root_node.rhs.node_index];
  viennacl::context ctx = detail::extract_context(root_node);

  // Part 1: Check whether temporaries are required //

  statement_node new_root_lhs;
  statement_node new_root_rhs;

  bool lhs_needs_temporary = detail::matrix_prod_temporary_required(s, leaf.lhs);
  bool rhs_needs_temporary = detail::matrix_prod_temporary_required(s, leaf.rhs);

  // check for temporary on lhs:
  if (lhs_needs_temporary)
  {
    std::cout << "Temporary for LHS!" << std::endl;
    detail::new_element(new_root_lhs.lhs, root_node.lhs, ctx);

    new_root_lhs.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
    new_root_lhs.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

    new_root_lhs.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
    new_root_lhs.rhs.subtype      = INVALID_SUBTYPE;
    new_root_lhs.rhs.numeric_type = INVALID_NUMERIC_TYPE;
    new_root_lhs.rhs.node_index   = leaf.lhs.node_index;

    // work on subexpression:
    // TODO: Catch exception, free temporary, then rethrow
    detail::execute_composite(s, new_root_lhs);
  }

  // check for temporary on rhs:
  if (rhs_needs_temporary)
  {
    detail::new_element(new_root_rhs.lhs, root_node.lhs, ctx);

    new_root_rhs.op.type_family = OPERATION_BINARY_TYPE_FAMILY;
    new_root_rhs.op.type        = OPERATION_BINARY_ASSIGN_TYPE;

    new_root_rhs.rhs.type_family  = COMPOSITE_OPERATION_FAMILY;
    new_root_rhs.rhs.subtype      = INVALID_SUBTYPE;
    new_root_rhs.rhs.numeric_type = INVALID_NUMERIC_TYPE;
    new_root_rhs.rhs.node_index   = leaf.rhs.node_index;

    // work on subexpression:
    // TODO: Catch exception, free temporary, then rethrow
    detail::execute_composite(s, new_root_rhs);
  }

  // Part 2: Run the actual computations //

  lhs_rhs_element x = lhs_needs_temporary ? new_root_lhs.lhs : leaf.lhs;
  lhs_rhs_element y = rhs_needs_temporary ? new_root_rhs.lhs : leaf.rhs;

  if (root_node.lhs.type_family == VECTOR_TYPE_FAMILY)
  {
    if (root_node.op.type != OPERATION_BINARY_ASSIGN_TYPE)
    {
      //split y += A*x
      statement_node new_root_z;
      detail::new_element(new_root_z.lhs, root_node.lhs, ctx);

      // compute z = A * x
      detail::matrix_vector_prod(s, new_root_z.lhs, x, y);

      // assignment y = z
      double alpha = 0;
      if (root_node.op.type == OPERATION_BINARY_INPLACE_ADD_TYPE)
        alpha = 1.0;
      else if (root_node.op.type == OPERATION_BINARY_INPLACE_SUB_TYPE)
        alpha = -1.0;
      else
        throw statement_not_supported_exception("Invalid assignment type for matrix-vector product");

      lhs_rhs_element y2 = root_node.lhs;
      detail::axbx(y2,
                   y2, 1.0, 1, false, false,
                   new_root_z.lhs, alpha, 1, false, false);

      detail::delete_element(new_root_z.lhs);
    }
    else
      detail::matrix_vector_prod(s, root_node.lhs, x, y);
  }
  else
  {
    double alpha = (root_node.op.type == OPERATION_BINARY_INPLACE_SUB_TYPE) ? -1.0 : 1.0;
    double beta  = (root_node.op.type != OPERATION_BINARY_ASSIGN_TYPE)      ?  1.0 : 0.0;

    detail::matrix_matrix_prod(s, root_node.lhs, x, y, alpha, beta);
  }

  // Part 3: Clean up //

  if (lhs_needs_temporary)
    detail::delete_element(new_root_lhs.lhs);

  if (rhs_needs_temporary)
    detail::delete_element(new_root_rhs.lhs);
}

} // namespace scheduler
} // namespace viennacl

#endif

