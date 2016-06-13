#ifndef VIENNACL_SCHEDULER_EXECUTE_UTIL_HPP
#define VIENNACL_SCHEDULER_EXECUTE_UTIL_HPP

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


/** @file viennacl/scheduler/execute_util.hpp
    @brief Provides various utilities for implementing the execution of statements
*/

#include <assert.h>

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/scheduler/forwards.h"

namespace viennacl
{
namespace scheduler
{
namespace detail
{
//
inline lhs_rhs_element const & extract_representative_vector(statement const & s, lhs_rhs_element const & element)
{
  switch (element.type_family)
  {
  case VECTOR_TYPE_FAMILY:
    return element;
  case COMPOSITE_OPERATION_FAMILY:
  {
    statement_node const & leaf = s.array()[element.node_index];

    if (leaf.op.type_family == OPERATION_UNARY_TYPE_FAMILY)
      return extract_representative_vector(s, leaf.lhs);
    switch (leaf.op.type)
    {
    case OPERATION_BINARY_ADD_TYPE:
    case OPERATION_BINARY_SUB_TYPE:
    case OPERATION_BINARY_MULT_TYPE:
    case OPERATION_BINARY_DIV_TYPE:
    case OPERATION_BINARY_ELEMENT_PROD_TYPE:
    case OPERATION_BINARY_ELEMENT_DIV_TYPE:
    case OPERATION_BINARY_ELEMENT_POW_TYPE:
      return extract_representative_vector(s, leaf.lhs);
    case OPERATION_BINARY_MAT_VEC_PROD_TYPE:
      return extract_representative_vector(s, leaf.rhs);
    default:
      throw statement_not_supported_exception("Vector leaf encountered an invalid binary operation!");
    }
  }
  default:
    throw statement_not_supported_exception("Vector leaf encountered an invalid node type!");
  }
}


// helper routines for extracting the scalar type
inline float convert_to_float(float f) { return f; }
inline float convert_to_float(double d) { return static_cast<float>(d); }
inline float convert_to_float(lhs_rhs_element const & el)
{
  if (el.type_family == SCALAR_TYPE_FAMILY && el.subtype == HOST_SCALAR_TYPE && el.numeric_type == FLOAT_TYPE)
    return el.host_float;
  if (el.type_family == SCALAR_TYPE_FAMILY && el.subtype == DEVICE_SCALAR_TYPE && el.numeric_type == FLOAT_TYPE)
    return *el.scalar_float;

  throw statement_not_supported_exception("Cannot convert to float");
}

// helper routines for extracting the scalar type
inline double convert_to_double(float d) { return static_cast<double>(d); }
inline double convert_to_double(double d) { return d; }
inline double convert_to_double(lhs_rhs_element const & el)
{
  if (el.type_family == SCALAR_TYPE_FAMILY && el.subtype == HOST_SCALAR_TYPE && el.numeric_type == DOUBLE_TYPE)
    return el.host_double;
  if (el.type_family == SCALAR_TYPE_FAMILY && el.subtype == DEVICE_SCALAR_TYPE && el.numeric_type == DOUBLE_TYPE)
    return *el.scalar_double;

  throw statement_not_supported_exception("Cannot convert to double");
}

/** @brief Helper routine for extracting the context in which a statement is executed.
  *
  * As all statements are of the type x = EXPR, it is sufficient to extract the context from x.
  */
inline viennacl::context extract_context(statement_node const & root_node)
{
  if (root_node.lhs.type_family == SCALAR_TYPE_FAMILY)
  {
    switch (root_node.lhs.numeric_type)
    {
    case CHAR_TYPE:
      return viennacl::traits::context(*root_node.lhs.scalar_char);
    case UCHAR_TYPE:
      return viennacl::traits::context(*root_node.lhs.scalar_char);
    case SHORT_TYPE:
      return viennacl::traits::context(*root_node.lhs.scalar_short);
    case USHORT_TYPE:
      return viennacl::traits::context(*root_node.lhs.scalar_ushort);
    case INT_TYPE:
      return viennacl::traits::context(*root_node.lhs.scalar_int);
    case UINT_TYPE:
      return viennacl::traits::context(*root_node.lhs.scalar_uint);
    case LONG_TYPE:
      return viennacl::traits::context(*root_node.lhs.scalar_long);
    case ULONG_TYPE:
      return viennacl::traits::context(*root_node.lhs.scalar_ulong);
      //case HALF_TYPE:
      //return viennacl::traits::context(*root_node.lhs.scalar_half);
    case FLOAT_TYPE:
      return viennacl::traits::context(*root_node.lhs.scalar_float);
    case DOUBLE_TYPE:
      return viennacl::traits::context(*root_node.lhs.scalar_double);
    default:
      throw statement_not_supported_exception("Invalid numeric type for extraction of context from scalar");
    }
  }
  else if (root_node.lhs.type_family == VECTOR_TYPE_FAMILY)
  {
    switch (root_node.lhs.numeric_type)
    {
    case CHAR_TYPE:
      return viennacl::traits::context(*root_node.lhs.vector_char);
    case UCHAR_TYPE:
      return viennacl::traits::context(*root_node.lhs.vector_char);
    case SHORT_TYPE:
      return viennacl::traits::context(*root_node.lhs.vector_short);
    case USHORT_TYPE:
      return viennacl::traits::context(*root_node.lhs.vector_ushort);
    case INT_TYPE:
      return viennacl::traits::context(*root_node.lhs.vector_int);
    case UINT_TYPE:
      return viennacl::traits::context(*root_node.lhs.vector_uint);
    case LONG_TYPE:
      return viennacl::traits::context(*root_node.lhs.vector_long);
    case ULONG_TYPE:
      return viennacl::traits::context(*root_node.lhs.vector_ulong);
      //case HALF_TYPE:
      //return viennacl::traits::context(*root_node.lhs.vector_half);
    case FLOAT_TYPE:
      return viennacl::traits::context(*root_node.lhs.vector_float);
    case DOUBLE_TYPE:
      return viennacl::traits::context(*root_node.lhs.vector_double);
    default:
      throw statement_not_supported_exception("Invalid numeric type for extraction of context from vector");
    }
  }
  else if (root_node.lhs.type_family == MATRIX_TYPE_FAMILY)
  {
    switch (root_node.lhs.numeric_type)
    {
    case CHAR_TYPE:
      return viennacl::traits::context(*root_node.lhs.matrix_char);
    case UCHAR_TYPE:
      return viennacl::traits::context(*root_node.lhs.matrix_char);
    case SHORT_TYPE:
      return viennacl::traits::context(*root_node.lhs.matrix_short);
    case USHORT_TYPE:
      return viennacl::traits::context(*root_node.lhs.matrix_ushort);
    case INT_TYPE:
      return viennacl::traits::context(*root_node.lhs.matrix_int);
    case UINT_TYPE:
      return viennacl::traits::context(*root_node.lhs.matrix_uint);
    case LONG_TYPE:
      return viennacl::traits::context(*root_node.lhs.matrix_long);
    case ULONG_TYPE:
      return viennacl::traits::context(*root_node.lhs.matrix_ulong);
      //case HALF_TYPE:
      //return viennacl::traits::context(*root_node.lhs.matrix_half);
    case FLOAT_TYPE:
      return viennacl::traits::context(*root_node.lhs.matrix_float);
    case DOUBLE_TYPE:
      return viennacl::traits::context(*root_node.lhs.matrix_double);
    default:
      throw statement_not_supported_exception("Invalid numeric type for extraction of context from matrix");
    }
  }

  throw statement_not_supported_exception("Invalid type for context extraction");
}


/////////////////// Create/Destory temporary vector ///////////////////////

inline void new_element(lhs_rhs_element & new_elem, lhs_rhs_element const & old_element, viennacl::context const & ctx)
{
  new_elem.type_family  = old_element.type_family;
  new_elem.subtype      = old_element.subtype;
  new_elem.numeric_type = old_element.numeric_type;
  if (new_elem.type_family == SCALAR_TYPE_FAMILY)
  {
    assert(new_elem.subtype == DEVICE_SCALAR_TYPE && bool("Expected a device scalar in root node"));

    switch (new_elem.numeric_type)
    {
    case FLOAT_TYPE:
      new_elem.scalar_float = new viennacl::scalar<float>(0, ctx);
      return;
    case DOUBLE_TYPE:
      new_elem.scalar_double = new viennacl::scalar<double>(0, ctx);
      return;
    default:
      throw statement_not_supported_exception("Invalid vector type for vector construction");
    }
  }
  else if (new_elem.type_family == VECTOR_TYPE_FAMILY)
  {
    assert(new_elem.subtype == DENSE_VECTOR_TYPE && bool("Expected a dense vector in root node"));

    switch (new_elem.numeric_type)
    {
    case FLOAT_TYPE:
      new_elem.vector_float = new viennacl::vector<float>((old_element.vector_float)->size(), ctx);
      return;
    case DOUBLE_TYPE:
      new_elem.vector_double = new viennacl::vector<double>((old_element.vector_float)->size(), ctx);
      return;
    default:
      throw statement_not_supported_exception("Invalid vector type for vector construction");
    }
  }
  else if (new_elem.type_family == MATRIX_TYPE_FAMILY)
  {
    assert( (new_elem.subtype == DENSE_MATRIX_TYPE) && bool("Expected a dense matrix in root node"));

    if (new_elem.subtype == DENSE_MATRIX_TYPE)
    {
      switch (new_elem.numeric_type)
      {
      case FLOAT_TYPE:
        new_elem.matrix_float = new viennacl::matrix_base<float>((old_element.matrix_float)->size1(), (old_element.matrix_float)->size2(), (old_element.matrix_float)->row_major(), ctx);
        return;
      case DOUBLE_TYPE:
        new_elem.matrix_double = new viennacl::matrix_base<double>((old_element.matrix_double)->size1(), (old_element.matrix_double)->size2(), (old_element.matrix_double)->row_major(), ctx);
        return;
      default:
        throw statement_not_supported_exception("Invalid vector type for vector construction");
      }
    }
    else
      throw statement_not_supported_exception("Expected a dense matrix in root node when creating a temporary");
  }
  else
    throw statement_not_supported_exception("Unknown type familty when creating new temporary object");
}

inline void delete_element(lhs_rhs_element & elem)
{
  if (elem.type_family == SCALAR_TYPE_FAMILY)
  {
    switch (elem.numeric_type)
    {
    case FLOAT_TYPE:
      delete elem.scalar_float;
      return;
    case DOUBLE_TYPE:
      delete elem.scalar_double;
      return;
    default:
      throw statement_not_supported_exception("Invalid vector type for vector destruction");
    }
  }
  else if (elem.type_family == VECTOR_TYPE_FAMILY)
  {
    switch (elem.numeric_type)
    {
    case FLOAT_TYPE:
      delete elem.vector_float;
      return;
    case DOUBLE_TYPE:
      delete elem.vector_double;
      return;
    default:
      throw statement_not_supported_exception("Invalid vector type for vector destruction");
    }
  }
  else if (elem.type_family == MATRIX_TYPE_FAMILY)
  {
    if (elem.subtype == DENSE_MATRIX_TYPE)
    {
      switch (elem.numeric_type)
      {
      case FLOAT_TYPE:
        delete elem.matrix_float;
        return;
      case DOUBLE_TYPE:
        delete elem.matrix_double;
        return;
      default:
        throw statement_not_supported_exception("Invalid vector type for vector destruction");
      }
    }
    else
      throw statement_not_supported_exception("Expected a dense matrix in root node when deleting temporary");
  }
  else
    throw statement_not_supported_exception("Unknown type familty when deleting temporary object");
}

} // namespace detail
} // namespace scheduler
} // namespace viennacl

#endif

