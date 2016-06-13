#ifndef VIENNACL_SCHEDULER_EXECUTE_MATRIX_DISPATCHER_HPP
#define VIENNACL_SCHEDULER_EXECUTE_MATRIX_DISPATCHER_HPP

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


/** @file viennacl/scheduler/execute_matrix_dispatcher.hpp
    @brief Provides wrappers for am(), ambm(), ambm_m(), etc. in viennacl/linalg/matrix_operations.hpp such that scheduler logic is not cluttered with numeric type decutions
*/

#include <assert.h>

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/execute_util.hpp"
#include "viennacl/linalg/matrix_operations.hpp"

namespace viennacl
{
namespace scheduler
{
namespace detail
{

/** @brief Wrapper for viennacl::linalg::av(), taking care of the argument unwrapping */
template<typename ScalarType1>
void am(lhs_rhs_element & mat1,
        lhs_rhs_element const & mat2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
  assert(   mat1.type_family == MATRIX_TYPE_FAMILY && mat2.type_family == MATRIX_TYPE_FAMILY
            && bool("Arguments are not matrix types!"));

  assert(mat1.numeric_type == mat2.numeric_type && bool("Matrices do not have the same scalar type"));

  if (mat1.subtype == DENSE_MATRIX_TYPE)
  {
    switch (mat1.numeric_type)
    {
    case FLOAT_TYPE:
      viennacl::linalg::am(*mat1.matrix_float,
                           *mat2.matrix_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha);
      break;
    case DOUBLE_TYPE:
      viennacl::linalg::am(*mat1.matrix_double,
                           *mat2.matrix_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha);
      break;

    default:
      throw statement_not_supported_exception("Invalid arguments in scheduler when calling am()");
    }
  }
  else
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling am()");
}

/** @brief Wrapper for viennacl::linalg::avbv(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void ambm(lhs_rhs_element & mat1,
          lhs_rhs_element const & mat2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          lhs_rhs_element const & mat3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   mat1.type_family == MATRIX_TYPE_FAMILY
            && mat2.type_family == MATRIX_TYPE_FAMILY
            && mat3.type_family == MATRIX_TYPE_FAMILY
            && bool("Arguments are not matrix types!"));

  assert(   (mat1.subtype == mat2.subtype)
            && (mat2.subtype == mat3.subtype)
            && bool("Matrices do not have the same layout"));

  assert(   (mat1.numeric_type == mat2.numeric_type)
            && (mat2.numeric_type == mat3.numeric_type)
            && bool("Matrices do not have the same scalar type"));

  if (mat1.subtype == DENSE_MATRIX_TYPE)
  {
    switch (mat1.numeric_type)
    {
    case FLOAT_TYPE:
      viennacl::linalg::ambm(*mat1.matrix_float,
                             *mat2.matrix_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                             *mat3.matrix_float, convert_to_float(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
      break;
    case DOUBLE_TYPE:
      viennacl::linalg::ambm(*mat1.matrix_double,
                             *mat2.matrix_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                             *mat3.matrix_double, convert_to_double(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
      break;
    default:
      throw statement_not_supported_exception("Invalid arguments in scheduler when calling ambm()");
    }
  }
  else
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling ambm()");
}

/** @brief Wrapper for viennacl::linalg::avbv_v(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void ambm_m(lhs_rhs_element & mat1,
            lhs_rhs_element const & mat2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
            lhs_rhs_element const & mat3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   mat1.type_family == MATRIX_TYPE_FAMILY
            && mat2.type_family == MATRIX_TYPE_FAMILY
            && mat3.type_family == MATRIX_TYPE_FAMILY
            && bool("Arguments are not matrix types!"));

  assert(   (mat1.subtype == mat2.subtype)
            && (mat2.subtype == mat3.subtype)
            && bool("Matrices do not have the same layout"));

  assert(   (mat1.numeric_type == mat2.numeric_type)
            && (mat2.numeric_type == mat3.numeric_type)
            && bool("Matrices do not have the same scalar type"));

  if (mat1.subtype == DENSE_MATRIX_TYPE)
  {
    switch (mat1.numeric_type)
    {
    case FLOAT_TYPE:
      viennacl::linalg::ambm_m(*mat1.matrix_float,
                               *mat2.matrix_float, convert_to_float(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                               *mat3.matrix_float, convert_to_float(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
      break;
    case DOUBLE_TYPE:
      viennacl::linalg::ambm_m(*mat1.matrix_double,
                               *mat2.matrix_double, convert_to_double(alpha), len_alpha, reciprocal_alpha, flip_sign_alpha,
                               *mat3.matrix_double, convert_to_double(beta),  len_beta,  reciprocal_beta,  flip_sign_beta);
      break;
    default:
      throw statement_not_supported_exception("Invalid arguments in scheduler when calling ambm_m()");
    }
  }
  else
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling ambm_m()");
}

/** @brief Scheduler unwrapper for A =/+=/-= trans(B) */
inline void assign_trans(lhs_rhs_element const & A,
                         lhs_rhs_element const & B)
{
  assert(   A.type_family == MATRIX_TYPE_FAMILY && B.type_family == MATRIX_TYPE_FAMILY
            && bool("Arguments are not matrix types!"));

  assert(A.numeric_type == B.numeric_type && bool("Matrices do not have the same scalar type"));

  if (A.subtype == DENSE_MATRIX_TYPE)
  {
    switch (A.numeric_type)
    {
    case FLOAT_TYPE:
      *A.matrix_float = viennacl::trans(*B.matrix_float);
      break;
    case DOUBLE_TYPE:
      *A.matrix_double = viennacl::trans(*B.matrix_double);
      break;
    default:
      throw statement_not_supported_exception("Invalid arguments in scheduler when calling assign_trans()");
    }
  }
  else
    throw statement_not_supported_exception("Invalid arguments in scheduler when calling assign_trans()");
}

} // namespace detail
} // namespace scheduler
} // namespace viennacl

#endif

