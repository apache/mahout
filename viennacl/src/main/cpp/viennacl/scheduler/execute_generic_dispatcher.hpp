#ifndef VIENNACL_SCHEDULER_EXECUTE_GENERIC_DISPATCHER_HPP
#define VIENNACL_SCHEDULER_EXECUTE_GENERIC_DISPATCHER_HPP

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


/** @file viennacl/scheduler/execute_generic_dispatcher.hpp
    @brief Provides unified wrappers for the common routines {as(), asbs(), asbs_s()}, {av(), avbv(), avbv_v()}, and {am(), ambm(), ambm_m()} such that scheduler logic is not cluttered with numeric type decutions
*/

#include <assert.h>

#include "viennacl/forwards.h"
#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/execute_util.hpp"
#include "viennacl/scheduler/execute_scalar_dispatcher.hpp"
#include "viennacl/scheduler/execute_vector_dispatcher.hpp"
#include "viennacl/scheduler/execute_matrix_dispatcher.hpp"

namespace viennacl
{
namespace scheduler
{
namespace detail
{

/** @brief Wrapper for viennacl::linalg::av(), taking care of the argument unwrapping */
template<typename ScalarType1>
void ax(lhs_rhs_element & x1,
        lhs_rhs_element const & x2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
  assert(x1.type_family == x2.type_family && bool("Arguments are not of the same type family!"));

  switch (x1.type_family)
  {
  case SCALAR_TYPE_FAMILY:
    detail::as(x1, x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
    break;
  case VECTOR_TYPE_FAMILY:
    detail::av(x1, x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
    break;
  case MATRIX_TYPE_FAMILY:
    detail::am(x1, x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
    break;
  default:
    throw statement_not_supported_exception("Invalid argument in scheduler ax() while dispatching.");
  }
}

/** @brief Wrapper for viennacl::linalg::avbv(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void axbx(lhs_rhs_element & x1,
          lhs_rhs_element const & x2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          lhs_rhs_element const & x3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   x1.type_family == x2.type_family
            && x2.type_family == x3.type_family
            && bool("Arguments are not of the same type family!"));

  switch (x1.type_family)
  {
  case SCALAR_TYPE_FAMILY:
    detail::asbs(x1,
                 x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                 x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case VECTOR_TYPE_FAMILY:
    detail::avbv(x1,
                 x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                 x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case MATRIX_TYPE_FAMILY:
    detail::ambm(x1,
                 x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                 x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  default:
    throw statement_not_supported_exception("Invalid argument in scheduler ax() while dispatching.");
  }
}

/** @brief Wrapper for viennacl::linalg::avbv_v(), taking care of the argument unwrapping */
template<typename ScalarType1, typename ScalarType2>
void axbx_x(lhs_rhs_element & x1,
            lhs_rhs_element const & x2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
            lhs_rhs_element const & x3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  assert(   x1.type_family == x2.type_family
            && x2.type_family == x3.type_family
            && bool("Arguments are not of the same type family!"));

  switch (x1.type_family)
  {
  case SCALAR_TYPE_FAMILY:
    detail::asbs_s(x1,
                   x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                   x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case VECTOR_TYPE_FAMILY:
    detail::avbv_v(x1,
                   x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                   x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  case MATRIX_TYPE_FAMILY:
    detail::ambm_m(x1,
                   x2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                   x3, beta,  len_beta,  reciprocal_beta,  flip_sign_beta);
    break;
  default:
    throw statement_not_supported_exception("Invalid argument in scheduler ax() while dispatching.");
  }
}

} // namespace detail
} // namespace scheduler
} // namespace viennacl

#endif

