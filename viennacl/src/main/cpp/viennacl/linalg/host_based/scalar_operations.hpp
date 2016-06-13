#ifndef VIENNACL_LINALG_HOST_BASED_SCALAR_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_SCALAR_OPERATIONS_HPP_

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

/** @file viennacl/linalg/host_based/scalar_operations.hpp
    @brief Implementations of scalar operations using a plain single-threaded or OpenMP-enabled execution on CPU
*/

#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/host_based/common.hpp"

namespace viennacl
{
namespace linalg
{
namespace host_based
{
template<typename ScalarT1,
         typename ScalarT2, typename FactorT>
typename viennacl::enable_if< viennacl::is_scalar<ScalarT1>::value
                              && viennacl::is_scalar<ScalarT2>::value
                              && viennacl::is_any_scalar<FactorT>::value
                            >::type
as(ScalarT1       & s1,
   ScalarT2 const & s2, FactorT const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha)
{
  typedef typename viennacl::result_of::cpu_value_type<ScalarT1>::type        value_type;

  value_type       * data_s1 = detail::extract_raw_pointer<value_type>(s1);
  value_type const * data_s2 = detail::extract_raw_pointer<value_type>(s2);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;
  if (reciprocal_alpha)
    data_alpha = static_cast<value_type>(1) / data_alpha;

  *data_s1 = *data_s2 * data_alpha;
}


template<typename ScalarT1,
         typename ScalarT2, typename FactorT2,
         typename ScalarT3, typename FactorT3>
typename viennacl::enable_if< viennacl::is_scalar<ScalarT1>::value
                              && viennacl::is_scalar<ScalarT2>::value
                              && viennacl::is_scalar<ScalarT3>::value
                              && viennacl::is_any_scalar<FactorT2>::value
                              && viennacl::is_any_scalar<FactorT3>::value
                            >::type
asbs(ScalarT1       & s1,
     ScalarT2 const & s2, FactorT2 const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
     ScalarT3 const & s3, FactorT3 const & beta,  vcl_size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef typename viennacl::result_of::cpu_value_type<ScalarT1>::type        value_type;

  value_type       * data_s1 = detail::extract_raw_pointer<value_type>(s1);
  value_type const * data_s2 = detail::extract_raw_pointer<value_type>(s2);
  value_type const * data_s3 = detail::extract_raw_pointer<value_type>(s3);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;
  if (reciprocal_alpha)
    data_alpha = static_cast<value_type>(1) / data_alpha;

  value_type data_beta = beta;
  if (flip_sign_beta)
    data_beta = -data_beta;
  if (reciprocal_beta)
    data_beta = static_cast<value_type>(1) / data_beta;

  *data_s1 = *data_s2 * data_alpha + *data_s3 * data_beta;
}


template<typename ScalarT1,
         typename ScalarT2, typename FactorT2,
         typename ScalarT3, typename FactorT3>
typename viennacl::enable_if< viennacl::is_scalar<ScalarT1>::value
                              && viennacl::is_scalar<ScalarT2>::value
                              && viennacl::is_scalar<ScalarT3>::value
                              && viennacl::is_any_scalar<FactorT2>::value
                              && viennacl::is_any_scalar<FactorT3>::value
                            >::type
asbs_s(ScalarT1       & s1,
       ScalarT2 const & s2, FactorT2 const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
       ScalarT3 const & s3, FactorT3 const & beta,  vcl_size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef typename viennacl::result_of::cpu_value_type<ScalarT1>::type        value_type;

  value_type       * data_s1 = detail::extract_raw_pointer<value_type>(s1);
  value_type const * data_s2 = detail::extract_raw_pointer<value_type>(s2);
  value_type const * data_s3 = detail::extract_raw_pointer<value_type>(s3);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;
  if (reciprocal_alpha)
    data_alpha = static_cast<value_type>(1) / data_alpha;

  value_type data_beta = beta;
  if (flip_sign_beta)
    data_beta = -data_beta;
  if (reciprocal_beta)
    data_beta = static_cast<value_type>(1) / data_beta;

  *data_s1 += *data_s2 * data_alpha + *data_s3 * data_beta;
}


/** @brief Swaps the contents of two scalars, data is copied
*
* @param s1   The first scalar
* @param s2   The second scalar
*/
template<typename ScalarT1, typename ScalarT2>
typename viennacl::enable_if<    viennacl::is_scalar<ScalarT1>::value
                              && viennacl::is_scalar<ScalarT2>::value
                            >::type
swap(ScalarT1 & s1, ScalarT2 & s2)
{
  typedef typename viennacl::result_of::cpu_value_type<ScalarT1>::type        value_type;

  value_type * data_s1 = detail::extract_raw_pointer<value_type>(s1);
  value_type * data_s2 = detail::extract_raw_pointer<value_type>(s2);

  value_type temp = *data_s2;
  *data_s2 = *data_s1;
  *data_s1 = temp;
}



} //namespace host_based
} //namespace linalg
} //namespace viennacl


#endif
