#ifndef VIENNACL_LINALG_HOST_BASED_VECTOR_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_VECTOR_OPERATIONS_HPP_

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

/** @file viennacl/linalg/host_based/vector_operations.hpp
    @brief Implementations of vector operations using a plain single-threaded or OpenMP-enabled execution on CPU
*/

#include <cmath>
#include <algorithm>  //for std::max and std::min

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/linalg/host_based/common.hpp"
#include "viennacl/linalg/detail/op_applier.hpp"
#include "viennacl/traits/stride.hpp"

#ifdef VIENNACL_WITH_OPENMP
#include <omp.h>
#endif

// Minimum vector size for using OpenMP on vector operations:
#ifndef VIENNACL_OPENMP_VECTOR_MIN_SIZE
  #define VIENNACL_OPENMP_VECTOR_MIN_SIZE  5000
#endif

namespace viennacl
{
namespace linalg
{
namespace host_based
{
namespace detail
{
  template<typename NumericT>
  NumericT flip_sign(NumericT val) { return -val; }
  inline unsigned long  flip_sign(unsigned long  val) { return val; }
  inline unsigned int   flip_sign(unsigned int   val) { return val; }
  inline unsigned short flip_sign(unsigned short val) { return val; }
  inline unsigned char  flip_sign(unsigned char  val) { return val; }
}

//
// Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
//
template<typename DestNumericT, typename SrcNumericT>
void convert(vector_base<DestNumericT> & dest, vector_base<SrcNumericT> const & src)
{
  DestNumericT      * data_dest = detail::extract_raw_pointer<DestNumericT>(dest);
  SrcNumericT const * data_src  = detail::extract_raw_pointer<SrcNumericT>(src);

  vcl_size_t start_dest = viennacl::traits::start(dest);
  vcl_size_t inc_dest   = viennacl::traits::stride(dest);
  vcl_size_t size_dest  = viennacl::traits::size(dest);

  vcl_size_t start_src = viennacl::traits::start(src);
  vcl_size_t inc_src   = viennacl::traits::stride(src);

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for if (size_dest > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
  for (long i = 0; i < static_cast<long>(size_dest); ++i)
    data_dest[static_cast<vcl_size_t>(i)*inc_dest+start_dest] = static_cast<DestNumericT>(data_src[static_cast<vcl_size_t>(i)*inc_src+start_src]);
}

template<typename NumericT, typename ScalarT1>
void av(vector_base<NumericT> & vec1,
        vector_base<NumericT> const & vec2, ScalarT1 const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha)
{
  typedef NumericT        value_type;

  value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = detail::flip_sign(data_alpha);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  vcl_size_t start2 = viennacl::traits::start(vec2);
  vcl_size_t inc2   = viennacl::traits::stride(vec2);

  if (reciprocal_alpha)
  {
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
    for (long i = 0; i < static_cast<long>(size1); ++i)
      data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] = data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] / data_alpha;
  }
  else
  {
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
    for (long i = 0; i < static_cast<long>(size1); ++i)
      data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] = data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] * data_alpha;
  }
}


template<typename NumericT, typename ScalarT1, typename ScalarT2>
void avbv(vector_base<NumericT> & vec1,
          vector_base<NumericT> const & vec2, ScalarT1 const & alpha, vcl_size_t /* len_alpha */, bool reciprocal_alpha, bool flip_sign_alpha,
          vector_base<NumericT> const & vec3, ScalarT2 const & beta,  vcl_size_t /* len_beta */,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef NumericT      value_type;

  value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);
  value_type const * data_vec3 = detail::extract_raw_pointer<value_type>(vec3);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = detail::flip_sign(data_alpha);

  value_type data_beta = beta;
  if (flip_sign_beta)
    data_beta = detail::flip_sign(data_beta);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  vcl_size_t start2 = viennacl::traits::start(vec2);
  vcl_size_t inc2   = viennacl::traits::stride(vec2);

  vcl_size_t start3 = viennacl::traits::start(vec3);
  vcl_size_t inc3   = viennacl::traits::stride(vec3);

  if (reciprocal_alpha)
  {
    if (reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] = data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] / data_alpha + data_vec3[static_cast<vcl_size_t>(i)*inc3+start3] / data_beta;
    }
    else
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] = data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] / data_alpha + data_vec3[static_cast<vcl_size_t>(i)*inc3+start3] * data_beta;
    }
  }
  else
  {
    if (reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] = data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] * data_alpha + data_vec3[static_cast<vcl_size_t>(i)*inc3+start3] / data_beta;
    }
    else
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] = data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] * data_alpha + data_vec3[static_cast<vcl_size_t>(i)*inc3+start3] * data_beta;
    }
  }
}


template<typename NumericT, typename ScalarT1, typename ScalarT2>
void avbv_v(vector_base<NumericT> & vec1,
            vector_base<NumericT> const & vec2, ScalarT1 const & alpha, vcl_size_t /*len_alpha*/, bool reciprocal_alpha, bool flip_sign_alpha,
            vector_base<NumericT> const & vec3, ScalarT2 const & beta,  vcl_size_t /*len_beta*/,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef NumericT        value_type;

  value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);
  value_type const * data_vec3 = detail::extract_raw_pointer<value_type>(vec3);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = detail::flip_sign(data_alpha);

  value_type data_beta = beta;
  if (flip_sign_beta)
    data_beta = detail::flip_sign(data_beta);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  vcl_size_t start2 = viennacl::traits::start(vec2);
  vcl_size_t inc2   = viennacl::traits::stride(vec2);

  vcl_size_t start3 = viennacl::traits::start(vec3);
  vcl_size_t inc3   = viennacl::traits::stride(vec3);

  if (reciprocal_alpha)
  {
    if (reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] += data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] / data_alpha + data_vec3[static_cast<vcl_size_t>(i)*inc3+start3] / data_beta;
    }
    else
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] += data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] / data_alpha + data_vec3[static_cast<vcl_size_t>(i)*inc3+start3] * data_beta;
    }
  }
  else
  {
    if (reciprocal_beta)
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] += data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] * data_alpha + data_vec3[static_cast<vcl_size_t>(i)*inc3+start3] / data_beta;
    }
    else
    {
#ifdef VIENNACL_WITH_OPENMP
      #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
      for (long i = 0; i < static_cast<long>(size1); ++i)
        data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] += data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] * data_alpha + data_vec3[static_cast<vcl_size_t>(i)*inc3+start3] * data_beta;
    }
  }
}




/** @brief Assign a constant value to a vector (-range/-slice)
*
* @param vec1   The vector to which the value should be assigned
* @param alpha  The value to be assigned
* @param up_to_internal_size  Specifies whether alpha should also be written to padded memory (mostly used for clearing the whole buffer).
*/
template<typename NumericT>
void vector_assign(vector_base<NumericT> & vec1, const NumericT & alpha, bool up_to_internal_size = false)
{
  typedef NumericT       value_type;

  value_type * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);
  vcl_size_t loop_bound  = up_to_internal_size ? vec1.internal_size() : size1;  //Note: Do NOT use traits::internal_size() here, because vector proxies don't require padding.

  value_type data_alpha = static_cast<value_type>(alpha);

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for if (loop_bound > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
  for (long i = 0; i < static_cast<long>(loop_bound); ++i)
    data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] = data_alpha;
}


/** @brief Swaps the contents of two vectors, data is copied
*
* @param vec1   The first vector (or -range, or -slice)
* @param vec2   The second vector (or -range, or -slice)
*/
template<typename NumericT>
void vector_swap(vector_base<NumericT> & vec1, vector_base<NumericT> & vec2)
{
  typedef NumericT      value_type;

  value_type * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  vcl_size_t start2 = viennacl::traits::start(vec2);
  vcl_size_t inc2   = viennacl::traits::stride(vec2);

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
  for (long i = 0; i < static_cast<long>(size1); ++i)
  {
    value_type temp = data_vec2[static_cast<vcl_size_t>(i)*inc2+start2];
    data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] = data_vec1[static_cast<vcl_size_t>(i)*inc1+start1];
    data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] = temp;
  }
}


///////////////////////// Elementwise operations /////////////

/** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
*
* @param vec1   The result vector (or -range, or -slice)
* @param proxy  The proxy object holding v2, v3 and the operation
*/
template<typename NumericT, typename OpT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_binary<OpT> > const & proxy)
{
  typedef NumericT                                           value_type;
  typedef viennacl::linalg::detail::op_applier<op_element_binary<OpT> >    OpFunctor;

  value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(proxy.lhs());
  value_type const * data_vec3 = detail::extract_raw_pointer<value_type>(proxy.rhs());

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  vcl_size_t start2 = viennacl::traits::start(proxy.lhs());
  vcl_size_t inc2   = viennacl::traits::stride(proxy.lhs());

  vcl_size_t start3 = viennacl::traits::start(proxy.rhs());
  vcl_size_t inc3   = viennacl::traits::stride(proxy.rhs());

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
  for (long i = 0; i < static_cast<long>(size1); ++i)
    OpFunctor::apply(data_vec1[static_cast<vcl_size_t>(i)*inc1+start1], data_vec2[static_cast<vcl_size_t>(i)*inc2+start2], data_vec3[static_cast<vcl_size_t>(i)*inc3+start3]);
}

/** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
*
* @param vec1   The result vector (or -range, or -slice)
* @param proxy  The proxy object holding v2, v3 and the operation
*/
template<typename NumericT, typename OpT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<OpT> > const & proxy)
{
  typedef NumericT      value_type;
  typedef viennacl::linalg::detail::op_applier<op_element_unary<OpT> >    OpFunctor;

  value_type       * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(proxy.lhs());

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  vcl_size_t start2 = viennacl::traits::start(proxy.lhs());
  vcl_size_t inc2   = viennacl::traits::stride(proxy.lhs());

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
  for (long i = 0; i < static_cast<long>(size1); ++i)
    OpFunctor::apply(data_vec1[static_cast<vcl_size_t>(i)*inc1+start1], data_vec2[static_cast<vcl_size_t>(i)*inc2+start2]);
}


///////////////////////// Norms and inner product ///////////////////


//implementation of inner product:

namespace detail
{

// the following circumvents problems when trying to use a variable of template parameter type for a reduction.
// Such a behavior is not covered by the OpenMP standard, hence we manually apply some preprocessor magic to resolve the problem.
// See https://github.com/viennacl/viennacl-dev/issues/112 for a detailed explanation and discussion.

#define VIENNACL_INNER_PROD_IMPL_1(RESULTSCALART, TEMPSCALART) \
  inline RESULTSCALART inner_prod_impl(RESULTSCALART const * data_vec1, vcl_size_t start1, vcl_size_t inc1, vcl_size_t size1, \
                                       RESULTSCALART const * data_vec2, vcl_size_t start2, vcl_size_t inc2) { \
    TEMPSCALART temp = 0;

#define VIENNACL_INNER_PROD_IMPL_2(RESULTSCALART) \
    for (long i = 0; i < static_cast<long>(size1); ++i) \
      temp += data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] * data_vec2[static_cast<vcl_size_t>(i)*inc2+start2]; \
    return static_cast<RESULTSCALART>(temp); \
  }

// char
VIENNACL_INNER_PROD_IMPL_1(char, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_INNER_PROD_IMPL_2(char)

VIENNACL_INNER_PROD_IMPL_1(unsigned char, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_INNER_PROD_IMPL_2(unsigned char)


// short
VIENNACL_INNER_PROD_IMPL_1(short, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_INNER_PROD_IMPL_2(short)

VIENNACL_INNER_PROD_IMPL_1(unsigned short, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_INNER_PROD_IMPL_2(unsigned short)


// int
VIENNACL_INNER_PROD_IMPL_1(int, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_INNER_PROD_IMPL_2(int)

VIENNACL_INNER_PROD_IMPL_1(unsigned int, unsigned int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_INNER_PROD_IMPL_2(unsigned int)


// long
VIENNACL_INNER_PROD_IMPL_1(long, long)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_INNER_PROD_IMPL_2(long)

VIENNACL_INNER_PROD_IMPL_1(unsigned long, unsigned long)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_INNER_PROD_IMPL_2(unsigned long)


// float
VIENNACL_INNER_PROD_IMPL_1(float, float)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_INNER_PROD_IMPL_2(float)

// double
VIENNACL_INNER_PROD_IMPL_1(double, double)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_INNER_PROD_IMPL_2(double)

#undef VIENNACL_INNER_PROD_IMPL_1
#undef VIENNACL_INNER_PROD_IMPL_2
}

/** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
*
* @param vec1 The first vector
* @param vec2 The second vector
* @param result The result scalar (on the gpu)
*/
template<typename NumericT, typename ScalarT>
void inner_prod_impl(vector_base<NumericT> const & vec1,
                     vector_base<NumericT> const & vec2,
                     ScalarT & result)
{
  typedef NumericT      value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type const * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  vcl_size_t start2 = viennacl::traits::start(vec2);
  vcl_size_t inc2   = viennacl::traits::stride(vec2);

  result = detail::inner_prod_impl(data_vec1, start1, inc1, size1,
                                   data_vec2, start2, inc2);  //Note: Assignment to result might be expensive, thus a temporary is introduced here
}

template<typename NumericT>
void inner_prod_impl(vector_base<NumericT> const & x,
                     vector_tuple<NumericT> const & vec_tuple,
                     vector_base<NumericT> & result)
{
  typedef NumericT        value_type;

  value_type const * data_x = detail::extract_raw_pointer<value_type>(x);

  vcl_size_t start_x = viennacl::traits::start(x);
  vcl_size_t inc_x   = viennacl::traits::stride(x);
  vcl_size_t size_x  = viennacl::traits::size(x);

  std::vector<value_type> temp(vec_tuple.const_size());
  std::vector<value_type const *> data_y(vec_tuple.const_size());
  std::vector<vcl_size_t> start_y(vec_tuple.const_size());
  std::vector<vcl_size_t> stride_y(vec_tuple.const_size());

  for (vcl_size_t j=0; j<vec_tuple.const_size(); ++j)
  {
    data_y[j] = detail::extract_raw_pointer<value_type>(vec_tuple.const_at(j));
    start_y[j] = viennacl::traits::start(vec_tuple.const_at(j));
    stride_y[j] = viennacl::traits::stride(vec_tuple.const_at(j));
  }

  // Note: No OpenMP here because it cannot perform a reduction on temp-array. Savings in memory bandwidth are expected to still justify this approach...
  for (vcl_size_t i = 0; i < size_x; ++i)
  {
    value_type entry_x = data_x[i*inc_x+start_x];
    for (vcl_size_t j=0; j < vec_tuple.const_size(); ++j)
      temp[j] += entry_x * data_y[j][i*stride_y[j]+start_y[j]];
  }

  for (vcl_size_t j=0; j < vec_tuple.const_size(); ++j)
    result[j] = temp[j];  //Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
}


namespace detail
{

#define VIENNACL_NORM_1_IMPL_1(RESULTSCALART, TEMPSCALART) \
  inline RESULTSCALART norm_1_impl(RESULTSCALART const * data_vec1, vcl_size_t start1, vcl_size_t inc1, vcl_size_t size1) { \
    TEMPSCALART temp = 0;

#define VIENNACL_NORM_1_IMPL_2(RESULTSCALART, TEMPSCALART) \
    for (long i = 0; i < static_cast<long>(size1); ++i) \
      temp += static_cast<TEMPSCALART>(std::fabs(static_cast<double>(data_vec1[static_cast<vcl_size_t>(i)*inc1+start1]))); \
    return static_cast<RESULTSCALART>(temp); \
  }

// char
VIENNACL_NORM_1_IMPL_1(char, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_1_IMPL_2(char, int)

VIENNACL_NORM_1_IMPL_1(unsigned char, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_1_IMPL_2(unsigned char, int)

// short
VIENNACL_NORM_1_IMPL_1(short, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_1_IMPL_2(short, int)

VIENNACL_NORM_1_IMPL_1(unsigned short, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_1_IMPL_2(unsigned short, int)


// int
VIENNACL_NORM_1_IMPL_1(int, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_1_IMPL_2(int, int)

VIENNACL_NORM_1_IMPL_1(unsigned int, unsigned int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_1_IMPL_2(unsigned int, unsigned int)


// long
VIENNACL_NORM_1_IMPL_1(long, long)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_1_IMPL_2(long, long)

VIENNACL_NORM_1_IMPL_1(unsigned long, unsigned long)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_1_IMPL_2(unsigned long, unsigned long)


// float
VIENNACL_NORM_1_IMPL_1(float, float)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_1_IMPL_2(float, float)

// double
VIENNACL_NORM_1_IMPL_1(double, double)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_1_IMPL_2(double, double)

#undef VIENNACL_NORM_1_IMPL_1
#undef VIENNACL_NORM_1_IMPL_2

}

/** @brief Computes the l^1-norm of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void norm_1_impl(vector_base<NumericT> const & vec1,
                 ScalarT & result)
{
  typedef NumericT        value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  result = detail::norm_1_impl(data_vec1, start1, inc1, size1);  //Note: Assignment to result might be expensive, thus using a temporary for accumulation
}



namespace detail
{

#define VIENNACL_NORM_2_IMPL_1(RESULTSCALART, TEMPSCALART) \
  inline RESULTSCALART norm_2_impl(RESULTSCALART const * data_vec1, vcl_size_t start1, vcl_size_t inc1, vcl_size_t size1) { \
    TEMPSCALART temp = 0;

#define VIENNACL_NORM_2_IMPL_2(RESULTSCALART, TEMPSCALART) \
    for (long i = 0; i < static_cast<long>(size1); ++i) { \
      RESULTSCALART data = data_vec1[static_cast<vcl_size_t>(i)*inc1+start1]; \
      temp += static_cast<TEMPSCALART>(data * data); \
    } \
    return static_cast<RESULTSCALART>(temp); \
  }

// char
VIENNACL_NORM_2_IMPL_1(char, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_2_IMPL_2(char, int)

VIENNACL_NORM_2_IMPL_1(unsigned char, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_2_IMPL_2(unsigned char, int)


// short
VIENNACL_NORM_2_IMPL_1(short, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_2_IMPL_2(short, int)

VIENNACL_NORM_2_IMPL_1(unsigned short, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_2_IMPL_2(unsigned short, int)


// int
VIENNACL_NORM_2_IMPL_1(int, int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_2_IMPL_2(int, int)

VIENNACL_NORM_2_IMPL_1(unsigned int, unsigned int)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_2_IMPL_2(unsigned int, unsigned int)


// long
VIENNACL_NORM_2_IMPL_1(long, long)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_2_IMPL_2(long, long)

VIENNACL_NORM_2_IMPL_1(unsigned long, unsigned long)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_2_IMPL_2(unsigned long, unsigned long)


// float
VIENNACL_NORM_2_IMPL_1(float, float)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_2_IMPL_2(float, float)

// double
VIENNACL_NORM_2_IMPL_1(double, double)
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+: temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
VIENNACL_NORM_2_IMPL_2(double, double)

#undef VIENNACL_NORM_2_IMPL_1
#undef VIENNACL_NORM_2_IMPL_2

}


/** @brief Computes the l^2-norm of a vector - implementation
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void norm_2_impl(vector_base<NumericT> const & vec1,
                 ScalarT & result)
{
  typedef NumericT       value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  result = std::sqrt(detail::norm_2_impl(data_vec1, start1, inc1, size1));  //Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
}

/** @brief Computes the supremum-norm of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void norm_inf_impl(vector_base<NumericT> const & vec1,
                   ScalarT & result)
{
  typedef NumericT       value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  vcl_size_t thread_count=1;

  #ifdef VIENNACL_WITH_OPENMP
  if(size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
      thread_count = omp_get_max_threads();
  #endif

  std::vector<value_type> temp(thread_count);

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
  {
    vcl_size_t id = 0;
#ifdef VIENNACL_WITH_OPENMP
    id = omp_get_thread_num();
#endif

    vcl_size_t begin = (size1 * id) / thread_count;
    vcl_size_t end   = (size1 * (id + 1)) / thread_count;
    temp[id]         = 0;

    for (vcl_size_t i = begin; i < end; ++i)
      temp[id] = std::max<value_type>(temp[id], static_cast<value_type>(std::fabs(static_cast<double>(data_vec1[i*inc1+start1]))));  //casting to double in order to avoid problems if T is an integer type
  }
  for (vcl_size_t i = 1; i < thread_count; ++i)
    temp[0] = std::max<value_type>( temp[0], temp[i]);
  result  = temp[0];
}

//This function should return a CPU scalar, otherwise statements like
// vcl_rhs[index_norm_inf(vcl_rhs)]
// are ambiguous
/** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
*
* @param vec1 The vector
* @return The result. Note that the result must be a CPU scalar (unsigned int), since gpu scalars are floating point types.
*/
template<typename NumericT>
vcl_size_t index_norm_inf(vector_base<NumericT> const & vec1)
{
  typedef NumericT      value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);
  vcl_size_t thread_count=1;

#ifdef VIENNACL_WITH_OPENMP
  if(size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
      thread_count = omp_get_max_threads();
#endif

  std::vector<value_type> temp(thread_count);
  std::vector<vcl_size_t> index(thread_count);

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
  {
    vcl_size_t id = 0;
#ifdef VIENNACL_WITH_OPENMP
    id = omp_get_thread_num();
#endif
    vcl_size_t begin = (size1 * id) / thread_count;
    vcl_size_t end   = (size1 * (id + 1)) / thread_count;
    index[id]        = start1;
    temp[id]         = 0;
    value_type data;

    for (vcl_size_t i = begin; i < end; ++i)
    {
      data = static_cast<value_type>(std::fabs(static_cast<double>(data_vec1[i*inc1+start1])));  //casting to double in order to avoid problems if T is an integer type
      if (data > temp[id])
      {
        index[id] = i;
        temp[id]  = data;
      }
    }
  }
  for (vcl_size_t i = 1; i < thread_count; ++i)
  {
    if (temp[i] > temp[0])
    {
      index[0] = index[i];
      temp[0] = temp[i];
    }
  }
  return index[0];
}

/** @brief Computes the maximum of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void max_impl(vector_base<NumericT> const & vec1,
              ScalarT & result)
{
  typedef NumericT       value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  vcl_size_t thread_count=1;

#ifdef VIENNACL_WITH_OPENMP
  if(size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
      thread_count = omp_get_max_threads();
#endif

  std::vector<value_type> temp(thread_count);

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
  {
    vcl_size_t id = 0;
#ifdef VIENNACL_WITH_OPENMP
    id = omp_get_thread_num();
#endif
    vcl_size_t begin = (size1 * id) / thread_count;
    vcl_size_t end   = (size1 * (id + 1)) / thread_count;
    temp[id]         = data_vec1[start1];

    for (vcl_size_t i = begin; i < end; ++i)
    {
      value_type v = data_vec1[i*inc1+start1];//Note: Assignment to 'vec1' in std::min might be expensive, thus 'v' is used for the function
      temp[id] = std::max<value_type>(temp[id],v);
    }
  }
  for (vcl_size_t i = 1; i < thread_count; ++i)
    temp[0] = std::max<value_type>( temp[0], temp[i]);
  result  = temp[0];//Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
}

/** @brief Computes the minimum of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void min_impl(vector_base<NumericT> const & vec1,
              ScalarT & result)
{
  typedef NumericT       value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  vcl_size_t thread_count=1;

#ifdef VIENNACL_WITH_OPENMP
  if(size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
      thread_count = omp_get_max_threads();
#endif

  std::vector<value_type> temp(thread_count);

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
  {
    vcl_size_t id = 0;
#ifdef VIENNACL_WITH_OPENMP
    id = omp_get_thread_num();
#endif
    vcl_size_t begin = (size1 * id) / thread_count;
    vcl_size_t end   = (size1 * (id + 1)) / thread_count;
    temp[id]         = data_vec1[start1];

    for (vcl_size_t i = begin; i < end; ++i)
    {
      value_type v = data_vec1[i*inc1+start1];//Note: Assignment to 'vec1' in std::min might be expensive, thus 'v' is used for the function
      temp[id] = std::min<value_type>(temp[id],v);
    }
  }
  for (vcl_size_t i = 1; i < thread_count; ++i)
    temp[0] = std::min<value_type>( temp[0], temp[i]);
  result  = temp[0];//Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
}

/** @brief Computes the sum of all elements from the vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT, typename ScalarT>
void sum_impl(vector_base<NumericT> const & vec1,
              ScalarT & result)
{
  typedef NumericT       value_type;

  value_type const * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  value_type temp = 0;
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for reduction(+:temp) if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
  for (long i = 0; i < static_cast<long>(size1); ++i)
    temp += data_vec1[static_cast<vcl_size_t>(i)*inc1+start1];

  result = temp;  //Note: Assignment to result might be expensive, thus 'temp' is used for accumulation
}

/** @brief Computes a plane rotation of two vectors.
*
* Computes (x,y) <- (alpha * x + beta * y, -beta * x + alpha * y)
*
* @param vec1   The first vector
* @param vec2   The second vector
* @param alpha  The first transformation coefficient
* @param beta   The second transformation coefficient
*/
template<typename NumericT>
void plane_rotation(vector_base<NumericT> & vec1,
                    vector_base<NumericT> & vec2,
                    NumericT alpha, NumericT beta)
{
  typedef NumericT  value_type;

  value_type * data_vec1 = detail::extract_raw_pointer<value_type>(vec1);
  value_type * data_vec2 = detail::extract_raw_pointer<value_type>(vec2);

  vcl_size_t start1 = viennacl::traits::start(vec1);
  vcl_size_t inc1   = viennacl::traits::stride(vec1);
  vcl_size_t size1  = viennacl::traits::size(vec1);

  vcl_size_t start2 = viennacl::traits::start(vec2);
  vcl_size_t inc2   = viennacl::traits::stride(vec2);

  value_type data_alpha = alpha;
  value_type data_beta  = beta;

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
#endif
  for (long i = 0; i < static_cast<long>(size1); ++i)
  {
    value_type temp1 = data_vec1[static_cast<vcl_size_t>(i)*inc1+start1];
    value_type temp2 = data_vec2[static_cast<vcl_size_t>(i)*inc2+start2];

    data_vec1[static_cast<vcl_size_t>(i)*inc1+start1] = data_alpha * temp1 + data_beta * temp2;
    data_vec2[static_cast<vcl_size_t>(i)*inc2+start2] = data_alpha * temp2 - data_beta * temp1;
  }
}

namespace detail
{
  /** @brief Implementation of inclusive_scan and exclusive_scan for the host (OpenMP) backend. */
  template<typename NumericT>
  void vector_scan_impl(vector_base<NumericT> const & vec1,
                        vector_base<NumericT>       & vec2,
                        bool is_inclusive)
  {
    NumericT const * data_vec1 = detail::extract_raw_pointer<NumericT>(vec1);
    NumericT       * data_vec2 = detail::extract_raw_pointer<NumericT>(vec2);

    vcl_size_t start1 = viennacl::traits::start(vec1);
    vcl_size_t inc1   = viennacl::traits::stride(vec1);
    vcl_size_t size1  = viennacl::traits::size(vec1);
    if (size1 < 1)
      return;

    vcl_size_t start2 = viennacl::traits::start(vec2);
    vcl_size_t inc2   = viennacl::traits::stride(vec2);

#ifdef VIENNACL_WITH_OPENMP
    if (size1 > VIENNACL_OPENMP_VECTOR_MIN_SIZE)
    {
      std::vector<NumericT> thread_results(omp_get_max_threads());

      // inclusive scan each thread segment:
      #pragma omp parallel
      {
        vcl_size_t work_per_thread = (size1 - 1) / thread_results.size() + 1;
        vcl_size_t thread_start = work_per_thread * omp_get_thread_num();
        vcl_size_t thread_stop  = std::min<vcl_size_t>(thread_start + work_per_thread, size1);

        NumericT thread_sum = 0;
        for(vcl_size_t i = thread_start; i < thread_stop; i++)
          thread_sum += data_vec1[i * inc1 + start1];

        thread_results[omp_get_thread_num()] = thread_sum;
      }

      // exclusive-scan of thread results:
      NumericT current_offset = 0;
      for (vcl_size_t i=0; i<thread_results.size(); ++i)
      {
        NumericT tmp = thread_results[i];
        thread_results[i] = current_offset;
        current_offset += tmp;
      }

      // exclusive/inclusive scan of each segment with correct offset:
      #pragma omp parallel
      {
        vcl_size_t work_per_thread = (size1 - 1) / thread_results.size() + 1;
        vcl_size_t thread_start = work_per_thread * omp_get_thread_num();
        vcl_size_t thread_stop  = std::min<vcl_size_t>(thread_start + work_per_thread, size1);

        NumericT thread_sum = thread_results[omp_get_thread_num()];
        if (is_inclusive)
        {
          for(vcl_size_t i = thread_start; i < thread_stop; i++)
          {
            thread_sum += data_vec1[i * inc1 + start1];
            data_vec2[i * inc2 + start2] = thread_sum;
          }
        }
        else
        {
          for(vcl_size_t i = thread_start; i < thread_stop; i++)
          {
            NumericT tmp = data_vec1[i * inc1 + start1];
            data_vec2[i * inc2 + start2] = thread_sum;
            thread_sum += tmp;
          }
        }
      }
    } else
#endif
    {
      NumericT sum = 0;
      if (is_inclusive)
      {
        for(vcl_size_t i = 0; i < size1; i++)
        {
          sum += data_vec1[i * inc1 + start1];
          data_vec2[i * inc2 + start2] = sum;
        }
      }
      else
      {
        for(vcl_size_t i = 0; i < size1; i++)
        {
          NumericT tmp = data_vec1[i * inc1 + start1];
          data_vec2[i * inc2 + start2] = sum;
          sum += tmp;
        }
      }
    }

  }
}

/** @brief This function implements an inclusive scan on the host using OpenMP.
*
* Given an element vector (x_0, x_1, ..., x_{n-1}),
* this routine computes (x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-1})
*
* @param vec1       Input vector: Gets overwritten by the routine.
* @param vec2       The output vector. Either idential to vec1 or non-overlapping.
*/
template<typename NumericT>
void inclusive_scan(vector_base<NumericT> const & vec1,
                    vector_base<NumericT>       & vec2)
{
  detail::vector_scan_impl(vec1, vec2, true);
}

/** @brief This function implements an exclusive scan on the host using OpenMP.
*
* Given an element vector (x_0, x_1, ..., x_{n-1}),
* this routine computes (0, x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-2})
*
* @param vec1       Input vector: Gets overwritten by the routine.
* @param vec2       The output vector. Either idential to vec1 or non-overlapping.
*/
template<typename NumericT>
void exclusive_scan(vector_base<NumericT> const & vec1,
                    vector_base<NumericT>       & vec2)
{
  detail::vector_scan_impl(vec1, vec2, false);
}


} //namespace host_based
} //namespace linalg
} //namespace viennacl


#endif
