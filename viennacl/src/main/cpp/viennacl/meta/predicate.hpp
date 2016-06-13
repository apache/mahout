#ifndef VIENNACL_META_PREDICATE_HPP_
#define VIENNACL_META_PREDICATE_HPP_

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

/** @file predicate.hpp
    @brief All the predicates used within ViennaCL. Checks for expressions to be vectors, etc.
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"

#ifdef VIENNACL_WITH_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include "CL/cl.h"
#endif
#endif

namespace viennacl
{

//
// is_cpu_scalar: checks for float or double
//
//template<typename T>
//struct is_cpu_scalar
//{
//  enum { value = false };
//};

/** \cond */
template<> struct is_cpu_scalar<char>           { enum { value = true }; };
template<> struct is_cpu_scalar<unsigned char>  { enum { value = true }; };
template<> struct is_cpu_scalar<short>          { enum { value = true }; };
template<> struct is_cpu_scalar<unsigned short> { enum { value = true }; };
template<> struct is_cpu_scalar<int>            { enum { value = true }; };
template<> struct is_cpu_scalar<unsigned int>   { enum { value = true }; };
template<> struct is_cpu_scalar<long>           { enum { value = true }; };
template<> struct is_cpu_scalar<unsigned long>  { enum { value = true }; };
template<> struct is_cpu_scalar<float>          { enum { value = true }; };
template<> struct is_cpu_scalar<double>         { enum { value = true }; };
/** \endcond */


//
// is_scalar: checks for viennacl::scalar
//
//template<typename T>
//struct is_scalar
//{
//  enum { value = false };
//};

/** \cond */
template<typename T>
struct is_scalar<viennacl::scalar<T> >
{
  enum { value = true };
};
/** \endcond */

//
// is_flip_sign_scalar: checks for viennacl::scalar modified with unary operator-
//
//template<typename T>
//struct is_flip_sign_scalar
//{
//  enum { value = false };
//};

/** \cond */
template<typename T>
struct is_flip_sign_scalar<viennacl::scalar_expression< const scalar<T>,
    const scalar<T>,
    op_flip_sign> >
{
  enum { value = true };
};
/** \endcond */

//
// is_any_scalar: checks for either CPU and GPU scalars, i.e. is_cpu_scalar<>::value || is_scalar<>::value
//
//template<typename T>
//struct is_any_scalar
//{
//  enum { value = (is_scalar<T>::value || is_cpu_scalar<T>::value || is_flip_sign_scalar<T>::value )};
//};

//

/** \cond */
#define VIENNACL_MAKE_ANY_VECTOR_TRUE(type) template<> struct is_any_vector< type > { enum { value = 1 }; };
#define VIENNACL_MAKE_FOR_ALL_NumericT(type) \
  VIENNACL_MAKE_ANY_VECTOR_TRUE(type<float>)\
  VIENNACL_MAKE_ANY_VECTOR_TRUE(type<double>)

  VIENNACL_MAKE_FOR_ALL_NumericT(viennacl::vector)
  VIENNACL_MAKE_FOR_ALL_NumericT(viennacl::vector_range)
  VIENNACL_MAKE_FOR_ALL_NumericT(viennacl::vector_slice)
  VIENNACL_MAKE_FOR_ALL_NumericT(viennacl::unit_vector)
  VIENNACL_MAKE_FOR_ALL_NumericT(viennacl::zero_vector)
  VIENNACL_MAKE_FOR_ALL_NumericT(viennacl::one_vector)
  VIENNACL_MAKE_FOR_ALL_NumericT(viennacl::scalar_vector)

#undef VIENNACL_MAKE_FOR_ALL_NumericT
#undef VIENNACL_MAKE_ANY_VECTOR_TRUE
  /** \endcond */


  /** \cond */
#define VIENNACL_MAKE_ANY_MATRIX_TRUE(TYPE)\
template<> struct is_any_dense_matrix< TYPE > { enum { value = 1 }; };

#define VIENNACL_MAKE_FOR_ALL_NumericT(TYPE) \
  VIENNACL_MAKE_ANY_MATRIX_TRUE(TYPE<float>)\
  VIENNACL_MAKE_ANY_MATRIX_TRUE(TYPE<double>)

#define VIENNACL_COMMA ,
#define VIENNACL_MAKE_FOR_ALL_NumericT_LAYOUT(TYPE) \
  VIENNACL_MAKE_ANY_MATRIX_TRUE(TYPE<float VIENNACL_COMMA viennacl::row_major>)\
  VIENNACL_MAKE_ANY_MATRIX_TRUE(TYPE<double VIENNACL_COMMA viennacl::row_major>)\
  VIENNACL_MAKE_ANY_MATRIX_TRUE(TYPE<float VIENNACL_COMMA viennacl::column_major>)\
  VIENNACL_MAKE_ANY_MATRIX_TRUE(TYPE<double VIENNACL_COMMA viennacl::column_major>)

  VIENNACL_MAKE_FOR_ALL_NumericT_LAYOUT(viennacl::matrix)
  //    VIENNACL_MAKE_FOR_ALL_NumericT_LAYOUT(viennacl::matrix_range)
  //    VIENNACL_MAKE_FOR_ALL_NumericT_LAYOUT(viennacl::matrix_slice)
  VIENNACL_MAKE_FOR_ALL_NumericT(viennacl::identity_matrix)
  VIENNACL_MAKE_FOR_ALL_NumericT(viennacl::zero_matrix)
  VIENNACL_MAKE_FOR_ALL_NumericT(viennacl::scalar_matrix)

#undef VIENNACL_MAKE_FOR_ALL_NumericT_LAYOUT
#undef VIENNACL_MAKE_FOR_ALL_NumericT
#undef VIENNACL_MAKE_ANY_MATRIX_TRUE
#undef VIENNACL_COMMA
/** \endcond */

//
// is_row_major
//
//template<typename T>
//struct is_row_major
//{
//  enum { value = false };
//};

/** \cond */
template<typename ScalarType>
struct is_row_major<viennacl::matrix<ScalarType, viennacl::row_major> >
{
  enum { value = true };
};

template<>
struct is_row_major< viennacl::row_major >
{
  enum { value = true };
};

template<typename T>
struct is_row_major<viennacl::matrix_expression<T, T, viennacl::op_trans> >
{
  enum { value = is_row_major<T>::value };
};
/** \endcond */


//
// is_circulant_matrix
//
//template<typename T>
//struct is_circulant_matrix
//{
//  enum { value = false };
//};

/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_circulant_matrix<viennacl::circulant_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename ScalarType, unsigned int AlignmentV>
struct is_circulant_matrix<const viennacl::circulant_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */

//
// is_hankel_matrix
//
//template<typename T>
//struct is_hankel_matrix
//{
//  enum { value = false };
//};

/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_hankel_matrix<viennacl::hankel_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename ScalarType, unsigned int AlignmentV>
struct is_hankel_matrix<const viennacl::hankel_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */

//
// is_toeplitz_matrix
//
//template<typename T>
//struct is_toeplitz_matrix
//{
//  enum { value = false };
//};

/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_toeplitz_matrix<viennacl::toeplitz_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename ScalarType, unsigned int AlignmentV>
struct is_toeplitz_matrix<const viennacl::toeplitz_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */

//
// is_vandermonde_matrix
//
//template<typename T>
//struct is_vandermonde_matrix
//{
//  enum { value = false };
//};

/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_vandermonde_matrix<viennacl::vandermonde_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename ScalarType, unsigned int AlignmentV>
struct is_vandermonde_matrix<const viennacl::vandermonde_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */


//
// is_compressed_matrix
//

/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_compressed_matrix<viennacl::compressed_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */

//
// is_coordinate_matrix
//

/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_coordinate_matrix<viennacl::coordinate_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */

//
// is_ell_matrix
//
/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_ell_matrix<viennacl::ell_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */

//
// is_sliced_ell_matrix
//
/** \cond */
template<typename ScalarType, typename IndexT>
struct is_sliced_ell_matrix<viennacl::sliced_ell_matrix<ScalarType, IndexT> >
{
  enum { value = true };
};
/** \endcond */

//
// is_hyb_matrix
//
/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_hyb_matrix<viennacl::hyb_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};
/** \endcond */


//
// is_any_sparse_matrix
//
//template<typename T>
//struct is_any_sparse_matrix
//{
//  enum { value = false };
//};

/** \cond */
template<typename ScalarType, unsigned int AlignmentV>
struct is_any_sparse_matrix<viennacl::compressed_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename ScalarType>
struct is_any_sparse_matrix<viennacl::compressed_compressed_matrix<ScalarType> >
{
  enum { value = true };
};

template<typename ScalarType, unsigned int AlignmentV>
struct is_any_sparse_matrix<viennacl::coordinate_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename ScalarType, unsigned int AlignmentV>
struct is_any_sparse_matrix<viennacl::ell_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename ScalarType, typename IndexT>
struct is_any_sparse_matrix<viennacl::sliced_ell_matrix<ScalarType, IndexT> >
{
  enum { value = true };
};

template<typename ScalarType, unsigned int AlignmentV>
struct is_any_sparse_matrix<viennacl::hyb_matrix<ScalarType, AlignmentV> >
{
  enum { value = true };
};

template<typename T>
struct is_any_sparse_matrix<const T>
{
  enum { value = is_any_sparse_matrix<T>::value };
};

/** \endcond */

//////////////// Part 2: Operator predicates ////////////////////

//
// is_addition
//
/** @brief Helper metafunction for checking whether the provided type is viennacl::op_add (for addition) */
template<typename T>
struct is_addition
{
  enum { value = false };
};

/** \cond */
template<>
struct is_addition<viennacl::op_add>
{
  enum { value = true };
};
/** \endcond */

//
// is_subtraction
//
/** @brief Helper metafunction for checking whether the provided type is viennacl::op_sub (for subtraction) */
template<typename T>
struct is_subtraction
{
  enum { value = false };
};

/** \cond */
template<>
struct is_subtraction<viennacl::op_sub>
{
  enum { value = true };
};
/** \endcond */

//
// is_product
//
/** @brief Helper metafunction for checking whether the provided type is viennacl::op_prod (for products/multiplication) */
template<typename T>
struct is_product
{
  enum { value = false };
};

/** \cond */
template<>
struct is_product<viennacl::op_prod>
{
  enum { value = true };
};

template<>
struct is_product<viennacl::op_mult>
{
  enum { value = true };
};

template<>
struct is_product<viennacl::op_element_binary<op_prod> >
{
  enum { value = true };
};
/** \endcond */

//
// is_division
//
/** @brief Helper metafunction for checking whether the provided type is viennacl::op_div (for division) */
template<typename T>
struct is_division
{
  enum { value = false };
};

/** \cond */
template<>
struct is_division<viennacl::op_div>
{
  enum { value = true };
};

template<>
struct is_division<viennacl::op_element_binary<op_div> >
{
  enum { value = true };
};
/** \endcond */

// is_primitive_type
//

/** @brief Helper class for checking whether a type is a primitive type. */
template<class T>
struct is_primitive_type{ enum {value = false}; };

/** \cond */
template<> struct is_primitive_type<float>         { enum { value = true }; };
template<> struct is_primitive_type<double>        { enum { value = true }; };
template<> struct is_primitive_type<unsigned int>  { enum { value = true }; };
template<> struct is_primitive_type<int>           { enum { value = true }; };
template<> struct is_primitive_type<unsigned char> { enum { value = true }; };
template<> struct is_primitive_type<char>          { enum { value = true }; };
template<> struct is_primitive_type<unsigned long> { enum { value = true }; };
template<> struct is_primitive_type<long>          { enum { value = true }; };
template<> struct is_primitive_type<unsigned short>{ enum { value = true }; };
template<> struct is_primitive_type<short>         { enum { value = true }; };
/** \endcond */

#ifdef VIENNACL_WITH_OPENCL

/** @brief Helper class for checking whether a particular type is a native OpenCL type. */
template<class T>
struct is_cl_type{ enum { value = false }; };

/** \cond */
template<> struct is_cl_type<cl_float> { enum { value = true }; };
template<> struct is_cl_type<cl_double>{ enum { value = true }; };
template<> struct is_cl_type<cl_uint>  { enum { value = true }; };
template<> struct is_cl_type<cl_int>   { enum { value = true }; };
template<> struct is_cl_type<cl_uchar> { enum { value = true }; };
template<> struct is_cl_type<cl_char>  { enum { value = true }; };
template<> struct is_cl_type<cl_ulong> { enum { value = true }; };
template<> struct is_cl_type<cl_long>  { enum { value = true }; };
template<> struct is_cl_type<cl_ushort>{ enum { value = true }; };
template<> struct is_cl_type<cl_short> { enum { value = true }; };
/** \endcond */

/** @brief Helper class for checking whether a particular type is a floating point type. */
template<class T> struct is_floating_point { enum { value = false }; };
template<> struct is_floating_point<float> { enum { value = true }; };
template<> struct is_floating_point<double> { enum { value = true }; };

#endif

} //namespace viennacl


#endif
