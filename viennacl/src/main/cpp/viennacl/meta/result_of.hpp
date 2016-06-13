#ifndef VIENNACL_META_RESULT_OF_HPP_
#define VIENNACL_META_RESULT_OF_HPP_

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

/** @file viennacl/meta/result_of.hpp
    @brief A collection of compile time type deductions
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"


#ifdef VIENNACL_WITH_UBLAS
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#endif

#ifdef VIENNACL_WITH_ARMADILLO
#include <armadillo>
#endif

#ifdef VIENNACL_WITH_EIGEN
#include <Eigen/Core>
#include <Eigen/Sparse>
#endif

#ifdef VIENNACL_WITH_MTL4
#include <boost/numeric/mtl/mtl.hpp>
#endif

#ifdef VIENNACL_WITH_OPENCL
#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include "CL/cl.h"
#endif
#endif

#include <vector>
#include <map>

namespace viennacl
{
namespace result_of
{
//
// Retrieve alignment from vector
//
/** @brief Retrieves the alignment from a vector. Deprecated - will be replaced by a pure runtime facility in the future. */
template<typename T>
struct alignment
{
  typedef typename T::ERROR_ARGUMENT_PROVIDED_IS_NOT_A_VECTOR_OR_A_MATRIX   error_type;
  enum { value = 1 };
};

/** \cond */
template<typename T>
struct alignment<const T>
{
  enum { value = alignment<T>::value };
};

template<typename NumericT, unsigned int AlignmentV>
struct alignment< vector<NumericT, AlignmentV> >
{
  enum { value = AlignmentV };
};

template<typename T>
struct alignment< vector_range<T> >
{
  enum { value = alignment<T>::value };
};

template<typename T>
struct alignment< vector_slice<T> >
{
  enum { value = alignment<T>::value };
};

// support for a*x with scalar a and vector x
template<typename LHS, typename RHS, typename OP>
struct alignment< vector_expression<LHS, RHS, OP> >
{
  enum { value = alignment<LHS>::value };
};


// Matrices
template<typename NumericT, typename F, unsigned int AlignmentV>
struct alignment< matrix<NumericT, F, AlignmentV> >
{
  enum { value = AlignmentV };
};

template<typename T>
struct alignment< matrix_range<T> >
{
  enum { value = alignment<T>::value };
};

template<typename T>
struct alignment< matrix_slice<T> >
{
  enum { value = alignment<T>::value };
};

template<typename LHS, typename RHS>
struct alignment< matrix_expression<LHS, RHS, op_trans> >
{
  enum { value = alignment<LHS>::value };
};
/** \endcond */

//
// Retrieve size_type
//
/** @brief Generic meta-function for retrieving the size_type associated with type T */
template<typename T>
struct size_type
{
  typedef typename T::size_type   type;
};

/** \cond */
template<typename T, typename SizeType>
struct size_type< vector_base<T, SizeType> >
{
  typedef SizeType   type;
};

//
// Retrieve difference_type
//
/** @brief Generic meta-function for retrieving the difference_type associated with type T */
template<typename T>
struct difference_type
{
  typedef typename T::difference_type   type;
};

#ifdef VIENNACL_WITH_ARMADILLO
template<typename NumericT>
struct size_type<arma::Col<NumericT> > { typedef vcl_size_t  type; };

template<typename NumericT>
struct size_type<arma::Mat<NumericT> > { typedef vcl_size_t  type; };

template<typename NumericT>
struct size_type<arma::SpMat<NumericT> > { typedef vcl_size_t  type; };

#endif

#ifdef VIENNACL_WITH_EIGEN
template<class T, int a, int b, int c, int d, int e>
struct size_type< Eigen::Matrix<T, a, b, c, d, e> >
{
  typedef vcl_size_t   type;
};

template<class T, int a, int b, int c, int d, int e>
struct size_type< Eigen::Map<Eigen::Matrix<T, a, b, c, d, e> > >
{
  typedef vcl_size_t   type;
};

template<>
struct size_type<Eigen::VectorXf>
{
  typedef vcl_size_t   type;
};

template<>
struct size_type<Eigen::VectorXd>
{
  typedef vcl_size_t   type;
};

template<typename T, int options>
struct size_type<Eigen::SparseMatrix<T, options> >
{
  typedef vcl_size_t   type;
};
#endif
/** \endcond */

//
// Retrieve value_type:
//
/** @brief Generic helper function for retrieving the value_type associated with type T */
template<typename T>
struct value_type
{
  typedef typename T::value_type    type;
};

/** \cond */
#ifdef VIENNACL_WITH_ARMADILLO
template<typename NumericT>
struct value_type<arma::Col<NumericT> > { typedef NumericT  type; };

template<typename NumericT>
struct value_type<arma::Mat<NumericT> > { typedef NumericT  type; };

template<typename NumericT>
struct value_type<arma::SpMat<NumericT> > { typedef NumericT  type; };

#endif

#ifdef VIENNACL_WITH_EIGEN
template<>
struct value_type<Eigen::MatrixXf>
{
  typedef Eigen::MatrixXf::RealScalar    type;
};

template<>
struct value_type<Eigen::MatrixXd>
{
  typedef Eigen::MatrixXd::RealScalar    type;
};

template<typename ScalarType, int option>
struct value_type<Eigen::SparseMatrix<ScalarType, option> >
{
  typedef ScalarType    type;
};

template<>
struct value_type<Eigen::VectorXf>
{
  typedef Eigen::VectorXf::RealScalar    type;
};

template<>
struct value_type<Eigen::VectorXd>
{
  typedef Eigen::VectorXd::RealScalar    type;
};

#endif
/** \endcond */


//
// Retrieve cpu value_type:
//
/** @brief Helper meta function for retrieving the main RAM-based value type. Particularly important to obtain T from viennacl::scalar<T> in a generic way. */
template<typename T>
struct cpu_value_type
{
  typedef typename T::ERROR_CANNOT_DEDUCE_CPU_SCALAR_TYPE_FOR_T    type;
};

/** \cond */
template<typename T>
struct cpu_value_type<const T>
{
  typedef typename cpu_value_type<T>::type    type;
};

template<>
struct cpu_value_type<char>
{
  typedef char    type;
};

template<>
struct cpu_value_type<unsigned char>
{
  typedef unsigned char    type;
};

template<>
struct cpu_value_type<short>
{
  typedef short    type;
};

template<>
struct cpu_value_type<unsigned short>
{
  typedef unsigned short    type;
};

template<>
struct cpu_value_type<int>
{
  typedef int    type;
};

template<>
struct cpu_value_type<unsigned int>
{
  typedef unsigned int    type;
};

template<>
struct cpu_value_type<long>
{
  typedef int    type;
};

template<>
struct cpu_value_type<unsigned long>
{
  typedef unsigned long    type;
};


template<>
struct cpu_value_type<float>
{
  typedef float    type;
};

template<>
struct cpu_value_type<double>
{
  typedef double    type;
};

template<typename T>
struct cpu_value_type<viennacl::scalar<T> >
{
  typedef T    type;
};

template<typename T>
struct cpu_value_type<viennacl::vector_base<T> >
{
  typedef T    type;
};

template<typename T>
struct cpu_value_type<viennacl::implicit_vector_base<T> >
{
  typedef T    type;
};


template<typename T, unsigned int AlignmentV>
struct cpu_value_type<viennacl::vector<T, AlignmentV> >
{
  typedef T    type;
};

template<typename T>
struct cpu_value_type<viennacl::vector_range<T> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T>
struct cpu_value_type<viennacl::vector_slice<T> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T1, typename T2, typename OP>
struct cpu_value_type<viennacl::vector_expression<const T1, const T2, OP> >
{
  typedef typename cpu_value_type<T1>::type    type;
};

template<typename T1, typename T2, typename OP>
struct cpu_value_type<const viennacl::vector_expression<const T1, const T2, OP> >
{
  typedef typename cpu_value_type<T1>::type    type;
};


template<typename T>
struct cpu_value_type<viennacl::matrix_base<T> >
{
  typedef T    type;
};

template<typename T>
struct cpu_value_type<viennacl::implicit_matrix_base<T> >
{
  typedef T    type;
};


template<typename T, typename F, unsigned int AlignmentV>
struct cpu_value_type<viennacl::matrix<T, F, AlignmentV> >
{
  typedef T    type;
};

template<typename T>
struct cpu_value_type<viennacl::matrix_range<T> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T>
struct cpu_value_type<viennacl::matrix_slice<T> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<viennacl::compressed_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T>
struct cpu_value_type<viennacl::compressed_compressed_matrix<T> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<viennacl::coordinate_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<viennacl::ell_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, typename IndexT>
struct cpu_value_type<viennacl::sliced_ell_matrix<T, IndexT> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<viennacl::hyb_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<viennacl::circulant_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<viennacl::hankel_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<viennacl::toeplitz_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T, unsigned int AlignmentV>
struct cpu_value_type<viennacl::vandermonde_matrix<T, AlignmentV> >
{
  typedef typename cpu_value_type<T>::type    type;
};

template<typename T1, typename T2, typename OP>
struct cpu_value_type<viennacl::matrix_expression<T1, T2, OP> >
{
  typedef typename cpu_value_type<T1>::type    type;
};


//
// Deduce compatible vector type for a matrix type
//

template<typename T>
struct vector_for_matrix
{
  typedef typename T::ERROR_CANNOT_DEDUCE_VECTOR_FOR_MATRIX_TYPE   type;
};

//ViennaCL
template<typename T, typename F, unsigned int A>
struct vector_for_matrix< viennacl::matrix<T, F, A> >
{
  typedef viennacl::vector<T,A>   type;
};

template<typename T, unsigned int A>
struct vector_for_matrix< viennacl::compressed_matrix<T, A> >
{
  typedef viennacl::vector<T,A>   type;
};

template<typename T, unsigned int A>
struct vector_for_matrix< viennacl::coordinate_matrix<T, A> >
{
  typedef viennacl::vector<T,A>   type;
};

#ifdef VIENNACL_WITH_UBLAS
//Boost:
template<typename T, typename F, typename A>
struct vector_for_matrix< boost::numeric::ublas::matrix<T, F, A> >
{
  typedef boost::numeric::ublas::vector<T>   type;
};

template<typename T, typename U, vcl_size_t A, typename B, typename C>
struct vector_for_matrix< boost::numeric::ublas::compressed_matrix<T, U, A, B, C> >
{
  typedef boost::numeric::ublas::vector<T>   type;
};

template<typename T, typename U, vcl_size_t A, typename B, typename C>
struct vector_for_matrix< boost::numeric::ublas::coordinate_matrix<T, U, A, B, C> >
{
  typedef boost::numeric::ublas::vector<T>   type;
};
#endif

template<typename T>
struct reference_if_nonscalar
{
  typedef T &    type;
};

#define VIENNACL_REFERENCE_IF_NONSCALAR_INT(TNAME) \
template<> struct reference_if_nonscalar<TNAME>                { typedef                TNAME  type; }; \
template<> struct reference_if_nonscalar<const TNAME>          { typedef          const TNAME  type; }; \
template<> struct reference_if_nonscalar<unsigned TNAME>       { typedef       unsigned TNAME  type; }; \
template<> struct reference_if_nonscalar<const unsigned TNAME> { typedef const unsigned TNAME  type; };

  VIENNACL_REFERENCE_IF_NONSCALAR_INT(char)
  VIENNACL_REFERENCE_IF_NONSCALAR_INT(short)
  VIENNACL_REFERENCE_IF_NONSCALAR_INT(int)
  VIENNACL_REFERENCE_IF_NONSCALAR_INT(long)

#undef VIENNACL_REFERENCE_IF_NONSCALAR_INT

template<>
struct reference_if_nonscalar<float>
{
  typedef float    type;
};

template<>
struct reference_if_nonscalar<const float>
{
  typedef const float    type;
};

template<>
struct reference_if_nonscalar<double>
{
  typedef double    type;
};

template<>
struct reference_if_nonscalar<const double>
{
  typedef const double    type;
};

/** \endcond */

//OpenCL equivalent type
/** @brief Metafunction for deducing the OpenCL type for a numeric type, e.g. float -> cl_float */
template<typename T>
struct cl_type
{
  typedef T type;
};

/** \cond */
#ifdef VIENNACL_WITH_OPENCL
template<>
struct cl_type<float>{ typedef cl_float type; };

template<>
struct cl_type<double>{ typedef cl_double type; };

template<>
struct cl_type<int>{ typedef cl_int type; };

template<>
struct cl_type<unsigned int>{  typedef cl_uint type; };

template<>
struct cl_type<long>{  typedef cl_long type;  };

template<>
struct cl_type<unsigned long>{ typedef cl_ulong type; };

template<>
struct cl_type<short>{ typedef cl_short type;  };

template<>
struct cl_type<unsigned short>{ typedef cl_ushort type; };

template<>
struct cl_type<char>{ typedef cl_char type; };

template<>
struct cl_type<unsigned char>{ typedef cl_uchar type; };
#endif
  /** \endcond */

} //namespace result_of
} //namespace viennacl


#endif
