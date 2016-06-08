#ifndef VIENNACL_META_TAGOF_HPP_
#define VIENNACL_META_TAGOF_HPP_

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


/** @file tag_of.hpp
    @brief Dispatch facility for distinguishing between ublas, STL and ViennaCL types
*/

#include <vector>
#include <map>

#include "viennacl/forwards.h"

#ifdef VIENNACL_WITH_UBLAS
#include <boost/numeric/ublas/matrix_sparse.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/vector.hpp>
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

namespace viennacl
{

// ----------------------------------------------------
// TAGS
//
/** @brief A tag class for identifying 'unknown' types. */
struct tag_none     {};
/** @brief A tag class for identifying types from MTL4. */
struct tag_mtl4     {};
/** @brief A tag class for identifying types from Eigen. */
struct tag_eigen    {};
/** @brief A tag class for identifying types from uBLAS. */
struct tag_ublas    {};
/** @brief A tag class for identifying types from the C++ STL. */
struct tag_stl      {};
/** @brief A tag class for identifying types from ViennaCL. */
struct tag_viennacl {};

namespace traits
{
  // ----------------------------------------------------
  // GENERIC BASE
  //
  /** @brief Generic base for wrapping other linear algebra packages
  *
  *  Maps types to tags, e.g. viennacl::vector to tag_viennacl, ublas::vector to tag_ublas
  *  if the matrix type is unknown, tag_none is returned
  *
  *  This is an internal function only, there is no need for a library user of ViennaCL to care about it any further
  *
  * @tparam T   The type to be inspected
  */
  template< typename T, typename Active = void >
  struct tag_of;

  /** \cond */
  template< typename Sequence, typename Active >
  struct tag_of
  {
    typedef viennacl::tag_none  type;
  };

#ifdef VIENNACL_WITH_MTL4
  // ----------------------------------------------------
  // MTL4
  //
  template<typename ScalarType>
  struct tag_of< mtl::dense_vector<ScalarType> >
  {
    typedef viennacl::tag_mtl4  type;
  };

  template<typename ScalarType>
  struct tag_of< mtl::compressed2D<ScalarType> >
  {
    typedef viennacl::tag_mtl4  type;
  };

  template<typename ScalarType, typename T>
  struct tag_of< mtl::dense2D<ScalarType, T> >
  {
    typedef viennacl::tag_mtl4  type;
  };
#endif


#ifdef VIENNACL_WITH_EIGEN
  // ----------------------------------------------------
  // Eigen
  //
  template<>
  struct tag_of< Eigen::VectorXf >
  {
    typedef viennacl::tag_eigen  type;
  };

  template<>
  struct tag_of< Eigen::VectorXd >
  {
    typedef viennacl::tag_eigen  type;
  };

  template<>
  struct tag_of< Eigen::MatrixXf >
  {
    typedef viennacl::tag_eigen  type;
  };

  template<>
  struct tag_of< Eigen::MatrixXd >
  {
    typedef viennacl::tag_eigen  type;
  };

  template<typename ScalarType, int option>
  struct tag_of< Eigen::SparseMatrix<ScalarType, option> >
  {
    typedef viennacl::tag_eigen  type;
  };

#endif

#ifdef VIENNACL_WITH_UBLAS
  // ----------------------------------------------------
  // UBLAS
  //
  template< typename T >
  struct tag_of< boost::numeric::ublas::vector<T> >
  {
    typedef viennacl::tag_ublas  type;
  };

  template< typename T >
  struct tag_of< boost::numeric::ublas::matrix<T> >
  {
    typedef viennacl::tag_ublas  type;
  };

  template< typename T1, typename T2 >
  struct tag_of< boost::numeric::ublas::matrix_unary2<T1,T2> >
  {
    typedef viennacl::tag_ublas  type;
  };

  template< typename T1, typename T2 >
  struct tag_of< boost::numeric::ublas::compressed_matrix<T1,T2> >
  {
    typedef viennacl::tag_ublas  type;
  };

#endif

  // ----------------------------------------------------
  // STL types
  //

  //vector
  template< typename T, typename A >
  struct tag_of< std::vector<T, A> >
  {
    typedef viennacl::tag_stl  type;
  };

  //dense matrix
  template< typename T, typename A >
  struct tag_of< std::vector<std::vector<T, A>, A> >
  {
    typedef viennacl::tag_stl  type;
  };

  //sparse matrix (vector of maps)
  template< typename KEY, typename DATA, typename COMPARE, typename AMAP, typename AVEC>
  struct tag_of< std::vector<std::map<KEY, DATA, COMPARE, AMAP>, AVEC> >
  {
    typedef viennacl::tag_stl  type;
  };


  // ----------------------------------------------------
  // VIENNACL
  //
  template< typename T, unsigned int alignment >
  struct tag_of< viennacl::vector<T, alignment> >
  {
    typedef viennacl::tag_viennacl  type;
  };

  template< typename T, typename F, unsigned int alignment >
  struct tag_of< viennacl::matrix<T, F, alignment> >
  {
    typedef viennacl::tag_viennacl  type;
  };

  template< typename T1, typename T2, typename OP >
  struct tag_of< viennacl::matrix_expression<T1,T2,OP> >
  {
    typedef viennacl::tag_viennacl  type;
  };

  template< typename T >
  struct tag_of< viennacl::matrix_range<T> >
  {
    typedef viennacl::tag_viennacl  type;
  };

  template< typename T, unsigned int I>
  struct tag_of< viennacl::compressed_matrix<T,I> >
  {
    typedef viennacl::tag_viennacl  type;
  };

  template< typename T, unsigned int I>
  struct tag_of< viennacl::coordinate_matrix<T,I> >
  {
    typedef viennacl::tag_viennacl  type;
  };

  template< typename T, unsigned int I>
  struct tag_of< viennacl::ell_matrix<T,I> >
  {
    typedef viennacl::tag_viennacl  type;
  };

  template< typename T, typename I>
  struct tag_of< viennacl::sliced_ell_matrix<T,I> >
  {
    typedef viennacl::tag_viennacl  type;
  };


  template< typename T, unsigned int I>
  struct tag_of< viennacl::hyb_matrix<T,I> >
  {
    typedef viennacl::tag_viennacl  type;
  };

  template< typename T, unsigned int I>
  struct tag_of< viennacl::circulant_matrix<T,I> >
  {
    typedef viennacl::tag_viennacl  type;
  };

  template< typename T, unsigned int I>
  struct tag_of< viennacl::hankel_matrix<T,I> >
  {
    typedef viennacl::tag_viennacl  type;
  };

  template< typename T, unsigned int I>
  struct tag_of< viennacl::toeplitz_matrix<T,I> >
  {
    typedef viennacl::tag_viennacl  type;
  };

  template< typename T, unsigned int I>
  struct tag_of< viennacl::vandermonde_matrix<T,I> >
  {
    typedef viennacl::tag_viennacl  type;
  };
  /** \endcond */

  // ----------------------------------------------------
} // end namespace traits


/** @brief Meta function which checks whether a tag is tag_mtl4
*
*  This is an internal function only, there is no need for a library user of ViennaCL to care about it any further
*/
template<typename Tag>
struct is_mtl4
{
  enum { value = false };
};

/** \cond */
template<>
struct is_mtl4< viennacl::tag_mtl4 >
{
  enum { value = true };
};
/** \endcond */

/** @brief Meta function which checks whether a tag is tag_eigen
*
*  This is an internal function only, there is no need for a library user of ViennaCL to care about it any further
*/
template<typename Tag>
struct is_eigen
{
  enum { value = false };
};

/** \cond */
template<>
struct is_eigen< viennacl::tag_eigen >
{
  enum { value = true };
};
/** \endcond */


/** @brief Meta function which checks whether a tag is tag_ublas
*
*  This is an internal function only, there is no need for a library user of ViennaCL to care about it any further
*/
template<typename Tag>
struct is_ublas
{
  enum { value = false };
};

/** \cond */
template<>
struct is_ublas< viennacl::tag_ublas >
{
  enum { value = true };
};
/** \endcond */

/** @brief Meta function which checks whether a tag is tag_ublas
*
*  This is an internal function only, there is no need for a library user of ViennaCL to care about it any further
*/
template<typename Tag>
struct is_stl
{
  enum { value = false };
};

/** \cond */
template<>
struct is_stl< viennacl::tag_stl >
{
  enum { value = true };
};
/** \endcond */


/** @brief Meta function which checks whether a tag is tag_viennacl
*
*  This is an internal function only, there is no need for a library user of ViennaCL to care about it any further
*/
template<typename Tag>
struct is_viennacl
{
  enum { value = false };
};

/** \cond */
template<>
struct is_viennacl< viennacl::tag_viennacl >
{
  enum { value = true };
};
/** \endcond */

} // end namespace viennacl

#endif
