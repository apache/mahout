#ifndef VIENNACL_TOOLS_MATRIX_SIZE_DEDUCER_HPP_
#define VIENNACL_TOOLS_MATRIX_SIZE_DEDUCER_HPP_

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

/** @file viennacl/tools/matrix_size_deducer.hpp
    @brief Helper implementations that deduce the dimensions of the supplied matrix-valued expressions.
*/

#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <vector>
#include <map>

#include "viennacl/forwards.h"
#include "viennacl/tools/adapter.hpp"

namespace viennacl
{
namespace tools
{

/** @brief Deduces the size of the resulting vector represented by a vector_expression from the operands
*
* @tparam LHS   The left hand side operand
* @tparam RHS   The right hand side operand
* @tparam OP    The operation tag
*/
template<typename LHS, typename RHS, typename OP>
struct MATRIX_SIZE_DEDUCER
{
  //Standard case: size1 from lhs, size2 from rhs (fits most cases)
  static vcl_size_t size1(LHS & lhs, RHS & /*rhs*/) { return lhs.size1(); }
  static vcl_size_t size2(LHS & /*lhs*/, RHS & rhs) { return rhs.size2(); }
};

/** \cond */
//special case: outer vector product:
template<typename ScalarType>
struct MATRIX_SIZE_DEDUCER<const viennacl::vector_base<ScalarType>,
    const viennacl::vector_base<ScalarType>,
    viennacl::op_prod>
{
  static vcl_size_t size1(viennacl::vector_base<ScalarType> const & lhs,
                          viennacl::vector_base<ScalarType> const & /*rhs*/) { return lhs.size(); }

  static vcl_size_t size2(viennacl::vector_base<ScalarType> const & /*lhs*/,
                          viennacl::vector_base<ScalarType> const & rhs) { return rhs.size(); }
};


//special case: multiplication with a scalar
template<typename LHS, typename RHS, typename OP, typename ScalarType>
struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_expression<const LHS, const RHS, OP>,
    const ScalarType,
    viennacl::op_mult>
{
  static vcl_size_t size1(viennacl::matrix_expression<const LHS, const RHS, OP> const & lhs,
                          ScalarType const & /*rhs*/) { return MATRIX_SIZE_DEDUCER<const LHS, const RHS, OP>::size1(lhs.lhs(), lhs.rhs()); }

  static vcl_size_t size2(viennacl::matrix_expression<const LHS, const RHS, OP> const & lhs,
                          ScalarType const & /*rhs*/) { return MATRIX_SIZE_DEDUCER<const LHS, const RHS, OP>::size2(lhs.lhs(), lhs.rhs()); }
};

//special case: multiplication with a scalar
template<typename T, typename ScalarType>
struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_base<T>,
    const ScalarType,
    viennacl::op_mult>
{
  static vcl_size_t size1(viennacl::matrix_base<T> const & lhs,
                          ScalarType const & /*rhs*/) { return lhs.size1(); }

  static vcl_size_t size2(viennacl::matrix_base<T> const & lhs,
                          ScalarType const & /*rhs*/) { return lhs.size2(); }
};


//special case: division with a scalar
template<typename LHS, typename RHS, typename OP, typename ScalarType>
struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_expression<const LHS, const RHS, OP>,
    const ScalarType,
    viennacl::op_div>
{
  static vcl_size_t size1(viennacl::matrix_expression<const LHS, const RHS, OP> const & lhs,
                          ScalarType const & /*rhs*/) { return MATRIX_SIZE_DEDUCER<const LHS, const RHS, OP>::size1(lhs.lhs(), lhs.rhs()); }

  static vcl_size_t size2(viennacl::matrix_expression<const LHS, const RHS, OP> const & lhs,
                          ScalarType const & /*rhs*/) { return MATRIX_SIZE_DEDUCER<const LHS, const RHS, OP>::size2(lhs.lhs(), lhs.rhs()); }
};

//special case: division with a scalar
template<typename T, typename ScalarType>
struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_base<T>,
    const ScalarType,
    viennacl::op_div>
{
  static vcl_size_t size1(viennacl::matrix_base<T> const & lhs,
                          ScalarType const & /*rhs*/) { return lhs.size1(); }

  static vcl_size_t size2(viennacl::matrix_base<T> const & lhs,
                          ScalarType const & /*rhs*/) { return lhs.size2(); }
};

//special case: diagonal from vector
template<typename T>
struct MATRIX_SIZE_DEDUCER<const viennacl::vector_base<T>,
    const int,
    viennacl::op_vector_diag>
{
  static vcl_size_t size1(viennacl::vector_base<T> const & lhs,
                          const int k) { return lhs.size() + static_cast<vcl_size_t>(std::fabs(double(k))); }

  static vcl_size_t size2(viennacl::vector_base<T> const & lhs,
                          const int k) { return lhs.size() + static_cast<vcl_size_t>(std::fabs(double(k))); }
};

//special case: transposed matrix-vector product: Return the number of rows of the matrix
template<typename MatrixType>
struct MATRIX_SIZE_DEDUCER<MatrixType,
    MatrixType,
    viennacl::op_trans>
{
  static vcl_size_t size1(const MatrixType & lhs,
                          const MatrixType & /*rhs*/) { return lhs.size2(); }
  static vcl_size_t size2(const MatrixType & lhs,
                          const MatrixType & /*rhs*/) { return lhs.size1(); }
};

// A^T * B
template<typename ScalarType, typename T1>
struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_expression<T1,
    T1, op_trans>,
    const viennacl::matrix_base<ScalarType>,
    viennacl::op_mat_mat_prod>
{
  static vcl_size_t size1(viennacl::matrix_expression<T1,
                          T1,
                          op_trans> const & lhs,
                          viennacl::matrix_base<ScalarType> const & /*rhs*/) { return lhs.lhs().size2(); }
  static vcl_size_t size2(viennacl::matrix_expression<T1,
                          T1,
                          op_trans> const & /*lhs*/,
                          viennacl::matrix_base<ScalarType> const & rhs) { return rhs.size2(); }
};


// A * B^T
template<typename ScalarType, typename T2>
struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_base<ScalarType>,
    const viennacl::matrix_expression<T2,
    T2, op_trans>,
    viennacl::op_mat_mat_prod>
{
  static vcl_size_t size1(viennacl::matrix_base<ScalarType> const & lhs,
                          viennacl::matrix_expression<T2,
                          T2,
                          op_trans> const & /*rhs*/) { return lhs.size1(); }
  static vcl_size_t size2(viennacl::matrix_base<ScalarType> const & /*lhs*/,
                          viennacl::matrix_expression<T2,
                          T2,
                          op_trans> const & rhs) { return rhs.lhs().size1(); }
};

// A^T * B^T
template<typename T1, typename T2>
struct MATRIX_SIZE_DEDUCER<const viennacl::matrix_expression<T1,
    T1, op_trans>,
    const viennacl::matrix_expression<T2,
    T2, op_trans>,
    viennacl::op_mat_mat_prod>
{
  typedef viennacl::matrix_expression<T1, T1, op_trans>   LHSType;
  typedef viennacl::matrix_expression<T2, T2, op_trans>   RHSType;

  static vcl_size_t size1(LHSType const & lhs,
                          RHSType const & /*rhs*/) { return lhs.lhs().size2(); }
  static vcl_size_t size2(LHSType const & /*lhs*/,
                          RHSType const & rhs) { return rhs.lhs().size1(); }
};
/** \endcond */

}
}

#endif

