#ifndef VIENNACL_TRAITS_SIZE_HPP_
#define VIENNACL_TRAITS_SIZE_HPP_

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

/** @file viennacl/traits/size.hpp
    @brief Generic size and resize functionality for different vector and matrix types
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/meta/predicate.hpp"

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

#include <vector>
#include <map>

namespace viennacl
{
namespace traits
{

//
// Resize: Change the size of vectors and matrices
//
/** @brief Generic resize routine for resizing a matrix (ViennaCL, uBLAS, etc.) to a new size/dimension */
template<typename MatrixType>
void resize(MatrixType & matrix, vcl_size_t rows, vcl_size_t cols)
{
  matrix.resize(rows, cols);
}

/** @brief Generic resize routine for resizing a vector (ViennaCL, uBLAS, etc.) to a new size */
template<typename VectorType>
void resize(VectorType & vec, vcl_size_t new_size)
{
  vec.resize(new_size);
}

/** \cond */
#ifdef VIENNACL_WITH_UBLAS
//ublas needs separate treatment:
template<typename ScalarType>
void resize(boost::numeric::ublas::compressed_matrix<ScalarType> & matrix,
            vcl_size_t rows,
            vcl_size_t cols)
{
  matrix.resize(rows, cols, false); //Note: omitting third parameter leads to compile time error (not implemented in ublas <= 1.42)
}
#endif


#ifdef VIENNACL_WITH_MTL4
template<typename ScalarType>
void resize(mtl::compressed2D<ScalarType> & matrix,
            vcl_size_t rows,
            vcl_size_t cols)
{
  matrix.change_dim(rows, cols);
}

template<typename ScalarType>
void resize(mtl::dense_vector<ScalarType> & vec,
            vcl_size_t new_size)
{
  vec.change_dim(new_size);
}
#endif

#ifdef VIENNACL_WITH_ARMADILLO
template<typename NumericT>
inline void resize(arma::Mat<NumericT> & A,
                   vcl_size_t new_rows,
                   vcl_size_t new_cols)
{
  A.resize(new_rows, new_cols);
}

template<typename NumericT>
inline void resize(arma::SpMat<NumericT> & A,
                   vcl_size_t new_rows,
                   vcl_size_t new_cols)
{
  A.set_size(new_rows, new_cols);
}
#endif

#ifdef VIENNACL_WITH_EIGEN
template<typename NumericT, int Options>
inline void resize(Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, Options> & m,
                   vcl_size_t new_rows,
                   vcl_size_t new_cols)
{
  m.resize(new_rows, new_cols);
}

template<typename T, int options>
inline void resize(Eigen::SparseMatrix<T, options> & m,
                   vcl_size_t new_rows,
                   vcl_size_t new_cols)
{
  m.resize(new_rows, new_cols);
}

inline void resize(Eigen::VectorXf & v,
                   vcl_size_t new_size)
{
  v.resize(new_size);
}

inline void resize(Eigen::VectorXd & v,
                   vcl_size_t new_size)
{
  v.resize(new_size);
}
#endif
/** \endcond */




//
// size1: No. of rows for matrices
//
/** @brief Generic routine for obtaining the number of rows of a matrix (ViennaCL, uBLAS, etc.) */
template<typename MatrixType>
vcl_size_t
size1(MatrixType const & mat) { return mat.size1(); }

/** \cond */
template<typename RowType>
vcl_size_t
size1(std::vector< RowType > const & mat) { return mat.size(); }

#ifdef VIENNACL_WITH_ARMADILLO
template<typename NumericT>
inline vcl_size_t size1(arma::Mat<NumericT> const & A) { return A.n_rows; }
template<typename NumericT>
inline vcl_size_t size1(arma::SpMat<NumericT> const & A) { return A.n_rows; }
#endif

#ifdef VIENNACL_WITH_EIGEN
template<typename NumericT, int Options>
vcl_size_t size1(Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, Options> const & m) { return static_cast<vcl_size_t>(m.rows()); }
template<typename NumericT, int Options>
vcl_size_t size1(Eigen::Map< Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, Options> > const & m) { return static_cast<vcl_size_t>(m.rows()); }
template<typename T, int options>
inline vcl_size_t size1(Eigen::SparseMatrix<T, options> & m) { return static_cast<vcl_size_t>(m.rows()); }
#endif

#ifdef VIENNACL_WITH_MTL4
template<typename NumericT, typename T>
vcl_size_t size1(mtl::dense2D<NumericT, T> const & m) { return static_cast<vcl_size_t>(m.num_rows()); }
template<typename NumericT>
vcl_size_t size1(mtl::compressed2D<NumericT> const & m) { return static_cast<vcl_size_t>(m.num_rows()); }
#endif
/** \endcond */


//
// size2: No. of columns for matrices
//
/** @brief Generic routine for obtaining the number of columns of a matrix (ViennaCL, uBLAS, etc.) */
template<typename MatrixType>
typename result_of::size_type<MatrixType>::type
size2(MatrixType const & mat) { return mat.size2(); }

/** \cond */
template<typename RowType>
vcl_size_t
size2(std::vector< RowType > const & mat) { return mat[0].size(); }

#ifdef VIENNACL_WITH_ARMADILLO
template<typename NumericT>
inline vcl_size_t size2(arma::Mat<NumericT> const & A) { return A.n_cols; }
template<typename NumericT>
inline vcl_size_t size2(arma::SpMat<NumericT> const & A) { return A.n_cols; }
#endif

#ifdef VIENNACL_WITH_EIGEN
template<typename NumericT, int Options>
inline vcl_size_t size2(Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, Options> const & m) { return m.cols(); }
template<typename NumericT, int Options>
inline vcl_size_t size2(Eigen::Map< Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, Options> > const & m) { return m.cols(); }
template<typename T, int options>
inline vcl_size_t size2(Eigen::SparseMatrix<T, options> & m) { return m.cols(); }
#endif

#ifdef VIENNACL_WITH_MTL4
template<typename NumericT, typename T>
vcl_size_t size2(mtl::dense2D<NumericT, T> const & m) { return static_cast<vcl_size_t>(m.num_cols()); }
template<typename NumericT>
vcl_size_t size2(mtl::compressed2D<NumericT> const & m) { return static_cast<vcl_size_t>(m.num_cols()); }
#endif
/** \endcond */



//
// size: Returns the length of vectors
//
/** @brief Generic routine for obtaining the size of a vector (ViennaCL, uBLAS, etc.) */
template<typename VectorType>
vcl_size_t size(VectorType const & vec)
{
  return vec.size();
}

/** \cond */
template<typename SparseMatrixType, typename VectorType>
vcl_size_t size(vector_expression<const SparseMatrixType, const VectorType, op_prod> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename T, unsigned int A, typename VectorType>
vcl_size_t size(vector_expression<const circulant_matrix<T, A>, const VectorType, op_prod> const & proxy) { return proxy.lhs().size1();  }

template<typename T, unsigned int A, typename VectorType>
vcl_size_t size(vector_expression<const hankel_matrix<T, A>, const VectorType, op_prod> const & proxy) { return proxy.lhs().size1();  }

template<typename T, unsigned int A, typename VectorType>
vcl_size_t size(vector_expression<const toeplitz_matrix<T, A>, const VectorType, op_prod> const & proxy) { return proxy.lhs().size1();  }

template<typename T, unsigned int A, typename VectorType>
vcl_size_t size(vector_expression<const vandermonde_matrix<T, A>, const VectorType, op_prod> const & proxy) { return proxy.lhs().size1();  }

template<typename NumericT>
vcl_size_t size(vector_expression<const matrix_base<NumericT>, const vector_base<NumericT>, op_prod> const & proxy)  //matrix-vector product
{
  return proxy.lhs().size1();
}

template<typename NumericT, typename LhsT, typename RhsT, typename OpT>
vcl_size_t size(vector_expression<const matrix_base<NumericT>, const vector_expression<LhsT, RhsT, OpT>, op_prod> const & proxy)  //matrix-vector product
{
  return proxy.lhs().size1();
}

template<typename NumericT>
vcl_size_t size(vector_expression<const matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>,
                const vector_base<NumericT>,
                op_prod> const & proxy)  //transposed matrix-vector product
{
  return proxy.lhs().lhs().size2();
}


#ifdef VIENNACL_WITH_MTL4
template<typename ScalarType>
vcl_size_t size(mtl::dense_vector<ScalarType> const & vec) { return vec.used_memory(); }
#endif

#ifdef VIENNACL_WITH_ARMADILLO
template<typename NumericT>
inline vcl_size_t size(arma::Mat<NumericT> const & A) { return A.n_elem; }
#endif

#ifdef VIENNACL_WITH_EIGEN
inline vcl_size_t size(Eigen::VectorXf const & v) { return v.rows(); }
inline vcl_size_t size(Eigen::VectorXd const & v) { return v.rows(); }
#endif

template<typename LHS, typename RHS, typename OP>
vcl_size_t size(vector_expression<LHS, RHS, OP> const & proxy)
{
  return size(proxy.lhs());
}

template<typename LHS, typename RHS>
vcl_size_t size(vector_expression<LHS, const vector_tuple<RHS>, op_inner_prod> const & proxy)
{
  return proxy.rhs().const_size();
}

template<typename LhsT, typename RhsT, typename OpT, typename VectorT>
vcl_size_t size(vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                  VectorT,
                                  op_prod> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename LhsT, typename RhsT, typename OpT, typename NumericT>
vcl_size_t size(vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                  const vector_base<NumericT>,
                                  op_prod> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename LhsT1, typename RhsT1, typename OpT1,
         typename LhsT2, typename RhsT2, typename OpT2>
vcl_size_t size(vector_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                  const vector_expression<const LhsT2, const RhsT2, OpT2>,
                                  op_prod> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename NumericT>
vcl_size_t size(vector_expression<const matrix_base<NumericT>,
                                  const matrix_base<NumericT>,
                                  op_row_sum> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename LhsT, typename RhsT, typename OpT>
vcl_size_t size(vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                  const matrix_expression<const LhsT, const RhsT, OpT>,
                                  op_row_sum> const & proxy)
{
  return size1(proxy.lhs());
}

template<typename NumericT>
vcl_size_t size(vector_expression<const matrix_base<NumericT>,
                                  const matrix_base<NumericT>,
                                  op_col_sum> const & proxy)
{
  return size2(proxy.lhs());
}

template<typename LhsT, typename RhsT, typename OpT>
vcl_size_t size(vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                  const matrix_expression<const LhsT, const RhsT, OpT>,
                                  op_col_sum> const & proxy)
{
  return size2(proxy.lhs());
}

/** \endcond */

//
// internal_size: Returns the internal (padded) length of vectors
//
/** @brief Helper routine for obtaining the buffer length of a ViennaCL vector  */
template<typename NumericT>
vcl_size_t internal_size(vector_base<NumericT> const & vec)
{
  return vec.internal_size();
}


//
// internal_size1: No. of internal (padded) rows for matrices
//
/** @brief Helper routine for obtaining the internal number of entries per row of a ViennaCL matrix  */
template<typename NumericT>
vcl_size_t internal_size1(matrix_base<NumericT> const & mat) { return mat.internal_size1(); }


//
// internal_size2: No. of internal (padded) columns for matrices
//
/** @brief Helper routine for obtaining the internal number of entries per column of a ViennaCL matrix  */
template<typename NumericT>
vcl_size_t internal_size2(matrix_base<NumericT> const & mat) { return mat.internal_size2(); }

/** @brief Helper routine for obtaining the internal number of entries per row of a ViennaCL matrix  */
template<typename NumericT>
vcl_size_t ld(matrix_base<NumericT> const & mat)
{
  if (mat.row_major())
    return mat.internal_size2();
  return mat.internal_size1();
}

template<typename NumericT>
vcl_size_t nld(matrix_base<NumericT> const & mat)
{
  if (mat.row_major())
    return mat.stride2();
  return mat.stride1();
}

template<typename LHS>
vcl_size_t size(vector_expression<LHS, const int, op_matrix_diag> const & proxy)
{
  int k = proxy.rhs();
  int A_size1 = static_cast<int>(size1(proxy.lhs()));
  int A_size2 = static_cast<int>(size2(proxy.lhs()));

  int row_depth = std::min(A_size1, A_size1 + k);
  int col_depth = std::min(A_size2, A_size2 - k);

  return vcl_size_t(std::min(row_depth, col_depth));
}

template<typename LHS>
vcl_size_t size(vector_expression<LHS, const unsigned int, op_row> const & proxy)
{
  return size2(proxy.lhs());
}

template<typename LHS>
vcl_size_t size(vector_expression<LHS, const unsigned int, op_column> const & proxy)
{
  return size1(proxy.lhs());
}

} //namespace traits
} //namespace viennacl


#endif
