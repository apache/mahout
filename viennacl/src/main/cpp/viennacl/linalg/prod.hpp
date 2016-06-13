#ifndef VIENNACL_LINALG_PROD_HPP_
#define VIENNACL_LINALG_PROD_HPP_

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

/** @file viennacl/linalg/prod.hpp
    @brief Generic interface for matrix-vector and matrix-matrix products.
           See viennacl/linalg/vector_operations.hpp, viennacl/linalg/matrix_operations.hpp, and
           viennacl/linalg/sparse_matrix_operations.hpp for implementations.
*/

#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/tag_of.hpp"
#include <vector>
#include <map>

namespace viennacl
{
  //
  // generic prod function
  //   uses tag dispatch to identify which algorithm
  //   should be called
  //
  namespace linalg
  {
    #ifdef VIENNACL_WITH_MTL4
    // ----------------------------------------------------
    // mtl4
    //
    template< typename MatrixT, typename VectorT >
    typename viennacl::enable_if< viennacl::is_mtl4< typename viennacl::traits::tag_of< MatrixT >::type >::value,
                                  VectorT>::type
    prod(MatrixT const& matrix, VectorT const& vector)
    {
      return VectorT(matrix * vector);
    }
    #endif

    #ifdef VIENNACL_WITH_ARMADILLO
    // ----------------------------------------------------
    // Armadillo
    //
    template<typename NumericT, typename VectorT>
    VectorT prod(arma::SpMat<NumericT> const& A, VectorT const& vector)
    {
      return A * vector;
    }
    #endif

    #ifdef VIENNACL_WITH_EIGEN
    // ----------------------------------------------------
    // Eigen
    //
    template< typename MatrixT, typename VectorT >
    typename viennacl::enable_if< viennacl::is_eigen< typename viennacl::traits::tag_of< MatrixT >::type >::value,
                                  VectorT>::type
    prod(MatrixT const& matrix, VectorT const& vector)
    {
      return matrix * vector;
    }
    #endif

    #ifdef VIENNACL_WITH_UBLAS
    // ----------------------------------------------------
    // UBLAS
    //
    template< typename MatrixT, typename VectorT >
    typename viennacl::enable_if< viennacl::is_ublas< typename viennacl::traits::tag_of< MatrixT >::type >::value,
                                  VectorT>::type
    prod(MatrixT const& matrix, VectorT const& vector)
    {
      // std::cout << "ublas .. " << std::endl;
      return boost::numeric::ublas::prod(matrix, vector);
    }
    #endif


    // ----------------------------------------------------
    // STL type
    //

    // dense matrix-vector product:
    template< typename T, typename A1, typename A2, typename VectorT >
    VectorT
    prod(std::vector< std::vector<T, A1>, A2 > const & matrix, VectorT const& vector)
    {
      VectorT result(matrix.size());
      for (typename std::vector<T, A1>::size_type i=0; i<matrix.size(); ++i)
      {
        result[i] = 0; //we will not assume that VectorT is initialized to zero
        for (typename std::vector<T, A1>::size_type j=0; j<matrix[i].size(); ++j)
          result[i] += matrix[i][j] * vector[j];
      }
      return result;
    }

    // sparse matrix-vector product:
    template< typename KEY, typename DATA, typename COMPARE, typename AMAP, typename AVEC, typename VectorT >
    VectorT
    prod(std::vector< std::map<KEY, DATA, COMPARE, AMAP>, AVEC > const& matrix, VectorT const& vector)
    {
      typedef std::vector< std::map<KEY, DATA, COMPARE, AMAP>, AVEC > MatrixType;

      VectorT result(matrix.size());
      for (typename MatrixType::size_type i=0; i<matrix.size(); ++i)
      {
        result[i] = 0; //we will not assume that VectorT is initialized to zero
        for (typename std::map<KEY, DATA, COMPARE, AMAP>::const_iterator row_entries = matrix[i].begin();
             row_entries != matrix[i].end();
             ++row_entries)
          result[i] += row_entries->second * vector[row_entries->first];
      }
      return result;
    }


    /*template< typename MatrixT, typename VectorT >
    VectorT
    prod(MatrixT const& matrix, VectorT const& vector,
         typename viennacl::enable_if< viennacl::is_stl< typename viennacl::traits::tag_of< MatrixT >::type >::value
                                     >::type* dummy = 0)
    {
      // std::cout << "std .. " << std::endl;
      return prod_impl(matrix, vector);
    }*/

    // ----------------------------------------------------
    // VIENNACL
    //

    // standard product:
    template<typename NumericT>
    viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                 const viennacl::matrix_base<NumericT>,
                                 viennacl::op_mat_mat_prod >
    prod(viennacl::matrix_base<NumericT> const & A,
         viennacl::matrix_base<NumericT> const & B)
    {
      return viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                          const viennacl::matrix_base<NumericT>,
                                          viennacl::op_mat_mat_prod >(A, B);
    }

    // right factor is a matrix expression:
    template<typename NumericT, typename LhsT, typename RhsT, typename OpT>
    viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                 const viennacl::matrix_expression<const LhsT, const RhsT, OpT>,
                                 viennacl::op_mat_mat_prod >
    prod(viennacl::matrix_base<NumericT> const & A,
         viennacl::matrix_expression<const LhsT, const RhsT, OpT> const & B)
    {
      return viennacl::matrix_expression< const viennacl::matrix_base<NumericT>,
                                          const viennacl::matrix_expression<const LhsT, const RhsT, OpT>,
                                          viennacl::op_mat_mat_prod >(A, B);
    }

    // left factor is a matrix expression:
    template<typename LhsT, typename RhsT, typename OpT, typename NumericT>
    viennacl::matrix_expression< const viennacl::matrix_expression<const LhsT, const RhsT, OpT>,
                                 const viennacl::matrix_base<NumericT>,
                                 viennacl::op_mat_mat_prod >
    prod(viennacl::matrix_expression<const LhsT, const RhsT, OpT> const & A,
         viennacl::matrix_base<NumericT> const & B)
    {
      return viennacl::matrix_expression< const viennacl::matrix_expression<const LhsT, const RhsT, OpT>,
                                          const viennacl::matrix_base<NumericT>,
                                          viennacl::op_mat_mat_prod >(A, B);
    }


    // both factors transposed:
    template<typename LhsT1, typename RhsT1, typename OpT1,
             typename LhsT2, typename RhsT2, typename OpT2>
    viennacl::matrix_expression< const viennacl::matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                 const viennacl::matrix_expression<const LhsT2, const RhsT2, OpT2>,
                                 viennacl::op_mat_mat_prod >
    prod(viennacl::matrix_expression<const LhsT1, const RhsT1, OpT1> const & A,
         viennacl::matrix_expression<const LhsT2, const RhsT2, OpT2> const & B)
    {
      return viennacl::matrix_expression< const viennacl::matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                          const viennacl::matrix_expression<const LhsT2, const RhsT2, OpT2>,
                                          viennacl::op_mat_mat_prod >(A, B);
    }



    // matrix-vector product
    template< typename NumericT>
    viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                 const viennacl::vector_base<NumericT>,
                                 viennacl::op_prod >
    prod(viennacl::matrix_base<NumericT> const & A,
         viennacl::vector_base<NumericT> const & x)
    {
      return viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                          const viennacl::vector_base<NumericT>,
                                          viennacl::op_prod >(A, x);
    }

    // matrix-vector product (resolve ambiguity)
    template<typename NumericT, typename F>
    viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                 const viennacl::vector_base<NumericT>,
                                 viennacl::op_prod >
    prod(viennacl::matrix<NumericT, F> const & A,
         viennacl::vector_base<NumericT> const & x)
    {
      return viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                          const viennacl::vector_base<NumericT>,
                                          viennacl::op_prod >(A, x);
    }

    // matrix-vector product (resolve ambiguity)
    template<typename MatrixT, typename NumericT>
    viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                 const viennacl::vector_base<NumericT>,
                                 viennacl::op_prod >
    prod(viennacl::matrix_range<MatrixT> const & A,
         viennacl::vector_base<NumericT> const & x)
    {
      return viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                          const viennacl::vector_base<NumericT>,
                                          viennacl::op_prod >(A, x);
    }

    // matrix-vector product (resolve ambiguity)
    template<typename MatrixT, typename NumericT>
    viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                 const viennacl::vector_base<NumericT>,
                                 viennacl::op_prod >
    prod(viennacl::matrix_slice<MatrixT> const & A,
         viennacl::vector_base<NumericT> const & x)
    {
      return viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                          const viennacl::vector_base<NumericT>,
                                          viennacl::op_prod >(A, x);
    }

    // matrix-vector product with matrix expression (including transpose)
    template< typename NumericT, typename LhsT, typename RhsT, typename OpT>
    viennacl::vector_expression< const viennacl::matrix_expression<const LhsT, const RhsT, OpT>,
                                 const viennacl::vector_base<NumericT>,
                                 viennacl::op_prod >
    prod(viennacl::matrix_expression<const LhsT, const RhsT, OpT> const & A,
         viennacl::vector_base<NumericT> const & x)
    {
      return viennacl::vector_expression< const viennacl::matrix_expression<const LhsT, const RhsT, OpT>,
                                          const viennacl::vector_base<NumericT>,
                                          viennacl::op_prod >(A, x);
    }


    // matrix-vector product with vector expression
    template< typename NumericT, typename LhsT, typename RhsT, typename OpT>
    viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                 const viennacl::vector_expression<const LhsT, const RhsT, OpT>,
                                 viennacl::op_prod >
    prod(viennacl::matrix_base<NumericT> const & A,
         viennacl::vector_expression<const LhsT, const RhsT, OpT> const & x)
    {
      return viennacl::vector_expression< const viennacl::matrix_base<NumericT>,
                                          const viennacl::vector_expression<const LhsT, const RhsT, OpT>,
                                          viennacl::op_prod >(A, x);
    }


    // matrix-vector product with matrix expression (including transpose) and vector expression
    template<typename LhsT1, typename RhsT1, typename OpT1,
             typename LhsT2, typename RhsT2, typename OpT2>
    viennacl::vector_expression< const viennacl::matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                 const viennacl::vector_expression<const LhsT2, const RhsT2, OpT2>,
                                 viennacl::op_prod >
    prod(viennacl::matrix_expression<const LhsT1, const RhsT1, OpT1> const & A,
         viennacl::vector_expression<const LhsT2, const RhsT2, OpT2> const & x)
    {
      return viennacl::vector_expression< const viennacl::matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                          const viennacl::vector_expression<const LhsT2, const RhsT2, OpT2>,
                                          viennacl::op_prod >(A, x);
    }




    template< typename SparseMatrixType, typename SCALARTYPE>
    typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value,
                                  viennacl::matrix_expression<const SparseMatrixType,
                                                              const matrix_base <SCALARTYPE>,
                                                              op_prod >
                                 >::type
    prod(const SparseMatrixType & sp_mat,
         const viennacl::matrix_base<SCALARTYPE> & d_mat)
    {
      return viennacl::matrix_expression<const SparseMatrixType,
                                         const viennacl::matrix_base<SCALARTYPE>,
                                         op_prod >(sp_mat, d_mat);
    }

    // right factor is transposed
    template< typename SparseMatrixType, typename SCALARTYPE>
    typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value,
                                  viennacl::matrix_expression< const SparseMatrixType,
                                                               const viennacl::matrix_expression<const viennacl::matrix_base<SCALARTYPE>,
                                                                                                 const viennacl::matrix_base<SCALARTYPE>,
                                                                                                 op_trans>,
                                                               viennacl::op_prod >
                                  >::type
    prod(const SparseMatrixType & A,
         viennacl::matrix_expression<const viennacl::matrix_base<SCALARTYPE>,
                                     const viennacl::matrix_base<SCALARTYPE>,
                                     op_trans> const & B)
    {
      return viennacl::matrix_expression< const SparseMatrixType,
                                          const viennacl::matrix_expression<const viennacl::matrix_base<SCALARTYPE>,
                                                                            const viennacl::matrix_base<SCALARTYPE>,
                                                                            op_trans>,
                                          viennacl::op_prod >(A, B);
    }


    /** @brief Sparse matrix-matrix product with compressed_matrix objects */
    template<typename NumericT>
    viennacl::matrix_expression<const compressed_matrix<NumericT>,
                                const compressed_matrix<NumericT>,
                                op_prod >
    prod(compressed_matrix<NumericT> const & A,
         compressed_matrix<NumericT> const & B)
    {
      return viennacl::matrix_expression<const compressed_matrix<NumericT>,
                                         const compressed_matrix<NumericT>,
                                         op_prod >(A, B);
    }

    /** @brief Generic matrix-vector product with user-provided sparse matrix type */
    template<typename SparseMatrixType, typename NumericT>
    vector_expression<const SparseMatrixType,
                      const vector_base<NumericT>,
                      op_prod >
    prod(const SparseMatrixType & A,
         const vector_base<NumericT> & x)
    {
      return vector_expression<const SparseMatrixType,
                               const vector_base<NumericT>,
                               op_prod >(A, x);
    }

  } // end namespace linalg
} // end namespace viennacl
#endif





