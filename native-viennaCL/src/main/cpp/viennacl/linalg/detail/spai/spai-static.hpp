#ifndef VIENNACL_LINALG_DETAIL_SPAI_SPAI_STATIC_HPP
#define VIENNACL_LINALG_DETAIL_SPAI_SPAI_STATIC_HPP

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

/** @file viennacl/linalg/detail/spai/spai-static.hpp
    @brief Implementation of a static SPAI. Experimental.

    SPAI code contributed by Nikolay Lukash
*/

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <math.h>
#include <map>
//#include "spai-dynamic.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/storage.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/lu.hpp"
#include "boost/numeric/ublas/triangular.hpp"
#include "boost/numeric/ublas/matrix_expression.hpp"
// ViennaCL includes
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/sparse_matrix_operations.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/linalg/cg.hpp"
#include "viennacl/linalg/inner_prod.hpp"

//#include "boost/numeric/ublas/detail/matrix_assign.hpp"

namespace viennacl
{
namespace linalg
{
namespace detail
{
namespace spai
{

/** @brief Determines if element ind is in set {J}
 *
 * @param J     current set
 * @param ind   current element
 */
template<typename SizeT>
bool isInIndexSet(std::vector<SizeT> const & J, SizeT ind)
{
  return (std::find(J.begin(), J.end(), ind) != J.end());
}



/********************************* STATIC SPAI FUNCTIONS******************************************/

/** @brief Projects solution of LS problem onto original column m
 *
 * @param m_in   solution of LS
 * @param J      set of non-zero columns
 * @param m      original column of M
 */
template<typename VectorT, typename SparseVectorT>
void fanOutVector(VectorT const & m_in, std::vector<unsigned int> const & J, SparseVectorT & m)
{
  unsigned int  cnt = 0;
  for (vcl_size_t i = 0; i < J.size(); ++i)
    m[J[i]] = m_in(cnt++);
}

/** @brief Solution of linear:R*x=y system by backward substitution
 *
 * @param R   uppertriangular matrix
 * @param y   right handside vector
 * @param x   solution vector
 */
template<typename MatrixT, typename VectorT>
void backwardSolve(MatrixT const & R, VectorT const & y, VectorT & x)
{
  for (long i2 = static_cast<long>(R.size2())-1; i2 >= 0; i2--)
  {
    vcl_size_t i = static_cast<vcl_size_t>(i2);
    x(i) = y(i);
    for (vcl_size_t j = static_cast<vcl_size_t>(i)+1; j < R.size2(); ++j)
      x(i) -= R(i,j)*x(j);

    x(i) /= R(i,i);
  }
}

/** @brief Perform projection of set I on the unit-vector
 *
 * @param I     set of non-zero rows
 * @param y     result vector
 * @param ind   index of unit vector
 */
template<typename VectorT, typename NumericT>
void projectI(std::vector<unsigned int> const & I, VectorT & y, unsigned int ind)
{
  for (vcl_size_t i = 0; i < I.size(); ++i)
  {
    //y.resize(y.size()+1);
    if (I[i] == ind)
      y(i) = NumericT(1.0);
    else
      y(i) = NumericT(0.0);
  }
}

/** @brief Builds index set of projected columns for current column of preconditioner
 *
 * @param v    current column of preconditioner
 * @param J    output - index set of non-zero columns
 */
template<typename SparseVectorT>
void buildColumnIndexSet(SparseVectorT const & v, std::vector<unsigned int> & J)
{
  for (typename SparseVectorT::const_iterator vec_it = v.begin(); vec_it != v.end(); ++vec_it)
    J.push_back(vec_it->first);

  std::sort(J.begin(), J.end());
}

/** @brief Initialize preconditioner with sparcity pattern = p(A)
 *
 * @param A   input matrix
 * @param M   output matrix - initialized preconditioner
 */
template<typename SparseMatrixT>
void initPreconditioner(SparseMatrixT const & A, SparseMatrixT & M)
{
  typedef typename SparseMatrixT::value_type      NumericType;

  M.resize(A.size1(), A.size2(), false);
  for (typename SparseMatrixT::const_iterator1 row_it = A.begin1(); row_it!= A.end1(); ++row_it)
    for (typename SparseMatrixT::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
      M(col_it.index1(),col_it.index2()) = NumericType(1);
}

/** @brief Row projection for matrix A(:,J) -> A(I,J), building index set of non-zero rows
 *
 * @param A_v_c   input matrix
 * @param J       set of non-zero rows
 * @param I       output matrix
 */
template<typename SparseVectorT>
void projectRows(std::vector<SparseVectorT> const & A_v_c,
                 std::vector<unsigned int> const & J,
                 std::vector<unsigned int>       & I)
{
  for (vcl_size_t i = 0; i < J.size(); ++i)
  {
    for (typename SparseVectorT::const_iterator col_it = A_v_c[J[i]].begin(); col_it!=A_v_c[J[i]].end(); ++col_it)
    {
      if (!isInIndexSet(I, col_it->first))
        I.push_back(col_it->first);
    }
  }
  std::sort(I.begin(), I.end());
}


} //namespace spai
} //namespace detail
} //namespace linalg
} //namespace viennacl

#endif
