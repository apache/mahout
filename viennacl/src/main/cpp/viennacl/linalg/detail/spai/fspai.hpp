#ifndef VIENNACL_LINALG_DETAIL_SPAI_FSPAI_HPP
#define VIENNACL_LINALG_DETAIL_SPAI_FSPAI_HPP

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

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <math.h>
#include <map>

//boost includes
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
#include "viennacl/linalg/ilu.hpp"
//#include <omp.h>

/** @file viennacl/linalg/detail/spai/fspai.hpp
    @brief Implementation of FSPAI. Experimental.
*/

namespace viennacl
{
namespace linalg
{
namespace detail
{
namespace spai
{

/** @brief A tag for FSPAI. Experimental.
*
* Contains values for the algorithm.
* Must be passed to spai_precond constructor
*/
class fspai_tag
{
public:
  /** @brief Constructor
   *
   * @param residual_norm_threshold Calculate until the norm of the residual falls below this threshold
   * @param iteration_limit maximum number of iterations
   * @param is_static determines if static version of SPAI should be used
   * @param is_right determines if left or right preconditioner should be used
   */
  fspai_tag(
          double residual_norm_threshold = 1e-3,
          unsigned int iteration_limit = 5,
          bool is_static = false,
          bool is_right = false)
    : residual_norm_threshold_(residual_norm_threshold),
      iteration_limit_(iteration_limit),
      is_static_(is_static),
      is_right_(is_right) {}

  inline double getResidualNormThreshold() const { return residual_norm_threshold_; }
  inline unsigned long getIterationLimit () const { return iteration_limit_; }
  inline bool getIsStatic() const { return is_static_; }
  inline bool getIsRight() const  { return is_right_; }
  inline void setResidualNormThreshold(double residual_norm_threshold)
  {
    if (residual_norm_threshold > 0)
      residual_norm_threshold_ = residual_norm_threshold;
  }
  inline void setIterationLimit(unsigned long iteration_limit)
  {
    if (iteration_limit > 0)
      iteration_limit_ = iteration_limit;
  }
  inline void setIsRight(bool is_right)   { is_right_  = is_right; }
  inline void setIsStatic(bool is_static) { is_static_ = is_static; }

private:
  double residual_norm_threshold_;
  unsigned long iteration_limit_;
  bool is_static_;
  bool is_right_;
};


//
// Helper: Store A in an STL container of type, exploiting symmetry
// Reason: ublas interface does not allow to iterate over nonzeros of a particular row without starting an iterator1 from the very beginning of the matrix...
//
template<typename MatrixT, typename NumericT>
void sym_sparse_matrix_to_stl(MatrixT const & A, std::vector<std::map<unsigned int, NumericT> > & STL_A)
{
  STL_A.resize(A.size1());
  for (typename MatrixT::const_iterator1 row_it  = A.begin1();
                                         row_it != A.end1();
                                       ++row_it)
  {
    for (typename MatrixT::const_iterator2 col_it  = row_it.begin();
                                           col_it != row_it.end();
                                         ++col_it)
    {
      if (col_it.index1() >= col_it.index2())
        STL_A[col_it.index1()][static_cast<unsigned int>(col_it.index2())] = *col_it;
      else
        break; //go to next row
    }
  }
}


//
// Generate index sets J_k, k=0,...,N-1
//
template<typename MatrixT>
void generateJ(MatrixT const & A, std::vector<std::vector<vcl_size_t> > & J)
{
  for (typename MatrixT::const_iterator1 row_it  = A.begin1();
                                         row_it != A.end1();
                                       ++row_it)
  {
    for (typename MatrixT::const_iterator2 col_it  = row_it.begin();
                                           col_it != row_it.end();
                                         ++col_it)
    {
      if (col_it.index1() > col_it.index2()) //Matrix is symmetric, thus only work on lower triangular part
      {
        J[col_it.index2()].push_back(col_it.index1());
        J[col_it.index1()].push_back(col_it.index2());
      }
      else
        break; //go to next row
    }
  }
}


//
// Extracts the blocks A(\tilde{J}_k, \tilde{J}_k) from A
// Sets up y_k = A(\tilde{J}_k, k) for the inplace-solution after Cholesky-factoriation
//
template<typename NumericT, typename MatrixT, typename VectorT>
void fill_blocks(std::vector< std::map<unsigned int, NumericT> > & A,
                 std::vector<MatrixT>                            & blocks,
                 std::vector<std::vector<vcl_size_t> > const     & J,
                 std::vector<VectorT>                            & Y)
{
  for (vcl_size_t k=0; k<A.size(); ++k)
  {
    std::vector<vcl_size_t> const & Jk = J[k];
    VectorT & yk = Y[k];
    MatrixT & block_k = blocks[k];

    yk.resize(Jk.size());
    block_k.resize(Jk.size(), Jk.size());
    block_k.clear();

    for (vcl_size_t i=0; i<Jk.size(); ++i)
    {
      vcl_size_t row_index = Jk[i];
      std::map<unsigned int, NumericT> & A_row = A[row_index];

      //fill y_k:
      yk[i] = A_row[static_cast<unsigned int>(k)];

      for (vcl_size_t j=0; j<Jk.size(); ++j)
      {
        vcl_size_t col_index = Jk[j];
        if (col_index <= row_index && A_row.find(static_cast<unsigned int>(col_index)) != A_row.end()) //block is symmetric, thus store only lower triangular part
          block_k(i, j) = A_row[static_cast<unsigned int>(col_index)];
      }
    }
  }
}


//
// Perform Cholesky factorization of A inplace. Cf. Schwarz: Numerische Mathematik, vol 5, p. 58
//
template<typename MatrixT>
void cholesky_decompose(MatrixT & A)
{
  for (vcl_size_t k=0; k<A.size2(); ++k)
  {
    if (A(k,k) <= 0)
    {
      std::cout << "k: " << k << std::endl;
      std::cout << "A(k,k): " << A(k,k) << std::endl;
    }

    assert(A(k,k) > 0 && bool("Matrix not positive definite in Cholesky factorization."));

    A(k,k) = std::sqrt(A(k,k));

    for (vcl_size_t i=k+1; i<A.size1(); ++i)
    {
      A(i,k) /= A(k,k);
      for (vcl_size_t j=k+1; j<=i; ++j)
        A(i,j) -= A(i,k) * A(j,k);
    }
  }
}


//
// Compute x in Ax = b, where A is already Cholesky factored (A = L L^T)
//
template<typename MatrixT, typename VectorT>
void cholesky_solve(MatrixT const & L, VectorT & b)
{
  // inplace forward solve L x = b
  for (vcl_size_t i=0; i<L.size1(); ++i)
  {
    for (vcl_size_t j=0; j<i; ++j)
      b[i] -= L(i,j) * b[j];
    b[i] /= L(i,i);
  }

  // inplace backward solve L^T x = b:
  for (vcl_size_t i=L.size1()-1;; --i)
  {
    for (vcl_size_t k=i+1; k<L.size1(); ++k)
      b[i] -= L(k,i) * b[k];
    b[i] /= L(i,i);

    if (i==0) //vcl_size_t might be unsigned, therefore manual check for equality with zero here
      break;
  }
}



//
// Compute the Cholesky factor L from the sparse vectors y_k
//
template<typename MatrixT, typename VectorT>
void computeL(MatrixT const & A,
              MatrixT       & L,
              MatrixT       & L_trans,
              std::vector<VectorT> & Y,
              std::vector<std::vector<vcl_size_t> > & J)
{
  typedef typename VectorT::value_type                          NumericType;
  typedef std::vector<std::map<unsigned int, NumericType> >     STLSparseMatrixType;

  STLSparseMatrixType L_temp(A.size1());

  for (vcl_size_t k=0; k<A.size1(); ++k)
  {
    std::vector<vcl_size_t> const & Jk = J[k];
    VectorT const & yk = Y[k];

    //compute L(k,k):
    NumericType Lkk = A(k,k);
    for (vcl_size_t i=0; i<Jk.size(); ++i)
      Lkk -= A(Jk[i],k) * yk[i];

    Lkk = NumericType(1) / std::sqrt(Lkk);
    L_temp[k][static_cast<unsigned int>(k)] = Lkk;
    L_trans(k,k) = Lkk;

    //write lower diagonal entries:
    for (vcl_size_t i=0; i<Jk.size(); ++i)
    {
      L_temp[Jk[i]][static_cast<unsigned int>(k)] = -Lkk * yk[i];
      L_trans(k, Jk[i]) = -Lkk * yk[i];
    }
  } //for k


  //build L from L_temp
  for (vcl_size_t i=0; i<L_temp.size(); ++i)
    for (typename std::map<unsigned int, NumericType>::const_iterator it = L_temp[i].begin();
           it != L_temp[i].end();
         ++it)
      L(i, it->first) = it->second;
}


//
// Top level FSPAI function
//
template<typename MatrixT>
void computeFSPAI(MatrixT const & A,
                  MatrixT const & PatternA,
                  MatrixT       & L,
                  MatrixT       & L_trans,
                  fspai_tag)
{
  typedef typename MatrixT::value_type                    NumericT;
  typedef boost::numeric::ublas::matrix<NumericT>         DenseMatrixType;
  typedef std::vector<std::map<unsigned int, NumericT> >  SparseMatrixType;

  //
  // preprocessing: Store A in a STL container:
  //
  //std::cout << "Transferring to STL container:" << std::endl;
  std::vector<std::vector<NumericT> >    y_k(A.size1());
  SparseMatrixType   STL_A(A.size1());
  sym_sparse_matrix_to_stl(A, STL_A);


  //
  // Step 1: Generate pattern indices
  //
  //std::cout << "computeFSPAI(): Generating pattern..." << std::endl;
  std::vector<std::vector<vcl_size_t> > J(A.size1());
  generateJ(PatternA, J);

  //
  // Step 2: Set up matrix blocks
  //
  //std::cout << "computeFSPAI(): Setting up matrix blocks..." << std::endl;
  std::vector<DenseMatrixType>  subblocks_A(A.size1());
  fill_blocks(STL_A, subblocks_A, J, y_k);
  STL_A.clear(); //not needed anymore

  //
  // Step 3: Cholesky-factor blocks
  //
  //std::cout << "computeFSPAI(): Cholesky-factorization..." << std::endl;
  for (vcl_size_t i=0; i<subblocks_A.size(); ++i)
  {
    //std::cout << "Block before: " << subblocks_A[i] << std::endl;
    cholesky_decompose(subblocks_A[i]);
    //std::cout << "Block after: " << subblocks_A[i] << std::endl;
  }


  /*vcl_size_t num_bytes = 0;
  for (vcl_size_t i=0; i<subblocks_A.size(); ++i)
    num_bytes += 8*subblocks_A[i].size1()*subblocks_A[i].size2();*/
  //std::cout << "Memory for FSPAI matrix: " << num_bytes / (1024.0 * 1024.0) << " MB" << std::endl;

  //
  // Step 4: Solve for y_k
  //
  //std::cout << "computeFSPAI(): Cholesky-solve..." << std::endl;
  for (vcl_size_t i=0; i<y_k.size(); ++i)
  {
    if (subblocks_A[i].size1() > 0) //block might be empty...
    {
      //y_k[i].resize(subblocks_A[i].size1());
      //std::cout << "y_k[" << i << "]: ";
      //for (vcl_size_t j=0; j<y_k[i].size(); ++j)
      //  std::cout << y_k[i][j] << " ";
      //std::cout << std::endl;
      cholesky_solve(subblocks_A[i], y_k[i]);
    }
  }


  //
  // Step 5: Set up Cholesky factors L and L_trans
  //
  //std::cout << "computeFSPAI(): Computing L..." << std::endl;
  L.resize(A.size1(), A.size2(), false);
  L.reserve(A.nnz(), false);
  L_trans.resize(A.size1(), A.size2(), false);
  L_trans.reserve(A.nnz(), false);
  computeL(A, L, L_trans, y_k, J);

  //std::cout << "L: " << L << std::endl;
}



}
}
}
}

#endif
