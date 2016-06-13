#ifndef VIENNACL_HANKEL_MATRIX_HPP
#define VIENNACL_HANKEL_MATRIX_HPP

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


/** @file hankel_matrix.hpp
    @brief Implementation of the hankel_matrix class for efficient manipulation of Hankel matrices.  Experimental.
*/

#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"
#include "viennacl/ocl/backend.hpp"

#include "viennacl/toeplitz_matrix.hpp"
#include "viennacl/fft.hpp"

#include "viennacl/linalg/hankel_matrix_operations.hpp"

namespace viennacl
{
/** @brief A Hankel matrix class
  *
  * @tparam NumericT   The underlying scalar type (either float or double)
  * @tparam AlignmentV    The internal memory size is given by (size()/AlignmentV + 1) * AlignmentV. AlignmentV must be a power of two. Best values or usually 4, 8 or 16, higher values are usually a waste of memory.
  */
template<class NumericT, unsigned int AlignmentV>
class hankel_matrix
{
public:
  typedef viennacl::backend::mem_handle                                                              handle_type;
  typedef scalar<typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<NumericT>::ResultType>   value_type;

  /**
       * @brief The default constructor. Does not allocate any memory.
       *
       */
  explicit hankel_matrix() {}

  /**
       * @brief         Creates the matrix with the given size
       *
       * @param rows      Number of rows of the matrix
       * @param cols      Number of columns of the matrix
       */
  explicit hankel_matrix(vcl_size_t rows, vcl_size_t cols) : elements_(rows, cols)
  {
    assert(rows == cols && bool("Hankel matrix must be square!"));
    (void)cols;  // avoid 'unused parameter' warning in optimized builds
  }

  /** @brief Resizes the matrix.
      *   Existing entries can be preserved
      *
      * @param sz         New size of matrix
      * @param preserve   If true, existing values are preserved.
      */
  void resize(vcl_size_t sz, bool preserve = true)
  {
    elements_.resize(sz, preserve);
  }

  /** @brief Returns the OpenCL handle
      *
      *   @return OpenCL handle
      */
  handle_type const & handle() const { return elements_.handle(); }

  /**
       * @brief Returns an internal viennacl::toeplitz_matrix, which represents a Hankel matrix elements
       *
       */
  toeplitz_matrix<NumericT, AlignmentV> & elements() { return elements_; }
  toeplitz_matrix<NumericT, AlignmentV> const & elements() const { return elements_; }

  /**
       * @brief Returns the number of rows of the matrix
       */
  vcl_size_t size1() const { return elements_.size1(); }

  /**
       * @brief Returns the number of columns of the matrix
       */
  vcl_size_t size2() const { return elements_.size2(); }

  /** @brief Returns the internal size of matrix representtion.
      *   Usually required for launching OpenCL kernels only
      *
      *   @return Internal size of matrix representation
      */
  vcl_size_t internal_size() const { return elements_.internal_size(); }

  /**
       * @brief Read-write access to a element of the matrix
       *
       * @param row_index  Row index of accessed element
       * @param col_index  Column index of accessed element
       * @return Proxy for matrix entry
       */
  entry_proxy<NumericT> operator()(unsigned int row_index, unsigned int col_index)
  {
    assert(row_index < size1() && col_index < size2() && bool("Invalid access"));

    return elements_(size1() - row_index - 1, col_index);
  }

  /**
       * @brief += operation for Hankel matrices
       *
       * @param that Matrix which will be added
       * @return Result of addition
       */
  hankel_matrix<NumericT, AlignmentV>& operator +=(hankel_matrix<NumericT, AlignmentV>& that)
  {
    elements_ += that.elements();
    return *this;
  }

private:
  hankel_matrix(hankel_matrix const &) {}
  hankel_matrix & operator=(hankel_matrix const & t);

  toeplitz_matrix<NumericT, AlignmentV> elements_;
};

/** @brief Copies a Hankel matrix from the std::vector to the OpenCL device (either GPU or multi-core CPU)
  *
  *
  * @param cpu_vec   A std::vector on the host.
  * @param gpu_mat   A hankel_matrix from ViennaCL
  */
template<typename NumericT, unsigned int AlignmentV>
void copy(std::vector<NumericT> const & cpu_vec, hankel_matrix<NumericT, AlignmentV> & gpu_mat)
{
  assert((gpu_mat.size1() * 2 - 1)  == cpu_vec.size() && bool("Size mismatch"));

  copy(cpu_vec, gpu_mat.elements());
}

/** @brief Copies a Hankel matrix from the OpenCL device (either GPU or multi-core CPU) to the std::vector
  *
  *
  * @param gpu_mat   A hankel_matrix from ViennaCL
  * @param cpu_vec   A std::vector on the host.
  */
template<typename NumericT, unsigned int AlignmentV>
void copy(hankel_matrix<NumericT, AlignmentV> const & gpu_mat, std::vector<NumericT> & cpu_vec)
{
  assert((gpu_mat.size1() * 2 - 1)  == cpu_vec.size() && bool("Size mismatch"));

  copy(gpu_mat.elements(), cpu_vec);
}

/** @brief Copies a Hankel matrix from the OpenCL device (either GPU or multi-core CPU) to the matrix-like object
  *
  *
  * @param han_src   A hankel_matrix from ViennaCL
  * @param com_dst   A matrix-like object
  */
template<typename NumericT, unsigned int AlignmentV, typename MatrixT>
void copy(hankel_matrix<NumericT, AlignmentV> const & han_src, MatrixT& com_dst)
{
  assert( (viennacl::traits::size1(com_dst) == han_src.size1()) && bool("Size mismatch") );
  assert( (viennacl::traits::size2(com_dst) == han_src.size2()) && bool("Size mismatch") );

  vcl_size_t size = han_src.size1();
  std::vector<NumericT> tmp(size * 2 - 1);
  copy(han_src, tmp);

  for (vcl_size_t i = 0; i < size; i++)
    for (vcl_size_t j = 0; j < size; j++)
      com_dst(i, j) = tmp[i + j];
}

/** @brief Copies a the matrix-like object to the Hankel matrix from the OpenCL device (either GPU or multi-core CPU)
  *
  *
  * @param com_src   A std::vector on the host
  * @param han_dst   A hankel_matrix from ViennaCL
  */
template<typename NumericT, unsigned int AlignmentV, typename MatrixT>
void copy(MatrixT const & com_src, hankel_matrix<NumericT, AlignmentV>& han_dst)
{
  assert( (han_dst.size1() == 0 || viennacl::traits::size1(com_src) == han_dst.size1()) && bool("Size mismatch") );
  assert( (han_dst.size2() == 0 || viennacl::traits::size2(com_src) == han_dst.size2()) && bool("Size mismatch") );
  assert( viennacl::traits::size2(com_src) == viennacl::traits::size1(com_src) && bool("Logic error: non-square Hankel matrix!") );

  vcl_size_t size = viennacl::traits::size1(com_src);

  std::vector<NumericT> tmp(2*size - 1);

  for (vcl_size_t i = 0; i < size; i++)
    tmp[i] = com_src(0, i);

  for (vcl_size_t i = 1; i < size; i++)
    tmp[size + i - 1] = com_src(size - 1, i);

  viennacl::copy(tmp, han_dst);
}

/*template<typename NumericT, unsigned int AlignmentV, unsigned int VECTOR_AlignmentV>
  void prod_impl(hankel_matrix<NumericT, AlignmentV>& mat,
                 vector<NumericT, VECTOR_AlignmentV>& vec,
                 vector<NumericT, VECTOR_AlignmentV>& result)
  {
      prod_impl(mat.elements(), vec, result);
      fft::reverse(result);
  }*/

template<class NumericT, unsigned int AlignmentV>
std::ostream & operator<<(std::ostream & s, hankel_matrix<NumericT, AlignmentV>& gpu_matrix)
{
  vcl_size_t size = gpu_matrix.size1();
  std::vector<NumericT> tmp(2*size - 1);
  copy(gpu_matrix, tmp);
  s << "[" << size << "," << size << "](";

  for (vcl_size_t i = 0; i < size; i++)
  {
    s << "(";
    for (vcl_size_t j = 0; j < size; j++)
    {
      s << tmp[i + j];
      //s << (int)i - (int)j;
      if (j < (size - 1)) s << ",";
    }
    s << ")";
  }
  s << ")";
  return s;
}

//
// Specify available operations:
//

/** \cond */

namespace linalg
{
namespace detail
{
  // x = A * y
  template<typename T, unsigned int A>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const hankel_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hankel_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
    {
      // check for the special case x = A * x
      if (viennacl::traits::handle(lhs) == viennacl::traits::handle(rhs.rhs()))
      {
        viennacl::vector<T> temp(lhs);
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), temp);
        lhs = temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs);
    }
  };

  template<typename T, unsigned int A>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const hankel_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hankel_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), temp);
      lhs += temp;
    }
  };

  template<typename T, unsigned int A>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const hankel_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hankel_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), temp);
      lhs -= temp;
    }
  };


  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const hankel_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hankel_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs());
      viennacl::linalg::prod_impl(rhs.lhs(), temp, lhs);
    }
  };

  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const hankel_matrix<T, A>, vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hankel_matrix<T, A>, vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs());
      viennacl::vector<T> temp_result(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), temp, temp_result);
      lhs += temp_result;
    }
  };

  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const hankel_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hankel_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs());
      viennacl::vector<T> temp_result(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), temp, temp_result);
      lhs -= temp_result;
    }
  };



} // namespace detail
} // namespace linalg

/** \endcond */
}
#endif // VIENNACL_HANKEL_MATRIX_HPP
