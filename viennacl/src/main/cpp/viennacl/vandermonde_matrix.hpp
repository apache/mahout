#ifndef VIENNACL_VANDERMONDE_MATRIX_HPP
#define VIENNACL_VANDERMONDE_MATRIX_HPP

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

#include <cmath>

/** @file vandermonde_matrix.hpp
    @brief Implementation of the vandermonde_matrix class for efficient manipulation of Vandermonde matrices.  Experimental.
*/

#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"
#include "viennacl/ocl/backend.hpp"

#include "viennacl/fft.hpp"

#include "viennacl/linalg/vandermonde_matrix_operations.hpp"

namespace viennacl
{

/** @brief A Vandermonde matrix class
  *
  * @tparam NumericT   The underlying scalar type (either float or double)
  * @tparam AlignmentV    The internal memory size is given by (size()/AlignmentV + 1) * AlignmentV. AlignmentV must be a power of two. Best values or usually 4, 8 or 16, higher values are usually a waste of memory.
  */
template<class NumericT, unsigned int AlignmentV>
class vandermonde_matrix
{
public:
  typedef viennacl::backend::mem_handle                                                              handle_type;
  typedef scalar<typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<NumericT>::ResultType>   value_type;

  /**
       * @brief The default constructor. Does not allocate any memory.
       *
       */
  explicit vandermonde_matrix() {}

  /**
       * @brief         Creates the matrix with the given size
       *
       * @param rows      Number of rows of the matrix
       * @param cols      Number of columns of the matrix
       */
  explicit vandermonde_matrix(vcl_size_t rows, vcl_size_t cols) : elements_(rows)
  {
    assert(rows == cols && bool("Vandermonde matrix must be square in this release!"));
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
       * @brief Returns an internal viennacl::vector, which represents a Vandermonde matrix elements
       *
       */
  viennacl::vector<NumericT, AlignmentV> & elements() { return elements_; }
  viennacl::vector<NumericT, AlignmentV> const & elements() const { return elements_; }

  /**
       * @brief Returns the number of rows of the matrix
       */
  vcl_size_t size1() const { return elements_.size(); }

  /**
       * @brief Returns the number of columns of the matrix
       */
  vcl_size_t size2() const { return elements_.size(); }

  /** @brief Returns the internal size of matrix representtion.
      *   Usually required for launching OpenCL kernels only
      *
      *   @return Internal size of matrix representation
      */
  vcl_size_t internal_size() const { return elements_.internal_size(); }

  /**
       * @brief Read-write access to a base element of the matrix
       *
       * @param row_index  Row index of accessed element
       * @return Proxy for matrix entry
       */
  entry_proxy<NumericT> operator()(vcl_size_t row_index)
  {
    return elements_[row_index];
  }

  /**
       * @brief Read access to a element of the matrix
       *
       * @param row_index  Row index of accessed element
       * @param col_index  Column index of accessed element
       * @return Proxy for matrix entry
       */
  NumericT operator()(vcl_size_t row_index, vcl_size_t col_index) const
  {
    assert(row_index < size1() && col_index < size2() && bool("Invalid access"));

    return pow(elements_[row_index], static_cast<int>(col_index));
  }

private:
  vandermonde_matrix(vandermonde_matrix const &) {}
  vandermonde_matrix & operator=(vandermonde_matrix const & t);

  viennacl::vector<NumericT, AlignmentV> elements_;
};

/** @brief Copies a Vandermonde matrix from the std::vector to the OpenCL device (either GPU or multi-core CPU)
  *
  *
  * @param cpu_vec   A std::vector on the host.
  * @param gpu_mat   A vandermonde_matrix from ViennaCL
  */
template<typename NumericT, unsigned int AlignmentV>
void copy(std::vector<NumericT>& cpu_vec, vandermonde_matrix<NumericT, AlignmentV>& gpu_mat)
{
  assert(cpu_vec.size() == gpu_mat.size1()  && bool("Size mismatch"));
  copy(cpu_vec, gpu_mat.elements());
}

/** @brief Copies a Vandermonde matrix from the OpenCL device (either GPU or multi-core CPU) to the std::vector
  *
  *
  * @param gpu_mat   A vandermonde_matrix from ViennaCL
  * @param cpu_vec   A std::vector on the host.
  */
template<typename NumericT, unsigned int AlignmentV>
void copy(vandermonde_matrix<NumericT, AlignmentV>& gpu_mat, std::vector<NumericT>& cpu_vec)
{
  assert(cpu_vec.size() == gpu_mat.size1() && bool("Size mismatch"));
  copy(gpu_mat.elements(), cpu_vec);
}

/** @brief Copies a Vandermonde matrix from the OpenCL device (either GPU or multi-core CPU) to the matrix-like object
  *
  *
  * @param vander_src   A vandermonde_matrix from ViennaCL
  * @param com_dst   A matrix-like object
  */
template<typename NumericT, unsigned int AlignmentV, typename MatrixT>
void copy(vandermonde_matrix<NumericT, AlignmentV>& vander_src, MatrixT& com_dst)
{
  assert(vander_src.size1() == viennacl::traits::size1(com_dst) && bool("Size mismatch"));
  assert(vander_src.size2() == viennacl::traits::size2(com_dst) && bool("Size mismatch"));

  vcl_size_t size = vander_src.size1();
  std::vector<NumericT> tmp(size);
  copy(vander_src, tmp);

  for (vcl_size_t i = 0; i < size; i++)
    for (vcl_size_t j = 0; j < size; j++)
      com_dst(i, j) = std::pow(tmp[i], static_cast<int>(j));

}

/** @brief Copies a the matrix-like object to the Vandermonde matrix from the OpenCL device (either GPU or multi-core CPU)
  *
  *
  * @param com_src   A std::vector on the host
  * @param vander_dst   A vandermonde_matrix from ViennaCL
  */
template<typename NumericT, unsigned int AlignmentV, typename MatrixT>
void copy(MatrixT& com_src, vandermonde_matrix<NumericT, AlignmentV>& vander_dst)
{
  assert( (vander_dst.size1() == 0 || vander_dst.size1() == viennacl::traits::size1(com_src)) && bool("Size mismatch"));
  assert( (vander_dst.size2() == 0 || vander_dst.size2() == viennacl::traits::size2(com_src)) && bool("Size mismatch"));

  vcl_size_t size = vander_dst.size1();
  std::vector<NumericT> tmp(size);

  for (vcl_size_t i = 0; i < size; i++)
    tmp[i] = com_src(i, 1);

  copy(tmp, vander_dst);
}

/*template<typename NumericT, unsigned int AlignmentV, unsigned int VECTOR_AlignmentV>
  void prod_impl(vandermonde_matrix<NumericT, AlignmentV>& mat,
                 vector<NumericT, VECTOR_AlignmentV>& vec,
                 vector<NumericT, VECTOR_AlignmentV>& result) {
      assert(mat.size1() == vec.size());

      fft::vandermonde_prod<NumericT>(mat.handle(), vec.handle(), result.handle(), mat.size1());
  } */

/** @brief Prints the matrix. Output is compatible to boost::numeric::ublas
  *
  * @param s            STL output stream
  * @param gpu_matrix   A ViennaCL Vandermonde matrix
  */
template<class NumericT, unsigned int AlignmentV>
std::ostream & operator<<(std::ostream& s, vandermonde_matrix<NumericT, AlignmentV>& gpu_matrix)
{
  vcl_size_t size = gpu_matrix.size1();
  std::vector<NumericT> tmp(size);
  copy(gpu_matrix, tmp);
  s << "[" << size << "," << size << "](\n";

  for (vcl_size_t i = 0; i < size; i++)
  {
    s << "(";
    for (vcl_size_t j = 0; j < size; j++)
    {
      s << pow(tmp[i], static_cast<NumericT>(j));
      if (j < (size - 1))
        s << ",";
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
  struct op_executor<vector_base<T>, op_assign, vector_expression<const vandermonde_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vandermonde_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
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
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const vandermonde_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vandermonde_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), temp);
      lhs += temp;
    }
  };

  template<typename T, unsigned int A>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const vandermonde_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vandermonde_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), temp);
      lhs -= temp;
    }
  };


  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const vandermonde_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vandermonde_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs());
      viennacl::linalg::prod_impl(rhs.lhs(), temp, lhs);
    }
  };

  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const vandermonde_matrix<T, A>, vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vandermonde_matrix<T, A>, vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs());
      viennacl::vector<T> temp_result(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), temp, temp_result);
      lhs += temp_result;
    }
  };

  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const vandermonde_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vandermonde_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
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

#endif // VIENNACL_VANDERMONDE_MATRIX_HPP
