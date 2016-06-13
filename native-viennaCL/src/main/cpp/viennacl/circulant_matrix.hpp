#ifndef VIENNACL_CIRCULANT_MATRIX_HPP
#define VIENNACL_CIRCULANT_MATRIX_HPP

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

/** @file circulant_matrix.hpp
    @brief Implementation of the circulant_matrix class for efficient manipulation of circulant matrices.  Experimental.
*/

#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"
#include "viennacl/ocl/backend.hpp"

#include "viennacl/linalg/circulant_matrix_operations.hpp"

#include "viennacl/fft.hpp"

namespace viennacl
{
/** @brief A Circulant matrix class
  *
  * @tparam NumericT  The underlying scalar type (either float or double)
  * @tparam AlignmentV   The internal memory size is given by (size()/AlignmentV + 1) * AlignmentV. AlignmentV must be a power of two. Best values or usually 4, 8 or 16, higher values are usually a waste of memory.
  */
template<class NumericT, unsigned int AlignmentV>
class circulant_matrix
{
public:
  typedef viennacl::backend::mem_handle                                                              handle_type;
  typedef scalar<typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<NumericT>::ResultType>   value_type;

  /**
    * @brief The default constructor. Does not allocate any memory.
    *
    */
  explicit circulant_matrix() {}

  /**
    * @brief         Creates the matrix with the given size
    *
    * @param rows      Number of rows of the matrix
    * @param cols      Number of columns of the matrix
    */
  explicit circulant_matrix(vcl_size_t rows, vcl_size_t cols) : elements_(rows)
  {
    assert(rows == cols && bool("Circulant matrix must be square!"));
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
    * @brief Returns an internal viennacl::vector, which represents a circulant matrix elements
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
    * @brief Read-write access to a single element of the matrix
    *
    * @param row_index  Row index of accessed element
    * @param col_index  Column index of accessed element
    * @return Proxy for matrix entry
    */
  entry_proxy<NumericT> operator()(vcl_size_t row_index, vcl_size_t col_index)
  {
    long index = static_cast<long>(row_index) - static_cast<long>(col_index);

    assert(row_index < size1() && col_index < size2() && bool("Invalid access"));

    while (index < 0)
      index += static_cast<long>(size1());
    return elements_[static_cast<vcl_size_t>(index)];
  }

  /**
    * @brief += operation for circulant matrices
    *
    * @param that Matrix which will be added
    * @return Result of addition
    */
  circulant_matrix<NumericT, AlignmentV>& operator +=(circulant_matrix<NumericT, AlignmentV>& that)
  {
    elements_ += that.elements();
    return *this;
  }

private:
  circulant_matrix(circulant_matrix const &) {}
  circulant_matrix & operator=(circulant_matrix const & t);

  viennacl::vector<NumericT, AlignmentV> elements_;
};

/** @brief Copies a circulant matrix from the std::vector to the OpenCL device (either GPU or multi-core CPU)
  *
  *
  * @param cpu_vec   A std::vector on the host.
  * @param gpu_mat   A circulant_matrix from ViennaCL
  */
template<typename NumericT, unsigned int AlignmentV>
void copy(std::vector<NumericT>& cpu_vec, circulant_matrix<NumericT, AlignmentV>& gpu_mat)
{
  assert( (gpu_mat.size1() == 0 || cpu_vec.size() == gpu_mat.size1()) && bool("Size mismatch"));
  copy(cpu_vec, gpu_mat.elements());
}

/** @brief Copies a circulant matrix from the OpenCL device (either GPU or multi-core CPU) to the std::vector
  *
  *
  * @param gpu_mat   A circulant_matrix from ViennaCL
  * @param cpu_vec   A std::vector on the host.
  */
template<typename NumericT, unsigned int AlignmentV>
void copy(circulant_matrix<NumericT, AlignmentV>& gpu_mat, std::vector<NumericT>& cpu_vec)
{
  assert(cpu_vec.size() == gpu_mat.size1() && bool("Size mismatch"));
  copy(gpu_mat.elements(), cpu_vec);
}

/** @brief Copies a circulant matrix from the OpenCL device (either GPU or multi-core CPU) to the matrix-like object
  *
  *
  * @param circ_src   A circulant_matrix from ViennaCL
  * @param com_dst   A matrix-like object
  */
template<typename NumericT, unsigned int AlignmentV, typename MatrixT>
void copy(circulant_matrix<NumericT, AlignmentV>& circ_src, MatrixT& com_dst)
{
  vcl_size_t size = circ_src.size1();
  assert(size == viennacl::traits::size1(com_dst) && bool("Size mismatch"));
  assert(size == viennacl::traits::size2(com_dst) && bool("Size mismatch"));
  std::vector<NumericT> tmp(size);
  copy(circ_src, tmp);

  for (vcl_size_t i = 0; i < size; i++)
  {
    for (vcl_size_t j = 0; j < size; j++)
    {
      long index = static_cast<long>(i) - static_cast<long>(j);
      if (index < 0)
        index += static_cast<long>(size);
      com_dst(i, j) = tmp[static_cast<vcl_size_t>(index)];
    }
  }
}

/** @brief Copies a the matrix-like object to the circulant matrix from the OpenCL device (either GPU or multi-core CPU)
  *
  *
  * @param com_src   A std::vector on the host
  * @param circ_dst   A circulant_matrix from ViennaCL
  */
template<typename NumericT, unsigned int AlignmentV, typename MatrixT>
void copy(MatrixT& com_src, circulant_matrix<NumericT, AlignmentV>& circ_dst)
{
  assert( (circ_dst.size1() == 0 || circ_dst.size1() == viennacl::traits::size1(com_src)) && bool("Size mismatch"));
  assert( (circ_dst.size2() == 0 || circ_dst.size2() == viennacl::traits::size2(com_src)) && bool("Size mismatch"));

  vcl_size_t size = viennacl::traits::size1(com_src);

  std::vector<NumericT> tmp(size);

  for (vcl_size_t i = 0; i < size; i++) tmp[i] = com_src(i, 0);

  copy(tmp, circ_dst);
}

/*namespace linalg
  {
    template<typename NumericT, unsigned int AlignmentV, unsigned int VECTOR_AlignmentV>
    void prod_impl(circulant_matrix<NumericT, AlignmentV> const & mat,
                    vector<NumericT, VECTOR_AlignmentV> const & vec,
                    vector<NumericT, VECTOR_AlignmentV>& result) {
        viennacl::vector<NumericT, VECTOR_AlignmentV> circ(mat.elements().size() * 2);
        fft::real_to_complex(mat.elements(), circ, mat.elements().size());

        viennacl::vector<NumericT, VECTOR_AlignmentV> tmp(vec.size() * 2);
        viennacl::vector<NumericT, VECTOR_AlignmentV> tmp2(vec.size() * 2);

        fft::real_to_complex(vec, tmp, vec.size());
        fft::convolve(circ, tmp, tmp2);
        fft::complex_to_real(tmp2, result, vec.size());
    }
  }*/

/** @brief Prints the matrix. Output is compatible to boost::numeric::ublas
  *
  * @param s            STL output stream
  * @param gpu_matrix   A ViennaCL circulant matrix
  */
template<class NumericT, unsigned int AlignmentV>
std::ostream & operator<<(std::ostream& s, circulant_matrix<NumericT, AlignmentV>& gpu_matrix)
{
  vcl_size_t size = gpu_matrix.size1();
  std::vector<NumericT> tmp(size);
  copy(gpu_matrix, tmp);
  s << "[" << size << "," << size << "](";

  for (vcl_size_t i = 0; i < size; i++)
  {
    s << "(";
    for (vcl_size_t j = 0; j < size; j++)
    {
      long index = static_cast<long>(i) - static_cast<long>(j);
      if (index < 0) index = static_cast<long>(size) + index;
      s << tmp[vcl_size_t(index)];
      //s << index;
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
  struct op_executor<vector_base<T>, op_assign, vector_expression<const circulant_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const circulant_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
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
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const circulant_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const circulant_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), temp);
      lhs += temp;
    }
  };

  template<typename T, unsigned int A>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const circulant_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const circulant_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), temp);
      lhs -= temp;
    }
  };


  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const circulant_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const circulant_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs());
      viennacl::linalg::prod_impl(rhs.lhs(), temp, lhs);
    }
  };

  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const circulant_matrix<T, A>, vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const circulant_matrix<T, A>, vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs());
      viennacl::vector<T> temp_result(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), temp, temp_result);
      lhs += temp_result;
    }
  };

  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const circulant_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const circulant_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
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

#endif // VIENNACL_CIRCULANT_MATRIX_HPP
