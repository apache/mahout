#ifndef VIENNACL_MATRIX_HPP_
#define VIENNACL_MATRIX_HPP_

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

/** @file viennacl/matrix.hpp
    @brief Implementation of the dense matrix class
*/

#include "viennacl/forwards.h"
#include "viennacl/detail/matrix_def.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include "viennacl/linalg/sparse_matrix_operations.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/tools/matrix_size_deducer.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/row_major.hpp"

namespace viennacl
{

//#ifdef VIENNACL_WITH_OPENCL
//  template<class NumericT, class DISTRIBUTION>
//  rand::random_matrix_t<NumericT, DISTRIBUTION> random_matrix(unsigned int size1, unsigned int size2, DISTRIBUTION const & distribution){
//      return rand::random_matrix_t<NumericT,DISTRIBUTION>(size1,size2,distribution);
//  }
//#endif

/** @brief Expression template class for representing a tree of expressions which ultimately result in a matrix.
  *
  * @tparam LHS   The left hand side of the expression tree
  * @tparam RHS   The right hand side of the expression tree
  * @tparam OP    The operator to apply to LHS and RHS to obtain the result.
  */
template<typename LHS, typename RHS, typename OP>
class matrix_expression
{
  typedef typename viennacl::result_of::reference_if_nonscalar<LHS>::type     lhs_reference_type;
  typedef typename viennacl::result_of::reference_if_nonscalar<RHS>::type     rhs_reference_type;

public:
  typedef vcl_size_t       size_type;

  matrix_expression(LHS & lhs, RHS & rhs) : lhs_(lhs), rhs_(rhs) {}

  /** @brief Get left hand side operand
    */
  LHS & lhs() const { return lhs_; }
  /** @brief Get right hand side operand
    */
  RHS & rhs() const { return rhs_; }

  /** @brief Returns the size of the result vector */
  vcl_size_t size1() const { return viennacl::tools::MATRIX_SIZE_DEDUCER<LHS, RHS, OP>::size1(lhs_, rhs_); }
  vcl_size_t size2() const { return viennacl::tools::MATRIX_SIZE_DEDUCER<LHS, RHS, OP>::size2(lhs_, rhs_); }

private:
  /** @brief The left hand side operand */
  lhs_reference_type lhs_;
  /** @brief The right hand side operand */
  rhs_reference_type rhs_;
};


/** @brief A tag indicating iteration along increasing row index of a matrix */
struct row_iteration {};

/** @brief A tag indicating iteration along increasing columns index of a matrix */
struct col_iteration {};

//STL-like iterator. TODO: STL-compliance...
/** @brief uBLAS-like iterator class for iterating over the entries of a dense matrix. */
template<typename ROWCOL, typename MatrixT>
class matrix_iterator
{
  typedef matrix_iterator<ROWCOL, MatrixT>    self_type;
public:
  typedef typename MatrixT::value_type       value_type;

  matrix_iterator(MatrixT & mat,
                  vcl_size_t start_row,
                  vcl_size_t start_col) : mat_(mat), row_(start_row), col_(start_col) {}

  value_type operator*(void) { return mat_(row_, col_); }
  self_type & operator++(void) { viennacl::tools::MATRIX_ITERATOR_INCREMENTER<ROWCOL, MatrixT>::apply(mat_, row_, col_); return *this; }
  self_type operator++(int) { self_type tmp = *this; ++(*this); return tmp; }

  bool operator==(self_type const & other) { return (row_ == other.row_) && (col_ == other.col_); }
  bool operator!=(self_type const & other) { return !(*this == other); }

  vcl_size_t index1() { return row_; }
  vcl_size_t index2() { return col_; }

  MatrixT & operator()(void) const { return mat_; }

private:
  MatrixT & mat_;
  vcl_size_t row_;
  vcl_size_t col_;
};

/** @brief Creates the matrix with the given dimensions
*
* @param rows     Number of rows
* @param columns  Number of columns
* @param ctx      Optional context in which the matrix is created (one out of multiple OpenCL contexts, CUDA, host)
*/
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT>::matrix_base(size_type rows, size_type columns, bool is_row_major, viennacl::context ctx)
  : size1_(rows), size2_(columns), start1_(0), start2_(0), stride1_(1), stride2_(1),
    internal_size1_(viennacl::tools::align_to_multiple<size_type>(rows, dense_padding_size)),
    internal_size2_(viennacl::tools::align_to_multiple<size_type>(columns, dense_padding_size)),
    row_major_fixed_(true), row_major_(is_row_major)
{
  if (rows > 0 && columns > 0)
  {
    viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), ctx);
    clear();
  }
}

/** @brief Constructor for creating a matrix_range or matrix_stride from some other matrix/matrix_range/matrix_stride */

template<class NumericT, typename SizeT, typename DistanceT>
template<typename LHS, typename RHS, typename OP>
matrix_base<NumericT, SizeT, DistanceT>::matrix_base(matrix_expression<const LHS, const RHS, OP> const & proxy) :
  size1_(viennacl::traits::size1(proxy)), size2_(viennacl::traits::size2(proxy)), start1_(0), start2_(0), stride1_(1), stride2_(1),
  internal_size1_(viennacl::tools::align_to_multiple<size_type>(size1_, dense_padding_size)),
  internal_size2_(viennacl::tools::align_to_multiple<size_type>(size2_, dense_padding_size)),
  row_major_fixed_(true), row_major_(viennacl::traits::row_major(proxy))
{
  elements_.switch_active_handle_id(viennacl::traits::active_handle_id(proxy));
  if (internal_size() > 0)
  {
    viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), viennacl::traits::context(proxy));
    clear();
    self_type::operator=(proxy);
  }
}

// CUDA or host memory:
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT>::matrix_base(NumericT * ptr_to_mem, viennacl::memory_types mem_type,
                                                        size_type mat_size1, size_type mat_start1, size_type mat_stride1, size_type mat_internal_size1,
                                                        size_type mat_size2, size_type mat_start2, size_type mat_stride2, size_type mat_internal_size2,
                                                        bool is_row_major)
  : size1_(mat_size1), size2_(mat_size2),
    start1_(mat_start1), start2_(mat_start2),
    stride1_(mat_stride1), stride2_(mat_stride2),
    internal_size1_(mat_internal_size1), internal_size2_(mat_internal_size2),
    row_major_fixed_(true), row_major_(is_row_major)
{
  if (mem_type == viennacl::CUDA_MEMORY)
  {
#ifdef VIENNACL_WITH_CUDA
    elements_.switch_active_handle_id(viennacl::CUDA_MEMORY);
    elements_.cuda_handle().reset(reinterpret_cast<char*>(ptr_to_mem));
    elements_.cuda_handle().inc(); //prevents that the user-provided memory is deleted once the vector object is destroyed.
#else
    throw cuda_not_available_exception();
#endif
  }
  else if (mem_type == viennacl::MAIN_MEMORY)
  {
    elements_.switch_active_handle_id(viennacl::MAIN_MEMORY);
    elements_.ram_handle().reset(reinterpret_cast<char*>(ptr_to_mem));
    elements_.ram_handle().inc(); //prevents that the user-provided memory is deleted once the vector object is destroyed.
  }

  elements_.raw_size(sizeof(NumericT) * internal_size());
}

#ifdef VIENNACL_WITH_OPENCL
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT>::matrix_base(cl_mem mem, size_type rows, size_type columns, bool is_row_major, viennacl::context ctx)
  : size1_(rows), size2_(columns),
    start1_(0), start2_(0),
    stride1_(1), stride2_(1),
    internal_size1_(rows), internal_size2_(columns),
    row_major_fixed_(true), row_major_(is_row_major)
{
  elements_.switch_active_handle_id(viennacl::OPENCL_MEMORY);
  elements_.opencl_handle() = mem;
  elements_.opencl_handle().inc();  //prevents that the user-provided memory is deleted once the vector object is destroyed.
  elements_.opencl_handle().context(ctx.opencl_context());
  elements_.raw_size(sizeof(NumericT)*internal_size());
}

template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT>::matrix_base(cl_mem mem, viennacl::context ctx,
                                                        size_type mat_size1, size_type mat_start1, size_type mat_stride1, size_type mat_internal_size1,
                                                        size_type mat_size2, size_type mat_start2, size_type mat_stride2, size_type mat_internal_size2,
                                                        bool is_row_major)
  : size1_(mat_size1), size2_(mat_size2),
    start1_(mat_start1), start2_(mat_start2),
    stride1_(mat_stride1), stride2_(mat_stride2),
    internal_size1_(mat_internal_size1), internal_size2_(mat_internal_size2),
    row_major_fixed_(true), row_major_(is_row_major)
{
  elements_.switch_active_handle_id(viennacl::OPENCL_MEMORY);
  elements_.opencl_handle() = mem;
  elements_.opencl_handle().inc();  //prevents that the user-provided memory is deleted once the vector object is destroyed.
  elements_.opencl_handle().context(ctx.opencl_context());
  elements_.raw_size(sizeof(NumericT)*internal_size());
}
#endif

// Copy CTOR
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT>::matrix_base(const matrix_base<NumericT, SizeT, DistanceT> & other) :
  size1_(other.size1()), size2_(other.size2()), start1_(0), start2_(0), stride1_(1), stride2_(1),
  internal_size1_(viennacl::tools::align_to_multiple<size_type>(size1_, dense_padding_size)),
  internal_size2_(viennacl::tools::align_to_multiple<size_type>(size2_, dense_padding_size)),
  row_major_fixed_(true), row_major_(other.row_major())
{
  elements_.switch_active_handle_id(viennacl::traits::active_handle_id(other));
  if (internal_size() > 0)
  {
    viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), viennacl::traits::context(other));
    clear();
    self_type::operator=(other);
  }
}

// Conversion CTOR
template<typename NumericT, typename SizeT, typename DistanceT>
template<typename OtherNumericT>
matrix_base<NumericT, SizeT, DistanceT>::matrix_base(const matrix_base<OtherNumericT, SizeT, DistanceT> & other) :
  size1_(other.size1()), size2_(other.size2()), start1_(0), start2_(0), stride1_(1), stride2_(1),
  internal_size1_(viennacl::tools::align_to_multiple<size_type>(size1_, dense_padding_size)),
  internal_size2_(viennacl::tools::align_to_multiple<size_type>(size2_, dense_padding_size)),
  row_major_fixed_(true), row_major_(other.row_major())
{
  elements_.switch_active_handle_id(viennacl::traits::active_handle_id(other));
  if (internal_size() > 0)
  {
    viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), viennacl::traits::context(other));
    clear();
    self_type::operator=(other);
  }
}

template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator=(const self_type & other)  //enables implicit conversions
{
  if (&other==this)
    return *this;

  if (internal_size() == 0)
  {
    if (other.internal_size() == 0)
      return *this;
    if (!row_major_fixed_)
      row_major_ = other.row_major();
    resize(other.size1(), other.size2(), false);
  }

  viennacl::linalg::am(*this,
                       other, cpu_value_type(1.0), 1, false, false);
  return *this;
}

// Conversion assignment
template<class NumericT, typename SizeT, typename DistanceT>
template<typename OtherNumericT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator=(const matrix_base<OtherNumericT, SizeT, DistanceT> & other)
{
  if (internal_size() == 0)
  {
    if (other.internal_size() == 0)
      return *this;
    if (!row_major_fixed_)
      row_major_ = other.row_major();
    resize(other.size1(), other.size2(), false);
  }

  viennacl::linalg::convert(*this, other);
  return *this;
}

/** @brief Implementation of the operation m1 = m2 @ alpha, where @ denotes either multiplication or division, and alpha is either a CPU or a GPU scalar
*
* @param proxy  An expression template proxy class.
*/
template<class NumericT, typename SizeT, typename DistanceT>
template<typename LHS, typename RHS, typename OP>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator=(const matrix_expression<const LHS, const RHS, OP> & proxy)
{
  assert(  (viennacl::traits::size1(proxy) == size1() || size1() == 0)
           && (viennacl::traits::size2(proxy) == size2() || size2() == 0)
           && bool("Incompatible matrix sizes!"));
  if (internal_size() == 0 && viennacl::traits::size1(proxy) > 0 && viennacl::traits::size2(proxy) > 0)
  {
    size1_ = viennacl::traits::size1(proxy);
    size2_ = viennacl::traits::size2(proxy);
    internal_size1_ = viennacl::tools::align_to_multiple<size_type>(size1_, dense_padding_size);
    internal_size2_ = viennacl::tools::align_to_multiple<size_type>(size2_, dense_padding_size);
    if (!row_major_fixed_)
      row_major_ = viennacl::traits::row_major(proxy);
    viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), viennacl::traits::context(proxy));
    if (size1_ != internal_size1_ || size2_ != internal_size2_)
      clear();
  }

  if (internal_size() > 0)
    linalg::detail::op_executor<self_type, op_assign, matrix_expression<const LHS, const RHS, OP> >::apply(*this, proxy);

  return *this;
}


// A = trans(B)
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator=(const matrix_expression<const self_type, const self_type, op_trans> & proxy)
{
  if ( internal_size() == 0 && viennacl::traits::size1(proxy) > 0 && viennacl::traits::size2(proxy) > 0 )
  {
    size1_ = viennacl::traits::size1(proxy);
    size2_ = viennacl::traits::size2(proxy);
    internal_size1_ = viennacl::tools::align_to_multiple<size_type>(size1_, dense_padding_size);
    internal_size2_ = viennacl::tools::align_to_multiple<size_type>(size2_, dense_padding_size);
    if (!row_major_fixed_)
      row_major_ = viennacl::traits::row_major(proxy);
  }

  if ( handle() == proxy.lhs().handle() )
  {
    viennacl::matrix_base<NumericT> temp(proxy.lhs().size2(), proxy.lhs().size1(),proxy.lhs().row_major());
    viennacl::linalg::trans(proxy, temp);
    if ( proxy.lhs().size1() != proxy.lhs().size2() )
      this->resize(proxy.lhs().size2(), proxy.lhs().size1());
    elements_ = temp.handle();
  }
  else
  {
    if ( proxy.lhs().size1() != proxy.lhs().size2() )
      this->resize(proxy.lhs().size2(), proxy.lhs().size1(), false);
    viennacl::linalg::trans(proxy, *this);
  }
  return *this;
}

template<class NumericT, typename SizeT, typename DistanceT>
template<typename LHS, typename RHS, typename OP>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator+=(const matrix_expression<const LHS, const RHS, OP> & proxy)
{
  assert(  (viennacl::traits::size1(proxy) == size1())
           && (viennacl::traits::size2(proxy) == size2())
           && bool("Incompatible matrix sizes!"));
  assert( (size1() > 0) && bool("Vector not yet initialized!") );
  assert( (size2() > 0) && bool("Vector not yet initialized!") );

  linalg::detail::op_executor<self_type, op_inplace_add, matrix_expression<const LHS, const RHS, OP> >::apply(*this, proxy);

  return *this;
}

template<class NumericT, typename SizeT, typename DistanceT>
template<typename LHS, typename RHS, typename OP>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator-=(const matrix_expression<const LHS, const RHS, OP> & proxy)
{
  assert(  (viennacl::traits::size1(proxy) == size1())
           && (viennacl::traits::size2(proxy) == size2())
           && bool("Incompatible matrix sizes!"));
  assert( (size1() > 0) && bool("Vector not yet initialized!") );
  assert( (size2() > 0) && bool("Vector not yet initialized!") );

  linalg::detail::op_executor<self_type, op_inplace_sub, matrix_expression<const LHS, const RHS, OP> >::apply(*this, proxy);

  return *this;
}

/** @brief Assigns the supplied identity matrix to the matrix. */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator = (identity_matrix<NumericT> const & m)
{
  assert( (m.size1() == size1_ || size1_ == 0) && bool("Size mismatch!") );
  assert( (m.size2() == size2_ || size2_ == 0) && bool("Size mismatch!") );

  if (internal_size() == 0)
  {
    size1_ = m.size1();
    size2_ = m.size2();
    internal_size1_ = viennacl::tools::align_to_multiple<size_type>(size1_, dense_padding_size);
    internal_size2_ = viennacl::tools::align_to_multiple<size_type>(size2_, dense_padding_size);
    if (internal_size() > 0)
    {
      viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), m.context());
      clear();
    }
  }
  else
    viennacl::linalg::matrix_assign(*this, NumericT(0));

  if (internal_size() > 0)
    viennacl::linalg::matrix_diagonal_assign(*this, m(0,0));

  return *this;
}

/** @brief Assigns the supplied zero matrix to the matrix. */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator = (zero_matrix<NumericT> const & m)
{
  assert( (m.size1() == size1_ || size1_ == 0) && bool("Size mismatch!") );
  assert( (m.size2() == size2_ || size2_ == 0) && bool("Size mismatch!") );

  if (internal_size() == 0)
  {
    size1_ = m.size1();
    size2_ = m.size2();
    internal_size1_ = viennacl::tools::align_to_multiple<size_type>(size1_, dense_padding_size);
    internal_size2_ = viennacl::tools::align_to_multiple<size_type>(size2_, dense_padding_size);
    if (internal_size() > 0)
    {
      viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), m.context());
      clear();
    }
  }
  else
    viennacl::linalg::matrix_assign(*this, NumericT(0));

  return *this;
}

/** @brief Assigns the supplied scalar vector to the matrix. */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator = (scalar_matrix<NumericT> const & m)
{
  assert( (m.size1() == size1_ || size1_ == 0) && bool("Size mismatch!") );
  assert( (m.size2() == size2_ || size2_ == 0) && bool("Size mismatch!") );

  if (internal_size() == 0)
  {
    size1_ = m.size1();
    size2_ = m.size2();
    internal_size1_ = viennacl::tools::align_to_multiple<size_type>(size1_, dense_padding_size);
    internal_size2_ = viennacl::tools::align_to_multiple<size_type>(size2_, dense_padding_size);
    if (internal_size() > 0)
    {
      viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), m.context());
      clear();
    }
  }

  if (internal_size() > 0)
  {
    viennacl::linalg::matrix_assign(*this, m(0,0));
  }

  return *this;
}


//read-write access to an element of the matrix/matrix_range/matrix_slice
/** @brief Read-write access to a single element of the matrix/matrix_range/matrix_slice
*/
template<class NumericT, typename SizeT, typename DistanceT>
entry_proxy<NumericT> matrix_base<NumericT, SizeT, DistanceT>::operator()(size_type row_index, size_type col_index)
{
  if (row_major_)
    return entry_proxy<NumericT>(row_major::mem_index(start1_ + stride1_ * row_index, start2_ + stride2_ * col_index, internal_size1(), internal_size2()), elements_);
  return entry_proxy<NumericT>(column_major::mem_index(start1_ + stride1_ * row_index, start2_ + stride2_ * col_index, internal_size1(), internal_size2()), elements_);
}

/** @brief Read access to a single element of the matrix/matrix_range/matrix_slice
*/
template<class NumericT, typename SizeT, typename DistanceT>
const_entry_proxy<NumericT> matrix_base<NumericT, SizeT, DistanceT>::operator()(size_type row_index, size_type col_index) const
{
  if (row_major_)
    return const_entry_proxy<NumericT>(row_major::mem_index(start1_ + stride1_ * row_index, start2_ + stride2_ * col_index, internal_size1(), internal_size2()), elements_);
  return const_entry_proxy<NumericT>(column_major::mem_index(start1_ + stride1_ * row_index, start2_ + stride2_ * col_index, internal_size1(), internal_size2()), elements_);
}

//
// Operator overloads for enabling implicit conversions:
//
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator += (const matrix_base<NumericT, SizeT, DistanceT> & other)
{
  viennacl::linalg::ambm(*this,
                         *this, NumericT(1.0), 1, false, false,
                         other, NumericT(1.0), 1, false, false);
  return *this;
}

template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator -= (const matrix_base<NumericT, SizeT, DistanceT> & other)
{
  viennacl::linalg::ambm(*this,
                         *this, NumericT(1.0), 1, false, false,
                         other, NumericT(1.0), 1, false, true);
  return *this;
}

/** @brief Scales a matrix by a char (8-bit integer) value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator *= (char val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, false, false);
  return *this;
}

/** @brief Scales a matrix by a char (8-bit integer) value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator *= (short val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, false, false);
  return *this;
}

/** @brief Scales a matrix by a char (8-bit integer) value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator *= (int val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, false, false);
  return *this;
}

/** @brief Scales a matrix by a char (8-bit integer) value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator *= (long val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, false, false);
  return *this;
}

/** @brief Scales a matrix by a char (8-bit integer) value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator *= (float val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, false, false);
  return *this;
}

/** @brief Scales a matrix by a char (8-bit integer) value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator *= (double val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, false, false);
  return *this;
}



/** @brief Scales this matrix by a char (8-bit integer) value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator /= (char val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, true, false);
  return *this;
}

/** @brief Scales this matrix by a short integer value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator /= (short val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, true, false);
  return *this;
}

/** @brief Scales this matrix by an integer value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator /= (int val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, true, false);
  return *this;
}

/** @brief Scales this matrix by a long integer value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator /= (long val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, true, false);
  return *this;
}

/** @brief Scales this matrix by a single precision floating point value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator /= (float val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, true, false);
  return *this;
}

/** @brief Scales this matrix by a double precision floating point value */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_base<NumericT, SizeT, DistanceT> & matrix_base<NumericT, SizeT, DistanceT>::operator /= (double val)
{
  viennacl::linalg::am(*this,
                       *this, NumericT(val), 1, true, false);
  return *this;
}


/** @brief Sign flip for the matrix. Emulated to be equivalent to -1.0 * matrix */
template<class NumericT, typename SizeT, typename DistanceT>
matrix_expression<const matrix_base<NumericT, SizeT, DistanceT>, const NumericT, op_mult> matrix_base<NumericT, SizeT, DistanceT>::operator-() const
{
  return matrix_expression<const self_type, const NumericT, op_mult>(*this, NumericT(-1));
}

template<class NumericT, typename SizeT, typename DistanceT>
void matrix_base<NumericT, SizeT, DistanceT>::clear() { viennacl::linalg::matrix_assign(*this, NumericT(0), true); }


template<class NumericT, typename SizeT, typename DistanceT>
void matrix_base<NumericT, SizeT, DistanceT>::resize(size_type rows, size_type columns, bool preserve)
{
  assert( (rows > 0 && columns > 0) && bool("Check failed in matrix::resize(): Number of rows and columns must be positive!"));

  if (preserve && internal_size() > 0)
  {
    //get old entries:
    std::vector< NumericT > old_entries(internal_size());
    viennacl::backend::memory_read(elements_, 0, sizeof(NumericT)*internal_size(), &(old_entries[0]));

    //set up entries of new matrix:
    std::vector< NumericT > new_entries(  viennacl::tools::align_to_multiple<vcl_size_t>(rows,    dense_padding_size)
                                          * viennacl::tools::align_to_multiple<vcl_size_t>(columns, dense_padding_size));
    for (size_type i=0; i<rows; ++i)
    {
      if (i >= size1_)
        continue;

      for (size_type j=0; j<columns; ++j)
      {
        if (j >= size2_)
          continue;
        if (row_major_)
          new_entries[row_major::mem_index(i, j, viennacl::tools::align_to_multiple<vcl_size_t>(rows, dense_padding_size), viennacl::tools::align_to_multiple<vcl_size_t>(columns, dense_padding_size))]
              = old_entries[row_major::mem_index(i, j, internal_size1(), internal_size2())];
        else
          new_entries[column_major::mem_index(i, j, viennacl::tools::align_to_multiple<vcl_size_t>(rows, dense_padding_size), viennacl::tools::align_to_multiple<vcl_size_t>(columns, dense_padding_size))]
              = old_entries[column_major::mem_index(i, j, internal_size1(), internal_size2())];
      }
    }

    //copy new entries to GPU:
    size1_ = rows;
    size2_ = columns;
    internal_size1_ = viennacl::tools::align_to_multiple<size_type>(size1_, dense_padding_size);
    internal_size2_ = viennacl::tools::align_to_multiple<size_type>(size2_, dense_padding_size);
    viennacl::backend::memory_create(elements_, sizeof(NumericT)*new_entries.size(), viennacl::traits::context(elements_), &(new_entries[0]));
  }
  else //discard old entries:
  {
    size1_ = rows;
    size2_ = columns;
    internal_size1_ = viennacl::tools::align_to_multiple<size_type>(size1_, dense_padding_size);
    internal_size2_ = viennacl::tools::align_to_multiple<size_type>(size2_, dense_padding_size);

    viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), viennacl::traits::context(elements_));
    clear();
  }
}


/** @brief A dense matrix class
*
* @tparam NumericT   The underlying scalar type (either float or double)
* @tparam F            Storage layout: Either row_major or column_major
* @tparam AlignmentV   The internal memory size is given by (size()/AlignmentV + 1) * AlignmentV. AlignmentV must be a power of two. Best values or usually 4, 8 or 16, higher values are usually a waste of memory.
*/
template<class NumericT, typename F, unsigned int AlignmentV>
class matrix : public matrix_base<NumericT>
{
  typedef matrix<NumericT, F, AlignmentV>          self_type;
  typedef matrix_base<NumericT>                   base_type;
public:
  typedef typename base_type::size_type             size_type;

  /** @brief The default constructor. Does not allocate any memory. */
  explicit matrix() : base_type(static_cast<bool>(viennacl::is_row_major<F>::value)) {}

  /** @brief Creates the matrix with the given dimensions
    *
    * @param rows     Number of rows
    * @param columns  Number of columns
    * @param ctx      Optional context in which the matrix is created (one out of multiple OpenCL contexts, CUDA, host)
    */
  explicit matrix(size_type rows, size_type columns, viennacl::context ctx = viennacl::context()) : base_type(rows, columns, viennacl::is_row_major<F>::value, ctx) {}

  /** @brief Wraps a CUDA or host buffer provided by the user.
    *
    * @param ptr_to_mem   The pointer to existing memory
    * @param mem_type     Type of the memory (either viennacl::CUDA_MEMORY if available, or viennacl::HOST_MEMORY)
    * @param rows         Number of rows of the matrix
    * @param cols         Number of columns of the matrix
    */
  explicit matrix(NumericT * ptr_to_mem, viennacl::memory_types mem_type, size_type rows, size_type cols)
    : base_type(ptr_to_mem, mem_type,
                rows, 0, 1, rows,
                cols, 0, 1, cols,
                viennacl::is_row_major<F>::value) {}


  /** @brief Wraps a CUDA or host buffer provided by the user including padding of rows and columns.
    *
    * @param ptr_to_mem          The pointer to existing memory
    * @param mem_type            Type of the memory (either viennacl::CUDA_MEMORY if available, or viennacl::HOST_MEMORY)
    * @param rows                Number of rows of the matrix
    * @param internal_row_count  Number of rows including padding the buffer by e.g. zeros.
    * @param cols                Number of columns of the matrix
    * @param internal_col_count  Number of columns including padding the buffer by e.g. zeros.
    */
  explicit matrix(NumericT * ptr_to_mem, viennacl::memory_types mem_type,
                  size_type rows, size_type internal_row_count,
                  size_type cols, size_type internal_col_count)
    : base_type(ptr_to_mem, mem_type,
                rows, 0, 1, internal_row_count,
                cols, 0, 1, internal_col_count,
                true, viennacl::is_row_major<F>::value) {}

#ifdef VIENNACL_WITH_OPENCL
  explicit matrix(cl_mem mem, size_type rows, size_type columns) : base_type(mem, rows, columns, viennacl::is_row_major<F>::value) {}
#endif

  template<typename LHS, typename RHS, typename OP>
  matrix(matrix_expression< LHS, RHS, OP> const & proxy) : base_type(proxy) {}

  /** @brief Creates the matrix from the supplied identity matrix. */
  matrix(identity_matrix<NumericT> const & m) : base_type(m.size1(), m.size2(), viennacl::is_row_major<F>::value, m.context())
  {
    if (base_type::internal_size() > 0)
      base_type::operator=(m);
  }

  /** @brief Creates the matrix from the supplied zero matrix. */
  matrix(zero_matrix<NumericT> const & m) : base_type(m.size1(), m.size2(), viennacl::is_row_major<F>::value, m.context())
  {
    if (base_type::internal_size() > 0)
      base_type::operator=(m);
  }

  /** @brief Creates the matrix from the supplied scalar matrix. */
  matrix(scalar_matrix<NumericT> const & m) : base_type(m.size1(), m.size2(), viennacl::is_row_major<F>::value, m.context())
  {
    if (base_type::internal_size() > 0)
      base_type::operator=(m);
  }

  matrix(const base_type & other) : base_type(other.size1(), other.size2(), viennacl::is_row_major<F>::value, viennacl::traits::context(other))
  {
    base_type::operator=(other);
  }


  //copy constructor:
  matrix(const self_type & other) : base_type(other.size1(), other.size2(), viennacl::is_row_major<F>::value, viennacl::traits::context(other))
  {
    base_type::operator=(other);
  }


  /*template<typename M1>
    self_type & operator=(const matrix_expression< const M1, const M1, op_trans> & proxy)
    {
      self_type temp(proxy.lhs());
      *this = trans(temp);
      return *this;
    }*/

  using base_type::operator=;

  // the following are needed for Visual Studio:
  template<typename OtherNumericT, typename F2>
  base_type & operator=(viennacl::matrix<OtherNumericT, F2> const & B)                          { return base_type::operator=(static_cast<viennacl::matrix_base<OtherNumericT> const &>(B)); }

  template<typename OtherNumericT, typename F2>
  base_type & operator=(viennacl::matrix_range<viennacl::matrix<OtherNumericT, F2> > const & B) { return base_type::operator=(static_cast<viennacl::matrix_base<OtherNumericT> const &>(B)); }

  template<typename OtherNumericT, typename F2>
  base_type & operator=(viennacl::matrix_slice<viennacl::matrix<OtherNumericT, F2> > const & B) { return base_type::operator=(static_cast<viennacl::matrix_base<OtherNumericT> const &>(B)); }

  /** @brief Resizes the matrix.
    *   Existing entries can optionally be preserved
    *
    * @param rows       New number of rows
    * @param columns    New number of columns
    * @param preserve   If true, existing values are preserved.
    */
  void resize(size_type rows, size_type columns, bool preserve = true)
  {
    base_type::resize(rows, columns, preserve);
  }

}; //matrix



/** @brief Prints the matrix. Output is compatible to boost::numeric::ublas
*
* @param s            STL output stream
* @param gpu_matrix   A dense ViennaCL matrix
*/
template<class NumericT>
std::ostream & operator<<(std::ostream & s, const matrix_base<NumericT> & gpu_matrix)
{
  typedef typename matrix_base<NumericT>::size_type      size_type;

  std::vector<NumericT> tmp(gpu_matrix.internal_size());
  viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(NumericT) * gpu_matrix.internal_size(), &(tmp[0]));

  s << "[" << gpu_matrix.size1() << "," << gpu_matrix.size2() << "]";

  s << "(";
  for (size_type i = 0; i < gpu_matrix.size1(); ++i)
  {
    s << "(";
    for (size_type j = 0; j < gpu_matrix.size2(); ++j)
    {
      if (gpu_matrix.row_major())
        s << tmp[row_major::mem_index(i * gpu_matrix.stride1() + gpu_matrix.start1(), j * gpu_matrix.stride2() + gpu_matrix.start2(), gpu_matrix.internal_size1(), gpu_matrix.internal_size2())];
      else
        s << tmp[column_major::mem_index(i * gpu_matrix.stride1() + gpu_matrix.start1(), j * gpu_matrix.stride2() + gpu_matrix.start2(), gpu_matrix.internal_size1(), gpu_matrix.internal_size2())];

      if (j < gpu_matrix.size2() - 1)
        s << ",";
    }
    s << ")";
    if (i < gpu_matrix.size1() - 1)
      s << ",";
  }
  s << ")";
  return s;
}

/** @brief Prints the matrix. Output is compatible to boost::numeric::ublas
*
* @param s            STL output stream
* @param expr         A matrix expression
*/
template<typename LHS, typename RHS, typename OP>
std::ostream & operator<<(std::ostream & s, const matrix_expression<LHS, RHS, OP> & expr)
{
  typedef typename viennacl::tools::CPU_SCALAR_TYPE_DEDUCER< typename tools::CONST_REMOVER<LHS>::ResultType >::ResultType     ScalarType;

  matrix<ScalarType> temp = expr;
  s << temp;
  return s;
}

/** @brief Returns an expression template class representing a transposed matrix */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>
trans(const matrix_base<NumericT> & mat)
{
  return matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>(mat, mat);
}

/** @brief Returns an expression template class representing the transposed matrix expression */
template<typename LhsT, typename RhsT, typename OpT>
matrix_expression< const matrix_expression<const LhsT, const RhsT, OpT>, const matrix_expression<const LhsT, const RhsT, OpT>, op_trans>
trans(const matrix_expression<const LhsT, const RhsT, OpT> & proxy)
{
  return matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                           const matrix_expression<const LhsT, const RhsT, OpT>,
                           op_trans>(proxy, proxy);
}

//diag():
template<typename NumericT>
vector_expression< const matrix_base<NumericT>, const int, op_matrix_diag>
diag(const matrix_base<NumericT> & A, int k = 0)
{
  return vector_expression< const matrix_base<NumericT>, const int, op_matrix_diag>(A, k);
}

template<typename NumericT>
matrix_expression< const vector_base<NumericT>, const int, op_vector_diag>
diag(const vector_base<NumericT> & v, int k = 0)
{
  return matrix_expression< const vector_base<NumericT>, const int, op_vector_diag>(v, k);
}

// row():
template<typename NumericT, typename F>
vector_expression< const matrix_base<NumericT, F>, const unsigned int, op_row>
row(const matrix_base<NumericT, F> & A, unsigned int i)
{
  return vector_expression< const matrix_base<NumericT, F>, const unsigned int, op_row>(A, i);
}

// column():
template<typename NumericT, typename F>
vector_expression< const matrix_base<NumericT, F>, const unsigned int, op_column>
column(const matrix_base<NumericT, F> & A, unsigned int j)
{
  return vector_expression< const matrix_base<NumericT, F>, const unsigned int, op_column>(A, j);
}

/////////////////////// transfer operations: //////////////////////////////////////

//
//cpu to gpu, generic type:
//
/** @brief Copies a dense matrix from the host (CPU) to the OpenCL device (GPU or multi-core CPU)
*
* @param cpu_matrix   A dense matrix on the host. Type requirements: .size1() returns number of rows, .size2() returns number of columns. Access to entries via operator()
* @param gpu_matrix   A dense ViennaCL matrix
*/
template<typename CPUMatrixT, typename NumericT, typename F, unsigned int AlignmentV>
void copy(const CPUMatrixT & cpu_matrix,
          matrix<NumericT, F, AlignmentV> & gpu_matrix )
{
  typedef typename matrix<NumericT, F, AlignmentV>::size_type      size_type;

  //std::cout << "Copying CPUMatrixT!" << std::endl;
  //std::cout << "Size at begin: " << gpu_matrix.size1() << ", " << gpu_matrix.size2() << std::endl;
  if (gpu_matrix.size1() == 0 || gpu_matrix.size2() == 0)
  {
    gpu_matrix.resize(cpu_matrix.size1(),
                      cpu_matrix.size2(), false);
  }

  assert( (gpu_matrix.size1() == cpu_matrix.size1()) && (gpu_matrix.size2() == cpu_matrix.size2()) && bool("Matrix dimensions mismatch.") );

  std::vector<NumericT> data(gpu_matrix.internal_size());
  for (size_type i = 0; i < gpu_matrix.size1(); ++i)
  {
    for (size_type j = 0; j < gpu_matrix.size2(); ++j)
      data[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())] = cpu_matrix(i,j);
  }

  viennacl::backend::memory_write(gpu_matrix.handle(), 0, sizeof(NumericT) * data.size(), &(data[0]));
  //gpu_matrix.elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, data);
  //std::cout << "Size at end: " << gpu_matrix.size1() << ", " << gpu_matrix.size2() << std::endl;
}

//
//cpu to gpu, STL type:
//
/** @brief Copies a dense STL-type matrix from the host (CPU) to the OpenCL device (GPU or multi-core CPU)
*
* @param cpu_matrix   A dense matrix on the host of type std::vector< std::vector<> >. cpu_matrix[i][j] returns the element in the i-th row and j-th columns (both starting with zero)
* @param gpu_matrix   A dense ViennaCL matrix
*/
template<typename NumericT, typename A1, typename A2, typename F, unsigned int AlignmentV>
void copy(const std::vector< std::vector<NumericT, A1>, A2> & cpu_matrix,
          matrix<NumericT, F, AlignmentV> & gpu_matrix )
{
  typedef typename matrix<NumericT, F, AlignmentV>::size_type      size_type;

  if (gpu_matrix.size1() == 0 || gpu_matrix.size2() == 0)
  {
    gpu_matrix.resize(cpu_matrix.size(),
                      cpu_matrix[0].size(),
        false);
  }

  assert( (gpu_matrix.size1() == cpu_matrix.size()) && bool("Matrix dimensions mismatch.") );

  std::vector<NumericT> data(gpu_matrix.internal_size());
  for (size_type i = 0; i < gpu_matrix.size1(); ++i)
  {
    assert( (gpu_matrix.size2() == cpu_matrix[i].size()) && bool("Matrix dimensions mismatch.") );

    for (size_type j = 0; j < gpu_matrix.size2(); ++j)
      data[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())] = cpu_matrix[i][j];
  }

  viennacl::backend::memory_write(gpu_matrix.handle(), 0, sizeof(NumericT) * data.size(), &(data[0]));
  //gpu_matrix.elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, data);
}


//
//cpu to gpu, another STL type:
//
/** @brief Copies a dense matrix from the host (CPU) to the OpenCL device (GPU or multi-core CPU) without temporary. Matrix-Layout on CPU must be equal to the matrix-layout on the GPU.
*
* See \ref manual-types-matrix in the manual for the underlying data layout including padding rows and columns by zero.
*
* @param cpu_matrix_begin   Pointer to the first matrix entry. Cf. iterator concept in STL
* @param cpu_matrix_end     Pointer past the last matrix entry. Cf. iterator concept in STL
* @param gpu_matrix         A dense ViennaCL matrix
*/
template<typename NumericT, typename F, unsigned int AlignmentV>
void fast_copy(NumericT * cpu_matrix_begin,
               NumericT * cpu_matrix_end,
               matrix<NumericT, F, AlignmentV> & gpu_matrix)
{
  if (gpu_matrix.internal_size() == 0)
    viennacl::backend::memory_create(gpu_matrix.handle(), sizeof(NumericT) * static_cast<vcl_size_t>(cpu_matrix_end - cpu_matrix_begin), viennacl::traits::context(gpu_matrix), cpu_matrix_begin);
  else
  {
    assert( (gpu_matrix.internal_size() >= static_cast<vcl_size_t>(cpu_matrix_end - cpu_matrix_begin)) && bool("fast_copy(): Matrix not large enough to fit data!"));
    viennacl::backend::memory_write(gpu_matrix.handle(), 0, sizeof(NumericT) * static_cast<vcl_size_t>(cpu_matrix_end - cpu_matrix_begin), cpu_matrix_begin);
  }
}

#ifdef VIENNACL_WITH_ARMADILLO
/** @brief Copies a dense Armadillo matrix from the host (CPU) to a ViennaCL vector
*
* @param arma_matrix   A dense MTL matrix. cpu_matrix(i, j) returns the element in the i-th row and j-th columns (both starting with zero)
* @param gpu_matrix   A dense ViennaCL matrix
*/
template<typename NumericT, typename F, unsigned int AlignmentV>
void copy(arma::Mat<NumericT>                       const & arma_matrix,
          viennacl::matrix<NumericT, F, AlignmentV>       & vcl_matrix)
{
  typedef typename viennacl::matrix<NumericT, F, AlignmentV>::size_type      size_type;

  if (vcl_matrix.size1() == 0 || vcl_matrix.size2() == 0)
  {
    vcl_matrix.resize(arma_matrix.n_rows,
                      arma_matrix.n_cols,
                      false);
  }
  else
  {
    assert(    (vcl_matrix.size1() == static_cast<vcl_size_t>(arma_matrix.n_rows))
            && (vcl_matrix.size2() == static_cast<vcl_size_t>(arma_matrix.n_cols))
            && bool("matrix size mismatch")
            );
  }

  // prepare buffer:
  viennacl::backend::typesafe_host_array<NumericT> data(vcl_matrix.handle(), vcl_matrix.internal_size());
  for (size_type j = 0; j < vcl_matrix.size2(); ++j) // iterate along columns is certainly fast for arma_matrix
    for (size_type i = 0; i < vcl_matrix.size1(); ++i)
      data.set(F::mem_index(i, j, vcl_matrix.internal_size1(), vcl_matrix.internal_size2()), arma_matrix(i,j));

  // copy over:
  viennacl::backend::memory_write(vcl_matrix.handle(), 0, data.raw_size(), data.get());
}
#endif

#ifdef VIENNACL_WITH_EIGEN
namespace detail
{
  template<typename EigenMatrixTypeT, typename NumericT, typename F, unsigned int AlignmentV>
  void copy_from_eigen_matrix(EigenMatrixTypeT const & cpu_matrix,
                              viennacl::matrix<NumericT, F, AlignmentV> & gpu_matrix)
  {
    typedef typename viennacl::matrix<NumericT, F, AlignmentV>::size_type      size_type;

    if (gpu_matrix.size1() == 0 || gpu_matrix.size2() == 0)
    {
      gpu_matrix.resize(cpu_matrix.rows(),
                        cpu_matrix.cols(),
                        false);
    }
    else
    {
      assert(    (gpu_matrix.size1() == static_cast<vcl_size_t>(cpu_matrix.rows()))
              && (gpu_matrix.size2() == static_cast<vcl_size_t>(cpu_matrix.cols()))
              && bool("matrix size mismatch")
              );
    }

    std::vector<NumericT> data(gpu_matrix.internal_size());
    for (size_type i = 0; i < gpu_matrix.size1(); ++i)
    {
      for (size_type j = 0; j < gpu_matrix.size2(); ++j)
        data[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())] = cpu_matrix(i,j);
    }

    viennacl::backend::memory_write(gpu_matrix.handle(), 0, sizeof(NumericT) * data.size(), &(data[0]));
  }

}

/** @brief Copies a dense Eigen matrix from the host (CPU) to a ViennaCL matrix (host, CUDA, or OpenCL)
*
* @param cpu_matrix   A dense MTL matrix. cpu_matrix(i, j) returns the element in the i-th row and j-th columns (both starting with zero)
* @param vcl_matrix   A dense ViennaCL matrix
*/
template<typename NumericT, int EigenOptions, typename F, unsigned int AlignmentV>
void copy(Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, EigenOptions> const & cpu_matrix,
          viennacl::matrix<NumericT, F, AlignmentV> & vcl_matrix)
{
  detail::copy_from_eigen_matrix(cpu_matrix, vcl_matrix);
}

/** @brief Copies a dense Eigen matrix from the host (CPU) to a ViennaCL matrix (host, CUDA, or OpenCL)
*
* @param cpu_matrix   A dense MTL matrix. cpu_matrix(i, j) returns the element in the i-th row and j-th columns (both starting with zero)
* @param vcl_matrix   A dense ViennaCL matrix
*/
template<typename NumericT, int EigenOptions, int EigenMatTypeV, typename EigenStrideT, typename F, unsigned int AlignmentV>
void copy(Eigen::Map<Eigen::Matrix<NumericT, Eigen::Dynamic, Eigen::Dynamic, EigenOptions>, EigenMatTypeV, EigenStrideT> const & cpu_matrix,
          viennacl::matrix<NumericT, F, AlignmentV> & vcl_matrix)
{
  detail::copy_from_eigen_matrix(cpu_matrix, vcl_matrix);
}
#endif

#ifdef VIENNACL_WITH_MTL4
/** @brief Copies a dense MTL matrix from the host (CPU) to the OpenCL device (GPU or multi-core CPU)
*
* @param cpu_matrix   A dense MTL matrix. cpu_matrix(i, j) returns the element in the i-th row and j-th columns (both starting with zero)
* @param gpu_matrix   A dense ViennaCL matrix
*/
template<typename NumericT, typename T, typename F, unsigned int AlignmentV>
void copy(const mtl::dense2D<NumericT, T>& cpu_matrix,
          matrix<NumericT, F, AlignmentV> & gpu_matrix)
{
  typedef typename matrix<NumericT, F, AlignmentV>::size_type      size_type;

  if (gpu_matrix.size1() == 0 || gpu_matrix.size2() == 0)
  {
    gpu_matrix.resize(cpu_matrix.num_rows(),
                      cpu_matrix.num_cols(),
                      false);
  }
  else
  {
    assert( (gpu_matrix.size1() == cpu_matrix.num_rows())
            && (gpu_matrix.size2() == cpu_matrix.num_cols())
            && bool("matrix size mismatch")
            );
  }

  std::vector<NumericT> data(gpu_matrix.internal_size());
  for (size_type i = 0; i < gpu_matrix.size1(); ++i)
  {
    for (size_type j = 0; j < gpu_matrix.size2(); ++j)
      data[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())] = cpu_matrix[i][j];
  }

  viennacl::backend::memory_write(gpu_matrix.handle(), 0, sizeof(NumericT) * data.size(), &(data[0]));
  //gpu_matrix.elements_ = viennacl::ocl::current_context().create_memory(CL_MEM_READ_WRITE, data);
}
#endif




//
//gpu to cpu, generic type
//
/** @brief Copies a dense matrix from the OpenCL device (GPU or multi-core CPU) to the host (CPU).
*
* @param gpu_matrix   A dense ViennaCL matrix
* @param cpu_matrix   A dense memory on the host. Must have at least as many rows and columns as the gpu_matrix! Type requirement: Access to entries via operator()
*/
template<typename CPUMatrixT, typename NumericT, typename F, unsigned int AlignmentV>
void copy(const matrix<NumericT, F, AlignmentV> & gpu_matrix,
          CPUMatrixT & cpu_matrix )
{
  typedef typename matrix<float, F, AlignmentV>::size_type      size_type;

  if ( (gpu_matrix.size1() > 0) && (gpu_matrix.size2() > 0) )
  {
    assert( viennacl::traits::size1(cpu_matrix) == gpu_matrix.size1() && bool("Matrix dimensions mismatch: rows"));

    std::vector<NumericT> temp_buffer(gpu_matrix.internal_size());
    viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(NumericT)*gpu_matrix.internal_size(), &(temp_buffer[0]));

    //now copy entries to cpu_matrix:
    for (size_type i = 0; i < gpu_matrix.size1(); ++i)
    {
      assert( viennacl::traits::size2(cpu_matrix) == gpu_matrix.size2() && bool("Matrix dimensions mismatch: columns"));
      for (size_type j = 0; j < gpu_matrix.size2(); ++j)
        cpu_matrix(i,j) = temp_buffer[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())];
    }
  }
}

//gpu to cpu, STL type
/** @brief Copies a dense matrix from the OpenCL device (GPU or multi-core CPU) to the host (CPU).
*
* @param gpu_matrix   A dense ViennaCL matrix
* @param cpu_matrix   A dense memory on the host using STL types, typically std::vector< std::vector<> > Must have at least as many rows and columns as the gpu_matrix! Type requirement: Access to entries via operator()
*/
template<typename NumericT, typename A1, typename A2, typename F, unsigned int AlignmentV>
void copy(const matrix<NumericT, F, AlignmentV> & gpu_matrix,
          std::vector< std::vector<NumericT, A1>, A2> & cpu_matrix)
{
  typedef typename matrix<float, F, AlignmentV>::size_type      size_type;

  if ( (gpu_matrix.size1() > 0) && (gpu_matrix.size2() > 0) )
  {
    assert( (cpu_matrix.size() == gpu_matrix.size1()) && bool("Matrix dimensions mismatch: rows"));

    std::vector<NumericT> temp_buffer(gpu_matrix.internal_size());
    viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(NumericT)*gpu_matrix.internal_size(), &(temp_buffer[0]));

    //now copy entries to cpu_matrix:
    for (size_type i = 0; i < gpu_matrix.size1(); ++i)
    {
      assert( (cpu_matrix[i].size() == gpu_matrix.size2()) && bool("Matrix dimensions mismatch: columns"));

      for (size_type j = 0; j < gpu_matrix.size2(); ++j)
        cpu_matrix[i][j] = temp_buffer[F::mem_index(i, j, gpu_matrix.internal_size1(), gpu_matrix.internal_size2())];
    }
  }
}

//gpu to cpu, STL type
/** @brief Copies a dense matrix from the OpenCL device (GPU or multi-core CPU) to the host (CPU).
*
* See \ref manual-types-matrix in the manual for the underlying data layout including padding rows and columns by zero.
*
* @param gpu_matrix         A dense ViennaCL matrix
* @param cpu_matrix_begin   Pointer to the output memory on the CPU. User must ensure that provided memory is large enough.
*/
template<typename NumericT, typename F, unsigned int AlignmentV>
void fast_copy(const matrix<NumericT, F, AlignmentV> & gpu_matrix,
               NumericT * cpu_matrix_begin)
{
  viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(NumericT)*gpu_matrix.internal_size(), cpu_matrix_begin);
}



/////////////////////// matrix operator overloads to follow ////////////////////////////////////////////


// operator +
/** @brief Generic 'catch-all' overload, which enforces a temporary if the expression tree gets too deep. */
template<typename LHS1, typename RHS1, typename OP1,
         typename LHS2, typename RHS2, typename OP2>
matrix_expression< const matrix_expression<const LHS1, const RHS1, OP1>,
const matrix_expression<const LHS2, const RHS2, OP2>,
op_add>
operator + (matrix_expression<const LHS1, const RHS1, OP1> const & proxy1,
            matrix_expression<const LHS2, const RHS2, OP2> const & proxy2)
{
  assert(    (viennacl::traits::size1(proxy1) == viennacl::traits::size1(proxy2))
             && (viennacl::traits::size2(proxy1) == viennacl::traits::size2(proxy2))
             && bool("Incompatible matrix sizes!"));
  return matrix_expression< const matrix_expression<const LHS1, const RHS1, OP1>,
      const matrix_expression<const LHS2, const RHS2, OP2>,
      op_add>(proxy1, proxy2);
}

template<typename LHS1, typename RHS1, typename OP1,
         typename NumericT>
matrix_expression< const matrix_expression<const LHS1, const RHS1, OP1>,
const matrix_base<NumericT>,
op_add>
operator + (matrix_expression<const LHS1, const RHS1, OP1> const & proxy1,
            matrix_base<NumericT> const & proxy2)
{
  assert(    (viennacl::traits::size1(proxy1) == viennacl::traits::size1(proxy2))
             && (viennacl::traits::size2(proxy1) == viennacl::traits::size2(proxy2))
             && bool("Incompatible matrix sizes!"));
  return matrix_expression< const matrix_expression<const LHS1, const RHS1, OP1>,
      const matrix_base<NumericT>,
      op_add>(proxy1, proxy2);
}

template<typename NumericT,
         typename LHS2, typename RHS2, typename OP2>
matrix_expression< const matrix_base<NumericT>,
const matrix_expression<const LHS2, const RHS2, OP2>,
op_add>
operator + (matrix_base<NumericT> const & proxy1,
            matrix_expression<const LHS2, const RHS2, OP2> const & proxy2)
{
  assert(    (viennacl::traits::size1(proxy1) == viennacl::traits::size1(proxy2))
             && (viennacl::traits::size2(proxy1) == viennacl::traits::size2(proxy2))
             && bool("Incompatible matrix sizes!"));
  return  matrix_expression< const matrix_base<NumericT>,
      const matrix_expression<const LHS2, const RHS2, OP2>,
      op_add>(proxy1, proxy2);
}

/** @brief Operator overload for m1 + m2, where m1 and m2 are either dense matrices, matrix ranges, or matrix slices. No mixing of different storage layouts allowed at the moment. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_add >
operator + (const matrix_base<NumericT> & m1, const matrix_base<NumericT> & m2)
{
  return matrix_expression< const matrix_base<NumericT>,
      const matrix_base<NumericT>,
      op_add > (m1, m2);
}


// operator -
template<typename LHS1, typename RHS1, typename OP1,
         typename LHS2, typename RHS2, typename OP2>
matrix_expression< const matrix_expression<const LHS1, const RHS1, OP1>,
const matrix_expression<const LHS2, const RHS2, OP2>,
op_sub>
operator - (matrix_expression<const LHS1, const RHS1, OP1> const & proxy1,
            matrix_expression<const LHS2, const RHS2, OP2> const & proxy2)
{
  assert(    (viennacl::traits::size1(proxy1) == viennacl::traits::size1(proxy2))
             && (viennacl::traits::size2(proxy1) == viennacl::traits::size2(proxy2))
             && bool("Incompatible matrix sizes!"));
  return matrix_expression< const matrix_expression<const LHS1, const RHS1, OP1>,
      const matrix_expression<const LHS2, const RHS2, OP2>,
      op_sub>(proxy1, proxy2);
}

template<typename LHS1, typename RHS1, typename OP1,
         typename NumericT>
matrix_expression< const matrix_expression<const LHS1, const RHS1, OP1>,
const matrix_base<NumericT>,
op_sub>
operator - (matrix_expression<const LHS1, const RHS1, OP1> const & proxy1,
            matrix_base<NumericT> const & proxy2)
{
  assert(    (viennacl::traits::size1(proxy1) == viennacl::traits::size1(proxy2))
             && (viennacl::traits::size2(proxy1) == viennacl::traits::size2(proxy2))
             && bool("Incompatible matrix sizes!"));
  return matrix_expression< const matrix_expression<const LHS1, const RHS1, OP1>,
      const matrix_base<NumericT>,
      op_sub>(proxy1, proxy2);
}

template<typename NumericT,
         typename LHS2, typename RHS2, typename OP2>
matrix_expression< const matrix_base<NumericT>,
const matrix_expression<const LHS2, const RHS2, OP2>,
op_sub>
operator - (matrix_base<NumericT> const & proxy1,
            matrix_expression<const LHS2, const RHS2, OP2> const & proxy2)
{
  assert(    (viennacl::traits::size1(proxy1) == viennacl::traits::size1(proxy2))
             && (viennacl::traits::size2(proxy1) == viennacl::traits::size2(proxy2))
             && bool("Incompatible matrix sizes!"));
  return  matrix_expression< const matrix_base<NumericT>,
      const matrix_expression<const LHS2, const RHS2, OP2>,
      op_sub>(proxy1, proxy2);
}

/** @brief Operator overload for m1 - m2, where m1 and m2 are either dense matrices, matrix ranges, or matrix slices. No mixing of different storage layouts allowed at the moment. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_sub >
operator - (const matrix_base<NumericT> & m1, const matrix_base<NumericT> & m2)
{
  return matrix_expression< const matrix_base<NumericT>,
      const matrix_base<NumericT>,
      op_sub > (m1, m2);
}



// operator *
/** @brief Operator overload for the expression alpha * m1, where alpha is a host scalar (float or double) and m1 is a ViennaCL matrix.
*
* @param value   The host scalar (float or double)
* @param m1      A ViennaCL matrix
*/
template<typename S1, typename NumericT>
typename viennacl::enable_if<    viennacl::is_any_scalar<S1>::value,
matrix_expression< const matrix_base<NumericT>, const S1, op_mult>
>::type
operator * (S1 const & value, matrix_base<NumericT> const & m1)
{
  return matrix_expression< const matrix_base<NumericT>, const S1, op_mult>(m1, value);
}

/** @brief Operator overload for the expression alpha * m1, where alpha is a char (8-bit integer) */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (char value, matrix_base<NumericT> const & m1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(value));
}

/** @brief Operator overload for the expression alpha * m1, where alpha is a short integer */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (short value, matrix_base<NumericT> const & m1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(value));
}

/** @brief Operator overload for the expression alpha * m1, where alpha is an integer */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (int value, matrix_base<NumericT> const & m1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(value));
}

/** @brief Operator overload for the expression alpha * m1, where alpha is a long integer */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (long value, matrix_base<NumericT> const & m1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(value));
}

/** @brief Operator overload for the expression alpha * m1, where alpha is a single precision floating point value */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (float value, matrix_base<NumericT> const & m1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(value));
}

/** @brief Operator overload for the expression alpha * m1, where alpha is a double precision floating point value */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (double value, matrix_base<NumericT> const & m1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(value));
}



/** @brief Operator overload for the multiplication of a matrix expression with a scalar from the right, e.g. (beta * m1) * alpha. Here, beta * m1 is wrapped into a matrix_expression and then multiplied with alpha from the right.
*
* @param proxy   Left hand side matrix expression
* @param val     Right hand side scalar
*/
template<typename LHS, typename RHS, typename OP, typename S1>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
matrix_expression< const matrix_expression< LHS, RHS, OP>, const S1, op_mult> >::type
operator * (matrix_expression< LHS, RHS, OP> const & proxy,
            S1 const & val)
{
  return matrix_expression< const matrix_expression< LHS, RHS, OP>, const S1, op_mult>(proxy, val);
}


/** @brief Operator overload for the multiplication of a matrix expression with a ViennaCL scalar from the left, e.g. alpha * (beta * m1). Here, beta * m1 is wrapped into a matrix_expression and then multiplied with alpha from the left.
*
* @param val     Right hand side scalar
* @param proxy   Left hand side matrix expression
*/
template<typename S1, typename LHS, typename RHS, typename OP>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
matrix_expression< const matrix_expression< LHS, RHS, OP>, const S1, op_mult> >::type
operator * (S1 const & val,
            matrix_expression< LHS, RHS, OP> const & proxy)
{
  return matrix_expression< const matrix_expression< LHS, RHS, OP>, const S1, op_mult>(proxy, val);
}

/** @brief Scales the matrix by a GPU scalar 'alpha' and returns an expression template
*/
template<typename NumericT, typename S1>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
matrix_expression< const matrix_base<NumericT>, const S1, op_mult> >::type
operator * (matrix_base<NumericT> const & m1, S1 const & s1)
{
  return matrix_expression< const matrix_base<NumericT>, const S1, op_mult>(m1, s1);
}

/** @brief Scales the matrix by a char (8-bit integer) 'alpha' and returns an expression template. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (matrix_base<NumericT> const & m1, char s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(s1));
}

/** @brief Scales the matrix by a short integer 'alpha' and returns an expression template. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (matrix_base<NumericT> const & m1, short s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(s1));
}

/** @brief Scales the matrix by an integer 'alpha' and returns an expression template. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (matrix_base<NumericT> const & m1, int s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(s1));
}

/** @brief Scales the matrix by a long integer 'alpha' and returns an expression template. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (matrix_base<NumericT> const & m1, long s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(s1));
}

/** @brief Scales the matrix by a single precision floating point number 'alpha' and returns an expression template. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (matrix_base<NumericT> const & m1, float s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(s1));
}

/** @brief Scales the matrix by a double precision floating point number 'alpha' and returns an expression template. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>
operator * (matrix_base<NumericT> const & m1, double s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_mult>(m1, NumericT(s1));
}


// operator *=

/** @brief Scales a matrix by a GPU scalar value */
template<typename NumericT, typename S1>
typename viennacl::enable_if< viennacl::is_scalar<S1>::value, matrix_base<NumericT> & >::type
operator *= (matrix_base<NumericT> & m1, S1 const & gpu_val)
{
  bool is_sign_flip = viennacl::is_flip_sign_scalar<S1>::value;
  viennacl::linalg::am(m1,
                       m1, gpu_val, 1, false, is_sign_flip ? true : false);
  return m1;
}

/** @brief Scales a matrix by a char (8-bit) value. */
template<typename NumericT>
matrix_base<NumericT> &
operator *= (matrix_base<NumericT> & m1, char gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, NumericT(gpu_val), 1, false, false);
  return m1;
}

/** @brief Scales a matrix by a short integer value. */
template<typename NumericT>
matrix_base<NumericT> &
operator *= (matrix_base<NumericT> & m1, short gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, NumericT(gpu_val), 1, false, false);
  return m1;
}

/** @brief Scales a matrix by an integer value. */
template<typename NumericT>
matrix_base<NumericT> &
operator *= (matrix_base<NumericT> & m1, int gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, NumericT(gpu_val), 1, false, false);
  return m1;
}

/** @brief Scales a matrix by a long integer value. */
template<typename NumericT>
matrix_base<NumericT> &
operator *= (matrix_base<NumericT> & m1, long gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, NumericT(gpu_val), 1, false, false);
  return m1;
}

/** @brief Scales a matrix by a single precision floating point value. */
template<typename NumericT>
matrix_base<NumericT> &
operator *= (matrix_base<NumericT> & m1, float gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, NumericT(gpu_val), 1, false, false);
  return m1;
}

/** @brief Scales a matrix by a double precision floating point value. */
template<typename NumericT>
matrix_base<NumericT> &
operator *= (matrix_base<NumericT> & m1, double gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, NumericT(gpu_val), 1, false, false);
  return m1;
}



// operator /


/** @brief Operator overload for the division of a matrix expression by a scalar from the right, e.g. (beta * m1) / alpha. Here, beta * m1 is wrapped into a matrix_expression and then divided by alpha.
*
* @param proxy   Left hand side matrix expression
* @param val     Right hand side scalar
*/
template<typename LHS, typename RHS, typename OP, typename S1>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
matrix_expression< const matrix_expression<const LHS, const RHS, OP>, const S1, op_div> >::type
operator / (matrix_expression<const LHS, const RHS, OP> const & proxy,
            S1 const & val)
{
  return matrix_expression< const matrix_expression<const LHS, const RHS, OP>, const S1, op_div>(proxy, val);
}


/** @brief Returns an expression template for scaling the matrix by a GPU scalar 'alpha' */
template<typename NumericT, typename S1>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
matrix_expression< const matrix_base<NumericT>, const S1, op_div> >::type
operator / (matrix_base<NumericT> const & m1, S1 const & s1)
{
  return matrix_expression< const matrix_base<NumericT>, const S1, op_div>(m1, s1);
}

/** @brief Returns an expression template for scaling the matrix by a char (8-bit integer) 'alpha'. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>
operator / (matrix_base<NumericT> const & m1, char s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>(m1, NumericT(s1));
}

/** @brief Returns an expression template for scaling the matrix by a short integer 'alpha'. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>
operator / (matrix_base<NumericT> const & m1, short s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>(m1, NumericT(s1));
}

/** @brief Returns an expression template for scaling the matrix by an integer 'alpha'. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>
operator / (matrix_base<NumericT> const & m1, int s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>(m1, NumericT(s1));
}

/** @brief Returns an expression template for scaling the matrix by a long integer 'alpha'. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>
operator / (matrix_base<NumericT> const & m1, long s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>(m1, NumericT(s1));
}

/** @brief Returns an expression template for scaling the matrix by a single precision floating point number 'alpha'. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>
operator / (matrix_base<NumericT> const & m1, float s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>(m1, NumericT(s1));
}

/** @brief Returns an expression template for scaling the matrix by a double precision floating point number 'alpha'. */
template<typename NumericT>
matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>
operator / (matrix_base<NumericT> const & m1, double s1)
{
  return matrix_expression< const matrix_base<NumericT>, const NumericT, op_div>(m1, NumericT(s1));
}



// operator /=

/** @brief Scales a matrix by a GPU scalar value */
template<typename NumericT, typename S1>
typename viennacl::enable_if< viennacl::is_scalar<S1>::value, matrix_base<NumericT> & >::type
operator /= (matrix_base<NumericT> & m1, S1 const & gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, gpu_val, 1, true, false);
  return m1;
}

/** @brief Scales a matrix by a char (8-bit integer) value */
template<typename NumericT>
matrix_base<NumericT> &
operator /= (matrix_base<NumericT> & m1, char gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, NumericT(gpu_val), 1, true, false);
  return m1;
}

/** @brief Scales a matrix by a short integer value */
template<typename NumericT>
matrix_base<NumericT> &
operator /= (matrix_base<NumericT> & m1, short gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, gpu_val, 1, true, false);
  return m1;
}

/** @brief Scales a matrix by an integer value */
template<typename NumericT>
matrix_base<NumericT> &
operator /= (matrix_base<NumericT> & m1, int gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, gpu_val, 1, true, false);
  return m1;
}

/** @brief Scales a matrix by a long integer value */
template<typename NumericT>
matrix_base<NumericT> &
operator /= (matrix_base<NumericT> & m1, long gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, gpu_val, 1, true, false);
  return m1;
}

/** @brief Scales a matrix by a single precision floating point value */
template<typename NumericT>
matrix_base<NumericT> &
operator /= (matrix_base<NumericT> & m1, float gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, gpu_val, 1, true, false);
  return m1;
}

/** @brief Scales a matrix by a double precision floating point value */
template<typename NumericT>
matrix_base<NumericT> &
operator /= (matrix_base<NumericT> & m1, double gpu_val)
{
  viennacl::linalg::am(m1,
                       m1, gpu_val, 1, true, false);
  return m1;
}





// outer_prod(v1, v2) * val;
template<typename NumericT, typename S1>
typename viennacl::enable_if< viennacl::is_scalar<S1>::value,
viennacl::matrix_expression< const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod>,
const S1,
op_mult>
>::type
operator*(const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod> & proxy,
          const S1 & val)
{
  return viennacl::matrix_expression< const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod>,
      const S1,
      op_mult>(proxy, val);
}

template<typename NumericT, typename S1>
typename viennacl::enable_if< viennacl::is_cpu_scalar<S1>::value,
viennacl::matrix_expression< const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod>,
const NumericT,
op_mult>
>::type
operator*(const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod> & proxy,
          const S1 & val)
{
  return viennacl::matrix_expression< const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod>,
      const NumericT,
      op_mult>(proxy, NumericT(val));
}

// val * outer_prod(v1, v2);
template<typename NumericT, typename S1>
typename viennacl::enable_if< viennacl::is_scalar<S1>::value,
viennacl::matrix_expression< const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod>,
const S1,
op_mult>
>::type
operator*(const S1 & val,
          const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod> & proxy)
{
  return viennacl::matrix_expression< const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod>,
      const S1,
      op_mult>(proxy, val);
}

template<typename NumericT, typename S1>
typename viennacl::enable_if< viennacl::is_cpu_scalar<S1>::value,
viennacl::matrix_expression< const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod>,
const NumericT,
op_mult>
>::type
operator*(const S1 & val,
          const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod> & proxy)
{
  return viennacl::matrix_expression< const viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod>,
      const NumericT,
      op_mult>(proxy, NumericT(val));
}



//
// Specify available operations:
//

/** \cond */

namespace linalg
{
namespace detail
{

  // x = y
  template<typename T>
  struct op_executor<matrix_base<T>, op_assign, matrix_base<T> >
  {
    static void apply(matrix_base<T> & lhs, matrix_base<T> const & rhs)
    {
      viennacl::linalg::am(lhs, rhs, T(1), 1, false, false);
    }
  };

  // x = trans(y)
  template<typename T>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans> const & rhs)
    {
      matrix_base<T> temp(rhs);
      viennacl::linalg::am(lhs, temp, T(1), 1, false, false);
    }
  };

  // x = trans(expr)
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>, const matrix_expression<const LhsT, const RhsT, OpT>, op_trans> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                                              const matrix_expression<const LhsT, const RhsT, OpT>,
                                                              op_trans> const & rhs)
    {
      matrix_base<T> temp1(rhs.rhs());
      matrix_base<T> temp2(viennacl::trans(temp1));
      viennacl::linalg::am(lhs, temp2, T(1), 1, false, false);
    }
  };


  // x += y
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_base<T> >
  {
    static void apply(matrix_base<T> & lhs, matrix_base<T> const & rhs)
    {
      viennacl::linalg::ambm(lhs, lhs, T(1), 1, false, false, rhs, T(1), 1, false, false);
    }
  };

  // x += trans(y)
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans> const & rhs)
    {
      matrix_base<T> temp(rhs);
      viennacl::linalg::ambm(lhs, lhs, T(1), 1, false, false, temp, T(1), 1, false, false);
    }
  };

  // x += trans(expr)
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>, const matrix_expression<const LhsT, const RhsT, OpT>, op_trans> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                                              const matrix_expression<const LhsT, const RhsT, OpT>,
                                                              op_trans> const & rhs)
    {
      matrix_base<T> temp1(rhs.rhs());
      matrix_base<T> temp2(viennacl::trans(temp1));
      viennacl::linalg::ambm(lhs, lhs, T(1), 1, false, false, temp2, T(1), 1, false, false);
    }
  };

  // x -= y
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_base<T> >
  {
    static void apply(matrix_base<T> & lhs, matrix_base<T> const & rhs)
    {
      viennacl::linalg::ambm(lhs, lhs, T(1), 1, false, false, rhs, T(1), 1, false, true);
    }
  };

  // x -= trans(y)
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans> const & rhs)
    {
      matrix_base<T> temp(rhs);
      viennacl::linalg::ambm(lhs, lhs, T(1), 1, false, false, temp, T(1), 1, false, true);
    }
  };

  // x -= trans(expr)
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>, const matrix_expression<const LhsT, const RhsT, OpT>, op_trans> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                                              const matrix_expression<const LhsT, const RhsT, OpT>,
                                                              op_trans> const & rhs)
    {
      matrix_base<T> temp1(rhs.rhs());
      matrix_base<T> temp2(viennacl::trans(temp1));
      viennacl::linalg::ambm(lhs, lhs, T(1), 1, false, false, temp2, T(1), 1, false, true);
    }
  };

  ///////////// x  OP  y * alpha ////////////////////////


  // x = alpha * y
  template<typename T, typename ScalarType>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const matrix_base<T>, const ScalarType, op_mult> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const ScalarType, op_mult> const & proxy)
    {
      viennacl::linalg::am(lhs, proxy.lhs(), proxy.rhs(), 1, false, false);
    }
  };

  // x += alpha * y
  template<typename T, typename ScalarType>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const matrix_base<T>, const ScalarType, op_mult> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const ScalarType, op_mult> const & proxy)
    {
      viennacl::linalg::ambm(lhs, lhs, T(1), 1, false, false, proxy.lhs(), proxy.rhs(), 1, false, false);
    }
  };

  // x -= alpha * y
  template<typename T, typename ScalarType>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const matrix_base<T>, const ScalarType, op_mult> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const ScalarType, op_mult> const & proxy)
    {
      viennacl::linalg::ambm(lhs, lhs, T(1), 1, false, false, proxy.lhs(), proxy.rhs(), 1, false, true);
    }
  };


  ///////////// x  OP  vec_expr * alpha ////////////////////////

  // x = alpha * vec_expr
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> const & proxy)
    {
      if (lhs.row_major())
      {
        matrix<T> temp(proxy.lhs());
        lhs = temp * proxy.rhs();
      }
      else
      {
        matrix<T, column_major> temp(proxy.lhs());
        lhs = temp * proxy.rhs();
      }
    }
  };

  // x += alpha * vec_expr
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> const & proxy)
    {
      if (lhs.row_major())
      {
        matrix<T> temp(proxy.lhs());
        lhs += temp * proxy.rhs();
      }
      else
      {
        matrix<T, column_major> temp(proxy.lhs());
        lhs += temp * proxy.rhs();
      }
    }
  };

  // x -= alpha * vec_expr
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> const & proxy)
    {
      if (lhs.row_major())
      {
        matrix<T> temp(proxy.lhs());
        lhs -= temp * proxy.rhs();
      }
      else
      {
        matrix<T, column_major> temp(proxy.lhs());
        lhs -= temp * proxy.rhs();
      }
    }
  };


  ///////////// x  OP  y / alpha ////////////////////////

  // x = y / alpha
  template<typename T, typename ScalarType>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const matrix_base<T>, const ScalarType, op_div> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const ScalarType, op_div> const & proxy)
    {
      viennacl::linalg::am(lhs, proxy.lhs(), proxy.rhs(), 1, true, false);
    }
  };

  // x += y / alpha
  template<typename T, typename ScalarType>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const matrix_base<T>, const ScalarType, op_div> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const ScalarType, op_div> const & proxy)
    {
      viennacl::linalg::ambm(lhs, lhs, T(1), 1, false, false, proxy.lhs(), proxy.rhs(), 1, true, false);
    }
  };

  // x -= y / alpha
  template<typename T, typename ScalarType>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const matrix_base<T>, const ScalarType, op_div> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const ScalarType, op_div> const & proxy)
    {
      viennacl::linalg::ambm(lhs, lhs, T(1), 1, false, false, proxy.lhs(), proxy.rhs(), 1, true, true);
    }
  };


  ///////////// x  OP  vec_expr / alpha ////////////////////////

  // x = vec_expr / alpha
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_div> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_div> const & proxy)
    {
      if (lhs.row_major())
      {
        matrix<T> temp(proxy.lhs());
        lhs = temp / proxy.rhs();
      }
      else
      {
        matrix<T, column_major> temp(proxy.lhs());
        lhs = temp / proxy.rhs();
      }
    }
  };

  // x += vec_expr / alpha
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_div> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_div> const & proxy)
    {
      if (lhs.row_major())
      {
        matrix<T> temp(proxy.lhs());
        lhs += temp / proxy.rhs();
      }
      else
      {
        matrix<T, column_major> temp(proxy.lhs());
        lhs += temp / proxy.rhs();
      }
    }
  };

  // x -= vec_expr / alpha
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_div> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS, const RHS, OP>, const ScalarType, op_div> const & proxy)
    {
      if (lhs.row_major())
      {
        matrix<T, row_major> temp(proxy.lhs());
        lhs -= temp / proxy.rhs();
      }
      else
      {
        matrix<T, column_major> temp(proxy.lhs());
        lhs -= temp / proxy.rhs();
      }
    }
  };



  // generic x = vec_expr1 + vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const LHS, const RHS, op_add> >
  {
    // generic x = vec_expr1 + vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(matrix_base<T> & lhs, matrix_expression<const LHS1, const RHS1, op_add> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        matrix_base<T> temp(proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_add, RHS>::apply(temp, proxy.rhs());
        lhs = temp;
      }
      else
      {
        op_executor<matrix_base<T>, op_assign, LHS>::apply(lhs, proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_add, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x = y + z
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_add> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs(), T(1), 1, false, false);
    }

    // x = alpha * y + z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      const matrix_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs(), T(1), 1, false, false);
    }

    // x = y / alpha + z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      const matrix_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs(), T(1), 1, false, false);
    }

    // x = y + beta * z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x = y + z / beta
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x = alpha * y + beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x = alpha * y + z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x = y / alpha + beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x = y / alpha + z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }
  };

  // dense = sparse * dense
  template<typename T, typename LHS, typename RHS>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const LHS, const RHS, op_prod> >
  {
    template< typename SparseMatrixType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const SparseMatrixType,
                      const viennacl::matrix_base<T>,
                      viennacl::op_prod> const & proxy)
    {
      // check for x = A * x
      if (op_aliasing(lhs, proxy.rhs()))
      {
        matrix_base<T> temp(proxy);
        lhs = temp;
      }
      else
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), lhs);
    }

    // dense = sparse * trans(dense)
    template< typename SparseMatrixType >
    static void apply(matrix_base<T> & lhs, matrix_expression<const SparseMatrixType,
                      const viennacl::matrix_expression< const viennacl::matrix_base<T>,
                      const viennacl::matrix_base<T>,
                      viennacl::op_trans >,
                      viennacl::op_prod> const & proxy)
    {
      // check for x = A * x
      if (op_aliasing(lhs, proxy.rhs()))
      {
        matrix_base<T> temp(proxy);
        lhs = temp;
      }
      else
        viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), lhs);
    }

  };

  // generic x += vec_expr1 + vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const LHS, const RHS, op_add> >
  {
    // generic x += vec_expr1 + vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(matrix_base<T> & lhs, matrix_expression<const LHS1, const RHS1, op_add> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        matrix_base<T> temp(proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_add, RHS>::apply(temp, proxy.rhs());
        lhs += temp;
      }
      else
      {
        op_executor<matrix_base<T>, op_inplace_add, LHS>::apply(lhs, proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_add, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x += y + z
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x += alpha * y + z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      const matrix_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x += y / alpha + z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      const matrix_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x += y + beta * z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x += y + z / beta
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x += alpha * y + beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x += alpha * y + z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x += y / alpha + beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x += y / alpha + z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }
  };



  // generic x -= vec_expr1 + vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const LHS, const RHS, op_add> >
  {
    // generic x -= vec_expr1 + vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(matrix_base<T> & lhs, matrix_expression<const LHS1, const RHS1, op_add> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        matrix_base<T> temp(proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_add, RHS>::apply(temp, proxy.rhs());
        lhs -= temp;
      }
      else
      {
        op_executor<matrix_base<T>, op_inplace_sub, LHS>::apply(lhs, proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_sub, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x -= y + z
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x -= alpha * y + z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      const matrix_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x -= y / alpha + z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      const matrix_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x -= y + beta * z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x -= y + z / beta
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x -= alpha * y + beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x -= alpha * y + z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x -= y / alpha + beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x -= y / alpha + z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }
  };



  ///////////////////////



  // generic x = vec_expr1 - vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const LHS, const RHS, op_sub> >
  {
    // generic x = vec_expr1 - vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(matrix_base<T> & lhs, matrix_expression<const LHS1, const RHS1, op_sub> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        matrix_base<T> temp(proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_sub, RHS>::apply(temp, proxy.rhs());
        lhs = temp;
      }
      else
      {
        op_executor<matrix_base<T>, op_assign, LHS>::apply(lhs, proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_sub, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x = y - z
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_sub> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs(), T(1), 1, false, true);
    }

    // x = alpha * y - z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      const matrix_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs(), T(1), 1, false, true);
    }

    // x = y / alpha - z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      const matrix_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs(), T(1), 1, false, true);
    }

    // x = y - beta * z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x = y - z / beta
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x = alpha * y - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x = alpha * y - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x = y / alpha - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x = y / alpha - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }
  };


  // generic x += vec_expr1 - vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const LHS, const RHS, op_sub> >
  {
    // generic x += vec_expr1 - vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(matrix_base<T> & lhs, matrix_expression<const LHS1, const RHS1, op_sub> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        matrix_base<T> temp(proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_sub, RHS>::apply(temp, proxy.rhs());
        lhs += temp;
      }
      else
      {
        op_executor<matrix_base<T>, op_inplace_add, LHS>::apply(lhs, proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_sub, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x += y - z
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x += alpha * y - z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      const matrix_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x += y / alpha - z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      const matrix_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x += y - beta * z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x += y - z / beta
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x += alpha * y - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x += alpha * y - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x += y / alpha - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x += y / alpha - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }
  };



  // generic x -= vec_expr1 - vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const LHS, const RHS, op_sub> >
  {
    // generic x -= vec_expr1 - vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(matrix_base<T> & lhs, matrix_expression<const LHS1, const RHS1, op_sub> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        matrix_base<T> temp(proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_sub, RHS>::apply(temp, proxy.rhs());
        lhs -= temp;
      }
      else
      {
        op_executor<matrix_base<T>, op_inplace_sub, LHS>::apply(lhs, proxy.lhs());
        op_executor<matrix_base<T>, op_inplace_add, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x -= y - z
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x -= alpha * y - z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      const matrix_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x -= y / alpha - z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      const matrix_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x -= y - beta * z
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x -= y - z / beta
    template<typename ScalarType>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const ScalarType, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x -= alpha * y - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x -= alpha * y - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_mult>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x -= y / alpha - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x -= y / alpha - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const ScalarType1, op_div>,
                      const matrix_expression<const matrix_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::ambm_m(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }
  };


  //////////////////// diag(), row(), column() operations ////////////////////////////////////////

  template<typename T, typename LHS>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const LHS, const int, op_vector_diag> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const vector_base<T>, const int, op_vector_diag> const & proxy)
    {
      viennacl::linalg::matrix_diag_from_vector(proxy.lhs(), proxy.rhs(), lhs);
    }
  };


  template<typename T, typename LHS>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const LHS, const int, op_matrix_diag> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_base<T>, const int, op_matrix_diag> const & proxy)
    {
      viennacl::linalg::matrix_diag_to_vector(proxy.lhs(), proxy.rhs(), lhs);
    }
  };

  template<typename T, typename LHS>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const LHS, const unsigned int, op_row> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_base<T>, const unsigned int, op_row> const & proxy)
    {
      viennacl::linalg::matrix_row(proxy.lhs(), proxy.rhs(), lhs);
    }
  };


  template<typename T, typename LHS>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const LHS, const unsigned int, op_column> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_base<T>, const unsigned int, op_column> const & proxy)
    {
      viennacl::linalg::matrix_column(proxy.lhs(), proxy.rhs(), lhs);
    }
  };

  //////////////////// row_sum(), column_sum() operations ////////////////////////////////////////

  template<typename T>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const matrix_base<T>, const matrix_base<T>, op_row_sum> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_base<T>, const matrix_base<T>, op_row_sum> const & proxy)
    {
      viennacl::linalg::row_sum_impl(proxy.lhs(), lhs);
    }
  };

  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const matrix_expression<LHS, RHS, OP>, const matrix_expression<LHS, RHS, OP>, op_row_sum> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_expression<LHS, RHS, OP>, const matrix_expression<LHS, RHS, OP>, op_row_sum> const & proxy)
    {
      matrix_base<T> tmp(proxy.lhs());
      viennacl::linalg::row_sum_impl(tmp, lhs);
    }
  };

  template<typename T>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const matrix_base<T>, const matrix_base<T>, op_col_sum> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_base<T>, const matrix_base<T>, op_col_sum> const & proxy)
    {
      viennacl::linalg::column_sum_impl(proxy.lhs(), lhs);
    }
  };


  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const matrix_expression<LHS, RHS, OP>, const matrix_expression<LHS, RHS, OP>, op_col_sum> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_expression<LHS, RHS, OP>, const matrix_expression<LHS, RHS, OP>, op_col_sum> const & proxy)
    {
      matrix_base<T> tmp(proxy.lhs());
      viennacl::linalg::column_sum_impl(tmp, lhs);
    }
  };

  //////////////////// Element-wise operations ////////////////////////////////////////

  // generic x = mat_expr1 .* mat_expr2:
  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const LHS, const RHS, op_element_binary<OP> > >
  {
    // x = y .* z
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> > const & proxy)
    {
      viennacl::linalg::element_op(lhs, proxy);
    }

    // x = y .* mat_expr
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_expression<const LHS2, const RHS2, OP2>, op_element_binary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy.rhs());
      viennacl::linalg::element_op(lhs, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> >(proxy.lhs(), temp));
    }

    // x = mat_expr .* z
    template<typename LHS1, typename RHS1, typename OP1>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS1, const RHS1, OP1>, const matrix_base<T>, op_element_binary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy.lhs());
      viennacl::linalg::element_op(lhs, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> >(temp, proxy.rhs()));
    }

    // x = mat_expr .* mat_expr
    template<typename LHS1, typename RHS1, typename OP1,
             typename LHS2, typename RHS2, typename OP2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS1, const RHS1, OP1>,
                      const matrix_expression<const LHS2, const RHS2, OP2>,
                      op_element_binary<OP> > const & proxy)
    {
      matrix_base<T> temp1(proxy.lhs());
      matrix_base<T> temp2(proxy.rhs());
      viennacl::linalg::element_op(lhs, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> >(temp1, temp2));
    }
  };

  // generic x += mat_expr .* mat_expr:
  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const LHS, const RHS, op_element_binary<OP> > >
  {
    // x += y .* z
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy);
      lhs += temp;
    }

    // x += y .* mat_expr
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_expression<const LHS2, const RHS2, OP2>, op_element_binary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy.rhs());
      matrix_base<T> temp2(temp.size1(), temp.size2(), lhs.row_major(), viennacl::traits::context(lhs));
      viennacl::linalg::element_op(temp2, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> >(proxy.lhs(), temp));
      lhs += temp2;
    }

    // x += mat_expr .* z
    template<typename LHS1, typename RHS1, typename OP1>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS1, const RHS1, OP1>, const matrix_base<T>, op_element_binary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy.lhs());
      matrix_base<T> temp2(temp.size1(), temp.size2(), lhs.row_major(), viennacl::traits::context(lhs));
      viennacl::linalg::element_op(temp2, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> >(temp, proxy.rhs()));
      lhs += temp2;
    }

    // x += mat_expr .* mat_expr
    template<typename LHS1, typename RHS1, typename OP1,
             typename LHS2, typename RHS2, typename OP2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS1, const RHS1, OP1>,
                      const matrix_expression<const LHS2, const RHS2, OP2>,
                      op_element_binary<OP> > const & proxy)
    {
      matrix_base<T> temp1(proxy.lhs());
      matrix_base<T> temp2(proxy.rhs());
      matrix_base<T> temp3(temp1.size1(), temp1.size2(), lhs.row_major(), viennacl::traits::context(lhs));
      viennacl::linalg::element_op(temp3, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> >(temp1, temp2));
      lhs += temp3;
    }
  };

  // generic x -= mat_expr1 .* mat_expr2:
  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const LHS, const RHS, op_element_binary<OP> > >
  {

    // x -= y .* z
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy);
      lhs -= temp;
    }

    // x -= y .* mat_expr
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_expression<const LHS2, const RHS2, OP2>, op_element_binary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy.rhs());
      matrix_base<T> temp2(temp.size1(), temp.size2(), lhs.row_major(), viennacl::traits::context(lhs));
      viennacl::linalg::element_op(temp2, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> >(proxy.lhs(), temp));
      lhs -= temp2;
    }

    // x -= mat_expr .* z
    template<typename LHS1, typename RHS1, typename OP1>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS1, const RHS1, OP1>, const matrix_base<T>, op_element_binary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy.lhs());
      matrix_base<T> temp2(temp.size1(), temp.size2(), lhs.row_major(), viennacl::traits::context(lhs));
      viennacl::linalg::element_op(temp2, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> >(temp, proxy.rhs()));
      lhs -= temp2;
    }

    // x -= mat_expr .* mat_expr
    template<typename LHS1, typename RHS1, typename OP1,
             typename LHS2, typename RHS2, typename OP2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS1, const RHS1, OP1>,
                      const matrix_expression<const LHS2, const RHS2, OP2>,
                      op_element_binary<OP> > const & proxy)
    {
      matrix_base<T> temp1(proxy.lhs());
      matrix_base<T> temp2(proxy.rhs());
      matrix_base<T> temp3(temp1.size1(), temp1.size2(), lhs.row_major(), viennacl::traits::context(lhs));
      viennacl::linalg::element_op(temp3, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<OP> >(temp1, temp2));
      lhs -= temp3;
    }
  };

  //////////////// unary expressions

  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const LHS, const RHS, op_element_unary<OP> > >
  {
    // x = OP(y)
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<OP> > const & proxy)
    {
      viennacl::linalg::element_op(lhs, proxy);
    }

    // x = OP(vec_expr)
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS2, const RHS2, OP2>,
                      const matrix_expression<const LHS2, const RHS2, OP2>,
                      op_element_unary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy.rhs());
      viennacl::linalg::element_op(lhs, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<OP> >(temp, temp));
    }
  };

  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const LHS, const RHS, op_element_unary<OP> > >
  {
    // x += OP(y)
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy);
      lhs += temp;
    }

    // x += OP(vec_expr)
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS2, const RHS2, OP2>,
                      const matrix_expression<const LHS2, const RHS2, OP2>,
                      op_element_unary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy.rhs());
      viennacl::linalg::element_op(temp, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<OP> >(temp, temp)); // inplace operation is safe here
      lhs += temp;
    }
  };

  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const LHS, const RHS, op_element_unary<OP> > >
  {
    // x -= OP(y)
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy);
      lhs -= temp;
    }

    // x -= OP(vec_expr)
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const LHS2, const RHS2, OP2>,
                      const matrix_expression<const LHS2, const RHS2, OP2>,
                      op_element_unary<OP> > const & proxy)
    {
      matrix_base<T> temp(proxy.rhs());
      viennacl::linalg::element_op(temp, viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<OP> >(temp, temp)); // inplace operation is safe here
      lhs -= temp;
    }
  };



  //////////////// Matrix - Matrix products ////////////////

  // C = A * B
  template<typename T>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs()) || op_aliasing(lhs, rhs.rhs()))
      {
        matrix_base<T> temp(rhs);
        lhs = temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(1.0), T(0));
    }
  };

  // C = A * B^T
  template<typename T>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const matrix_base<T>,
      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs()) || op_aliasing(lhs, rhs.rhs().lhs()))
      {
        matrix_base<T> temp(rhs);
        lhs = temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(1.0), T(0));
    }
  };

  // C = A * EXPR   for some matrix expression EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<matrix_base<T>,
                     op_assign,
                     matrix_expression<const matrix_base<T>,
                                       const matrix_expression<const LhsT, const RhsT, OpT>,
                                       op_mat_mat_prod>
                    >
  {
    static void apply(matrix_base<T> & lhs,
                      matrix_expression<const matrix_base<T>,
                                        const matrix_expression<const LhsT, const RhsT, OpT>,
                                        op_mat_mat_prod> const & rhs)
    {
      matrix_base<T> temp(rhs.rhs());
      viennacl::linalg::prod_impl(rhs.lhs(), temp, lhs, T(1.0), T(0));
    }
  };



  // C = A^T * B
  template<typename T>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      const matrix_base<T>,
      op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      const matrix_base<T>,
                      op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs().lhs()) || op_aliasing(lhs, rhs.rhs()))
      {
        matrix_base<T> temp(rhs);
        lhs = temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(1.0), T(0));
    }
  };

  // C = EXPR * B   for some matrix expression EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<matrix_base<T>,
                     op_assign,
                     matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                       const matrix_base<T>,
                                       op_mat_mat_prod>
                    >
  {
    static void apply(matrix_base<T> & lhs,
                      matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                        const matrix_base<T>,
                                        op_mat_mat_prod> const & rhs)
    {
      matrix_base<T> temp(rhs.lhs());
      viennacl::linalg::prod_impl(temp, rhs.rhs(), lhs, T(1.0), T(0));
    }
  };


  // C = A^T * B^T
  template<typename T>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs().lhs()) || op_aliasing(lhs, rhs.rhs().lhs()))
      {
        matrix_base<T> temp(rhs);
        lhs = temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(1.0), T(0));
    }
  };

  // C = EXPR1 * EXPR2   for some matrix expressions EXPR1 and EXPR2
  template<typename T,
           typename LhsT1, typename RhsT1, typename OpT1,
           typename LhsT2, typename RhsT2, typename OpT2>
  struct op_executor<matrix_base<T>,
                     op_assign,
                     matrix_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                       const matrix_expression<const LhsT2, const RhsT2, OpT2>,
                                       op_mat_mat_prod>
                    >
  {
    static void apply(matrix_base<T> & lhs,
                      matrix_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                        const matrix_expression<const LhsT2, const RhsT2, OpT2>,
                                        op_mat_mat_prod> const & rhs)
    {
      matrix_base<T> temp1(rhs.lhs());
      matrix_base<T> temp2(rhs.rhs());
      viennacl::linalg::prod_impl(temp1, temp2, lhs, T(1.0), T(0));
    }
  };




  // C += A * B
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs()) || op_aliasing(lhs, rhs.rhs()))
      {
        matrix_base<T> temp(rhs);
        lhs += temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(1.0), T(1.0));
    }
  };

  // C += A * B^T
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const matrix_base<T>,
      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs()) || op_aliasing(lhs, rhs.rhs().lhs()))
      {
        matrix_base<T> temp(rhs);
        lhs += temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(1.0), T(1.0));
    }
  };

  // C += A * EXPR   for some matrix expression EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<matrix_base<T>,
                     op_inplace_add,
                     matrix_expression<const matrix_base<T>,
                                       const matrix_expression<const LhsT, const RhsT, OpT>,
                                       op_mat_mat_prod>
                    >
  {
    static void apply(matrix_base<T> & lhs,
                      matrix_expression<const matrix_base<T>,
                                        const matrix_expression<const LhsT, const RhsT, OpT>,
                                        op_mat_mat_prod> const & rhs)
    {
      matrix_base<T> temp(rhs.rhs());
      viennacl::linalg::prod_impl(rhs.lhs(), temp, lhs, T(1.0), T(1.0));
    }
  };


  // C += A^T * B
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      const matrix_base<T>,
      op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      const matrix_base<T>,
                      op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs().lhs()) || op_aliasing(lhs, rhs.rhs()))
      {
        matrix_base<T> temp(rhs);
        lhs += temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(1.0), T(1.0));
    }
  };

  // C += EXPR * B   for some matrix expression EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<matrix_base<T>,
                     op_inplace_add,
                     matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                       const matrix_base<T>,
                                       op_mat_mat_prod>
                    >
  {
    static void apply(matrix_base<T> & lhs,
                      matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                        const matrix_base<T>,
                                        op_mat_mat_prod> const & rhs)
    {
      matrix_base<T> temp(rhs.lhs());
      viennacl::linalg::prod_impl(temp, rhs.rhs(), lhs, T(1.0), T(1.0));
    }
  };


  // C += A^T * B^T
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs().lhs()) || op_aliasing(lhs, rhs.rhs().lhs()))
      {
        matrix_base<T> temp(rhs);
        lhs += temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(1.0), T(1.0));
    }
  };


  // C += EXPR1 * EXPR2   for some matrix expressions EXPR1 and EXPR2
  template<typename T,
           typename LhsT1, typename RhsT1, typename OpT1,
           typename LhsT2, typename RhsT2, typename OpT2>
  struct op_executor<matrix_base<T>,
                     op_inplace_add,
                     matrix_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                       const matrix_expression<const LhsT2, const RhsT2, OpT2>,
                                       op_mat_mat_prod>
                    >
  {
    static void apply(matrix_base<T> & lhs,
                      matrix_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                        const matrix_expression<const LhsT2, const RhsT2, OpT2>,
                                        op_mat_mat_prod> const & rhs)
    {
      matrix_base<T> temp1(rhs.lhs());
      matrix_base<T> temp2(rhs.rhs());
      viennacl::linalg::prod_impl(temp1, temp2, lhs, T(1.0), T(1.0));
    }
  };



  // C -= A * B
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>, const matrix_base<T>, op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs()) || op_aliasing(lhs, rhs.rhs()))
      {
        matrix_base<T> temp(rhs);
        lhs -= temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(-1.0), T(1.0));
    }
  };

  // C -= A * B^T
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const matrix_base<T>,
      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_base<T>,
                      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs()) || op_aliasing(lhs, rhs.rhs().lhs()))
      {
        matrix_base<T> temp(rhs);
        lhs -= temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(-1.0), T(1.0));
    }
  };

  // C -= A * EXPR   for some matrix expression EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<matrix_base<T>,
                     op_inplace_sub,
                     matrix_expression<const matrix_base<T>,
                                       const matrix_expression<const LhsT, const RhsT, OpT>,
                                       op_mat_mat_prod>
                    >
  {
    static void apply(matrix_base<T> & lhs,
                      matrix_expression<const matrix_base<T>,
                                        const matrix_expression<const LhsT, const RhsT, OpT>,
                                        op_mat_mat_prod> const & rhs)
    {
      matrix_base<T> temp(rhs.rhs());
      viennacl::linalg::prod_impl(rhs.lhs(), temp, lhs, T(-1.0), T(1.0));
    }
  };


  // C -= A^T * B
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      const matrix_base<T>,
      op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      const matrix_base<T>,
                      op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs().lhs()) || op_aliasing(lhs, rhs.rhs()))
      {
        matrix_base<T> temp(rhs);
        lhs -= temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(-1.0), T(1.0));
    }
  };

  // C += EXPR * B   for some matrix expression EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<matrix_base<T>,
                     op_inplace_sub,
                     matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                       const matrix_base<T>,
                                       op_mat_mat_prod>
                    >
  {
    static void apply(matrix_base<T> & lhs,
                      matrix_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                        const matrix_base<T>,
                                        op_mat_mat_prod> const & rhs)
    {
      matrix_base<T> temp(rhs.lhs());
      viennacl::linalg::prod_impl(temp, rhs.rhs(), lhs, T(-1.0), T(1.0));
    }
  };


  // C -= A^T * B^T
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      op_mat_mat_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      op_mat_mat_prod> const & rhs)
    {
      if (op_aliasing(lhs, rhs.lhs().lhs()) || op_aliasing(lhs, rhs.rhs().lhs()))
      {
        matrix_base<T> temp(rhs);
        lhs -= temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs, T(-1.0), T(1.0));
    }
  };

  // C -= EXPR1 * EXPR2   for some matrix expressions EXPR1 and EXPR2
  template<typename T,
           typename LhsT1, typename RhsT1, typename OpT1,
           typename LhsT2, typename RhsT2, typename OpT2>
  struct op_executor<matrix_base<T>,
                     op_inplace_sub,
                     matrix_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                       const matrix_expression<const LhsT2, const RhsT2, OpT2>,
                                       op_mat_mat_prod>
                    >
  {
    static void apply(matrix_base<T> & lhs,
                      matrix_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                        const matrix_expression<const LhsT2, const RhsT2, OpT2>,
                                        op_mat_mat_prod> const & rhs)
    {
      matrix_base<T> temp1(rhs.lhs());
      matrix_base<T> temp2(rhs.rhs());
      viennacl::linalg::prod_impl(temp1, temp2, lhs, T(-1.0), T(1.0));
    }
  };

  ////////////////// Matrix-Vector Products ///////////////

  // y = A * x
  template<typename T>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const matrix_base<T>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_base<T>, const vector_base<T>, op_prod> const & rhs)
    {
      // check for x = A * x
      if (op_aliasing(lhs, rhs.rhs()))
      {
        vector_base<T> temp(rhs);
        lhs = temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs);
    }
  };

  // y = A^T * x
  template<typename T>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      const vector_base<T>,
      op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      const vector_base<T>,
                      op_prod> const & rhs)
    {
      // check for x = A^T * x
      if (op_aliasing(lhs, rhs.rhs()))
      {
        vector_base<T> temp(rhs);
        lhs = temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), lhs);
    }
  };

  // y = MAT_EXPR * x   for a matrix expression MAT_EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<vector_base<T>,
                     op_assign,
                     vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                       const vector_base<T>,
                                       op_prod>
                    >
  {
    static void apply(vector_base<T> & lhs,
                      vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                        const vector_base<T>,
                                        op_prod> const & rhs)
    {
      matrix_base<T> temp(rhs.lhs());
      viennacl::linalg::prod_impl(temp, rhs.rhs(), lhs);
    }
  };

  // y = A * VEC_EXPR   for a vector expression VEC_EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<vector_base<T>,
                     op_assign,
                     vector_expression<const matrix_base<T>,
                                       const vector_expression<const LhsT, const RhsT, OpT>,
                                       op_prod>
                    >
  {
    static void apply(vector_base<T> & lhs,
                      vector_expression<const matrix_base<T>,
                                        const vector_expression<const LhsT, const RhsT, OpT>,
                                        op_prod> const & rhs)
    {
      vector_base<T> x(rhs.rhs());
      viennacl::linalg::prod_impl(rhs.lhs(), x, lhs);
    }
  };

  // y = MAT_EXPR * VEC_EXPR   for a matrix expression MAT_EXPR and a vector expression VEC_EXPR
  template<typename T,
           typename LhsT1, typename RhsT1, typename OpT1,
           typename LhsT2, typename RhsT2, typename OpT2>
  struct op_executor<vector_base<T>,
                     op_assign,
                     vector_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                       const vector_expression<const LhsT2, const RhsT2, OpT2>,
                                       op_prod>
                    >
  {
    static void apply(vector_base<T> & lhs,
                      vector_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                        const vector_expression<const LhsT2, const RhsT2, OpT2>,
                                        op_prod> const & rhs)
    {
      matrix_base<T> A(rhs.lhs());
      vector_base<T> x(rhs.rhs());
      viennacl::linalg::prod_impl(A, x, lhs);
    }
  };



  // y += A * x
  template<typename T>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const matrix_base<T>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_base<T>, const vector_base<T>, op_prod> const & rhs)
    {
      vector_base<T> temp(rhs);
      lhs += temp;
    }
  };

  // y += A^T * x
  template<typename T>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      const vector_base<T>,
      op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      const vector_base<T>,
                      op_prod> const & rhs)
    {
      vector_base<T> temp(rhs);
      lhs += temp;
    }
  };

  // y += MAT_EXPR * x   for a matrix expression MAT_EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<vector_base<T>,
                     op_inplace_add,
                     vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                       const vector_base<T>,
                                       op_prod>
                    >
  {
    static void apply(vector_base<T> & lhs,
                      vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                        const vector_base<T>,
                                        op_prod> const & rhs)
    {
      matrix_base<T> A(rhs.lhs());
      vector_base<T> y(lhs);
      viennacl::linalg::prod_impl(A, rhs.rhs(), y);
      lhs += y;
    }
  };

  // y += A * VEC_EXPR   for a vector expression VEC_EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<vector_base<T>,
                     op_inplace_add,
                     vector_expression<const matrix_base<T>,
                                       const vector_expression<const LhsT, const RhsT, OpT>,
                                       op_prod>
                    >
  {
    static void apply(vector_base<T> & lhs,
                      vector_expression<const matrix_base<T>,
                                        const vector_expression<const LhsT, const RhsT, OpT>,
                                        op_prod> const & rhs)
    {
      vector_base<T> x(rhs.rhs());
      vector_base<T> y(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), x, y);
      lhs += y;
    }
  };

  // y += MAT_EXPR * VEC_EXPR   for a matrix expression MAT_EXPR and a vector expression VEC_EXPR
  template<typename T,
           typename LhsT1, typename RhsT1, typename OpT1,
           typename LhsT2, typename RhsT2, typename OpT2>
  struct op_executor<vector_base<T>,
                     op_inplace_add,
                     vector_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                       const vector_expression<const LhsT2, const RhsT2, OpT2>,
                                       op_prod>
                    >
  {
    static void apply(vector_base<T> & lhs,
                      vector_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                        const vector_expression<const LhsT2, const RhsT2, OpT2>,
                                        op_prod> const & rhs)
    {
      matrix_base<T> A(rhs.lhs());
      vector_base<T> x(rhs.rhs());
      vector_base<T> y(lhs);
      viennacl::linalg::prod_impl(A, x, y);
      lhs += y;
    }
  };



  // y -= A * x
  template<typename T>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const matrix_base<T>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_base<T>, const vector_base<T>, op_prod> const & rhs)
    {
      vector_base<T> temp(rhs);
      lhs -= temp;
    }
  };

  // y -= A^T * x
  template<typename T>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
      const vector_base<T>,
      op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const matrix_expression<const matrix_base<T>, const matrix_base<T>, op_trans>,
                      const vector_base<T>,
                      op_prod> const & rhs)
    {
      vector_base<T> temp(rhs);
      lhs -= temp;
    }
  };

  // y -= MAT_EXPR * x   for a matrix expression MAT_EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<vector_base<T>,
                     op_inplace_sub,
                     vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                       const vector_base<T>,
                                       op_prod>
                    >
  {
    static void apply(vector_base<T> & lhs,
                      vector_expression<const matrix_expression<const LhsT, const RhsT, OpT>,
                                        const vector_base<T>,
                                        op_prod> const & rhs)
    {
      matrix_base<T> A(rhs.lhs());
      vector_base<T> y(lhs);
      viennacl::linalg::prod_impl(A, rhs.rhs(), y);
      lhs -= y;
    }
  };

  // y -= A * VEC_EXPR   for a vector expression VEC_EXPR
  template<typename T, typename LhsT, typename RhsT, typename OpT>
  struct op_executor<vector_base<T>,
                     op_inplace_sub,
                     vector_expression<const matrix_base<T>,
                                       const vector_expression<const LhsT, const RhsT, OpT>,
                                       op_prod>
                    >
  {
    static void apply(vector_base<T> & lhs,
                      vector_expression<const matrix_base<T>,
                                        const vector_expression<const LhsT, const RhsT, OpT>,
                                        op_prod> const & rhs)
    {
      vector_base<T> x(rhs.rhs());
      vector_base<T> y(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), x, y);
      lhs -= y;
    }
  };

  // y -= MAT_EXPR * VEC_EXPR   for a matrix expression MAT_EXPR and a vector expression VEC_EXPR
  template<typename T,
           typename LhsT1, typename RhsT1, typename OpT1,
           typename LhsT2, typename RhsT2, typename OpT2>
  struct op_executor<vector_base<T>,
                     op_inplace_sub,
                     vector_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                       const vector_expression<const LhsT2, const RhsT2, OpT2>,
                                       op_prod>
                    >
  {
    static void apply(vector_base<T> & lhs,
                      vector_expression<const matrix_expression<const LhsT1, const RhsT1, OpT1>,
                                        const vector_expression<const LhsT2, const RhsT2, OpT2>,
                                        op_prod> const & rhs)
    {
      matrix_base<T> A(rhs.lhs());
      vector_base<T> x(rhs.rhs());
      vector_base<T> y(lhs);
      viennacl::linalg::prod_impl(A, x, y);
      lhs -= y;
    }
  };



  ////////////////// Rank-1 Updates ///////////////

  // A = v1 * v2^T
  template<typename T>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression<const vector_base<T>, const vector_base<T>, op_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const vector_base<T>, const vector_base<T>, op_prod> const & rhs)
    {
      lhs.clear();
      viennacl::linalg::scaled_rank_1_update(lhs, T(1.0), 1, false, false, rhs.lhs(), rhs.rhs());
    }
  };

  // A = alpha * v1 * v2^T
  template<typename T, typename ScalarType>
  struct op_executor<matrix_base<T>, op_assign, matrix_expression< const matrix_expression<const vector_base<T>, const vector_base<T>, op_prod>,
      const ScalarType,
      op_mult> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression< const matrix_expression<const vector_base<T>, const vector_base<T>, op_prod>,
                      const ScalarType,
                      op_mult> const & rhs)
    {
      lhs.clear();
      viennacl::linalg::scaled_rank_1_update(lhs, rhs.rhs(), 1, false, false, rhs.lhs().lhs(), rhs.lhs().rhs());
    }
  };

  // A += v1 * v2^T
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression<const vector_base<T>, const vector_base<T>, op_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const vector_base<T>, const vector_base<T>, op_prod> const & rhs)
    {
      viennacl::linalg::scaled_rank_1_update(lhs, T(1.0), 1, false, false, rhs.lhs(), rhs.rhs());
    }
  };

  // A += alpha * v1 * v2^T
  template<typename T, typename ScalarType>
  struct op_executor<matrix_base<T>, op_inplace_add, matrix_expression< const matrix_expression<const vector_base<T>, const vector_base<T>, op_prod>,
      const ScalarType,
      op_mult> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression< const matrix_expression<const vector_base<T>, const vector_base<T>, op_prod>,
                      const ScalarType,
                      op_mult> const & rhs)
    {
      viennacl::linalg::scaled_rank_1_update(lhs, rhs.rhs(), 1, false, false, rhs.lhs().lhs(), rhs.lhs().rhs());
    }
  };

  // A -= v1 * v2^T
  template<typename T>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression<const vector_base<T>, const vector_base<T>, op_prod> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression<const vector_base<T>, const vector_base<T>, op_prod> const & rhs)
    {
      viennacl::linalg::scaled_rank_1_update(lhs, T(1.0), 1, false, true, rhs.lhs(), rhs.rhs());
    }
  };

  // A -= alpha * v1 * v2^T
  template<typename T, typename ScalarType>
  struct op_executor<matrix_base<T>, op_inplace_sub, matrix_expression< const matrix_expression<const vector_base<T>, const vector_base<T>, op_prod>,
      const ScalarType,
      op_mult> >
  {
    static void apply(matrix_base<T> & lhs, matrix_expression< const matrix_expression<const vector_base<T>, const vector_base<T>, op_prod>,
                      const ScalarType,
                      op_mult> const & rhs)
    {
      viennacl::linalg::scaled_rank_1_update(lhs, rhs.rhs(), 1, false, true, rhs.lhs().lhs(), rhs.lhs().rhs());
    }
  };


} // namespace detail

} // namespace linalg

/** \endcond */

} //namespace viennacl

#endif
