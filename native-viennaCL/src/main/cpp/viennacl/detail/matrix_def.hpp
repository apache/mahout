#ifndef VIENNACL_DETAIL_MATRIX_DEF_HPP_
#define VIENNACL_DETAIL_MATRIX_DEF_HPP_

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

/** @file viennacl/detail/matrix_def.hpp
    @brief Forward declaration of dense matrix classes
*/

#include "viennacl/forwards.h"
#include "viennacl/tools/entry_proxy.hpp"

namespace viennacl
{
/** @brief Base class for representing matrices where the individual entries are not all stored explicitly, e.g. identity_matrix<>
  *
  * Examples are identity_matrix, scalar_matrix, and zero_matrix.
  */
template<typename NumericT>
class implicit_matrix_base
{
protected:
  typedef vcl_size_t        size_type;
  implicit_matrix_base(size_type size1, size_type size2, NumericT value, bool diag, viennacl::context ctx) : size1_(size1), size2_(size2), value_(value), diag_(diag), off_diag_(0), ctx_(ctx){ }
public:
  typedef NumericT const & const_reference;
  typedef NumericT cpu_value_type;

  size_type size1() const { return size1_; }
  size_type size2() const { return size2_; }
  viennacl::context context() const { return ctx_; }
  NumericT  value() const { return value_; }
  bool diag() const { return diag_; }

  const_reference operator()(size_type i, size_type j) const
  {
    if (diag_) return (i == j) ? value_ : off_diag_;
    return value_;
  }
protected:
  size_type size1_;
  size_type size2_;
  NumericT value_;
  bool diag_;
  NumericT off_diag_;
  viennacl::context ctx_;
};

//
// Initializer types
//
/** @brief Represents a vector consisting of 1 at a given index and zeros otherwise. To be used as an initializer for viennacl::vector, vector_range, or vector_slize only. */
template<typename NumericT>
class identity_matrix : public implicit_matrix_base<NumericT>
{
public:
  typedef vcl_size_t         size_type;
  typedef NumericT const & const_reference;

  identity_matrix(size_type s, viennacl::context ctx = viennacl::context()) : implicit_matrix_base<NumericT>(s, s, 1, true, ctx){}
};


/** @brief Represents a vector consisting of zeros only. To be used as an initializer for viennacl::vector, vector_range, or vector_slize only. */
template<typename NumericT>
class zero_matrix : public implicit_matrix_base<NumericT>
{
public:
  typedef vcl_size_t         size_type;
  typedef NumericT const & const_reference;

  zero_matrix(size_type s1, size_type s2, viennacl::context ctx = viennacl::context()) : implicit_matrix_base<NumericT>(s1, s2, 0, false, ctx){}
};


/** @brief Represents a vector consisting of scalars 's' only, i.e. v[i] = s for all i. To be used as an initializer for viennacl::vector, vector_range, or vector_slize only. */
template<typename NumericT>
class scalar_matrix : public implicit_matrix_base<NumericT>
{
public:
  typedef vcl_size_t         size_type;
  typedef NumericT const & const_reference;

  scalar_matrix(size_type s1, size_type s2, const_reference val, viennacl::context ctx = viennacl::context()) : implicit_matrix_base<NumericT>(s1, s2, val, false, ctx) {}
};

template<class NumericT, typename SizeT, typename DistanceT>
class matrix_base
{
  typedef matrix_base<NumericT, SizeT, DistanceT>          self_type;
public:

  typedef matrix_iterator<row_iteration, self_type >   iterator1;
  typedef matrix_iterator<col_iteration, self_type >   iterator2;
  typedef scalar<NumericT>                                                  value_type;
  typedef NumericT                                                          cpu_value_type;
  typedef SizeT                                                            size_type;
  typedef DistanceT                                                        difference_type;
  typedef viennacl::backend::mem_handle                                       handle_type;

  /** @brief The default constructor. Does not allocate any memory. */
  explicit matrix_base(): size1_(0), size2_(0), start1_(0), start2_(0), stride1_(1), stride2_(1), internal_size1_(0), internal_size2_(0), row_major_fixed_(false), row_major_(true) {}

  /** @brief The layout constructor. Does not allocate any memory. */
  explicit matrix_base(bool is_row_major) : size1_(0), size2_(0), start1_(0), start2_(0), stride1_(1), stride2_(1), internal_size1_(0), internal_size2_(0), row_major_fixed_(true), row_major_(is_row_major) {}

  /** @brief Creates the matrix with the given dimensions
    *
    * @param rows     Number of rows
    * @param columns  Number of columns
    * @param is_row_major  Boolean flag stating whether this matrix is stored row-major
    * @param ctx      Optional context in which the matrix is created (one out of multiple OpenCL contexts, CUDA, host)
    */
  explicit matrix_base(size_type rows, size_type columns, bool is_row_major, viennacl::context ctx = viennacl::context());

  /** @brief Constructor for creating a matrix_range or matrix_stride from some other matrix/matrix_range/matrix_stride */
  explicit matrix_base(viennacl::backend::mem_handle & h,
                       size_type mat_size1, size_type mat_start1, size_type mat_stride1, size_type mat_internal_size1,
                       size_type mat_size2, size_type mat_start2, size_type mat_stride2, size_type mat_internal_size2,
                       bool is_row_major): size1_(mat_size1), size2_(mat_size2),
        start1_(mat_start1), start2_(mat_start2),
        stride1_(mat_stride1), stride2_(mat_stride2),
        internal_size1_(mat_internal_size1), internal_size2_(mat_internal_size2),
        row_major_fixed_(true), row_major_(is_row_major),
        elements_(h) {}


  template<typename LHS, typename RHS, typename OP>
  explicit matrix_base(matrix_expression<const LHS, const RHS, OP> const & proxy);

  // CUDA or host memory:
  explicit matrix_base(NumericT * ptr_to_mem, viennacl::memory_types mem_type,
                       size_type mat_size1, size_type mat_start1, size_type mat_stride1, size_type mat_internal_size1,
                       size_type mat_size2, size_type mat_start2, size_type mat_stride2, size_type mat_internal_size2,
                       bool is_row_major);

#ifdef VIENNACL_WITH_OPENCL
  explicit matrix_base(cl_mem mem, size_type rows, size_type columns, bool is_row_major, viennacl::context ctx = viennacl::context());
  explicit matrix_base(cl_mem mem, viennacl::context ctx,
                       size_type mat_size1, size_type mat_start1, size_type mat_stride1, size_type mat_internal_size1,
                       size_type mat_size2, size_type mat_start2, size_type mat_stride2, size_type mat_internal_size2,
                       bool is_row_major);
#endif

  /* Copy CTOR */
  matrix_base(const self_type & other);

  /* Conversion CTOR */
  template<typename OtherNumericT>
  matrix_base(const matrix_base<OtherNumericT, SizeT, DistanceT> & other);

  self_type & operator=(const self_type & other);
  template<typename OtherNumericT>
  self_type & operator=(const matrix_base<OtherNumericT, SizeT, DistanceT> & other);

  /** @brief Implementation of the operation m1 = m2 @ alpha, where @ denotes either multiplication or division, and alpha is either a CPU or a GPU scalar
    * @param proxy  An expression template proxy class. */
  template<typename LHS, typename RHS, typename OP>
  self_type & operator=(const matrix_expression<const LHS, const RHS, OP> & proxy);
  // A = trans(B). Currently achieved in CPU memory
  self_type & operator=(const matrix_expression< const self_type, const self_type, op_trans> & proxy);
  template<typename LHS, typename RHS, typename OP>
  self_type & operator+=(const matrix_expression<const LHS, const RHS, OP> & proxy);
  template<typename LHS, typename RHS, typename OP>
  self_type & operator-=(const matrix_expression<const LHS, const RHS, OP> & proxy);
  /** @brief Assigns the supplied identity matrix to the matrix. */
  self_type & operator = (identity_matrix<NumericT> const & m);
  /** @brief Assigns the supplied zero matrix to the matrix. */
  self_type & operator = (zero_matrix<NumericT> const & m);
  /** @brief Assigns the supplied scalar vector to the matrix. */
  self_type & operator = (scalar_matrix<NumericT> const & m);
  //read-write access to an element of the matrix/matrix_range/matrix_slice
  /** @brief Read-write access to a single element of the matrix/matrix_range/matrix_slice */
  entry_proxy<NumericT> operator()(size_type row_index, size_type col_index);
  /** @brief Read access to a single element of the matrix/matrix_range/matrix_slice */
  const_entry_proxy<NumericT> operator()(size_type row_index, size_type col_index) const;
  self_type & operator += (const self_type & other);
  self_type & operator -= (const self_type & other);

  /** @brief Scales the matrix by a char (8-bit integer) */
  self_type & operator *= (char val);
  /** @brief Scales the matrix by a short integer */
  self_type & operator *= (short val);
  /** @brief Scales the matrix by an integer */
  self_type & operator *= (int val);
  /** @brief Scales the matrix by a long integer */
  self_type & operator *= (long val);
  /** @brief Scales the matrix by a single precision floating point value */
  self_type & operator *= (float val);
  /** @brief Scales the matrix by a double precision floating point value */
  self_type & operator *= (double val);

  /** @brief Scales the matrix by a char (8-bit integer) */
  self_type & operator /= (char val);
  /** @brief Scales the matrix by a short integer */
  self_type & operator /= (short val);
  /** @brief Scales the matrix by an integer */
  self_type & operator /= (int val);
  /** @brief Scales the matrix by a long integer */
  self_type & operator /= (long val);
  /** @brief Scales the matrix by a single precision floating point value */
  self_type & operator /= (float val);
  /** @brief Scales the matrix by a double precision floating point value */
  self_type & operator /= (double val);

  /** @brief Sign flip for the matrix. Emulated to be equivalent to -1.0 * matrix */
  matrix_expression<const self_type, const NumericT, op_mult> operator-() const;
  /** @brief Returns the number of rows */
  size_type size1() const { return size1_;}
  /** @brief Returns the number of columns */
  size_type size2() const { return size2_; }
  /** @brief Returns the number of rows */
  size_type start1() const { return start1_;}
  /** @brief Returns the number of columns */
  size_type start2() const { return start2_; }
  /** @brief Returns the number of rows */
  size_type stride1() const { return stride1_;}
  /** @brief Returns the number of columns */
  size_type stride2() const { return stride2_; }
  /** @brief Resets all entries to zero */
  void clear();
  /** @brief Returns the internal number of rows. Usually required for launching OpenCL kernels only */
  size_type internal_size1() const { return internal_size1_; }
  /** @brief Returns the internal number of columns. Usually required for launching OpenCL kernels only */
  size_type internal_size2() const { return internal_size2_; }
  /** @brief Returns the total amount of allocated memory in multiples of sizeof(NumericT) */
  size_type internal_size() const { return internal_size1() * internal_size2(); }
  /** @brief Returns the OpenCL handle, non-const-version */
  handle_type & handle()       { return elements_; }
  /** @brief Returns the OpenCL handle, const-version */
  const handle_type & handle() const { return elements_; }
  viennacl::memory_types memory_domain() const { return elements_.get_active_handle_id(); }
  bool row_major() const { return row_major_; }
  void switch_memory_context(viennacl::context new_ctx) { viennacl::backend::switch_memory_context<NumericT>(elements_, new_ctx); }

protected:
  void set_handle(viennacl::backend::mem_handle const & h);
  void resize(size_type rows, size_type columns, bool preserve = true);
private:
  size_type size1_;
  size_type size2_;
  size_type start1_;
  size_type start2_;
  size_type stride1_;
  size_type stride2_;
  size_type internal_size1_;
  size_type internal_size2_;
  bool row_major_fixed_; //helper flag to make layout of matrix<T, row_major> A; persistent
  bool row_major_;
  handle_type elements_;
}; //matrix

}

#endif
