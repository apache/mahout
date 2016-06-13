#ifndef VIENNACL_VECTOR_HPP_
#define VIENNACL_VECTOR_HPP_

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

/** @file  viennacl/vector.hpp
    @brief The vector type with operator-overloads and proxy classes is defined here.
           Linear algebra operations such as norms and inner products are located in linalg/vector_operations.hpp
*/

#include "viennacl/forwards.h"
#include "viennacl/detail/vector_def.hpp"
#include "viennacl/backend/memory.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/tools/entry_proxy.hpp"
#include "viennacl/linalg/detail/op_executor.hpp"
#include "viennacl/linalg/vector_operations.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/context.hpp"
#include "viennacl/traits/handle.hpp"

namespace viennacl
{

//
// Vector expression
//

/** @brief An expression template class that represents a binary operation that yields a vector
*
* In contrast to full expression templates as introduced by Veldhuizen, ViennaCL does not allow nested expressions.
* The reason is that this requires automated GPU viennacl::ocl::kernel generation, which then has to be compiles just-in-time.
* For performance-critical applications, one better writes the appropriate viennacl::ocl::kernels by hand.
*
* Assumption: dim(LHS) >= dim(RHS), where dim(scalar) = 0, dim(vector) = 1 and dim(matrix = 2)
*
* @tparam LHS   left hand side operand
* @tparam RHS   right hand side operand
* @tparam OP    the operator
*/
template<typename LHS, typename RHS, typename OP>
class vector_expression
{
  typedef typename viennacl::result_of::reference_if_nonscalar<LHS>::type     lhs_reference_type;
  typedef typename viennacl::result_of::reference_if_nonscalar<RHS>::type     rhs_reference_type;

public:
  enum { alignment = 1 };

  /** @brief Extracts the vector type from the two operands.
    */
  typedef vcl_size_t       size_type;

  vector_expression(LHS & l, RHS & r) : lhs_(l), rhs_(r) {}

  /** @brief Get left hand side operand
    */
  lhs_reference_type lhs() const { return lhs_; }
  /** @brief Get right hand side operand
    */
  rhs_reference_type rhs() const { return rhs_; }

  /** @brief Returns the size of the result vector */
  size_type size() const { return viennacl::traits::size(*this); }

private:
  /** @brief The left hand side operand */
  lhs_reference_type lhs_;
  /** @brief The right hand side operand */
  rhs_reference_type rhs_;
};

/** @brief A STL-type const-iterator for vector elements. Elements can be accessed, but cannot be manipulated. VERY SLOW!!
*
* Every dereference operation initiates a transfer from the GPU to the CPU. The overhead of such a transfer is around 50us, so 20.000 dereferences take one second.
* This is four orders of magnitude slower than similar dereferences on the CPU. However, increments and comparisons of iterators is as fast as for CPU types.
* If you need a fast iterator, copy the whole vector to the CPU first and iterate over the CPU object, e.g.
* std::vector<float> temp;
* copy(gpu_vector, temp);
* for (std::vector<float>::const_iterator iter = temp.begin();
*      iter != temp.end();
*      ++iter)
* {
*   //do something
* }
* Note that you may obtain inconsistent data if entries of gpu_vector are manipulated elsewhere in the meanwhile.
*
* @tparam NumericT  The underlying floating point type (either float or double)
* @tparam AlignmentV   Alignment of the underlying vector, @see vector
*/
template<class NumericT, unsigned int AlignmentV>
class const_vector_iterator
{
  typedef const_vector_iterator<NumericT, AlignmentV>    self_type;
public:
  typedef scalar<NumericT>            value_type;
  typedef vcl_size_t                size_type;
  typedef vcl_ptrdiff_t                 difference_type;
  typedef viennacl::backend::mem_handle handle_type;

  //const_vector_iterator() {}

  /** @brief Constructor
    *   @param vec    The vector over which to iterate
    *   @param index  The starting index of the iterator
    *   @param start  First index of the element in the vector pointed to be the iterator (for vector_range and vector_slice)
    *   @param stride Stride for the support of vector_slice
    */
  const_vector_iterator(vector_base<NumericT> const & vec,
                        size_type index,
                        size_type start = 0,
                        size_type stride = 1) : elements_(vec.handle()), index_(index), start_(start), stride_(stride) {}

  /** @brief Constructor for vector-like treatment of arbitrary buffers
    *   @param elements  The buffer over which to iterate
    *   @param index     The starting index of the iterator
    *   @param start     First index of the element in the vector pointed to be the iterator (for vector_range and vector_slice)
    *   @param stride    Stride for the support of vector_slice
    */
  const_vector_iterator(handle_type const & elements,
                        size_type index,
                        size_type start = 0,
                        size_type stride = 1) : elements_(elements), index_(index), start_(start), stride_(stride) {}

  /** @brief Dereferences the iterator and returns the value of the element. For convenience only, performance is poor due to OpenCL overhead! */
  value_type operator*(void) const
  {
    value_type result;
    result = const_entry_proxy<NumericT>(start_ + index_ * stride(), elements_);
    return result;
  }
  self_type operator++(void) { ++index_; return *this; }
  self_type operator++(int) { self_type tmp = *this; ++(*this); return tmp; }

  bool operator==(self_type const & other) const { return index_ == other.index_; }
  bool operator!=(self_type const & other) const { return index_ != other.index_; }

  //        self_type & operator=(self_type const & other)
  //        {
  //           index_ = other._index;
  //           elements_ = other._elements;
  //           return *this;
  //        }

  difference_type operator-(self_type const & other) const
  {
    assert( (other.start_ == start_) && (other.stride_ == stride_) && bool("Iterators are not from the same vector (proxy)!"));
    return static_cast<difference_type>(index_) - static_cast<difference_type>(other.index_);
  }
  self_type operator+(difference_type diff) const { return self_type(elements_, size_type(difference_type(index_) + diff), start_, stride_); }

  //vcl_size_t index() const { return index_; }
  /** @brief Offset of the current element index with respect to the beginning of the buffer */
  size_type offset() const { return start_ + index_ * stride(); }

  /** @brief Index increment in the underlying buffer when incrementing the iterator to the next element */
  size_type stride() const { return stride_; }
  handle_type const & handle() const { return elements_; }

protected:
  /** @brief  The index of the entry the iterator is currently pointing to */
  handle_type const & elements_;
  size_type index_;  //offset from the beginning of elements_
  size_type start_;
  size_type stride_;
};


/** @brief A STL-type iterator for vector elements. Elements can be accessed and manipulated. VERY SLOW!!
*
* Every dereference operation initiates a transfer from the GPU to the CPU. The overhead of such a transfer is around 50us, so 20.000 dereferences take one second.
* This is four orders of magnitude slower than similar dereferences on the CPU. However, increments and comparisons of iterators is as fast as for CPU types.
* If you need a fast iterator, copy the whole vector to the CPU first and iterate over the CPU object, e.g.
* std::vector<float> temp;
* copy(gpu_vector, temp);
* for (std::vector<float>::const_iterator iter = temp.begin();
*      iter != temp.end();
*      ++iter)
* {
*   //do something
* }
* copy(temp, gpu_vector);
* Note that you may obtain inconsistent data if you manipulate entries of gpu_vector in the meanwhile.
*
* @tparam NumericT  The underlying floating point type (either float or double)
* @tparam AlignmentV   Alignment of the underlying vector, @see vector
*/
template<class NumericT, unsigned int AlignmentV>
class vector_iterator : public const_vector_iterator<NumericT, AlignmentV>
{
  typedef const_vector_iterator<NumericT, AlignmentV>  base_type;
  typedef vector_iterator<NumericT, AlignmentV>        self_type;
public:
  typedef typename base_type::handle_type               handle_type;
  typedef typename base_type::size_type             size_type;
  typedef typename base_type::difference_type           difference_type;

  vector_iterator(handle_type const & elements,
                  size_type index,
                  size_type start = 0,
                  size_type stride = 1)  : base_type(elements, index, start, stride), elements_(elements) {}
  /** @brief Constructor
    *   @param vec    The vector over which to iterate
    *   @param index  The starting index of the iterator
    *   @param start  Offset from the beginning of the underlying vector (for ranges and slices)
    *   @param stride Stride for slices
    */
  vector_iterator(vector_base<NumericT> & vec,
                  size_type index,
                  size_type start = 0,
                  size_type stride = 1) : base_type(vec, index, start, stride), elements_(vec.handle()) {}
  //vector_iterator(base_type const & b) : base_type(b) {}

  entry_proxy<NumericT> operator*(void)
  {
    return entry_proxy<NumericT>(base_type::start_ + base_type::index_ * base_type::stride(), elements_);
  }

  difference_type operator-(self_type const & other) const { difference_type result = base_type::index_; return (result - static_cast<difference_type>(other.index_)); }
  self_type operator+(difference_type diff) const { return self_type(elements_, static_cast<vcl_size_t>(static_cast<difference_type>(base_type::index_) + diff), base_type::start_, base_type::stride_); }

  handle_type       & handle()       { return elements_; }
  handle_type const & handle() const { return base_type::elements_; }

  //operator base_type() const
  //{
  //  return base_type(base_type::elements_, base_type::index_, base_type::start_, base_type::stride_);
  //}
private:
  handle_type elements_;
};


template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT>::vector_base() : size_(0), start_(0), stride_(1), internal_size_(0) { /* Note: One must not call ::init() here because a vector might have been created globally before the backend has become available */ }

template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT>::vector_base(viennacl::backend::mem_handle & h,
                                                     size_type vec_size, size_type vec_start, size_type vec_stride)
  : size_(vec_size), start_(vec_start), stride_(vec_stride), internal_size_(vec_size), elements_(h) {}

template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT>::vector_base(size_type vec_size, viennacl::context ctx)
  : size_(vec_size), start_(0), stride_(1), internal_size_(viennacl::tools::align_to_multiple<size_type>(size_, dense_padding_size))
{
  if (size_ > 0)
  {
    viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), ctx);
    clear();
  }
}

// CUDA or host memory:
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT>::vector_base(NumericT * ptr_to_mem, viennacl::memory_types mem_type, size_type vec_size, vcl_size_t start, size_type stride)
  : size_(vec_size), start_(start), stride_(stride), internal_size_(vec_size)
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

  elements_.raw_size(sizeof(NumericT) * vec_size);

}

#ifdef VIENNACL_WITH_OPENCL
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT>::vector_base(cl_mem existing_mem, size_type vec_size, size_type start, size_type stride, viennacl::context ctx)
  : size_(vec_size), start_(start), stride_(stride), internal_size_(vec_size)
{
  elements_.switch_active_handle_id(viennacl::OPENCL_MEMORY);
  elements_.opencl_handle() = existing_mem;
  elements_.opencl_handle().inc();  //prevents that the user-provided memory is deleted once the vector object is destroyed.
  elements_.opencl_handle().context(ctx.opencl_context());
  elements_.raw_size(sizeof(NumericT) * vec_size);
}
#endif


template<class NumericT, typename SizeT, typename DistanceT>
template<typename LHS, typename RHS, typename OP>
vector_base<NumericT, SizeT, DistanceT>::vector_base(vector_expression<const LHS, const RHS, OP> const & proxy)
  : size_(viennacl::traits::size(proxy)), start_(0), stride_(1), internal_size_(viennacl::tools::align_to_multiple<size_type>(size_, dense_padding_size))
{
  if (size_ > 0)
  {
    viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), viennacl::traits::context(proxy));
    clear();
  }
  self_type::operator=(proxy);
}

// Copy CTOR:
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT>::vector_base(const vector_base<NumericT, SizeT, DistanceT> & other) :
  size_(other.size_), start_(0), stride_(1),
  internal_size_(viennacl::tools::align_to_multiple<size_type>(other.size_, dense_padding_size))
{
  elements_.switch_active_handle_id(viennacl::traits::active_handle_id(other));
  if (internal_size() > 0)
  {
    viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), viennacl::traits::context(other));
    clear();
    self_type::operator=(other);
  }
}

// Conversion CTOR:
template<typename NumericT, typename SizeT, typename DistanceT>
template<typename OtherNumericT>
vector_base<NumericT, SizeT, DistanceT>::vector_base(const vector_base<OtherNumericT> & other) :
  size_(other.size()), start_(0), stride_(1),
  internal_size_(viennacl::tools::align_to_multiple<size_type>(other.size(), dense_padding_size))
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
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator=(const self_type & vec)
{
  assert( ( (vec.size() == size()) || (size() == 0) )
          && bool("Incompatible vector sizes!"));

  if (&vec==this)
    return *this;

  if (vec.size() > 0)
  {
    if (size_ == 0)
    {
      size_ = vec.size();
      internal_size_ = viennacl::tools::align_to_multiple<size_type>(size_, dense_padding_size);
      elements_.switch_active_handle_id(vec.handle().get_active_handle_id());
      viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), viennacl::traits::context(vec));
      pad();
    }

    viennacl::linalg::av(*this,
                         vec, cpu_value_type(1.0), 1, false, false);
  }

  return *this;
}


/** @brief Implementation of the operation v1 = v2 @ alpha, where @ denotes either multiplication or division, and alpha is either a CPU or a GPU scalar
*
* @param proxy  An expression template proxy class.
*/
template<class NumericT, typename SizeT, typename DistanceT>
template<typename LHS, typename RHS, typename OP>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator=(const vector_expression<const LHS, const RHS, OP> & proxy)
{
  assert( ( (viennacl::traits::size(proxy) == size()) || (size() == 0) )
          && bool("Incompatible vector sizes!"));

  // initialize the necessary buffer
  if (size() == 0)
  {
    size_ = viennacl::traits::size(proxy);
    internal_size_ = viennacl::tools::align_to_multiple<size_type>(size_, dense_padding_size);
    viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), viennacl::traits::context(proxy));
    pad();
  }

  linalg::detail::op_executor<self_type, op_assign, vector_expression<const LHS, const RHS, OP> >::apply(*this, proxy);

  return *this;
}

// convert from vector with other numeric type
template<class NumericT, typename SizeT, typename DistanceT>
template<typename OtherNumericT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>:: operator = (const vector_base<OtherNumericT> & v1)
{
  assert( ( (v1.size() == size()) || (size() == 0) )
          && bool("Incompatible vector sizes!"));

  if (size() == 0)
  {
    size_ = v1.size();
    if (size_ > 0)
    {
      internal_size_ = viennacl::tools::align_to_multiple<size_type>(size_, dense_padding_size);
      viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), viennacl::traits::context(v1));
      pad();
    }
  }

  viennacl::linalg::convert(*this, v1);

  return *this;
}

/** @brief Creates the vector from the supplied unit vector. */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator = (unit_vector<NumericT> const & v)
{
  assert( ( (v.size() == size()) || (size() == 0) )
          && bool("Incompatible vector sizes!"));

  if (size() == 0)
  {
    size_ = v.size();
    internal_size_ = viennacl::tools::align_to_multiple<size_type>(size_, dense_padding_size);
    if (size_ > 0)
    {
      viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), v.context());
      clear();
    }
  }
  else
    viennacl::linalg::vector_assign(*this, NumericT(0));

  if (size_ > 0)
    this->operator()(v.index()) = NumericT(1);

  return *this;
}

/** @brief Creates the vector from the supplied zero vector. */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator = (zero_vector<NumericT> const & v)
{
  assert( ( (v.size() == size()) || (size() == 0) )
          && bool("Incompatible vector sizes!"));

  if (size() == 0)
  {
    size_ = v.size();
    internal_size_ = viennacl::tools::align_to_multiple<size_type>(size_, dense_padding_size);
    if (size_ > 0)
    {
      viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), v.context());
      clear();
    }
  }
  else
    viennacl::linalg::vector_assign(*this, NumericT(0));

  return *this;
}

/** @brief Creates the vector from the supplied scalar vector. */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator = (scalar_vector<NumericT> const & v)
{
  assert( ( (v.size() == size()) || (size() == 0) )
          && bool("Incompatible vector sizes!"));

  if (size() == 0)
  {
    size_ = v.size();
    internal_size_ = viennacl::tools::align_to_multiple<size_type>(size_, dense_padding_size);
    if (size_ > 0)
    {
      viennacl::backend::memory_create(elements_, sizeof(NumericT)*internal_size(), v.context());
      pad();
    }
  }

  if (size_ > 0)
    viennacl::linalg::vector_assign(*this, v[0]);

  return *this;
}



///////////////////////////// Matrix Vector interaction start ///////////////////////////////////

//Note: The following operator overloads are defined in matrix_operations.hpp, compressed_matrix_operations.hpp and coordinate_matrix_operations.hpp
//This is certainly not the nicest approach and will most likely by changed in the future, but it works :-)

//matrix<>
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator=(const viennacl::vector_expression< const matrix_base<NumericT>, const vector_base<NumericT>, viennacl::op_prod> & proxy)
{
  assert(viennacl::traits::size1(proxy.lhs()) == size() && bool("Size check failed for v1 = A * v2: size1(A) != size(v1)"));

  // check for the special case x = A * x
  if (viennacl::traits::handle(proxy.rhs()) == viennacl::traits::handle(*this))
  {
    viennacl::vector<NumericT> result(viennacl::traits::size1(proxy.lhs()));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    *this = result;
  }
  else
  {
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this);
  }
  return *this;
}


//transposed_matrix_proxy:
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator=(const vector_expression< const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans >,
                                                                                             const vector_base<NumericT>,
                                                                                             op_prod> & proxy)
{
  assert(viennacl::traits::size1(proxy.lhs()) == size() && bool("Size check failed in v1 = trans(A) * v2: size2(A) != size(v1)"));

  // check for the special case x = trans(A) * x
  if (viennacl::traits::handle(proxy.rhs()) == viennacl::traits::handle(*this))
  {
    viennacl::vector<NumericT> result(viennacl::traits::size1(proxy.lhs()));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    *this = result;
  }
  else
  {
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), *this);
  }
  return *this;
}

///////////////////////////// Matrix Vector interaction end ///////////////////////////////////


//////////////////////////// Read-write access to an element of the vector start ///////////////////
//read-write access to an element of the vector

template<class NumericT, typename SizeT, typename DistanceT>
entry_proxy<NumericT> vector_base<NumericT, SizeT, DistanceT>::operator()(size_type index)
{
  assert( (size() > 0)  && bool("Cannot apply operator() to vector of size zero!"));
  assert( index < size() && bool("Index out of bounds!") );

  return entry_proxy<NumericT>(start_ + stride_ * index, elements_);
}

template<class NumericT, typename SizeT, typename DistanceT>
entry_proxy<NumericT> vector_base<NumericT, SizeT, DistanceT>::operator[](size_type index)
{
  assert( (size() > 0)  && bool("Cannot apply operator() to vector of size zero!"));
  assert( index < size() && bool("Index out of bounds!") );

  return entry_proxy<NumericT>(start_ + stride_ * index, elements_);
}

template<class NumericT, typename SizeT, typename DistanceT>
const_entry_proxy<NumericT> vector_base<NumericT, SizeT, DistanceT>::operator()(size_type index) const
{
  assert( (size() > 0)  && bool("Cannot apply operator() to vector of size zero!"));
  assert( index < size() && bool("Index out of bounds!") );

  return const_entry_proxy<NumericT>(start_ + stride_ * index, elements_);
}

template<class NumericT, typename SizeT, typename DistanceT>
const_entry_proxy<NumericT> vector_base<NumericT, SizeT, DistanceT>::operator[](size_type index) const
{
  assert( (size() > 0)  && bool("Cannot apply operator() to vector of size zero!"));
  assert( index < size() && bool("Index out of bounds!") );

  return const_entry_proxy<NumericT>(start_ + stride_ * index, elements_);
}

//////////////////////////// Read-write access to an element of the vector end ///////////////////


//
// Operator overloads with implicit conversion (thus cannot be made global without introducing additional headache)
//
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator += (const self_type & vec)
{
  assert(vec.size() == size() && bool("Incompatible vector sizes!"));

  if (size() > 0)
    viennacl::linalg::avbv(*this,
                           *this, NumericT(1.0), 1, false, false,
                           vec,   NumericT(1.0), 1, false, false);
  return *this;
}

template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator -= (const self_type & vec)
{
  assert(vec.size() == size() && bool("Incompatible vector sizes!"));

  if (size() > 0)
    viennacl::linalg::avbv(*this,
                           *this, NumericT(1.0),  1, false, false,
                           vec,   NumericT(-1.0), 1, false, false);
  return *this;
}

/** @brief Scales a vector (or proxy) by a char (8-bit integer) value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator *= (char val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, false, false);
  return *this;
}
/** @brief Scales a vector (or proxy) by a short integer value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator *= (short val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, false, false);
  return *this;
}
/** @brief Scales a vector (or proxy) by an integer value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator *= (int val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, false, false);
  return *this;
}
/** @brief Scales a vector (or proxy) by a long integer value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator *= (long val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, false, false);
  return *this;
}
/** @brief Scales a vector (or proxy) by a single precision floating point value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator *= (float val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, false, false);
  return *this;
}
/** @brief Scales a vector (or proxy) by a double precision floating point value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator *= (double val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, false, false);
  return *this;
}


/** @brief Scales this vector by a char (8-bit) value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator /= (char val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, true, false);
  return *this;
}
/** @brief Scales this vector by a short integer value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator /= (short val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, true, false);
  return *this;
}
/** @brief Scales this vector by an integer value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator /= (int val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, true, false);
  return *this;
}
/** @brief Scales this vector by a long integer value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator /= (long val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, true, false);
  return *this;
}
/** @brief Scales this vector by a single precision floating point value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator /= (float val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, true, false);
  return *this;
}
/** @brief Scales this vector by a double precision floating point value */
template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::operator /= (double val)
{
  if (size() > 0)
    viennacl::linalg::av(*this,
                         *this, NumericT(val), 1, true, false);
  return *this;
}


/** @brief Scales the vector by a char (8-bit value) 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_mult>
vector_base<NumericT, SizeT, DistanceT>::operator * (char value) const
{
  return vector_expression< const self_type, const NumericT, op_mult>(*this, NumericT(value));
}
/** @brief Scales the vector by a short integer 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_mult>
vector_base<NumericT, SizeT, DistanceT>::operator * (short value) const
{
  return vector_expression< const self_type, const NumericT, op_mult>(*this, NumericT(value));
}
/** @brief Scales the vector by an integer 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_mult>
vector_base<NumericT, SizeT, DistanceT>::operator * (int value) const
{
  return vector_expression< const self_type, const NumericT, op_mult>(*this, NumericT(value));
}
/** @brief Scales the vector by a long integer 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_mult>
vector_base<NumericT, SizeT, DistanceT>::operator * (long value) const
{
  return vector_expression< const self_type, const NumericT, op_mult>(*this, NumericT(value));
}
/** @brief Scales the vector by a single precision floating point number 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_mult>
vector_base<NumericT, SizeT, DistanceT>::operator * (float value) const
{
  return vector_expression< const self_type, const NumericT, op_mult>(*this, NumericT(value));
}
/** @brief Scales the vector by a single precision floating point number 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_mult>
vector_base<NumericT, SizeT, DistanceT>::operator * (double value) const
{
  return vector_expression< const self_type, const NumericT, op_mult>(*this, NumericT(value));
}


/** @brief Scales the vector by a char (8-bit value) 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_div>
vector_base<NumericT, SizeT, DistanceT>::operator / (char value) const
{
  return vector_expression< const self_type, const NumericT, op_div>(*this, NumericT(value));
}
/** @brief Scales the vector by a short integer 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_div>
vector_base<NumericT, SizeT, DistanceT>::operator / (short value) const
{
  return vector_expression< const self_type, const NumericT, op_div>(*this, NumericT(value));
}
/** @brief Scales the vector by an integer 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_div>
vector_base<NumericT, SizeT, DistanceT>::operator / (int value) const
{
  return vector_expression< const self_type, const NumericT, op_div>(*this, NumericT(value));
}
/** @brief Scales the vector by a long integer 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_div>
vector_base<NumericT, SizeT, DistanceT>::operator / (long value) const
{
  return vector_expression< const self_type, const NumericT, op_div>(*this, NumericT(value));
}
/** @brief Scales the vector by a single precision floating point number 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_div>
vector_base<NumericT, SizeT, DistanceT>::operator / (float value) const
{
  return vector_expression< const self_type, const NumericT, op_div>(*this, NumericT(value));
}
/** @brief Scales the vector by a double precision floating point number 'alpha' and returns an expression template */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression< const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_div>
vector_base<NumericT, SizeT, DistanceT>::operator / (double value) const
{
  return vector_expression< const self_type, const NumericT, op_div>(*this, NumericT(value));
}


/** @brief Sign flip for the vector. Emulated to be equivalent to -1.0 * vector */
template<class NumericT, typename SizeT, typename DistanceT>
vector_expression<const vector_base<NumericT, SizeT, DistanceT>, const NumericT, op_mult>
vector_base<NumericT, SizeT, DistanceT>::operator-() const
{
  return vector_expression<const self_type, const NumericT, op_mult>(*this, NumericT(-1.0));
}

//
//// iterators:
//

/** @brief Returns an iterator pointing to the beginning of the vector  (STL like)*/
template<class NumericT, typename SizeT, typename DistanceT>
typename vector_base<NumericT, SizeT, DistanceT>::iterator vector_base<NumericT, SizeT, DistanceT>::begin()
{
  return iterator(*this, 0, start_, stride_);
}

/** @brief Returns an iterator pointing to the end of the vector (STL like)*/
template<class NumericT, typename SizeT, typename DistanceT>
typename vector_base<NumericT, SizeT, DistanceT>::iterator vector_base<NumericT, SizeT, DistanceT>::end()
{
  return iterator(*this, size(), start_, stride_);
}

/** @brief Returns a const-iterator pointing to the beginning of the vector (STL like)*/
template<class NumericT, typename SizeT, typename DistanceT>
typename vector_base<NumericT, SizeT, DistanceT>::const_iterator vector_base<NumericT, SizeT, DistanceT>::begin() const
{
  return const_iterator(*this, 0, start_, stride_);
}

template<class NumericT, typename SizeT, typename DistanceT>
typename vector_base<NumericT, SizeT, DistanceT>::const_iterator vector_base<NumericT, SizeT, DistanceT>::end() const
{
  return const_iterator(*this, size(), start_, stride_);
}

template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::swap(self_type & other)
{
  viennacl::linalg::vector_swap(*this, other);
  return *this;
}


template<class NumericT, typename SizeT, typename DistanceT>
void vector_base<NumericT, SizeT, DistanceT>::clear()
{
  viennacl::linalg::vector_assign(*this, cpu_value_type(0.0), true);
}

template<class NumericT, typename SizeT, typename DistanceT>
vector_base<NumericT, SizeT, DistanceT> & vector_base<NumericT, SizeT, DistanceT>::fast_swap(self_type & other)
{
  assert(this->size_ == other.size_ && bool("Vector size mismatch"));
  this->elements_.swap(other.elements_);
  return *this;
}

template<class NumericT, typename SizeT, typename DistanceT>
void vector_base<NumericT, SizeT, DistanceT>::pad()
{
  if (internal_size() != size())
  {
    std::vector<NumericT> pad(internal_size() - size());
    viennacl::backend::memory_write(elements_, sizeof(NumericT) * size(), sizeof(NumericT) * pad.size(), &(pad[0]));
  }
}

template<class NumericT, typename SizeT, typename DistanceT>
void vector_base<NumericT, SizeT, DistanceT>::switch_memory_context(viennacl::context new_ctx)
{
  viennacl::backend::switch_memory_context<NumericT>(elements_, new_ctx);
}

//TODO: Think about implementing the following public member functions
//void insert_element(unsigned int i, NumericT val){}
//void erase_element(unsigned int i){}

template<class NumericT, typename SizeT, typename DistanceT>
void vector_base<NumericT, SizeT, DistanceT>::resize(size_type new_size, bool preserve)
{
  resize_impl(new_size, viennacl::traits::context(*this), preserve);
}

template<class NumericT, typename SizeT, typename DistanceT>
void vector_base<NumericT, SizeT, DistanceT>::resize(size_type new_size, viennacl::context ctx, bool preserve)
{
  resize_impl(new_size, ctx, preserve);
}

template<class NumericT, typename SizeT, typename DistanceT>
void vector_base<NumericT, SizeT, DistanceT>::resize_impl(size_type new_size, viennacl::context ctx, bool preserve)
{
  assert(new_size > 0 && bool("Positive size required when resizing vector!"));

  if (new_size != size_)
  {
    vcl_size_t new_internal_size = viennacl::tools::align_to_multiple<vcl_size_t>(new_size, dense_padding_size);

    std::vector<NumericT> temp(size_);
    if (preserve && size_ > 0)
      fast_copy(*this, temp);
    temp.resize(new_size);  //drop all entries above new_size
    temp.resize(new_internal_size); //enlarge to fit new internal size

    if (new_internal_size != internal_size())
    {
      viennacl::backend::memory_create(elements_, sizeof(NumericT)*new_internal_size, ctx, NULL);
    }

    fast_copy(temp, *this);
    size_ = new_size;
    internal_size_ = viennacl::tools::align_to_multiple<size_type>(size_, dense_padding_size);
    pad();
  }

}


template<class NumericT, unsigned int AlignmentV>
class vector : public vector_base<NumericT>
{
  typedef vector<NumericT, AlignmentV>         self_type;
  typedef vector_base<NumericT>               base_type;

public:
  typedef typename base_type::size_type                  size_type;
  typedef typename base_type::difference_type            difference_type;

  /** @brief Default constructor in order to be compatible with various containers.
  */
  explicit vector() : base_type() { /* Note: One must not call ::init() here because the vector might have been created globally before the backend has become available */ }

  /** @brief An explicit constructor for the vector, allocating the given amount of memory (plus a padding specified by 'AlignmentV')
  *
  * @param vec_size   The length (i.e. size) of the vector.
  */
  explicit vector(size_type vec_size) : base_type(vec_size) {}

  explicit vector(size_type vec_size, viennacl::context ctx) : base_type(vec_size, ctx) {}

  explicit vector(NumericT * ptr_to_mem, viennacl::memory_types mem_type, size_type vec_size, size_type start = 0, size_type stride = 1)
    : base_type(ptr_to_mem, mem_type, vec_size, start, stride) {}

#ifdef VIENNACL_WITH_OPENCL
  /** @brief Create a vector from existing OpenCL memory
  *
  * Note: The provided memory must take an eventual AlignmentV into account, i.e. existing_mem must be at least of size internal_size()!
  * This is trivially the case with the default alignment, but should be considered when using vector<> with an alignment parameter not equal to 1.
  *
  * @param existing_mem   An OpenCL handle representing the memory
  * @param vec_size       The size of the vector.
  */
  explicit vector(cl_mem existing_mem, size_type vec_size, size_type start = 0, size_type stride = 1) : base_type(existing_mem, vec_size, start, stride) {}

  /** @brief An explicit constructor for the vector, allocating the given amount of memory (plus a padding specified by 'AlignmentV') and the OpenCL context provided
  *
  * @param vec_size   The length (i.e. size) of the vector.
  * @param ctx        The context
  */
  explicit vector(size_type vec_size, viennacl::ocl::context const & ctx) : base_type(vec_size, ctx) {}
#endif

  template<typename LHS, typename RHS, typename OP>
  vector(vector_expression<const LHS, const RHS, OP> const & proxy) : base_type(proxy) {}

  vector(const base_type & v) : base_type(v.size(), viennacl::traits::context(v))
  {
    if (v.size() > 0)
      base_type::operator=(v);
  }

  vector(const self_type & v) : base_type(v.size(), viennacl::traits::context(v))
  {
    if (v.size() > 0)
      base_type::operator=(v);
  }

  /** @brief Creates the vector from the supplied unit vector. */
  vector(unit_vector<NumericT> const & v) : base_type(v.size())
  {
    if (v.size() > 0)
      this->operator()(v.index()) = NumericT(1);;
  }

  /** @brief Creates the vector from the supplied zero vector. */
  vector(zero_vector<NumericT> const & v) : base_type(v.size(), v.context())
  {
    if (v.size() > 0)
      viennacl::linalg::vector_assign(*this, NumericT(0.0));
  }

  /** @brief Creates the vector from the supplied scalar vector. */
  vector(scalar_vector<NumericT> const & v) : base_type(v.size(), v.context())
  {
    if (v.size() > 0)
      viennacl::linalg::vector_assign(*this, v[0]);
  }

  // the following is used to circumvent an issue with Clang 3.0 when 'using base_type::operator=;' directly
  template<typename T>
  self_type & operator=(T const & other)
  {
    base_type::operator=(other);
    return *this;
  }

  using base_type::operator+=;
  using base_type::operator-=;

  //enlarge or reduce allocated memory and set unused memory to zero
  /** @brief Resizes the allocated memory for the vector. Pads the memory to be a multiple of 'AlignmentV'
  *
  *  @param new_size  The new size of the vector
  *  @param preserve  If true, old entries of the vector are preserved, otherwise eventually discarded.
  */
  void resize(size_type new_size, bool preserve = true)
  {
    base_type::resize(new_size, preserve);
  }

  void resize(size_type new_size, viennacl::context ctx, bool preserve = true)
  {
    base_type::resize(new_size, ctx, preserve);
  }

  /** @brief Swaps the handles of two vectors by swapping the OpenCL handles only, no data copy
  */
  self_type & fast_swap(self_type & other)
  {
    base_type::fast_swap(other);
    return *this;
  }

  void switch_memory_context(viennacl::context new_ctx)
  {
    base_type::switch_memory_context(new_ctx);
  }

}; //vector

/** @brief Tuple class holding pointers to multiple vectors. Mainly used as a temporary object returned from viennacl::tie(). */
template<typename ScalarT>
class vector_tuple
{
  typedef vector_base<ScalarT>   VectorType;

public:
  // 2 vectors

  vector_tuple(VectorType const & v0, VectorType const & v1) : const_vectors_(2), non_const_vectors_()
  {
    const_vectors_[0] = &v0;
    const_vectors_[1] = &v1;
  }
  vector_tuple(VectorType       & v0, VectorType       & v1) : const_vectors_(2), non_const_vectors_(2)
  {
    const_vectors_[0] = &v0; non_const_vectors_[0] = &v0;
    const_vectors_[1] = &v1; non_const_vectors_[1] = &v1;
  }

  // 3 vectors

  vector_tuple(VectorType const & v0, VectorType const & v1, VectorType const & v2) : const_vectors_(3), non_const_vectors_()
  {
    const_vectors_[0] = &v0;
    const_vectors_[1] = &v1;
    const_vectors_[2] = &v2;
  }
  vector_tuple(VectorType       & v0, VectorType       & v1, VectorType       & v2) : const_vectors_(3), non_const_vectors_(3)
  {
    const_vectors_[0] = &v0; non_const_vectors_[0] = &v0;
    const_vectors_[1] = &v1; non_const_vectors_[1] = &v1;
    const_vectors_[2] = &v2; non_const_vectors_[2] = &v2;
  }

  // 4 vectors

  vector_tuple(VectorType const & v0, VectorType const & v1, VectorType const & v2, VectorType const & v3) : const_vectors_(4), non_const_vectors_()
  {
    const_vectors_[0] = &v0;
    const_vectors_[1] = &v1;
    const_vectors_[2] = &v2;
    const_vectors_[3] = &v3;
  }
  vector_tuple(VectorType       & v0, VectorType       & v1, VectorType       & v2, VectorType       & v3) : const_vectors_(4), non_const_vectors_(4)
  {
    const_vectors_[0] = &v0; non_const_vectors_[0] = &v0;
    const_vectors_[1] = &v1; non_const_vectors_[1] = &v1;
    const_vectors_[2] = &v2; non_const_vectors_[2] = &v2;
    const_vectors_[3] = &v3; non_const_vectors_[3] = &v3;
  }

  // add more overloads here

  // generic interface:

  vector_tuple(std::vector<VectorType const *> const & vecs) : const_vectors_(vecs.size()), non_const_vectors_()
  {
    for (vcl_size_t i=0; i<vecs.size(); ++i)
      const_vectors_[i] = vecs[i];
  }

  vector_tuple(std::vector<VectorType *> const & vecs) : const_vectors_(vecs.size()), non_const_vectors_(vecs.size())
  {
    for (vcl_size_t i=0; i<vecs.size(); ++i)
    {
      const_vectors_[i] = vecs[i];
      non_const_vectors_[i] = vecs[i];
    }
  }

  vcl_size_t size()       const { return non_const_vectors_.size(); }
  vcl_size_t const_size() const { return     const_vectors_.size(); }

  VectorType       &       at(vcl_size_t i) const { return *(non_const_vectors_.at(i)); }
  VectorType const & const_at(vcl_size_t i) const { return     *(const_vectors_.at(i)); }

private:
  std::vector<VectorType const *>   const_vectors_;
  std::vector<VectorType *>         non_const_vectors_;
};

// 2 args
template<typename ScalarT>
vector_tuple<ScalarT> tie(vector_base<ScalarT> const & v0, vector_base<ScalarT> const & v1) { return vector_tuple<ScalarT>(v0, v1); }

template<typename ScalarT>
vector_tuple<ScalarT> tie(vector_base<ScalarT>       & v0, vector_base<ScalarT>       & v1) { return vector_tuple<ScalarT>(v0, v1); }

// 3 args
template<typename ScalarT>
vector_tuple<ScalarT> tie(vector_base<ScalarT> const & v0, vector_base<ScalarT> const & v1, vector_base<ScalarT> const & v2) { return vector_tuple<ScalarT>(v0, v1, v2); }

template<typename ScalarT>
vector_tuple<ScalarT> tie(vector_base<ScalarT>       & v0, vector_base<ScalarT>       & v1, vector_base<ScalarT>       & v2) { return vector_tuple<ScalarT>(v0, v1, v2); }

// 4 args
template<typename ScalarT>
vector_tuple<ScalarT> tie(vector_base<ScalarT> const & v0, vector_base<ScalarT> const & v1, vector_base<ScalarT> const & v2, vector_base<ScalarT> const & v3)
{
  return vector_tuple<ScalarT>(v0, v1, v2, v3);
}

template<typename ScalarT>
vector_tuple<ScalarT> tie(vector_base<ScalarT>       & v0, vector_base<ScalarT>       & v1, vector_base<ScalarT>       & v2, vector_base<ScalarT>       & v3)
{
  return vector_tuple<ScalarT>(v0, v1, v2, v3);
}

// 5 args
template<typename ScalarT>
vector_tuple<ScalarT> tie(vector_base<ScalarT> const & v0,
                          vector_base<ScalarT> const & v1,
                          vector_base<ScalarT> const & v2,
                          vector_base<ScalarT> const & v3,
                          vector_base<ScalarT> const & v4)
{
  typedef vector_base<ScalarT> const *       VectorPointerType;
  std::vector<VectorPointerType> vec(5);
  vec[0] = &v0;
  vec[1] = &v1;
  vec[2] = &v2;
  vec[3] = &v3;
  vec[4] = &v4;
  return vector_tuple<ScalarT>(vec);
}

template<typename ScalarT>
vector_tuple<ScalarT> tie(vector_base<ScalarT> & v0,
                          vector_base<ScalarT> & v1,
                          vector_base<ScalarT> & v2,
                          vector_base<ScalarT> & v3,
                          vector_base<ScalarT> & v4)
{
  typedef vector_base<ScalarT> *       VectorPointerType;
  std::vector<VectorPointerType> vec(5);
  vec[0] = &v0;
  vec[1] = &v1;
  vec[2] = &v2;
  vec[3] = &v3;
  vec[4] = &v4;
  return vector_tuple<ScalarT>(vec);
}

// TODO: Add more arguments to tie() here. Maybe use some preprocessor magic to accomplish this.

//
//////////////////// Copy from GPU to CPU //////////////////////////////////
//


/** @brief STL-like transfer of a GPU vector to the CPU. The cpu type is assumed to reside in a linear piece of memory, such as e.g. for std::vector.
*
* This method is faster than the plain copy() function, because entries are
* directly written to the cpu vector, starting with &(*cpu.begin()) However,
* keep in mind that the cpu type MUST represent a linear piece of
* memory, otherwise you will run into undefined behavior.
*
* @param gpu_begin  GPU iterator pointing to the beginning of the gpu vector (STL-like)
* @param gpu_end    GPU iterator pointing to the end of the vector (STL-like)
* @param cpu_begin  Output iterator for the cpu vector. The cpu vector must be at least as long as the gpu vector!
*/
template<typename NumericT, unsigned int AlignmentV, typename CPU_ITERATOR>
void fast_copy(const const_vector_iterator<NumericT, AlignmentV> & gpu_begin,
               const const_vector_iterator<NumericT, AlignmentV> & gpu_end,
               CPU_ITERATOR cpu_begin )
{
  if (gpu_begin != gpu_end)
  {
    if (gpu_begin.stride() == 1)
    {
      viennacl::backend::memory_read(gpu_begin.handle(),
                                     sizeof(NumericT)*gpu_begin.offset(),
                                     sizeof(NumericT)*gpu_begin.stride() * static_cast<vcl_size_t>(gpu_end - gpu_begin),
                                     &(*cpu_begin));
    }
    else
    {
      vcl_size_t gpu_size = static_cast<vcl_size_t>(gpu_end - gpu_begin);
      std::vector<NumericT> temp_buffer(gpu_begin.stride() * gpu_size);
      viennacl::backend::memory_read(gpu_begin.handle(), sizeof(NumericT)*gpu_begin.offset(), sizeof(NumericT)*temp_buffer.size(), &(temp_buffer[0]));

      for (vcl_size_t i=0; i<gpu_size; ++i)
      {
        (&(*cpu_begin))[i] = temp_buffer[i * gpu_begin.stride()];
      }
    }
  }
}

/** @brief Transfer from a gpu vector to a cpu vector. Convenience wrapper for viennacl::linalg::fast_copy(gpu_vec.begin(), gpu_vec.end(), cpu_vec.begin());
*
* @param gpu_vec    A gpu vector.
* @param cpu_vec    The cpu vector. Type requirements: Output iterator pointing to entries linear in memory can be obtained via member function .begin()
*/
template<typename NumericT, typename CPUVECTOR>
void fast_copy(vector_base<NumericT> const & gpu_vec, CPUVECTOR & cpu_vec )
{
  viennacl::fast_copy(gpu_vec.begin(), gpu_vec.end(), cpu_vec.begin());
}


/** @brief Asynchronous version of fast_copy(), copying data from device to host. The host iterator cpu_begin needs to reside in a linear piece of memory, such as e.g. for std::vector.
*
* This method allows for overlapping data transfer with host computation and returns immediately if the gpu vector has a unit-stride.
* In order to wait for the transfer to complete, use viennacl::backend::finish().
* Note that data pointed to by cpu_begin must not be modified prior to completion of the transfer.
*
* @param gpu_begin  GPU iterator pointing to the beginning of the gpu vector (STL-like)
* @param gpu_end    GPU iterator pointing to the end of the vector (STL-like)
* @param cpu_begin  Output iterator for the cpu vector. The cpu vector must be at least as long as the gpu vector!
*/
template<typename NumericT, unsigned int AlignmentV, typename CPU_ITERATOR>
void async_copy(const const_vector_iterator<NumericT, AlignmentV> & gpu_begin,
                const const_vector_iterator<NumericT, AlignmentV> & gpu_end,
                CPU_ITERATOR cpu_begin )
{
  if (gpu_begin != gpu_end)
  {
    if (gpu_begin.stride() == 1)
    {
      viennacl::backend::memory_read(gpu_begin.handle(),
                                     sizeof(NumericT)*gpu_begin.offset(),
                                     sizeof(NumericT)*gpu_begin.stride() * static_cast<vcl_size_t>(gpu_end - gpu_begin),
                                     &(*cpu_begin),
                                     true);
    }
    else // no async copy possible, so fall-back to fast_copy
      fast_copy(gpu_begin, gpu_end, cpu_begin);
  }
}

/** @brief Transfer from a gpu vector to a cpu vector. Convenience wrapper for viennacl::linalg::fast_copy(gpu_vec.begin(), gpu_vec.end(), cpu_vec.begin());
*
* @param gpu_vec    A gpu vector.
* @param cpu_vec    The cpu vector. Type requirements: Output iterator pointing to entries linear in memory can be obtained via member function .begin()
*/
template<typename NumericT, typename CPUVECTOR>
void async_copy(vector_base<NumericT> const & gpu_vec, CPUVECTOR & cpu_vec )
{
  viennacl::async_copy(gpu_vec.begin(), gpu_vec.end(), cpu_vec.begin());
}


/** @brief STL-like transfer for the entries of a GPU vector to the CPU. The cpu type does not need to lie in a linear piece of memory.
*
* @param gpu_begin  GPU constant iterator pointing to the beginning of the gpu vector (STL-like)
* @param gpu_end    GPU constant iterator pointing to the end of the vector (STL-like)
* @param cpu_begin  Output iterator for the cpu vector. The cpu vector must be at least as long as the gpu vector!
*/
template<typename NumericT, unsigned int AlignmentV, typename CPU_ITERATOR>
void copy(const const_vector_iterator<NumericT, AlignmentV> & gpu_begin,
          const const_vector_iterator<NumericT, AlignmentV> & gpu_end,
          CPU_ITERATOR cpu_begin )
{
  assert(gpu_end - gpu_begin >= 0 && bool("Iterators incompatible"));
  if (gpu_end - gpu_begin != 0)
  {
    std::vector<NumericT> temp_buffer(static_cast<vcl_size_t>(gpu_end - gpu_begin));
    fast_copy(gpu_begin, gpu_end, temp_buffer.begin());

    //now copy entries to cpu_vec:
    std::copy(temp_buffer.begin(), temp_buffer.end(), cpu_begin);
  }
}

/** @brief STL-like transfer for the entries of a GPU vector to the CPU. The cpu type does not need to lie in a linear piece of memory.
*
* @param gpu_begin  GPU iterator pointing to the beginning of the gpu vector (STL-like)
* @param gpu_end    GPU iterator pointing to the end of the vector (STL-like)
* @param cpu_begin  Output iterator for the cpu vector. The cpu vector must be at least as long as the gpu vector!
*/
template<typename NumericT, unsigned int AlignmentV, typename CPU_ITERATOR>
void copy(const vector_iterator<NumericT, AlignmentV> & gpu_begin,
          const vector_iterator<NumericT, AlignmentV> & gpu_end,
          CPU_ITERATOR cpu_begin )

{
  viennacl::copy(const_vector_iterator<NumericT, AlignmentV>(gpu_begin),
                 const_vector_iterator<NumericT, AlignmentV>(gpu_end),
                 cpu_begin);
}

/** @brief Transfer from a gpu vector to a cpu vector. Convenience wrapper for viennacl::linalg::copy(gpu_vec.begin(), gpu_vec.end(), cpu_vec.begin());
*
* @param gpu_vec    A gpu vector
* @param cpu_vec    The cpu vector. Type requirements: Output iterator can be obtained via member function .begin()
*/
template<typename NumericT, typename CPUVECTOR>
void copy(vector_base<NumericT> const & gpu_vec, CPUVECTOR & cpu_vec )
{
  viennacl::copy(gpu_vec.begin(), gpu_vec.end(), cpu_vec.begin());
}



#ifdef VIENNACL_WITH_EIGEN
template<typename NumericT, unsigned int AlignmentV>
void copy(vector<NumericT, AlignmentV> const & gpu_vec,
          Eigen::Matrix<NumericT, Eigen::Dynamic, 1> & eigen_vec)
{
  viennacl::fast_copy(gpu_vec.begin(), gpu_vec.end(), &(eigen_vec[0]));
}

template<typename NumericT, unsigned int AlignmentV, int EigenMapTypeV, typename EigenStrideT>
void copy(vector<NumericT, AlignmentV> const & gpu_vec,
          Eigen::Map<Eigen::Matrix<NumericT, Eigen::Dynamic, 1>, EigenMapTypeV, EigenStrideT> & eigen_vec)
{
  viennacl::fast_copy(gpu_vec.begin(), gpu_vec.end(), &(eigen_vec[0]));
}
#endif


//
//////////////////// Copy from CPU to GPU //////////////////////////////////
//

/** @brief STL-like transfer of a CPU vector to the GPU. The cpu type is assumed to reside in a linear piece of memory, such as e.g. for std::vector.
*
* This method is faster than the plain copy() function, because entries are
* directly read from the cpu vector, starting with &(*cpu.begin()). However,
* keep in mind that the cpu type MUST represent a linear piece of
* memory, otherwise you will run into undefined behavior.
*
* @param cpu_begin  CPU iterator pointing to the beginning of the cpu vector (STL-like)
* @param cpu_end    CPU iterator pointing to the end of the vector (STL-like)
* @param gpu_begin  Output iterator for the gpu vector. The gpu iterator must be incrementable (cpu_end - cpu_begin) times, otherwise the result is undefined.
*/
template<typename CPU_ITERATOR, typename NumericT, unsigned int AlignmentV>
void fast_copy(CPU_ITERATOR const & cpu_begin,
               CPU_ITERATOR const & cpu_end,
               vector_iterator<NumericT, AlignmentV> gpu_begin)
{
  if (cpu_end - cpu_begin > 0)
  {
    if (gpu_begin.stride() == 1)
    {
      viennacl::backend::memory_write(gpu_begin.handle(),
                                      sizeof(NumericT)*gpu_begin.offset(),
                                      sizeof(NumericT)*gpu_begin.stride() * static_cast<vcl_size_t>(cpu_end - cpu_begin), &(*cpu_begin));
    }
    else //writing to slice:
    {
      vcl_size_t cpu_size = static_cast<vcl_size_t>(cpu_end - cpu_begin);
      std::vector<NumericT> temp_buffer(gpu_begin.stride() * cpu_size);

      viennacl::backend::memory_read(gpu_begin.handle(), sizeof(NumericT)*gpu_begin.offset(), sizeof(NumericT)*temp_buffer.size(), &(temp_buffer[0]));

      for (vcl_size_t i=0; i<cpu_size; ++i)
        temp_buffer[i * gpu_begin.stride()] = (&(*cpu_begin))[i];

      viennacl::backend::memory_write(gpu_begin.handle(), sizeof(NumericT)*gpu_begin.offset(), sizeof(NumericT)*temp_buffer.size(), &(temp_buffer[0]));
    }
  }
}


/** @brief Transfer from a cpu vector to a gpu vector. Convenience wrapper for viennacl::linalg::fast_copy(cpu_vec.begin(), cpu_vec.end(), gpu_vec.begin());
*
* @param cpu_vec    A cpu vector. Type requirements: Iterator can be obtained via member function .begin() and .end()
* @param gpu_vec    The gpu vector.
*/
template<typename CPUVECTOR, typename NumericT>
void fast_copy(const CPUVECTOR & cpu_vec, vector_base<NumericT> & gpu_vec)
{
  viennacl::fast_copy(cpu_vec.begin(), cpu_vec.end(), gpu_vec.begin());
}

/** @brief Asynchronous version of fast_copy(), copying data from host to device. The host iterator cpu_begin needs to reside in a linear piece of memory, such as e.g. for std::vector.
*
* This method allows for overlapping data transfer with host computation and returns immediately if the gpu vector has a unit-stride.
* In order to wait for the transfer to complete, use viennacl::backend::finish().
* Note that data pointed to by cpu_begin must not be modified prior to completion of the transfer.
*
* @param cpu_begin  CPU iterator pointing to the beginning of the cpu vector (STL-like)
* @param cpu_end    CPU iterator pointing to the end of the vector (STL-like)
* @param gpu_begin  Output iterator for the gpu vector. The gpu iterator must be incrementable (cpu_end - cpu_begin) times, otherwise the result is undefined.
*/
template<typename CPU_ITERATOR, typename NumericT, unsigned int AlignmentV>
void async_copy(CPU_ITERATOR const & cpu_begin,
                CPU_ITERATOR const & cpu_end,
                vector_iterator<NumericT, AlignmentV> gpu_begin)
{
  if (cpu_end - cpu_begin > 0)
  {
    if (gpu_begin.stride() == 1)
    {
      viennacl::backend::memory_write(gpu_begin.handle(),
                                      sizeof(NumericT)*gpu_begin.offset(),
                                      sizeof(NumericT)*gpu_begin.stride() * static_cast<vcl_size_t>(cpu_end - cpu_begin), &(*cpu_begin),
                                      true);
    }
    else // fallback to blocking copy. There's nothing we can do to prevent this
      fast_copy(cpu_begin, cpu_end, gpu_begin);
  }
}


/** @brief Transfer from a cpu vector to a gpu vector. Convenience wrapper for viennacl::linalg::fast_copy(cpu_vec.begin(), cpu_vec.end(), gpu_vec.begin());
*
* @param cpu_vec    A cpu vector. Type requirements: Iterator can be obtained via member function .begin() and .end()
* @param gpu_vec    The gpu vector.
*/
template<typename CPUVECTOR, typename NumericT>
void async_copy(const CPUVECTOR & cpu_vec, vector_base<NumericT> & gpu_vec)
{
  viennacl::async_copy(cpu_vec.begin(), cpu_vec.end(), gpu_vec.begin());
}

//from cpu to gpu. Safe assumption: cpu_vector does not necessarily occupy a linear memory segment, but is not larger than the allocated memory on the GPU
/** @brief STL-like transfer for the entries of a GPU vector to the CPU. The cpu type does not need to lie in a linear piece of memory.
*
* @param cpu_begin  CPU iterator pointing to the beginning of the gpu vector (STL-like)
* @param cpu_end    CPU iterator pointing to the end of the vector (STL-like)
* @param gpu_begin  Output iterator for the gpu vector. The gpu vector must be at least as long as the cpu vector!
*/
template<typename NumericT, unsigned int AlignmentV, typename CPU_ITERATOR>
void copy(CPU_ITERATOR const & cpu_begin,
          CPU_ITERATOR const & cpu_end,
          vector_iterator<NumericT, AlignmentV> gpu_begin)
{
  assert(cpu_end - cpu_begin > 0 && bool("Iterators incompatible"));
  if (cpu_begin != cpu_end)
  {
    //we require that the size of the gpu_vector is larger or equal to the cpu-size
    std::vector<NumericT> temp_buffer(static_cast<vcl_size_t>(cpu_end - cpu_begin));
    std::copy(cpu_begin, cpu_end, temp_buffer.begin());
    viennacl::fast_copy(temp_buffer.begin(), temp_buffer.end(), gpu_begin);
  }
}

// for things like copy(std_vec.begin(), std_vec.end(), vcl_vec.begin() + 1);

/** @brief Transfer from a host vector object to a ViennaCL vector proxy. Requires the vector proxy to have the necessary size. Convenience wrapper for viennacl::linalg::copy(cpu_vec.begin(), cpu_vec.end(), gpu_vec.begin());
*
* @param cpu_vec    A cpu vector. Type requirements: Iterator can be obtained via member function .begin() and .end()
* @param gpu_vec    The gpu vector.
*/
template<typename HostVectorT, typename T>
void copy(HostVectorT const & cpu_vec, vector_base<T> & gpu_vec)
{
  viennacl::copy(cpu_vec.begin(), cpu_vec.end(), gpu_vec.begin());
}

/** @brief Transfer from a host vector object to a ViennaCL vector. Resizes the ViennaCL vector if it has zero size. Convenience wrapper for viennacl::linalg::copy(cpu_vec.begin(), cpu_vec.end(), gpu_vec.begin());
*
* @param cpu_vec    A host vector. Type requirements: Iterator can be obtained via member function .begin() and .end()
* @param gpu_vec    The gpu (ViennaCL) vector.
*/
template<typename HostVectorT, typename T, unsigned int AlignmentV>
void copy(HostVectorT const & cpu_vec, vector<T, AlignmentV> & gpu_vec)
{
  if (gpu_vec.size() == 0)
    gpu_vec.resize(static_cast<vcl_size_t>(cpu_vec.end() - cpu_vec.begin()));
  viennacl::copy(cpu_vec.begin(), cpu_vec.end(), gpu_vec.begin());
}


#ifdef VIENNACL_WITH_EIGEN
template<typename NumericT, unsigned int AlignmentV>
void copy(Eigen::Matrix<NumericT, Eigen::Dynamic, 1> const & eigen_vec,
          vector<NumericT, AlignmentV> & gpu_vec)
{
  viennacl::fast_copy(eigen_vec.data(), eigen_vec.data() + eigen_vec.size(), gpu_vec.begin());
}

template<typename NumericT, int EigenMapTypeV, typename EigenStrideT, unsigned int AlignmentV>
void copy(Eigen::Map<Eigen::Matrix<NumericT, Eigen::Dynamic, 1>, EigenMapTypeV, EigenStrideT> const & eigen_vec,
          vector<NumericT, AlignmentV> & gpu_vec)
{
  viennacl::fast_copy(eigen_vec.data(), eigen_vec.data() + eigen_vec.size(), gpu_vec.begin());
}
#endif



//
//////////////////// Copy from GPU to GPU //////////////////////////////////
//
/** @brief Copy (parts of a) GPU vector to another GPU vector
*
* @param gpu_src_begin    GPU iterator pointing to the beginning of the gpu vector (STL-like)
* @param gpu_src_end      GPU iterator pointing to the end of the vector (STL-like)
* @param gpu_dest_begin   Output iterator for the gpu vector. The gpu_dest vector must be at least as long as the gpu_src vector!
*/
template<typename NumericT, unsigned int AlignmentV_SRC, unsigned int AlignmentV_DEST>
void copy(const_vector_iterator<NumericT, AlignmentV_SRC> const & gpu_src_begin,
          const_vector_iterator<NumericT, AlignmentV_SRC> const & gpu_src_end,
          vector_iterator<NumericT, AlignmentV_DEST> gpu_dest_begin)
{
  assert(gpu_src_end - gpu_src_begin >= 0);
  assert(gpu_src_begin.stride() == 1 && bool("ViennaCL ERROR: copy() for GPU->GPU not implemented for slices! Use operator= instead for the moment."));

  if (gpu_src_begin.stride() == 1 && gpu_dest_begin.stride() == 1)
  {
    if (gpu_src_begin != gpu_src_end)
      viennacl::backend::memory_copy(gpu_src_begin.handle(), gpu_dest_begin.handle(),
                                     sizeof(NumericT) * gpu_src_begin.offset(),
                                     sizeof(NumericT) * gpu_dest_begin.offset(),
                                     sizeof(NumericT) * (gpu_src_end.offset() - gpu_src_begin.offset()));
  }
  else
  {
    assert( false && bool("not implemented yet"));
  }
}

/** @brief Copy (parts of a) GPU vector to another GPU vector
*
* @param gpu_src_begin   GPU iterator pointing to the beginning of the gpu vector (STL-like)
* @param gpu_src_end     GPU iterator pointing to the end of the vector (STL-like)
* @param gpu_dest_begin  Output iterator for the gpu vector. The gpu vector must be at least as long as the cpu vector!
*/
template<typename NumericT, unsigned int AlignmentV_SRC, unsigned int AlignmentV_DEST>
void copy(vector_iterator<NumericT, AlignmentV_SRC> const & gpu_src_begin,
          vector_iterator<NumericT, AlignmentV_SRC> const & gpu_src_end,
          vector_iterator<NumericT, AlignmentV_DEST> gpu_dest_begin)
{
  viennacl::copy(static_cast<const_vector_iterator<NumericT, AlignmentV_SRC> >(gpu_src_begin),
                 static_cast<const_vector_iterator<NumericT, AlignmentV_SRC> >(gpu_src_end),
                 gpu_dest_begin);
}

/** @brief Transfer from a ViennaCL vector to another ViennaCL vector. Convenience wrapper for viennacl::linalg::copy(gpu_src_vec.begin(), gpu_src_vec.end(), gpu_dest_vec.begin());
*
* @param gpu_src_vec    A gpu vector
* @param gpu_dest_vec    The cpu vector. Type requirements: Output iterator can be obtained via member function .begin()
*/
template<typename NumericT, unsigned int AlignmentV_SRC, unsigned int AlignmentV_DEST>
void copy(vector<NumericT, AlignmentV_SRC> const & gpu_src_vec,
          vector<NumericT, AlignmentV_DEST> & gpu_dest_vec )
{
  viennacl::copy(gpu_src_vec.begin(), gpu_src_vec.end(), gpu_dest_vec.begin());
}






//global functions for handling vectors:
/** @brief Output stream. Output format is ublas compatible.
* @param os   STL output stream
* @param val  The vector that should be printed
*/
template<typename T>
std::ostream & operator<<(std::ostream & os, vector_base<T> const & val)
{
  std::vector<T> tmp(val.size());
  viennacl::copy(val.begin(), val.end(), tmp.begin());
  os << "[" << val.size() << "](";
  for (typename std::vector<T>::size_type i=0; i<val.size(); ++i)
  {
    if (i > 0)
      os << ",";
    os << tmp[i];
  }
  os << ")";
  return os;
}

template<typename LHS, typename RHS, typename OP>
std::ostream & operator<<(std::ostream & os, vector_expression<LHS, RHS, OP> const & proxy)

{
  typedef typename viennacl::result_of::cpu_value_type<typename LHS::value_type>::type ScalarType;
  viennacl::vector<ScalarType> result = proxy;
  os << result;
  return os;
}

/** @brief Swaps the contents of two vectors, data is copied
*
* @param vec1   The first vector
* @param vec2   The second vector
*/
template<typename T>
void swap(vector_base<T> & vec1, vector_base<T> & vec2)
{
  viennacl::linalg::vector_swap(vec1, vec2);
}

/** @brief Swaps the content of two vectors by swapping OpenCL handles only, NO data is copied
*
* @param v1   The first vector
* @param v2   The second vector
*/
template<typename NumericT, unsigned int AlignmentV>
vector<NumericT, AlignmentV> & fast_swap(vector<NumericT, AlignmentV> & v1,
                                         vector<NumericT, AlignmentV> & v2)
{
  return v1.fast_swap(v2);
}





//
//
////////// operations /////////////////////////////////////////////////////////////////////////////////
//
//


//
// operator *=
//

/** @brief Scales this vector by a GPU scalar value
*/
template<typename T, typename S1>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
vector_base<T> &
>::type
operator *= (vector_base<T> & v1, S1 const & gpu_val)
{
  bool flip_sign = viennacl::is_flip_sign_scalar<S1>::value;
  if (v1.size() > 0)
    viennacl::linalg::av(v1,
                         v1, gpu_val, 1, false, flip_sign);
  return v1;
}


//
// operator /=
//


/** @brief Scales this vector by a GPU scalar value
*/
template<typename T, typename S1>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
vector_base<T> &
>::type
operator /= (vector_base<T> & v1, S1 const & gpu_val)
{
  bool flip_sign = viennacl::is_flip_sign_scalar<S1>::value;
  if (v1.size() > 0)
    viennacl::linalg::av(v1,
                         v1, gpu_val, 1, true, flip_sign);
  return v1;
}


//
// operator +
//


/** @brief Operator overload for the addition of two vector expressions.
*
* @param proxy1  Left hand side vector expression
* @param proxy2  Right hand side vector expression
*/
template<typename LHS1, typename RHS1, typename OP1,
         typename LHS2, typename RHS2, typename OP2>
vector_expression< const vector_expression< LHS1, RHS1, OP1>,
const vector_expression< LHS2, RHS2, OP2>,
viennacl::op_add>
operator + (vector_expression<LHS1, RHS1, OP1> const & proxy1,
            vector_expression<LHS2, RHS2, OP2> const & proxy2)
{
  assert(proxy1.size() == proxy2.size() && bool("Incompatible vector sizes!"));
  return   vector_expression< const vector_expression<LHS1, RHS1, OP1>,
      const vector_expression<LHS2, RHS2, OP2>,
      viennacl::op_add>(proxy1, proxy2);
}

/** @brief Operator overload for the addition of a vector expression with a vector or another vector expression. This is the default implementation for all cases that are too complex in order to be covered within a single kernel, hence a temporary vector is created.
*
* @param proxy   Left hand side vector expression
* @param vec     Right hand side vector (also -range and -slice is allowed)
*/
template<typename LHS, typename RHS, typename OP, typename T>
vector_expression< const vector_expression<LHS, RHS, OP>,
const vector_base<T>,
viennacl::op_add>
operator + (vector_expression<LHS, RHS, OP> const & proxy,
            vector_base<T> const & vec)
{
  assert(proxy.size() == vec.size() && bool("Incompatible vector sizes!"));
  return vector_expression< const vector_expression<LHS, RHS, OP>,
      const vector_base<T>,
      viennacl::op_add>(proxy, vec);
}

/** @brief Operator overload for the addition of a vector with a vector expression. This is the default implementation for all cases that are too complex in order to be covered within a single kernel, hence a temporary vector is created.
*
* @param proxy   Left hand side vector expression
* @param vec     Right hand side vector (also -range and -slice is allowed)
*/
template<typename T, typename LHS, typename RHS, typename OP>
vector_expression< const vector_base<T>,
const vector_expression<LHS, RHS, OP>,
viennacl::op_add>
operator + (vector_base<T> const & vec,
            vector_expression<LHS, RHS, OP> const & proxy)
{
  assert(proxy.size() == vec.size() && bool("Incompatible vector sizes!"));
  return vector_expression< const vector_base<T>,
      const vector_expression<LHS, RHS, OP>,
      viennacl::op_add>(vec, proxy);
}

/** @brief Returns an expression template object for adding up two vectors, i.e. v1 + v2
*/
template<typename T>
vector_expression< const vector_base<T>, const vector_base<T>, op_add>
operator + (const vector_base<T> & v1, const vector_base<T> & v2)
{
  return vector_expression< const vector_base<T>, const vector_base<T>, op_add>(v1, v2);
}



//
// operator -
//

/** @brief Operator overload for the subtraction of two vector expressions.
*
* @param proxy1  Left hand side vector expression
* @param proxy2  Right hand side vector expression
*/
template<typename LHS1, typename RHS1, typename OP1,
         typename LHS2, typename RHS2, typename OP2>
vector_expression< const vector_expression< LHS1, RHS1, OP1>,
const vector_expression< LHS2, RHS2, OP2>,
viennacl::op_sub>
operator - (vector_expression<LHS1, RHS1, OP1> const & proxy1,
            vector_expression<LHS2, RHS2, OP2> const & proxy2)
{
  assert(proxy1.size() == proxy2.size() && bool("Incompatible vector sizes!"));
  return   vector_expression< const vector_expression<LHS1, RHS1, OP1>,
      const vector_expression<LHS2, RHS2, OP2>,
      viennacl::op_sub>(proxy1, proxy2);
}


/** @brief Operator overload for the subtraction of a vector expression with a vector or another vector expression. This is the default implementation for all cases that are too complex in order to be covered within a single kernel, hence a temporary vector is created.
*
* @param proxy   Left hand side vector expression
* @param vec     Right hand side vector (also -range and -slice is allowed)
*/
template<typename LHS, typename RHS, typename OP, typename T>
vector_expression< const vector_expression<LHS, RHS, OP>,
const vector_base<T>,
viennacl::op_sub>
operator - (vector_expression<LHS, RHS, OP> const & proxy,
            vector_base<T> const & vec)
{
  assert(proxy.size() == vec.size() && bool("Incompatible vector sizes!"));
  return vector_expression< const vector_expression<LHS, RHS, OP>,
      const vector_base<T>,
      viennacl::op_sub>(proxy, vec);
}

/** @brief Operator overload for the subtraction of a vector expression with a vector or another vector expression. This is the default implementation for all cases that are too complex in order to be covered within a single kernel, hence a temporary vector is created.
*
* @param proxy   Left hand side vector expression
* @param vec     Right hand side vector (also -range and -slice is allowed)
*/
template<typename T, typename LHS, typename RHS, typename OP>
vector_expression< const vector_base<T>,
const vector_expression<LHS, RHS, OP>,
viennacl::op_sub>
operator - (vector_base<T> const & vec,
            vector_expression<LHS, RHS, OP> const & proxy)
{
  assert(proxy.size() == vec.size() && bool("Incompatible vector sizes!"));
  return vector_expression< const vector_base<T>,
      const vector_expression<LHS, RHS, OP>,
      viennacl::op_sub>(vec, proxy);
}

/** @brief Returns an expression template object for subtracting two vectors, i.e. v1 - v2
*/
template<typename T>
vector_expression< const vector_base<T>, const vector_base<T>, op_sub>
operator - (const vector_base<T> & v1, const vector_base<T> & v2)
{
  return vector_expression< const vector_base<T>, const vector_base<T>, op_sub>(v1, v2);
}


//
// operator *
//


/** @brief Operator overload for the expression alpha * v1, where alpha is a host scalar (float or double) and v1 is a ViennaCL vector.
*
* @param value   The host scalar (float or double)
* @param vec     A ViennaCL vector
*/
template<typename S1, typename T>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
vector_expression< const vector_base<T>, const S1, op_mult> >::type
operator * (S1 const & value, vector_base<T> const & vec)
{
  return vector_expression< const vector_base<T>, const S1, op_mult>(vec, value);
}

/** @brief Operator overload for the expression alpha * v1, where alpha is a char
*
* @param value   The host scalar (float or double)
* @param vec     A ViennaCL vector
*/
template<typename T>
vector_expression< const vector_base<T>, const T, op_mult>
operator * (char value, vector_base<T> const & vec)
{
  return vector_expression< const vector_base<T>, const T, op_mult>(vec, T(value));
}

/** @brief Operator overload for the expression alpha * v1, where alpha is a short
*
* @param value   The host scalar (float or double)
* @param vec     A ViennaCL vector
*/
template<typename T>
vector_expression< const vector_base<T>, const T, op_mult>
operator * (short value, vector_base<T> const & vec)
{
  return vector_expression< const vector_base<T>, const T, op_mult>(vec, T(value));
}

/** @brief Operator overload for the expression alpha * v1, where alpha is a int
*
* @param value   The host scalar (float or double)
* @param vec     A ViennaCL vector
*/
template<typename T>
vector_expression< const vector_base<T>, const T, op_mult>
operator * (int value, vector_base<T> const & vec)
{
  return vector_expression< const vector_base<T>, const T, op_mult>(vec, T(value));
}

/** @brief Operator overload for the expression alpha * v1, where alpha is a long
*
* @param value   The host scalar (float or double)
* @param vec     A ViennaCL vector
*/
template<typename T>
vector_expression< const vector_base<T>, const T, op_mult>
operator * (long value, vector_base<T> const & vec)
{
  return vector_expression< const vector_base<T>, const T, op_mult>(vec, T(value));
}

/** @brief Operator overload for the expression alpha * v1, where alpha is a float
*
* @param value   The host scalar (float or double)
* @param vec     A ViennaCL vector
*/
template<typename T>
vector_expression< const vector_base<T>, const T, op_mult>
operator * (float value, vector_base<T> const & vec)
{
  return vector_expression< const vector_base<T>, const T, op_mult>(vec, T(value));
}

/** @brief Operator overload for the expression alpha * v1, where alpha is a double
*
* @param value   The host scalar (float or double)
* @param vec     A ViennaCL vector
*/
template<typename T>
vector_expression< const vector_base<T>, const T, op_mult>
operator * (double value, vector_base<T> const & vec)
{
  return vector_expression< const vector_base<T>, const T, op_mult>(vec, T(value));
}



/** @brief Operator overload for the expression alpha * v1, where alpha is a scalar expression and v1 is a ViennaCL vector.
*
* @param expr    The scalar expression
* @param vec     A ViennaCL vector
*/
template<typename LHS, typename RHS, typename OP, typename T>
vector_expression< const vector_base<T>, const scalar_expression<LHS, RHS, OP>, op_mult>
operator * (scalar_expression<LHS, RHS, OP> const & expr, vector_base<T> const & vec)
{
  return vector_expression< const vector_base<T>, const scalar_expression<LHS, RHS, OP>, op_mult>(vec, expr);
}

/** @brief Scales the vector by a scalar 'alpha' and returns an expression template
*/
template<typename T, typename S1>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
vector_expression< const vector_base<T>, const S1, op_mult> >::type
operator * (vector_base<T> const & vec, S1 const & value)
{
  return vector_expression< const vector_base<T>, const S1, op_mult>(vec, value);
}

template<typename T>
vector_expression< const vector_base<T>, const T, op_mult>
operator * (vector_base<T> const & vec, T const & value)
{
  return vector_expression< const vector_base<T>, const T, op_mult>(vec, value);
}

/** @brief Operator overload for the multiplication of a vector expression with a scalar from the right, e.g. (beta * vec1) * alpha. Here, beta * vec1 is wrapped into a vector_expression and then multiplied with alpha from the right.
*
* @param proxy   Left hand side vector expression
* @param val     Right hand side scalar
*/
template<typename LHS, typename RHS, typename OP, typename S1>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
viennacl::vector_expression<const vector_expression<LHS, RHS, OP>, const S1, op_mult>  >::type
operator * (vector_expression< LHS, RHS, OP> const & proxy,
            S1 const & val)
{
  return viennacl::vector_expression<const vector_expression<LHS, RHS, OP>, const S1, op_mult>(proxy, val);
}

/** @brief Operator overload for the multiplication of a vector expression with a ViennaCL scalar from the left, e.g. alpha * (beta * vec1). Here, beta * vec1 is wrapped into a vector_expression and then multiplied with alpha from the left.
*
* @param val     Right hand side scalar
* @param proxy   Left hand side vector expression
*/
template<typename S1, typename LHS, typename RHS, typename OP>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
viennacl::vector_expression<const vector_expression<LHS, RHS, OP>, const S1, op_mult>  >::type
operator * (S1 const & val,
            vector_expression<LHS, RHS, OP> const & proxy)
{
  return viennacl::vector_expression<const vector_expression<LHS, RHS, OP>, const S1, op_mult>(proxy, val);
}

//
// operator /
//

/** @brief Operator overload for the division of a vector expression by a scalar from the right, e.g. (beta * vec1) / alpha. Here, beta * vec1 is wrapped into a vector_expression and then divided by alpha.
*
* @param proxy   Left hand side vector expression
* @param val     Right hand side scalar
*/
template<typename S1, typename LHS, typename RHS, typename OP>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
viennacl::vector_expression<const vector_expression<LHS, RHS, OP>, const S1, op_div>  >::type
operator / (vector_expression< LHS, RHS, OP> const & proxy,
            S1 const & val)
{
  return viennacl::vector_expression<const vector_expression<LHS, RHS, OP>, const S1, op_div>(proxy, val);
}


/** @brief Returns an expression template for scaling the vector by a GPU scalar 'alpha'
*/
template<typename T, typename S1>
typename viennacl::enable_if< viennacl::is_any_scalar<S1>::value,
vector_expression< const vector_base<T>, const S1, op_div> >::type
operator / (vector_base<T> const & v1, S1 const & s1)
{
  return vector_expression<const vector_base<T>, const S1, op_div>(v1, s1);
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
  struct op_executor<vector_base<T>, op_assign, vector_base<T> >
  {
    static void apply(vector_base<T> & lhs, vector_base<T> const & rhs)
    {
      viennacl::linalg::av(lhs, rhs, T(1), 1, false, false);
    }
  };

  // x = inner_prod(z, {y0, y1, ...})
  template<typename T>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const vector_base<T>, const vector_tuple<T>, op_inner_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_tuple<T>, op_inner_prod> const & rhs)
    {
      viennacl::linalg::inner_prod_impl(rhs.lhs(), rhs.rhs(), lhs);
    }
  };

  // x += y
  template<typename T>
  struct op_executor<vector_base<T>, op_inplace_add, vector_base<T> >
  {
    static void apply(vector_base<T> & lhs, vector_base<T> const & rhs)
    {
      viennacl::linalg::avbv(lhs, lhs, T(1), 1, false, false, rhs, T(1), 1, false, false);
    }
  };

  // x -= y
  template<typename T>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_base<T> >
  {
    static void apply(vector_base<T> & lhs, vector_base<T> const & rhs)
    {
      viennacl::linalg::avbv(lhs, lhs, T(1), 1, false, false, rhs, T(1), 1, false, true);
    }
  };

  ///////////// x  OP  y * alpha ////////////////////////


  // x = alpha * y
  template<typename T, typename ScalarType>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const vector_base<T>, const ScalarType, op_mult> >
  {
    // generic case: ScalarType is a scalar expression
    template<typename LHS, typename RHS, typename OP>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const scalar_expression<LHS, RHS, OP>, op_mult> const & proxy)
    {
      T alpha = proxy.rhs();
      viennacl::linalg::av(lhs, proxy.lhs(), alpha, 1, false, false);
    }

    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const scalar<T>, op_mult> const & proxy)
    {
      viennacl::linalg::av(lhs, proxy.lhs(), proxy.rhs(), 1, false, false);
    }

    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const T, op_mult> const & proxy)
    {
      viennacl::linalg::av(lhs, proxy.lhs(), proxy.rhs(), 1, false, false);
    }
  };

  // x += alpha * y
  template<typename T, typename ScalarType>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const vector_base<T>, const ScalarType, op_mult> >
  {
    // generic case: ScalarType is a scalar expression
    template<typename LHS, typename RHS, typename OP>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const scalar_expression<LHS, RHS, OP>, op_mult> const & proxy)
    {
      T alpha = proxy.rhs();
      viennacl::linalg::avbv(lhs, lhs, T(1), 1, false, false, proxy.lhs(), alpha, 1, false, false);
    }

    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const scalar<T>, op_mult> const & proxy)
    {
      viennacl::linalg::avbv(lhs, lhs, T(1), 1, false, false, proxy.lhs(), proxy.rhs(), 1, false, false);
    }

    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const T, op_mult> const & proxy)
    {
      viennacl::linalg::avbv(lhs, lhs, T(1), 1, false, false, proxy.lhs(), proxy.rhs(), 1, false, false);
    }
  };

  // x -= alpha * y
  template<typename T, typename ScalarType>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const vector_base<T>, const ScalarType, op_mult> >
  {
    // generic case: ScalarType is a scalar expression
    template<typename LHS, typename RHS, typename OP>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const scalar_expression<LHS, RHS, OP>, op_mult> const & proxy)
    {
      T alpha = proxy.rhs();
      viennacl::linalg::avbv(lhs, lhs, T(1), 1, false, false, proxy.lhs(), alpha, 1, false, true);
    }

    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const scalar<T>, op_mult> const & proxy)
    {
      viennacl::linalg::avbv(lhs, lhs, T(1), 1, false, false, proxy.lhs(), proxy.rhs(), 1, false, true);
    }

    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const T, op_mult> const & proxy)
    {
      viennacl::linalg::avbv(lhs, lhs, T(1), 1, false, false, proxy.lhs(), proxy.rhs(), 1, false, true);
    }
  };


  ///////////// x  OP  vec_expr * alpha ////////////////////////

  // x = alpha * vec_expr
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> const & proxy)
    {
      vector<T> temp(proxy.lhs());
      lhs = temp * proxy.rhs();
    }
  };

  // x += alpha * vec_expr
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> const & proxy)
    {
      vector<T> temp(proxy.lhs());
      lhs += temp * proxy.rhs();
    }
  };

  // x -= alpha * vec_expr
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_mult> const & proxy)
    {
      vector<T> temp(proxy.lhs());
      lhs -= temp * proxy.rhs();
    }
  };


  ///////////// x  OP  y / alpha ////////////////////////

  // x = y / alpha
  template<typename T, typename ScalarType>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const vector_base<T>, const ScalarType, op_div> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const ScalarType, op_div> const & proxy)
    {
      viennacl::linalg::av(lhs, proxy.lhs(), proxy.rhs(), 1, true, false);
    }
  };

  // x += y / alpha
  template<typename T, typename ScalarType>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const vector_base<T>, const ScalarType, op_div> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const ScalarType, op_div> const & proxy)
    {
      viennacl::linalg::avbv(lhs, lhs, T(1), 1, false, false, proxy.lhs(), proxy.rhs(), 1, true, false);
    }
  };

  // x -= y / alpha
  template<typename T, typename ScalarType>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const vector_base<T>, const ScalarType, op_div> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const ScalarType, op_div> const & proxy)
    {
      viennacl::linalg::avbv(lhs, lhs, T(1), 1, false, false, proxy.lhs(), proxy.rhs(), 1, true, true);
    }
  };


  ///////////// x  OP  vec_expr / alpha ////////////////////////

  // x = vec_expr / alpha
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_div> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_div> const & proxy)
    {
      vector<T> temp(proxy.lhs());
      lhs = temp / proxy.rhs();
    }
  };

  // x += vec_expr / alpha
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_div> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_div> const & proxy)
    {
      vector<T> temp(proxy.lhs());
      lhs += temp / proxy.rhs();
    }
  };

  // x -= vec_expr / alpha
  template<typename T, typename LHS, typename RHS, typename OP, typename ScalarType>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_div> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS, const RHS, OP>, const ScalarType, op_div> const & proxy)
    {
      vector<T> temp(proxy.lhs());
      lhs -= temp / proxy.rhs();
    }
  };



  // generic x = vec_expr1 + vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const LHS, const RHS, op_add> >
  {
    // generic x = vec_expr1 + vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(vector_base<T> & lhs, vector_expression<const LHS1, const RHS1, op_add> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        vector_base<T> temp(proxy.lhs());
        op_executor<vector_base<T>, op_inplace_add, RHS>::apply(temp, proxy.rhs());
        lhs = temp;
      }
      else
      {
        op_executor<vector_base<T>, op_assign, LHS>::apply(lhs, proxy.lhs());
        op_executor<vector_base<T>, op_inplace_add, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x = y + z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_add> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs(), T(1), 1, false, false);
    }

    // x = alpha * y + z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const T, op_mult>,
                      const vector_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs(), T(1), 1, false, false);
    }

    // x = y / alpha + z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const T, op_div>,
                      const vector_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs(), T(1), 1, false, false);
    }

    // x = y + beta * z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const T, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x = y + z / beta
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const T, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x = alpha * y + beta * z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const T, op_mult>,
                      const vector_expression<const vector_base<T>, const T, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x = alpha * y + z / beta
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const T, op_mult>,
                      const vector_expression<const vector_base<T>, const T, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x = y / alpha + beta * z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const T, op_div>,
                      const vector_expression<const vector_base<T>, const T, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x = y / alpha + z / beta
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const T, op_div>,
                      const vector_expression<const vector_base<T>, const T, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }
  };


  // generic x += vec_expr1 + vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const LHS, const RHS, op_add> >
  {
    // generic x += vec_expr1 + vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(vector_base<T> & lhs, vector_expression<const LHS1, const RHS1, op_add> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        vector_base<T> temp(proxy.lhs());
        op_executor<vector_base<T>, op_inplace_add, RHS>::apply(temp, proxy.rhs());
        lhs += temp;
      }
      else
      {
        op_executor<vector_base<T>, op_inplace_add, LHS>::apply(lhs, proxy.lhs());
        op_executor<vector_base<T>, op_inplace_add, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x += y + z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x += alpha * y + z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType, op_mult>,
                      const vector_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x += y / alpha + z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType, op_div>,
                      const vector_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x += y + beta * z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const ScalarType, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x += y + z / beta
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const ScalarType, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x += alpha * y + beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_mult>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x += alpha * y + z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_mult>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x += y / alpha + beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_div>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x += y / alpha + z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_div>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }
  };



  // generic x -= vec_expr1 + vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const LHS, const RHS, op_add> >
  {
    // generic x -= vec_expr1 + vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(vector_base<T> & lhs, vector_expression<const LHS1, const RHS1, op_add> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        vector_base<T> temp(proxy.lhs());
        op_executor<vector_base<T>, op_inplace_add, RHS>::apply(temp, proxy.rhs());
        lhs -= temp;
      }
      else
      {
        op_executor<vector_base<T>, op_inplace_sub, LHS>::apply(lhs, proxy.lhs());
        op_executor<vector_base<T>, op_inplace_sub, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x -= y + z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x -= alpha * y + z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType, op_mult>,
                      const vector_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x -= y / alpha + z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType, op_div>,
                      const vector_base<T>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x -= y + beta * z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const ScalarType, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x -= y + z / beta
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const ScalarType, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x -= alpha * y + beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_mult>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x -= alpha * y + z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_mult>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x -= y / alpha + beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_div>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_mult>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x -= y / alpha + z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_div>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_div>,
                      op_add> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }
  };



  ///////////////////////



  // generic x = vec_expr1 - vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const LHS, const RHS, op_sub> >
  {
    // generic x = vec_expr1 - vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(vector_base<T> & lhs, vector_expression<const LHS1, const RHS1, op_sub> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        vector_base<T> temp(proxy.lhs());
        op_executor<vector_base<T>, op_inplace_sub, RHS>::apply(temp, proxy.rhs());
        lhs = temp;
      }
      else
      {
        op_executor<vector_base<T>, op_assign, LHS>::apply(lhs, proxy.lhs());
        op_executor<vector_base<T>, op_inplace_sub, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x = y - z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_sub> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs(), T(1), 1, false, true);
    }

    // x = alpha * y - z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType, op_mult>,
                      const vector_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs(), T(1), 1, false, true);
    }

    // x = y / alpha - z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType, op_div>,
                      const vector_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs(), T(1), 1, false, true);
    }

    // x = y - beta * z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const ScalarType, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x = y - z / beta
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const ScalarType, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs(), T(1), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x = alpha * y - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_mult>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x = alpha * y - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_mult>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x = y / alpha - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_div>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x = y / alpha - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_div>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv(lhs,
                             proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                             proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }
  };


  // generic x += vec_expr1 - vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const LHS, const RHS, op_sub> >
  {
    // generic x += vec_expr1 - vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(vector_base<T> & lhs, vector_expression<const LHS1, const RHS1, op_sub> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        vector_base<T> temp(proxy.lhs());
        op_executor<vector_base<T>, op_inplace_sub, RHS>::apply(temp, proxy.rhs());
        lhs += temp;
      }
      else
      {
        op_executor<vector_base<T>, op_inplace_add, LHS>::apply(lhs, proxy.lhs());
        op_executor<vector_base<T>, op_inplace_sub, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x += y - z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x += alpha * y - z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType, op_mult>,
                      const vector_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x += y / alpha - z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType, op_div>,
                      const vector_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs(), T(1), 1, false, true);
    }

    // x += y - beta * z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const ScalarType, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x += y - z / beta
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const ScalarType, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x += alpha * y - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_mult>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x += alpha * y - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_mult>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }

    // x += y / alpha - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_div>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, true);
    }

    // x += y / alpha - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_div>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, false,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, true);
    }
  };



  // generic x -= vec_expr1 - vec_expr2:
  template<typename T, typename LHS, typename RHS>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const LHS, const RHS, op_sub> >
  {
    // generic x -= vec_expr1 - vec_expr2:
    template<typename LHS1, typename RHS1>
    static void apply(vector_base<T> & lhs, vector_expression<const LHS1, const RHS1, op_sub> const & proxy)
    {
      bool op_aliasing_lhs = op_aliasing(lhs, proxy.lhs());
      bool op_aliasing_rhs = op_aliasing(lhs, proxy.rhs());

      if (op_aliasing_lhs || op_aliasing_rhs)
      {
        vector_base<T> temp(proxy.lhs());
        op_executor<vector_base<T>, op_inplace_sub, RHS>::apply(temp, proxy.rhs());
        lhs -= temp;
      }
      else
      {
        op_executor<vector_base<T>, op_inplace_sub, LHS>::apply(lhs, proxy.lhs());
        op_executor<vector_base<T>, op_inplace_add, RHS>::apply(lhs, proxy.rhs());
      }
    }

    // x -= y - z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x -= alpha * y - z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType, op_mult>,
                      const vector_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x -= y / alpha - z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType, op_div>,
                      const vector_base<T>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs(), T(1), 1, false, false);
    }

    // x -= y - beta * z
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const ScalarType, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x -= y - z / beta
    template<typename ScalarType>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>,
                      const vector_expression<const vector_base<T>, const ScalarType, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs(), T(1), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x -= alpha * y - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_mult>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x -= alpha * y - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_mult>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, false, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }

    // x -= y / alpha - beta * z
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_div>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_mult>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, false, false);
    }

    // x -= y / alpha - z / beta
    template<typename ScalarType1, typename ScalarType2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const vector_base<T>, const ScalarType1, op_div>,
                      const vector_expression<const vector_base<T>, const ScalarType2, op_div>,
                      op_sub> const & proxy)
    {
      viennacl::linalg::avbv_v(lhs,
                               proxy.lhs().lhs(), proxy.lhs().rhs(), 1, true, true,
                               proxy.rhs().lhs(), proxy.rhs().rhs(), 1, true, false);
    }
  };


















  //////////////////// Element-wise operations ////////////////////////////////////////

  // generic x = vec_expr1 .* vec_expr2:
  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const LHS, const RHS, op_element_binary<OP> > >
  {
    // x = y .* z  or  x = y ./ z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> > const & proxy)
    {
      viennacl::linalg::element_op(lhs, proxy);
    }

    // x = y .* vec_expr  or  x = y ./ vec_expr
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_expression<const LHS2, const RHS2, OP2>, op_element_binary<OP> > const & proxy)
    {
      vector<T> temp(proxy.rhs());
      viennacl::linalg::element_op(lhs, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> >(proxy.lhs(), temp));
    }

    // x = vec_expr .* z  or  x = vec_expr ./ z
    template<typename LHS1, typename RHS1, typename OP1>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS1, const RHS1, OP1>, const vector_base<T>, op_element_binary<OP> > const & proxy)
    {
      vector<T> temp(proxy.lhs());
      viennacl::linalg::element_op(lhs, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> >(temp, proxy.rhs()));
    }

    // x = vec_expr .* vec_expr  or  z = vec_expr .* vec_expr
    template<typename LHS1, typename RHS1, typename OP1,
             typename LHS2, typename RHS2, typename OP2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS1, const RHS1, OP1>,
                      const vector_expression<const LHS2, const RHS2, OP2>,
                      op_element_binary<OP> > const & proxy)
    {
      vector<T> temp1(proxy.lhs());
      vector<T> temp2(proxy.rhs());
      viennacl::linalg::element_op(lhs, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> >(temp1, temp2));
    }
  };

  // generic x += vec_expr1 .* vec_expr2:
  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const LHS, const RHS, op_element_binary<OP> > >
  {
    // x += y .* z  or  x += y ./ z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> > const & proxy)
    {
      viennacl::vector<T> temp(proxy);
      lhs += temp;
    }

    // x += y .* vec_expr  or  x += y ./ vec_expr
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_expression<const LHS2, const RHS2, OP2>,  op_element_binary<OP> > const & proxy)
    {
      vector<T> temp(proxy.rhs());
      vector<T> temp2(temp.size());
      viennacl::linalg::element_op(temp2, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> >(proxy.lhs(), temp));
      lhs += temp2;
    }

    // x += vec_expr .* z  or  x += vec_expr ./ z
    template<typename LHS1, typename RHS1, typename OP1>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS1, const RHS1, OP1>, const vector_base<T>, op_element_binary<OP> > const & proxy)
    {
      vector<T> temp(proxy.lhs());
      vector<T> temp2(temp.size());
      viennacl::linalg::element_op(temp2, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> >(temp, proxy.rhs()));
      lhs += temp2;
    }

    // x += vec_expr .* vec_expr  or  x += vec_expr ./ vec_expr
    template<typename LHS1, typename RHS1, typename OP1,
             typename LHS2, typename RHS2, typename OP2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS1, const RHS1, OP1>,
                      const vector_expression<const LHS2, const RHS2, OP2>,
                      op_element_binary<OP> > const & proxy)
    {
      vector<T> temp1(proxy.lhs());
      vector<T> temp2(proxy.rhs());
      vector<T> temp3(temp1.size());
      viennacl::linalg::element_op(temp3, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> >(temp1, temp2));
      lhs += temp3;
    }
  };

  // generic x -= vec_expr1 .* vec_expr2:
  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const LHS, const RHS, op_element_binary<OP> > >
  {

    // x -= y .* z  or  x -= y ./ z
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> > const & proxy)
    {
      viennacl::vector<T> temp(proxy);
      lhs -= temp;
    }

    // x -= y .* vec_expr  or  x -= y ./ vec_expr
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_expression<const LHS2, const RHS2, OP2>, op_element_binary<OP> > const & proxy)
    {
      vector<T> temp(proxy.rhs());
      vector<T> temp2(temp.size());
      viennacl::linalg::element_op(temp2, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> >(proxy.lhs(), temp));
      lhs -= temp2;
    }

    // x -= vec_expr .* z  or  x -= vec_expr ./ z
    template<typename LHS1, typename RHS1, typename OP1>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS1, const RHS1, OP1>, const vector_base<T>, op_element_binary<OP> > const & proxy)
    {
      vector<T> temp(proxy.lhs());
      vector<T> temp2(temp.size());
      viennacl::linalg::element_op(temp2, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> >(temp, proxy.rhs()));
      lhs -= temp2;
    }

    // x -= vec_expr .* vec_expr  or  x -= vec_expr ./ vec_expr
    template<typename LHS1, typename RHS1, typename OP1,
             typename LHS2, typename RHS2, typename OP2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS1, const RHS1, OP1>,
                      const vector_expression<const LHS2, const RHS2, OP2>,
                      op_element_binary<OP> > const & proxy)
    {
      vector<T> temp1(proxy.lhs());
      vector<T> temp2(proxy.rhs());
      vector<T> temp3(temp1.size());
      viennacl::linalg::element_op(temp3, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<OP> >(temp1, temp2));
      lhs -= temp3;
    }
  };

  //////////////// unary expressions

  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const LHS, const RHS, op_element_unary<OP> > >
  {
    // x = OP(y)
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<OP> > const & proxy)
    {
      viennacl::linalg::element_op(lhs, proxy);
    }

    // x = OP(vec_expr)
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS2, const RHS2, OP2>,
                      const vector_expression<const LHS2, const RHS2, OP2>,
                      op_element_unary<OP> > const & proxy)
    {
      vector<T> temp(proxy.rhs());
      viennacl::linalg::element_op(lhs, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<OP> >(temp, temp));
    }
  };

  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const LHS, const RHS, op_element_unary<OP> > >
  {
    // x += OP(y)
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<OP> > const & proxy)
    {
      vector<T> temp(proxy);
      lhs += temp;
    }

    // x += OP(vec_expr)
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS2, const RHS2, OP2>,
                      const vector_expression<const LHS2, const RHS2, OP2>,
                      op_element_unary<OP> > const & proxy)
    {
      vector<T> temp(proxy.rhs());
      viennacl::linalg::element_op(temp, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<OP> >(temp, temp)); // inplace operation is safe here
      lhs += temp;
    }
  };

  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const LHS, const RHS, op_element_unary<OP> > >
  {
    // x -= OP(y)
    static void apply(vector_base<T> & lhs, vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<OP> > const & proxy)
    {
      vector<T> temp(proxy);
      lhs -= temp;
    }

    // x -= OP(vec_expr)
    template<typename LHS2, typename RHS2, typename OP2>
    static void apply(vector_base<T> & lhs, vector_expression<const vector_expression<const LHS2, const RHS2, OP2>,
                      const vector_expression<const LHS2, const RHS2, OP2>,
                      op_element_unary<OP> > const & proxy)
    {
      vector<T> temp(proxy.rhs());
      viennacl::linalg::element_op(temp, viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<OP> >(temp, temp)); // inplace operation is safe here
      lhs -= temp;
    }
  };



  //////////// Generic user-provided routines //////////////

  template<typename T, typename UserMatrixT>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const UserMatrixT, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const UserMatrixT, const vector_base<T>, op_prod> const & rhs)
    {
      rhs.lhs().apply(rhs.rhs(), lhs);
    }
  };


} // namespace detail
} // namespace linalg

/** \endcond */

} // namespace viennacl

#endif
