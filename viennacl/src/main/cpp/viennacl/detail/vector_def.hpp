#ifndef VIENNACL_DETAIL_VECTOR_DEF_HPP_
#define VIENNACL_DETAIL_VECTOR_DEF_HPP_

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

/** @file  viennacl/detail/vector_def.hpp
    @brief Forward declarations of the implicit_vector_base, vector_base class.
*/

#include "viennacl/forwards.h"
#include "viennacl/tools/entry_proxy.hpp"

namespace viennacl
{

/** @brief Common base class for representing vectors where the entries are not all stored explicitly.
  *
  * Typical examples are zero_vector or scalar_vector.
  */
template<typename NumericT>
class implicit_vector_base
{
protected:
  implicit_vector_base(vcl_size_t s, vcl_size_t i, NumericT v, viennacl::context ctx) : size_(s), index_(std::make_pair(true,i)), value_(v), ctx_(ctx){ }
  implicit_vector_base(vcl_size_t s, NumericT v, viennacl::context ctx) : size_(s), index_(std::make_pair(false,0)), value_(v), ctx_(ctx){ }

public:
  typedef NumericT const & const_reference;
  typedef NumericT cpu_value_type;

  viennacl::context context() const { return ctx_; }
  vcl_size_t size() const { return size_; }
  cpu_value_type  value() const { return value_; }
  vcl_size_t index() const { return index_.second; }
  bool has_index() const { return index_.first; }

  cpu_value_type operator()(vcl_size_t i) const
  {
    if (index_.first)
      return (i==index_.second)?value_:0;
    return value_;
  }

  cpu_value_type operator[](vcl_size_t i) const
  {
    if (index_.first)
      return (i==index_.second)?value_:0;
    return
        value_;
  }

protected:
  vcl_size_t size_;
  std::pair<bool, vcl_size_t> index_;
  NumericT value_;
  viennacl::context ctx_;
};

/** @brief Represents a vector consisting of 1 at a given index and zeros otherwise.*/
template<typename NumericT>
struct unit_vector : public implicit_vector_base<NumericT>
{
  unit_vector(vcl_size_t s, vcl_size_t ind, viennacl::context ctx = viennacl::context()) : implicit_vector_base<NumericT>(s, ind, 1, ctx)
  {
    assert( (ind < s) && bool("Provided index out of range!") );
  }
};


/** @brief Represents a vector consisting of scalars 's' only, i.e. v[i] = s for all i. To be used as an initializer for viennacl::vector, vector_range, or vector_slize only. */
template<typename NumericT>
struct scalar_vector : public implicit_vector_base<NumericT>
{
  scalar_vector(vcl_size_t s, NumericT val, viennacl::context ctx = viennacl::context()) : implicit_vector_base<NumericT>(s, val, ctx) {}
};

template<typename NumericT>
struct zero_vector : public scalar_vector<NumericT>
{
  zero_vector(vcl_size_t s, viennacl::context ctx = viennacl::context()) : scalar_vector<NumericT>(s, 0, ctx){}
};


/** @brief Common base class for dense vectors, vector ranges, and vector slices.
  *
  * @tparam NumericT   The floating point type, either 'float' or 'double'
  */
template<class NumericT, typename SizeT /* see forwards.h for default type */, typename DistanceT /* see forwards.h for default type */>
class vector_base
{
  typedef vector_base<NumericT, SizeT, DistanceT>         self_type;

public:
  typedef scalar<NumericT>                                value_type;
  typedef NumericT                                        cpu_value_type;
  typedef viennacl::backend::mem_handle                     handle_type;
  typedef SizeT                                          size_type;
  typedef DistanceT                                      difference_type;
  typedef const_vector_iterator<NumericT, 1>              const_iterator;
  typedef vector_iterator<NumericT, 1>                    iterator;

  /** @brief Returns the length of the vector (cf. std::vector)  */
  size_type size() const { return size_; }
  /** @brief Returns the internal length of the vector, which is given by size() plus the extra memory due to padding the memory with zeros up to a multiple of 'AlignmentV' */
  size_type internal_size() const { return internal_size_; }
  /** @brief Returns the offset within the buffer  */
  size_type start() const { return start_; }
  /** @brief Returns the stride within the buffer (in multiples of sizeof(NumericT)) */
  size_type stride() const { return stride_; }
  /** @brief Returns true is the size is zero */
  bool empty() const { return size_ == 0; }
  /** @brief Returns the memory handle. */
  const handle_type & handle() const { return elements_; }
  /** @brief Returns the memory handle. */
  handle_type & handle() { return elements_; }
  viennacl::memory_types memory_domain() const { return elements_.get_active_handle_id();  }

  /** @brief Default constructor in order to be compatible with various containers.
    */
  explicit vector_base();

  /** @brief An explicit constructor for wrapping an existing vector into a vector_range or vector_slice.
     *
     * @param h          The existing memory handle from a vector/vector_range/vector_slice
     * @param vec_size   The length (i.e. size) of the buffer
     * @param vec_start  The offset from the beginning of the buffer identified by 'h'
     * @param vec_stride Increment between two elements in the original buffer (in multiples of NumericT)
    */
  explicit vector_base(viennacl::backend::mem_handle & h, size_type vec_size, size_type vec_start, size_type vec_stride);

  /** @brief Creates a vector and allocates the necessary memory */
  explicit vector_base(size_type vec_size, viennacl::context ctx = viennacl::context());

  // CUDA or host memory:
  explicit vector_base(NumericT * ptr_to_mem, viennacl::memory_types mem_type, size_type vec_size, vcl_size_t start = 0, size_type stride = 1);

#ifdef VIENNACL_WITH_OPENCL
  /** @brief Create a vector from existing OpenCL memory
    *
    * Note: The provided memory must take an eventual AlignmentV into account, i.e. existing_mem must be at least of size internal_size()!
    * This is trivially the case with the default alignment, but should be considered when using vector<> with an alignment parameter not equal to 1.
    *
    * @param existing_mem   An OpenCL handle representing the memory
    * @param vec_size       The size of the vector.
    */
  explicit vector_base(cl_mem existing_mem, size_type vec_size, size_type start = 0, size_type stride = 1, viennacl::context ctx = viennacl::context());
#endif

  template<typename LHS, typename RHS, typename OP>
  explicit vector_base(vector_expression<const LHS, const RHS, OP> const & proxy);

  // Copy CTOR:
  vector_base(const self_type & other);

  // Conversion CTOR:
  template<typename OtherNumericT>
  vector_base(const vector_base<OtherNumericT> & v1);

  /** @brief Assignment operator. Other vector needs to be of the same size, or this vector is not yet initialized.
    */
  self_type & operator=(const self_type & vec);
  /** @brief Implementation of the operation v1 = v2 @ alpha, where @ denotes either multiplication or division, and alpha is either a CPU or a GPU scalar
    * @param proxy  An expression template proxy class.
    */
  template<typename LHS, typename RHS, typename OP>
  self_type & operator=(const vector_expression<const LHS, const RHS, OP> & proxy);
  /** @brief Converts a vector of a different numeric type to the current numeric type */
  template<typename OtherNumericT>
  self_type &  operator = (const vector_base<OtherNumericT> & v1);
  /** @brief Creates the vector from the supplied unit vector. */
  self_type & operator = (unit_vector<NumericT> const & v);
  /** @brief Creates the vector from the supplied zero vector. */
  self_type & operator = (zero_vector<NumericT> const & v);
  /** @brief Creates the vector from the supplied scalar vector. */
  self_type & operator = (scalar_vector<NumericT> const & v);


  ///////////////////////////// Matrix Vector interaction start ///////////////////////////////////
  /** @brief Operator overload for v1 = A * v2, where v1, v2 are vectors and A is a dense matrix.
    * @param proxy An expression template proxy class
    */
  self_type & operator=(const viennacl::vector_expression< const matrix_base<NumericT>, const vector_base<NumericT>, viennacl::op_prod> & proxy);

  //transposed_matrix_proxy:
  /** @brief Operator overload for v1 = trans(A) * v2, where v1, v2 are vectors and A is a dense matrix.
    * @param proxy An expression template proxy class
    */
  self_type & operator=(const vector_expression< const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans >,
                        const vector_base<NumericT>,
                        op_prod> & proxy);

  ///////////////////////////// Matrix Vector interaction end ///////////////////////////////////


  //read-write access to an element of the vector
  /** @brief Read-write access to a single element of the vector */
  entry_proxy<NumericT> operator()(size_type index);
  /** @brief Read-write access to a single element of the vector */
  entry_proxy<NumericT> operator[](size_type index);
  /** @brief Read access to a single element of the vector */
  const_entry_proxy<NumericT> operator()(size_type index) const;
  /** @brief Read access to a single element of the vector */
  const_entry_proxy<NumericT> operator[](size_type index) const;
  self_type & operator += (const self_type & vec);
  self_type & operator -= (const self_type & vec);

  /** @brief Scales a vector (or proxy) by a char (8-bit integer) */
  self_type & operator *= (char val);
  /** @brief Scales a vector (or proxy) by a short integer */
  self_type & operator *= (short val);
  /** @brief Scales a vector (or proxy) by an integer */
  self_type & operator *= (int val);
  /** @brief Scales a vector (or proxy) by a long integer */
  self_type & operator *= (long val);
  /** @brief Scales a vector (or proxy) by a single precision floating point value */
  self_type & operator *= (float val);
  /** @brief Scales a vector (or proxy) by a double precision floating point value */
  self_type & operator *= (double val);


  /** @brief Scales a vector (or proxy) by a char (8-bit integer) */
  self_type & operator /= (char val);
  /** @brief Scales a vector (or proxy) by a short integer */
  self_type & operator /= (short val);
  /** @brief Scales a vector (or proxy) by an integer */
  self_type & operator /= (int val);
  /** @brief Scales a vector (or proxy) by a long integer */
  self_type & operator /= (long val);
  /** @brief Scales a vector (or proxy) by a single precision floating point value */
  self_type & operator /= (float val);
  /** @brief Scales a vector (or proxy) by a double precision floating point value */
  self_type & operator /= (double val);

  /** @brief Scales the vector by a char (8-bit integer) 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_mult>
  operator * (char value) const;
  /** @brief Scales the vector by a short integer 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_mult>
  operator * (short value) const;
  /** @brief Scales the vector by an integer 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_mult>
  operator * (int value) const;
  /** @brief Scales the vector by a long integer 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_mult>
  operator * (long value) const;
  /** @brief Scales the vector by a single precision floating point value 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_mult>
  operator * (float value) const;
  /** @brief Scales the vector by a double precision floating point value 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_mult>
  operator * (double value) const;

  /** @brief Scales the vector by a char (8-bit integer) 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_div>
  operator / (char value) const;
  /** @brief Scales the vector by a short integer 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_div>
  operator / (short value) const;
  /** @brief Scales the vector by an integer 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_div>
  operator / (int value) const;
  /** @brief Scales the vector by a long integer 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_div>
  operator / (long value) const;
  /** @brief Scales the vector by a single precision floating point value 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_div>
  operator / (float value) const;
  /** @brief Scales the vector by a double precision floating point value 'alpha' and returns an expression template */
  vector_expression< const self_type, const NumericT, op_div>
  operator / (double value) const;

  /** @brief Sign flip for the vector. Emulated to be equivalent to -1.0 * vector */
  vector_expression<const self_type, const NumericT, op_mult> operator-() const;
  /** @brief Returns an iterator pointing to the beginning of the vector  (STL like)*/
  iterator begin();
  /** @brief Returns an iterator pointing to the end of the vector (STL like)*/
  iterator end();
  /** @brief Returns a const-iterator pointing to the beginning of the vector (STL like)*/
  const_iterator begin() const;
  /** @brief Returns a const-iterator pointing to the end of the vector (STL like)*/
  const_iterator end() const;
  /** @brief Swaps the entries of the two vectors */
  self_type & swap(self_type & other);

  /** @brief Resets all entries to zero. Does not change the size of the vector. */
  void clear();

protected:

  void set_handle(viennacl::backend::mem_handle const & h) {  elements_ = h; }

  /** @brief Swaps the handles of two vectors by swapping the OpenCL handles only, no data copy */
  self_type & fast_swap(self_type & other);

  /** @brief Pads vectors with alignment > 1 with trailing zeros if the internal size is larger than the visible size */
  void pad();

  void switch_memory_context(viennacl::context new_ctx);

  //TODO: Think about implementing the following public member functions
  //void insert_element(unsigned int i, NumericT val){}
  //void erase_element(unsigned int i){}

  //enlarge or reduce allocated memory and set unused memory to zero
  /** @brief Resizes the allocated memory for the vector. Pads the memory to be a multiple of 'AlignmentV'
    *
    *  @param new_size  The new size of the vector
    *  @param preserve  If true, old entries of the vector are preserved, otherwise eventually discarded.
    */
  void resize(size_type new_size, bool preserve = true);

  /** @brief Resizes the allocated memory for the vector. Convenience function for setting an OpenCL context in case reallocation is needed
    *
    *  @param new_size  The new size of the vector
    *  @param ctx       The context within which the new memory should be allocated
    *  @param preserve  If true, old entries of the vector are preserved, otherwise eventually discarded.
    */
  void resize(size_type new_size, viennacl::context ctx, bool preserve = true);
private:

  void resize_impl(size_type new_size, viennacl::context ctx, bool preserve = true);

  size_type       size_;
  size_type       start_;
  size_type       stride_;
  size_type       internal_size_;
  handle_type elements_;
}; //vector_base

/** \endcond */

} // namespace viennacl

#endif
