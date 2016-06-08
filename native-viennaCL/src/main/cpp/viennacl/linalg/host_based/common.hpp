#ifndef VIENNACL_LINALG_HOST_BASED_COMMON_HPP_
#define VIENNACL_LINALG_HOST_BASED_COMMON_HPP_

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

/** @file viennacl/linalg/host_based/common.hpp
    @brief Common routines for single-threaded or OpenMP-enabled execution on CPU
*/

#include "viennacl/traits/handle.hpp"

namespace viennacl
{
namespace linalg
{
namespace host_based
{
namespace detail
{

template<typename ResultT, typename VectorT>
ResultT * extract_raw_pointer(VectorT & vec)
{
  return reinterpret_cast<ResultT *>(viennacl::traits::ram_handle(vec).get());
}

template<typename ResultT, typename VectorT>
ResultT const * extract_raw_pointer(VectorT const & vec)
{
  return reinterpret_cast<ResultT const *>(viennacl::traits::ram_handle(vec).get());
}

/** @brief Helper class for accessing a strided subvector of a larger vector. */
template<typename NumericT>
class vector_array_wrapper
{
public:
  typedef NumericT   value_type;

  vector_array_wrapper(value_type * A,
                       vcl_size_t start,
                       vcl_size_t inc)
   : A_(A),
     start_(start),
     inc_(inc) {}

  value_type & operator()(vcl_size_t i) { return A_[i * inc_ + start_]; }

private:
  value_type * A_;
  vcl_size_t start_;
  vcl_size_t inc_;
};


/** @brief Helper array for accessing a strided submatrix embedded in a larger matrix. */
template<typename NumericT, typename LayoutT, bool is_transposed>
class matrix_array_wrapper
{
  public:
    typedef NumericT   value_type;

    matrix_array_wrapper(value_type * A,
                         vcl_size_t start1, vcl_size_t start2,
                         vcl_size_t inc1,   vcl_size_t inc2,
                         vcl_size_t internal_size1, vcl_size_t internal_size2)
     : A_(A),
       start1_(start1), start2_(start2),
       inc1_(inc1), inc2_(inc2),
       internal_size1_(internal_size1), internal_size2_(internal_size2) {}

    value_type & operator()(vcl_size_t i, vcl_size_t j)
    {
      return A_[LayoutT::mem_index(i * inc1_ + start1_,
                                   j * inc2_ + start2_,
                                   internal_size1_, internal_size2_)];
    }

    // convenience overloads to address signed index types for OpenMP:
    value_type & operator()(vcl_size_t i, long j) { return operator()(i, static_cast<vcl_size_t>(j)); }
    value_type & operator()(long i, vcl_size_t j) { return operator()(static_cast<vcl_size_t>(i), j); }
    value_type & operator()(long i, long j)       { return operator()(static_cast<vcl_size_t>(i), static_cast<vcl_size_t>(j)); }

  private:
    value_type * A_;
    vcl_size_t start1_, start2_;
    vcl_size_t inc1_, inc2_;
    vcl_size_t internal_size1_, internal_size2_;
};

/** \cond */
template<typename NumericT, typename LayoutT>
class matrix_array_wrapper<NumericT, LayoutT, true>
{
public:
  typedef NumericT   value_type;

  matrix_array_wrapper(value_type * A,
                       vcl_size_t start1, vcl_size_t start2,
                       vcl_size_t inc1,   vcl_size_t inc2,
                       vcl_size_t internal_size1, vcl_size_t internal_size2)
   : A_(A),
     start1_(start1), start2_(start2),
     inc1_(inc1), inc2_(inc2),
     internal_size1_(internal_size1), internal_size2_(internal_size2) {}

  value_type & operator()(vcl_size_t i, vcl_size_t j)
  {
    //swapping row and column indices here
    return A_[LayoutT::mem_index(j * inc1_ + start1_,
                                 i * inc2_ + start2_,
                                 internal_size1_, internal_size2_)];
  }

  // convenience overloads to address signed index types for OpenMP:
  value_type & operator()(vcl_size_t i, long j) { return operator()(i, static_cast<vcl_size_t>(j)); }
  value_type & operator()(long i, vcl_size_t j) { return operator()(static_cast<vcl_size_t>(i), j); }
  value_type & operator()(long i, long j) { return operator()(static_cast<vcl_size_t>(i), static_cast<vcl_size_t>(j)); }

private:
  value_type * A_;
  vcl_size_t start1_, start2_;
  vcl_size_t inc1_, inc2_;
  vcl_size_t internal_size1_, internal_size2_;
};
/** \endcond */

} //namespace detail
} //namespace host_based
} //namespace linalg
} //namespace viennacl


#endif
