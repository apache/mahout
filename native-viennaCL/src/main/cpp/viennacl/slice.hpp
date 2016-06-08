#ifndef VIENNACL_SLICE_HPP_
#define VIENNACL_SLICE_HPP_

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

/** @file slice.hpp
    @brief Implementation of a slice object for use with proxy objects
*/

#include <vector>
#include <stddef.h>
#include <assert.h>
#include "viennacl/forwards.h"

namespace viennacl
{

/** @brief A slice class that refers to an interval [start, stop), where 'start' is included, and 'stop' is excluded.
 *
 * Similar to the boost::numeric::ublas::basic_range class.
 */
template<typename SizeT /* see forwards.h for default argument*/,
         typename DistanceT /* see forwards.h for default argument*/>
class basic_slice
{
public:
  typedef SizeT             size_type;
  typedef DistanceT         difference_type;
  typedef size_type            value_type;
  typedef value_type           const_reference;
  typedef const_reference      reference;

  basic_slice() : start_(0), stride_(1), size_(0) {}
  basic_slice(size_type start_index,
              size_type stride_arg,
              size_type size_arg) : start_(start_index), stride_(stride_arg), size_(size_arg) {}


  size_type       start() const { return start_; }
  size_type       stride() const { return stride_; }
  size_type       size() const { return size_; }

  const_reference operator()(size_type i) const
  {
    assert(i < size());
    return start_ + i * stride_;
  }
  const_reference operator[](size_type i) const { return operator()(i); }

  bool operator==(const basic_slice & s) const { return (start_ == s.start_) && (stride_ == s.stride_) && (size_ == s.size_); }
  bool operator!=(const basic_slice & s) const { return !(*this == s); }

private:
  size_type start_;
  size_type stride_;
  size_type size_;
};


}

#endif
