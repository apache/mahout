#ifndef VIENNACL_RANGE_HPP_
#define VIENNACL_RANGE_HPP_

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

/** @file range.hpp
    @brief Implementation of a range object for use with proxy objects
*/

#include <vector>
#include <stddef.h>
#include <assert.h>
#include "viennacl/forwards.h"

namespace viennacl
{

/** @brief A range class that refers to an interval [start, stop), where 'start' is included, and 'stop' is excluded.
 *
 * Similar to the boost::numeric::ublas::basic_range class.
 */
template<typename SizeT /* see forwards.h for default argument*/,
         typename DistanceT /* see forwards.h for default argument*/>
class basic_range
{
public:
  typedef SizeT             size_type;
  typedef DistanceT         difference_type;
  typedef size_type            value_type;
  typedef value_type           const_reference;
  typedef const_reference      reference;

  basic_range() : start_(0), size_(0) {}
  basic_range(size_type start_index, size_type stop_index) : start_(start_index), size_(stop_index - start_index)
  {
    assert(start_index <= stop_index);
  }


  size_type start() const { return start_; }
  size_type size() const { return size_; }

  const_reference operator()(size_type i) const
  {
    assert(i < size());
    return start_ + i;
  }
  const_reference operator[](size_type i) const { return operator()(i); }

  bool operator==(const basic_range & r) const { return (start_ == r.start_) && (size_ == r.size_); }
  bool operator!=(const basic_range & r) const { return !(*this == r); }

private:
  size_type start_;
  size_type size_;
};


}

#endif
