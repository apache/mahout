#ifndef VIENNACL_TRAITS_CLEAR_HPP_
#define VIENNACL_TRAITS_CLEAR_HPP_

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

/** @file viennacl/traits/clear.hpp
    @brief Generic clear functionality for different vector and matrix types
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"

#include "viennacl/traits/size.hpp"

#include <vector>
#include <map>

namespace viennacl
{
namespace traits
{

//clear:
/** @brief Generic routine for setting all entries of a vector to zero. This is the version for non-ViennaCL objects. */
template<typename VectorType>
void clear(VectorType & vec)
{
  typedef typename viennacl::result_of::size_type<VectorType>::type  size_type;

  for (size_type i=0; i<viennacl::traits::size(vec); ++i)
    vec[i] = 0;  //TODO: Quantity access can also be wrapped...
}

/** @brief Generic routine for setting all entries of a vector to zero. This is the version for ViennaCL objects. */
template<typename ScalarType, unsigned int AlignmentV>
void clear(viennacl::vector<ScalarType, AlignmentV> & vec)
{
  vec.clear();
}

} //namespace traits
} //namespace viennacl


#endif
