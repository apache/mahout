#ifndef VIENNACL_TRAITS_START_HPP_
#define VIENNACL_TRAITS_START_HPP_

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

/** @file viennacl/traits/start.hpp
    @brief Extracts the underlying OpenCL start index handle from a vector, a matrix, an expression etc.
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"

#include "viennacl/meta/result_of.hpp"

namespace viennacl
{
namespace traits
{

//
// start: Mostly for vectors
//

// Default: Try to get the start index from the .start() member function
template<typename T>
typename result_of::size_type<T>::type
start(T const & obj)
{
  return obj.start();
}

//ViennaCL vector leads to start index 0:
template<typename ScalarType, unsigned int AlignmentV>
typename result_of::size_type<viennacl::vector<ScalarType, AlignmentV> >::type
start(viennacl::vector<ScalarType, AlignmentV> const &)
{
  return 0;
}


//
// start1: Row start index
//

// Default: Try to get the start index from the .start1() member function
template<typename T>
typename result_of::size_type<T>::type
start1(T const & obj)
{
  return obj.start1();
}

//ViennaCL matrix leads to start index 0:
template<typename ScalarType, typename F, unsigned int AlignmentV>
typename result_of::size_type<viennacl::matrix<ScalarType, F, AlignmentV> >::type
start1(viennacl::matrix<ScalarType, F, AlignmentV> const &)
{
  return 0;
}


//
// start2: Column start index
//
template<typename T>
typename result_of::size_type<T>::type
start2(T const & obj)
{
  return obj.start2();
}

//ViennaCL matrix leads to start index 0:
template<typename ScalarType, typename F, unsigned int AlignmentV>
typename result_of::size_type<viennacl::matrix<ScalarType, F, AlignmentV> >::type
start2(viennacl::matrix<ScalarType, F, AlignmentV> const &)
{
  return 0;
}


} //namespace traits
} //namespace viennacl


#endif
