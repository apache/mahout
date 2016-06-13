#ifndef VIENNACL_TRAITS_STRIDE_HPP_
#define VIENNACL_TRAITS_STRIDE_HPP_

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

/** @file viennacl/traits/stride.hpp
    @brief Determines row and column increments for matrices and matrix proxies
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"
#include "viennacl/meta/result_of.hpp"


#include <vector>
#include <map>

namespace viennacl
{
namespace traits
{

//
// inc: Increment for vectors. Defaults to 1
//
template<typename T>
typename result_of::size_type< viennacl::vector_base<T> >::type
stride(viennacl::vector_base<T> const & s) { return s.stride(); }

//
// inc1: Row increment for matrices. Defaults to 1
//
//template<typename MatrixType>
//typename result_of::size_type<MatrixType>::type
//stride1(MatrixType const &) { return 1; }
template<typename NumericT>
typename result_of::size_type< matrix_base<NumericT> >::type
stride1(matrix_base<NumericT> const & s) { return s.stride1(); }

//
// inc2: Column increment for matrices. Defaults to 1
//
//template<typename MatrixType>
//typename result_of::size_type<MatrixType>::type
//stride2(MatrixType const &) { return 1; }
template<typename NumericT>
typename result_of::size_type< matrix_base<NumericT> >::type
stride2(matrix_base<NumericT> const & s) { return s.stride2(); }


} //namespace traits
} //namespace viennacl


#endif
