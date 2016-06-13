#ifndef VIENNACL_TRAITS_ROW_MAJOR_HPP_
#define VIENNACL_TRAITS_ROW_MAJOR_HPP_

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

/** @file viennacl/traits/row_major.hpp
    @brief Determines whether a given expression has a row-major matrix layout
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

template<typename T>
bool row_major(T const &) { return true; } //default implementation: If there is no underlying matrix type, we take the result to be row-major

template<typename NumericT>
bool row_major(matrix_base<NumericT> const & A) { return A.row_major(); }

template<typename LHS, typename RHS, typename OP>
bool row_major(matrix_expression<LHS, RHS, OP> const & proxy) { return viennacl::traits::row_major(proxy.lhs()); }

} //namespace traits
} //namespace viennacl


#endif
