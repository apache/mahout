#ifndef VIENNACL_TRAITS_FILL_HPP_
#define VIENNACL_TRAITS_FILL_HPP_

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

/** @file viennacl/traits/fill.hpp
    @brief Generic fill functionality for different matrix types
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"
#include "viennacl/meta/result_of.hpp"

#ifdef VIENNACL_WITH_EIGEN
#include <Eigen/Core>
#include <Eigen/Sparse>
#endif

#include <vector>
#include <map>

namespace viennacl
{
namespace traits
{

/** @brief Generic filler routine for setting an entry of a matrix to a particular value */
template<typename MatrixType, typename NumericT>
void fill(MatrixType & matrix, vcl_size_t row_index, vcl_size_t col_index, NumericT value)
{
  matrix(row_index, col_index) = value;
}

#ifdef VIENNACL_WITH_EIGEN
/** @brief Generic filler routine for setting an entry of a matrix to a particular value. Special case for Eigen sparse matrices. */
template<typename T, int options, typename NumericT>
inline void fill(Eigen::SparseMatrix<T, options> & m,
                 vcl_size_t row_index,
                 vcl_size_t col_index,
                 NumericT value
                 )
{
  m.insert(row_index, col_index) = value;
}
#endif

} //namespace traits
} //namespace viennacl


#endif
