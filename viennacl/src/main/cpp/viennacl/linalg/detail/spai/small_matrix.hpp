#ifndef VIENNACL_LINALG_DETAIL_SPAI_SMALL_MATRIX_HPP
#define VIENNACL_LINALG_DETAIL_SPAI_SMALL_MATRIX_HPP

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

/** @file viennacl/linalg/detail/spai/small_matrix.hpp
    @brief Implementation of a routines for small matrices (helper for SPAI). Experimental.

    SPAI code contributed by Nikolay Lukash
*/

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <math.h>
#include <map>
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/storage.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/lu.hpp"
#include "boost/numeric/ublas/triangular.hpp"
#include "boost/numeric/ublas/matrix_expression.hpp"
#include "boost/numeric/ublas/detail/matrix_assign.hpp"

#include "viennacl/forwards.h"

namespace viennacl
{
namespace linalg
{
namespace detail
{
namespace spai
{

//
// Constructs an orthonormal sparse matrix M (with M^T M = Id). Is composed of elementary 2x2 rotation matrices with suitable renumbering.
//
template<typename MatrixT>
void make_rotation_matrix(MatrixT & mat,
                          vcl_size_t new_size,
                          vcl_size_t off_diagonal_distance = 4)
{
  mat.resize(new_size, new_size, false);
  mat.clear();

  double val = 1.0 / std::sqrt(2.0);

  for (vcl_size_t i=0; i<new_size; ++i)
    mat(i,i) = val;

  for (vcl_size_t i=off_diagonal_distance; i<new_size; ++i)
  {
    mat(i-off_diagonal_distance, i)                       = val;
    mat(i,                       i-off_diagonal_distance) = -val;
  }

}


//calcualtes matrix determinant
template<typename MatrixT>
double determinant(boost::numeric::ublas::matrix_expression<MatrixT> const & mat_r)
{
  double det = 1.0;

  MatrixT mLu(mat_r());
  boost::numeric::ublas::permutation_matrix<vcl_size_t> pivots(mat_r().size1());

  int is_singular = static_cast<int>(lu_factorize(mLu, pivots));

  if (!is_singular)
  {
    for (vcl_size_t i=0; i < pivots.size(); ++i)
    {
      if (pivots(i) != i)
        det *= -1.0;

      det *= mLu(i,i);
    }
  }
  else
    det = 0.0;

  return det;
}

}
}
}
}
#endif
