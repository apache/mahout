#ifndef VIENNACL_TOOLS_MATRIX_GENERATION_HPP_
#define VIENNACL_TOOLS_MATRIX_GENERATION_HPP_

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

/** @file viennacl/tools/matrix_generation.hpp
    @brief Helper routines for generating sparse matrices
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/tools/adapter.hpp"

#include <vector>
#include <map>

namespace viennacl
{
namespace tools
{

/** @brief Generates a sparse matrix obtained from a simple finite-difference discretization of the Laplace equation on the unit square (2d).
  *
  * @tparam MatrixType  An uBLAS-compatible matrix type supporting .clear(), .resize(), and operator()-access
  * @param A            A sparse matrix object from ViennaCL, total number of unknowns will be points_x*points_y
  * @param points_x     Number of points in x-direction
  * @param points_y     Number of points in y-direction
  */
template<typename MatrixType>
void generate_fdm_laplace(MatrixType & A, vcl_size_t points_x, vcl_size_t points_y)
{
  vcl_size_t total_unknowns = points_x * points_y;

  A.clear();
  A.resize(total_unknowns, total_unknowns, false);

  for (vcl_size_t i=0; i<points_x; ++i)
  {
    for (vcl_size_t j=0; j<points_y; ++j)
    {
      vcl_size_t row = i + j * points_x;

      A(row, row) = 4.0;

      if (i > 0)
      {
        vcl_size_t col = (i-1) + j * points_x;
        A(row, col) = -1.0;
      }

      if (j > 0)
      {
        vcl_size_t col = i + (j-1) * points_x;
        A(row, col) = -1.0;
      }

      if (i < points_x-1)
      {
        vcl_size_t col = (i+1) + j * points_x;
        A(row, col) = -1.0;
      }

      if (j < points_y-1)
      {
        vcl_size_t col = i + (j+1) * points_x;
        A(row, col) = -1.0;
      }
    }
  }

}

template<typename NumericT>
void generate_fdm_laplace(viennacl::compressed_matrix<NumericT> & A, vcl_size_t points_x, vcl_size_t points_y)
{
  // Assemble into temporary matrix on CPU, then copy over:
  std::vector< std::map<unsigned int, NumericT> > temp_A;
  viennacl::tools::sparse_matrix_adapter<NumericT> adapted_A(temp_A);
  generate_fdm_laplace(adapted_A, points_x, points_y);
  viennacl::copy(temp_A, A);
}

template<typename NumericT>
void generate_fdm_laplace(viennacl::coordinate_matrix<NumericT> & A, vcl_size_t points_x, vcl_size_t points_y)
{
  // Assemble into temporary matrix on CPU, then copy over:
  std::vector< std::map<unsigned int, NumericT> > temp_A;
  viennacl::tools::sparse_matrix_adapter<NumericT> adapted_A(temp_A);
  generate_fdm_laplace(adapted_A, points_x, points_y);
  viennacl::copy(temp_A, A);
}

template<typename NumericT>
void generate_fdm_laplace(viennacl::ell_matrix<NumericT> & A, vcl_size_t points_x, vcl_size_t points_y)
{
  // Assemble into temporary matrix on CPU, then copy over:
  std::vector< std::map<unsigned int, NumericT> > temp_A;
  viennacl::tools::sparse_matrix_adapter<NumericT> adapted_A(temp_A);
  generate_fdm_laplace(adapted_A, points_x, points_y);
  viennacl::copy(temp_A, A);
}

template<typename NumericT>
void generate_fdm_laplace(viennacl::sliced_ell_matrix<NumericT> & A, vcl_size_t points_x, vcl_size_t points_y)
{
  // Assemble into temporary matrix on CPU, then copy over:
  std::vector< std::map<unsigned int, NumericT> > temp_A;
  viennacl::tools::sparse_matrix_adapter<NumericT> adapted_A(temp_A);
  generate_fdm_laplace(adapted_A, points_x, points_y);
  viennacl::copy(temp_A, A);
}

template<typename NumericT>
void generate_fdm_laplace(viennacl::hyb_matrix<NumericT> & A, vcl_size_t points_x, vcl_size_t points_y)
{
  // Assemble into temporary matrix on CPU, then copy over:
  std::vector< std::map<unsigned int, NumericT> > temp_A;
  viennacl::tools::sparse_matrix_adapter<NumericT> adapted_A(temp_A);
  generate_fdm_laplace(adapted_A, points_x, points_y);
  viennacl::copy(temp_A, A);
}


} //namespace tools
} //namespace viennacl


#endif
