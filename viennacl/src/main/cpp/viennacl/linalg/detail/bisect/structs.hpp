#ifndef VIENNACL_LINALG_DETAIL_BISECT_STRUCTS_HPP_
#define VIENNACL_LINALG_DETAIL_BISECT_STRUCTS_HPP_

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


/** @file viennacl/linalg/detail//bisect/structs.hpp
    @brief  Helper structures to simplify variable handling

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/



#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <assert.h>

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"

namespace viennacl
{
namespace linalg
{
namespace detail
{

/////////////////////////////////////////////////////////////////////////////////
//! In this class the input matrix is stored
/////////////////////////////////////////////////////////////////////////////////
template<typename NumericT>
struct InputData
{
  //! host side representation of diagonal
  std::vector<NumericT> std_a;
  //! host side representation superdiagonal
  std::vector<NumericT> std_b;
  //! device side representation of diagonal
  viennacl::vector<NumericT> g_a;
  //!device side representation of superdiagonal
  viennacl::vector<NumericT> g_b;

  /** @brief Initialize the input data to the algorithm
   *
   * @param diagonal        vector with the diagonal elements
   * @param superdiagonal   vector with the superdiagonal elements
   * @param sz              size of the matrix
   */
  InputData(std::vector<NumericT> diagonal, std::vector<NumericT> superdiagonal, const unsigned int sz) :
              std_a(sz), std_b(sz), g_a(sz), g_b(sz)
  {
   std_a = diagonal;
   std_b = superdiagonal;

   viennacl::copy(std_b, g_b);
   viennacl::copy(std_a, g_a);
  }

  InputData(viennacl::vector<NumericT> diagonal, viennacl::vector<NumericT> superdiagonal, const unsigned int sz) :
              std_a(sz), std_b(sz), g_a(sz), g_b(sz)
  {
   g_a = diagonal;
   g_b = superdiagonal;

   viennacl::copy(g_a, std_a);
   viennacl::copy(g_b, std_b);
  }
};


/////////////////////////////////////////////////////////////////////////////////
//! In this class the data of the result for small matrices is stored
/////////////////////////////////////////////////////////////////////////////////
template<typename NumericT>
struct ResultDataSmall
{
  //! eigenvalues (host side)
  std::vector<NumericT> std_eigenvalues;
  //! left interval limits at the end of the computation
  viennacl::vector<NumericT> vcl_g_left;
  //! right interval limits at the end of the computation
  viennacl::vector<NumericT> vcl_g_right;
  //! number of eigenvalues smaller than the left interval limit
  viennacl::vector<unsigned int> vcl_g_left_count;
  //! number of eigenvalues bigger than the right interval limit
  viennacl::vector<unsigned int> vcl_g_right_count;


  ////////////////////////////////////////////////////////////////////////////////
  //! Initialize variables and memory for the result for small matrices
  ////////////////////////////////////////////////////////////////////////////////
  ResultDataSmall(const unsigned int mat_size) :
    std_eigenvalues(mat_size), vcl_g_left(mat_size), vcl_g_right(mat_size), vcl_g_left_count(mat_size), vcl_g_right_count(mat_size) {}
};





/////////////////////////////////////////////////////////////////////////////////
//! In this class the data of the result for large matrices is stored
/////////////////////////////////////////////////////////////////////////////////
template<typename NumericT>
struct ResultDataLarge
{
//! eigenvalues
  std::vector<NumericT> std_eigenvalues;

  //! number of intervals containing one eigenvalue after the first step
  viennacl::scalar<unsigned int> g_num_one;

  //! number of (thread) blocks of intervals containing multiple eigenvalues after the first steo
  viennacl::scalar<unsigned int> g_num_blocks_mult;

  //! left interval limits of intervals containing one eigenvalue after the first iteration step
  viennacl::vector<NumericT> g_left_one;

  //! right interval limits of intervals containing one eigenvalue after the first iteration step
  viennacl::vector<NumericT> g_right_one;

  //! interval indices (position in sorted listed of eigenvalues) of intervals containing one eigenvalue after the first iteration step
  viennacl::vector<unsigned int> g_pos_one;

  //! left interval limits of intervals containing multiple eigenvalues after the first iteration step
  viennacl::vector<NumericT> g_left_mult;
  //! right interval limits of intervals containing multiple eigenvalues after the first iteration step
  viennacl::vector<NumericT> g_right_mult;

  //! number of eigenvalues less than the left limit of the eigenvalue intervals containing multiple eigenvalues
  viennacl::vector<unsigned int> g_left_count_mult;

  //! number of eigenvalues less than the right limit of the eigenvalue intervals containing multiple eigenvalues
  viennacl::vector<unsigned int> g_right_count_mult;
  //! start addresses in g_left_mult etc. of blocks of intervals containing more than one eigenvalue after the first step
  viennacl::vector<unsigned int> g_blocks_mult;

  //! accumulated number of intervals in g_left_mult etc. of blocks of intervals containing more than one eigenvalue after the first step
  viennacl::vector<unsigned int> g_blocks_mult_sum;

  //! eigenvalues that have been generated in the second step from intervals that still contained multiple eigenvalues after the first step
  viennacl::vector<NumericT> g_lambda_mult;

  //! eigenvalue index of intervals that have been generated in the second processing step
  viennacl::vector<unsigned int> g_pos_mult;

  /** @brief Initialize variables and memory for result
   *
   * @param  mat_size  size of the matrix
   */
  ResultDataLarge(unsigned int mat_size) :
    std_eigenvalues(mat_size), g_num_one(0), g_num_blocks_mult(0),
    g_left_one(mat_size), g_right_one(mat_size), g_pos_one(mat_size),
    g_left_mult(mat_size), g_right_mult(mat_size), g_left_count_mult(mat_size), g_right_count_mult(mat_size),
    g_blocks_mult(mat_size), g_blocks_mult_sum(mat_size), g_lambda_mult(mat_size), g_pos_mult(mat_size) {}

};
} // namespace detail
} // namespace linalg
} // namespace viennacl
#endif // #ifndef VIENNACL_LINALG_DETAIL_STRUCTS_HPP_

