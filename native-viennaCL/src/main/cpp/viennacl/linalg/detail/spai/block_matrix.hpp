#ifndef VIENNACL_LINALG_DETAIL_SPAI_BLOCK_MATRIX_HPP
#define VIENNACL_LINALG_DETAIL_SPAI_BLOCK_MATRIX_HPP

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

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include "viennacl/ocl/backend.hpp"
#include "viennacl/tools/tools.hpp"

/** @file viennacl/linalg/detail/spai/block_matrix.hpp
    @brief Implementation of a bunch of (small) matrices on GPU. Experimental.

    SPAI code contributed by Nikolay Lukash
*/

namespace viennacl
{
namespace linalg
{
namespace detail
{
namespace spai
{

/**
* @brief Represents contigious matrices on GPU
*/

class block_matrix
{
public:

  ////////// non-const

  /** @brief Returns a handle to the elements */
  viennacl::ocl::handle<cl_mem>& handle(){ return elements_; }

  /** @brief Returns a handle to the matrix dimensions */
  viennacl::ocl::handle<cl_mem>& handle1() { return matrix_dimensions_; }

  /** @brief Returns a handle to the start indices of matrix */
  viennacl::ocl::handle<cl_mem>& handle2() { return start_block_inds_; }

  ////////// const

  /** @brief Returns a handle to the const elements */
  const viennacl::ocl::handle<cl_mem>& handle() const { return elements_; }

  /** @brief Returns a handle to the const matrix dimensions */
  const viennacl::ocl::handle<cl_mem>& handle1() const { return matrix_dimensions_; }

  /** @brief Returns a handle to the const start indices of matrix */
  const viennacl::ocl::handle<cl_mem>& handle2() const { return start_block_inds_; }

private:
  viennacl::ocl::handle<cl_mem> elements_;
  viennacl::ocl::handle<cl_mem> matrix_dimensions_;
  viennacl::ocl::handle<cl_mem> start_block_inds_;
};


}
}
}
}
#endif
