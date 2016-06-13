#ifndef VIENNACL_LINALG_OPENCL_VANDERMONDE_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_OPENCL_VANDERMONDE_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/opencl/vandermonde_matrix_operations.hpp
    @brief Implementations of operations using vandermonde_matrix
*/

#include "viennacl/forwards.h"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/fft.hpp"
#include "viennacl/linalg/opencl/kernels/fft.hpp"

namespace viennacl
{
namespace linalg
{
namespace opencl
{

/** @brief Carries out matrix-vector multiplication with a vandermonde_matrix
*
* Implementation of the convenience expression y = prod(A, x);
*
* @param A    The Vandermonde matrix
* @param x    The vector
* @param y    The result vector
*/
template<typename NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::vandermonde_matrix<NumericT, AlignmentV> const & A,
               viennacl::vector_base<NumericT> const & x,
               viennacl::vector_base<NumericT>       & y)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::fft<NumericT>::init(ctx);

  viennacl::ocl::kernel & kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::fft<NumericT>::program_name(), "vandermonde_prod");
  viennacl::ocl::enqueue(kernel(viennacl::traits::opencl_handle(A),
                                viennacl::traits::opencl_handle(x),
                                viennacl::traits::opencl_handle(y),
                                static_cast<cl_uint>(A.size1())));
}

} //namespace opencl
} //namespace linalg
} //namespace viennacl


#endif
