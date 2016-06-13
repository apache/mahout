#ifndef VIENNACL_LINALG_CUDA_MISC_OPERATIONS_HPP_
#define VIENNACL_LINALG_CUDA_MISC_OPERATIONS_HPP_

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

/** @file viennacl/linalg/cuda/misc_operations.hpp
    @brief Implementations of miscellaneous operations using CUDA
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/cuda/common.hpp"


namespace viennacl
{
namespace linalg
{
namespace cuda
{
namespace detail
{

template<typename NumericT>
__global__ void level_scheduling_substitute_kernel(
          const unsigned int * row_index_array,
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
          NumericT * vec,
          unsigned int size)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < size;
                    row += gridDim.x * blockDim.x)
  {
    unsigned int eq_row = row_index_array[row];
    NumericT vec_entry = vec[eq_row];
    unsigned int row_end = row_indices[row+1];

    for (unsigned int j = row_indices[row]; j < row_end; ++j)
      vec_entry -= vec[column_indices[j]] * elements[j];

    vec[eq_row] = vec_entry;
  }
}



template<typename NumericT>
void level_scheduling_substitute(vector<NumericT> & vec,
                             viennacl::backend::mem_handle const & row_index_array,
                             viennacl::backend::mem_handle const & row_buffer,
                             viennacl::backend::mem_handle const & col_buffer,
                             viennacl::backend::mem_handle const & element_buffer,
                             vcl_size_t num_rows
                            )
{
  level_scheduling_substitute_kernel<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(row_index_array),
                                                   viennacl::cuda_arg<unsigned int>(row_buffer),
                                                   viennacl::cuda_arg<unsigned int>(col_buffer),
                                                   viennacl::cuda_arg<NumericT>(element_buffer),
                                                   viennacl::cuda_arg(vec),
                                                   static_cast<unsigned int>(num_rows)
                                                  );
}

} //namespace detail
} //namespace cuda
} //namespace linalg
} //namespace viennacl


#endif
