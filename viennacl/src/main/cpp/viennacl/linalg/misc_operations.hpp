#ifndef VIENNACL_LINALG_MISC_OPERATIONS_HPP_
#define VIENNACL_LINALG_MISC_OPERATIONS_HPP_

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

/** @file viennacl/linalg/misc_operations.hpp
    @brief Implementations of miscellaneous operations
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/host_based/misc_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/misc_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/misc_operations.hpp"
#endif

namespace viennacl
{
  namespace linalg
  {

    namespace detail
    {

      template<typename ScalarType>
      void level_scheduling_substitute(vector<ScalarType> & vec,
                                  viennacl::backend::mem_handle const & row_index_array,
                                  viennacl::backend::mem_handle const & row_buffer,
                                  viennacl::backend::mem_handle const & col_buffer,
                                  viennacl::backend::mem_handle const & element_buffer,
                                  vcl_size_t num_rows
                                  )
      {
        assert( viennacl::traits::handle(vec).get_active_handle_id() == row_index_array.get_active_handle_id() && bool("Incompatible memory domains"));
        assert( viennacl::traits::handle(vec).get_active_handle_id() ==      row_buffer.get_active_handle_id() && bool("Incompatible memory domains"));
        assert( viennacl::traits::handle(vec).get_active_handle_id() ==      col_buffer.get_active_handle_id() && bool("Incompatible memory domains"));
        assert( viennacl::traits::handle(vec).get_active_handle_id() ==  element_buffer.get_active_handle_id() && bool("Incompatible memory domains"));

        switch (viennacl::traits::handle(vec).get_active_handle_id())
        {
          case viennacl::MAIN_MEMORY:
            viennacl::linalg::host_based::detail::level_scheduling_substitute(vec, row_index_array, row_buffer, col_buffer, element_buffer, num_rows);
            break;
#ifdef VIENNACL_WITH_OPENCL
          case viennacl::OPENCL_MEMORY:
            viennacl::linalg::opencl::detail::level_scheduling_substitute(vec, row_index_array, row_buffer, col_buffer, element_buffer, num_rows);
            break;
#endif
#ifdef VIENNACL_WITH_CUDA
          case viennacl::CUDA_MEMORY:
            viennacl::linalg::cuda::detail::level_scheduling_substitute(vec, row_index_array, row_buffer, col_buffer, element_buffer, num_rows);
            break;
#endif
          case viennacl::MEMORY_NOT_INITIALIZED:
            throw memory_exception("not initialised!");
          default:
            throw memory_exception("not implemented");
        }
      }




    } //namespace detail


  } //namespace linalg
} //namespace viennacl


#endif
