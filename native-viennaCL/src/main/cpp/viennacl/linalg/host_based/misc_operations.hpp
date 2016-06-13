#ifndef VIENNACL_LINALG_HOST_BASED_MISC_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_MISC_OPERATIONS_HPP_

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

/** @file viennacl/linalg/host_based/misc_operations.hpp
    @brief Implementations of miscellaneous operations on the CPU using a single thread or OpenMP.
*/

#include <list>

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/host_based/common.hpp"

namespace viennacl
{
namespace linalg
{
namespace host_based
{
namespace detail
{
  template<typename NumericT>
  void level_scheduling_substitute(vector<NumericT> & vec,
                                   viennacl::backend::mem_handle const & row_index_array,
                                   viennacl::backend::mem_handle const & row_buffer,
                                   viennacl::backend::mem_handle const & col_buffer,
                                   viennacl::backend::mem_handle const & element_buffer,
                                   vcl_size_t num_rows
                                  )
  {
    NumericT * vec_buf = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(vec.handle());

    unsigned int const * elim_row_index  = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(row_index_array);
    unsigned int const * elim_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(row_buffer);
    unsigned int const * elim_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(col_buffer);
    NumericT     const * elim_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(element_buffer);

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
    for (long row=0; row < static_cast<long>(num_rows); ++row)
    {
      unsigned int  eq_row = elim_row_index[row];
      unsigned int row_end = elim_row_buffer[row+1];
      NumericT   vec_entry = vec_buf[eq_row];

      for (vcl_size_t j = elim_row_buffer[row]; j < row_end; ++j)
        vec_entry -= vec_buf[elim_col_buffer[j]] * elim_elements[j];

      vec_buf[eq_row] = vec_entry;
    }

  }
}

} // namespace host_based
} //namespace linalg
} //namespace viennacl


#endif
