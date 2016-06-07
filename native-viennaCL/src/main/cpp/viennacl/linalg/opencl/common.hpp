#ifndef VIENNACL_LINALG_OPENCL_COMMON_HPP_
#define VIENNACL_LINALG_OPENCL_COMMON_HPP_

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

/** @file viennacl/linalg/opencl/common.hpp
    @brief Common implementations shared by OpenCL-based operations
*/

#include <cmath>

#include "viennacl/forwards.h"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/traits/handle.hpp"

namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace detail
{



inline cl_uint make_options(vcl_size_t length, bool reciprocal, bool flip_sign)
{
  return static_cast<cl_uint>( ((length > 1) ? (cl_uint(length) << 2) : 0) + (reciprocal ? 2 : 0) + (flip_sign ? 1 : 0) );
}


/** @brief Returns the OpenCL kernel string for the operation C = A * B with A sparse, B, C dense matrices. */
inline std::string sparse_dense_matmult_kernel_name(bool B_transposed, bool B_row_major, bool C_row_major)
{
  if (B_transposed)
  {
    if (B_row_major && C_row_major)
      return "trans_mat_mult_row_row";
    if (B_row_major && !C_row_major)
      return "trans_mat_mult_row_col";
    if (!B_row_major && C_row_major)
      return "trans_mat_mult_col_row";

    return "trans_mat_mult_col_col";
  }

  if (B_row_major && C_row_major)
    return "mat_mult_row_row";
  if (B_row_major && !C_row_major)
    return "mat_mult_row_col";
  if (!B_row_major && C_row_major)
    return "mat_mult_col_row";

  return "mat_mult_col_col";
}



template<typename SomeT>
ocl::device const & current_device(SomeT const & obj) {  return traits::opencl_handle(obj).context().current_device(); }

inline std::string op_to_string(op_abs)   { return "abs";   }
inline std::string op_to_string(op_acos)  { return "acos";  }
inline std::string op_to_string(op_asin)  { return "asin";  }
inline std::string op_to_string(op_atan)  { return "atan";  }
inline std::string op_to_string(op_ceil)  { return "ceil";  }
inline std::string op_to_string(op_cos)   { return "cos";   }
inline std::string op_to_string(op_cosh)  { return "cosh";  }
inline std::string op_to_string(op_exp)   { return "exp";   }
inline std::string op_to_string(op_fabs)  { return "fabs";  }
inline std::string op_to_string(op_floor) { return "floor"; }
inline std::string op_to_string(op_log)   { return "log";   }
inline std::string op_to_string(op_log10) { return "log10"; }
inline std::string op_to_string(op_sin)   { return "sin";   }
inline std::string op_to_string(op_sinh)  { return "sinh";  }
inline std::string op_to_string(op_sqrt)  { return "sqrt";  }
inline std::string op_to_string(op_tan)   { return "tan";   }
inline std::string op_to_string(op_tanh)  { return "tanh";  }

} //namespace detail
} //namespace opencl
} //namespace linalg
} //namespace viennacl


#endif
