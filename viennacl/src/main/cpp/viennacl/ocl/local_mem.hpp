#ifndef VIENNACL_OCL_LOCAL_MEM_HPP_
#define VIENNACL_OCL_LOCAL_MEM_HPP_

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


/** @file viennacl/ocl/local_mem.hpp
    @brief A local (shared) memory object for OpenCL
*/

#include "viennacl/forwards.h"

namespace viennacl
{
namespace ocl
{
/** @brief A class representing local (shared) OpenCL memory. Typically used as kernel argument */
class local_mem
{
public:
  local_mem(vcl_size_t s) : size_(s) {}

  /** @brief Returns size in bytes */
  vcl_size_t size() const { return size_; }

  /** @brief Sets the size of the local memory in bytes */
  void size(vcl_size_t s) { size_ = s; }

private:
  vcl_size_t size_;
};

}
}
#endif

