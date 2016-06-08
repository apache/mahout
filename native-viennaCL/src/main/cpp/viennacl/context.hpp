#ifndef VIENNACL_CONTEXT_HPP_
#define VIENNACL_CONTEXT_HPP_

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

/** @file viennacl/context.hpp
    @brief Implementation of a OpenCL-like context, which serves as a unification of {OpenMP, CUDA, OpenCL} at the user API.
*/

#include <vector>
#include <stddef.h>
#include <assert.h>
#include "viennacl/forwards.h"
#include "viennacl/ocl/forwards.h"
#include "viennacl/backend/mem_handle.hpp"

namespace viennacl
{
/** @brief Represents a generic 'context' similar to an OpenCL context, but is backend-agnostic and thus also suitable for CUDA and OpenMP
  *
  * Context objects are used to distinguish between different memory domains. One context may refer to an OpenCL device, another context may refer to a CUDA device, and a third context to main RAM.
  * Thus, operations are only defined on objects residing on the same context.
  */
class context
{
public:
  context() : mem_type_(viennacl::backend::default_memory_type())
  {
#ifdef VIENNACL_WITH_OPENCL
    if (mem_type_ == OPENCL_MEMORY)
      ocl_context_ptr_ = &viennacl::ocl::current_context();
    else
      ocl_context_ptr_ = NULL;
#endif
  }

  explicit context(viennacl::memory_types mtype) : mem_type_(mtype)
  {
    if (mem_type_ == MEMORY_NOT_INITIALIZED)
      mem_type_ = viennacl::backend::default_memory_type();
#ifdef VIENNACL_WITH_OPENCL
    if (mem_type_ == OPENCL_MEMORY)
      ocl_context_ptr_ = &viennacl::ocl::current_context();
    else
      ocl_context_ptr_ = NULL;
#endif
  }

#ifdef VIENNACL_WITH_OPENCL
  context(viennacl::ocl::context const & ctx) : mem_type_(OPENCL_MEMORY), ocl_context_ptr_(&ctx) {}

  viennacl::ocl::context const & opencl_context() const
  {
    assert(mem_type_ == OPENCL_MEMORY && bool("Context type is not OpenCL"));
    return *ocl_context_ptr_;
  }
#endif

  // TODO: Add CUDA and OpenMP contexts

  viennacl::memory_types  memory_type() const { return mem_type_; }

private:
  viennacl::memory_types   mem_type_;
#ifdef VIENNACL_WITH_OPENCL
  viennacl::ocl::context const * ocl_context_ptr_;
#endif
};


}

#endif
