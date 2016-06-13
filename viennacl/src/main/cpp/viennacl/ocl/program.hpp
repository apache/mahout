#ifndef VIENNACL_OCL_PROGRAM_HPP_
#define VIENNACL_OCL_PROGRAM_HPP_

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

/** @file viennacl/ocl/program.hpp
    @brief Implements an OpenCL program class for ViennaCL
*/

#include <string>
#include <vector>
#include "viennacl/ocl/forwards.h"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/tools/shared_ptr.hpp"

namespace viennacl
{
namespace ocl
{

/** @brief Wrapper class for an OpenCL program.
  *
  * This class was written when the OpenCL C++ bindings haven't been standardized yet.
  * Regardless, it takes care about some additional details and is supposed to provide higher convenience by holding the kernels defined in the program.
  */
class program
{
  typedef std::vector<tools::shared_ptr<viennacl::ocl::kernel> >    kernel_container_type;

public:
  program() : p_context_(NULL) {}
  program(cl_program program_handle, viennacl::ocl::context const & program_context, std::string const & prog_name = std::string())
    : handle_(program_handle, program_context), p_context_(&program_context), name_(prog_name) {}

  program(program const & other) : handle_(other.handle_), p_context_(other.p_context_), name_(other.name_), kernels_(other.kernels_) {      }

  viennacl::ocl::program & operator=(const program & other)
  {
    handle_ = other.handle_;
    name_ = other.name_;
    p_context_ = other.p_context_;
    kernels_ = other.kernels_;
    return *this;
  }

  viennacl::ocl::context const * p_context() const { return p_context_; }

  std::string const & name() const { return name_; }

  /** @brief Adds a kernel to the program */
  inline viennacl::ocl::kernel & add_kernel(cl_kernel kernel_handle, std::string const & kernel_name);   //see context.hpp for implementation

  /** @brief Returns the kernel with the provided name */
  inline viennacl::ocl::kernel & get_kernel(std::string const & name);    //see context.hpp for implementation

  const viennacl::ocl::handle<cl_program> & handle() const { return handle_; }

private:

  viennacl::ocl::handle<cl_program> handle_;
  viennacl::ocl::context const * p_context_;
  std::string name_;
  kernel_container_type kernels_;
};

} //namespace ocl
} //namespace viennacl


#endif
