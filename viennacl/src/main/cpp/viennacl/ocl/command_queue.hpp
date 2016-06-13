#ifndef VIENNACL_OCL_COMMAND_QUEUE_HPP_
#define VIENNACL_OCL_COMMAND_QUEUE_HPP_

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

/** @file viennacl/ocl/command_queue.hpp
    @brief Implementations of command queue representations
*/

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <vector>
#include <string>
#include <sstream>
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/handle.hpp"

namespace viennacl
{
namespace ocl
{

/** @brief A class representing a command queue
*
*/
class command_queue
{
public:
  command_queue() {}
  command_queue(viennacl::ocl::handle<cl_command_queue> h) : handle_(h) {}

  //Copy constructor:
  command_queue(command_queue const & other)
  {
    handle_ = other.handle_;
  }

  //assignment operator:
  command_queue & operator=(command_queue const & other)
  {
    handle_ = other.handle_;
    return *this;
  }

  bool operator==(command_queue const & other) const
  {
    return handle_ == other.handle_;
  }

  /** @brief Waits until all kernels in the queue have finished their execution */
  void finish() const
  {
    clFinish(handle_.get());
  }

  /** @brief Waits until all kernels in the queue have started their execution */
  void flush() const
  {
    clFlush(handle_.get());
  }

  viennacl::ocl::handle<cl_command_queue> const & handle() const { return handle_; }
  viennacl::ocl::handle<cl_command_queue>       & handle()       { return handle_; }

private:

  viennacl::ocl::handle<cl_command_queue> handle_;
};

} //namespace ocl
} //namespace viennacl

#endif
