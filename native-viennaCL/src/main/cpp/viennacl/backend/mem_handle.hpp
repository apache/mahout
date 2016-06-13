#ifndef VIENNACL_BACKEND_MEM_HANDLE_HPP
#define VIENNACL_BACKEND_MEM_HANDLE_HPP

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

/** @file viennacl/backend/mem_handle.hpp
    @brief Implements the multi-memory-domain handle
*/

#include <vector>
#include <cassert>
#include "viennacl/forwards.h"
#include "viennacl/tools/shared_ptr.hpp"
#include "viennacl/backend/cpu_ram.hpp"

#ifdef VIENNACL_WITH_OPENCL
#include "viennacl/backend/opencl.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
#include "viennacl/backend/cuda.hpp"
#endif


namespace viennacl
{
namespace backend
{

namespace detail
{
  /** @brief Singleton for managing the default memory type.
  *
  * @param new_mem_type    If NULL, returns the current memory type. Otherwise, sets the memory type to the provided value.
  */
  inline memory_types get_set_default_memory_type(memory_types * new_mem_type)
  {
    // if a user compiles with CUDA, it is reasonable to expect that CUDA should be the default
#ifdef VIENNACL_WITH_CUDA
    static memory_types mem_type = CUDA_MEMORY;
#elif defined(VIENNACL_WITH_OPENCL)
    static memory_types mem_type = OPENCL_MEMORY;
#else
    static memory_types mem_type = MAIN_MEMORY;
#endif

    if (new_mem_type)
      mem_type = *new_mem_type;

    return mem_type;
  }
}

/** @brief Returns the default memory type for the given configuration.
 *
 * CUDA has precedence over OpenCL, which has precedence over main memory. Depends on which VIENNACL_WITH_{CUDA/OPENCL/OPENMP} macros are defined.
 */
inline memory_types default_memory_type() { return detail::get_set_default_memory_type(NULL); }

/** @brief Sets the default memory type for the given configuration.
 *
 * Make sure the respective new memory type is enabled.
 * For example, passing CUDA_MEMORY if no CUDA backend is selected will result in exceptions being thrown as soon as you try to allocate buffers.
 */
inline memory_types default_memory_type(memory_types new_memory_type) { return detail::get_set_default_memory_type(&new_memory_type); }


/** @brief Main abstraction class for multiple memory domains. Represents a buffer in either main RAM, an OpenCL context, or a CUDA device.
 *
 * The idea is to wrap all possible handle types inside this class so that higher-level code does not need to be cluttered with preprocessor switches.
 * Instead, this class collects all the necessary conditional compilations.
 *
 */
class mem_handle
{
public:
  typedef viennacl::tools::shared_ptr<char>      ram_handle_type;
  typedef viennacl::tools::shared_ptr<char>      cuda_handle_type;

  /** @brief Default CTOR. No memory is allocated */
  mem_handle() : active_handle_(MEMORY_NOT_INITIALIZED), size_in_bytes_(0) {}

  /** @brief Returns the handle to a buffer in CPU RAM. NULL is returned if no such buffer has been allocated. */
  ram_handle_type       & ram_handle()       { return ram_handle_; }
  /** @brief Returns the handle to a buffer in CPU RAM. NULL is returned if no such buffer has been allocated. */
  ram_handle_type const & ram_handle() const { return ram_handle_; }

#ifdef VIENNACL_WITH_OPENCL
  /** @brief Returns the handle to an OpenCL buffer. The handle contains NULL if no such buffer has been allocated. */
  viennacl::ocl::handle<cl_mem>       & opencl_handle()       { return opencl_handle_; }
  /** @brief Returns the handle to an OpenCL buffer. The handle contains NULL if no such buffer has been allocated. */
  viennacl::ocl::handle<cl_mem> const & opencl_handle() const { return opencl_handle_; }
#endif

#ifdef VIENNACL_WITH_CUDA
  /** @brief Returns the handle to a CUDA buffer. The handle contains NULL if no such buffer has been allocated. */
  cuda_handle_type       & cuda_handle()       { return cuda_handle_; }
  /** @brief Returns the handle to a CUDA buffer. The handle contains NULL if no such buffer has been allocated. */
  cuda_handle_type const & cuda_handle() const { return cuda_handle_; }
#endif

  /** @brief Returns an ID for the currently active memory buffer. Other memory buffers might contain old or no data. */
  memory_types get_active_handle_id() const { return active_handle_; }

  /** @brief Switches the currently active handle. If no support for that backend is provided, an exception is thrown. */
  void switch_active_handle_id(memory_types new_id)
  {
    if (new_id != active_handle_)
    {
      if (active_handle_ == MEMORY_NOT_INITIALIZED)
        active_handle_ = new_id;
      else if (active_handle_ == MAIN_MEMORY)
      {
        active_handle_ = new_id;
      }
      else if (active_handle_ == OPENCL_MEMORY)
      {
#ifdef VIENNACL_WITH_OPENCL
        active_handle_ = new_id;
#else
        throw memory_exception("compiled without OpenCL suppport!");
#endif
      }
      else if (active_handle_ == CUDA_MEMORY)
      {
#ifdef VIENNACL_WITH_CUDA
        active_handle_ = new_id;
#else
        throw memory_exception("compiled without CUDA suppport!");
#endif
      }
      else
        throw memory_exception("invalid new memory region!");
    }
  }

  /** @brief Compares the two handles and returns true if the active memory handles in the two mem_handles point to the same buffer. */
  bool operator==(mem_handle const & other) const
  {
    if (active_handle_ != other.active_handle_)
      return false;

    switch (active_handle_)
    {
    case MAIN_MEMORY:
      return ram_handle_.get() == other.ram_handle_.get();
#ifdef VIENNACL_WITH_OPENCL
    case OPENCL_MEMORY:
      return opencl_handle_.get() == other.opencl_handle_.get();
#endif
#ifdef VIENNACL_WITH_CUDA
    case CUDA_MEMORY:
      return cuda_handle_.get() == other.cuda_handle_.get();
#endif
    default: break;
    }

    return false;
  }

  /** @brief Compares the two handles and returns true if the active memory handles in the two mem_handles point a buffer with inferior address
     * useful to store handles into a map, since they naturally have strong ordering
     */
  bool operator<(mem_handle const & other) const
  {
    if (active_handle_ != other.active_handle_)
      return false;

    switch (active_handle_)
    {
    case MAIN_MEMORY:
      return ram_handle_.get() < other.ram_handle_.get();
#ifdef VIENNACL_WITH_OPENCL
    case OPENCL_MEMORY:
      return opencl_handle_.get() < other.opencl_handle_.get();
#endif
#ifdef VIENNACL_WITH_CUDA
    case CUDA_MEMORY:
      return cuda_handle_.get() < other.cuda_handle_.get();
#endif
    default: break;
    }

    return false;
  }


  bool operator!=(mem_handle const & other) const { return !(*this == other); }

  /** @brief Implements a fast swapping method. No data is copied, only the handles are exchanged. */
  void swap(mem_handle & other)
  {
    // swap handle type:
    memory_types active_handle_tmp = other.active_handle_;
    other.active_handle_ = active_handle_;
    active_handle_ = active_handle_tmp;

    // swap ram handle:
    ram_handle_type ram_handle_tmp = other.ram_handle_;
    other.ram_handle_ = ram_handle_;
    ram_handle_ = ram_handle_tmp;

    // swap OpenCL handle:
#ifdef VIENNACL_WITH_OPENCL
    opencl_handle_.swap(other.opencl_handle_);
#endif
#ifdef VIENNACL_WITH_CUDA
    cuda_handle_type cuda_handle_tmp = other.cuda_handle_;
    other.cuda_handle_ = cuda_handle_;
    cuda_handle_ = cuda_handle_tmp;
#endif
  }

  /** @brief Returns the number of bytes of the currently active buffer */
  vcl_size_t raw_size() const               { return size_in_bytes_; }

  /** @brief Sets the size of the currently active buffer. Use with care! */
  void        raw_size(vcl_size_t new_size) { size_in_bytes_ = new_size; }

private:
  memory_types active_handle_;
  ram_handle_type ram_handle_;
#ifdef VIENNACL_WITH_OPENCL
  viennacl::ocl::handle<cl_mem> opencl_handle_;
#endif
#ifdef VIENNACL_WITH_CUDA
  cuda_handle_type        cuda_handle_;
#endif
  vcl_size_t size_in_bytes_;
};


} //backend
} //viennacl
#endif
