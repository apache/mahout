#ifndef VIENNACL_BACKEND_UTIL_HPP
#define VIENNACL_BACKEND_UTIL_HPP

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

/** @file viennacl/backend/util.hpp
    @brief Helper functionality for working with different memory domains
*/

#include <vector>
#include <cassert>

#include "viennacl/forwards.h"
#include "viennacl/backend/mem_handle.hpp"

#ifdef VIENNACL_WITH_OPENCL
#include "viennacl/backend/opencl.hpp"
#endif


namespace viennacl
{
namespace backend
{
namespace detail
{

  /** @brief Helper struct for converting a type to its OpenCL pendant. */
  template<typename T>
  struct convert_to_opencl
  {
    typedef T    type;
    enum { special = 0 };
  };

#ifdef VIENNACL_WITH_OPENCL
  template<>
  struct convert_to_opencl<unsigned int>
  {
    typedef cl_uint    type;
    //enum { special = (sizeof(unsigned int) != sizeof(cl_uint)) };
    enum { special = 1 };
  };

  template<>
  struct convert_to_opencl<int>
  {
    typedef cl_int    type;
    //enum { special = (sizeof(int) != sizeof(cl_int)) };
    enum { special = 1 };
  };


  template<>
  struct convert_to_opencl<unsigned long>
  {
    typedef cl_ulong    type;
    //enum { special = (sizeof(unsigned long) != sizeof(cl_ulong)) };
    enum { special = 1 };
  };

  template<>
  struct convert_to_opencl<long>
  {
    typedef cl_long    type;
    //enum { special = (sizeof(long) != sizeof(cl_long)) };
    enum { special = 1 };
  };
#endif


} //namespace detail


/** @brief Helper class implementing an array on the host. Default case: No conversion necessary */
template<typename T, bool special = detail::convert_to_opencl<T>::special>
class typesafe_host_array
{
  typedef T                                              cpu_type;
  typedef typename detail::convert_to_opencl<T>::type    target_type;

public:
  explicit typesafe_host_array() : bytes_buffer_(NULL), buffer_size_(0) {}

  explicit typesafe_host_array(mem_handle const & handle, vcl_size_t num = 0) : bytes_buffer_(NULL), buffer_size_(sizeof(cpu_type) * num)
  {
    resize(handle, num);
  }

  ~typesafe_host_array() { delete[] bytes_buffer_; }

  //
  // Setter and Getter
  //
  void * get() { return reinterpret_cast<void *>(bytes_buffer_); }
  vcl_size_t raw_size() const { return buffer_size_; }
  vcl_size_t element_size() const  {  return sizeof(cpu_type); }
  vcl_size_t size() const { return buffer_size_ / element_size(); }
  template<typename U>
  void set(vcl_size_t index, U value)
  {
    reinterpret_cast<cpu_type *>(bytes_buffer_)[index] = static_cast<cpu_type>(value);
  }

  //
  // Resize functionality
  //

  /** @brief Resize without initializing the new memory */
  void raw_resize(mem_handle const & /*handle*/, vcl_size_t num)
  {
    buffer_size_ = sizeof(cpu_type) * num;

    if (num > 0)
    {
      delete[] bytes_buffer_;

      bytes_buffer_ = new char[buffer_size_];
    }
  }

  /** @brief Resize including initialization of new memory (cf. std::vector<>) */
  void resize(mem_handle const & handle, vcl_size_t num)
  {
    raw_resize(handle, num);

    if (num > 0)
    {
      for (vcl_size_t i=0; i<buffer_size_; ++i)
        bytes_buffer_[i] = 0;
    }
  }

  cpu_type operator[](vcl_size_t index) const
  {
    assert(index < size() && bool("index out of bounds"));

    return reinterpret_cast<cpu_type *>(bytes_buffer_)[index];
  }

private:
  char * bytes_buffer_;
  vcl_size_t buffer_size_;
};




/** @brief Special host array type for conversion between OpenCL types and pure CPU types */
template<typename T>
class typesafe_host_array<T, true>
{
  typedef T                                              cpu_type;
  typedef typename detail::convert_to_opencl<T>::type    target_type;

public:
  explicit typesafe_host_array() : convert_to_opencl_( (default_memory_type() == OPENCL_MEMORY) ? true : false), bytes_buffer_(NULL), buffer_size_(0) {}

  explicit typesafe_host_array(mem_handle const & handle, vcl_size_t num = 0) : convert_to_opencl_(false), bytes_buffer_(NULL), buffer_size_(sizeof(cpu_type) * num)
  {
    resize(handle, num);
  }

  ~typesafe_host_array() { delete[] bytes_buffer_; }

  //
  // Setter and Getter
  //

  template<typename U>
  void set(vcl_size_t index, U value)
  {
#ifdef VIENNACL_WITH_OPENCL
    if (convert_to_opencl_)
      reinterpret_cast<target_type *>(bytes_buffer_)[index] = static_cast<target_type>(value);
    else
#endif
      reinterpret_cast<cpu_type *>(bytes_buffer_)[index] = static_cast<cpu_type>(value);
  }

  void * get() { return reinterpret_cast<void *>(bytes_buffer_); }
  cpu_type operator[](vcl_size_t index) const
  {
    assert(index < size() && bool("index out of bounds"));
#ifdef VIENNACL_WITH_OPENCL
    if (convert_to_opencl_)
      return static_cast<cpu_type>(reinterpret_cast<target_type *>(bytes_buffer_)[index]);
#endif
    return reinterpret_cast<cpu_type *>(bytes_buffer_)[index];
  }

  vcl_size_t raw_size() const { return buffer_size_; }
  vcl_size_t element_size() const
  {
#ifdef VIENNACL_WITH_OPENCL
    if (convert_to_opencl_)
      return sizeof(target_type);
#endif
    return sizeof(cpu_type);
  }
  vcl_size_t size() const { return buffer_size_ / element_size(); }

  //
  // Resize functionality
  //

  /** @brief Resize without initializing the new memory */
  void raw_resize(mem_handle const & handle, vcl_size_t num)
  {
    buffer_size_ = sizeof(cpu_type) * num;
    (void)handle; //silence unused variable warning if compiled without OpenCL support

#ifdef VIENNACL_WITH_OPENCL
    memory_types mem_type = handle.get_active_handle_id();
    if (mem_type == MEMORY_NOT_INITIALIZED)
      mem_type = default_memory_type();

    if (mem_type == OPENCL_MEMORY)
    {
      convert_to_opencl_ = true;
      buffer_size_ = sizeof(target_type) * num;
    }
#endif

    if (num > 0)
    {
      delete[] bytes_buffer_;

      bytes_buffer_ = new char[buffer_size_];
    }
  }

  /** @brief Resize including initialization of new memory (cf. std::vector<>) */
  void resize(mem_handle const & handle, vcl_size_t num)
  {
    raw_resize(handle, num);

    if (num > 0)
    {
      for (vcl_size_t i=0; i<buffer_size_; ++i)
        bytes_buffer_[i] = 0;
    }
  }

private:
  bool convert_to_opencl_;
  char * bytes_buffer_;
  vcl_size_t buffer_size_;
};

} //backend
} //viennacl
#endif
