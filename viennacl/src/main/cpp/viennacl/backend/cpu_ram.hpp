#ifndef VIENNACL_BACKEND_CPU_RAM_HPP_
#define VIENNACL_BACKEND_CPU_RAM_HPP_

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

/** @file viennacl/backend/cpu_ram.hpp
    @brief Implementations for the OpenCL backend functionality
*/

#include <cassert>
#include <vector>
#ifdef VIENNACL_WITH_AVX2
#include <stdlib.h>
#endif

#include "viennacl/forwards.h"
#include "viennacl/tools/shared_ptr.hpp"

namespace viennacl
{
namespace backend
{
namespace cpu_ram
{
typedef viennacl::tools::shared_ptr<char>  handle_type;
// Requirements for backend:

// * memory_create(size, host_ptr)
// * memory_copy(src, dest, offset_src, offset_dest, size)
// * memory_write_from_main_memory(src, offset, size,
//                                 dest, offset, size)
// * memory_read_to_main_memory(src, offset, size
//                              dest, offset, size)
// *
//

namespace detail
{
  /** @brief Helper struct for deleting an pointer to an array */
  template<class U>
  struct array_deleter
  {
#ifdef VIENNACL_WITH_AVX2
    void operator()(U* p) const { free(p); }
#else
    void operator()(U* p) const { delete[] p; }
#endif
  };

}

/** @brief Creates an array of the specified size in main RAM. If the second argument is provided, the buffer is initialized with data from that pointer.
 *
 * @param size_in_bytes   Number of bytes to allocate
 * @param host_ptr        Pointer to data which will be copied to the new array. Must point to at least 'size_in_bytes' bytes of data.
 *
 */
inline handle_type  memory_create(vcl_size_t size_in_bytes, const void * host_ptr = NULL)
{
#ifdef VIENNACL_WITH_AVX2
  // Note: aligned_alloc not available on all compilers. Consider platform-specific alternatives such as posix_memalign()
  if (!host_ptr)
    return handle_type(reinterpret_cast<char*>(aligned_alloc(32, size_in_bytes)), detail::array_deleter<char>());

  handle_type new_handle(reinterpret_cast<char*>(aligned_alloc(32, size_in_bytes)), detail::array_deleter<char>());
#else
  if (!host_ptr)
    return handle_type(new char[size_in_bytes], detail::array_deleter<char>());

  handle_type new_handle(new char[size_in_bytes], detail::array_deleter<char>());
#endif

  // copy data:
  char * raw_ptr = new_handle.get();
  const char * data_ptr = static_cast<const char *>(host_ptr);
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
  for (long i=0; i<long(size_in_bytes); ++i)
    raw_ptr[i] = data_ptr[i];

  return new_handle;
}

/** @brief Copies 'bytes_to_copy' bytes from address 'src_buffer + src_offset' to memory starting at address 'dst_buffer + dst_offset'.
 *
 *  @param src_buffer     A smart pointer to the begin of an allocated buffer
 *  @param dst_buffer     A smart pointer to the end of an allocated buffer
 *  @param src_offset     Offset of the first byte to be written from the address given by 'src_buffer' (in bytes)
 *  @param dst_offset     Offset of the first byte to be written to the address given by 'dst_buffer' (in bytes)
 *  @param bytes_to_copy  Number of bytes to be copied
 */
inline void memory_copy(handle_type const & src_buffer,
                        handle_type & dst_buffer,
                        vcl_size_t src_offset,
                        vcl_size_t dst_offset,
                        vcl_size_t bytes_to_copy)
{
  assert( (dst_buffer.get() != NULL) && bool("Memory not initialized!"));
  assert( (src_buffer.get() != NULL) && bool("Memory not initialized!"));

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i=0; i<long(bytes_to_copy); ++i)
    dst_buffer.get()[vcl_size_t(i)+dst_offset] = src_buffer.get()[vcl_size_t(i) + src_offset];
}

/** @brief Writes data from main RAM identified by 'ptr' to the buffer identified by 'dst_buffer'
 *
 * @param dst_buffer    A smart pointer to the beginning of an allocated buffer
 * @param dst_offset    Offset of the first written byte from the beginning of 'dst_buffer' (in bytes)
 * @param bytes_to_copy Number of bytes to be copied
 * @param ptr           Pointer to the first byte to be written
 */
inline void memory_write(handle_type & dst_buffer,
                         vcl_size_t dst_offset,
                         vcl_size_t bytes_to_copy,
                         const void * ptr,
                         bool /*async*/)
{
  assert( (dst_buffer.get() != NULL) && bool("Memory not initialized!"));

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i=0; i<long(bytes_to_copy); ++i)
    dst_buffer.get()[vcl_size_t(i)+dst_offset] = static_cast<const char *>(ptr)[i];
}

/** @brief Reads data from a buffer back to main RAM.
 *
 * @param src_buffer         A smart pointer to the beginning of an allocated source buffer
 * @param src_offset         Offset of the first byte to be read from the beginning of src_buffer (in bytes_
 * @param bytes_to_copy      Number of bytes to be read
 * @param ptr                Location in main RAM where to read data should be written to
 */
inline void memory_read(handle_type const & src_buffer,
                        vcl_size_t src_offset,
                        vcl_size_t bytes_to_copy,
                        void * ptr,
                        bool /*async*/)
{
  assert( (src_buffer.get() != NULL) && bool("Memory not initialized!"));

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i=0; i<long(bytes_to_copy); ++i)
    static_cast<char *>(ptr)[i] = src_buffer.get()[vcl_size_t(i)+src_offset];
}

}
} //backend
} //viennacl
#endif
