#ifndef VIENNACL_BACKEND_CUDA_HPP_
#define VIENNACL_BACKEND_CUDA_HPP_

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

/** @file viennacl/backend/cuda.hpp
    @brief Implementations for the CUDA backend functionality
*/


#include <iostream>
#include <vector>
#include <cassert>
#include <stdexcept>
#include <sstream>

#include "viennacl/forwards.h"
#include "viennacl/tools/shared_ptr.hpp"

// includes CUDA
#include <cuda_runtime.h>

#define VIENNACL_CUDA_ERROR_CHECK(err)  detail::cuda_error_check (err, __FILE__, __LINE__)

namespace viennacl
{
namespace backend
{
namespace cuda
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

class cuda_exception : public std::runtime_error
{
public:
  cuda_exception(std::string const & what_arg, cudaError_t err_code) : std::runtime_error(what_arg), error_code_(err_code) {}

  cudaError_t error_code() const { return error_code_; }

private:
  cudaError_t error_code_;
};

namespace detail
{

  inline void cuda_error_check(cudaError error_code, const char *file, const int line )
  {
    if (cudaSuccess != error_code)
    {
      std::stringstream ss;
      ss << file << "(" << line << "): " << ": CUDA Runtime API error " << error_code << ": " << cudaGetErrorString( error_code ) << std::endl;
      throw viennacl::backend::cuda::cuda_exception(ss.str(), error_code);
    }
  }


  /** @brief Functor for deleting a CUDA handle. Used within the smart pointer class. */
  template<typename U>
  struct cuda_deleter
  {
    void operator()(U * p) const
    {
      //std::cout << "Freeing handle " << reinterpret_cast<void *>(p) << std::endl;
      cudaFree(p);
    }
  };

}

/** @brief Creates an array of the specified size on the CUDA device. If the second argument is provided, the buffer is initialized with data from that pointer.
 *
 * @param size_in_bytes   Number of bytes to allocate
 * @param host_ptr        Pointer to data which will be copied to the new array. Must point to at least 'size_in_bytes' bytes of data.
 *
 */
inline handle_type  memory_create(vcl_size_t size_in_bytes, const void * host_ptr = NULL)
{
  void * dev_ptr = NULL;
  VIENNACL_CUDA_ERROR_CHECK( cudaMalloc(&dev_ptr, size_in_bytes) );
  //std::cout << "Allocated new dev_ptr " << dev_ptr << " of size " <<  size_in_bytes << std::endl;

  if (!host_ptr)
    return handle_type(reinterpret_cast<char *>(dev_ptr), detail::cuda_deleter<char>());

  handle_type new_handle(reinterpret_cast<char*>(dev_ptr), detail::cuda_deleter<char>());

  // copy data:
  //std::cout << "Filling new handle from host_ptr " << host_ptr << std::endl;
  cudaMemcpy(new_handle.get(), host_ptr, size_in_bytes, cudaMemcpyHostToDevice);

  return new_handle;
}


/** @brief Copies 'bytes_to_copy' bytes from address 'src_buffer + src_offset' on the CUDA device to memory starting at address 'dst_buffer + dst_offset' on the same CUDA device.
 *
 *  @param src_buffer     A smart pointer to the begin of an allocated CUDA buffer
 *  @param dst_buffer     A smart pointer to the end of an allocated CUDA buffer
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

  cudaMemcpy(reinterpret_cast<void *>(dst_buffer.get() + dst_offset),
             reinterpret_cast<void *>(src_buffer.get() + src_offset),
             bytes_to_copy,
             cudaMemcpyDeviceToDevice);
}


/** @brief Writes data from main RAM identified by 'ptr' to the CUDA buffer identified by 'dst_buffer'
 *
 * @param dst_buffer    A smart pointer to the beginning of an allocated CUDA buffer
 * @param dst_offset    Offset of the first written byte from the beginning of 'dst_buffer' (in bytes)
 * @param bytes_to_copy Number of bytes to be copied
 * @param ptr           Pointer to the first byte to be written
 * @param async              Whether the operation should be asynchronous
 */
inline void memory_write(handle_type & dst_buffer,
                         vcl_size_t dst_offset,
                         vcl_size_t bytes_to_copy,
                         const void * ptr,
                         bool async = false)
{
  assert( (dst_buffer.get() != NULL) && bool("Memory not initialized!"));

  if (async)
    cudaMemcpyAsync(reinterpret_cast<char *>(dst_buffer.get()) + dst_offset,
                    reinterpret_cast<const char *>(ptr),
                    bytes_to_copy,
                    cudaMemcpyHostToDevice);
  else
    cudaMemcpy(reinterpret_cast<char *>(dst_buffer.get()) + dst_offset,
               reinterpret_cast<const char *>(ptr),
               bytes_to_copy,
               cudaMemcpyHostToDevice);
}


/** @brief Reads data from a CUDA buffer back to main RAM.
 *
 * @param src_buffer         A smart pointer to the beginning of an allocated CUDA source buffer
 * @param src_offset         Offset of the first byte to be read from the beginning of src_buffer (in bytes_
 * @param bytes_to_copy      Number of bytes to be read
 * @param ptr                Location in main RAM where to read data should be written to
 * @param async              Whether the operation should be asynchronous
 */
inline void memory_read(handle_type const & src_buffer,
                        vcl_size_t src_offset,
                        vcl_size_t bytes_to_copy,
                        void * ptr,
                        bool async = false)
{
  assert( (src_buffer.get() != NULL) && bool("Memory not initialized!"));

  if (async)
    cudaMemcpyAsync(reinterpret_cast<char *>(ptr),
                    reinterpret_cast<char *>(src_buffer.get()) + src_offset,
                    bytes_to_copy,
                    cudaMemcpyDeviceToHost);
  else
    cudaMemcpy(reinterpret_cast<char *>(ptr),
               reinterpret_cast<char *>(src_buffer.get()) + src_offset,
               bytes_to_copy,
               cudaMemcpyDeviceToHost);
}

} //cuda
} //backend
} //viennacl
#endif
