#ifndef VIENNACL_BACKEND_OPENCL_HPP_
#define VIENNACL_BACKEND_OPENCL_HPP_

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

/** @file viennacl/backend/opencl.hpp
    @brief Implementations for the OpenCL backend functionality
*/


#include <vector>
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/backend.hpp"

namespace viennacl
{
namespace backend
{
namespace opencl
{

// Requirements for backend:

// * memory_create(size, host_ptr)
// * memory_copy(src, dest, offset_src, offset_dest, size)
// * memory_write_from_main_memory(src, offset, size,
//                                 dest, offset, size)
// * memory_read_to_main_memory(src, offset, size
//                              dest, offset, size)
// *
//

/** @brief Creates an array of the specified size in the current OpenCL context. If the second argument is provided, the buffer is initialized with data from that pointer.
 *
 * @param size_in_bytes   Number of bytes to allocate
 * @param host_ptr        Pointer to data which will be copied to the new array. Must point to at least 'size_in_bytes' bytes of data.
 * @param ctx             Optional context in which the matrix is created (one out of multiple OpenCL contexts, CUDA, host)
 *
 */
inline cl_mem memory_create(viennacl::ocl::context const & ctx, vcl_size_t size_in_bytes, const void * host_ptr = NULL)
{
  //std::cout << "Creating buffer (" << size_in_bytes << " bytes) host buffer " << host_ptr << " in context " << &ctx << std::endl;
  return ctx.create_memory_without_smart_handle(CL_MEM_READ_WRITE, static_cast<unsigned int>(size_in_bytes), const_cast<void *>(host_ptr));
}

/** @brief Copies 'bytes_to_copy' bytes from address 'src_buffer + src_offset' in the OpenCL context to memory starting at address 'dst_buffer + dst_offset' in the same OpenCL context.
 *
 *  @param src_buffer     A smart pointer to the begin of an allocated OpenCL buffer
 *  @param dst_buffer     A smart pointer to the end of an allocated OpenCL buffer
 *  @param src_offset     Offset of the first byte to be written from the address given by 'src_buffer' (in bytes)
 *  @param dst_offset     Offset of the first byte to be written to the address given by 'dst_buffer' (in bytes)
 *  @param bytes_to_copy  Number of bytes to be copied
 */
inline void memory_copy(viennacl::ocl::handle<cl_mem> const & src_buffer,
                        viennacl::ocl::handle<cl_mem> & dst_buffer,
                        vcl_size_t src_offset,
                        vcl_size_t dst_offset,
                        vcl_size_t bytes_to_copy)
{
  assert( &src_buffer.context() == &dst_buffer.context() && bool("Transfer between memory buffers in different contexts not supported yet!"));

  viennacl::ocl::context & memory_context = const_cast<viennacl::ocl::context &>(src_buffer.context());
  cl_int err = clEnqueueCopyBuffer(memory_context.get_queue().handle().get(),
                                   src_buffer.get(),
                                   dst_buffer.get(),
                                   src_offset,
                                   dst_offset,
                                   bytes_to_copy,
                                   0, NULL, NULL);  //events
  VIENNACL_ERR_CHECK(err);
}


/** @brief Writes data from main RAM identified by 'ptr' to the OpenCL buffer identified by 'dst_buffer'
 *
 * @param dst_buffer    A smart pointer to the beginning of an allocated OpenCL buffer
 * @param dst_offset    Offset of the first written byte from the beginning of 'dst_buffer' (in bytes)
 * @param bytes_to_copy Number of bytes to be copied
 * @param ptr           Pointer to the first byte to be written
 * @param async         Whether the operation should be asynchronous
 */
inline void memory_write(viennacl::ocl::handle<cl_mem> & dst_buffer,
                         vcl_size_t dst_offset,
                         vcl_size_t bytes_to_copy,
                         const void * ptr,
                         bool async = false)
{

  viennacl::ocl::context & memory_context = const_cast<viennacl::ocl::context &>(dst_buffer.context());

#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_DEVICE)
  std::cout << "Writing data (" << bytes_to_copy << " bytes, offset " << dst_offset << ") to OpenCL buffer " << dst_buffer.get() << " with queue " << memory_context.get_queue().handle().get() << " from " << ptr << std::endl;
#endif

  cl_int err = clEnqueueWriteBuffer(memory_context.get_queue().handle().get(),
                                    dst_buffer.get(),
                                    async ? CL_FALSE : CL_TRUE,             //blocking
                                    dst_offset,
                                    bytes_to_copy,
                                    ptr,
                                    0, NULL, NULL);      //events
  VIENNACL_ERR_CHECK(err);
}


/** @brief Reads data from an OpenCL buffer back to main RAM.
 *
 * @param src_buffer         A smart pointer to the beginning of an allocated OpenCL source buffer
 * @param src_offset         Offset of the first byte to be read from the beginning of src_buffer (in bytes_
 * @param bytes_to_copy      Number of bytes to be read
 * @param ptr                Location in main RAM where to read data should be written to
 * @param async         Whether the operation should be asynchronous
 */
inline void memory_read(viennacl::ocl::handle<cl_mem> const & src_buffer,
                        vcl_size_t src_offset,
                        vcl_size_t bytes_to_copy,
                        void * ptr,
                        bool async = false)
{
  //std::cout << "Reading data (" << bytes_to_copy << " bytes, offset " << src_offset << ") from OpenCL buffer " << src_buffer.get() << " to " << ptr << std::endl;
  viennacl::ocl::context & memory_context = const_cast<viennacl::ocl::context &>(src_buffer.context());
  cl_int err =  clEnqueueReadBuffer(memory_context.get_queue().handle().get(),
                                    src_buffer.get(),
                                    async ? CL_FALSE : CL_TRUE,             //blocking
                                    src_offset,
                                    bytes_to_copy,
                                    ptr,
                                    0, NULL, NULL);      //events
  VIENNACL_ERR_CHECK(err);
}


}
} //backend
} //viennacl
#endif
