#ifndef VIENNACL_BACKEND_MEMORY_HPP
#define VIENNACL_BACKEND_MEMORY_HPP

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

/** @file viennacl/backend/memory.hpp
    @brief Main interface routines for memory management
*/

#include <vector>
#include <cassert>
#include "viennacl/forwards.h"
#include "viennacl/backend/mem_handle.hpp"
#include "viennacl/context.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/context.hpp"
#include "viennacl/backend/util.hpp"

#include "viennacl/backend/cpu_ram.hpp"

#ifdef VIENNACL_WITH_OPENCL
#include "viennacl/backend/opencl.hpp"
#include "viennacl/ocl/backend.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
#include "viennacl/backend/cuda.hpp"
#endif


namespace viennacl
{
namespace backend
{


  // if a user compiles with CUDA, it is reasonable to expect that CUDA should be the default
  /** @brief Synchronizes the execution. finish() will only return after all compute kernels (CUDA, OpenCL) have completed. */
  inline void finish()
  {
#ifdef VIENNACL_WITH_CUDA
    cudaDeviceSynchronize();
#endif
#ifdef VIENNACL_WITH_OPENCL
    viennacl::ocl::get_queue().finish();
#endif
  }




  // Requirements for backend:

  // ---- Memory ----
  //
  // * memory_create(size, host_ptr)
  // * memory_copy(src, dest, offset_src, offset_dest, size)
  // * memory_write(src, offset, size, ptr)
  // * memory_read(src, offset, size, ptr)
  //

  /** @brief Creates an array of the specified size. If the second argument is provided, the buffer is initialized with data from that pointer.
  *
  * This is the generic version for CPU RAM, CUDA, and OpenCL. Creates the memory in the currently active memory domain.
  *
  * @param handle          The generic wrapper handle for multiple memory domains which will hold the new buffer.
  * @param size_in_bytes   Number of bytes to allocate
  * @param ctx             Optional context in which the matrix is created (one out of multiple OpenCL contexts, CUDA, host)
  * @param host_ptr        Pointer to data which will be copied to the new array. Must point to at least 'size_in_bytes' bytes of data.
  *
  */
  inline void memory_create(mem_handle & handle, vcl_size_t size_in_bytes, viennacl::context const & ctx, const void * host_ptr = NULL)
  {
    if (size_in_bytes > 0)
    {
      if (handle.get_active_handle_id() == MEMORY_NOT_INITIALIZED)
        handle.switch_active_handle_id(ctx.memory_type());

      switch (handle.get_active_handle_id())
      {
      case MAIN_MEMORY:
        handle.ram_handle() = cpu_ram::memory_create(size_in_bytes, host_ptr);
        handle.raw_size(size_in_bytes);
        break;
#ifdef VIENNACL_WITH_OPENCL
      case OPENCL_MEMORY:
        handle.opencl_handle().context(ctx.opencl_context());
        handle.opencl_handle() = opencl::memory_create(handle.opencl_handle().context(), size_in_bytes, host_ptr);
        handle.raw_size(size_in_bytes);
        break;
#endif
#ifdef VIENNACL_WITH_CUDA
      case CUDA_MEMORY:
        handle.cuda_handle() = cuda::memory_create(size_in_bytes, host_ptr);
        handle.raw_size(size_in_bytes);
        break;
#endif
      case MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("unknown memory handle!");
      }
    }
  }

  /*
  inline void memory_create(mem_handle & handle, vcl_size_t size_in_bytes, const void * host_ptr = NULL)
  {
    viennacl::context  ctx(default_memory_type());
    memory_create(handle, size_in_bytes, ctx, host_ptr);
  }*/


  /** @brief Copies 'bytes_to_copy' bytes from address 'src_buffer + src_offset' to memory starting at address 'dst_buffer + dst_offset'.
  *
  * This is the generic version for CPU RAM, CUDA, and OpenCL. Copies the memory in the currently active memory domain.
  *
  *
  *  @param src_buffer     A smart pointer to the begin of an allocated buffer
  *  @param dst_buffer     A smart pointer to the end of an allocated buffer
  *  @param src_offset     Offset of the first byte to be written from the address given by 'src_buffer' (in bytes)
  *  @param dst_offset     Offset of the first byte to be written to the address given by 'dst_buffer' (in bytes)
  *  @param bytes_to_copy  Number of bytes to be copied
  */
  inline void memory_copy(mem_handle const & src_buffer,
                          mem_handle & dst_buffer,
                          vcl_size_t src_offset,
                          vcl_size_t dst_offset,
                          vcl_size_t bytes_to_copy)
  {
    assert( src_buffer.get_active_handle_id() == dst_buffer.get_active_handle_id() && bool("memory_copy() must be called on buffers from the same domain") );

    if (bytes_to_copy > 0)
    {
      switch (src_buffer.get_active_handle_id())
      {
      case MAIN_MEMORY:
        cpu_ram::memory_copy(src_buffer.ram_handle(), dst_buffer.ram_handle(), src_offset, dst_offset, bytes_to_copy);
        break;
#ifdef VIENNACL_WITH_OPENCL
      case OPENCL_MEMORY:
        opencl::memory_copy(src_buffer.opencl_handle(), dst_buffer.opencl_handle(), src_offset, dst_offset, bytes_to_copy);
        break;
#endif
#ifdef VIENNACL_WITH_CUDA
      case CUDA_MEMORY:
        cuda::memory_copy(src_buffer.cuda_handle(), dst_buffer.cuda_handle(), src_offset, dst_offset, bytes_to_copy);
        break;
#endif
      case MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("unknown memory handle!");
      }
    }
  }

  // TODO: Refine this concept. Maybe move to constructor?
  /** @brief A 'shallow' copy operation from an initialized buffer to an uninitialized buffer.
   * The uninitialized buffer just copies the raw handle.
   */
  inline void memory_shallow_copy(mem_handle const & src_buffer,
                                  mem_handle & dst_buffer)
  {
    assert( (dst_buffer.get_active_handle_id() == MEMORY_NOT_INITIALIZED) && bool("Shallow copy on already initialized memory not supported!"));

    switch (src_buffer.get_active_handle_id())
    {
    case MAIN_MEMORY:
      dst_buffer.switch_active_handle_id(src_buffer.get_active_handle_id());
      dst_buffer.ram_handle() = src_buffer.ram_handle();
      dst_buffer.raw_size(src_buffer.raw_size());
      break;
#ifdef VIENNACL_WITH_OPENCL
    case OPENCL_MEMORY:
      dst_buffer.switch_active_handle_id(src_buffer.get_active_handle_id());
      dst_buffer.opencl_handle() = src_buffer.opencl_handle();
      dst_buffer.raw_size(src_buffer.raw_size());
      break;
#endif
#ifdef VIENNACL_WITH_CUDA
    case CUDA_MEMORY:
      dst_buffer.switch_active_handle_id(src_buffer.get_active_handle_id());
      dst_buffer.cuda_handle() = src_buffer.cuda_handle();
      dst_buffer.raw_size(src_buffer.raw_size());
      break;
#endif
    case MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("unknown memory handle!");
    }
  }

  /** @brief Writes data from main RAM identified by 'ptr' to the buffer identified by 'dst_buffer'
  *
  * This is the generic version for CPU RAM, CUDA, and OpenCL. Writes the memory in the currently active memory domain.
  *
  * @param dst_buffer     A smart pointer to the beginning of an allocated buffer
  * @param dst_offset     Offset of the first written byte from the beginning of 'dst_buffer' (in bytes)
  * @param bytes_to_write Number of bytes to be written
  * @param ptr            Pointer to the first byte to be written
  * @param async              Whether the operation should be asynchronous
  */
  inline void memory_write(mem_handle & dst_buffer,
                           vcl_size_t dst_offset,
                           vcl_size_t bytes_to_write,
                           const void * ptr,
                           bool async = false)
  {
    if (bytes_to_write > 0)
    {
      switch (dst_buffer.get_active_handle_id())
      {
      case MAIN_MEMORY:
        cpu_ram::memory_write(dst_buffer.ram_handle(), dst_offset, bytes_to_write, ptr, async);
        break;
#ifdef VIENNACL_WITH_OPENCL
      case OPENCL_MEMORY:
        opencl::memory_write(dst_buffer.opencl_handle(), dst_offset, bytes_to_write, ptr, async);
        break;
#endif
#ifdef VIENNACL_WITH_CUDA
      case CUDA_MEMORY:
        cuda::memory_write(dst_buffer.cuda_handle(), dst_offset, bytes_to_write, ptr, async);
        break;
#endif
      case MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("unknown memory handle!");
      }
    }
  }

  /** @brief Reads data from a buffer back to main RAM.
  *
  * This is the generic version for CPU RAM, CUDA, and OpenCL. Reads the memory from the currently active memory domain.
  *
  * @param src_buffer         A smart pointer to the beginning of an allocated source buffer
  * @param src_offset         Offset of the first byte to be read from the beginning of src_buffer (in bytes_
  * @param bytes_to_read      Number of bytes to be read
  * @param ptr                Location in main RAM where to read data should be written to
  * @param async              Whether the operation should be asynchronous
  */
  inline void memory_read(mem_handle const & src_buffer,
                          vcl_size_t src_offset,
                          vcl_size_t bytes_to_read,
                          void * ptr,
                          bool async = false)
  {
    //finish(); //Fixes some issues with AMD APP SDK. However, might sacrifice a few percents of performance in some cases.

    if (bytes_to_read > 0)
    {
      switch (src_buffer.get_active_handle_id())
      {
      case MAIN_MEMORY:
        cpu_ram::memory_read(src_buffer.ram_handle(), src_offset, bytes_to_read, ptr, async);
        break;
#ifdef VIENNACL_WITH_OPENCL
      case OPENCL_MEMORY:
        opencl::memory_read(src_buffer.opencl_handle(), src_offset, bytes_to_read, ptr, async);
        break;
#endif
#ifdef VIENNACL_WITH_CUDA
      case CUDA_MEMORY:
        cuda::memory_read(src_buffer.cuda_handle(), src_offset, bytes_to_read, ptr, async);
        break;
#endif
      case MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("unknown memory handle!");
      }
    }
  }



  namespace detail
  {
    template<typename T>
    vcl_size_t element_size(memory_types /* mem_type */)
    {
      return sizeof(T);
    }


    template<>
    inline vcl_size_t element_size<unsigned long>(memory_types
                                            #ifdef VIENNACL_WITH_OPENCL
                                                  mem_type  //in order to compile cleanly at -Wextra in GCC
                                            #endif
                                                  )
    {
#ifdef VIENNACL_WITH_OPENCL
      if (mem_type == OPENCL_MEMORY)
        return sizeof(cl_ulong);
#endif
      return sizeof(unsigned long);
    }

    template<>
    inline vcl_size_t element_size<long>(memory_types
                                   #ifdef VIENNACL_WITH_OPENCL
                                         mem_type  //in order to compile cleanly at -Wextra in GCC
                                   #endif
                                         )
    {
#ifdef VIENNACL_WITH_OPENCL
      if (mem_type == OPENCL_MEMORY)
        return sizeof(cl_long);
#endif
      return sizeof(long);
    }


    template<>
    inline vcl_size_t element_size<unsigned int>(memory_types
                                           #ifdef VIENNACL_WITH_OPENCL
                                                 mem_type  //in order to compile cleanly at -Wextra in GCC
                                           #endif
                                                 )
    {
#ifdef VIENNACL_WITH_OPENCL
      if (mem_type == OPENCL_MEMORY)
        return sizeof(cl_uint);
#endif
      return sizeof(unsigned int);
    }

    template<>
    inline vcl_size_t element_size<int>(memory_types
                                  #ifdef VIENNACL_WITH_OPENCL
                                        mem_type  //in order to compile cleanly at -Wextra in GCC
                                  #endif
                                        )
    {
#ifdef VIENNACL_WITH_OPENCL
      if (mem_type == OPENCL_MEMORY)
        return sizeof(cl_int);
#endif
      return sizeof(int);
    }


  }


  /** @brief Switches the active memory domain within a memory handle. Data is copied if the new active domain differs from the old one. Memory in the source handle is not free'd. */
  template<typename DataType>
  void switch_memory_context(mem_handle & handle, viennacl::context new_ctx)
  {
    if (handle.get_active_handle_id() == new_ctx.memory_type())
      return;

    if (handle.get_active_handle_id() == viennacl::MEMORY_NOT_INITIALIZED || handle.raw_size() == 0)
    {
      handle.switch_active_handle_id(new_ctx.memory_type());
#ifdef VIENNACL_WITH_OPENCL
      if (new_ctx.memory_type() == OPENCL_MEMORY)
        handle.opencl_handle().context(new_ctx.opencl_context());
#endif
      return;
    }

    vcl_size_t size_dst = detail::element_size<DataType>(handle.get_active_handle_id());
    vcl_size_t size_src = detail::element_size<DataType>(new_ctx.memory_type());

    if (size_dst != size_src)  // OpenCL data element size not the same as host data element size
    {
      throw memory_exception("Heterogeneous data element sizes not yet supported!");
    }
    else //no data conversion required
    {
      if (handle.get_active_handle_id() == MAIN_MEMORY) //we can access the existing data directly
      {
        switch (new_ctx.memory_type())
        {
#ifdef VIENNACL_WITH_OPENCL
        case OPENCL_MEMORY:
          handle.opencl_handle().context(new_ctx.opencl_context());
          handle.opencl_handle() = opencl::memory_create(handle.opencl_handle().context(), handle.raw_size(), handle.ram_handle().get());
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case CUDA_MEMORY:
          handle.cuda_handle() = cuda::memory_create(handle.raw_size(), handle.ram_handle().get());
          break;
#endif
        case MAIN_MEMORY:
        default:
          throw memory_exception("Invalid destination domain");
        }
      }
#ifdef VIENNACL_WITH_OPENCL
      else if (handle.get_active_handle_id() == OPENCL_MEMORY) // data can be dumped into destination directly
      {
        std::vector<DataType> buffer;

        switch (new_ctx.memory_type())
        {
        case MAIN_MEMORY:
          handle.ram_handle() = cpu_ram::memory_create(handle.raw_size());
          opencl::memory_read(handle.opencl_handle(), 0, handle.raw_size(), handle.ram_handle().get());
          break;
#ifdef VIENNACL_WITH_CUDA
        case CUDA_MEMORY:
          buffer.resize(handle.raw_size() / sizeof(DataType));
          opencl::memory_read(handle.opencl_handle(), 0, handle.raw_size(), &(buffer[0]));
          cuda::memory_create(handle.cuda_handle(), handle.raw_size(), &(buffer[0]));
          break;
#endif
        default:
          throw memory_exception("Invalid destination domain");
        }
      }
#endif
#ifdef VIENNACL_WITH_CUDA
      else //CUDA_MEMORY
      {
        std::vector<DataType> buffer;

        // write
        switch (new_ctx.memory_type())
        {
        case MAIN_MEMORY:
          handle.ram_handle() = cpu_ram::memory_create(handle.raw_size());
          cuda::memory_read(handle.cuda_handle(), 0, handle.raw_size(), handle.ram_handle().get());
          break;
#ifdef VIENNACL_WITH_OPENCL
        case OPENCL_MEMORY:
          buffer.resize(handle.raw_size() / sizeof(DataType));
          cuda::memory_read(handle.cuda_handle(), 0, handle.raw_size(), &(buffer[0]));
          handle.opencl_handle() = opencl::memory_create(handle.raw_size(), &(buffer[0]));
          break;
#endif
        default:
          throw memory_exception("Unsupported source memory domain");
        }
      }
#endif

      // everything succeeded so far, now switch to new domain:
      handle.switch_active_handle_id(new_ctx.memory_type());

    } // no data conversion
  }



  /** @brief Copies data of the provided 'DataType' from 'handle_src' to 'handle_dst' and converts the data if the binary representation of 'DataType' among the memory domains differs. */
  template<typename DataType>
  void typesafe_memory_copy(mem_handle const & handle_src, mem_handle & handle_dst)
  {
    if (handle_dst.get_active_handle_id() == MEMORY_NOT_INITIALIZED)
      handle_dst.switch_active_handle_id(default_memory_type());

    vcl_size_t element_size_src = detail::element_size<DataType>(handle_src.get_active_handle_id());
    vcl_size_t element_size_dst = detail::element_size<DataType>(handle_dst.get_active_handle_id());

    if (element_size_src != element_size_dst)
    {
      // Data needs to be converted.

      typesafe_host_array<DataType> buffer_src(handle_src);
      typesafe_host_array<DataType> buffer_dst(handle_dst, handle_src.raw_size() / element_size_src);

      //
      // Step 1: Fill buffer_dst depending on where the data resides:
      //
      DataType const * src_data;
      switch (handle_src.get_active_handle_id())
      {
      case MAIN_MEMORY:
        src_data = reinterpret_cast<DataType const *>(handle_src.ram_handle().get());
        for (vcl_size_t i=0; i<buffer_dst.size(); ++i)
          buffer_dst.set(i, src_data[i]);
        break;

#ifdef VIENNACL_WITH_OPENCL
      case OPENCL_MEMORY:
        buffer_src.resize(handle_src, handle_src.raw_size() / element_size_src);
        opencl::memory_read(handle_src.opencl_handle(), 0, buffer_src.raw_size(), buffer_src.get());
        for (vcl_size_t i=0; i<buffer_dst.size(); ++i)
          buffer_dst.set(i, buffer_src[i]);
        break;
#endif
#ifdef VIENNACL_WITH_CUDA
      case CUDA_MEMORY:
        buffer_src.resize(handle_src, handle_src.raw_size() / element_size_src);
        cuda::memory_read(handle_src.cuda_handle(), 0, buffer_src.raw_size(), buffer_src.get());
        for (vcl_size_t i=0; i<buffer_dst.size(); ++i)
          buffer_dst.set(i, buffer_src[i]);
        break;
#endif

      default:
        throw memory_exception("unsupported memory domain");
      }

      //
      // Step 2: Write to destination
      //
      if (handle_dst.raw_size() == buffer_dst.raw_size())
        viennacl::backend::memory_write(handle_dst, 0, buffer_dst.raw_size(), buffer_dst.get());
      else
        viennacl::backend::memory_create(handle_dst, buffer_dst.raw_size(), viennacl::traits::context(handle_dst), buffer_dst.get());

    }
    else
    {
      // No data conversion required.
      typesafe_host_array<DataType> buffer(handle_src);

      switch (handle_src.get_active_handle_id())
      {
      case MAIN_MEMORY:
        switch (handle_dst.get_active_handle_id())
        {
        case MAIN_MEMORY:
        case OPENCL_MEMORY:
        case CUDA_MEMORY:
          if (handle_dst.raw_size() == handle_src.raw_size())
            viennacl::backend::memory_write(handle_dst, 0, handle_src.raw_size(), handle_src.ram_handle().get());
          else
            viennacl::backend::memory_create(handle_dst, handle_src.raw_size(), viennacl::traits::context(handle_dst), handle_src.ram_handle().get());
          break;

        default:
          throw memory_exception("unsupported destination memory domain");
        }
        break;

      case OPENCL_MEMORY:
        switch (handle_dst.get_active_handle_id())
        {
        case MAIN_MEMORY:
          if (handle_dst.raw_size() != handle_src.raw_size())
            viennacl::backend::memory_create(handle_dst, handle_src.raw_size(), viennacl::traits::context(handle_dst));
          viennacl::backend::memory_read(handle_src, 0, handle_src.raw_size(), handle_dst.ram_handle().get());
          break;

        case OPENCL_MEMORY:
          if (handle_dst.raw_size() != handle_src.raw_size())
            viennacl::backend::memory_create(handle_dst, handle_src.raw_size(), viennacl::traits::context(handle_dst));
          viennacl::backend::memory_copy(handle_src, handle_dst, 0, 0, handle_src.raw_size());
          break;

        case CUDA_MEMORY:
          if (handle_dst.raw_size() != handle_src.raw_size())
            viennacl::backend::memory_create(handle_dst, handle_src.raw_size(), viennacl::traits::context(handle_dst));
          buffer.resize(handle_src, handle_src.raw_size() / element_size_src);
          viennacl::backend::memory_read(handle_src, 0, handle_src.raw_size(), buffer.get());
          viennacl::backend::memory_write(handle_dst, 0, handle_src.raw_size(), buffer.get());
          break;

        default:
          throw memory_exception("unsupported destination memory domain");
        }
        break;

      case CUDA_MEMORY:
        switch (handle_dst.get_active_handle_id())
        {
        case MAIN_MEMORY:
          if (handle_dst.raw_size() != handle_src.raw_size())
            viennacl::backend::memory_create(handle_dst, handle_src.raw_size(), viennacl::traits::context(handle_dst));
          viennacl::backend::memory_read(handle_src, 0, handle_src.raw_size(), handle_dst.ram_handle().get());
          break;

        case OPENCL_MEMORY:
          if (handle_dst.raw_size() != handle_src.raw_size())
            viennacl::backend::memory_create(handle_dst, handle_src.raw_size(), viennacl::traits::context(handle_dst));
          buffer.resize(handle_src, handle_src.raw_size() / element_size_src);
          viennacl::backend::memory_read(handle_src, 0, handle_src.raw_size(), buffer.get());
          viennacl::backend::memory_write(handle_dst, 0, handle_src.raw_size(), buffer.get());
          break;

        case CUDA_MEMORY:
          if (handle_dst.raw_size() != handle_src.raw_size())
            viennacl::backend::memory_create(handle_dst, handle_src.raw_size(), viennacl::traits::context(handle_dst));
          viennacl::backend::memory_copy(handle_src, handle_dst, 0, 0, handle_src.raw_size());
          break;

        default:
          throw memory_exception("unsupported destination memory domain");
        }
        break;

      default:
        throw memory_exception("unsupported source memory domain");
      }

    }
  }


} //backend

//
// Convenience layer:
//
/** @brief Generic convenience routine for migrating data of an object to a new memory domain */
template<typename T>
void switch_memory_context(T & obj, viennacl::context new_ctx)
{
  obj.switch_memory_context(new_ctx);
}

} //viennacl
#endif
