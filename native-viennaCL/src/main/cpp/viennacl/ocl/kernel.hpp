#ifndef VIENNACL_OCL_KERNEL_HPP_
#define VIENNACL_OCL_KERNEL_HPP_

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

/** @file viennacl/ocl/kernel.hpp
    @brief Representation of an OpenCL kernel in ViennaCL.
*/

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include "viennacl/ocl/forwards.h"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/program.hpp"
#include "viennacl/ocl/device.hpp"
#include "viennacl/ocl/local_mem.hpp"

namespace viennacl
{
  namespace ocl
  {
    /** @brief Helper class for packing four cl_uint numbers into a uint4 type for access inside an OpenCL kernel.
      *
      * Since the primary use is for dealing with ranges and strides, the four members are termed accordingly.
      */
    struct packed_cl_uint
    {
      /** @brief Starting value of the integer stride. */
      cl_uint start;
      /** @brief Increment between integers. */
      cl_uint stride;
      /** @brief Number of values in the stride. */
      cl_uint size;
      /** @brief Internal length of the buffer. Might be larger than 'size' due to padding. */
      cl_uint internal_size;
    };

    /** @brief Represents an OpenCL kernel within ViennaCL */
    class kernel
    {
      template<typename KernelType>
      friend void enqueue(KernelType & k, viennacl::ocl::command_queue const & queue);

    public:
      typedef vcl_size_t            size_type;

      kernel() : handle_(), p_program_(NULL), p_context_(NULL), name_()
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Creating kernel object (default CTOR): " << name_ << std::endl;
        #endif
      }

      kernel(cl_kernel kernel_handle, viennacl::ocl::program const & kernel_program, viennacl::ocl::context const & kernel_context, std::string const & name)
        : handle_(kernel_handle, kernel_context), p_program_(&kernel_program), p_context_(&kernel_context), name_(name)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Creating kernel object (full CTOR): " << name_ << std::endl;
        #endif
        set_work_size_defaults();
      }

      kernel(kernel const & other)
        : handle_(other.handle_), p_program_(other.p_program_), p_context_(other.p_context_), name_(other.name_)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Creating kernel object (Copy CTOR): " << name_ << std::endl;
        #endif
        local_work_size_[0] = other.local_work_size_[0];
        local_work_size_[1] = other.local_work_size_[1];
        local_work_size_[2] = other.local_work_size_[2];

        global_work_size_[0] = other.global_work_size_[0];
        global_work_size_[1] = other.global_work_size_[1];
        global_work_size_[2] = other.global_work_size_[2];
      }

      viennacl::ocl::kernel & operator=(const kernel & other)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Assigning kernel object: " << other.name_ << std::endl;
        #endif
        handle_ = other.handle_;
        p_program_ = other.p_program_;
        p_context_ = other.p_context_;
        name_ = other.name_;
        local_work_size_[0] = other.local_work_size_[0];
        local_work_size_[1] = other.local_work_size_[1];
        local_work_size_[2] = other.local_work_size_[2];
        global_work_size_[0] = other.global_work_size_[0];
        global_work_size_[1] = other.global_work_size_[1];
        global_work_size_[2] = other.global_work_size_[2];
        return *this;
      }

      /** @brief Sets a char argument at the provided position */
      void arg(unsigned int pos, cl_char val)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting char kernel argument " << val << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(cl_char), (void*)&val);
        VIENNACL_ERR_CHECK(err);
      }

      /** @brief Sets a char argument at the provided position */
      void arg(unsigned int pos, cl_uchar val)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting unsigned char kernel argument " << val << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(cl_uchar), (void*)&val);
        VIENNACL_ERR_CHECK(err);
      }

      /** @brief Sets a argument of type short at the provided position */
      void arg(unsigned int pos, cl_short val)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting short kernel argument " << val << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(cl_short), (void*)&val);
        VIENNACL_ERR_CHECK(err);
      }

      /** @brief Sets a argument of type unsigned short at the provided position */
      void arg(unsigned int pos, cl_ushort val)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting unsigned short kernel argument " << val << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(cl_ushort), (void*)&val);
        VIENNACL_ERR_CHECK(err);
      }


      /** @brief Sets an unsigned integer argument at the provided position */
      void arg(unsigned int pos, cl_uint val)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting unsigned int kernel argument " << val << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(cl_uint), (void*)&val);
        VIENNACL_ERR_CHECK(err);
      }

      /** @brief Sets four packed unsigned integers as argument at the provided position */
      void arg(unsigned int pos, packed_cl_uint val)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting packed_cl_uint kernel argument (" << val.start << ", " << val.stride << ", " << val.size << ", " << val.internal_size << ") at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(packed_cl_uint), (void*)&val);
        VIENNACL_ERR_CHECK(err);
      }

      /** @brief Sets a single precision floating point argument at the provided position */
      void arg(unsigned int pos, float val)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting floating point kernel argument " << val << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(float), (void*)&val);
        VIENNACL_ERR_CHECK(err);
      }

      /** @brief Sets a double precision floating point argument at the provided position */
      void arg(unsigned int pos, double val)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting double precision kernel argument " << val << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(double), (void*)&val);
        VIENNACL_ERR_CHECK(err);
      }

      /** @brief Sets an int argument at the provided position */
      void arg(unsigned int pos, cl_int val)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting int precision kernel argument " << val << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(cl_int), (void*)&val);
        VIENNACL_ERR_CHECK(err);
      }

      /** @brief Sets an unsigned long argument at the provided position */
      void arg(unsigned int pos, cl_ulong val)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting ulong precision kernel argument " << val << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(cl_ulong), (void*)&val);
        VIENNACL_ERR_CHECK(err);
      }

      /** @brief Sets an unsigned long argument at the provided position */
      void arg(unsigned int pos, cl_long val)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting long precision kernel argument " << val << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(cl_long), (void*)&val);
        VIENNACL_ERR_CHECK(err);
      }

      //generic handling: call .handle() member
      /** @brief Sets an OpenCL memory object at the provided position */
      template<class VCL_TYPE>
      void arg(unsigned int pos, VCL_TYPE const & val)
      {
        assert(&val.handle().opencl_handle().context() == &handle_.context() && bool("Kernel and memory object not in the same context!"));

        cl_mem temp = val.handle().opencl_handle().get();
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting generic kernel argument " << temp << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(cl_mem), (void*)&temp);
        VIENNACL_ERR_CHECK(err);
      }

      //forward handles directly:
      /** @brief Sets an OpenCL object at the provided position */
      template<class CL_TYPE>
      void arg(unsigned int pos, viennacl::ocl::handle<CL_TYPE> const & h)
      {
        CL_TYPE temp = h.get();
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting handle kernel argument " << temp << " at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, sizeof(CL_TYPE), (void*)&temp);
        VIENNACL_ERR_CHECK(err);
      }


      //local buffer argument:
      /** @brief Sets an OpenCL local memory object at the provided position */
      void arg(unsigned int pos, const local_mem & mem)
      {
        cl_uint size = static_cast<cl_uint>(mem.size());
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting local memory kernel argument of size " << size << " bytes at pos " << pos << " for kernel " << name_ << std::endl;
        #endif
        cl_int err = clSetKernelArg(handle_.get(), pos, size, 0);
        VIENNACL_ERR_CHECK(err);
      }



      /** @brief Convenience function for setting one kernel parameter */
      template<typename T0>
      kernel & operator()(T0 const & t0)
      {
         arg(0, t0);
         return *this;
      }

      /** @brief Convenience function for setting two kernel parameters */
      template<typename T0, typename T1>
      kernel & operator()(T0 const & t0, T1 const & t1)
      {
         arg(0, t0); arg(1, t1);
         return *this;
      }

      /** @brief Convenience function for setting three kernel parameters */
      template<typename T0, typename T1, typename T2>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2)
      {
         arg(0, t0); arg(1, t1); arg(2, t2);
         return *this;
      }

      /** @brief Convenience function for setting four kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3);
         return *this;
      }

      /** @brief Convenience function for setting five kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4);
         return *this;
      }

      /** @brief Convenience function for setting six kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         return *this;
      }

      /** @brief Convenience function for setting seven kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5, T6 const & t6)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5); arg(6, t6);
         return *this;
      }

      /** @brief Convenience function for setting eight kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5, T6 const & t6, T7 const & t7)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5); arg(6, t6); arg(7, t7);
         return *this;
      }

      /** @brief Convenience function for setting nine kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5); arg(6, t6); arg(7, t7); arg(8, t8);
         return *this;
      }

      /** @brief Convenience function for setting ten kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4,
                typename T5, typename T6, typename T7, typename T8, typename T9>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4,
                          T5 const & t5, T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5); arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9);
         return *this;
      }

      /** @brief Convenience function for setting eleven kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5); arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10);
         return *this;
      }

      /** @brief Convenience function for setting twelve kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         return *this;
      }

      /** @brief Convenience function for setting thirteen kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11, T12 const & t12)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11); arg(12, t12);
         return *this;
      }

      /** @brief Convenience function for setting fourteen kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13);
         return *this;
      }

      /** @brief Convenience function for setting fifteen kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14);
         return *this;
      }

      /** @brief Convenience function for setting sixteen kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15);
         return *this;
      }

      /** @brief Convenience function for setting seventeen kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16);
         return *this;
      }

      /** @brief Convenience function for setting eighteen kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17)
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         return *this;
      }

      /** @brief Convenience function for setting nineteen kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18);
         return *this;
      }

      /** @brief Convenience function for setting twenty kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19);
         return *this;
      }

      /** @brief Convenience function for setting twentyone kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20);
         return *this;
      }

      /** @brief Convenience function for setting twentytwo kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20, typename T21>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20, T21 const & t21
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20); arg(21, t21);
         return *this;
      }

      /** @brief Convenience function for setting 23 kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20, typename T21, typename T22>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20, T21 const & t21, T22 const & t22
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20); arg(21, t21);  arg(22, t22);
         return *this;
      }

      /** @brief Convenience function for setting 24 kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20, typename T21, typename T22, typename T23>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20, T21 const & t21, T22 const & t22, T23 const & t23
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20); arg(21, t21); arg(22, t22); arg(23, t23);
         return *this;
      }

      /** @brief Convenience function for setting 25 kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20, typename T21, typename T22, typename T23,
                typename T24>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20, T21 const & t21, T22 const & t22, T23 const & t23,
                          T24 const & t24
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20); arg(21, t21); arg(22, t22); arg(23, t23);
         arg(24, t24);
         return *this;
      }

      /** @brief Convenience function for setting 26 kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20, typename T21, typename T22, typename T23,
                typename T24, typename T25>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20, T21 const & t21, T22 const & t22, T23 const & t23,
                          T24 const & t24, T25 const & t25
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20); arg(21, t21); arg(22, t22); arg(23, t23);
         arg(24, t24); arg(25, t25);
         return *this;
      }

      /** @brief Convenience function for setting 27 kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20, typename T21, typename T22, typename T23,
                typename T24, typename T25, typename T26>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20, T21 const & t21, T22 const & t22, T23 const & t23,
                          T24 const & t24, T25 const & t25, T26 const & t26
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20); arg(21, t21); arg(22, t22); arg(23, t23);
         arg(24, t24); arg(25, t25); arg(26, t26);
         return *this;
      }

      /** @brief Convenience function for setting 28 kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20, typename T21, typename T22, typename T23,
                typename T24, typename T25, typename T26, typename T27>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20, T21 const & t21, T22 const & t22, T23 const & t23,
                          T24 const & t24, T25 const & t25, T26 const & t26, T27 const & t27
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20); arg(21, t21); arg(22, t22); arg(23, t23);
         arg(24, t24); arg(25, t25); arg(26, t26); arg(27, t27);
         return *this;
      }

      /** @brief Convenience function for setting 29 kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20, typename T21, typename T22, typename T23,
                typename T24, typename T25, typename T26, typename T27, typename T28>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20, T21 const & t21, T22 const & t22, T23 const & t23,
                          T24 const & t24, T25 const & t25, T26 const & t26, T27 const & t27, T28 const & t28
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20); arg(21, t21); arg(22, t22); arg(23, t23);
         arg(24, t24); arg(25, t25); arg(26, t26); arg(27, t27); arg(28, t28);
         return *this;
      }

      /** @brief Convenience function for setting 30 kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20, typename T21, typename T22, typename T23,
                typename T24, typename T25, typename T26, typename T27, typename T28, typename T29>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20, T21 const & t21, T22 const & t22, T23 const & t23,
                          T24 const & t24, T25 const & t25, T26 const & t26, T27 const & t27, T28 const & t28, T29 const & t29
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20); arg(21, t21); arg(22, t22); arg(23, t23);
         arg(24, t24); arg(25, t25); arg(26, t26); arg(27, t27); arg(28, t28); arg(29, t29);
         return *this;
      }

      /** @brief Convenience function for setting 31 kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20, typename T21, typename T22, typename T23,
                typename T24, typename T25, typename T26, typename T27, typename T28, typename T29,
                typename T30>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20, T21 const & t21, T22 const & t22, T23 const & t23,
                          T24 const & t24, T25 const & t25, T26 const & t26, T27 const & t27, T28 const & t28, T29 const & t29,
                          T30 const & t30
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20); arg(21, t21); arg(22, t22); arg(23, t23);
         arg(24, t24); arg(25, t25); arg(26, t26); arg(27, t27); arg(28, t28); arg(29, t29);
         arg(30, t30);
         return *this;
      }

      /** @brief Convenience function for setting 32 kernel parameters */
      template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5,
                typename T6, typename T7, typename T8, typename T9, typename T10, typename T11,
                typename T12, typename T13, typename T14, typename T15, typename T16, typename T17,
                typename T18, typename T19, typename T20, typename T21, typename T22, typename T23,
                typename T24, typename T25, typename T26, typename T27, typename T28, typename T29,
                typename T30, typename T31>
      kernel & operator()(T0 const & t0, T1 const & t1, T2 const & t2, T3 const & t3, T4 const & t4, T5 const & t5,
                          T6 const & t6, T7 const & t7, T8 const & t8, T9 const & t9, T10 const & t10, T11 const & t11,
                          T12 const & t12, T13 const & t13, T14 const & t14, T15 const & t15, T16 const & t16, T17 const & t17,
                          T18 const & t18, T19 const & t19, T20 const & t20, T21 const & t21, T22 const & t22, T23 const & t23,
                          T24 const & t24, T25 const & t25, T26 const & t26, T27 const & t27, T28 const & t28, T29 const & t29,
                          T30 const & t30, T31 const & t31
                         )
      {
         arg(0, t0); arg(1, t1); arg(2, t2); arg(3, t3); arg(4, t4); arg(5, t5);
         arg(6, t6); arg(7, t7); arg(8, t8); arg(9, t9); arg(10, t10); arg(11, t11);
         arg(12, t12); arg(13, t13); arg(14, t14); arg(15, t15); arg(16, t16); arg(17, t17);
         arg(18, t18); arg(19, t19); arg(20, t20); arg(21, t21); arg(22, t22); arg(23, t23);
         arg(24, t24); arg(25, t25); arg(26, t26); arg(27, t27); arg(28, t28); arg(29, t29);
         arg(30, t30); arg(31, t31);
         return *this;
      }




      /** @brief Returns the local work size at the respective dimension
      *
      * @param index   Dimension index (currently either 0 or 1)
      */
      size_type local_work_size(int index = 0) const
      {
        assert(index < 3 && bool("Work size index out of bounds"));
        return local_work_size_[index];
      }
      /** @brief Returns the global work size at the respective dimension
      *
      * @param index   Dimension index (currently either 0 or 1)
      */
      size_type global_work_size(int index = 0) const
      {
        assert(index < 3 && bool("Work size index out of bounds"));
        return global_work_size_[index];
      }

      /** @brief Sets the local work size at the respective dimension
      *
      * @param index   Dimension index (currently either 0 or 1)
      * @param s       The new local work size
      */
      void local_work_size(int index, size_type s)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting local work size to " << s << " at index " << index << " for kernel " << name_ << std::endl;
        #endif
        assert(index < 3 && bool("Work size index out of bounds"));
        local_work_size_[index] = s;
      }
      /** @brief Sets the global work size at the respective dimension
      *
      * @param index   Dimension index (currently either 0 or 1)
      * @param s       The new global work size
      */
      void global_work_size(int index, size_type s)
      {
        #if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_KERNEL)
        std::cout << "ViennaCL: Setting global work size to " << s << " at index " << index << " for kernel " << name_ << std::endl;
        #endif
        assert(index < 3 && bool("Work size index out of bounds"));
        global_work_size_[index] = s;
      }

      std::string const & name() const { return name_; }

      viennacl::ocl::handle<cl_kernel> const & handle() const { return handle_; }

      viennacl::ocl::context const & context() const { return *p_context_; }

    private:

      inline void set_work_size_defaults();    //see context.hpp for implementation

      viennacl::ocl::handle<cl_kernel> handle_;
      viennacl::ocl::program const * p_program_;
      viennacl::ocl::context const * p_context_;
      std::string name_;
      size_type local_work_size_[3];
      size_type global_work_size_[3];
    };

  } //namespace ocl
} //namespace viennacl

#endif
