#ifndef VIENNACL_OCL_ERROR_HPP_
#define VIENNACL_OCL_ERROR_HPP_

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

/** @file viennacl/ocl/error.hpp
    @brief Error handling for the OpenCL layer of ViennaCL
*/

//error levels:
//#define VIENNACL_DEBUG_ALL           //print all of the following
//#define VIENNACL_DEBUG_KERNEL        //debug any modifications on viennacl::ocl::kernel objects
//#define VIENNACL_DEBUG_COPY          //print infos related to setting up/modifying memory objects
//#define VIENNACL_DEBUG_OPENCL        //display debug info for the OpenCL layer (platform/context/queue creation,
//#define VIENNACL_DEBUG_DEVICE        //Show device info upon allocation
//#define VIENNACL_DEBUG_CONTEXT       //Debug queries to context
//#define VIENNACL_DEBUG_BUILD         //Show debug info from OpenCL compiler


//backwards compatibility:
#ifdef VIENNACL_BUILD_INFO
#define VIENNACL_DEBUG_ALL
#endif


#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <iostream>
#include <exception>
#include <stdexcept>

#define VIENNACL_BUG_REPORT_STRING  \
  "\nIf you think that this is a bug in ViennaCL, please report it at viennacl-support@lists.sourceforge.net and supply at least the following information:\n"\
  " * Operating System\n"\
  " * Which OpenCL implementation (AMD, NVIDIA, etc.)\n"\
  " * ViennaCL version\n"\
  "Many thanks in advance!";\

namespace viennacl
{
namespace ocl
{
//
// Generic errors
//
class queue_not_found : public std::runtime_error
{
public:
  queue_not_found(std::string const & what_arg) : std::runtime_error(what_arg) {}
};

class program_not_found : public std::runtime_error
{
public:
  program_not_found(std::string const & what_arg) : std::runtime_error(what_arg) {}
};

class kernel_not_found : public std::runtime_error
{
public:
  kernel_not_found(std::string const & what_arg) : std::runtime_error(what_arg) {}
};


//
// Wrapper for OpenCL exceptions:
//

/** @brief Exception thrown in the case that a requested compute device was not found.
  *
  * This exception usually shows up if a user requests a GPU for computation, but the OpenCL SDK does not support the GPU.
  */
class device_not_found : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_DEVICE_NOT_FOUND \n ViennaCL could not find a suitable device. Please check whether an OpenCL implementation is properly installed and a suitable device available."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the selected compute device is not available (maybe locked by another process). */
class device_not_available : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_DEVICE_NOT_AVAILABLE \n ViennaCL could not use the compute device because it is not available."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the OpenCL just-in-time compiler is not available. */
class compiler_not_available : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_COMPILER_NOT_AVAILABLE \n Your OpenCL framework does not provide an OpenCL compiler. Unfortunately, ViennaCL cannot be used without such a compiler."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if a memory object cannot be allocated. Usually the requested memory buffer is simply too large. */
class mem_object_allocation_failure : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_MEM_OBJECT_ALLOCATION_FAILURE \n ViennaCL could not allocate memory on the device. Most likely the device simply ran out of memory."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the compute device is out of resources (either global memory, registers, etc.) for the requested operation. */
class out_of_resources : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_OUT_OF_RESOURCES \n ViennaCL tried to launch a compute kernel, but the device does not provide enough resources. Try changing the global and local work item sizes."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the host cannot provide enough memory for the datastructures in the OpenCL backend (temporary arrays, etc.) to perform the requested operation. */
class out_of_host_memory : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_OUT_OF_HOST_MEMORY \n The host ran out of memory (usually CPU RAM). Please try again on smaller problems."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the OpenCL context does not have CL_QUEUE_PROFILING_ENABLE set, if the execution is not complete, or the event object is a user event object. */
class profiling_info_not_available : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_PROFILING_INFO_NOT_AVAILABLE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the source buffer overlaps the destination buffer when copying from device memory to device memory. */
class mem_copy_overlap : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_MEM_COPY_OVERLAP."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if there is a mismatch in image formats for the operands. */
class image_format_mismatch : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_IMAGE_FORMAT_MISMATCH."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the requested image format is not supported. */
class image_format_not_supported : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_IMAGE_FORMAT_NOT_SUPPORTED."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the OpenCL program cannot be built, usually due to a syntax error in the OpenCL code. */
class build_program_failure : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_BUILD_PROGRAM_FAILURE \n The OpenCL compiler encountered an error during the compilation of ViennaCL sources. This is most likely a bug in ViennaCL."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the mapping of device memory to the host memory space failed. */
class map_failure : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_MAP_FAILURE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown is an invalid value is provided to an OpenCL function. */
class invalid_value : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_VALUE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if an invalid device type is specified. */
class invalid_device_type : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_DEVICE_TYPE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if an invalid OpenCL platform is provided to an OpenCL function. */
class invalid_platform : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_PLATFORM."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if an invalid OpenCL device is provided to an OpenCL function. */
class invalid_device : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_DEVICE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if an invalid OpenCL context is provided to an OpenCL function. */
class invalid_context : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_CONTEXT."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if invalid OpenCL command queue properties are provided when creating a command queue. */
class invalid_queue_properties : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_QUEUE_PROPERTIES."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if an invalid OpenCL command queue is provided to an OpenCL function. */
class invalid_command_queue : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_COMMAND_QUEUE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the provided pointer to host memory is invalid. */
class invalid_host_ptr : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_HOST_PTR."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if an invalid OpenCL memory object (of type cl_mem) is passed to an OpenCL funciton. */
class invalid_mem_object : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_MEM_OBJECT."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if an invalid image format descriptor is provided. */
class invalid_image_format_descriptor : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_IMAGE_FORMAT_DESCRIPTOR."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the image size provided is invalid (e.g. zero). */
class invalid_image_size : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_IMAGE_SIZE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if an invalid sampler is provided for an image. */
class invalid_sampler : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_SAMPLER."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the OpenCL binary (generated from the jit-compiler or loaded from some other location) won't work on the device (e.g. due to a lack of double precision support). */
class invalid_binary : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_BINARY."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if invalid build options are passed to the OpenCL just-in-time compiler. */
class invalid_build_options : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_BUILD_OPTIONS."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if an OpenCL program object handle is invalid (e.g. not initialized). */
class invalid_program : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_PROGRAM."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if there is no built program exectuable available for the device. */
class invalid_program_executable : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_PROGRAM_EXECUTABLE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the provided kernel name is invalid (e.g. not part of the program provided). */
class invalid_kernel_name : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_KERNEL_NAME \n The supplied kernel name is invalid. If you have written your own OpenCL kernel, please check that the correct kernel name is used in the initalization of the kernel object."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the kernel definition (number of arguments, argument types, etc.) is not the same for all devices for which the program has been built. */
class invalid_kernel_definition : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_KERNEL_DEFINITION."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the provided kernel object (of type cl_kernel) is invalid (e.g. not initialized, from different context, or corrupted). */
class invalid_kernel : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_KERNEL \n The supplied kernel argument is invalid."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the kernel argument index is invalid, e.g. an arg index larger than the number of kernel arguments was provided. */
class invalid_arg_index : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_ARG_INDEX."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the kernel argument provided has an invalid value. */
class invalid_arg_value : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_ARG_VALUE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the arguments to an OpenCL kernel have an invalid size e.g. not sizeof(cl_mem)). */
class invalid_arg_size : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_ARG_SIZE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the kernel arguments are invalid and/or do not fit the kernel parameter list. */
class invalid_kernel_args : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_KERNEL_ARGS \n The supplied kernel arguments do not fit the kernel parameter list. If you have written your own OpenCL kernel, please check that the correct kernel arguments are set in the appropriate order."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the work dimension is invalid (usually this means that the work dimension was set to be larger than three. */
class invalid_work_dimension : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_WORK_DIMENSION"
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the number of work groups is invalid (usually this means that more than 256/512/768/1024 work groups have been specified, but the device(s) cannot support this. */
class invalid_work_group_size : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_WORK_GROUP_SIZE \n The supplied work group size is invalid. If you have set this value manually, please reconsider your choice."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the number of work items per work group invalid (usually this means that more than 256/512/768/1024 work items have been specified, but the device(s) cannot support this. */
class invalid_work_item_size : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_WORK_ITEM_SIZE \n The work item size is invalid. If you have set this value manually, please reconsider your choice."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the provided offset for get_global_id() in OpenCL kernels is invalid. */
class invalid_global_offset : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_GLOBAL_OFFSET."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the provided event wait list is invalid. */
class invalid_event_wait_list : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_EVENT_WAIT_LIST."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the provided event object (of type cl_event) is invalid. */
class invalid_event : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_EVENT."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if interoperability of OpenCL with other frameworks collide. */
class invalid_operation : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_OPERATION."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the provided OpenGL (not OpenCL) object is invalid. */
class invalid_gl_object : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_GL_OBJECT."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the provided buffer size is invalid (e.g. zero) */
class invalid_buffer_size : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_BUFFER_SIZE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the provided miplevel is greater than zero, but the OpenGL implementation does not support creating from non-zero mipmap levels. */
class invalid_mip_level : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_MIP_LEVEL."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the total number of work items is invalid (for example, not divisible by the number of work items per work group). */
class invalid_global_work_size : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_GLOBAL_WORK_SIZE."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if an invalid property is provided to a function (vague value). */
class invalid_property : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: CL_INVALID_PROPERTY."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the returned error cannot be resolved to some defined error constant. Might result from invalid sources, invalid memory operations, etc. */
class unknown_error : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: ViennaCL encountered an unknown OpenCL error. Most likely your OpenCL SDK or driver is not installed properly. In some cases, this error is due to an invalid global work size or several kernel compilation errors."
        VIENNACL_BUG_REPORT_STRING;
  }
};

/** @brief Exception thrown if the user wants to use double precision arithmetics, but the device does not support double precision. */
class double_precision_not_provided_error : public std::exception
{
  virtual const char* what() const throw()
  {
    return "ViennaCL: FATAL ERROR: You requested to create a ViennaCL type using double precision. However, double precision is not supported by your device."
        VIENNACL_BUG_REPORT_STRING;
  }
};


/** @brief An error reporting class. Template argument is used to avoid problems with external linkage.
*
*  Do not use this class directly, use the macro CL_ERROR_CHECK instead.
*  @tparam T   Useless. Helps to avoid troubles with external linkage of namespace functions.
*/
template<typename T>
struct error_checker
{

  /** @brief Trows exceptions that reflect OpenCL error codes */
  static void raise_exception(cl_int err)
  {
    switch (err)
    {
    case CL_DEVICE_NOT_FOUND:               throw device_not_found();
    case CL_DEVICE_NOT_AVAILABLE:           throw device_not_available();
    case CL_COMPILER_NOT_AVAILABLE:         throw compiler_not_available();
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:  throw mem_object_allocation_failure();
    case CL_OUT_OF_RESOURCES:               throw out_of_resources();
    case CL_OUT_OF_HOST_MEMORY:             throw out_of_host_memory();
    case CL_PROFILING_INFO_NOT_AVAILABLE:   throw profiling_info_not_available();
    case CL_MEM_COPY_OVERLAP:               throw mem_copy_overlap();
    case CL_IMAGE_FORMAT_MISMATCH:          throw image_format_mismatch();
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:     throw image_format_not_supported();
    case CL_BUILD_PROGRAM_FAILURE:          throw build_program_failure();
    case CL_MAP_FAILURE:                    throw map_failure();

    case CL_INVALID_VALUE:                  throw invalid_value();
    case CL_INVALID_DEVICE_TYPE:            throw invalid_device_type();
    case CL_INVALID_PLATFORM:               throw invalid_platform();
    case CL_INVALID_DEVICE:                 throw invalid_device();
    case CL_INVALID_CONTEXT:                throw invalid_context();
    case CL_INVALID_QUEUE_PROPERTIES:       throw invalid_queue_properties();
    case CL_INVALID_COMMAND_QUEUE:          throw invalid_command_queue();
    case CL_INVALID_HOST_PTR:               throw invalid_host_ptr();
    case CL_INVALID_MEM_OBJECT:             throw invalid_mem_object();
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: throw invalid_image_format_descriptor();
    case CL_INVALID_IMAGE_SIZE:             throw invalid_image_size();
    case CL_INVALID_SAMPLER:                throw invalid_sampler();
    case CL_INVALID_BINARY:                 throw invalid_binary();
    case CL_INVALID_BUILD_OPTIONS:          throw invalid_build_options();
    case CL_INVALID_PROGRAM:                throw invalid_program();
    case CL_INVALID_PROGRAM_EXECUTABLE:     throw invalid_program_executable();
    case CL_INVALID_KERNEL_NAME:            throw invalid_kernel_name();
    case CL_INVALID_KERNEL_DEFINITION:      throw invalid_kernel_definition();
    case CL_INVALID_KERNEL:                 throw invalid_kernel();
    case CL_INVALID_ARG_INDEX:              throw invalid_arg_index();
    case CL_INVALID_ARG_VALUE:              throw invalid_arg_value();
    case CL_INVALID_ARG_SIZE:               throw invalid_arg_size();
    case CL_INVALID_KERNEL_ARGS:            throw invalid_kernel_args();
    case CL_INVALID_WORK_DIMENSION:         throw invalid_work_dimension();
    case CL_INVALID_WORK_GROUP_SIZE:        throw invalid_work_group_size();
    case CL_INVALID_WORK_ITEM_SIZE:         throw invalid_work_item_size();
    case CL_INVALID_GLOBAL_OFFSET:          throw invalid_global_offset();
    case CL_INVALID_EVENT_WAIT_LIST:        throw invalid_event_wait_list();
    case CL_INVALID_EVENT:                  throw invalid_event();
    case CL_INVALID_OPERATION:              throw invalid_operation();
    case CL_INVALID_GL_OBJECT:              throw invalid_gl_object();
    case CL_INVALID_BUFFER_SIZE:            throw invalid_buffer_size();
    case CL_INVALID_MIP_LEVEL:              throw invalid_mip_level();
    case CL_INVALID_GLOBAL_WORK_SIZE:       throw invalid_global_work_size();
#ifdef CL_INVALID_PROPERTY
    case CL_INVALID_PROPERTY:               throw invalid_property();
#endif
      //  return "CL_INVALID_GLOBAL_WORK_SIZE";

    default: throw unknown_error();
    }

  } //getErrorString

  /** @brief Checks whether an OpenCL error has occured.
  *
  *  Do not use this function directly, use the macro CL_ERROR_CHECK instead.
  */
  static void checkError(cl_int err,

#ifdef VIENNACL_DEBUG_ALL
                         const char * file,
                         const char * func,
                         int line)
#else
                         const char *,
                         const char *,
                         int)
#endif
  {
    if (err != CL_SUCCESS)
    {
#ifdef VIENNACL_DEBUG_ALL
      std::cerr << "ViennaCL: Error " << err  << " in function " << func << " ( "<< file << ":" << line << " ) " << std::endl;
#endif
      raise_exception(err);
    }
  } //checkError()

}; //struct

#define VIENNACL_ERR_CHECK(err) viennacl::ocl::error_checker<void>::checkError(err, __FILE__, __FUNCTION__, __LINE__);

} //namespace ocl
} //namespace viennacl

#endif

