#ifndef VIENNACL_OCL_DEVICE_HPP_
#define VIENNACL_OCL_DEVICE_HPP_

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

/** @file viennacl/ocl/device.hpp
    @brief Represents an OpenCL device within ViennaCL
*/

#ifdef __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif

#include<stdio.h>

#include <vector>
#include <string>
#include <sstream>
#include <assert.h>
#include "viennacl/ocl/device_utils.hpp"
#include "viennacl/ocl/handle.hpp"
#include "viennacl/ocl/error.hpp"

namespace viennacl
{
namespace ocl
{

/** @brief A class representing a compute device (e.g. a GPU)
*
*/
class device
{
public:
  explicit device() : device_(0) { flush_cache(); }

  explicit device(cl_device_id dev) : device_(dev)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_DEVICE)
    std::cout << "ViennaCL: Creating device object (CTOR with cl_device_id)" << std::endl;
#endif
    flush_cache();
  }

  device(const device & other) : device_(0)
  {
#if defined(VIENNACL_DEBUG_ALL) || defined(VIENNACL_DEBUG_DEVICE)
    std::cout << "ViennaCL: Creating device object (Copy CTOR)" << std::endl;
#endif
    if (device_ != other.device_)
    {
      device_ = other.device_;
      flush_cache();
    }
  }

  /** @brief The default compute device address space size specified as an unsigned integer value in bits. Currently supported values are 32 or 64 bits. */
  cl_uint address_bits() const
  {
    if (!address_bits_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_ADDRESS_BITS, sizeof(cl_uint), static_cast<void *>(&address_bits_), NULL);
      VIENNACL_ERR_CHECK(err);
      address_bits_valid_ = true;
    }
    return address_bits_;
  }

  /** @brief Is CL_TRUE if the device is available and CL_FALSE if the device is not available. */
  cl_bool available() const
  {
    if (!available_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_AVAILABLE, sizeof(cl_bool), static_cast<void *>(&available_), NULL);
      VIENNACL_ERR_CHECK(err);
      available_valid_ = true;
    }
    return available_;
  }

  /** @brief Is CL_FALSE if the implementation does not have a compiler available to compile the program source. Is CL_TRUE if the compiler is available. This can be CL_FALSE for the embedded platform profile only. */
  cl_bool compiler_available() const
  {
    if (!compiler_available_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_COMPILER_AVAILABLE , sizeof(cl_bool), static_cast<void *>(&compiler_available_), NULL);
      VIENNACL_ERR_CHECK(err);
      compiler_available_valid_ = true;
    }
    return compiler_available_;
  }

#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
  /** @brief Describes the OPTIONAL double precision floating-point capability of the OpenCL device.
      *
      * This is a bit-field that describes one or more of the following values:
      *   CL_FP_DENORM - denorms are supported.
      *   CL_FP_INF_NAN - INF and NaNs are supported.
      *   CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported.
      *   CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported.
      *   CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported.
      *   CP_FP_FMA - IEEE754-2008 fused multiply-add is supported.
      *
      * The mandated minimum double precision floating-point capability is
      * CL_FP_FMA | CL_FP_ROUND_TO_NEAREST | CL_FP_ROUND_TO_ZERO | CL_FP_ROUND_TO_INF | CL_FP_INF_NAN | CL_FP_DENORM.
      */
  cl_device_fp_config double_fp_config() const
  {
    if (double_support() && !double_fp_config_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_DOUBLE_FP_CONFIG, sizeof(cl_device_fp_config), static_cast<void *>(&double_fp_config_), NULL);
      VIENNACL_ERR_CHECK(err);
      double_fp_config_valid_ = true;
    }
    return double_fp_config_;
  }
#endif

  /** @brief Is CL_TRUE if the OpenCL device is a little endian device and CL_FALSE otherwise. */
  cl_bool endian_little() const
  {
    if (!endian_little_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_ENDIAN_LITTLE, sizeof(cl_bool), static_cast<void *>(&endian_little_), NULL);
      VIENNACL_ERR_CHECK(err);
      endian_little_valid_ = true;
    }
    return endian_little_;
  }

  /** @brief Is CL_TRUE if the device implements error correction for all accesses to compute device memory (global and constant) and CL_FALSE otherwise. */
  cl_bool error_correction_support() const
  {
    if (!error_correction_support_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_ERROR_CORRECTION_SUPPORT , sizeof(cl_bool), static_cast<void *>(&error_correction_support_), NULL);
      VIENNACL_ERR_CHECK(err);
      error_correction_support_valid_ = true;
    }
    return error_correction_support_;
  }

  /** @brief Describes the execution capabilities of the device.
      *
      * This is a bit-field that describes one or more of the following values:
      *   CL_EXEC_KERNEL - The OpenCL device can execute OpenCL kernels.
      *   CL_EXEC_NATIVE_KERNEL - The OpenCL device can execute native kernels.
      * The mandated minimum capability is CL_EXEC_KERNEL.
      */
  cl_device_exec_capabilities execution_capabilities() const
  {
    if (!execution_capabilities_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_EXECUTION_CAPABILITIES  , sizeof(cl_device_exec_capabilities), static_cast<void *>(&execution_capabilities_), NULL);
      VIENNACL_ERR_CHECK(err);
      execution_capabilities_valid_ = true;
    }
    return execution_capabilities_;
  }

  /** @brief Returns a space-separated list of extension names (the extension names themselves do not contain any spaces).
      *
      * The list of extension names returned currently can include one or more of the following approved extension names:
      *   cl_khr_fp64
      *   cl_khr_int64_base_atomics
      *   cl_khr_int64_extended_atomics
      *   cl_khr_fp16
      *   cl_khr_gl_sharing
      *   cl_khr_gl_event
      *   cl_khr_d3d10_sharing
      */
  std::string extensions() const
  {
    if (!extensions_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_EXTENSIONS, sizeof(char) * 2048, static_cast<void *>(&extensions_), NULL);
      VIENNACL_ERR_CHECK(err);
      extensions_valid_ = true;
    }
    return extensions_;
  }

  /** @brief Size of global memory cache in bytes. */
  cl_ulong global_mem_cache_size() const
  {
    if (!global_mem_cache_size_valid_)
    {
      cl_int err = clGetDeviceInfo(device_,  CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(cl_ulong), static_cast<void *>(&global_mem_cache_size_), NULL);
      VIENNACL_ERR_CHECK(err);
      global_mem_cache_size_valid_ = true;
    }
    return global_mem_cache_size_;
  }

  /** @brief Type of global memory cache supported. Valid values are: CL_NONE, CL_READ_ONLY_CACHE, and CL_READ_WRITE_CACHE. */
  cl_device_mem_cache_type global_mem_cache_type() const
  {
    if (!global_mem_cache_type_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, sizeof(cl_device_mem_cache_type), static_cast<void *>(&global_mem_cache_type_), NULL);
      VIENNACL_ERR_CHECK(err);
      global_mem_cache_type_valid_ = true;
    }
    return global_mem_cache_type_;
  }

  /** @brief Size of global memory cache in bytes. */
  cl_uint global_mem_cacheline_size() const
  {
    if (!global_mem_cacheline_size_valid_)
    {
      cl_int err = clGetDeviceInfo(device_,  CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, sizeof(cl_uint), static_cast<void *>(&global_mem_cacheline_size_), NULL);
      VIENNACL_ERR_CHECK(err);
      global_mem_cacheline_size_valid_ = true;
    }
    return global_mem_cacheline_size_;
  }

  /** @brief Size of global memory in bytes. */
  cl_ulong global_mem_size() const
  {
    if (!global_mem_size_valid_)
    {
      cl_int err = clGetDeviceInfo(device_,  CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), static_cast<void *>(&global_mem_size_), NULL);
      VIENNACL_ERR_CHECK(err);
      global_mem_size_valid_ = true;
    }
    return global_mem_size_;
  }

#ifdef CL_DEVICE_HALF_FP_CONFIG
  /** @brief Describes the OPTIONAL half precision floating-point capability of the OpenCL device.
      *
      * This is a bit-field that describes one or more of the following values:
      *   CL_FP_DENORM - denorms are supported.
      *   CL_FP_INF_NAN - INF and NaNs are supported.
      *   CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported.
      *   CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported.
      *   CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported.
      *   CP_FP_FMA - IEEE754-2008 fused multiply-add is supported.
      *
      * The required minimum half precision floating-point capability as implemented by this extension is CL_FP_ROUND_TO_ZERO or CL_FP_ROUND_TO_INF | CL_FP_INF_NAN.
      */
  cl_device_fp_config half_fp_config() const
  {
    if (!half_fp_config_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_HALF_FP_CONFIG, sizeof(cl_device_fp_config), static_cast<void *>(&half_fp_config_), NULL);
      VIENNACL_ERR_CHECK(err);
      half_fp_config_valid_ = true;
    }
    return half_fp_config_;
  }
#endif

  /** @brief Is CL_TRUE if the device and the host have a unified memory subsystem and is CL_FALSE otherwise. */
#ifdef CL_DEVICE_HOST_UNIFIED_MEMORY
  cl_bool host_unified_memory() const
  {
    if (!host_unified_memory_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_HOST_UNIFIED_MEMORY, sizeof(cl_bool), static_cast<void *>(&host_unified_memory_), NULL);
      VIENNACL_ERR_CHECK(err);
      host_unified_memory_valid_ = true;
    }
    return host_unified_memory_;
  }
#endif

  /** @brief Is CL_TRUE if images are supported by the OpenCL device and CL_FALSE otherwise. */
  cl_bool image_support() const
  {
    if (!image_support_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), static_cast<void *>(&image_support_), NULL);
      VIENNACL_ERR_CHECK(err);
      image_support_valid_ = true;
    }
    return image_support_;
  }

  /** @brief Max height of 2D image in pixels. The minimum value is 8192 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE. */
  size_t image2d_max_height() const
  {
    if (!image2d_max_height_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), static_cast<void *>(&image2d_max_height_), NULL);
      VIENNACL_ERR_CHECK(err);
      image2d_max_height_valid_ = true;
    }
    return image2d_max_height_;
  }

  /** @brief Max width of 2D image in pixels. The minimum value is 8192 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE. */
  size_t image2d_max_width() const
  {
    if (!image2d_max_width_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), static_cast<void *>(&image2d_max_width_), NULL);
      VIENNACL_ERR_CHECK(err);
      image2d_max_width_valid_ = true;
    }
    return image2d_max_width_;
  }

  /** @brief Max depth of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE. */
  size_t image3d_max_depth() const
  {
    if (!image3d_max_depth_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_IMAGE3D_MAX_DEPTH, sizeof(size_t), static_cast<void *>(&image3d_max_depth_), NULL);
      VIENNACL_ERR_CHECK(err);
      image3d_max_depth_valid_ = true;
    }
    return image3d_max_depth_;
  }

  /** @brief Max height of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE. */
  size_t image3d_max_height() const
  {
    if (!image3d_max_height_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_IMAGE3D_MAX_HEIGHT, sizeof(size_t), static_cast<void *>(&image3d_max_height_), NULL);
      VIENNACL_ERR_CHECK(err);
      image3d_max_height_valid_ = true;
    }
    return image3d_max_height_;
  }

  /** @brief Max width of 3D image in pixels. The minimum value is 2048 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE. */
  size_t image3d_max_width() const
  {
    if (!image3d_max_width_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_IMAGE3D_MAX_WIDTH, sizeof(size_t), static_cast<void *>(&image3d_max_width_), NULL);
      VIENNACL_ERR_CHECK(err);
      image3d_max_width_valid_ = true;
    }
    return image3d_max_width_;
  }

  /** @brief Size of local memory arena in bytes. The minimum value is 32 KB. */
  cl_ulong local_mem_size() const
  {
    if (!local_mem_size_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(cl_ulong), static_cast<void *>(&local_mem_size_), NULL);
      VIENNACL_ERR_CHECK(err);
      local_mem_size_valid_ = true;
    }
    return local_mem_size_;
  }

  /** @brief Type of local memory supported. This can be set to CL_LOCAL implying dedicated local memory storage such as SRAM, or CL_GLOBAL. */
  cl_device_local_mem_type local_mem_type() const
  {
    if (!local_mem_type_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_LOCAL_MEM_TYPE, sizeof(cl_device_local_mem_type), static_cast<void *>(&local_mem_type_), NULL);
      VIENNACL_ERR_CHECK(err);
      local_mem_type_valid_ = true;
    }
    return local_mem_type_;
  }

  /** @brief Maximum configured clock frequency of the device in MHz. */
  cl_uint max_clock_frequency() const
  {
    if (!max_clock_frequency_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(cl_uint), static_cast<void *>(&max_clock_frequency_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_clock_frequency_valid_ = true;
    }
    return max_clock_frequency_;
  }

  /** @brief The number of parallel compute cores on the OpenCL device. The minimum value is 1. */
  cl_uint max_compute_units() const
  {
    if (!max_compute_units_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), static_cast<void *>(&max_compute_units_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_compute_units_valid_ = true;
    }
    return max_compute_units_;
  }

  /** @brief Max number of arguments declared with the __constant qualifier in a kernel. The minimum value is 8. */
  cl_uint max_constant_args() const
  {
    if (!max_constant_args_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_CONSTANT_ARGS, sizeof(cl_uint), static_cast<void *>(&max_constant_args_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_constant_args_valid_ = true;
    }
    return max_constant_args_;
  }

  /** @brief Max size in bytes of a constant buffer allocation. The minimum value is 64 KB. */
  cl_ulong max_constant_buffer_size() const
  {
    if (!max_constant_buffer_size_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, sizeof(cl_ulong), static_cast<void *>(&max_constant_buffer_size_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_constant_buffer_size_valid_ = true;
    }
    return max_constant_buffer_size_;
  }

  /** @brief Max size of memory object allocation in bytes. The minimum value is max(1/4th of CL_DEVICE_GLOBAL_MEM_SIZE, 128*1024*1024) */
  cl_ulong max_mem_alloc_size() const
  {
    if (!max_mem_alloc_size_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), static_cast<void *>(&max_mem_alloc_size_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_mem_alloc_size_valid_ = true;
    }
    return max_mem_alloc_size_;
  }

  /** @brief Max size in bytes of the arguments that can be passed to a kernel. The minimum value is 1024.
      *
      * For this minimum value, only a maximum of 128 arguments can be passed to a kernel.
      */
  size_t max_parameter_size() const
  {
    if (!max_parameter_size_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_PARAMETER_SIZE, sizeof(size_t), static_cast<void *>(&max_parameter_size_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_parameter_size_valid_ = true;
    }
    return max_parameter_size_;
  }

  /** @brief Max number of simultaneous image objects that can be read by a kernel. The minimum value is 128 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE. */
  cl_uint max_read_image_args() const
  {
    if (!max_read_image_args_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_READ_IMAGE_ARGS, sizeof(cl_uint), static_cast<void *>(&max_read_image_args_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_read_image_args_valid_ = true;
    }
    return max_read_image_args_;
  }

  /** @brief Max number of simultaneous image objects that can be read by a kernel. The minimum value is 128 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE. */
  cl_uint max_samplers() const
  {
    if (!max_samplers_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_SAMPLERS, sizeof(cl_uint), static_cast<void *>(&max_samplers_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_samplers_valid_ = true;
    }
    return max_samplers_;
  }

  /** @brief Maximum number of work-items in a work-group executing a kernel using the data parallel execution model. The minimum value is 1. */
  size_t max_work_group_size() const
  {
    if (!max_work_group_size_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), static_cast<void *>(&max_work_group_size_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_work_group_size_valid_ = true;
    }
    return max_work_group_size_;
  }

  /** @brief Maximum dimensions that specify the global and local work-item IDs used by the data parallel execution model. The minimum value is 3. */
  cl_uint max_work_item_dimensions() const
  {
    if (!max_work_item_dimensions_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, sizeof(cl_uint), static_cast<void *>(&max_work_item_dimensions_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_work_item_dimensions_valid_ = true;
    }
    return max_work_item_dimensions_;
  }

  /** @brief Maximum number of work-items that can be specified in each dimension of the work-group.
      *
      * Returns n size_t entries, where n is the value returned by the query for CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS. The minimum value is (1, 1, 1).
      */
  std::vector<size_t> max_work_item_sizes() const
  {
    std::vector<size_t> result(max_work_item_dimensions());

    assert(result.size() < 16 && bool("Supported work item dimensions exceed available capacity!"));

    if (!max_work_item_sizes_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(size_t) * 16, static_cast<void *>(&max_work_item_sizes_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_work_item_sizes_valid_ = true;
    }

    for (vcl_size_t i=0; i<result.size(); ++i)
      result[i] = max_work_item_sizes_[i];

    return result;
  }

  /** @brief Max number of simultaneous image objects that can be written to by a kernel. The minimum value is 8 if CL_DEVICE_IMAGE_SUPPORT is CL_TRUE. */
  cl_uint max_write_image_args() const
  {
    if (!max_write_image_args_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MAX_WRITE_IMAGE_ARGS, sizeof(cl_uint), static_cast<void *>(&max_write_image_args_), NULL);
      VIENNACL_ERR_CHECK(err);
      max_write_image_args_valid_ = true;
    }
    return max_write_image_args_;
  }

  /** @brief Describes the alignment in bits of the base address of any allocated memory object. */
  cl_uint mem_base_addr_align() const
  {
    if (!mem_base_addr_align_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MEM_BASE_ADDR_ALIGN, sizeof(cl_uint), static_cast<void *>(&mem_base_addr_align_), NULL);
      VIENNACL_ERR_CHECK(err);
      mem_base_addr_align_valid_ = true;
    }
    return mem_base_addr_align_;
  }

  /** @brief The smallest alignment in bytes which can be used for any data type. */
  cl_uint min_data_type_align_size() const
  {
    if (!min_data_type_align_size_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE, sizeof(cl_uint), static_cast<void *>(&min_data_type_align_size_), NULL);
      VIENNACL_ERR_CHECK(err);
      min_data_type_align_size_valid_ = true;
    }
    return min_data_type_align_size_;
  }

  /** @brief Device name string. */
  std::string name() const
  {
    if (!name_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(char) * 256, static_cast<void *>(name_), NULL);
      VIENNACL_ERR_CHECK(err);
      name_valid_ = true;
    }
    return name_;
  }

  /** @brief Device architecture family. */
  device_architecture_family architecture_family() const
  {
    if ( !architecture_family_valid_)
    {
      architecture_family_ = get_architecture_family(vendor_id(), name());
      architecture_family_valid_ = true;
    }
    return architecture_family_;
  }

#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR
  /** @brief Returns the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector. */
  cl_uint native_vector_width_char() const
  {
    if (!native_vector_width_char_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR, sizeof(cl_uint), static_cast<void *>(&native_vector_width_char_), NULL);
      VIENNACL_ERR_CHECK(err);
      native_vector_width_char_valid_ = true;
    }
    return native_vector_width_char_;
  }
#endif

#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT
  /** @brief Returns the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector. */
  cl_uint native_vector_width_short() const
  {
    if (!native_vector_width_short_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT, sizeof(cl_uint), static_cast<void *>(&native_vector_width_short_), NULL);
      VIENNACL_ERR_CHECK(err);
      native_vector_width_short_valid_ = true;
    }
    return native_vector_width_short_;
  }
#endif

#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_INT
  /** @brief Returns the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector. */
  cl_uint native_vector_width_int() const
  {
    if (!native_vector_width_int_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_NATIVE_VECTOR_WIDTH_INT, sizeof(cl_uint), static_cast<void *>(&native_vector_width_int_), NULL);
      VIENNACL_ERR_CHECK(err);
      native_vector_width_int_valid_ = true;
    }
    return native_vector_width_int_;
  }
#endif

#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG
  /** @brief Returns the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector. */
  cl_uint native_vector_width_long() const
  {
    if (!native_vector_width_long_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG, sizeof(cl_uint), static_cast<void *>(&native_vector_width_long_), NULL);
      VIENNACL_ERR_CHECK(err);
      native_vector_width_long_valid_ = true;
    }
    return native_vector_width_long_;
  }
#endif

#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT
  /** @brief Returns the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector. */
  cl_uint native_vector_width_float() const
  {
    if (!native_vector_width_float_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), static_cast<void *>(&native_vector_width_float_), NULL);
      VIENNACL_ERR_CHECK(err);
      native_vector_width_float_valid_ = true;
    }
    return native_vector_width_float_;
  }
#endif

#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE
  /** @brief Returns the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector.
      *
      * If the cl_khr_fp64 extension is not supported, this function returns 0.
      */
  cl_uint native_vector_width_double() const
  {
    if (!native_vector_width_double_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), static_cast<void *>(&native_vector_width_double_), NULL);
      VIENNACL_ERR_CHECK(err);
      native_vector_width_double_valid_ = true;
    }
    return native_vector_width_double_;
  }
#endif

#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF
  /** @brief Returns the native ISA vector width. The vector width is defined as the number of scalar elements that can be stored in the vector.
      *
      * If the cl_khr_fp16 extension is not supported, this function returns 0.
      */
  cl_uint native_vector_width_half() const
  {
    if (!native_vector_width_half_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF, sizeof(cl_uint), static_cast<void *>(&native_vector_width_half_), NULL);
      VIENNACL_ERR_CHECK(err);
      native_vector_width_half_valid_ = true;
    }
    return native_vector_width_half_;
  }
#endif

#if CL_DEVICE_OPENCL_C_VERSION
  /** @brief OpenCL C version string. Returns the highest OpenCL C version supported by the compiler for this device.
      *
      * This version string has the following format:
      *   OpenCL[space]C[space][major_version.minor_version][space][vendor-specific information]
      * The major_version.minor_version value must be 1.1 if CL_DEVICE_VERSION is OpenCL 1.1.
      * The major_version.minor_version value returned can be 1.0 or 1.1 if CL_DEVICE_VERSION is OpenCL 1.0.
      * If OpenCL C 1.1 is returned, this implies that the language feature set defined in section 6 of the OpenCL 1.1 specification is supported by the OpenCL 1.0 device.
      */
  std::string opencl_c_version() const
  {
    if (!opencl_c_version_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_OPENCL_C_VERSION, sizeof(char) * 128, static_cast<void *>(opencl_c_version_), NULL);
      VIENNACL_ERR_CHECK(err);
      opencl_c_version_valid_ = true;
    }
    return opencl_c_version_;
  }
#endif

  /** @brief The platform associated with this device. */
  cl_platform_id platform() const
  {
    if (!platform_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), static_cast<void *>(&platform_), NULL);
      VIENNACL_ERR_CHECK(err);
      platform_valid_ = true;
    }
    return platform_;
  }

  /** @brief Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector. */
  cl_uint preferred_vector_width_char() const
  {
    if (!preferred_vector_width_char_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR, sizeof(cl_uint), static_cast<void *>(&preferred_vector_width_char_), NULL);
      VIENNACL_ERR_CHECK(err);
      preferred_vector_width_char_valid_ = true;
    }
    return preferred_vector_width_char_;
  }

  /** @brief Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector. */
  cl_uint preferred_vector_width_short() const
  {
    if (!preferred_vector_width_short_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT, sizeof(cl_uint), static_cast<void *>(&preferred_vector_width_short_), NULL);
      VIENNACL_ERR_CHECK(err);
      preferred_vector_width_short_valid_ = true;
    }
    return preferred_vector_width_short_;
  }

  /** @brief Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector. */
  cl_uint preferred_vector_width_int() const
  {
    if (!preferred_vector_width_int_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT, sizeof(cl_uint), static_cast<void *>(&preferred_vector_width_int_), NULL);
      VIENNACL_ERR_CHECK(err);
      preferred_vector_width_int_valid_ = true;
    }
    return preferred_vector_width_int_;
  }

  /** @brief Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector. */
  cl_uint preferred_vector_width_long() const
  {
    if (!preferred_vector_width_long_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG, sizeof(cl_uint), static_cast<void *>(&preferred_vector_width_long_), NULL);
      VIENNACL_ERR_CHECK(err);
      preferred_vector_width_long_valid_ = true;
    }
    return preferred_vector_width_long_;
  }

  /** @brief Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector. */
  cl_uint preferred_vector_width_float() const
  {
    if (!preferred_vector_width_float_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT, sizeof(cl_uint), static_cast<void *>(&preferred_vector_width_float_), NULL);
      VIENNACL_ERR_CHECK(err);
      preferred_vector_width_float_valid_ = true;
    }
    return preferred_vector_width_float_;
  }

  /** @brief Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
      *
      * If the cl_khr_fp64 extension is not supported, this function returns 0.
      */
  cl_uint preferred_vector_width_double() const
  {
    if (!preferred_vector_width_double_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE, sizeof(cl_uint), static_cast<void *>(&preferred_vector_width_double_), NULL);
      VIENNACL_ERR_CHECK(err);
      preferred_vector_width_double_valid_ = true;
    }
    return preferred_vector_width_double_;
  }

  /** @brief Preferred native vector width size for built-in scalar types that can be put into vectors. The vector width is defined as the number of scalar elements that can be stored in the vector.
      *
      * If the cl_khr_fp16 extension is not supported, this function returns 0.
      */
#ifdef CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF
  cl_uint preferred_vector_width_half() const
  {
    if (!preferred_vector_width_half_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF, sizeof(cl_uint), static_cast<void *>(&preferred_vector_width_half_), NULL);
      VIENNACL_ERR_CHECK(err);
      preferred_vector_width_half_valid_ = true;
    }
    return preferred_vector_width_half_;
  }
#endif

  /** @brief OpenCL profile string. Returns the profile name supported by the device.
      *
      * The profile name returned can be one of the following strings:
      *   FULL_PROFILE - if the device supports the OpenCL specification
      *   EMBEDDED_PROFILE - if the device supports the OpenCL embedded profile.
      */
  std::string profile() const
  {
    if (!profile_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_PROFILE, sizeof(char) * 32, static_cast<void *>(profile_), NULL);
      VIENNACL_ERR_CHECK(err);
      profile_valid_ = true;
    }
    return profile_;
  }

  /** @brief Describes the resolution of device timer. This is measured in nanoseconds. */
  size_t profiling_timer_resolution() const
  {
    if (!profiling_timer_resolution_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_PROFILING_TIMER_RESOLUTION, sizeof(size_t), static_cast<void *>(&profiling_timer_resolution_), NULL);
      VIENNACL_ERR_CHECK(err);
      profiling_timer_resolution_valid_ = true;
    }
    return profiling_timer_resolution_;
  }

  /** @brief Describes the command-queue properties supported by the device.
      *
      * This is a bit-field that describes one or more of the following values:
      *   CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE
      *   CL_QUEUE_PROFILING_ENABLE3
      * These properties are described in the table for clCreateCommandQueue in the OpenCL standard.
      * The mandated minimum capability is CL_QUEUE_PROFILING_ENABLE.
      */
  cl_command_queue_properties queue_properties() const
  {
    if (!queue_properties_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_QUEUE_PROPERTIES, sizeof(cl_command_queue_properties), static_cast<void *>(&queue_properties_), NULL);
      VIENNACL_ERR_CHECK(err);
      queue_properties_valid_ = true;
    }
    return queue_properties_;
  }

  /** @brief Describes single precision floating-point capability of the OpenCL device.
      *
      * This is a bit-field that describes one or more of the following values:
      *   CL_FP_DENORM - denorms are supported.
      *   CL_FP_INF_NAN - INF and NaNs are supported.
      *   CL_FP_ROUND_TO_NEAREST - round to nearest even rounding mode supported.
      *   CL_FP_ROUND_TO_ZERO - round to zero rounding mode supported.
      *   CL_FP_ROUND_TO_INF - round to +ve and -ve infinity rounding modes supported.
      *   CP_FP_FMA - IEEE754-2008 fused multiply-add is supported.
      *   CL_FP_SOFT_FLOAT - Basic floating-point operations (such as addition, subtraction, multiplication) are implemented in software.
      *
      * The mandated minimum floating-point capability is CL_FP_ROUND_TO_NEAREST | CL_FP_INF_NAN.
      */
  cl_device_fp_config single_fp_config() const
  {
    if (!single_fp_config_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_SINGLE_FP_CONFIG, sizeof(cl_device_fp_config), static_cast<void *>(&single_fp_config_), NULL);
      VIENNACL_ERR_CHECK(err);
      single_fp_config_valid_ = true;
    }
    return single_fp_config_;
  }

  /** @brief The OpenCL device type.
      *
      * Currently supported values are one of or a combination of: CL_DEVICE_TYPE_CPU, CL_DEVICE_TYPE_GPU, CL_DEVICE_TYPE_ACCELERATOR, or CL_DEVICE_TYPE_DEFAULT.
      */
  cl_device_type type() const
  {
    if (!type_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_TYPE, sizeof(cl_device_type), static_cast<void *>(&type_), NULL);
      VIENNACL_ERR_CHECK(err);
      type_valid_ = true;
    }
    return type_;
  }

  /** @brief Vendor name string. */
  std::string vendor() const
  {
    if (!vendor_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_VENDOR, sizeof(char) * 256, static_cast<void *>(vendor_), NULL);
      VIENNACL_ERR_CHECK(err);
      vendor_valid_ = true;
    }
    return vendor_;
  }

  /** @brief A unique device vendor identifier. An example of a unique device identifier could be the PCIe ID. */
  cl_uint vendor_id() const
  {
    if (!vendor_id_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_VENDOR_ID, sizeof(cl_uint), static_cast<void *>(&vendor_id_), NULL);
      VIENNACL_ERR_CHECK(err);
      vendor_id_valid_ = true;
    }
    return vendor_id_;
  }

  /** @brief Vendor name string. */
  std::string version() const
  {
    if (!version_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DEVICE_VERSION, sizeof(char) * 256, static_cast<void *>(version_), NULL);
      VIENNACL_ERR_CHECK(err);
      version_valid_ = true;
    }
    return version_;
  }

  /** @brief Vendor name string. */
  std::string driver_version() const
  {
    if (!driver_version_valid_)
    {
      cl_int err = clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(char) * 256, static_cast<void *>(driver_version_), NULL);
      VIENNACL_ERR_CHECK(err);
      driver_version_valid_ = true;
    }
    return driver_version_;
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////


  /** @brief ViennaCL convenience function: Returns true if the device supports double precision */
  bool double_support() const
  {
    std::string ext = extensions();

    if (ext.find("cl_khr_fp64") != std::string::npos || ext.find("cl_amd_fp64") != std::string::npos)
      return true;

    return false;
  }

  /** @brief ViennaCL convenience function: Returns the device extension which enables double precision (usually cl_khr_fp64, but AMD used cl_amd_fp64 in the past) */
  std::string double_support_extension() const
  {
    std::string ext = extensions();

    if (ext.find("cl_amd_fp64") != std::string::npos) //AMD extension
      return "cl_amd_fp64";

    if (ext.find("cl_khr_fp64") != std::string::npos) //Khronos-certified standard extension for double precision
      return "cl_khr_fp64";

    return "";
  }

  /** @brief Returns the OpenCL device id */
  cl_device_id id() const
  {
    assert(device_ != 0 && bool("Device ID invalid!"));
    return device_;
  }

  /** @brief Returns an info string with a few properties of the device. Use full_info() to get all details.
      *
      * Returns the following device properties:
      * name, vendor, type, availability, max compute units, max work group size, global mem size, local mem size, local mem type, host unified memory
      *
      * @param indent      Number of optional blanks to be added at the start of each line
      * @param indent_char Character to be used for indenting
      */
  std::string info(vcl_size_t indent = 0, char indent_char = ' ') const
  {
    std::string line_indent(indent, indent_char);
    std::ostringstream oss;
    oss << line_indent << "Name:                " << name() << std::endl;
    oss << line_indent << "Vendor:              " << vendor() << std::endl;
    oss << line_indent << "Type:                " << device_type_to_string(type()) << std::endl;
    oss << line_indent << "Available:           " << available() << std::endl;
    oss << line_indent << "Max Compute Units:   " << max_compute_units() << std::endl;
    oss << line_indent << "Max Work Group Size: " << max_work_group_size() << std::endl;
    oss << line_indent << "Global Mem Size:     " << global_mem_size() << std::endl;
    oss << line_indent << "Local Mem Size:      " << local_mem_size() << std::endl;
    oss << line_indent << "Local Mem Type:      " << local_mem_type() << std::endl;
#ifdef CL_DEVICE_HOST_UNIFIED_MEMORY
    oss << line_indent << "Host Unified Memory: " << host_unified_memory() << std::endl;
#endif

    return oss.str();
  }

  /** @brief Returns an info string with all device properties defined in the OpenCL 1.1 standard, listed in alphabetical order. Use info() for a short overview.
    *
    * @param indent   Number of optional blanks to be added at the start of each line
    * @param indent_char Character to be used for indenting
    */
  std::string full_info(vcl_size_t indent = 0, char indent_char = ' ') const
  {
    std::string line_indent(indent, indent_char);
    std::ostringstream oss;
    oss << line_indent << "Address Bits:                  " << address_bits() << std::endl;
    oss << line_indent << "Available:                     " << available() << std::endl;
    oss << line_indent << "Compiler Available:            " << compiler_available() << std::endl;
#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
    oss << line_indent << "Double FP Config:              " << fp_config_to_string(double_fp_config()) << std::endl;
#endif
    oss << line_indent << "Endian Little:                 " << endian_little() << std::endl;
    oss << line_indent << "Error Correction Support:      " << error_correction_support() << std::endl;
    oss << line_indent << "Execution Capabilities:        " << exec_capabilities_to_string(execution_capabilities()) << std::endl;
    oss << line_indent << "Extensions:                    " << extensions() << std::endl;
    oss << line_indent << "Global Mem Cache Size:         " << global_mem_cache_size() << " Bytes" << std::endl;
    oss << line_indent << "Global Mem Cache Type:         " << mem_cache_type_to_string(global_mem_cache_type()) << std::endl;
    oss << line_indent << "Global Mem Cacheline Size:     " << global_mem_cacheline_size() << " Bytes" << std::endl;
    oss << line_indent << "Global Mem Size:               " << global_mem_size() << " Bytes" << std::endl;
#ifdef CL_DEVICE_HALF_FP_CONFIG
    oss << line_indent << "Half PF Config:                " << fp_config_to_string(half_fp_config()) << std::endl;
#endif
#ifdef CL_DEVICE_HOST_UNIFIED_MEMORY
    oss << line_indent << "Host Unified Memory:           " << host_unified_memory() << std::endl;
#endif
    oss << line_indent << "Image Support:                 " << image_support() << std::endl;
    oss << line_indent << "Image2D Max Height:            " << image2d_max_height() << std::endl;
    oss << line_indent << "Image2D Max Width:             " << image2d_max_width() << std::endl;
    oss << line_indent << "Image3D Max Depth:             " << image3d_max_depth() << std::endl;
    oss << line_indent << "Image3D Max Height:            " << image3d_max_height() << std::endl;
    oss << line_indent << "Image3D Max Width:             " << image3d_max_width() << std::endl;
    oss << line_indent << "Local Mem Size:                " << local_mem_size() << " Bytes" << std::endl;
    oss << line_indent << "Local Mem Type:                " << local_mem_type_to_string(local_mem_type()) << std::endl;
    oss << line_indent << "Max Clock Frequency:           " << max_clock_frequency() << " MHz" << std::endl;
    oss << line_indent << "Max Compute Units:             " << max_compute_units() << std::endl;
    oss << line_indent << "Max Constant Args:             " << max_constant_args() << std::endl;
    oss << line_indent << "Max Constant Buffer Size:      " << max_constant_buffer_size() << " Bytes" << std::endl;
    oss << line_indent << "Max Mem Alloc Size:            " << max_mem_alloc_size() << " Bytes" << std::endl;
    oss << line_indent << "Max Parameter Size:            " << max_parameter_size() << " Bytes" << std::endl;
    oss << line_indent << "Max Read Image Args:           " << max_read_image_args() << std::endl;
    oss << line_indent << "Max Samplers:                  " << max_samplers() << std::endl;
    oss << line_indent << "Max Work Group Size:           " << max_work_group_size() << std::endl;
    oss << line_indent << "Max Work Item Dimensions:      " << max_work_item_dimensions() << std::endl;
    oss << line_indent << "Max Work Item Sizes:           " << convert_to_string(max_work_item_sizes()) << std::endl;
    oss << line_indent << "Max Write Image Args:          " << max_write_image_args() << std::endl;
    oss << line_indent << "Mem Base Addr Align:           " << mem_base_addr_align() << std::endl;
    oss << line_indent << "Min Data Type Align Size:      " << min_data_type_align_size() << " Bytes" << std::endl;
    oss << line_indent << "Name:                          " << name() << std::endl;
#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_CHAR
    oss << line_indent << "Native Vector Width char:      " << native_vector_width_char() << std::endl;
#endif
#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_SHORT
    oss << line_indent << "Native Vector Width short:     " << native_vector_width_short() << std::endl;
#endif
#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_INT
    oss << line_indent << "Native Vector Width int:       " << native_vector_width_int() << std::endl;
#endif
#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_LONG
    oss << line_indent << "Native Vector Width long:      " << native_vector_width_long() << std::endl;
#endif
#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT
    oss << line_indent << "Native Vector Width float:     " << native_vector_width_float() << std::endl;
#endif
#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_DOUBLE
    oss << line_indent << "Native Vector Width double:    " << native_vector_width_double() << std::endl;
#endif
#ifdef CL_DEVICE_NATIVE_VECTOR_WIDTH_HALF
    oss << line_indent << "Native Vector Width half:      " << native_vector_width_half() << std::endl;
#endif
#ifdef CL_DEVICE_OPENCL_C_VERSION
    oss << line_indent << "OpenCL C Version:              " << opencl_c_version() << std::endl;
#endif
    oss << line_indent << "Platform:                      " << platform() << std::endl;
    oss << line_indent << "Preferred Vector Width char:   " << preferred_vector_width_char() << std::endl;
    oss << line_indent << "Preferred Vector Width short:  " << preferred_vector_width_short() << std::endl;
    oss << line_indent << "Preferred Vector Width int:    " << preferred_vector_width_int() << std::endl;
    oss << line_indent << "Preferred Vector Width long:   " << preferred_vector_width_long() << std::endl;
    oss << line_indent << "Preferred Vector Width float:  " << preferred_vector_width_float() << std::endl;
    oss << line_indent << "Preferred Vector Width double: " << preferred_vector_width_double() << std::endl;
#ifdef CL_DEVICE_PREFERRED_VECTOR_WIDTH_HALF
    oss << line_indent << "Preferred Vector Width half:   " << preferred_vector_width_half() << std::endl;
#endif
    oss << line_indent << "Profile:                       " << profile() << std::endl;
    oss << line_indent << "Profiling Timer Resolution:    " << profiling_timer_resolution() << " ns" << std::endl;
    oss << line_indent << "Queue Properties:              " << queue_properties_to_string(queue_properties()) << std::endl;
    oss << line_indent << "Single FP Config:              " << fp_config_to_string(single_fp_config()) << std::endl;
    oss << line_indent << "Type:                          " << device_type_to_string(type()) << std::endl;
    oss << line_indent << "Vendor:                        " << vendor() << std::endl;
    oss << line_indent << "Vendor ID:                     " << vendor_id() << std::endl;
    oss << line_indent << "Version:                       " << version() << std::endl;
    oss << line_indent << "Driver Version:                " << driver_version() << std::endl;

    return oss.str();
  }

  bool operator==(device const & other) const
  {
    return device_ == other.device_;
  }

  bool operator==(cl_device_id other) const
  {
    return device_ == other;
  }

  /** @brief Helper function converting a floating point configuration to a string */
  std::string fp_config_to_string(cl_device_fp_config conf) const
  {
    std::ostringstream oss;
    if (conf & CL_FP_DENORM)
      oss << "CL_FP_DENORM ";
    if (conf & CL_FP_INF_NAN)
      oss << "CL_FP_INF_NAN ";
    if (conf & CL_FP_ROUND_TO_NEAREST)
      oss << "CL_FP_ROUND_TO_NEAREST ";
    if (conf & CL_FP_ROUND_TO_ZERO)
      oss << "CL_FP_ROUND_TO_ZERO ";
    if (conf & CL_FP_ROUND_TO_INF)
      oss << "CL_FP_ROUND_TO_INF ";
    if (conf & CL_FP_FMA)
      oss << "CL_FP_FMA ";
#ifdef CL_FP_SOFT_FLOAT
    if (conf & CL_FP_SOFT_FLOAT)
      oss << "CL_FP_SOFT_FLOAT ";
#endif

    return oss.str();
  }

  std::string exec_capabilities_to_string(cl_device_exec_capabilities cap) const
  {
    std::ostringstream oss;
    if (cap & CL_EXEC_KERNEL)
      oss << "CL_EXEC_KERNEL ";
    if (cap & CL_EXEC_NATIVE_KERNEL)
      oss << "CL_EXEC_NATIVE_KERNEL ";

    return oss.str();
  }

  std::string mem_cache_type_to_string(cl_device_mem_cache_type cachetype) const
  {
    std::ostringstream oss;
    if (cachetype == CL_NONE)
      oss << "CL_NONE ";
    else if (cachetype == CL_READ_ONLY_CACHE)
      oss << "CL_READ_ONLY_CACHE ";
    else if (cachetype == CL_READ_WRITE_CACHE)
      oss << "CL_READ_WRITE_CACHE ";

    return oss.str();
  }

  std::string local_mem_type_to_string(cl_device_local_mem_type loc_mem_type) const
  {
    std::ostringstream oss;
    if (loc_mem_type & CL_LOCAL)
      oss << "CL_LOCAL ";
    if (loc_mem_type & CL_GLOBAL)
      oss << "CL_GLOBAL ";

    return oss.str();
  }

  std::string convert_to_string(std::vector<size_t> const & vec) const
  {
    std::ostringstream oss;
    for (vcl_size_t i=0; i<vec.size(); ++i)
      oss << vec[i] << " ";

    return oss.str();
  }

  std::string queue_properties_to_string(cl_command_queue_properties queue_prop) const
  {
    std::ostringstream oss;
    if (queue_prop & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
      oss << "CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE ";
    if (queue_prop & CL_QUEUE_PROFILING_ENABLE)
      oss << "CL_QUEUE_PROFILING_ENABLE ";

    return oss.str();
  }

  std::string device_type_to_string(cl_device_type dev_type) const
  {
    std::ostringstream oss;
    if (dev_type & CL_DEVICE_TYPE_GPU)
      oss << "GPU ";
    if (dev_type & CL_DEVICE_TYPE_CPU)
      oss << "CPU ";
    if (dev_type & CL_DEVICE_TYPE_ACCELERATOR)
      oss << "Accelerator ";
    if (dev_type & CL_DEVICE_TYPE_DEFAULT)
      oss << "(default)";

    return oss.str();
  }

private:

  void flush_cache()
  {
    address_bits_valid_       = false;
    architecture_family_valid_ = false;
    available_valid_          = false;
    compiler_available_valid_ = false;
#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
    double_fp_config_valid_   = false;
#endif
    endian_little_valid_      = false;
    error_correction_support_valid_  = false;
    execution_capabilities_valid_    = false;
    extensions_valid_                = false;
    global_mem_cache_size_valid_     = false;
    global_mem_cache_type_valid_     = false;
    global_mem_cacheline_size_valid_ = false;
    global_mem_size_valid_           = false;
#ifdef CL_DEVICE_HALF_FP_CONFIG
    half_fp_config_valid_      = false;
#endif
    host_unified_memory_valid_ = false;
    image_support_valid_       = false;
    image2d_max_height_valid_  = false;
    image2d_max_width_valid_   = false;
    image3d_max_depth_valid_   = false;
    image3d_max_height_valid_  = false;
    image3d_max_width_valid_   = false;
    local_mem_size_valid_      = false;
    local_mem_type_valid_      = false;
    max_clock_frequency_valid_ = false;
    max_compute_units_valid_   = false;
    max_constant_args_valid_   = false;
    max_constant_buffer_size_valid_ = false;
    max_mem_alloc_size_valid_  = false;
    max_parameter_size_valid_  = false;
    max_read_image_args_valid_ = false;
    max_samplers_valid_        = false;
    max_work_group_size_valid_ = false;
    max_work_item_dimensions_valid_ = false;
    max_work_item_sizes_valid_  = false;
    max_write_image_args_valid_ = false;
    mem_base_addr_align_valid_  = false;
    min_data_type_align_size_valid_ = false;
    name_valid_ = false;
    native_vector_width_char_valid_   = false;
    native_vector_width_short_valid_  = false;
    native_vector_width_int_valid_    = false;
    native_vector_width_long_valid_   = false;
    native_vector_width_float_valid_  = false;
    native_vector_width_double_valid_ = false;
    native_vector_width_half_valid_   = false;
    opencl_c_version_valid_ = false;
    platform_valid_ = false;
    preferred_vector_width_char_valid_   = false;
    preferred_vector_width_short_valid_  = false;
    preferred_vector_width_int_valid_    = false;
    preferred_vector_width_long_valid_   = false;
    preferred_vector_width_float_valid_  = false;
    preferred_vector_width_double_valid_ = false;
    preferred_vector_width_half_valid_   = false;
    profile_valid_ = false;
    profiling_timer_resolution_valid_ = false;
    queue_properties_valid_ = false;
    single_fp_config_valid_ = false;
    type_valid_             = false;
    vendor_valid_           = false;
    vendor_id_valid_        = false;
    version_valid_          = false;
    driver_version_valid_   = false;
  }

  cl_device_id    device_;

  //
  // Device information supported by OpenCL 1.0 to follow
  // cf. http://www.khronos.org/registry/cl/sdk/1.0/docs/man/xhtml/clGetDeviceInfo.html
  // Note that all members are declared 'mutable', as they represent a caching mechanism in order to circumvent repeated potentially expensive calls to the OpenCL SDK
  //

  mutable bool    address_bits_valid_;
  mutable cl_uint address_bits_;

  mutable bool    available_valid_;
  mutable cl_bool available_;

  mutable bool    compiler_available_valid_;
  mutable cl_bool compiler_available_;

#ifdef CL_DEVICE_DOUBLE_FP_CONFIG
  mutable bool                double_fp_config_valid_;
  mutable cl_device_fp_config double_fp_config_;
#endif

  mutable bool    endian_little_valid_;
  mutable cl_bool endian_little_;

  mutable bool    error_correction_support_valid_;
  mutable cl_bool error_correction_support_;

  mutable bool                        execution_capabilities_valid_;
  mutable cl_device_exec_capabilities execution_capabilities_;

  mutable bool extensions_valid_;
  mutable char extensions_[2048];    // don't forget to adjust member function accordingly when changing array size

  mutable bool     global_mem_cache_size_valid_;
  mutable cl_ulong global_mem_cache_size_;

  mutable bool                     global_mem_cache_type_valid_;
  mutable cl_device_mem_cache_type global_mem_cache_type_;

  mutable bool    global_mem_cacheline_size_valid_;
  mutable cl_uint global_mem_cacheline_size_;

  mutable bool     global_mem_size_valid_;
  mutable cl_ulong global_mem_size_;

#ifdef CL_DEVICE_HALF_FP_CONFIG
  mutable bool                half_fp_config_valid_;
  mutable cl_device_fp_config half_fp_config_;
#endif

  mutable bool    host_unified_memory_valid_;
  mutable cl_bool host_unified_memory_;

  mutable bool    image_support_valid_;
  mutable cl_bool image_support_;

  mutable bool   image2d_max_height_valid_;
  mutable size_t image2d_max_height_;

  mutable bool   image2d_max_width_valid_;
  mutable size_t image2d_max_width_;

  mutable bool   image3d_max_depth_valid_;
  mutable size_t image3d_max_depth_;

  mutable bool   image3d_max_height_valid_;
  mutable size_t image3d_max_height_;

  mutable bool   image3d_max_width_valid_;
  mutable size_t image3d_max_width_;

  mutable bool     local_mem_size_valid_;
  mutable cl_ulong local_mem_size_;

  mutable bool                     local_mem_type_valid_;
  mutable cl_device_local_mem_type local_mem_type_;

  mutable bool    max_clock_frequency_valid_;
  mutable cl_uint max_clock_frequency_;

  mutable bool    max_compute_units_valid_;
  mutable cl_uint max_compute_units_;

  mutable bool    max_constant_args_valid_;
  mutable cl_uint max_constant_args_;

  mutable bool     max_constant_buffer_size_valid_;
  mutable cl_ulong max_constant_buffer_size_;

  mutable bool     max_mem_alloc_size_valid_;
  mutable cl_ulong max_mem_alloc_size_;

  mutable bool   max_parameter_size_valid_;
  mutable size_t max_parameter_size_;

  mutable bool    max_read_image_args_valid_;
  mutable cl_uint max_read_image_args_;

  mutable bool    max_samplers_valid_;
  mutable cl_uint max_samplers_;

  mutable bool   max_work_group_size_valid_;
  mutable size_t max_work_group_size_;

  mutable bool    max_work_item_dimensions_valid_;
  mutable cl_uint max_work_item_dimensions_;

  mutable bool   max_work_item_sizes_valid_;
  mutable size_t max_work_item_sizes_[16];   //we do not support execution models with more than 16 dimensions. This should totally suffice in practice, though.

  mutable bool    max_write_image_args_valid_;
  mutable cl_uint max_write_image_args_;

  mutable bool    mem_base_addr_align_valid_;
  mutable cl_uint mem_base_addr_align_;

  mutable bool    min_data_type_align_size_valid_;
  mutable cl_uint min_data_type_align_size_;

  mutable bool name_valid_;
  mutable char name_[256];    // don't forget to adjust member function accordingly when changing array size

  mutable bool    native_vector_width_char_valid_;
  mutable cl_uint native_vector_width_char_;

  mutable bool    native_vector_width_short_valid_;
  mutable cl_uint native_vector_width_short_;

  mutable bool    native_vector_width_int_valid_;
  mutable cl_uint native_vector_width_int_;

  mutable bool    native_vector_width_long_valid_;
  mutable cl_uint native_vector_width_long_;

  mutable bool    native_vector_width_float_valid_;
  mutable cl_uint native_vector_width_float_;

  mutable bool    native_vector_width_double_valid_;
  mutable cl_uint native_vector_width_double_;

  mutable bool    native_vector_width_half_valid_;
  mutable cl_uint native_vector_width_half_;

  mutable bool opencl_c_version_valid_;
  mutable char opencl_c_version_[128];    // don't forget to adjust member function accordingly when changing array size

  mutable bool           platform_valid_;
  mutable cl_platform_id platform_;

  mutable bool    preferred_vector_width_char_valid_;
  mutable cl_uint preferred_vector_width_char_;

  mutable bool    preferred_vector_width_short_valid_;
  mutable cl_uint preferred_vector_width_short_;

  mutable bool    preferred_vector_width_int_valid_;
  mutable cl_uint preferred_vector_width_int_;

  mutable bool    preferred_vector_width_long_valid_;
  mutable cl_uint preferred_vector_width_long_;

  mutable bool    preferred_vector_width_float_valid_;
  mutable cl_uint preferred_vector_width_float_;

  mutable bool    preferred_vector_width_double_valid_;
  mutable cl_uint preferred_vector_width_double_;

  mutable bool    preferred_vector_width_half_valid_;
  mutable cl_uint preferred_vector_width_half_;

  mutable bool profile_valid_;
  mutable char profile_[32];    // don't forget to adjust member function accordingly when changing array size

  mutable bool   profiling_timer_resolution_valid_;
  mutable size_t profiling_timer_resolution_;

  mutable bool                        queue_properties_valid_;
  mutable cl_command_queue_properties queue_properties_;

  mutable bool                single_fp_config_valid_;
  mutable cl_device_fp_config single_fp_config_;

  mutable bool           type_valid_;
  mutable cl_device_type type_;

  mutable bool vendor_valid_;
  mutable char vendor_[256];    // don't forget to adjust member function accordingly when changing array size

  mutable bool    vendor_id_valid_;
  mutable cl_uint vendor_id_;

  mutable bool version_valid_;
  mutable char version_[256];    // don't forget to adjust member function accordingly when changing array size

  mutable bool driver_version_valid_;
  mutable char driver_version_[256];    // don't forget to adjust member function accordingly when changing array size

  mutable bool architecture_family_valid_;
  mutable device_architecture_family architecture_family_;
};

} //namespace ocl
} //namespace viennacl

#endif
