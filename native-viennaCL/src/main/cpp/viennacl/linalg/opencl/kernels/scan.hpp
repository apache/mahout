#ifndef VIENNACL_LINALG_OPENCL_KERNELS_SCAN_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_SCAN_HPP

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

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

/** @file viennacl/linalg/opencl/kernels/scan.hpp
 *  @brief OpenCL kernel file for scan operations. To be merged back to vector operations. */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{


template <typename StringType>
void generate_scan_kernel_1(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void scan_1(__global "); source.append(numeric_string); source.append("* X, \n");
  source.append("                     unsigned int startX, \n");
  source.append("                     unsigned int incX, \n");
  source.append("                     unsigned int sizeX, \n");

  source.append("                     __global "); source.append(numeric_string); source.append("* Y, \n");
  source.append("                     unsigned int startY, \n");
  source.append("                     unsigned int incY, \n");

  source.append("                     unsigned int scan_offset, \n"); // 0 for inclusive scan, 1 for exclusive scan
  source.append("                     __global "); source.append(numeric_string); source.append("* carries) { \n");

  source.append("  __local "); source.append(numeric_string); source.append(" shared_buffer[256]; \n");
  source.append("  "); source.append(numeric_string); source.append(" my_value; \n");

  source.append("  unsigned int work_per_thread = (sizeX - 1) / get_global_size(0) + 1; \n");
  source.append("  unsigned int block_start = work_per_thread * get_local_size(0) *  get_group_id(0); \n");
  source.append("  unsigned int block_stop  = work_per_thread * get_local_size(0) * (get_group_id(0) + 1); \n");
  source.append("  unsigned int block_offset = 0; \n");

  // run scan on each section:
  source.append("  for (unsigned int i = block_start + get_local_id(0); i < block_stop; i += get_local_size(0)) { \n");

  // load data
  source.append("    my_value = (i < sizeX) ? X[i * incX + startX] : 0; \n");

  // inclusive scan in shared buffer:
  source.append("    for(unsigned int stride = 1; stride < get_local_size(0); stride *= 2) { \n");
  source.append("       barrier(CLK_LOCAL_MEM_FENCE);   \n");
  source.append("       shared_buffer[get_local_id(0)] = my_value;   \n");
  source.append("       barrier(CLK_LOCAL_MEM_FENCE);   \n");
  source.append("       if (get_local_id(0) >= stride)   \n");
  source.append("         my_value += shared_buffer[get_local_id(0) - stride];   \n");
  source.append("    } \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE);   \n");
  source.append("    shared_buffer[get_local_id(0)] = my_value;   \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE);   \n");

  // write to output array:
  source.append("    if (scan_offset > 0) \n");
  source.append("      my_value = (get_local_id(0) > 0) ? shared_buffer[get_local_id(0) - 1] : 0; \n");

  source.append("    if (i < sizeX) \n");
  source.append("      Y[i * incY + startY] = block_offset + my_value; \n");

  source.append("    block_offset += shared_buffer[get_local_size(0)-1]; \n");
  source.append("  } \n");

  // write carry:
  source.append("  if (get_local_id(0) == 0) carries[get_group_id(0)] = block_offset; \n");

  source.append("} \n");
}

template <typename StringType>
void generate_scan_kernel_2(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void scan_2(__global "); source.append(numeric_string); source.append("* carries) { \n");

  source.append("  __local "); source.append(numeric_string); source.append(" shared_buffer[256]; \n");       //section size

  // load data
  source.append("  "); source.append(numeric_string); source.append(" my_carry = carries[get_local_id(0)]; \n");

  // scan in shared buffer:
  source.append("  for(unsigned int stride = 1; stride < get_local_size(0); stride *= 2) { \n");
  source.append("     barrier(CLK_LOCAL_MEM_FENCE);   \n");
  source.append("     shared_buffer[get_local_id(0)] = my_carry;   \n");
  source.append("     barrier(CLK_LOCAL_MEM_FENCE);   \n");
  source.append("     if (get_local_id(0) >= stride)   \n");
  source.append("       my_carry += shared_buffer[get_local_id(0) - stride];   \n");
  source.append("  } \n");
  source.append("  barrier(CLK_LOCAL_MEM_FENCE);   \n");
  source.append("  shared_buffer[get_local_id(0)] = my_carry;   \n");
  source.append("  barrier(CLK_LOCAL_MEM_FENCE);   \n");

  // write to output array:
  source.append("  carries[get_local_id(0)] = (get_local_id(0) > 0) ? shared_buffer[get_local_id(0) - 1] : 0;  \n");

  source.append("} \n");
}

template <typename StringType>
void generate_scan_kernel_3(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void scan_3(__global "); source.append(numeric_string); source.append(" * Y, \n");
  source.append("                     unsigned int startY, \n");
  source.append("                     unsigned int incY, \n");
  source.append("                     unsigned int sizeY, \n");

  source.append("                     __global "); source.append(numeric_string); source.append("* carries) { \n");

  source.append("  unsigned int work_per_thread = (sizeY - 1) / get_global_size(0) + 1; \n");
  source.append("  unsigned int block_start = work_per_thread * get_local_size(0) *  get_group_id(0); \n");
  source.append("  unsigned int block_stop  = work_per_thread * get_local_size(0) * (get_group_id(0) + 1); \n");

  source.append("  __local "); source.append(numeric_string); source.append(" shared_offset; \n");

  source.append("  if (get_local_id(0) == 0) shared_offset = carries[get_group_id(0)]; \n");
  source.append("  barrier(CLK_LOCAL_MEM_FENCE);   \n");

  source.append("  for (unsigned int i = block_start + get_local_id(0); i < block_stop; i += get_local_size(0)) \n");
  source.append("    if (i < sizeY) \n");
  source.append("      Y[i * incY + startY] += shared_offset; \n");

  source.append("} \n");
}




// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for singular value decomposition of dense matrices. */
template<typename NumericT>
struct scan
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_scan";
  }

  static void init(viennacl::ocl::context & ctx)
  {
    static std::map<cl_context, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
      std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();

      std::string source;
      source.reserve(1024);

      viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

      generate_scan_kernel_1(source, numeric_string);
      generate_scan_kernel_2(source, numeric_string);
      generate_scan_kernel_3(source, numeric_string);

      std::string prog_name = program_name();
      #ifdef VIENNACL_BUILD_INFO
      std::cout << "Creating program " << prog_name << std::endl;
      #endif
      ctx.add_program(source, prog_name);
      init_done[ctx.handle().get()] = true;
    } //if
  } //init
};

}  // namespace kernels
}  // namespace opencl
}  // namespace linalg
}  // namespace viennacl
#endif

