#ifndef VIENNACL_LINALG_OPENCL_KERNELS_COMPRESSED_COMPRESSED_MATRIX_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_COMPRESSED_COMPRESSED_MATRIX_HPP

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

/** @file viennacl/linalg/opencl/kernels/compressed_compressed_matrix.hpp
 *  @brief OpenCL kernel file for vector operations */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{

//////////////////////////// Part 1: Kernel generation routines ////////////////////////////////////

template<typename StringT>
void generate_vec_mul(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void vec_mul( \n");
  source.append("  __global const unsigned int * row_jumper, \n");
  source.append("  __global const unsigned int * row_indices, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  uint nonzero_rows, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * x, \n");
  source.append("  uint4 layout_x, \n");
  source.append("  "); source.append(numeric_string); source.append(" alpha, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * result, \n");
  source.append("  uint4 layout_result, \n");
  source.append("  "); source.append(numeric_string); source.append(" beta) \n");
  source.append("{ \n");
  source.append("  for (unsigned int i = get_global_id(0); i < nonzero_rows; i += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    "); source.append(numeric_string); source.append(" dot_prod = 0; \n");
  source.append("    unsigned int row_end = row_jumper[i+1]; \n");
  source.append("    for (unsigned int j = row_jumper[i]; j < row_end; ++j) \n");
  source.append("      dot_prod += elements[j] * x[column_indices[j] * layout_x.y + layout_x.x]; \n");

  source.append("    if (beta != 0) result[row_indices[i] * layout_result.y + layout_result.x] += alpha * dot_prod; \n");
  source.append("    else           result[row_indices[i] * layout_result.y + layout_result.x]  = alpha * dot_prod; \n");
  source.append("  } \n");
  source.append(" } \n");
}

//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

/** @brief Main kernel class for generating OpenCL kernels for compressed_compressed_matrix. */
template<typename NumericT>
struct compressed_compressed_matrix
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_compressed_compressed_matrix";
  }

  static void init(viennacl::ocl::context & ctx)
  {
    static std::map<cl_context, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
      std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();

      std::string source;
      source.reserve(8192);

      viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

      // fully parametrized kernels:
      generate_vec_mul(source, numeric_string);

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

