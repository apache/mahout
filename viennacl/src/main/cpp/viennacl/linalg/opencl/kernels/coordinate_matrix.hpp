#ifndef VIENNACL_LINALG_OPENCL_KERNELS_COORDINATE_MATRIX_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_COORDINATE_MATRIX_HPP

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

#include "viennacl/linalg/opencl/common.hpp"

/** @file viennacl/linalg/opencl/kernels/coordinate_matrix.hpp
 *  @brief OpenCL kernel file for coordinate_matrix operations */
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
void generate_coordinate_matrix_vec_mul(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void vec_mul( \n");
  source.append("  __global const uint2 * coords,  \n");//(row_index, column_index)
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const uint  * group_boundaries, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * x, \n");
  source.append("  uint4 layout_x, \n");
  source.append("  "); source.append(numeric_string); source.append(" alpha, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * result, \n");
  source.append("  uint4 layout_result, \n");
  source.append("  "); source.append(numeric_string); source.append(" beta, \n");
  source.append("  __local unsigned int * shared_rows, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * inter_results) \n");
  source.append("{ \n");
  source.append("  uint2 tmp; \n");
  source.append("  "); source.append(numeric_string); source.append(" val; \n");
  source.append("  uint group_start = group_boundaries[get_group_id(0)]; \n");
  source.append("  uint group_end   = group_boundaries[get_group_id(0) + 1]; \n");
  source.append("  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : 0; \n");   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)

  source.append("  uint local_index = 0; \n");

  source.append("  for (uint k = 0; k < k_end; ++k) { \n");
  source.append("    local_index = group_start + k * get_local_size(0) + get_local_id(0); \n");

  source.append("    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0; \n");
  source.append("    val = (local_index < group_end) ? elements[local_index] * x[tmp.y * layout_x.y + layout_x.x] : 0; \n");

  //check for carry from previous loop run:
  source.append("    if (get_local_id(0) == 0 && k > 0) { \n");
  source.append("      if (tmp.x == shared_rows[get_local_size(0)-1]) \n");
  source.append("        val += inter_results[get_local_size(0)-1]; \n");
  source.append("      else if (beta != 0) \n");
  source.append("        result[shared_rows[get_local_size(0)-1] * layout_result.y + layout_result.x] += alpha * inter_results[get_local_size(0)-1]; \n");
  source.append("      else \n");
  source.append("        result[shared_rows[get_local_size(0)-1] * layout_result.y + layout_result.x]  = alpha * inter_results[get_local_size(0)-1]; \n");
  source.append("    } \n");

  //segmented parallel reduction begin
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    shared_rows[get_local_id(0)] = tmp.x; \n");
  source.append("    inter_results[get_local_id(0)] = val; \n");
  source.append("    "); source.append(numeric_string); source.append(" left = 0; \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");

  source.append("    for (unsigned int stride = 1; stride < get_local_size(0); stride *= 2) { \n");
  source.append("      left = (get_local_id(0) >= stride && tmp.x == shared_rows[get_local_id(0) - stride]) ? inter_results[get_local_id(0) - stride] : 0; \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("      inter_results[get_local_id(0)] += left; \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    } \n");
  //segmented parallel reduction end

  source.append("    if (local_index < group_end - 1 && get_local_id(0) < get_local_size(0) - 1 && \n");
  source.append("      shared_rows[get_local_id(0)] != shared_rows[get_local_id(0) + 1]) { \n");
  source.append("      if (beta != 0) result[tmp.x * layout_result.y + layout_result.x] += alpha * inter_results[get_local_id(0)]; \n");
  source.append("      else           result[tmp.x * layout_result.y + layout_result.x]  = alpha * inter_results[get_local_id(0)]; \n");
  source.append("    } \n");

  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("  }  \n"); //for k

  source.append("  if (local_index + 1 == group_end) {\n");  //write results of last active entry (this may not necessarily be the case already)
  source.append("    if (beta != 0) result[tmp.x * layout_result.y + layout_result.x] += alpha * inter_results[get_local_id(0)]; \n");
  source.append("    else           result[tmp.x * layout_result.y + layout_result.x]  = alpha * inter_results[get_local_id(0)]; \n");
  source.append("  } \n");
  source.append("} \n");

}

namespace detail
{
  /** @brief Generate kernel for C = A * B with A being a compressed_matrix, B and C dense */
  template<typename StringT>
  void generate_coordinate_matrix_dense_matrix_mul(StringT & source, std::string const & numeric_string,
                                                   bool B_transposed, bool B_row_major, bool C_row_major)
  {
    source.append("__kernel void ");
    source.append(viennacl::linalg::opencl::detail::sparse_dense_matmult_kernel_name(B_transposed, B_row_major, C_row_major));
    source.append("( \n");
    source.append("  __global const uint2 * coords,  \n");//(row_index, column_index)
    source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
    source.append("  __global const uint  * group_boundaries, \n");
    source.append("  __global const "); source.append(numeric_string); source.append(" * d_mat, \n");
    source.append("  unsigned int d_mat_row_start, \n");
    source.append("  unsigned int d_mat_col_start, \n");
    source.append("  unsigned int d_mat_row_inc, \n");
    source.append("  unsigned int d_mat_col_inc, \n");
    source.append("  unsigned int d_mat_row_size, \n");
    source.append("  unsigned int d_mat_col_size, \n");
    source.append("  unsigned int d_mat_internal_rows, \n");
    source.append("  unsigned int d_mat_internal_cols, \n");
    source.append("  __global "); source.append(numeric_string); source.append(" * result, \n");
    source.append("  unsigned int result_row_start, \n");
    source.append("  unsigned int result_col_start, \n");
    source.append("  unsigned int result_row_inc, \n");
    source.append("  unsigned int result_col_inc, \n");
    source.append("  unsigned int result_row_size, \n");
    source.append("  unsigned int result_col_size, \n");
    source.append("  unsigned int result_internal_rows, \n");
    source.append("  unsigned int result_internal_cols, \n");
    source.append("  __local unsigned int * shared_rows, \n");
    source.append("  __local "); source.append(numeric_string); source.append(" * inter_results) \n");
    source.append("{ \n");
    source.append("  uint2 tmp; \n");
    source.append("  "); source.append(numeric_string); source.append(" val; \n");
    source.append("  uint group_start = group_boundaries[get_group_id(0)]; \n");
    source.append("  uint group_end   = group_boundaries[get_group_id(0) + 1]; \n");
    source.append("  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : 0; \n");   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)

    source.append("  uint local_index = 0; \n");

    source.append("  for (uint result_col = 0; result_col < result_col_size; ++result_col) { \n");
    source.append("   for (uint k = 0; k < k_end; ++k) { \n");
    source.append("    local_index = group_start + k * get_local_size(0) + get_local_id(0); \n");

    source.append("    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0; \n");
    if (B_transposed && B_row_major)
      source.append("    val = (local_index < group_end) ? elements[local_index] * d_mat[ (d_mat_row_start + result_col * d_mat_row_inc) * d_mat_internal_cols + d_mat_col_start +      tmp.y * d_mat_col_inc ] : 0; \n");
    else if (B_transposed && !B_row_major)
      source.append("    val = (local_index < group_end) ? elements[local_index] * d_mat[ (d_mat_row_start + result_col * d_mat_row_inc)                       + (d_mat_col_start +      tmp.y * d_mat_col_inc) * d_mat_internal_rows ] : 0; \n");
    else if (!B_transposed && B_row_major)
      source.append("    val = (local_index < group_end) ? elements[local_index] * d_mat[ (d_mat_row_start +      tmp.y * d_mat_row_inc) * d_mat_internal_cols + d_mat_col_start + result_col * d_mat_col_inc ] : 0; \n");
    else
      source.append("    val = (local_index < group_end) ? elements[local_index] * d_mat[ (d_mat_row_start +      tmp.y * d_mat_row_inc)                       + (d_mat_col_start + result_col * d_mat_col_inc) * d_mat_internal_rows ] : 0; \n");

    //check for carry from previous loop run:
    source.append("    if (get_local_id(0) == 0 && k > 0) { \n");
    source.append("      if (tmp.x == shared_rows[get_local_size(0)-1]) \n");
    source.append("        val += inter_results[get_local_size(0)-1]; \n");
    source.append("      else \n");
    if (C_row_major)
      source.append("        result[(shared_rows[get_local_size(0)-1] * result_row_inc + result_row_start) * result_internal_cols + result_col_start + result_col * result_col_inc ] = inter_results[get_local_size(0)-1]; \n");
    else
      source.append("        result[(shared_rows[get_local_size(0)-1] * result_row_inc + result_row_start)                        + (result_col_start + result_col * result_col_inc) * result_internal_rows ] = inter_results[get_local_size(0)-1]; \n");
    source.append("    } \n");

    //segmented parallel reduction begin
    source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
    source.append("    shared_rows[get_local_id(0)] = tmp.x; \n");
    source.append("    inter_results[get_local_id(0)] = val; \n");
    source.append("    "); source.append(numeric_string); source.append(" left = 0; \n");
    source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");

    source.append("    for (unsigned int stride = 1; stride < get_local_size(0); stride *= 2) { \n");
    source.append("      left = (get_local_id(0) >= stride && tmp.x == shared_rows[get_local_id(0) - stride]) ? inter_results[get_local_id(0) - stride] : 0; \n");
    source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
    source.append("      inter_results[get_local_id(0)] += left; \n");
    source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
    source.append("    } \n");
    //segmented parallel reduction end

    source.append("    if (local_index < group_end && get_local_id(0) < get_local_size(0) - 1 && \n");
    source.append("      shared_rows[get_local_id(0)] != shared_rows[get_local_id(0) + 1]) { \n");
    if (C_row_major)
      source.append("      result[(tmp.x * result_row_inc + result_row_start) * result_internal_cols + result_col_start + result_col * result_col_inc ] = inter_results[get_local_id(0)]; \n");
    else
      source.append("      result[(tmp.x * result_row_inc + result_row_start)                        + (result_col_start + result_col * result_col_inc) * result_internal_rows ] = inter_results[get_local_id(0)]; \n");
    source.append("    } \n");

    source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
    source.append("   }  \n"); //for k

    source.append("   if (local_index + 1 == group_end) \n");  //write results of last active entry (this may not necessarily be the case already)
    if (C_row_major)
      source.append("    result[(tmp.x  * result_row_inc + result_row_start) * result_internal_cols + result_col_start + result_col * result_col_inc ] = inter_results[get_local_id(0)]; \n");
    else
      source.append("    result[(tmp.x  * result_row_inc + result_row_start)                        + (result_col_start + result_col * result_col_inc) * result_internal_rows ] = inter_results[get_local_id(0)]; \n");
    source.append("  } \n"); //for result_col
    source.append("} \n");

  }
}

template<typename StringT>
void generate_coordinate_matrix_dense_matrix_multiplication(StringT & source, std::string const & numeric_string)
{
  detail::generate_coordinate_matrix_dense_matrix_mul(source, numeric_string, false, false, false);
  detail::generate_coordinate_matrix_dense_matrix_mul(source, numeric_string, false, false,  true);
  detail::generate_coordinate_matrix_dense_matrix_mul(source, numeric_string, false,  true, false);
  detail::generate_coordinate_matrix_dense_matrix_mul(source, numeric_string, false,  true,  true);

  detail::generate_coordinate_matrix_dense_matrix_mul(source, numeric_string, true, false, false);
  detail::generate_coordinate_matrix_dense_matrix_mul(source, numeric_string, true, false,  true);
  detail::generate_coordinate_matrix_dense_matrix_mul(source, numeric_string, true,  true, false);
  detail::generate_coordinate_matrix_dense_matrix_mul(source, numeric_string, true,  true,  true);
}

template<typename StringT>
void generate_coordinate_matrix_row_info_extractor(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void row_info_extractor( \n");
  source.append("          __global const uint2 * coords,  \n");//(row_index, column_index)
  source.append("          __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("          __global const uint  * group_boundaries, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * result, \n");
  source.append("          unsigned int option, \n");
  source.append("          __local unsigned int * shared_rows, \n");
  source.append("          __local "); source.append(numeric_string); source.append(" * inter_results) \n");
  source.append("{ \n");
  source.append("  uint2 tmp; \n");
  source.append("  "); source.append(numeric_string); source.append(" val; \n");
  source.append("  uint last_index  = get_local_size(0) - 1; \n");
  source.append("  uint group_start = group_boundaries[get_group_id(0)]; \n");
  source.append("  uint group_end   = group_boundaries[get_group_id(0) + 1]; \n");
  source.append("  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : ("); source.append(numeric_string); source.append(")0; \n");   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)

  source.append("  uint local_index = 0; \n");

  source.append("  for (uint k = 0; k < k_end; ++k) \n");
  source.append("  { \n");
  source.append("    local_index = group_start + k * get_local_size(0) + get_local_id(0); \n");

  source.append("    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0; \n");
  source.append("    val = (local_index < group_end && (option != 3 || tmp.x == tmp.y) ) ? elements[local_index] : 0; \n");

      //check for carry from previous loop run:
  source.append("    if (get_local_id(0) == 0 && k > 0) \n");
  source.append("    { \n");
  source.append("      if (tmp.x == shared_rows[last_index]) \n");
  source.append("      { \n");
  source.append("        switch (option) \n");
  source.append("        { \n");
  source.append("          case 0: \n"); //inf-norm
  source.append("          case 3: \n"); //diagonal entry
  source.append("            val = max(val, fabs(inter_results[last_index])); \n");
  source.append("            break; \n");

  source.append("          case 1: \n"); //1-norm
  source.append("            val = fabs(val) + inter_results[last_index]; \n");
  source.append("            break; \n");

  source.append("          case 2: \n"); //2-norm
  source.append("            val = sqrt(val * val + inter_results[last_index]); \n");
  source.append("            break; \n");

  source.append("          default: \n");
  source.append("            break; \n");
  source.append("        } \n");
  source.append("      } \n");
  source.append("      else \n");
  source.append("      { \n");
  source.append("        switch (option) \n");
  source.append("        { \n");
  source.append("          case 0: \n"); //inf-norm
  source.append("          case 1: \n"); //1-norm
  source.append("          case 3: \n"); //diagonal entry
  source.append("            result[shared_rows[last_index]] = inter_results[last_index]; \n");
  source.append("            break; \n");

  source.append("          case 2: \n"); //2-norm
  source.append("            result[shared_rows[last_index]] = sqrt(inter_results[last_index]); \n");
  source.append("          default: \n");
  source.append("            break; \n");
  source.append("        } \n");
  source.append("      } \n");
  source.append("    } \n");

      //segmented parallel reduction begin
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    shared_rows[get_local_id(0)] = tmp.x; \n");
  source.append("    switch (option) \n");
  source.append("    { \n");
  source.append("      case 0: \n");
  source.append("      case 3: \n");
  source.append("        inter_results[get_local_id(0)] = val; \n");
  source.append("        break; \n");
  source.append("      case 1: \n");
  source.append("        inter_results[get_local_id(0)] = fabs(val); \n");
  source.append("        break; \n");
  source.append("      case 2: \n");
  source.append("        inter_results[get_local_id(0)] = val * val; \n");
  source.append("      default: \n");
  source.append("        break; \n");
  source.append("    } \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");

  source.append("    for (unsigned int stride = 1; stride < get_local_size(0); stride *= 2) \n");
  source.append("    { \n");
  source.append("      "); source.append(numeric_string); source.append(" left = (get_local_id(0) >= stride && tmp.x == shared_rows[get_local_id(0) - stride]) ? inter_results[get_local_id(0) - stride] : ("); source.append(numeric_string); source.append(")0; \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("      switch (option) \n");
  source.append("      { \n");
  source.append("        case 0: \n"); //inf-norm
  source.append("        case 3: \n"); //diagonal entry
  source.append("          inter_results[get_local_id(0)] = max(inter_results[get_local_id(0)], left); \n");
  source.append("          break; \n");

  source.append("        case 1: \n"); //1-norm
  source.append("          inter_results[get_local_id(0)] += left; \n");
  source.append("          break; \n");

  source.append("        case 2: \n"); //2-norm
  source.append("          inter_results[get_local_id(0)] += left; \n");
  source.append("          break; \n");

  source.append("        default: \n");
  source.append("          break; \n");
  source.append("      } \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    } \n");
      //segmented parallel reduction end

  source.append("    if (get_local_id(0) != last_index && \n");
  source.append("        shared_rows[get_local_id(0)] != shared_rows[get_local_id(0) + 1] && \n");
  source.append("        inter_results[get_local_id(0)] != 0) \n");
  source.append("    { \n");
  source.append("      result[tmp.x] = (option == 2) ? sqrt(inter_results[get_local_id(0)]) : inter_results[get_local_id(0)]; \n");
  source.append("    } \n");

  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("  } \n"); //for k

  source.append("  if (local_index + 1 == group_end && inter_results[get_local_id(0)] != 0) \n");
  source.append("    result[tmp.x] = (option == 2) ? sqrt(inter_results[get_local_id(0)]) : inter_results[get_local_id(0)]; \n");
  source.append("} \n");
}

//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for coordinate_matrix. */
template<typename NumericT>
struct coordinate_matrix
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_coordinate_matrix";
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

      generate_coordinate_matrix_vec_mul(source, numeric_string);
      generate_coordinate_matrix_dense_matrix_multiplication(source, numeric_string);
      generate_coordinate_matrix_row_info_extractor(source, numeric_string);

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

