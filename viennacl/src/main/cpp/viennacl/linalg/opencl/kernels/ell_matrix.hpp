#ifndef VIENNACL_LINALG_OPENCL_KERNELS_ELL_MATRIX_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_ELL_MATRIX_HPP

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

/** @file viennacl/linalg/opencl/kernels/ell_matrix.hpp
 *  @brief OpenCL kernel file for ell_matrix operations */
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
void generate_ell_vec_mul(StringT & source, std::string const & numeric_string, bool with_alpha_beta)
{
  if (with_alpha_beta)
    source.append("__kernel void vec_mul_alpha_beta( \n");
  else
    source.append("__kernel void vec_mul( \n");
  source.append("  __global const unsigned int * coords, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * x, \n");
  source.append("  uint4 layout_x, \n");
  if (with_alpha_beta) { source.append("  "); source.append(numeric_string); source.append(" alpha, \n"); }
  source.append("  __global "); source.append(numeric_string); source.append(" * result, \n");
  source.append("  uint4 layout_result, \n");
  if (with_alpha_beta) { source.append("  "); source.append(numeric_string); source.append(" beta, \n"); }
  source.append("  unsigned int row_num, \n");
  source.append("  unsigned int col_num, \n");
  source.append("  unsigned int internal_row_num, \n");
  source.append("  unsigned int items_per_row, \n");
  source.append("  unsigned int aligned_items_per_row) \n");
  source.append("{ \n");
  source.append("  uint glb_id = get_global_id(0); \n");
  source.append("  uint glb_sz = get_global_size(0); \n");

  source.append("  for (uint row_id = glb_id; row_id < row_num; row_id += glb_sz) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint offset = row_id; \n");
  source.append("    for (uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num) { \n");
  source.append("      "); source.append(numeric_string); source.append(" val = elements[offset]; \n");

  source.append("       if (val != 0.0f) { \n");
  source.append("          int col = coords[offset]; \n");
  source.append("          sum += (x[col * layout_x.y + layout_x.x] * val); \n");
  source.append("       } \n");

  source.append("    } \n");

  if (with_alpha_beta)
    source.append("    result[row_id * layout_result.y + layout_result.x] = alpha * sum + ((beta != 0) ? beta * result[row_id * layout_result.y + layout_result.x] : 0); \n");
  else
    source.append("    result[row_id * layout_result.y + layout_result.x] = sum; \n");
  source.append("  } \n");
  source.append("} \n");
}

namespace detail
{
  template<typename StringT>
  void generate_ell_matrix_dense_matrix_mul(StringT & source, std::string const & numeric_string,
                                            bool B_transposed, bool B_row_major, bool C_row_major)
  {
    source.append("__kernel void ");
    source.append(viennacl::linalg::opencl::detail::sparse_dense_matmult_kernel_name(B_transposed, B_row_major, C_row_major));
    source.append("( \n");
    source.append("    __global const unsigned int * sp_mat_coords, \n");
    source.append("    __global const "); source.append(numeric_string); source.append(" * sp_mat_elems, \n");
    source.append("    unsigned int sp_mat_row_num, \n");
    source.append("    unsigned int sp_mat_col_num, \n");
    source.append("    unsigned int sp_mat_internal_row_num, \n");
    source.append("    unsigned int sp_mat_items_per_row, \n");
    source.append("    unsigned int sp_mat_aligned_items_per_row, \n");
    source.append("    __global const "); source.append(numeric_string); source.append("* d_mat, \n");
    source.append("    unsigned int d_mat_row_start, \n");
    source.append("    unsigned int d_mat_col_start, \n");
    source.append("    unsigned int d_mat_row_inc, \n");
    source.append("    unsigned int d_mat_col_inc, \n");
    source.append("    unsigned int d_mat_row_size, \n");
    source.append("    unsigned int d_mat_col_size, \n");
    source.append("    unsigned int d_mat_internal_rows, \n");
    source.append("    unsigned int d_mat_internal_cols, \n");
    source.append("    __global "); source.append(numeric_string); source.append(" * result, \n");
    source.append("    unsigned int result_row_start, \n");
    source.append("    unsigned int result_col_start, \n");
    source.append("    unsigned int result_row_inc, \n");
    source.append("    unsigned int result_col_inc, \n");
    source.append("    unsigned int result_row_size, \n");
    source.append("    unsigned int result_col_size, \n");
    source.append("    unsigned int result_internal_rows, \n");
    source.append("    unsigned int result_internal_cols) { \n");

    source.append("    uint glb_id = get_global_id(0); \n");
    source.append("    uint glb_sz = get_global_size(0); \n");

    source.append("    for ( uint rc = glb_id; rc < (sp_mat_row_num * result_col_size); rc += glb_sz) { \n");
    source.append("      uint row = rc % sp_mat_row_num; \n");
    source.append("      uint col = rc / sp_mat_row_num; \n");

    source.append("      uint offset = row; \n");
    source.append("      "); source.append(numeric_string); source.append(" r = ("); source.append(numeric_string); source.append(")0; \n");

    source.append("      for ( uint k = 0; k < sp_mat_items_per_row; k++, offset += sp_mat_internal_row_num) { \n");

    source.append("        uint j = sp_mat_coords[offset]; \n");
    source.append("        "); source.append(numeric_string); source.append(" x = sp_mat_elems[offset]; \n");

    source.append("        if (x != ("); source.append(numeric_string); source.append(")0) { \n");
    source.append("          "); source.append(numeric_string);
    if (B_transposed && B_row_major)
      source.append(" y = d_mat[ (d_mat_row_start + col * d_mat_row_inc) * d_mat_internal_cols + d_mat_col_start + j * d_mat_col_inc ]; \n");
    else if (B_transposed && !B_row_major)
      source.append(" y = d_mat[ (d_mat_row_start + col * d_mat_row_inc)                       + (d_mat_col_start + j * d_mat_col_inc) * d_mat_internal_rows ]; \n");
    else if (!B_transposed && B_row_major)
      source.append(" y = d_mat[ (d_mat_row_start +   j * d_mat_row_inc) * d_mat_internal_cols + d_mat_col_start + col * d_mat_col_inc ]; \n");
    else
      source.append(" y = d_mat[ (d_mat_row_start +   j * d_mat_row_inc)                       + (d_mat_col_start + col * d_mat_col_inc) * d_mat_internal_rows ]; \n");

    source.append("          r += x*y; \n");
    source.append("        } \n");
    source.append("      } \n");

    if (C_row_major)
      source.append("      result[ (result_row_start + row * result_row_inc) * result_internal_cols + result_col_start + col * result_col_inc ] = r; \n");
    else
      source.append("      result[ (result_row_start + row * result_row_inc)                        + (result_col_start + col * result_col_inc) * result_internal_rows ] = r; \n");
    source.append("    } \n");
    source.append("} \n");

  }
}

template<typename StringT>
void generate_ell_matrix_dense_matrix_multiplication(StringT & source, std::string const & numeric_string)
{
  detail::generate_ell_matrix_dense_matrix_mul(source, numeric_string, false, false, false);
  detail::generate_ell_matrix_dense_matrix_mul(source, numeric_string, false, false,  true);
  detail::generate_ell_matrix_dense_matrix_mul(source, numeric_string, false,  true, false);
  detail::generate_ell_matrix_dense_matrix_mul(source, numeric_string, false,  true,  true);

  detail::generate_ell_matrix_dense_matrix_mul(source, numeric_string, true, false, false);
  detail::generate_ell_matrix_dense_matrix_mul(source, numeric_string, true, false,  true);
  detail::generate_ell_matrix_dense_matrix_mul(source, numeric_string, true,  true, false);
  detail::generate_ell_matrix_dense_matrix_mul(source, numeric_string, true,  true,  true);
}

//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for ell_matrix. */
template<typename NumericT>
struct ell_matrix
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_ell_matrix";
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

      // fully parameterized kernels:
      generate_ell_vec_mul(source, numeric_string, true);
      generate_ell_vec_mul(source, numeric_string, false);
      generate_ell_matrix_dense_matrix_multiplication(source, numeric_string);

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

