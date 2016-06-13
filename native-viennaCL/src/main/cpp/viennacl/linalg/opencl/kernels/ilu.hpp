#ifndef VIENNACL_LINALG_OPENCL_KERNELS_ILU_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_ILU_HPP

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

/** @file viennacl/linalg/opencl/kernels/ilu.hpp
 *  @brief OpenCL kernel file for nonnegative matrix factorization */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{
template<typename StringT>
void generate_ilu_level_scheduling_substitute(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void level_scheduling_substitute( \n");
  source.append("  __global const unsigned int * row_index_array, \n");
  source.append("  __global const unsigned int * row_indices, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * vec, \n");
  source.append("  unsigned int size) \n");
  source.append("{ \n");
  source.append("  for (unsigned int row  = get_global_id(0); \n");
  source.append("                    row  < size; \n");
  source.append("                    row += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    unsigned int eq_row = row_index_array[row]; \n");
  source.append("    "); source.append(numeric_string); source.append(" vec_entry = vec[eq_row]; \n");
  source.append("    unsigned int row_end = row_indices[row+1]; \n");

  source.append("    for (unsigned int j = row_indices[row]; j < row_end; ++j) \n");
  source.append("      vec_entry -= vec[column_indices[j]] * elements[j]; \n");

  source.append("    vec[eq_row] = vec_entry; \n");
  source.append("  } \n");
  source.append("} \n");
}

///////////// ICC ///////////////


template<typename StringT>
void generate_icc_extract_L_1(StringT & source)
{
  source.append("__kernel void extract_L_1( \n");
  source.append("  __global unsigned int const *A_row_indices, \n");
  source.append("  __global unsigned int const *A_col_indices, \n");
  source.append("  unsigned int A_size1, \n");
  source.append("  __global unsigned int *L_row_indices) { \n");

  source.append("  for (unsigned int row  = get_global_id(0); \n");
  source.append("                    row  < A_size1; \n");
  source.append("                    row += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    unsigned int row_begin = A_row_indices[row]; \n");
  source.append("    unsigned int row_end   = A_row_indices[row+1]; \n");

  source.append("    unsigned int num_entries_L = 0; \n");
  source.append("    for (unsigned int j=row_begin; j<row_end; ++j) { \n");
  source.append("      unsigned int col = A_col_indices[j]; \n");
  source.append("      if (col <= row) ++num_entries_L; \n");
  source.append("    } \n");

  source.append("    L_row_indices[row] = num_entries_L;   \n");
  source.append("  } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_icc_extract_L_2(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void extract_L_2( \n");
  source.append("  __global unsigned int const *A_row_indices, \n");
  source.append("  __global unsigned int const *A_col_indices, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const *A_elements, \n");
  source.append("  unsigned int A_size1, \n");
  source.append("  __global unsigned int const *L_row_indices, \n");
  source.append("  __global unsigned int       *L_col_indices, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" *L_elements) { \n");

  source.append("  for (unsigned int row  = get_global_id(0); \n");
  source.append("                    row  < A_size1; \n");
  source.append("                    row += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    unsigned int row_begin = A_row_indices[row]; \n");
  source.append("    unsigned int row_end   = A_row_indices[row+1]; \n");

  source.append("    unsigned int index_L = L_row_indices[row]; \n");
  source.append("    for (unsigned int j=row_begin; j<row_end; ++j) { \n");
  source.append("      unsigned int col = A_col_indices[j]; \n");
  source.append("      "); source.append(numeric_string); source.append(" value = A_elements[j]; \n");

  source.append("      if (col <= row) { \n");
  source.append("        L_col_indices[index_L] = col; \n");
  source.append("        L_elements[index_L]    = value; \n");
  source.append("        ++index_L; \n");
  source.append("      } \n");
  source.append("    } \n");

  source.append("  } \n");
  source.append("} \n");
}


template<typename StringT>
void generate_icc_chow_patel_sweep_kernel(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void icc_chow_patel_sweep_kernel( \n");
  source.append("  __global unsigned int const *L_row_indices, \n");
  source.append("  __global unsigned int const *L_col_indices, \n");
  source.append("  __global "); source.append(numeric_string); source.append("       *L_elements, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const *L_backup, \n");
  source.append("  unsigned int L_size1, \n");

  source.append("  __global "); source.append(numeric_string); source.append(" const *aij_L) { \n");

  source.append("  for (unsigned int row  = get_global_id(0); \n");
  source.append("                    row  < L_size1; \n");
  source.append("                    row += get_global_size(0)) \n");
  source.append("  { \n");

  //
  // Update L:
  //
  source.append("    unsigned int row_Li_start = L_row_indices[row]; \n");
  source.append("    unsigned int row_Li_end   = L_row_indices[row + 1]; \n");

  source.append("    for (unsigned int i = row_Li_start; i < row_Li_end; ++i) { \n");
  source.append("      unsigned int col = L_col_indices[i]; \n");

  source.append("      unsigned int row_Lj_start = L_row_indices[col]; \n");
  source.append("      unsigned int row_Lj_end   = L_row_indices[col + 1]; \n");

  source.append("      unsigned int index_Lj = row_Lj_start; \n");
  source.append("      unsigned int col_Lj = L_col_indices[index_Lj]; \n");

  source.append("      "); source.append(numeric_string); source.append(" s = aij_L[i]; \n");
  source.append("      for (unsigned int index_Li = row_Li_start; index_Li < i; ++index_Li) { \n");
  source.append("        unsigned int col_Li = L_col_indices[index_Li]; \n");

  source.append("        while (col_Lj < col_Li) { \n");
  source.append("          ++index_Lj; \n");
  source.append("          col_Lj = L_col_indices[index_Lj]; \n");
  source.append("        } \n");

  source.append("        if (col_Lj == col_Li) \n");
  source.append("          s -= L_backup[index_Li] * L_backup[index_Lj]; \n");
  source.append("      } \n");

  // update l_ij:
  source.append("      L_elements[i] = (row == col) ? sqrt(s) : (s / L_backup[row_Lj_end - 1]); \n");
  source.append("    } \n");

  source.append("  } \n");
  source.append("} \n");
}


///////////// ILU ///////////////

template<typename StringT>
void generate_ilu_extract_LU_1(StringT & source)
{
  source.append("__kernel void extract_LU_1( \n");
  source.append("  __global unsigned int const *A_row_indices, \n");
  source.append("  __global unsigned int const *A_col_indices, \n");
  source.append("  unsigned int A_size1, \n");
  source.append("  __global unsigned int *L_row_indices, \n");
  source.append("  __global unsigned int *U_row_indices) { \n");

  source.append("  for (unsigned int row  = get_global_id(0); \n");
  source.append("                    row  < A_size1; \n");
  source.append("                    row += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    unsigned int row_begin = A_row_indices[row]; \n");
  source.append("    unsigned int row_end   = A_row_indices[row+1]; \n");

  source.append("    unsigned int num_entries_L = 0; \n");
  source.append("    unsigned int num_entries_U = 0; \n");
  source.append("    for (unsigned int j=row_begin; j<row_end; ++j) { \n");
  source.append("      unsigned int col = A_col_indices[j]; \n");
  source.append("      if (col <= row) ++num_entries_L; \n");
  source.append("      if (col >= row) ++num_entries_U; \n");
  source.append("    } \n");

  source.append("    L_row_indices[row] = num_entries_L;   \n");
  source.append("    U_row_indices[row] = num_entries_U;   \n");
  source.append("  } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_ilu_extract_LU_2(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void extract_LU_2( \n");
  source.append("  __global unsigned int const *A_row_indices, \n");
  source.append("  __global unsigned int const *A_col_indices, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const *A_elements, \n");
  source.append("  unsigned int A_size1, \n");
  source.append("  __global unsigned int const *L_row_indices, \n");
  source.append("  __global unsigned int       *L_col_indices, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" *L_elements, \n");
  source.append("  __global unsigned int const *U_row_indices, \n");
  source.append("  __global unsigned int       *U_col_indices, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" *U_elements) { \n");

  source.append("  for (unsigned int row  = get_global_id(0); \n");
  source.append("                    row  < A_size1; \n");
  source.append("                    row += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    unsigned int row_begin = A_row_indices[row]; \n");
  source.append("    unsigned int row_end   = A_row_indices[row+1]; \n");

  source.append("    unsigned int index_L = L_row_indices[row]; \n");
  source.append("    unsigned int index_U = U_row_indices[row]; \n");
  source.append("    for (unsigned int j=row_begin; j<row_end; ++j) { \n");
  source.append("      unsigned int col = A_col_indices[j]; \n");
  source.append("      "); source.append(numeric_string); source.append(" value = A_elements[j]; \n");

  source.append("      if (col <= row) { \n");
  source.append("        L_col_indices[index_L] = col; \n");
  source.append("        L_elements[index_L]    = value; \n");
  source.append("        ++index_L; \n");
  source.append("      } \n");
  source.append("      if (col >= row) { \n");
  source.append("        U_col_indices[index_U] = col; \n");
  source.append("        U_elements[index_U]    = value; \n");
  source.append("        ++index_U; \n");
  source.append("      } \n");
  source.append("    } \n");

  source.append("  } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_ilu_scale_kernel_1(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void ilu_scale_kernel_1( \n");
  source.append("  __global unsigned int const *A_row_indices, \n");
  source.append("  __global unsigned int const *A_col_indices, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const *A_elements, \n");
  source.append("  unsigned int A_size1, \n");
  source.append("  __global "); source.append(numeric_string); source.append("       *D_elements) { \n");

  source.append("  for (unsigned int row  = get_global_id(0); \n");
  source.append("                    row  < A_size1; \n");
  source.append("                    row += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    unsigned int row_begin = A_row_indices[row]; \n");
  source.append("    unsigned int row_end   = A_row_indices[row+1]; \n");

  source.append("    for (unsigned int j=row_begin; j<row_end; ++j) { \n");
  source.append("      unsigned int col = A_col_indices[j]; \n");

  source.append("      if (col == row) { \n");
  source.append("        D_elements[row] = 1 / sqrt(fabs(A_elements[j])); \n");
  source.append("        break; \n");
  source.append("      } \n");
  source.append("    } \n");

  source.append("  } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_ilu_scale_kernel_2(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void ilu_scale_kernel_2( \n");
  source.append("  __global unsigned int const *R_row_indices, \n");
  source.append("  __global unsigned int const *R_col_indices, \n");
  source.append("  __global "); source.append(numeric_string); source.append("       *R_elements, \n");
  source.append("  unsigned int R_size1, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const *D_elements) { \n");

  source.append("  for (unsigned int row  = get_global_id(0); \n");
  source.append("                    row  < R_size1; \n");
  source.append("                    row += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    unsigned int row_begin = R_row_indices[row]; \n");
  source.append("    unsigned int row_end   = R_row_indices[row+1]; \n");

  source.append("    "); source.append(numeric_string); source.append(" D_row = D_elements[row]; \n");
  source.append("    for (unsigned int j=row_begin; j<row_end; ++j) \n");
  source.append("      R_elements[j] *= D_row * D_elements[R_col_indices[j]]; \n");

  source.append("  } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_ilu_chow_patel_sweep_kernel(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void ilu_chow_patel_sweep_kernel( \n");
  source.append("  __global unsigned int const *L_row_indices, \n");
  source.append("  __global unsigned int const *L_col_indices, \n");
  source.append("  __global "); source.append(numeric_string); source.append("       *L_elements, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const *L_backup, \n");
  source.append("  unsigned int L_size1, \n");

  source.append("  __global "); source.append(numeric_string); source.append(" const *aij_L, \n");

  source.append("  __global unsigned int const *U_trans_row_indices, \n");
  source.append("  __global unsigned int const *U_trans_col_indices, \n");
  source.append("  __global "); source.append(numeric_string); source.append("       *U_trans_elements, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const *U_trans_backup, \n");

  source.append("  __global "); source.append(numeric_string); source.append(" const *aij_U_trans) { \n");

  source.append("  for (unsigned int row  = get_global_id(0); \n");
  source.append("                    row  < L_size1; \n");
  source.append("                    row += get_global_size(0)) \n");
  source.append("  { \n");

  //
  // Update L:
  //
  source.append("    unsigned int row_L_start = L_row_indices[row]; \n");
  source.append("    unsigned int row_L_end   = L_row_indices[row + 1]; \n");

  source.append("    for (unsigned int j = row_L_start; j < row_L_end; ++j) { \n");
  source.append("      unsigned int col = L_col_indices[j]; \n");

  source.append("      if (col == row) continue; \n");

  source.append("      unsigned int row_U_start = U_trans_row_indices[col]; \n");
  source.append("      unsigned int row_U_end   = U_trans_row_indices[col + 1]; \n");

  source.append("      unsigned int index_U = row_U_start; \n");
  source.append("      unsigned int col_U = (index_U < row_U_end) ? U_trans_col_indices[index_U] : L_size1; \n");

  source.append("      "); source.append(numeric_string); source.append(" sum = 0; \n");
  source.append("      for (unsigned int k = row_L_start; k < j; ++k) { \n");
  source.append("        unsigned int col_L = L_col_indices[k]; \n");

  source.append("        while (col_U < col_L) { \n");
  source.append("          ++index_U; \n");
  source.append("          col_U = U_trans_col_indices[index_U]; \n");
  source.append("        } \n");

  source.append("        if (col_U == col_L) \n");
  source.append("          sum += L_backup[k] * U_trans_backup[index_U]; \n");
  source.append("      } \n");

  // update l_ij:
  source.append("      L_elements[j] = (aij_L[j] - sum) / U_trans_backup[row_U_end - 1]; \n");
  source.append("    } \n");

  //
  // Update U:
  //
  source.append("    unsigned int row_U_start = U_trans_row_indices[row]; \n");
  source.append("    unsigned int row_U_end   = U_trans_row_indices[row + 1]; \n");

  source.append("    for (unsigned int j = row_U_start; j < row_U_end; ++j) { \n");
  source.append("      unsigned int col = U_trans_col_indices[j]; \n");

  source.append("      row_L_start = L_row_indices[col]; \n");
  source.append("      row_L_end   = L_row_indices[col + 1]; \n");

  // compute \sum_{k=1}^{j-1} l_ik u_kj
  source.append("      unsigned int index_L = row_L_start; \n");
  source.append("      unsigned int col_L = (index_L < row_L_end) ? L_col_indices[index_L] : L_size1; \n");
  source.append("      "); source.append(numeric_string); source.append(" sum = 0; \n");
  source.append("      for (unsigned int k = row_U_start; k < j; ++k) { \n");
  source.append("        unsigned int col_U = U_trans_col_indices[k]; \n");

  // find element in L:
  source.append("        while (col_L < col_U) { \n");
  source.append("          ++index_L; \n");
  source.append("          col_L = L_col_indices[index_L]; \n");
  source.append("        } \n");

  source.append("        if (col_U == col_L) \n");
  source.append("          sum += L_backup[index_L] * U_trans_backup[k]; \n");
  source.append("      } \n");

  // update U_ij:
  source.append("      U_trans_elements[j] = aij_U_trans[j] - sum; \n");
  source.append("    } \n");

  source.append("  } \n");
  source.append("} \n");
}


template<typename StringT>
void generate_ilu_form_neumann_matrix_kernel(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void ilu_form_neumann_matrix_kernel( \n");
  source.append("  __global unsigned int const *R_row_indices, \n");
  source.append("  __global unsigned int const *R_col_indices, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" *R_elements, \n");
  source.append("  unsigned int R_size1, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" *D_elements) { \n");

  source.append("  for (unsigned int row  = get_global_id(0); \n");
  source.append("                    row  < R_size1; \n");
  source.append("                    row += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    unsigned int row_begin = R_row_indices[row]; \n");
  source.append("    unsigned int row_end   = R_row_indices[row+1]; \n");

  // Part 1: Extract and set diagonal entry
  source.append("    "); source.append(numeric_string); source.append(" diag = D_elements[row]; \n");
  source.append("    for (unsigned int j=row_begin; j<row_end; ++j) { \n");
  source.append("      unsigned int col = R_col_indices[j]; \n");
  source.append("      if (col == row) { \n");
  source.append("        diag = R_elements[j]; \n");
  source.append("        R_elements[j] = 0; \n");
  source.append("        break; \n");
  source.append("      } \n");
  source.append("    } \n");
  source.append("    D_elements[row] = diag; \n");

  // Part 2: Scale
  source.append("    for (unsigned int j=row_begin; j<row_end; ++j) \n");
  source.append("      R_elements[j] /= -diag; \n");

  source.append("  } \n");
  source.append("} \n");
}



// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for incomplete LU factorization preconditioners. */
template<class NumericT>
struct ilu
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_ilu";
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

      // only generate for floating points (forces error for integers)
      if (numeric_string == "float" || numeric_string == "double")
      {
        generate_ilu_level_scheduling_substitute(source, numeric_string);

        generate_icc_extract_L_1(source);
        generate_icc_extract_L_2(source, numeric_string);
        generate_icc_chow_patel_sweep_kernel(source, numeric_string);

        generate_ilu_extract_LU_1(source);
        generate_ilu_extract_LU_2(source, numeric_string);
        generate_ilu_scale_kernel_1(source, numeric_string);
        generate_ilu_scale_kernel_2(source, numeric_string);
        generate_ilu_chow_patel_sweep_kernel(source, numeric_string);
        generate_ilu_form_neumann_matrix_kernel(source, numeric_string);
      }

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

