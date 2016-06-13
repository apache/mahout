#ifndef VIENNACL_LINALG_OPENCL_KERNELS_SPAI_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_SPAI_HPP

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

/** @file viennacl/linalg/opencl/kernels/spai.hpp
 *  @brief OpenCL kernel file for sparse approximate inverse operations */
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
void generate_spai_assemble_blocks(StringT & source, std::string const & numeric_string)
{
  source.append("float get_element(__global const unsigned int * row_indices, \n");
  source.append("           __global const unsigned int * column_indices, \n");
  source.append("           __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("           unsigned int row, \n");
  source.append("           unsigned int col) \n");
  source.append("{ \n");
  source.append("  unsigned int row_end = row_indices[row+1]; \n");
  source.append("  for (unsigned int i = row_indices[row]; i < row_end; ++i){ \n");
  source.append("    if (column_indices[i] == col) \n");
  source.append("      return elements[i]; \n");
  source.append("    if (column_indices[i] > col) \n");
  source.append("      return 0; \n");
  source.append("  } \n");
  source.append("  return 0; \n");
  source.append("} \n");

  source.append("void block_assembly(__global const unsigned int * row_indices, \n");
  source.append("          __global const unsigned int * column_indices, \n");
  source.append("          __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("          __global const unsigned int * matrix_dimensions, \n");
  source.append("          __global const unsigned int * set_I, \n");
  source.append("          __global const unsigned int * set_J, \n");
  source.append("          unsigned int matrix_ind, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * com_A_I_J) \n");
  source.append("{ \n");
  source.append("  unsigned int row_n = matrix_dimensions[2*matrix_ind]; \n");
  source.append("  unsigned int col_n = matrix_dimensions[2*matrix_ind + 1]; \n");

  source.append("  for (unsigned int i = 0; i < col_n; ++i){ \n");
          //start row index
  source.append("        for (unsigned int j = 0; j < row_n; j++){ \n");
  source.append("          com_A_I_J[ i*row_n + j] = get_element(row_indices, column_indices, elements, set_I[j], set_J[i]); \n");
  source.append("        } \n");
  source.append("      } \n");
  source.append("} \n");

  source.append("__kernel void assemble_blocks( \n");
  source.append("          __global const unsigned int * row_indices, \n");
  source.append("          __global const unsigned int * column_indices, \n");
  source.append("          __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("          __global const unsigned int * set_I, \n");
  source.append("        __global const unsigned int * set_J, \n");
  source.append("      __global const unsigned int * i_ind, \n");
  source.append("      __global const unsigned int * j_ind, \n");
  source.append("        __global const unsigned int * block_ind, \n");
  source.append("        __global const unsigned int * matrix_dimensions, \n");
  source.append("      __global "); source.append(numeric_string); source.append(" * com_A_I_J, \n");
  source.append("      __global unsigned int * g_is_update, \n");
  source.append("                   unsigned int  block_elems_num) \n");
  source.append("{ \n");
  source.append("    for (unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){ \n");
  source.append("        if ((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){ \n");
  source.append("            block_assembly(row_indices, column_indices, elements, matrix_dimensions, set_I + i_ind[i], set_J + j_ind[i], i, com_A_I_J + block_ind[i]); \n");
  source.append("        } \n");
  source.append("    } \n");
  source.append("  } \n");
}

template<typename StringT>
void generate_spai_block_bv_assembly(StringT & source, std::string const & numeric_string)
{
  source.append("  void assemble_bv(__global "); source.append(numeric_string); source.append(" * g_bv_r, __global "); source.append(numeric_string); source.append(" * g_bv, unsigned int col_n){ \n");
  source.append("    for (unsigned int i = 0; i < col_n; ++i){ \n");
  source.append("      g_bv_r[i] = g_bv[ i]; \n");
  source.append("    } \n");
  source.append("  } \n");

  source.append("  void assemble_bv_block(__global "); source.append(numeric_string); source.append(" * g_bv_r, __global "); source.append(numeric_string); source.append(" * g_bv, unsigned int col_n, \n");
  source.append("               __global "); source.append(numeric_string); source.append(" * g_bv_u, unsigned int col_n_u) \n");
  source.append("  { \n");
  source.append("    assemble_bv(g_bv_r, g_bv, col_n); \n");
  source.append("    assemble_bv(g_bv_r + col_n, g_bv_u, col_n_u); \n");
  source.append("  } \n");

  source.append("  __kernel void block_bv_assembly(__global "); source.append(numeric_string); source.append(" * g_bv, \n");
  source.append("              __global unsigned int * start_bv_ind, \n");
  source.append("              __global unsigned int * matrix_dimensions, \n");
  source.append("              __global "); source.append(numeric_string); source.append(" * g_bv_u, \n");
  source.append("              __global unsigned int * start_bv_u_ind, \n");
  source.append("              __global unsigned int * matrix_dimensions_u, \n");
  source.append("              __global "); source.append(numeric_string); source.append(" * g_bv_r, \n");
  source.append("              __global unsigned int * start_bv_r_ind, \n");
  source.append("              __global unsigned int * matrix_dimensions_r, \n");
  source.append("              __global unsigned int * g_is_update, \n");
  source.append("              //__local  "); source.append(numeric_string); source.append(" * local_gb, \n");
  source.append("              unsigned int  block_elems_num) \n");
  source.append("  { \n");
  source.append("    for (unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){ \n");
  source.append("      if ((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){ \n");
  source.append("        assemble_bv_block(g_bv_r + start_bv_r_ind[i], g_bv + start_bv_ind[i], matrix_dimensions[2*i + 1], g_bv_u + start_bv_u_ind[i], matrix_dimensions_u[2*i + 1]); \n");
  source.append("      } \n");
  source.append("    } \n");
  source.append("  } \n");
}

template<typename StringT>
void generate_spai_block_least_squares(StringT & source, std::string const & numeric_string)
{
  source.append("void custom_dot_prod_ls(__global "); source.append(numeric_string); source.append(" * A, unsigned int row_n, __global "); source.append(numeric_string); source.append(" * v, unsigned int ind, "); source.append(numeric_string); source.append(" *res){ \n");
  source.append("  *res = 0.0; \n");
  source.append("  for (unsigned int j = ind; j < row_n; ++j){ \n");
  source.append("    if (j == ind){ \n");
  source.append("      *res += v[ j]; \n");
  source.append("    }else{ \n");
  source.append("      *res += A[ j + ind*row_n]*v[ j]; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");

  source.append("void backwardSolve(__global "); source.append(numeric_string); source.append(" * R,  unsigned int row_n, unsigned int col_n, __global "); source.append(numeric_string); source.append(" * y, __global "); source.append(numeric_string); source.append(" * x){ \n");
  source.append("  for (int i = col_n-1; i >= 0; i--) { \n");
  source.append("    x[ i] = y[ i]; \n");
  source.append("    for (int j = i+1; j < col_n; ++j) { \n");
  source.append("      x[ i] -= R[ i + j*row_n]*x[ j]; \n");
  source.append("    } \n");
  source.append("    x[i] /= R[ i + i*row_n]; \n");
  source.append("  } \n");
  source.append("} \n");


  source.append("void apply_q_trans_vec_ls(__global "); source.append(numeric_string); source.append(" * R, unsigned int row_n, unsigned int col_n, __global const "); source.append(numeric_string); source.append(" * b_v,  __global "); source.append(numeric_string); source.append(" * y){ \n");
  source.append("            "); source.append(numeric_string); source.append(" inn_prod = 0; \n");
  source.append("            for (unsigned int i = 0; i < col_n; ++i){ \n");
  source.append("                custom_dot_prod_ls(R, row_n, y, i, &inn_prod); \n");
  source.append("                for (unsigned int j = i; j < row_n; ++j){ \n");
  source.append("                    if (i == j){ \n");
  source.append("                        y[ j] -= b_v[ i]*inn_prod; \n");
  source.append("                    } \n");
  source.append("                    else{ \n");
  source.append("                        y[j] -= b_v[ i]*inn_prod*R[ j +i*row_n]; \n");
  source.append("                    } \n");
  source.append("                } \n");
  source.append("            } \n");
  source.append("        } \n");

  source.append("void ls(__global "); source.append(numeric_string); source.append(" * R, unsigned int row_n, unsigned int col_n, __global "); source.append(numeric_string); source.append(" * b_v, __global "); source.append(numeric_string); source.append(" * m_v, __global "); source.append(numeric_string); source.append(" * y_v){ \n");
  source.append("  apply_q_trans_vec_ls(R, row_n, col_n, b_v, y_v); \n");
  source.append("  //m_new - is m_v now \n");
  source.append("  backwardSolve(R, row_n, col_n, y_v, m_v); \n");
  source.append("} \n");

  source.append("__kernel void block_least_squares( \n");
  source.append("      __global "); source.append(numeric_string); source.append(" * global_R, \n");
  source.append("      __global unsigned int * block_ind, \n");
  source.append("      __global "); source.append(numeric_string); source.append(" * b_v, \n");
  source.append("      __global unsigned int * start_bv_inds, \n");
  source.append("      __global "); source.append(numeric_string); source.append(" * m_v, \n");
  source.append("      __global "); source.append(numeric_string); source.append(" * y_v, \n");
  source.append("      __global unsigned int * start_y_inds, \n");
  source.append("      __global unsigned int * matrix_dimensions, \n");
  source.append("      __global unsigned int * g_is_update, \n");
  source.append("      unsigned int  block_elems_num) \n");
  source.append("{ \n");
  source.append("    for (unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){ \n");
  source.append("        if ((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){ \n");
  source.append("            ls(global_R + block_ind[i], matrix_dimensions[2*i], matrix_dimensions[2*i + 1], b_v +start_bv_inds[i], m_v + start_bv_inds[i], y_v + start_y_inds[i] ); \n");
  source.append("        } \n");
  source.append("    } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_spai_block_q_mult(StringT & source, std::string const & numeric_string)
{
  source.append("void custom_dot_prod(__global "); source.append(numeric_string); source.append(" * A, unsigned int row_n, __local "); source.append(numeric_string); source.append(" * v, unsigned int ind, "); source.append(numeric_string); source.append(" *res){ \n");
  source.append("  *res = 0.0; \n");
  source.append("  for (unsigned int j = ind; j < row_n; ++j){ \n");
  source.append("    if (j == ind){ \n");
  source.append("      *res += v[j]; \n");
  source.append("    }else{ \n");
  source.append("      *res += A[j + ind*row_n]*v[j]; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");

  source.append("void apply_q_trans_vec(__global "); source.append(numeric_string); source.append(" * R, unsigned int row_n, unsigned int col_n, __global "); source.append(numeric_string); source.append(" * b_v, __local "); source.append(numeric_string); source.append(" * y){ \n");
  source.append("  "); source.append(numeric_string); source.append(" inn_prod = 0; \n");
  source.append("  for (unsigned int i = 0; i < col_n; ++i){ \n");
  source.append("    custom_dot_prod(R, row_n, y, i, &inn_prod); \n");
  source.append("    for (unsigned int j = i; j < row_n; ++j){ \n");
  source.append("      if (i == j){ \n");
  source.append("        y[j] -= b_v[ i]*inn_prod; \n");
  source.append("      } \n");
  source.append("      else{ \n");
  source.append("        y[j] -= b_v[ i]*inn_prod*R[ j + i*row_n]; \n");
  source.append("      } \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");

  source.append("void q_mult(__global "); source.append(numeric_string); source.append(" * R, unsigned int row_n, unsigned int col_n, __global "); source.append(numeric_string); source.append(" * b_v, __local "); source.append(numeric_string); source.append(" * R_u, unsigned int col_n_u){ \n");
  source.append("        for (unsigned int i = get_local_id(0); i < col_n_u; i+= get_local_size(0)){ \n");
  source.append("          apply_q_trans_vec(R, row_n, col_n, b_v, R_u + row_n*i); \n");
  source.append("        } \n");
  source.append("} \n");

  source.append("void matrix_from_global_to_local(__global "); source.append(numeric_string); source.append("* g_M, __local "); source.append(numeric_string); source.append("* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){ \n");
  source.append("  for (unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){ \n");
  source.append("    for (unsigned int j = 0; j < row_n; ++j){ \n");
  source.append("      l_M[i*row_n + j] = g_M[mat_start_ind + i*row_n + j]; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");

  source.append("void matrix_from_local_to_global(__global "); source.append(numeric_string); source.append("* g_M, __local "); source.append(numeric_string); source.append("* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){ \n");
  source.append("  for (unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){ \n");
  source.append("    for (unsigned int j = 0; j < row_n; ++j){ \n");
  source.append("      g_M[mat_start_ind + i*row_n + j] = l_M[i*row_n + j]; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");

  source.append("__kernel void block_q_mult(__global "); source.append(numeric_string); source.append(" * global_R, \n");
  source.append("  __global unsigned int * block_ind, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * global_R_u, \n");
  source.append("  __global unsigned int *block_ind_u, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * b_v, \n");
  source.append("  __global unsigned int * start_bv_inds, \n");
  source.append("  __global unsigned int * matrix_dimensions, \n");
  source.append("  __global unsigned int * matrix_dimensions_u, \n");
  source.append("  __global unsigned int * g_is_update, \n");
  source.append("  __local  "); source.append(numeric_string); source.append(" * local_R_u, \n");
  source.append("    unsigned int  block_elems_num){ \n");
  source.append("    for (unsigned int i  = get_group_id(0); i < block_elems_num; i += get_num_groups(0)){ \n");
  source.append("          if ((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && (g_is_update[i] > 0)){ \n");
          //matrix_from_global_to_local(R, local_buff_R, matrix_dimensions[2*i], matrix_dimensions[2*i + 1], start_matrix_inds[i]); \n");
  source.append("        matrix_from_global_to_local(global_R_u, local_R_u, matrix_dimensions_u[2*i], matrix_dimensions_u[2*i+ 1], block_ind_u[i]); \n");
  source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("              q_mult(global_R + block_ind[i], matrix_dimensions[2*i], matrix_dimensions[2*i + 1], b_v + start_bv_inds[i], local_R_u, \n");
  source.append("             matrix_dimensions_u[2*i + 1]); \n");
  source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("              matrix_from_local_to_global(global_R_u, local_R_u, matrix_dimensions_u[2*i], matrix_dimensions_u[2*i + 1], block_ind_u[i]); \n");
  source.append("          } \n");
  source.append("      } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_spai_block_qr(StringT & source, std::string const & numeric_string)
{
  source.append("void dot_prod(__local const "); source.append(numeric_string); source.append("* A, unsigned int n, unsigned int beg_ind, "); source.append(numeric_string); source.append("* res){ \n");
  source.append("    *res = 0; \n");
  source.append("    for (unsigned int i = beg_ind; i < n; ++i){ \n");
  source.append("        *res += A[(beg_ind-1)*n + i]*A[(beg_ind-1)*n + i]; \n");
  source.append("    } \n");
  source.append("} \n");

  source.append("void vector_div(__global "); source.append(numeric_string); source.append("* v, unsigned int beg_ind, "); source.append(numeric_string); source.append(" b, unsigned int n){ \n");
  source.append("    for (unsigned int i = beg_ind; i < n; ++i){ \n");
  source.append("        v[i] /= b; \n");
  source.append("    } \n");
  source.append("} \n");

  source.append("void copy_vector(__local const "); source.append(numeric_string); source.append("* A, __global "); source.append(numeric_string); source.append("* v, const unsigned int beg_ind, const unsigned int n){ \n");
  source.append("    for (unsigned int i = beg_ind; i < n; ++i){ \n");
  source.append("        v[i] = A[(beg_ind-1)*n + i]; \n");
  source.append("    } \n");
  source.append("} \n");


  source.append("void householder_vector(__local const "); source.append(numeric_string); source.append("* A, unsigned int j, unsigned int n, __global "); source.append(numeric_string); source.append("* v, __global "); source.append(numeric_string); source.append("* b){ \n");
  source.append("    "); source.append(numeric_string); source.append(" sg; \n");
  source.append("    dot_prod(A, n, j+1, &sg); \n");
  source.append("    copy_vector(A, v, j+1, n); \n");
  source.append("    "); source.append(numeric_string); source.append(" mu; \n");
  source.append("    v[j] = 1.0; \n");
      //print_contigious_vector(v, v_start_ind, n);
  source.append("    if (sg == 0){ \n");
  source.append("        *b = 0; \n");
  source.append("    } \n");
  source.append("    else{ \n");
  source.append("        mu = sqrt(A[j*n + j]*A[ j*n + j] + sg); \n");
  source.append("        if (A[ j*n + j] <= 0){ \n");
  source.append("            v[j] = A[ j*n + j] - mu; \n");
  source.append("        }else{ \n");
  source.append("            v[j] = -sg/(A[ j*n + j] + mu); \n");
  source.append("        } \n");
  source.append("    *b = 2*(v[j]*v[j])/(sg + v[j]*v[j]); \n");
          //*b = (2*v[j]*v[j])/(sg + (v[j])*(v[j]));
  source.append("        vector_div(v, j, v[j], n); \n");
          //print_contigious_vector(v, v_start_ind, n);
  source.append("    } \n");
  source.append("} \n");

  source.append("void custom_inner_prod(__local const "); source.append(numeric_string); source.append("* A, __global "); source.append(numeric_string); source.append("* v, unsigned int col_ind, unsigned int row_num, unsigned int start_ind, "); source.append(numeric_string); source.append("* res){ \n");
  source.append("    for (unsigned int i = start_ind; i < row_num; ++i){ \n");
  source.append("        *res += A[col_ind*row_num + i]*v[i]; \n");
  source.append("    } \n");
  source.append("} \n");
  //
  source.append("void apply_householder_reflection(__local "); source.append(numeric_string); source.append("* A,  unsigned int row_n, unsigned int col_n, unsigned int iter_cnt, __global "); source.append(numeric_string); source.append("* v, "); source.append(numeric_string); source.append(" b){ \n");
  source.append("    "); source.append(numeric_string); source.append(" in_prod_res; \n");
  source.append("    for (unsigned int i= iter_cnt + get_local_id(0); i < col_n; i+=get_local_size(0)){ \n");
  source.append("        in_prod_res = 0.0; \n");
  source.append("        custom_inner_prod(A, v, i, row_n, iter_cnt, &in_prod_res); \n");
  source.append("        for (unsigned int j = iter_cnt; j < row_n; ++j){ \n");
  source.append("            A[ i*row_n + j] -= b*in_prod_res* v[j]; \n");
  source.append("        } \n");
  source.append("    } \n");
  source.append("} \n");

  source.append("void store_householder_vector(__local "); source.append(numeric_string); source.append("* A,  unsigned int ind, unsigned int n, __global "); source.append(numeric_string); source.append("* v){ \n");
  source.append("    for (unsigned int i = ind; i < n; ++i){ \n");
  source.append("        A[ (ind-1)*n + i] = v[i]; \n");
  source.append("    } \n");
  source.append("} \n");

  source.append("void single_qr( __local "); source.append(numeric_string); source.append("* R, __global unsigned int* matrix_dimensions, __global "); source.append(numeric_string); source.append("* b_v, __global "); source.append(numeric_string); source.append("* v, unsigned int matrix_ind){ \n");
              //matrix_dimensions[0] - number of rows
                //matrix_dimensions[1] - number of columns
  source.append("  unsigned int col_n = matrix_dimensions[2*matrix_ind + 1]; \n");
  source.append("  unsigned int row_n = matrix_dimensions[2*matrix_ind]; \n");

  source.append("  if ((col_n == row_n)&&(row_n == 1)){ \n");
  source.append("    b_v[0] = 0.0; \n");
  source.append("      return; \n");
  source.append("  } \n");
  source.append("  for (unsigned int i = 0; i < col_n; ++i){ \n");
  source.append("    if (get_local_id(0) == 0){ \n");
  source.append("      householder_vector(R, i, row_n, v, b_v + i); \n");
  source.append("    } \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    apply_householder_reflection(R, row_n, col_n, i, v, b_v[i]); \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) == 0){ \n");
  source.append("      if (i < matrix_dimensions[2*matrix_ind]){ \n");
  source.append("        store_householder_vector(R, i+1, row_n, v); \n");
  source.append("      } \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");

  source.append("void matrix_from_global_to_local_qr(__global "); source.append(numeric_string); source.append("* g_M, __local "); source.append(numeric_string); source.append("* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){ \n");
  source.append("  for (unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){ \n");
  source.append("    for (unsigned int j = 0; j < row_n; ++j){ \n");
  source.append("      l_M[i*row_n + j] = g_M[mat_start_ind + i*row_n + j]; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");
  source.append("void matrix_from_local_to_global_qr(__global "); source.append(numeric_string); source.append("* g_M, __local "); source.append(numeric_string); source.append("* l_M, unsigned int row_n, unsigned int col_n, unsigned int mat_start_ind){ \n");
  source.append("  for (unsigned int i = get_local_id(0); i < col_n; i+= get_local_size(0)){ \n");
  source.append("    for (unsigned int j = 0; j < row_n; ++j){ \n");
  source.append("      g_M[mat_start_ind + i*row_n + j] = l_M[i*row_n + j]; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");


  source.append("__kernel void block_qr( \n");
  source.append("      __global "); source.append(numeric_string); source.append("* R, \n");
  source.append("      __global unsigned int* matrix_dimensions, \n");
  source.append("      __global "); source.append(numeric_string); source.append("* b_v, \n");
  source.append("      __global "); source.append(numeric_string); source.append("* v, \n");
  source.append("      __global unsigned int* start_matrix_inds, \n");
  source.append("      __global unsigned int* start_bv_inds, \n");
  source.append("      __global unsigned int* start_v_inds, \n");
  source.append("      __global unsigned int * g_is_update, \n");
  source.append("      __local "); source.append(numeric_string); source.append("* local_buff_R, \n");
  source.append("      unsigned int block_elems_num){ \n");
  source.append("    for (unsigned int i  = get_group_id(0); i < block_elems_num; i += get_num_groups(0)){ \n");
  source.append("        if ((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){ \n");
  source.append("      matrix_from_global_to_local_qr(R, local_buff_R, matrix_dimensions[2*i], matrix_dimensions[2*i + 1], start_matrix_inds[i]); \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("            single_qr(local_buff_R, matrix_dimensions, b_v + start_bv_inds[i], v + start_v_inds[i], i); \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("            matrix_from_local_to_global_qr(R, local_buff_R, matrix_dimensions[2*i], matrix_dimensions[2*i + 1], start_matrix_inds[i]); \n");
  source.append("        } \n");
  source.append("    } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_spai_block_qr_assembly(StringT & source, std::string const & numeric_string)
{
  source.append("void assemble_upper_part(__global "); source.append(numeric_string); source.append(" * R_q, \n");
  source.append("            unsigned int row_n_q, unsigned int col_n_q, __global "); source.append(numeric_string); source.append(" * R_u, \n");
  source.append("            unsigned int row_n_u, unsigned int col_n_u, \n");
  source.append("            unsigned int col_n, unsigned int diff){ \n");
  source.append("            for (unsigned int i = 0; i < col_n_q; ++i){ \n");
  source.append("                for (unsigned int j = 0; j < diff; ++j){ \n");
  source.append("          R_q[ i*row_n_q + j] = R_u[ i*row_n_u + j + col_n ]; \n");
  source.append("                } \n");
  source.append("            } \n");
  source.append("        } \n");

  source.append("void assemble_lower_part(__global "); source.append(numeric_string); source.append(" * R_q, unsigned int row_n_q, unsigned int col_n_q, __global "); source.append(numeric_string); source.append(" * R_u_u, \n");
  source.append("             unsigned int row_n_u_u, unsigned int col_n_u_u, \n");
  source.append("             unsigned int diff){ \n");
  source.append("  for (unsigned int i = 0; i < col_n_u_u; ++i){ \n");
  source.append("    for (unsigned int j = 0; j < row_n_u_u; ++j){ \n");
  source.append("      R_q[i*row_n_q + j + diff] = R_u_u[i*row_n_u_u + j]; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");

  source.append("void assemble_qr_block(__global "); source.append(numeric_string); source.append(" * R_q, unsigned int row_n_q, unsigned int col_n_q, __global "); source.append(numeric_string); source.append(" * R_u, unsigned int row_n_u, \n");
  source.append("            unsigned int col_n_u, __global "); source.append(numeric_string); source.append(" * R_u_u, unsigned int row_n_u_u, unsigned int col_n_u_u, unsigned int col_n){ \n");
  source.append("            unsigned int diff = row_n_u - col_n; \n");
  source.append("            assemble_upper_part(R_q, row_n_q, col_n_q, R_u, row_n_u, col_n_u, col_n, diff); \n");
  source.append("            if (diff > 0){ \n");
  source.append("              assemble_lower_part(R_q, row_n_q, col_n_q, R_u_u, row_n_u_u, col_n_u_u, diff); \n");
  source.append("            } \n");
  source.append("} \n");

  source.append("__kernel void block_qr_assembly( \n");
  source.append("      __global unsigned int * matrix_dimensions, \n");
  source.append("      __global "); source.append(numeric_string); source.append(" * R_u, \n");
  source.append("      __global unsigned int * block_ind_u, \n");
  source.append("      __global unsigned int * matrix_dimensions_u, \n");
  source.append("      __global "); source.append(numeric_string); source.append(" * R_u_u, \n");
  source.append("      __global unsigned int * block_ind_u_u, \n");
  source.append("      __global unsigned int * matrix_dimensions_u_u, \n");
  source.append("      __global "); source.append(numeric_string); source.append(" * R_q, \n");
  source.append("      __global unsigned int * block_ind_q, \n");
  source.append("      __global unsigned int * matrix_dimensions_q, \n");
  source.append("      __global unsigned int * g_is_update, \n");
  source.append("          //__local  "); source.append(numeric_string); source.append(" * local_R_q, \n");
  source.append("      unsigned int  block_elems_num) \n");
  source.append("{ \n");
  source.append("    for (unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){ \n");
  source.append("        if ((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){ \n");
  source.append("           assemble_qr_block(R_q + block_ind_q[i], matrix_dimensions_q[2*i], matrix_dimensions_q[2*i + 1], R_u + block_ind_u[i], matrix_dimensions_u[2*i], \n");
  source.append("             matrix_dimensions_u[2*i + 1], R_u_u + block_ind_u_u[i], matrix_dimensions_u_u[2*i], matrix_dimensions_u_u[2*i + 1], matrix_dimensions[2*i + 1]); \n");
  source.append("       } \n");
  source.append("   } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_spai_block_qr_assembly_1(StringT & source, std::string const & numeric_string)
{
  source.append("void assemble_upper_part_1(__global "); source.append(numeric_string); source.append(" * R_q, unsigned int row_n_q, unsigned int col_n_q, __global "); source.append(numeric_string); source.append(" * R_u, \n");
  source.append("             unsigned int row_n_u, unsigned int col_n_u, \n");
  source.append("             unsigned int col_n, unsigned int diff){ \n");
  source.append("            for (unsigned int i = 0; i < col_n_q; ++i){ \n");
  source.append("                for (unsigned int j = 0; j < diff; ++j){ \n");
  source.append("          R_q[ i*row_n_q + j] = R_u[i*row_n_u + j + col_n ]; \n");
  source.append("                } \n");
  source.append("            } \n");
  source.append("        } \n");


  source.append("void assemble_qr_block_1(__global "); source.append(numeric_string); source.append(" * R_q,  unsigned int row_n_q, unsigned int col_n_q, __global "); source.append(numeric_string); source.append(" * R_u, unsigned int row_n_u, \n");
  source.append("            unsigned int col_n_u, unsigned int col_n){ \n");
  source.append("            unsigned int diff = row_n_u - col_n; \n");
  source.append("            assemble_upper_part_1(R_q, row_n_q, col_n_q, R_u, row_n_u, col_n_u, col_n, diff); \n");
  source.append("} \n");

  source.append("__kernel void block_qr_assembly_1( \n");
  source.append("  __global unsigned int * matrix_dimensions, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * R_u, \n");
  source.append("  __global unsigned int * block_ind_u, \n");
  source.append("  __global unsigned int * matrix_dimensions_u, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * R_q, \n");
  source.append("  __global unsigned int * block_ind_q, \n");
  source.append("  __global unsigned int * matrix_dimensions_q, \n");
  source.append("  __global unsigned int * g_is_update, \n");
  source.append("  unsigned int  block_elems_num) \n");
  source.append("{ \n");
  source.append("    for (unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){ \n");
  source.append("        if ((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){ \n");
  source.append("            assemble_qr_block_1(R_q + block_ind_q[i], matrix_dimensions_q[2*i], matrix_dimensions_q[2*i + 1], R_u + block_ind_u[i], matrix_dimensions_u[2*i], \n");
  source.append("              matrix_dimensions_u[2*i + 1], matrix_dimensions[2*i + 1]); \n");
  source.append("        } \n");
  source.append("    } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_spai_block_r_assembly(StringT & source, std::string const & numeric_string)
{
  source.append("void assemble_r(__global "); source.append(numeric_string); source.append(" * gR, unsigned int row_n_r, unsigned int col_n_r, __global "); source.append(numeric_string); source.append(" * R, \n");
  source.append("        unsigned int row_n, unsigned int col_n) \n");
  source.append("{ \n");
  source.append("  for (unsigned int i = 0; i < col_n; ++i){ \n");
  source.append("     for (unsigned int j = 0; j < row_n; ++j){ \n");
  source.append("    gR[i*row_n_r + j] = R[i*row_n + j ]; \n");
  source.append("     } \n");
  source.append("  } \n");
  source.append("} \n");

  source.append("void assemble_r_u(__global "); source.append(numeric_string); source.append(" * gR, \n");
  source.append("          unsigned int row_n_r, unsigned int col_n_r, __global "); source.append(numeric_string); source.append(" * R_u, unsigned int row_n_u, unsigned int col_n_u, \n");
  source.append("          unsigned int col_n) \n");
  source.append("{ \n");
  source.append("  for (unsigned int i = 0; i < col_n_u; ++i){ \n");
  source.append("    for (unsigned int j = 0; j < col_n; ++j){ \n");
  source.append("      gR[ (i+col_n)*row_n_r + j] = R_u[ i*row_n_u + j]; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");


  source.append("void assemble_r_u_u(__global "); source.append(numeric_string); source.append(" * gR,  unsigned int row_n_r, unsigned int col_n_r, __global "); source.append(numeric_string); source.append(" * R_u_u, unsigned int row_n_u_u, \n");
  source.append("          unsigned int col_n_u_u, unsigned int col_n) \n");
  source.append("{ \n");
  source.append("  for (unsigned int i = 0; i < col_n_u_u; ++i){ \n");
  source.append("    for (unsigned int j = 0; j < row_n_u_u; ++j){ \n");
  source.append("      gR[(col_n+i)*row_n_r + j + col_n] = R_u_u[i*row_n_u_u + j]; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");

  source.append("void assemble_r_block(__global "); source.append(numeric_string); source.append(" * gR, unsigned int row_n_r, unsigned int col_n_r, __global "); source.append(numeric_string); source.append(" * R, unsigned int row_n, \n");
  source.append("        unsigned int col_n, __global "); source.append(numeric_string); source.append(" * R_u, unsigned int row_n_u, unsigned int col_n_u, __global "); source.append(numeric_string); source.append(" * R_u_u, \n");
  source.append("        unsigned int row_n_u_u, unsigned int col_n_u_u){ \n");
  source.append("        assemble_r(gR, row_n_r, col_n_r, R, row_n, col_n); \n");
  source.append("        assemble_r_u(gR, row_n_r, col_n_r, R_u, row_n_u, col_n_u, col_n); \n");
  source.append("        assemble_r_u_u(gR, row_n_r, col_n_r, R_u_u, row_n_u_u, col_n_u_u, col_n); \n");
  source.append("} \n");


  source.append("__kernel void block_r_assembly( \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * R, \n");
  source.append("  __global unsigned int * block_ind, \n");
  source.append("  __global unsigned int * matrix_dimensions, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * R_u, \n");
  source.append("  __global unsigned int * block_ind_u, \n");
  source.append("  __global unsigned int * matrix_dimensions_u, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * R_u_u, \n");
  source.append("  __global unsigned int * block_ind_u_u, \n");
  source.append("  __global unsigned int * matrix_dimensions_u_u, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * g_R, \n");
  source.append("  __global unsigned int * block_ind_r, \n");
  source.append("  __global unsigned int * matrix_dimensions_r, \n");
  source.append("  __global unsigned int * g_is_update, \n");
  source.append("  unsigned int  block_elems_num) \n");
  source.append("{ \n");
  source.append("    for (unsigned int i  = get_global_id(0); i < block_elems_num; i += get_global_size(0)){ \n");
  source.append("        if ((matrix_dimensions[2*i] > 0) && (matrix_dimensions[2*i + 1] > 0) && g_is_update[i] > 0){ \n");

  source.append("            assemble_r_block(g_R + block_ind_r[i], matrix_dimensions_r[2*i], matrix_dimensions_r[2*i + 1], R + block_ind[i], matrix_dimensions[2*i], \n");
  source.append("              matrix_dimensions[2*i + 1], R_u + block_ind_u[i], matrix_dimensions_u[2*i], matrix_dimensions_u[2*i + 1], \n");
  source.append("              R_u_u + block_ind_u_u[i], matrix_dimensions_u_u[2*i], matrix_dimensions_u_u[2*i + 1]); \n");

  source.append("        } \n");
  source.append("    } \n");
  source.append("} \n");
}

//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for the sparse approximate inverse preconditioners. */
template<typename NumericT>
struct spai
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_spai";
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

      generate_spai_assemble_blocks(source, numeric_string);
      generate_spai_block_bv_assembly(source, numeric_string);
      generate_spai_block_least_squares(source, numeric_string);
      generate_spai_block_q_mult(source, numeric_string);
      generate_spai_block_qr(source, numeric_string);
      generate_spai_block_qr_assembly(source, numeric_string);
      generate_spai_block_qr_assembly_1(source, numeric_string);
      generate_spai_block_r_assembly(source, numeric_string);

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

