#ifndef VIENNACL_LINALG_OPENCL_KERNELS_SVD_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_SVD_HPP

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

/** @file viennacl/linalg/opencl/kernels/svd.hpp
 *  @brief OpenCL kernel file for singular value decomposition */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{

template <typename StringType>
void generate_svd_bidiag_pack(StringType & source, std::string const & numeric_string, bool is_row_major)
{
  source.append("__kernel void bidiag_pack(__global "); source.append(numeric_string); source.append("* A, \n");
  source.append("  __global "); source.append(numeric_string); source.append("* D, \n");
  source.append("  __global "); source.append(numeric_string); source.append("* S, \n");
  source.append("  uint size1, \n");
  source.append("  uint size2, \n");
  source.append("  uint stride \n");
  source.append(") { \n");
  source.append("  uint size = min(size1, size2); \n");

  source.append("  if(get_global_id(0) == 0) \n");
  source.append("    S[0] = 0; \n");
  if(is_row_major)
    {
      source.append("  for(uint i = get_global_id(0); i < size ; i += get_global_size(0)) { \n");
      source.append("    D[i] = A[i*stride + i]; \n");
      source.append("    S[i + 1] = (i + 1 < size2) ? A[i*stride + (i + 1)] : 0; \n");
    }
  else
    {
      source.append("  for(uint i = get_global_id(0); i < size ; i += get_global_size(0)) { \n");
      source.append("    D[i] = A[i*stride + i]; \n");
      source.append("    S[i + 1] = (i + 1 < size2) ? A[i + (i + 1) * stride] : 0; \n");
    }
  source.append("  } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_svd_col_reduce_lcl_array(StringT & source, std::string const & numeric_string)
{
  // calculates a sum of local array elements
  source.append("void col_reduce_lcl_array(__local "); source.append(numeric_string); source.append("* sums, uint lcl_id, uint lcl_sz) { \n");
  source.append("    uint step = lcl_sz >> 1; \n");

  source.append("    while (step > 0) { \n");
  source.append("        if (lcl_id < step) { \n");
  source.append("            sums[lcl_id] += sums[lcl_id + step]; \n");
  source.append("        } \n");
  source.append("        step >>= 1; \n");
  source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    } \n");
  source.append("} \n");
}

template <typename StringType>
void generate_svd_copy_col(StringType & source, std::string const & numeric_string, bool is_row_major)
{
  // probably, this is a ugly way
  source.append("__kernel void copy_col(__global "); source.append(numeric_string); source.append("* A, \n");
  source.append("                       __global "); source.append(numeric_string); source.append("* V, \n");
  source.append("                       uint row_start, \n");
  source.append("                       uint col_start, \n");
  source.append("                       uint size, \n");
  source.append("                       uint stride \n");
  source.append("                       ) { \n");
  source.append("    uint glb_id = get_global_id(0); \n");
  source.append("    uint glb_sz = get_global_size(0); \n");
  if(is_row_major)
    {
      source.append("    for(uint i = row_start + glb_id; i < size; i += glb_sz) { \n");
      source.append("        V[i - row_start] = A[i * stride + col_start]; \n");
      source.append("    } \n");
    }
  else
    {
      source.append("    for(uint i = row_start + glb_id; i < size; i += glb_sz) { \n");
      source.append("        V[i - row_start] = A[i + col_start * stride]; \n");
      source.append("    } \n");
    }

  source.append("} \n");
}

template <typename StringType>
void generate_svd_copy_row(StringType & source, std::string const & numeric_string, bool is_row_major)
{
  // probably, this is too
  source.append("__kernel void copy_row(__global "); source.append(numeric_string); source.append("* A, \n");
  source.append("                       __global "); source.append(numeric_string); source.append("* V, \n");
  source.append("                       uint row_start, \n");
  source.append("                       uint col_start, \n");
  source.append("                       uint size, \n");
  source.append("                       uint stride \n");
  source.append("                       ) { \n");
  source.append("    uint glb_id = get_global_id(0); \n");
  source.append("    uint glb_sz = get_global_size(0); \n");
  if(is_row_major)
    {
      source.append("    for(uint i = col_start + glb_id; i < size; i += glb_sz) { \n");
      source.append("        V[i - col_start] = A[row_start * stride + i]; \n");
      source.append("    } \n");
    }
  else
    {
      source.append("    for(uint i = col_start + glb_id; i < size; i += glb_sz) { \n");
      source.append("        V[i - col_start] = A[row_start + i * stride]; \n");
      source.append("    } \n");
    }

  source.append("} \n");
}

template<typename StringT>
void generate_svd_final_iter_update(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void final_iter_update(__global "); source.append(numeric_string); source.append("* A, \n");
  source.append("                                uint stride, \n");
  source.append("                                uint n, \n");
  source.append("                                uint last_n, \n");
  source.append("                                "); source.append(numeric_string); source.append(" q, \n");
  source.append("                                "); source.append(numeric_string); source.append(" p \n");
  source.append("                                ) \n");
  source.append("{ \n");
  source.append("    uint glb_id = get_global_id(0); \n");
  source.append("    uint glb_sz = get_global_size(0); \n");

  source.append("    for (uint px = glb_id; px < last_n; px += glb_sz) \n");
  source.append("    { \n");
  source.append("        "); source.append(numeric_string); source.append(" v_in = A[n * stride + px]; \n");
  source.append("        "); source.append(numeric_string); source.append(" z = A[(n - 1) * stride + px]; \n");
  source.append("        A[(n - 1) * stride + px] = q * z + p * v_in; \n");
  source.append("        A[n * stride + px] = q * v_in - p * z; \n");
  source.append("    } \n");
  source.append("} \n");
}

template <typename StringType>
void generate_svd_givens_next(StringType & source, std::string const & numeric_string, bool is_row_major)
{
  source.append("__kernel void givens_next(__global "); source.append(numeric_string); source.append("* matr, \n");
  source.append("                            __global "); source.append(numeric_string); source.append("* cs, \n");
  source.append("                            __global "); source.append(numeric_string); source.append("* ss, \n");
  source.append("                            uint size, \n");
  source.append("                            uint stride, \n");
  source.append("                            uint start_i, \n");
  source.append("                            uint end_i \n");
  source.append("                            ) \n");
  source.append("{ \n");
  source.append("    uint glb_id = get_global_id(0); \n");
  source.append("    uint glb_sz = get_global_size(0); \n");

  source.append("    uint lcl_id = get_local_id(0); \n");
  source.append("    uint lcl_sz = get_local_size(0); \n");

  source.append("    uint j = glb_id; \n");

  source.append("    __local "); source.append(numeric_string); source.append(" cs_lcl[256]; \n");
  source.append("    __local "); source.append(numeric_string); source.append(" ss_lcl[256]; \n");
  if(is_row_major)
    {

      source.append("    "); source.append(numeric_string); source.append(" x = (j < size) ? matr[(end_i + 1) + j * stride] : 0; \n");

      source.append("    uint elems_num = end_i - start_i + 1; \n");
      source.append("    uint block_num = (elems_num + lcl_sz - 1) / lcl_sz; \n");

      source.append("    for(uint block_id = 0; block_id < block_num; block_id++) \n");
      source.append("    { \n");
      source.append("        uint to = min(elems_num - block_id * lcl_sz, lcl_sz); \n");

      source.append("        if(lcl_id < to) \n");
      source.append("        { \n");
      source.append("            cs_lcl[lcl_id] = cs[end_i - (lcl_id + block_id * lcl_sz)]; \n");
      source.append("            ss_lcl[lcl_id] = ss[end_i - (lcl_id + block_id * lcl_sz)]; \n");
      source.append("        } \n");

      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");

      source.append("        if(j < size) \n");
      source.append("        { \n");
      source.append("            for(uint ind = 0; ind < to; ind++) \n");
      source.append("            { \n");
      source.append("                uint i = end_i - (ind + block_id * lcl_sz); \n");

      source.append("                "); source.append(numeric_string); source.append(" z = matr[i + j * stride]; \n");

      source.append("                "); source.append(numeric_string); source.append(" cs_val = cs_lcl[ind]; \n");
      source.append("                "); source.append(numeric_string); source.append(" ss_val = ss_lcl[ind]; \n");

      source.append("                matr[(i + 1) + j * stride] = x * cs_val + z * ss_val; \n");
      source.append("                x = -x * ss_val + z * cs_val; \n");
      source.append("            } \n");
      source.append("        } \n");
      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
      source.append("    } \n");
      source.append("    if(j < size) \n");
      source.append("        matr[(start_i) + j * stride] = x; \n");
    }
  else
    {

      source.append("    "); source.append(numeric_string); source.append(" x = (j < size) ? matr[(end_i + 1) * stride + j] : 0; \n");

      source.append("    uint elems_num = end_i - start_i + 1; \n");
      source.append("    uint block_num = (elems_num + lcl_sz - 1) / lcl_sz; \n");

      source.append("    for(uint block_id = 0; block_id < block_num; block_id++) \n");
      source.append("    { \n");
      source.append("        uint to = min(elems_num - block_id * lcl_sz, lcl_sz); \n");

      source.append("        if(lcl_id < to) \n");
      source.append("        { \n");
      source.append("            cs_lcl[lcl_id] = cs[end_i - (lcl_id + block_id * lcl_sz)]; \n");
      source.append("            ss_lcl[lcl_id] = ss[end_i - (lcl_id + block_id * lcl_sz)]; \n");
      source.append("        } \n");

      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");

      source.append("        if(j < size) \n");
      source.append("        { \n");
      source.append("            for(uint ind = 0; ind < to; ind++) \n");
      source.append("            { \n");
      source.append("                uint i = end_i - (ind + block_id * lcl_sz); \n");

      source.append("                "); source.append(numeric_string); source.append(" z = matr[i * stride + j]; \n");

      source.append("                "); source.append(numeric_string); source.append(" cs_val = cs_lcl[ind]; \n");
      source.append("                "); source.append(numeric_string); source.append(" ss_val = ss_lcl[ind]; \n");

      source.append("                matr[(i + 1) * stride + j] = x * cs_val + z * ss_val; \n");
      source.append("                x = -x * ss_val + z * cs_val; \n");
      source.append("            } \n");
      source.append("        } \n");
      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
      source.append("    } \n");
      source.append("    if(j < size) \n");
      source.append("        matr[(start_i) * stride + j] = x; \n");
    }
  source.append("} \n");
}

template<typename StringT>
void generate_svd_givens_prev(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void givens_prev(__global "); source.append(numeric_string); source.append("* matr, \n");
  source.append("                            __global "); source.append(numeric_string); source.append("* cs, \n");
  source.append("                            __global "); source.append(numeric_string); source.append("* ss, \n");
  source.append("                            uint size, \n");
  source.append("                            uint stride, \n");
  source.append("                            uint start_i, \n");
  source.append("                            uint end_i \n");
  source.append("                            ) \n");
  source.append("{ \n");
  source.append("    uint glb_id = get_global_id(0); \n");
  source.append("    uint glb_sz = get_global_size(0); \n");

  source.append("    uint lcl_id = get_local_id(0); \n");
  source.append("    uint lcl_sz = get_local_size(0); \n");

  source.append("    uint j = glb_id; \n");

  source.append("    __local "); source.append(numeric_string); source.append(" cs_lcl[256]; \n");
  source.append("    __local "); source.append(numeric_string); source.append(" ss_lcl[256]; \n");

  source.append("    "); source.append(numeric_string); source.append(" x = (j < size) ? matr[(start_i - 1) * stride + j] : 0; \n");

  source.append("    uint elems_num = end_i - start_i; \n");
  source.append("    uint block_num = (elems_num + lcl_sz - 1) / lcl_sz; \n");

  source.append("    for (uint block_id = 0; block_id < block_num; block_id++) \n");
  source.append("    { \n");
  source.append("        uint to = min(elems_num - block_id * lcl_sz, lcl_sz); \n");

  source.append("        if (lcl_id < to) \n");
  source.append("        { \n");
  source.append("            cs_lcl[lcl_id] = cs[lcl_id + start_i + block_id * lcl_sz]; \n");
  source.append("            ss_lcl[lcl_id] = ss[lcl_id + start_i + block_id * lcl_sz]; \n");
  source.append("        } \n");

  source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");

  source.append("        if (j < size) \n");
  source.append("        { \n");
  source.append("            for (uint ind = 0; ind < to; ind++) \n");
  source.append("            { \n");
  source.append("                uint i = ind + start_i + block_id * lcl_sz; \n");

  source.append("                "); source.append(numeric_string); source.append(" z = matr[i * stride + j]; \n");

  source.append("                "); source.append(numeric_string); source.append(" cs_val = cs_lcl[ind];//cs[i]; \n");
  source.append("                "); source.append(numeric_string); source.append(" ss_val = ss_lcl[ind];//ss[i]; \n");

  source.append("                matr[(i - 1) * stride + j] = x * cs_val + z * ss_val; \n");
  source.append("                x = -x * ss_val + z * cs_val; \n");
  source.append("            } \n");
  source.append("        } \n");
  source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    } \n");
  source.append("    if (j < size) \n");
  source.append("        matr[(end_i - 1) * stride + j] = x; \n");
  source.append("} \n");
}

template <typename StringType>
void generate_svd_house_update_A_left(StringType & source, std::string const & numeric_string, bool is_row_major)
{
  source.append("__kernel void house_update_A_left( \n");
  source.append("                        __global "); source.append(numeric_string); source.append("* A, \n");
  source.append("                        __constant "); source.append(numeric_string); source.append("* V, \n"); //householder vector
  source.append("                        uint row_start, \n");
  source.append("                        uint col_start, \n");
  source.append("                        uint size1, \n");
  source.append("                        uint size2, \n");
  source.append("                        uint stride, \n");
  source.append("                        __local "); source.append(numeric_string); source.append("* sums \n");
  source.append("                        ) { \n");
  source.append("    uint glb_id = get_global_id(0); \n");
  source.append("    uint glb_sz = get_global_size(0); \n");

  source.append("    uint grp_id = get_group_id(0); \n");
  source.append("    uint grp_nm = get_num_groups(0); \n");

  source.append("    uint lcl_id = get_local_id(0); \n");
  source.append("    uint lcl_sz = get_local_size(0); \n");

  source.append("    "); source.append(numeric_string); source.append(" ss = 0; \n");

      // doing it in slightly different way to avoid cache misses
  if(is_row_major)
    {
      source.append("    for(uint i = glb_id + col_start; i < size2; i += glb_sz) { \n");
      source.append("        ss = 0; \n");
      source.append("        for(uint j = row_start; j < size1; j++) ss = ss + (V[j] * A[j * stride + i]); \n");

      source.append("        for(uint j = row_start; j < size1; j++) \n");
      source.append("            A[j * stride + i] = A[j * stride + i] - (2 * V[j] * ss); \n");
      source.append("    } \n");
    }
  else
    {
      source.append("    for(uint i = glb_id + col_start; i < size2; i += glb_sz) { \n");
      source.append("        ss = 0; \n");
      source.append("        for(uint j = row_start; j < size1; j++) ss = ss + (V[j] * A[j + i * stride]); \n");

      source.append("        for(uint j = row_start; j < size1; j++) \n");
      source.append("            A[j + i * stride] = A[j + i * stride] - (2 * V[j] * ss); \n");
      source.append("    } \n");
    }
  source.append("} \n");
}

template <typename StringType>
void generate_svd_house_update_A_right(StringType & source, std::string const & numeric_string, bool is_row_major)
{

  source.append("__kernel void house_update_A_right( \n");
  source.append("                        __global "); source.append(numeric_string); source.append("* A, \n");
  source.append("                        __global "); source.append(numeric_string); source.append("* V, \n"); // householder vector
  source.append("                        uint row_start, \n");
  source.append("                        uint col_start, \n");
  source.append("                        uint size1, \n");
  source.append("                        uint size2, \n");
  source.append("                        uint stride, \n");
  source.append("                        __local "); source.append(numeric_string); source.append("* sums \n");
  source.append("                        ) { \n");

  source.append("    uint glb_id = get_global_id(0); \n");

  source.append("    uint grp_id = get_group_id(0); \n");
  source.append("    uint grp_nm = get_num_groups(0); \n");

  source.append("    uint lcl_id = get_local_id(0); \n");
  source.append("    uint lcl_sz = get_local_size(0); \n");

  source.append("    "); source.append(numeric_string); source.append(" ss = 0; \n");

      // update of A matrix
  if(is_row_major)
    {
      source.append("    for(uint i = grp_id + row_start; i < size1; i += grp_nm) { \n");
      source.append("        ss = 0; \n");

      source.append("        for(uint j = lcl_id; j < size2; j += lcl_sz) ss = ss + (V[j] * A[i * stride + j]); \n");
      source.append("        sums[lcl_id] = ss; \n");

      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
      source.append("        col_reduce_lcl_array(sums, lcl_id, lcl_sz); \n");
      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");

      source.append("        "); source.append(numeric_string); source.append(" sum_Av = sums[0]; \n");

      source.append("        for(uint j = lcl_id; j < size2; j += lcl_sz) \n");
      source.append("            A[i * stride + j] = A[i * stride + j] - (2 * V[j] * sum_Av); \n");
      source.append("    } \n");
    }
  else
    {
      source.append("    for(uint i = grp_id + row_start; i < size1; i += grp_nm) { \n");
      source.append("        ss = 0; \n");

      source.append("        for(uint j = lcl_id; j < size2; j += lcl_sz) ss = ss + (V[j] * A[i + j * stride]); \n");
      source.append("        sums[lcl_id] = ss; \n");

      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
      source.append("        col_reduce_lcl_array(sums, lcl_id, lcl_sz); \n");
      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");

      source.append("        "); source.append(numeric_string); source.append(" sum_Av = sums[0]; \n");

      source.append("        for(uint j = lcl_id; j < size2; j += lcl_sz) \n");
      source.append("            A[i + j * stride] = A[i + j * stride] - (2 * V[j] * sum_Av); \n");
      source.append("    } \n");
    }

  source.append("} \n");

}

template <typename StringType>
void generate_svd_house_update_QL(StringType & source, std::string const & numeric_string, bool is_row_major)
{
  source.append("__kernel void house_update_QL(\n");
  source.append("                        __global "); source.append(numeric_string); source.append("* QL, \n");
  source.append("                        __constant "); source.append(numeric_string); source.append("* V, \n"); //householder vector
  source.append("                        uint size1, \n");
  source.append("                        uint strideQ, \n");
  source.append("                        __local "); source.append(numeric_string); source.append("* sums \n");
  source.append("                        ) { \n");
  source.append("    uint glb_id = get_global_id(0); \n");
  source.append("    uint glb_sz = get_global_size(0); \n");

  source.append("    uint grp_id = get_group_id(0); \n");
  source.append("    uint grp_nm = get_num_groups(0); \n");

  source.append("    uint lcl_id = get_local_id(0); \n");
  source.append("    uint lcl_sz = get_local_size(0); \n");

  source.append("    "); source.append(numeric_string); source.append(" ss = 0; \n");

  if(is_row_major)
    {
      source.append("    for(uint i = grp_id; i < size1; i += grp_nm) { \n");
      source.append("        ss = 0; \n");
      source.append("        for(uint j = lcl_id; j < size1; j += lcl_sz) ss = ss + (V[j] * QL[i * strideQ + j]); \n");
      source.append("        sums[lcl_id] = ss; \n");

      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
      source.append("        col_reduce_lcl_array(sums, lcl_id, lcl_sz); \n");
      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");

      source.append("        "); source.append(numeric_string); source.append(" sum_Qv = sums[0]; \n");

      source.append("        for(uint j = lcl_id; j < size1; j += lcl_sz) \n");
      source.append("            QL[i * strideQ + j] = QL[i * strideQ + j] - (2 * V[j] * sum_Qv); \n");
      source.append("    } \n");
    }
  else
    {
      source.append("    for(uint i = grp_id; i < size1; i += grp_nm) { \n");
      source.append("        ss = 0; \n");
      source.append("        for(uint j = lcl_id; j < size1; j += lcl_sz) ss = ss + (V[j] * QL[i + j * strideQ]); \n");
      source.append("        sums[lcl_id] = ss; \n");

      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
      source.append("        col_reduce_lcl_array(sums, lcl_id, lcl_sz); \n");
      source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");

      source.append("        "); source.append(numeric_string); source.append(" sum_Qv = sums[0]; \n");

      source.append("        for(uint j = lcl_id; j < size1; j += lcl_sz) \n");
      source.append("            QL[i + j * strideQ] = QL[i + j * strideQ] - (2 * V[j] * sum_Qv); \n");
      source.append("    } \n");
    }
  source.append("} \n");

}

template<typename StringT>
void generate_svd_house_update_QR(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void house_update_QR( \n");
  source.append("                        __global "); source.append(numeric_string); source.append("* QR, \n");
  source.append("                        __global "); source.append(numeric_string); source.append("* V, \n"); // householder vector
  source.append("                        uint size1, \n");
  source.append("                        uint size2, \n");
  source.append("                        uint strideQ, \n");
  source.append("                        __local "); source.append(numeric_string); source.append("* sums \n");
  source.append("                        ) { \n");

  source.append("    uint glb_id = get_global_id(0); \n");

  source.append("    uint grp_id = get_group_id(0); \n");
  source.append("    uint grp_nm = get_num_groups(0); \n");

  source.append("    uint lcl_id = get_local_id(0); \n");
  source.append("    uint lcl_sz = get_local_size(0); \n");

  source.append("   "); source.append(numeric_string); source.append(" ss = 0; \n");

      // update of QR matrix
      // Actually, we are calculating a transpose of right matrix. This allows to avoid cache
      // misses.
  source.append("    for (uint i = grp_id; i < size2; i += grp_nm) { \n");
  source.append("        ss = 0; \n");
  source.append("        for (uint j = lcl_id; j < size2; j += lcl_sz) ss = ss + (V[j] * QR[i * strideQ + j]); \n");
  source.append("        sums[lcl_id] = ss; \n");

  source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("        col_reduce_lcl_array(sums, lcl_id, lcl_sz); \n");
  source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");

  source.append("        "); source.append(numeric_string); source.append(" sum_Qv = sums[0]; \n");
  source.append("        for (uint j = lcl_id; j < size2; j += lcl_sz) \n");
  source.append("            QR[i * strideQ + j] = QR[i * strideQ + j] - (2 * V[j] * sum_Qv); \n");
  source.append("    } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_svd_inverse_signs(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void inverse_signs(__global "); source.append(numeric_string); source.append("* v, \n");
  source.append("                            __global "); source.append(numeric_string); source.append("* signs, \n");
  source.append("                            uint size, \n");
  source.append("                            uint stride \n");
  source.append("                            ) \n");
  source.append("{ \n");
  source.append("    uint glb_id_x = get_global_id(0); \n");
  source.append("    uint glb_id_y = get_global_id(1); \n");

  source.append("    if ((glb_id_x < size) && (glb_id_y < size)) \n");
  source.append("        v[glb_id_x * stride + glb_id_y] *= signs[glb_id_x]; \n");
  source.append("} \n");

}

template<typename StringT>
void generate_svd_transpose_inplace(StringT & source, std::string const & numeric_string)
{

  source.append("__kernel void transpose_inplace(__global "); source.append(numeric_string); source.append("* input, \n");
  source.append("                        unsigned int row_num, \n");
  source.append("                        unsigned int col_num) { \n");
  source.append("    unsigned int size = row_num * col_num; \n");
  source.append("    for (unsigned int i = get_global_id(0); i < size; i+= get_global_size(0)) { \n");
  source.append("        unsigned int row = i / col_num; \n");
  source.append("        unsigned int col = i - row*col_num; \n");

  source.append("        unsigned int new_pos = col * row_num + row; \n");

          //new_pos = (col < row) ? 0 : 1;
          //input[i] = new_pos;

  source.append("        if (i < new_pos) { \n");
  source.append("            "); source.append(numeric_string); source.append(" val = input[i]; \n");
  source.append("            input[i] = input[new_pos]; \n");
  source.append("            input[new_pos] = val; \n");
  source.append("        } \n");
  source.append("    } \n");
  source.append("} \n");

}

template<typename StringT>
void generate_svd_update_qr_column(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void update_qr_column(__global "); source.append(numeric_string); source.append("* A, \n");
  source.append("                               uint stride, \n");
  source.append("                               __global "); source.append(numeric_string); source.append("* buf, \n");
  source.append("                               int m, \n");
  source.append("                               int n, \n");
  source.append("                               int last_n) \n");
  source.append("{ \n");
  source.append("    uint glb_id = get_global_id(0); \n");
  source.append("    uint glb_sz = get_global_size(0); \n");

  source.append("    for (int i = glb_id; i < last_n; i += glb_sz) \n");
  source.append("    { \n");
  source.append("        "); source.append(numeric_string); source.append(" a_ik = A[m * stride + i], a_ik_1, a_ik_2; \n");

  source.append("        a_ik_1 = A[(m + 1) * stride + i]; \n");

  source.append("        for (int k = m; k < n; k++) \n");
  source.append("        { \n");
  source.append("            bool notlast = (k != n - 1); \n");

  source.append("            "); source.append(numeric_string); source.append(" p = buf[5 * k] * a_ik + buf[5 * k + 1] * a_ik_1; \n");

  source.append("            if (notlast) \n");
  source.append("            { \n");
  source.append("                a_ik_2 = A[(k + 2) * stride + i]; \n");
  source.append("                p = p + buf[5 * k + 2] * a_ik_2; \n");
  source.append("                a_ik_2 = a_ik_2 - p * buf[5 * k + 4]; \n");
  source.append("            } \n");

  source.append("            A[k * stride + i] = a_ik - p; \n");
  source.append("            a_ik_1 = a_ik_1 - p * buf[5 * k + 3]; \n");

  source.append("            a_ik = a_ik_1; \n");
  source.append("            a_ik_1 = a_ik_2; \n");
  source.append("        } \n");

  source.append("        A[n * stride + i] = a_ik; \n");
  source.append("    } \n");

  source.append("} \n");
}




// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for singular value decomposition of dense matrices. */
template<typename NumericT, typename MatrixLayout = row_major>
struct svd
{
  static std::string program_name()
  {
    bool is_row = viennacl::is_row_major<MatrixLayout>::value;
    return (viennacl::ocl::type_to_string<NumericT>::apply() + "_svd_") + (is_row ? "row" : "col");
  }

  static void init(viennacl::ocl::context & ctx)
  {
    static std::map<cl_context, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
      std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();
      bool is_row_major = viennacl::is_row_major<MatrixLayout>::value;

      std::string source;
      source.reserve(1024);

      viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

      // only generate for floating points (forces error for integers)
      if (numeric_string == "float" || numeric_string == "double")
      {
        //helper function used by multiple kernels:
        generate_svd_col_reduce_lcl_array(source, numeric_string);

        //kernels:
        generate_svd_bidiag_pack(source, numeric_string, is_row_major);
        generate_svd_copy_col(source, numeric_string, is_row_major);
        generate_svd_copy_row(source, numeric_string, is_row_major);
        generate_svd_final_iter_update(source, numeric_string);
        generate_svd_givens_next(source, numeric_string, is_row_major);
        generate_svd_givens_prev(source, numeric_string);
        generate_svd_house_update_A_left(source, numeric_string, is_row_major);
        generate_svd_house_update_A_right(source, numeric_string, is_row_major);
        generate_svd_house_update_QL(source, numeric_string, is_row_major);
        generate_svd_house_update_QR(source, numeric_string);
        generate_svd_inverse_signs(source, numeric_string);
        generate_svd_transpose_inplace(source, numeric_string);
        generate_svd_update_qr_column(source, numeric_string);
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

