#ifndef VIENNACL_LINALG_OPENCL_KERNELS_ITERATIVE_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_ITERATIVE_HPP

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

#include "viennacl/vector_proxy.hpp"

#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/io.hpp"
#include "viennacl/scheduler/preset.hpp"

#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

/** @file viennacl/linalg/opencl/kernels/iterative.hpp
 *  @brief OpenCL kernel file for specialized iterative solver kernels */
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
void generate_pipelined_cg_vector_update(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_vector_update( \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * result, \n");
  source.append("  "); source.append(numeric_string); source.append(" alpha, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * r, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * Ap, \n");
  source.append("  "); source.append(numeric_string); source.append(" beta, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_contrib = 0; \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" value_p = p[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_r = r[i]; \n");
  source.append("     \n");
  source.append("    result[i] += alpha * value_p; \n");
  source.append("    value_r   -= alpha * Ap[i]; \n");
  source.append("    value_p    = value_r + beta * value_p; \n");
  source.append("     \n");
  source.append("    p[i] = value_p; \n");
  source.append("    r[i] = value_r; \n");
  source.append("    inner_prod_contrib += value_r * value_r; \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array[get_local_id(0)] = inner_prod_contrib; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride)  \n");
  source.append("      shared_array[get_local_id(0)] += shared_array[get_local_id(0) + stride];  \n");
  source.append("  } ");

  // write results to result array
  source.append(" if (get_local_id(0) == 0) \n ");
  source.append("   inner_prod_buffer[get_group_id(0)] = shared_array[0]; ");

  source.append("} \n");
}

template<typename StringT>
void generate_compressed_matrix_pipelined_cg_blocked_prod(StringT & source, std::string const & numeric_string, unsigned int subwarp_size)
{
  std::stringstream ss;
  ss << subwarp_size;

  source.append("__kernel void cg_csr_blocked_prod( \n");
  source.append("    __global const unsigned int * row_indices, \n");
  source.append("    __global const unsigned int * column_indices, \n");
  source.append("    __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("    __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("    __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("    unsigned int size, \n");
  source.append("    __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("    unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  __local "); source.append(numeric_string); source.append(" shared_elements[256]; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");

  source.append("  const unsigned int id_in_row = get_local_id(0) % " + ss.str() + "; \n");
  source.append("  const unsigned int block_increment = get_local_size(0) * ((size - 1) / (get_global_size(0)) + 1); \n");
  source.append("  const unsigned int block_start = get_group_id(0) * block_increment; \n");
  source.append("  const unsigned int block_stop  = min(block_start + block_increment, size); \n");

  source.append("  for (unsigned int row  = block_start + get_local_id(0) / " + ss.str() + "; \n");
  source.append("                    row  < block_stop; \n");
  source.append("                    row += get_local_size(0) / " + ss.str() + ") \n");
  source.append("  { \n");
  source.append("    "); source.append(numeric_string); source.append(" dot_prod = 0; \n");
  source.append("    unsigned int row_end = row_indices[row+1]; \n");
  source.append("    for (unsigned int i = row_indices[row] + id_in_row; i < row_end; i += " + ss.str() + ") \n");
  source.append("      dot_prod += elements[i] * p[column_indices[i]]; \n");

  source.append("    shared_elements[get_local_id(0)] = dot_prod; \n");
  source.append("    #pragma unroll \n");
  source.append("    for (unsigned int k = 1; k < " + ss.str() + "; k *= 2) \n");
  source.append("      shared_elements[get_local_id(0)] += shared_elements[get_local_id(0) ^ k]; \n");

  source.append("    if (id_in_row == 0) { \n");
  source.append("      Ap[row] = shared_elements[get_local_id(0)]; \n");
  source.append("      inner_prod_ApAp += shared_elements[get_local_id(0)] * shared_elements[get_local_id(0)]; \n");
  source.append("      inner_prod_pAp  +=                           p[row] * shared_elements[get_local_id(0)]; \n");
  source.append("    } \n");
  source.append("  } \n");

  ////////// parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("  } \n");

  source.append("} \n");
}

template<typename StringT>
void generate_compressed_matrix_pipelined_cg_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_csr_prod( \n");
  source.append("  __global const unsigned int * row_indices, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const unsigned int * row_blocks, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  unsigned int num_blocks, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_elements) \n");
  source.append("{ \n");

  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");

  source.append("  for (unsigned int block_id = get_group_id(0); block_id < num_blocks; block_id += get_num_groups(0)) { \n");
  source.append("    unsigned int row_start = row_blocks[block_id]; \n");
  source.append("    unsigned int row_stop  = row_blocks[block_id + 1]; \n");
  source.append("    unsigned int rows_to_process = row_stop - row_start; \n");
  source.append("    unsigned int element_start = row_indices[row_start]; \n");
  source.append("    unsigned int element_stop = row_indices[row_stop]; \n");

  source.append("    if (rows_to_process > 1) { \n"); // CSR stream
      // load to shared buffer:
  source.append("      for (unsigned int i = element_start + get_local_id(0); i < element_stop; i += get_local_size(0)) \n");
  source.append("        shared_elements[i - element_start] = elements[i] * p[column_indices[i]]; \n");

  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");

      // use one thread per row to sum:
  source.append("      for (unsigned int row = row_start + get_local_id(0); row < row_stop; row += get_local_size(0)) { \n");
  source.append("        "); source.append(numeric_string); source.append(" dot_prod = 0; \n");
  source.append("        unsigned int thread_row_start = row_indices[row]     - element_start; \n");
  source.append("        unsigned int thread_row_stop  = row_indices[row + 1] - element_start; \n");
  source.append("        for (unsigned int i = thread_row_start; i < thread_row_stop; ++i) \n");
  source.append("          dot_prod += shared_elements[i]; \n");
  source.append("        Ap[row] = dot_prod; \n");
  source.append("        inner_prod_ApAp += dot_prod * dot_prod; \n");
  source.append("        inner_prod_pAp  +=   p[row] * dot_prod; \n");
  source.append("      } \n");
  source.append("    } \n");

  source.append("    else  \n"); // CSR vector for a single row
  source.append("    { \n");
      // load and sum to shared buffer:
  source.append("      shared_elements[get_local_id(0)] = 0; \n");
  source.append("      for (unsigned int i = element_start + get_local_id(0); i < element_stop; i += get_local_size(0)) \n");
  source.append("        shared_elements[get_local_id(0)] += elements[i] * p[column_indices[i]]; \n");

      // reduction to obtain final result
  source.append("      for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) { \n");
  source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("        if (get_local_id(0) < stride) \n");
  source.append("          shared_elements[get_local_id(0)] += shared_elements[get_local_id(0) + stride]; \n");
  source.append("      } \n");

  source.append("      if (get_local_id(0) == 0) { \n");
  source.append("        Ap[row_start] = shared_elements[0]; \n");
  source.append("        inner_prod_ApAp += shared_elements[0] * shared_elements[0]; \n");
  source.append("        inner_prod_pAp  +=       p[row_start] * shared_elements[0]; \n");
  source.append("      } \n");
  source.append("    } \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("  } \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("  } \n");

  source.append("} \n");

}


template<typename StringT>
void generate_coordinate_matrix_pipelined_cg_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_coo_prod( \n");
  source.append("  __global const uint2 * coords,  \n");//(row_index, column_index)
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const uint  * group_boundaries, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __local unsigned int * shared_rows, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * inter_results, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");

  ///////////// Sparse matrix-vector multiplication part /////////////
  source.append("  uint2 tmp; \n");
  source.append("  "); source.append(numeric_string); source.append(" val; \n");
  source.append("  uint group_start = group_boundaries[get_group_id(0)]; \n");
  source.append("  uint group_end   = group_boundaries[get_group_id(0) + 1]; \n");
  source.append("  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : 0; \n");   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)

  source.append("  uint local_index = 0; \n");

  source.append("  for (uint k = 0; k < k_end; ++k) { \n");
  source.append("    local_index = group_start + k * get_local_size(0) + get_local_id(0); \n");

  source.append("    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0; \n");
  source.append("    val = (local_index < group_end) ? elements[local_index] * p[tmp.y] : 0; \n");

  //check for carry from previous loop run:
  source.append("    if (get_local_id(0) == 0 && k > 0) { \n");
  source.append("      if (tmp.x == shared_rows[get_local_size(0)-1]) \n");
  source.append("        val += inter_results[get_local_size(0)-1]; \n");
  source.append("      else {\n");
  source.append("        "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_size(0)-1]; \n");
  source.append("        Ap[shared_rows[get_local_size(0)-1]] = Ap_entry; \n");
  source.append("        inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("        inner_prod_pAp  += p[shared_rows[get_local_size(0)-1]] * Ap_entry; \n");
  source.append("      } \n");
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
  source.append("      "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_id(0)]; \n");
  source.append("      Ap[tmp.x] = Ap_entry; \n");
  source.append("      inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("      inner_prod_pAp  += p[tmp.x] * Ap_entry; \n");
  source.append("    } \n");

  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("  }  \n"); //for k

  source.append("  if (local_index + 1 == group_end) {\n");  //write results of last active entry (this may not necessarily be the case already)
  source.append("    "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_id(0)]; \n");
  source.append("    Ap[tmp.x] = Ap_entry; \n");
  source.append("    inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("    inner_prod_pAp  += p[tmp.x] * Ap_entry; \n");
  source.append("  }  \n");

  //////////// parallel reduction of inner product contributions within work group ///////////////
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("  } \n");

  source.append("} \n \n");

}


template<typename StringT>
void generate_ell_matrix_pipelined_cg_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_ell_prod( \n");
  source.append("  __global const unsigned int * coords, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  unsigned int internal_row_num, \n");
  source.append("  unsigned int items_per_row, \n");
  source.append("  unsigned int aligned_items_per_row, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  uint glb_id = get_global_id(0); \n");
  source.append("  uint glb_sz = get_global_size(0); \n");

  source.append("  for (uint row = glb_id; row < size; row += glb_sz) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint offset = row; \n");
  source.append("    for (uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num) { \n");
  source.append("      "); source.append(numeric_string); source.append(" val = elements[offset]; \n");
  source.append("      sum += (val != 0) ? p[coords[offset]] * val : ("); source.append(numeric_string); source.append(")0; \n");
  source.append("    } \n");

  source.append("    Ap[row] = sum; \n");
  source.append("    inner_prod_ApAp += sum * sum; \n");
  source.append("    inner_prod_pAp  += p[row] * sum; \n");
  source.append("  }  \n");

  //////////// parallel reduction of inner product contributions within work group ///////////////
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}

template<typename StringT>
void generate_sliced_ell_matrix_pipelined_cg_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_sliced_ell_prod( \n");
  source.append("  __global const unsigned int * columns_per_block, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const unsigned int * block_start, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  unsigned int block_size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  uint blocks_per_workgroup = get_local_size(0) / block_size; \n");
  source.append("  uint id_in_block = get_local_id(0) % block_size; \n");
  source.append("  uint num_blocks  = (size - 1) / block_size + 1; \n");
  source.append("  uint global_warp_count  = blocks_per_workgroup * get_num_groups(0); \n");
  source.append("  uint global_warp_id     = blocks_per_workgroup * get_group_id(0) + get_local_id(0) / block_size; \n");

  source.append("  for (uint block_idx = global_warp_id; block_idx < num_blocks; block_idx += global_warp_count) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint row    = block_idx * block_size + id_in_block; \n");
  source.append("    uint offset = block_start[block_idx]; \n");
  source.append("    uint num_columns = columns_per_block[block_idx]; \n");
  source.append("    for (uint item_id = 0; item_id < num_columns; item_id++) { \n");
  source.append("      uint index = offset + item_id * block_size + id_in_block; \n");
  source.append("      "); source.append(numeric_string); source.append(" val = elements[index]; \n");
  source.append("      sum += (val != 0) ? (p[column_indices[index]] * val) : 0; \n");
  source.append("    } \n");

  source.append("    if (row < size) {\n");
  source.append("      Ap[row] = sum; \n");
  source.append("      inner_prod_ApAp += sum * sum; \n");
  source.append("      inner_prod_pAp  += p[row] * sum; \n");
  source.append("    }  \n");
  source.append("  }  \n");

  //////////// parallel reduction of inner product contributions within work group ///////////////
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}

template<typename StringT>
void generate_hyb_matrix_pipelined_cg_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void cg_hyb_prod( \n");
  source.append("  const __global int* ell_coords, \n");
  source.append("  const __global "); source.append(numeric_string); source.append("* ell_elements, \n");
  source.append("  const __global uint* csr_rows, \n");
  source.append("  const __global uint* csr_cols, \n");
  source.append("  const __global "); source.append(numeric_string); source.append("* csr_elements, \n");
  source.append("  unsigned int internal_row_num, \n");
  source.append("  unsigned int items_per_row, \n");
  source.append("  unsigned int aligned_items_per_row, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  uint glb_id = get_global_id(0); \n");
  source.append("  uint glb_sz = get_global_size(0); \n");

  source.append("  for (uint row = glb_id; row < size; row += glb_sz) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint offset = row; \n");
  source.append("    for (uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num) { \n");
  source.append("      "); source.append(numeric_string); source.append(" val = ell_elements[offset]; \n");
  source.append("      sum += (val != 0) ? (p[ell_coords[offset]] * val) : 0; \n");
  source.append("    } \n");

  source.append("    uint col_begin = csr_rows[row]; \n");
  source.append("    uint col_end   = csr_rows[row + 1]; \n");

  source.append("    for (uint item_id = col_begin; item_id < col_end; item_id++) {  \n");
  source.append("      sum += (p[csr_cols[item_id]] * csr_elements[item_id]); \n");
  source.append("    } \n");

  source.append("    Ap[row] = sum; \n");
  source.append("    inner_prod_ApAp += sum * sum; \n");
  source.append("    inner_prod_pAp  += p[row] * sum; \n");
  source.append("  }  \n");

  //////////// parallel reduction of inner product contributions within work group ///////////////
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}


//////////////////////////////////////////////////////


template<typename StringT>
void generate_pipelined_bicgstab_update_s(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_update_s( \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * s, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * r, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * Ap, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int chunk_size, \n");
  source.append("  unsigned int chunk_offset, \n");
  source.append("  unsigned int size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_Ap_in_r0) \n");
  source.append("{ \n");

  source.append("  "); source.append(numeric_string); source.append(" alpha = 0; \n");

  // parallel reduction in work group to compute <r, r0> / <Ap, r0>
  source.append("  shared_array[get_local_id(0)]  = inner_prod_buffer[get_local_id(0)]; \n");
  source.append("  shared_array_Ap_in_r0[get_local_id(0)] = inner_prod_buffer[get_local_id(0) + 3 * chunk_size]; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array[get_local_id(0)]  += shared_array[get_local_id(0) + stride];  \n");
  source.append("      shared_array_Ap_in_r0[get_local_id(0)] += shared_array_Ap_in_r0[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // compute alpha from reduced values:
  source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("  alpha = shared_array[0] / shared_array_Ap_in_r0[0]; ");

  source.append("  "); source.append(numeric_string); source.append(" inner_prod_contrib = 0; \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" value_s = s[i]; \n");
  source.append("     \n");
  source.append("    value_s = r[i] - alpha * Ap[i]; \n");
  source.append("    inner_prod_contrib += value_s * value_s; \n");
  source.append("     \n");
  source.append("    s[i] = value_s; \n");
  source.append("  }  \n");
  source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

  // parallel reduction in work group
  source.append("  shared_array[get_local_id(0)] = inner_prod_contrib; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride)  \n");
  source.append("      shared_array[get_local_id(0)] += shared_array[get_local_id(0) + stride];  \n");
  source.append("  } ");

  // write results to result array
  source.append(" if (get_local_id(0) == 0) \n ");
  source.append("   inner_prod_buffer[get_group_id(0) + chunk_offset] = shared_array[0]; ");

  source.append("} \n");

}



template<typename StringT>
void generate_pipelined_bicgstab_vector_update(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_vector_update( \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * result, \n");
  source.append("  "); source.append(numeric_string); source.append(" alpha, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  "); source.append(numeric_string); source.append(" omega, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * s, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * residual, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * As, \n");
  source.append("  "); source.append(numeric_string); source.append(" beta, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * Ap, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" const * r0star, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r_r0star = 0; \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" value_result = result[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_p = p[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_s = s[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_residual = residual[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_As = As[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_Ap = Ap[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" value_r0star = r0star[i]; \n");
  source.append("     \n");
  source.append("    value_result += alpha * value_p + omega * value_s; \n");
  source.append("    value_residual  = value_s - omega * value_As; \n");
  source.append("    value_p         = value_residual + beta * (value_p - omega * value_Ap); \n");
  source.append("     \n");
  source.append("    result[i]   = value_result; \n");
  source.append("    residual[i] = value_residual; \n");
  source.append("    p[i]        = value_p; \n");
  source.append("    inner_prod_r_r0star += value_residual * value_r0star; \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array[get_local_id(0)] = inner_prod_r_r0star; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride)  \n");
  source.append("      shared_array[get_local_id(0)] += shared_array[get_local_id(0) + stride];  \n");
  source.append("  } ");

  // write results to result array
  source.append(" if (get_local_id(0) == 0) \n ");
  source.append("   inner_prod_buffer[get_group_id(0)] = shared_array[0]; ");

  source.append("} \n");
}

template<typename StringT>
void generate_compressed_matrix_pipelined_bicgstab_blocked_prod(StringT & source, std::string const & numeric_string, unsigned int subwarp_size)
{
  std::stringstream ss;
  ss << subwarp_size;

  source.append("__kernel void bicgstab_csr_blocked_prod( \n");
  source.append("    __global const unsigned int * row_indices, \n");
  source.append("    __global const unsigned int * column_indices, \n");
  source.append("    __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("    __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("    __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * r0star, \n");
  source.append("    unsigned int size, \n");
  source.append("    __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  unsigned int buffer_offset, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_r0Ap) \n");
  source.append("{ \n");
  source.append("  __local "); source.append(numeric_string); source.append(" shared_elements[256]; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r0Ap = 0; \n");

  source.append("  const unsigned int id_in_row = get_local_id(0) % " + ss.str() + "; \n");
  source.append("  const unsigned int block_increment = get_local_size(0) * ((size - 1) / (get_global_size(0)) + 1); \n");
  source.append("  const unsigned int block_start = get_group_id(0) * block_increment; \n");
  source.append("  const unsigned int block_stop  = min(block_start + block_increment, size); \n");

  source.append("  for (unsigned int row  = block_start + get_local_id(0) / " + ss.str() + "; \n");
  source.append("                    row  < block_stop; \n");
  source.append("                    row += get_local_size(0) / " + ss.str() + ") \n");
  source.append("  { \n");
  source.append("    "); source.append(numeric_string); source.append(" dot_prod = 0; \n");
  source.append("    unsigned int row_end = row_indices[row+1]; \n");
  source.append("    for (unsigned int i = row_indices[row] + id_in_row; i < row_end; i += " + ss.str() + ") \n");
  source.append("      dot_prod += elements[i] * p[column_indices[i]]; \n");

  source.append("    shared_elements[get_local_id(0)] = dot_prod; \n");
  source.append("    #pragma unroll \n");
  source.append("    for (unsigned int k = 1; k < " + ss.str() + "; k *= 2) \n");
  source.append("      shared_elements[get_local_id(0)] += shared_elements[get_local_id(0) ^ k]; \n");

  source.append("    if (id_in_row == 0) { \n");
  source.append("      Ap[row] = shared_elements[get_local_id(0)]; \n");
  source.append("      inner_prod_ApAp += shared_elements[get_local_id(0)] * shared_elements[get_local_id(0)]; \n");
  source.append("      inner_prod_pAp  +=                           p[row] * shared_elements[get_local_id(0)]; \n");
  source.append("      inner_prod_r0Ap +=                      r0star[row] * shared_elements[get_local_id(0)]; \n");
  source.append("    } \n");
  source.append("  } \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  shared_array_r0Ap[get_local_id(0)] = inner_prod_r0Ap; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_r0Ap[get_local_id(0)]  += shared_array_r0Ap[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("    inner_prod_buffer[buffer_offset + get_group_id(0)] = shared_array_r0Ap[0]; \n");
  source.append("  } \n");

  source.append("} \n");
}

template<typename StringT>
void generate_compressed_matrix_pipelined_bicgstab_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_csr_prod( \n");
  source.append("  __global const unsigned int * row_indices, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const unsigned int * row_blocks, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  unsigned int num_blocks, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * r0star, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  unsigned int buffer_offset, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_r0Ap) \n");
  source.append("{ \n");
  source.append("  __local "); source.append(numeric_string); source.append(" shared_elements[1024]; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r0Ap = 0; \n");

  source.append("  for (unsigned int block_id = get_group_id(0); block_id < num_blocks; block_id += get_num_groups(0)) { \n");
  source.append("    unsigned int row_start = row_blocks[block_id]; \n");
  source.append("    unsigned int row_stop  = row_blocks[block_id + 1]; \n");
  source.append("    unsigned int rows_to_process = row_stop - row_start; \n");
  source.append("    unsigned int element_start = row_indices[row_start]; \n");
  source.append("    unsigned int element_stop = row_indices[row_stop]; \n");

  source.append("    if (rows_to_process > 1) { \n"); // CSR stream
      // load to shared buffer:
  source.append("      for (unsigned int i = element_start + get_local_id(0); i < element_stop; i += get_local_size(0)) \n");
  source.append("        shared_elements[i - element_start] = elements[i] * p[column_indices[i]]; \n");

  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");

      // use one thread per row to sum:
  source.append("      for (unsigned int row = row_start + get_local_id(0); row < row_stop; row += get_local_size(0)) { \n");
  source.append("        "); source.append(numeric_string); source.append(" dot_prod = 0; \n");
  source.append("        unsigned int thread_row_start = row_indices[row]     - element_start; \n");
  source.append("        unsigned int thread_row_stop  = row_indices[row + 1] - element_start; \n");
  source.append("        for (unsigned int i = thread_row_start; i < thread_row_stop; ++i) \n");
  source.append("          dot_prod += shared_elements[i]; \n");
  source.append("        Ap[row] = dot_prod; \n");
  source.append("        inner_prod_ApAp += dot_prod * dot_prod; \n");
  source.append("        inner_prod_pAp  +=   p[row] * dot_prod; \n");
  source.append("        inner_prod_r0Ap  += r0star[row] * dot_prod; \n");
  source.append("      } \n");
  source.append("    } \n");

  source.append("    else  \n"); // CSR vector for a single row
  source.append("    { \n");
      // load and sum to shared buffer:
  source.append("      shared_elements[get_local_id(0)] = 0; \n");
  source.append("      for (unsigned int i = element_start + get_local_id(0); i < element_stop; i += get_local_size(0)) \n");
  source.append("        shared_elements[get_local_id(0)] += elements[i] * p[column_indices[i]]; \n");

      // reduction to obtain final result
  source.append("      for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) { \n");
  source.append("        barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("        if (get_local_id(0) < stride) \n");
  source.append("          shared_elements[get_local_id(0)] += shared_elements[get_local_id(0) + stride]; \n");
  source.append("      } \n");

  source.append("      if (get_local_id(0) == 0) { \n");
  source.append("        Ap[row_start] = shared_elements[0]; \n");
  source.append("        inner_prod_ApAp += shared_elements[0] * shared_elements[0]; \n");
  source.append("        inner_prod_pAp  +=       p[row_start] * shared_elements[0]; \n");
  source.append("        inner_prod_r0Ap +=  r0star[row_start] * shared_elements[0]; \n");
  source.append("      } \n");
  source.append("    } \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("  } \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  shared_array_r0Ap[get_local_id(0)] = inner_prod_r0Ap; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_r0Ap[get_local_id(0)]  += shared_array_r0Ap[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("    inner_prod_buffer[buffer_offset + get_group_id(0)] = shared_array_r0Ap[0]; \n");
  source.append("  } \n");

  source.append("} \n \n");

}

template<typename StringT>
void generate_coordinate_matrix_pipelined_bicgstab_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_coo_prod( \n");
  source.append("  __global const uint2 * coords,  \n");//(row_index, column_index)
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const uint  * group_boundaries, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * r0star, \n");
  source.append("  unsigned int size, \n");
  source.append("  __local unsigned int * shared_rows, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * inter_results, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  unsigned int buffer_offset, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_r0Ap) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r0Ap = 0; \n");

  ///////////// Sparse matrix-vector multiplication part /////////////
  source.append("  uint2 tmp; \n");
  source.append("  "); source.append(numeric_string); source.append(" val; \n");
  source.append("  uint group_start = group_boundaries[get_group_id(0)]; \n");
  source.append("  uint group_end   = group_boundaries[get_group_id(0) + 1]; \n");
  source.append("  uint k_end = (group_end > group_start) ? 1 + (group_end - group_start - 1) / get_local_size(0) : 0; \n");   // -1 in order to have correct behavior if group_end - group_start == j * get_local_size(0)

  source.append("  uint local_index = 0; \n");

  source.append("  for (uint k = 0; k < k_end; ++k) { \n");
  source.append("    local_index = group_start + k * get_local_size(0) + get_local_id(0); \n");

  source.append("    tmp = (local_index < group_end) ? coords[local_index] : (uint2) 0; \n");
  source.append("    val = (local_index < group_end) ? elements[local_index] * p[tmp.y] : 0; \n");

  //check for carry from previous loop run:
  source.append("    if (get_local_id(0) == 0 && k > 0) { \n");
  source.append("      if (tmp.x == shared_rows[get_local_size(0)-1]) \n");
  source.append("        val += inter_results[get_local_size(0)-1]; \n");
  source.append("      else {\n");
  source.append("        "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_size(0)-1]; \n");
  source.append("        Ap[shared_rows[get_local_size(0)-1]] = Ap_entry; \n");
  source.append("        inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("        inner_prod_pAp  += p[shared_rows[get_local_size(0)-1]] * Ap_entry; \n");
  source.append("        inner_prod_r0Ap  += r0star[shared_rows[get_local_size(0)-1]] * Ap_entry; \n");
  source.append("      } \n");
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
  source.append("      "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_id(0)]; \n");
  source.append("      Ap[tmp.x] = Ap_entry; \n");
  source.append("      inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("      inner_prod_pAp  += p[tmp.x] * Ap_entry; \n");
  source.append("      inner_prod_r0Ap += r0star[tmp.x] * Ap_entry; \n");
  source.append("    } \n");

  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("  }  \n"); //for k

  source.append("  if (local_index + 1 == group_end) {\n");  //write results of last active entry (this may not necessarily be the case already)
  source.append("    "); source.append(numeric_string); source.append(" Ap_entry = inter_results[get_local_id(0)]; \n");
  source.append("    Ap[tmp.x] = Ap_entry; \n");
  source.append("    inner_prod_ApAp += Ap_entry * Ap_entry; \n");
  source.append("    inner_prod_pAp  += p[tmp.x] * Ap_entry; \n");
  source.append("    inner_prod_r0Ap += r0star[tmp.x] * Ap_entry; \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  shared_array_r0Ap[get_local_id(0)] = inner_prod_r0Ap; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_r0Ap[get_local_id(0)]  += shared_array_r0Ap[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("    inner_prod_buffer[buffer_offset + get_group_id(0)] = shared_array_r0Ap[0]; \n");
  source.append("  } \n");

  source.append("} \n \n");

}


template<typename StringT>
void generate_ell_matrix_pipelined_bicgstab_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_ell_prod( \n");
  source.append("  __global const unsigned int * coords, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  unsigned int internal_row_num, \n");
  source.append("  unsigned int items_per_row, \n");
  source.append("  unsigned int aligned_items_per_row, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * r0star, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  unsigned int buffer_offset, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_r0Ap) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r0Ap = 0; \n");
  source.append("  uint glb_id = get_global_id(0); \n");
  source.append("  uint glb_sz = get_global_size(0); \n");

  source.append("  for (uint row = glb_id; row < size; row += glb_sz) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint offset = row; \n");
  source.append("    for (uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num) { \n");
  source.append("      "); source.append(numeric_string); source.append(" val = elements[offset]; \n");
  source.append("      sum += (val != 0) ? p[coords[offset]] * val : ("); source.append(numeric_string); source.append(")0; \n");
  source.append("    } \n");

  source.append("    Ap[row] = sum; \n");
  source.append("    inner_prod_ApAp += sum * sum; \n");
  source.append("    inner_prod_pAp  += p[row] * sum; \n");
  source.append("    inner_prod_r0Ap += r0star[row] * sum; \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  shared_array_r0Ap[get_local_id(0)] = inner_prod_r0Ap; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_r0Ap[get_local_id(0)] += shared_array_r0Ap[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("    inner_prod_buffer[buffer_offset + get_group_id(0)] = shared_array_r0Ap[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}

template<typename StringT>
void generate_sliced_ell_matrix_pipelined_bicgstab_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_sliced_ell_prod( \n");
  source.append("  __global const unsigned int * columns_per_block, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const unsigned int * block_start, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * r0star, \n");
  source.append("  unsigned int size, \n");
  source.append("  unsigned int block_size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  unsigned int buffer_offset, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_r0Ap) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r0Ap = 0; \n");
  source.append("  uint blocks_per_workgroup = get_local_size(0) / block_size; \n");
  source.append("  uint id_in_block = get_local_id(0) % block_size; \n");
  source.append("  uint num_blocks  = (size - 1) / block_size + 1; \n");
  source.append("  uint global_warp_count  = blocks_per_workgroup * get_num_groups(0); \n");
  source.append("  uint global_warp_id     = blocks_per_workgroup * get_group_id(0) + get_local_id(0) / block_size; \n");

  source.append("  for (uint block_idx = global_warp_id; block_idx < num_blocks; block_idx += global_warp_count) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint row    = block_idx * block_size + id_in_block; \n");
  source.append("    uint offset = block_start[block_idx]; \n");
  source.append("    uint num_columns = columns_per_block[block_idx]; \n");
  source.append("    for (uint item_id = 0; item_id < num_columns; item_id++) { \n");
  source.append("      uint index = offset + item_id * block_size + id_in_block; \n");
  source.append("      "); source.append(numeric_string); source.append(" val = elements[index]; \n");
  source.append("      sum += (val != 0) ? (p[column_indices[index]] * val) : 0; \n");
  source.append("    } \n");

  source.append("    if (row < size) {\n");
  source.append("      Ap[row] = sum; \n");
  source.append("      inner_prod_ApAp += sum * sum; \n");
  source.append("      inner_prod_pAp  += p[row] * sum; \n");
  source.append("      inner_prod_r0Ap += r0star[row] * sum; \n");
  source.append("    }  \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  shared_array_r0Ap[get_local_id(0)] = inner_prod_r0Ap; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_r0Ap[get_local_id(0)] += shared_array_r0Ap[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("    inner_prod_buffer[buffer_offset + get_group_id(0)] = shared_array_r0Ap[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}

template<typename StringT>
void generate_hyb_matrix_pipelined_bicgstab_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bicgstab_hyb_prod( \n");
  source.append("  const __global int* ell_coords, \n");
  source.append("  const __global "); source.append(numeric_string); source.append("* ell_elements, \n");
  source.append("  const __global uint* csr_rows, \n");
  source.append("  const __global uint* csr_cols, \n");
  source.append("  const __global "); source.append(numeric_string); source.append("* csr_elements, \n");
  source.append("  unsigned int internal_row_num, \n");
  source.append("  unsigned int items_per_row, \n");
  source.append("  unsigned int aligned_items_per_row, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * r0star, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  unsigned int buffer_offset, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("   __local "); source.append(numeric_string); source.append(" * shared_array_r0Ap) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_ApAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_pAp = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" inner_prod_r0Ap = 0; \n");
  source.append("  uint glb_id = get_global_id(0); \n");
  source.append("  uint glb_sz = get_global_size(0); \n");

  source.append("  for (uint row = glb_id; row < size; row += glb_sz) { \n");
  source.append("    "); source.append(numeric_string); source.append(" sum = 0; \n");

  source.append("    uint offset = row; \n");
  source.append("    for (uint item_id = 0; item_id < items_per_row; item_id++, offset += internal_row_num) { \n");
  source.append("      "); source.append(numeric_string); source.append(" val = ell_elements[offset]; \n");
  source.append("      sum += (val != 0) ? (p[ell_coords[offset]] * val) : 0; \n");
  source.append("    } \n");

  source.append("    uint col_begin = csr_rows[row]; \n");
  source.append("    uint col_end   = csr_rows[row + 1]; \n");

  source.append("    for (uint item_id = col_begin; item_id < col_end; item_id++) {  \n");
  source.append("      sum += (p[csr_cols[item_id]] * csr_elements[item_id]); \n");
  source.append("    } \n");

  source.append("    Ap[row] = sum; \n");
  source.append("    inner_prod_ApAp += sum * sum; \n");
  source.append("    inner_prod_pAp  += p[row] * sum; \n");
  source.append("    inner_prod_r0Ap += r0star[row] * sum; \n");
  source.append("  }  \n");

  // parallel reduction in work group
  source.append("  shared_array_ApAp[get_local_id(0)] = inner_prod_ApAp; \n");
  source.append("  shared_array_pAp[get_local_id(0)]  = inner_prod_pAp; \n");
  source.append("  shared_array_r0Ap[get_local_id(0)] = inner_prod_r0Ap; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  source.append("      shared_array_ApAp[get_local_id(0)] += shared_array_ApAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_pAp[get_local_id(0)]  += shared_array_pAp[get_local_id(0) + stride];  \n");
  source.append("      shared_array_r0Ap[get_local_id(0)]  += shared_array_r0Ap[get_local_id(0) + stride];  \n");
  source.append("    } ");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) { \n ");
  source.append("    inner_prod_buffer[  buffer_size + get_group_id(0)] = shared_array_ApAp[0]; \n");
  source.append("    inner_prod_buffer[2*buffer_size + get_group_id(0)] = shared_array_pAp[0]; \n");
  source.append("    inner_prod_buffer[buffer_offset + get_group_id(0)] = shared_array_r0Ap[0]; \n");
  source.append("  } \n");
  source.append("} \n \n");
}

//////////////////////////////


template <typename StringType>
void generate_pipelined_gmres_gram_schmidt_stage1(StringType & source, std::string const & numeric_string, bool is_nvidia)
{
  source.append("__kernel void gmres_gram_schmidt_1( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" const * krylov_basis, \n");
  source.append("          unsigned int size, \n");
  source.append("          unsigned int internal_size, \n");
  source.append("          unsigned int k, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * vi_in_vk_buffer, \n");
  source.append("          unsigned int chunk_size) \n");
  source.append("{ \n");

  source.append("  __local "); source.append(numeric_string); source.append(" shared_array[7*128]; \n");
  if (!is_nvidia)  // use of thread-local variables entails a 2x performance drop on NVIDIA GPUs, but is faster an AMD
  {
    source.append("  "); source.append(numeric_string); source.append(" vi_in_vk[7]; \n");
  }
  source.append("  "); source.append(numeric_string); source.append(" value_vk = 0; \n");

  source.append("  unsigned int k_base = 0;   \n");
  source.append("  while (k_base < k) {   \n");
  source.append("    unsigned int vecs_in_iteration = (k - k_base > 7) ? 7 : (k - k_base);   \n");

  if (is_nvidia)
  {
    source.append("    for (uint j=0; j<vecs_in_iteration; ++j) \n");
    source.append("      shared_array[get_local_id(0) + j*chunk_size] = 0; \n");
  }
  else
  {
    source.append("    vi_in_vk[0] = 0;\n");
    source.append("    vi_in_vk[1] = 0;\n");
    source.append("    vi_in_vk[2] = 0;\n");
    source.append("    vi_in_vk[3] = 0;\n");
    source.append("    vi_in_vk[4] = 0;\n");
    source.append("    vi_in_vk[5] = 0;\n");
    source.append("    vi_in_vk[6] = 0;\n");
  }
  source.append("    for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("      value_vk = krylov_basis[i + k * internal_size]; \n");
  source.append("       \n");
  source.append("      for (unsigned int j=0; j<vecs_in_iteration; ++j) \n");
  if (is_nvidia)
    source.append("        shared_array[get_local_id(0) + j*chunk_size] += value_vk * krylov_basis[i + (k_base + j) * internal_size]; \n");
  else
    source.append("        vi_in_vk[j] += value_vk * krylov_basis[i + (k_base + j) * internal_size]; \n");
  source.append("    }  \n");

  // parallel reduction in work group
  if (!is_nvidia)
  {
    source.append("    for (uint j=0; j<vecs_in_iteration; ++j) \n");
    source.append("      shared_array[get_local_id(0) + j*chunk_size] = vi_in_vk[j]; \n");
  }
  source.append("    for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("    { \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("      if (get_local_id(0) < stride) { \n");
  source.append("        for (uint j=0; j<vecs_in_iteration; ++j) \n");
  source.append("          shared_array[get_local_id(0) + j*chunk_size] += shared_array[get_local_id(0) + j*chunk_size + stride];  \n");
  source.append("      } ");
  source.append("    } ");

  // write results to result array
  source.append("    if (get_local_id(0) == 0) \n ");
  source.append("      for (unsigned int j=0; j<vecs_in_iteration; ++j) \n");
  source.append("        vi_in_vk_buffer[get_group_id(0) + (k_base + j) * chunk_size] = shared_array[j*chunk_size]; ");

  source.append("    k_base += vecs_in_iteration;   \n");
  source.append("  }  \n");

  source.append("} \n");

}

template <typename StringType>
void generate_pipelined_gmres_gram_schmidt_stage2(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void gmres_gram_schmidt_2( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * krylov_basis, \n");
  source.append("          unsigned int size, \n");
  source.append("          unsigned int internal_size, \n");
  source.append("          unsigned int k, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" const * vi_in_vk_buffer, \n");
  source.append("          unsigned int chunk_size, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * R_buffer, \n");
  source.append("          unsigned int krylov_dim, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("         __local "); source.append(numeric_string); source.append(" * shared_array) \n");
  source.append("{ \n");

  source.append("  "); source.append(numeric_string); source.append(" vk_dot_vk = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" value_vk = 0; \n");

  source.append("  unsigned int k_base = 0;   \n");
  source.append("  while (k_base < k) {   \n");
  source.append("    unsigned int vecs_in_iteration = (k - k_base > 7) ? 7 : (k - k_base);   \n");

  // parallel reduction in work group for <v_i, v_k>
  source.append("    for (uint j=0; j<vecs_in_iteration; ++j) \n");
  source.append("      shared_array[get_local_id(0) + j*chunk_size] = vi_in_vk_buffer[get_local_id(0) + (k_base + j) * chunk_size]; \n");
  source.append("    for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("    { \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("      if (get_local_id(0) < stride) { \n");
  source.append("        for (uint j=0; j<vecs_in_iteration; ++j) \n");
  source.append("          shared_array[get_local_id(0) + j*chunk_size] += shared_array[get_local_id(0) + j*chunk_size + stride];  \n");
  source.append("      } ");
  source.append("    } ");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");

  // v_k -= <v_i, v_k> v_i:
  source.append("    for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("      value_vk = krylov_basis[i + k * internal_size]; \n");
  source.append("       \n");
  source.append("      for (unsigned int j=0; j<vecs_in_iteration; ++j) \n");
  source.append("        value_vk -= shared_array[j*chunk_size] * krylov_basis[i + (k_base + j) * internal_size]; \n");
  source.append("      vk_dot_vk += (k_base + vecs_in_iteration == k) ? (value_vk * value_vk) : 0;  \n");
  source.append("      krylov_basis[i + k * internal_size] = value_vk;  \n");
  source.append("    }  \n");

  // write to R: (to avoid thread divergence, all threads write the same value)
  source.append("    if (get_group_id(0) == 0) \n");
  source.append("      for (unsigned int j=0; j<vecs_in_iteration; ++j) \n");
  source.append("        R_buffer[(k_base + j) + k*krylov_dim] = shared_array[j*chunk_size]; ");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");

  source.append("    k_base += vecs_in_iteration;   \n");
  source.append("  }  \n");

  // parallel reduction in work group for <v_k, v_k>
  source.append("  shared_array[get_local_id(0)] = vk_dot_vk; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) \n");
  source.append("      shared_array[get_local_id(0)] += shared_array[get_local_id(0) + stride];  \n");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) \n ");
  source.append("    inner_prod_buffer[chunk_size+get_group_id(0)] = shared_array[0]; ");

  source.append("} \n");
}

template <typename StringType>
void generate_pipelined_gmres_normalize_vk(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void gmres_normalize_vk( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * vk, \n");
  source.append("          unsigned int vk_offset, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" const * residual, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * R_buffer, \n");
  source.append("          unsigned int R_offset, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" const * inner_prod_buffer, \n");
  source.append("          unsigned int chunk_size, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * r_dot_vk_buffer, \n");
  source.append("          unsigned int chunk_offset, \n");
  source.append("          unsigned int size, \n");
  source.append("         __local "); source.append(numeric_string); source.append(" * shared_array) \n");
  source.append("{ \n");

  source.append("  "); source.append(numeric_string); source.append(" norm_vk = 0; \n");

  // parallel reduction in work group to compute <vk, vk>
  source.append("  shared_array[get_local_id(0)] = inner_prod_buffer[get_local_id(0) + chunk_size]; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) \n");
  source.append("      shared_array[get_local_id(0)]  += shared_array[get_local_id(0) + stride];  \n");
  source.append("  } ");

  // compute alpha from reduced values:
  source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("  norm_vk = sqrt(shared_array[0]); \n");

  source.append("  "); source.append(numeric_string); source.append(" inner_prod_contrib = 0; \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" value_vk = vk[i + vk_offset] / norm_vk; \n");
  source.append("     \n");
  source.append("    inner_prod_contrib += residual[i] * value_vk; \n");
  source.append("     \n");
  source.append("    vk[i + vk_offset] = value_vk; \n");
  source.append("  }  \n");
  source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

  // parallel reduction in work group
  source.append("  shared_array[get_local_id(0)] = inner_prod_contrib; \n");
  source.append("  for (uint stride=get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride)  \n");
  source.append("      shared_array[get_local_id(0)] += shared_array[get_local_id(0) + stride];  \n");
  source.append("  } ");

  // write results to result array
  source.append("  if (get_local_id(0) == 0) \n ");
  source.append("    r_dot_vk_buffer[get_group_id(0) + chunk_offset] = shared_array[0]; ");
  source.append("  if (get_global_id(0) == 0) \n ");
  source.append("    R_buffer[R_offset] = norm_vk; \n");

  source.append("} \n");

}

template <typename StringType>
void generate_pipelined_gmres_update_result(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void gmres_update_result( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * result, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" const * residual, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" const * krylov_basis, \n");
  source.append("          unsigned int size, \n");
  source.append("          unsigned int internal_size, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" const * coefficients, \n");
  source.append("          unsigned int k) \n");
  source.append("{ \n");

  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" value_result = result[i] + coefficients[0] * residual[i]; \n");
  source.append("     \n");
  source.append("    for (unsigned int j = 1; j < k; ++j) \n");
  source.append("      value_result += coefficients[j] * krylov_basis[i + (j-1)*internal_size]; \n");
  source.append("     \n");
  source.append("    result[i] = value_result; \n");
  source.append("  }  \n");

  source.append("} \n");
}


template <typename StringType>
void generate_compressed_matrix_pipelined_gmres_blocked_prod(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void gmres_csr_blocked_prod( \n");
  source.append("  __global const unsigned int * row_indices, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  unsigned int offset_p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int offset_Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  cg_csr_blocked_prod(row_indices, column_indices, elements, p + offset_p, Ap + offset_Ap, size, inner_prod_buffer, buffer_size, shared_array_ApAp, shared_array_pAp); \n");
  source.append("} \n \n");

}

template <typename StringType>
void generate_compressed_matrix_pipelined_gmres_prod(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void gmres_csr_prod( \n");
  source.append("  __global const unsigned int * row_indices, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const unsigned int * row_blocks, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  unsigned int num_blocks, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  unsigned int offset_p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int offset_Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_elements) \n");
  source.append("{ \n");
  source.append("  cg_csr_prod(row_indices, column_indices, row_blocks, elements, num_blocks, p + offset_p, Ap + offset_Ap, size, inner_prod_buffer, buffer_size, shared_array_ApAp, shared_array_pAp, shared_elements); \n");
  source.append("} \n \n");

}

template <typename StringType>
void generate_coordinate_matrix_pipelined_gmres_prod(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void gmres_coo_prod( \n");
  source.append("          __global const uint2 * coords,  \n");//(row_index, column_index)
  source.append("          __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("          __global const uint  * group_boundaries, \n");
  source.append("          __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("          unsigned int offset_p, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("          unsigned int offset_Ap, \n");
  source.append("          unsigned int size, \n");
  source.append("          __local unsigned int * shared_rows, \n");
  source.append("          __local "); source.append(numeric_string); source.append(" * inter_results, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("          unsigned int buffer_size, \n");
  source.append("          __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("          __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  cg_coo_prod(coords, elements, group_boundaries, p + offset_p, Ap + offset_Ap, size, shared_rows, inter_results, inner_prod_buffer, buffer_size, shared_array_ApAp, shared_array_pAp); \n");
  source.append("} \n \n");

}


template <typename StringType>
void generate_ell_matrix_pipelined_gmres_prod(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void gmres_ell_prod( \n");
  source.append("  __global const unsigned int * coords, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  unsigned int internal_row_num, \n");
  source.append("  unsigned int items_per_row, \n");
  source.append("  unsigned int aligned_items_per_row, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  unsigned int offset_p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int offset_Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  cg_ell_prod(coords, elements, internal_row_num, items_per_row, aligned_items_per_row, p + offset_p, Ap + offset_Ap, size, inner_prod_buffer, buffer_size, shared_array_ApAp, shared_array_pAp); \n");
  source.append("} \n \n");
}

template <typename StringType>
void generate_sliced_ell_matrix_pipelined_gmres_prod(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void gmres_sliced_ell_prod( \n");
  source.append("  __global const unsigned int * columns_per_block, \n");
  source.append("  __global const unsigned int * column_indices, \n");
  source.append("  __global const unsigned int * block_start, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * elements, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  unsigned int offset_p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int offset_Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  unsigned int block_size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  cg_sliced_ell_prod(columns_per_block, column_indices, block_start, elements, p + offset_p, Ap + offset_Ap, size, block_size, inner_prod_buffer, buffer_size, shared_array_ApAp, shared_array_pAp); \n");
  source.append("} \n \n");
}

template <typename StringType>
void generate_hyb_matrix_pipelined_gmres_prod(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void gmres_hyb_prod( \n");
  source.append("  const __global int* ell_coords, \n");
  source.append("  const __global "); source.append(numeric_string); source.append("* ell_elements, \n");
  source.append("  const __global uint* csr_rows, \n");
  source.append("  const __global uint* csr_cols, \n");
  source.append("  const __global "); source.append(numeric_string); source.append("* csr_elements, \n");
  source.append("  unsigned int internal_row_num, \n");
  source.append("  unsigned int items_per_row, \n");
  source.append("  unsigned int aligned_items_per_row, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * p, \n");
  source.append("  unsigned int offset_p, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * Ap, \n");
  source.append("  unsigned int offset_Ap, \n");
  source.append("  unsigned int size, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * inner_prod_buffer, \n");
  source.append("  unsigned int buffer_size, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_ApAp, \n");
  source.append("  __local "); source.append(numeric_string); source.append(" * shared_array_pAp) \n");
  source.append("{ \n");
  source.append("  cg_hyb_prod(ell_coords, ell_elements, csr_rows, csr_cols, csr_elements, internal_row_num, items_per_row, aligned_items_per_row, p + offset_p, Ap + offset_Ap, size, inner_prod_buffer, buffer_size, shared_array_ApAp, shared_array_pAp); \n");
  source.append("} \n \n");
}




//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating specialized OpenCL kernels for fast iterative solvers. */
template<typename NumericT>
struct iterative
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_iterative";
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

      generate_pipelined_cg_vector_update(source, numeric_string);
      if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
        generate_compressed_matrix_pipelined_cg_blocked_prod(source, numeric_string, 16);
      generate_compressed_matrix_pipelined_cg_prod(source, numeric_string);
      generate_coordinate_matrix_pipelined_cg_prod(source, numeric_string);
      generate_ell_matrix_pipelined_cg_prod(source, numeric_string);
      generate_sliced_ell_matrix_pipelined_cg_prod(source, numeric_string);
      generate_hyb_matrix_pipelined_cg_prod(source, numeric_string);

      generate_pipelined_bicgstab_update_s(source, numeric_string);
      generate_pipelined_bicgstab_vector_update(source, numeric_string);
      if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
        generate_compressed_matrix_pipelined_bicgstab_blocked_prod(source, numeric_string, 16);
      generate_compressed_matrix_pipelined_bicgstab_prod(source, numeric_string);
      generate_coordinate_matrix_pipelined_bicgstab_prod(source, numeric_string);
      generate_ell_matrix_pipelined_bicgstab_prod(source, numeric_string);
      generate_sliced_ell_matrix_pipelined_bicgstab_prod(source, numeric_string);
      generate_hyb_matrix_pipelined_bicgstab_prod(source, numeric_string);

      generate_pipelined_gmres_gram_schmidt_stage1(source, numeric_string, ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id); // NVIDIA GPUs require special treatment here
      generate_pipelined_gmres_gram_schmidt_stage2(source, numeric_string);
      generate_pipelined_gmres_normalize_vk(source, numeric_string);
      generate_pipelined_gmres_update_result(source, numeric_string);
      if (ctx.current_device().vendor_id() == viennacl::ocl::nvidia_id)
        generate_compressed_matrix_pipelined_gmres_blocked_prod(source, numeric_string);
      generate_compressed_matrix_pipelined_gmres_prod(source, numeric_string);
      generate_coordinate_matrix_pipelined_gmres_prod(source, numeric_string);
      generate_ell_matrix_pipelined_gmres_prod(source, numeric_string);
      generate_sliced_ell_matrix_pipelined_gmres_prod(source, numeric_string);
      generate_hyb_matrix_pipelined_gmres_prod(source, numeric_string);

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

