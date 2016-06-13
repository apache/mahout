#ifndef VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_HPP

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

#include "viennacl/scheduler/forwards.h"
#include "viennacl/scheduler/io.hpp"
#include "viennacl/scheduler/preset.hpp"

#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"



/** @file viennacl/linalg/opencl/kernels/vector.hpp
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

/** @brief Enumeration for the scalar type in avbv-like operations */
enum avbv_scalar_type
{
  VIENNACL_AVBV_NONE = 0, // vector does not exist/contribute
  VIENNACL_AVBV_CPU,
  VIENNACL_AVBV_GPU
};

/** @brief Configuration struct for generating OpenCL kernels for linear combinations of vectors */
struct avbv_config
{
  avbv_config() : with_stride_and_range(true), a(VIENNACL_AVBV_CPU), b(VIENNACL_AVBV_NONE) {}

  bool with_stride_and_range;
  std::string      assign_op;
  avbv_scalar_type a;
  avbv_scalar_type b;
};

// just returns the for-loop
template <typename StringType>
void generate_avbv_impl2(StringType & source, std::string const & /*numeric_string*/, avbv_config const & cfg, bool mult_alpha, bool mult_beta)
{
  source.append("    for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0)) \n");
  if (cfg.with_stride_and_range)
  {
    source.append("      vec1[i*size1.y+size1.x] "); source.append(cfg.assign_op); source.append(" vec2[i*size2.y+size2.x] ");
    if (mult_alpha)
      source.append("* alpha ");
    else
      source.append("/ alpha ");
    if (cfg.b != VIENNACL_AVBV_NONE)
    {
      source.append("+ vec3[i*size3.y+size3.x] ");
      if (mult_beta)
        source.append("* beta");
      else
        source.append("/ beta");
    }
  }
  else
  {
    source.append("    vec1[i] "); source.append(cfg.assign_op); source.append(" vec2[i] ");
    if (mult_alpha)
      source.append("* alpha ");
    else
      source.append("/ alpha ");
    if (cfg.b != VIENNACL_AVBV_NONE)
    {
      source.append("+ vec3[i] ");
      if (mult_beta)
        source.append("* beta");
      else
        source.append("/ beta");
    }
  }
  source.append("; \n");
}

template <typename StringType>
void generate_avbv_impl(StringType & source, std::string const & numeric_string, avbv_config const & cfg)
{
  source.append("__kernel void av");
  if (cfg.b != VIENNACL_AVBV_NONE)
    source.append("bv");
  if (cfg.assign_op != "=")
    source.append("_v");

  if (cfg.a == VIENNACL_AVBV_CPU)
    source.append("_cpu");
  else if (cfg.a == VIENNACL_AVBV_GPU)
    source.append("_gpu");

  if (cfg.b == VIENNACL_AVBV_CPU)
    source.append("_cpu");
  else if (cfg.b == VIENNACL_AVBV_GPU)
    source.append("_gpu");
  source.append("( \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * vec1, \n");
  source.append("  uint4 size1, \n");
  source.append(" \n");
  if (cfg.a == VIENNACL_AVBV_CPU)
  {
    source.append("  "); source.append(numeric_string); source.append(" fac2, \n");
  }
  else if (cfg.a == VIENNACL_AVBV_GPU)
  {
    source.append("  __global "); source.append(numeric_string); source.append(" * fac2, \n");
  }
  source.append("  unsigned int options2, \n");  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse
  source.append("  __global const "); source.append(numeric_string); source.append(" * vec2, \n");
  source.append("  uint4 size2");

  if (cfg.b != VIENNACL_AVBV_NONE)
  {
    source.append(", \n\n");
    if (cfg.b == VIENNACL_AVBV_CPU)
    {
      source.append("  "); source.append(numeric_string); source.append(" fac3, \n");
    }
    else if (cfg.b == VIENNACL_AVBV_GPU)
    {
      source.append("  __global "); source.append(numeric_string); source.append(" * fac3, \n");
    }
    source.append("  unsigned int options3, \n");  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse
    source.append("  __global const "); source.append(numeric_string); source.append(" * vec3, \n");
    source.append("  uint4 size3 \n");
  }
  source.append(") { \n");

  if (cfg.a == VIENNACL_AVBV_CPU)
  {
    source.append("  "); source.append(numeric_string); source.append(" alpha = fac2; \n");
  }
  else if (cfg.a == VIENNACL_AVBV_GPU)
  {
    source.append("  "); source.append(numeric_string); source.append(" alpha = fac2[0]; \n");
  }
  source.append("  if (options2 & (1 << 0)) \n");
  source.append("    alpha = -alpha; \n");
  source.append(" \n");

  if (cfg.b == VIENNACL_AVBV_CPU)
  {
    source.append("  "); source.append(numeric_string); source.append(" beta = fac3; \n");
  }
  else if (cfg.b == VIENNACL_AVBV_GPU)
  {
    source.append("  "); source.append(numeric_string); source.append(" beta = fac3[0]; \n");
  }
  if (cfg.b != VIENNACL_AVBV_NONE)
  {
    source.append("  if (options3 & (1 << 0)) \n");
    source.append("    beta = -beta; \n");
    source.append(" \n");
  }
  source.append("  if (options2 & (1 << 1)) { \n");
  if (cfg.b != VIENNACL_AVBV_NONE)
  {
    source.append("    if (options3 & (1 << 1)) {\n");
    generate_avbv_impl2(source, numeric_string, cfg, false, false);
    source.append("    } else {\n");
    generate_avbv_impl2(source, numeric_string, cfg, false, true);
    source.append("    } \n");
  }
  else
    generate_avbv_impl2(source, numeric_string, cfg, false, true);
  source.append("  } else { \n");
  if (cfg.b != VIENNACL_AVBV_NONE)
  {
    source.append("    if (options3 & (1 << 1)) {\n");
    generate_avbv_impl2(source, numeric_string, cfg, true, false);
    source.append("    } else {\n");
    generate_avbv_impl2(source, numeric_string, cfg, true, true);
    source.append("    } \n");
  }
  else
    generate_avbv_impl2(source, numeric_string, cfg, true, true);
  source.append("  } \n");
  source.append("} \n");
}

template <typename StringType>
void generate_avbv(StringType & source, std::string const & numeric_string)
{
  avbv_config cfg;
  cfg.assign_op = "=";
  cfg.with_stride_and_range = true;

  // av
  cfg.b = VIENNACL_AVBV_NONE; cfg.a = VIENNACL_AVBV_CPU; generate_avbv_impl(source, numeric_string, cfg);
  cfg.b = VIENNACL_AVBV_NONE; cfg.a = VIENNACL_AVBV_GPU; generate_avbv_impl(source, numeric_string, cfg);

  // avbv
  cfg.a = VIENNACL_AVBV_CPU; cfg.b = VIENNACL_AVBV_CPU; generate_avbv_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_AVBV_CPU; cfg.b = VIENNACL_AVBV_GPU; generate_avbv_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_AVBV_GPU; cfg.b = VIENNACL_AVBV_CPU; generate_avbv_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_AVBV_GPU; cfg.b = VIENNACL_AVBV_GPU; generate_avbv_impl(source, numeric_string, cfg);

  // avbv
  cfg.assign_op = "+=";

  cfg.a = VIENNACL_AVBV_CPU; cfg.b = VIENNACL_AVBV_CPU; generate_avbv_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_AVBV_CPU; cfg.b = VIENNACL_AVBV_GPU; generate_avbv_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_AVBV_GPU; cfg.b = VIENNACL_AVBV_CPU; generate_avbv_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_AVBV_GPU; cfg.b = VIENNACL_AVBV_GPU; generate_avbv_impl(source, numeric_string, cfg);
}

template <typename StringType>
void generate_plane_rotation(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void plane_rotation( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
  source.append("          unsigned int start1, \n");
  source.append("          unsigned int inc1, \n");
  source.append("          unsigned int size1, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * vec2, \n");
  source.append("          unsigned int start2, \n");
  source.append("          unsigned int inc2, \n");
  source.append("          unsigned int size2, \n");
  source.append("          "); source.append(numeric_string); source.append(" alpha, \n");
  source.append("          "); source.append(numeric_string); source.append(" beta) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" tmp1 = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" tmp2 = 0; \n");
  source.append(" \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0)) \n");
  source.append(" { \n");
  source.append("    tmp1 = vec1[i*inc1+start1]; \n");
  source.append("    tmp2 = vec2[i*inc2+start2]; \n");
  source.append(" \n");
  source.append("    vec1[i*inc1+start1] = alpha * tmp1 + beta * tmp2; \n");
  source.append("    vec2[i*inc2+start2] = alpha * tmp2 - beta * tmp1; \n");
  source.append("  } \n");
  source.append(" \n");
  source.append("} \n");
}

template <typename StringType>
void generate_vector_swap(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void swap( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
  source.append("          unsigned int start1, \n");
  source.append("          unsigned int inc1, \n");
  source.append("          unsigned int size1, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * vec2, \n");
  source.append("          unsigned int start2, \n");
  source.append("          unsigned int inc2, \n");
  source.append("          unsigned int size2 \n");
  source.append("          ) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" tmp; \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    tmp = vec2[i*inc2+start2]; \n");
  source.append("    vec2[i*inc2+start2] = vec1[i*inc1+start1]; \n");
  source.append("    vec1[i*inc1+start1] = tmp; \n");
  source.append("  } \n");
  source.append("} \n");
}

template <typename StringType>
void generate_assign_cpu(StringType & source, std::string const & numeric_string)
{
  source.append("__kernel void assign_cpu( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
  source.append("          unsigned int start1, \n");
  source.append("          unsigned int inc1, \n");
  source.append("          unsigned int size1, \n");
  source.append("          unsigned int internal_size1, \n");
  source.append("          "); source.append(numeric_string); source.append(" alpha) \n");
  source.append("{ \n");
  source.append("  for (unsigned int i = get_global_id(0); i < internal_size1; i += get_global_size(0)) \n");
  source.append("    vec1[i*inc1+start1] = (i < size1) ? alpha : 0; \n");
  source.append("} \n");

}

template <typename StringType>
void generate_inner_prod(StringType & source, std::string const & numeric_string, vcl_size_t vector_num)
{
  std::stringstream ss;
  ss << vector_num;
  std::string vector_num_string = ss.str();

  source.append("__kernel void inner_prod"); source.append(vector_num_string); source.append("( \n");
  source.append("          __global const "); source.append(numeric_string); source.append(" * x, \n");
  source.append("          uint4 params_x, \n");
  for (vcl_size_t i=0; i<vector_num; ++i)
  {
    ss.str("");
    ss << i;
    source.append("          __global const "); source.append(numeric_string); source.append(" * y"); source.append(ss.str()); source.append(", \n");
    source.append("          uint4 params_y"); source.append(ss.str()); source.append(", \n");
  }
  source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * group_buffer) \n");
  source.append("{ \n");
  source.append("  unsigned int entries_per_thread = (params_x.z - 1) / get_global_size(0) + 1; \n");
  source.append("  unsigned int vec_start_index = get_group_id(0) * get_local_size(0) * entries_per_thread; \n");
  source.append("  unsigned int vec_stop_index  = min((unsigned int)((get_group_id(0) + 1) * get_local_size(0) * entries_per_thread), params_x.z); \n");

  // compute partial results within group:
  for (vcl_size_t i=0; i<vector_num; ++i)
  {
    ss.str("");
    ss << i;
    source.append("  "); source.append(numeric_string); source.append(" tmp"); source.append(ss.str()); source.append(" = 0; \n");
  }
  source.append("  for (unsigned int i = vec_start_index + get_local_id(0); i < vec_stop_index; i += get_local_size(0)) { \n");
  source.append("    ");  source.append(numeric_string); source.append(" val_x = x[i*params_x.y + params_x.x]; \n");
  for (vcl_size_t i=0; i<vector_num; ++i)
  {
    ss.str("");
    ss << i;
    source.append("    tmp"); source.append(ss.str()); source.append(" += val_x * y"); source.append(ss.str()); source.append("[i * params_y"); source.append(ss.str()); source.append(".y + params_y"); source.append(ss.str()); source.append(".x]; \n");
  }
  source.append("  } \n");
  for (vcl_size_t i=0; i<vector_num; ++i)
  {
    ss.str("");
    ss << i;
    source.append("  tmp_buffer[get_local_id(0) + "); source.append(ss.str()); source.append(" * get_local_size(0)] = tmp"); source.append(ss.str()); source.append("; \n");
  }

  // now run reduction:
  source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) { \n");
  for (vcl_size_t i=0; i<vector_num; ++i)
  {
    ss.str("");
    ss << i;
    source.append("      tmp_buffer[get_local_id(0) + "); source.append(ss.str()); source.append(" * get_local_size(0)] += tmp_buffer[get_local_id(0) + "); source.append(ss.str()); source.append(" * get_local_size(0) + stride]; \n");
  }
  source.append("    } \n");
  source.append("  } \n");
  source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

  source.append("  if (get_local_id(0) == 0) { \n");
  for (vcl_size_t i=0; i<vector_num; ++i)
  {
    ss.str("");
    ss << i;
    source.append("    group_buffer[get_group_id(0) + "); source.append(ss.str()); source.append(" * get_num_groups(0)] = tmp_buffer["); source.append(ss.str()); source.append(" * get_local_size(0)]; \n");
  }
  source.append("  } \n");
  source.append("} \n");

}

template <typename StringType>
void generate_norm(StringType & source, std::string const & numeric_string)
{
  bool is_float_or_double = (numeric_string == "float" || numeric_string == "double");

  source.append(numeric_string); source.append(" impl_norm( \n");
  source.append("          __global const "); source.append(numeric_string); source.append(" * vec, \n");
  source.append("          unsigned int start1, \n");
  source.append("          unsigned int inc1, \n");
  source.append("          unsigned int size1, \n");
  source.append("          unsigned int norm_selector, \n");
  source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" tmp = 0; \n");
  source.append("  if (norm_selector == 1) \n"); //norm_1
  source.append("  { \n");
  source.append("    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0)) \n");
  if (is_float_or_double)
    source.append("      tmp += fabs(vec[i*inc1 + start1]); \n");
  else if (numeric_string[0] == 'u') // abs may not be defined for unsigned types
    source.append("      tmp += vec[i*inc1 + start1]; \n");
  else
    source.append("      tmp += abs(vec[i*inc1 + start1]); \n");
  source.append("  } \n");
  source.append("  else if (norm_selector == 2) \n"); //norm_2
  source.append("  { \n");
  source.append("    "); source.append(numeric_string); source.append(" vec_entry = 0; \n");
  source.append("    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0)) \n");
  source.append("    { \n");
  source.append("      vec_entry = vec[i*inc1 + start1]; \n");
  source.append("      tmp += vec_entry * vec_entry; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("  else if (norm_selector == 0) \n"); //norm_inf
  source.append("  { \n");
  source.append("    for (unsigned int i = get_local_id(0); i < size1; i += get_local_size(0)) \n");
  if (is_float_or_double)
    source.append("      tmp = fmax(fabs(vec[i*inc1 + start1]), tmp); \n");
  else if (numeric_string[0] == 'u') // abs may not be defined for unsigned types
    source.append("      tmp = max(vec[i*inc1 + start1], tmp); \n");
  else
  {
    source.append("      tmp = max(("); source.append(numeric_string); source.append(")abs(vec[i*inc1 + start1]), tmp); \n");
  }
  source.append("  } \n");

  source.append("  tmp_buffer[get_local_id(0)] = tmp; \n");

  source.append("  if (norm_selector > 0) \n"); //norm_1 or norm_2:
  source.append("  { \n");
  source.append("    for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("    { \n");
  source.append("      barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("      if (get_local_id(0) < stride) \n");
  source.append("        tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0)+stride]; \n");
  source.append("    } \n");
  source.append("    return tmp_buffer[0]; \n");
  source.append("  } \n");

  //norm_inf:
  source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) \n");
  if (is_float_or_double)
    source.append("      tmp_buffer[get_local_id(0)] = fmax(tmp_buffer[get_local_id(0)], tmp_buffer[get_local_id(0)+stride]); \n");
  else
    source.append("      tmp_buffer[get_local_id(0)] = max(tmp_buffer[get_local_id(0)], tmp_buffer[get_local_id(0)+stride]); \n");
  source.append("  } \n");

  source.append("  return tmp_buffer[0]; \n");
  source.append("}; \n");

  source.append("__kernel void norm( \n");
  source.append("          __global const "); source.append(numeric_string); source.append(" * vec, \n");
  source.append("          unsigned int start1, \n");
  source.append("          unsigned int inc1, \n");
  source.append("          unsigned int size1, \n");
  source.append("          unsigned int norm_selector, \n");
  source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * group_buffer) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" tmp = impl_norm(vec, \n");
  source.append("                        (        get_group_id(0)  * size1) / get_num_groups(0) * inc1 + start1, \n");
  source.append("                        inc1, \n");
  source.append("                        (   (1 + get_group_id(0)) * size1) / get_num_groups(0) \n");
  source.append("                      - (        get_group_id(0)  * size1) / get_num_groups(0), \n");
  source.append("                        norm_selector, \n");
  source.append("                        tmp_buffer); \n");

  source.append("  if (get_local_id(0) == 0) \n");
  source.append("    group_buffer[get_group_id(0)] = tmp; \n");
  source.append("} \n");

}

template <typename StringType>
void generate_inner_prod_sum(StringType & source, std::string const & numeric_string)
{
  // sums the array 'vec1' and writes to result. Each work group computes the inner product for a subvector of size 'size_per_workgroup'.
  source.append("__kernel void sum_inner_prod( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
  source.append("          unsigned int size_per_workgroup, \n");
  source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * result, \n");
  source.append("          unsigned int start_result, \n");
  source.append("          unsigned int inc_result) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" thread_sum = 0; \n");
  source.append("  for (unsigned int i = get_local_id(0); i<size_per_workgroup; i += get_local_size(0)) \n");
  source.append("    thread_sum += vec1[size_per_workgroup * get_group_id(0) + i]; \n");

  source.append("  tmp_buffer[get_local_id(0)] = thread_sum; \n");

  source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) \n");
  source.append("      tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0) + stride]; \n");
  source.append("  } \n");
  source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

  source.append("  if (get_local_id(0) == 0) \n");
  source.append("    result[start_result + inc_result * get_group_id(0)] = tmp_buffer[0]; \n");
  source.append("} \n");

}

template <typename StringType>
void generate_sum(StringType & source, std::string const & numeric_string)
{
  // sums the array 'vec1' and writes to result. Makes use of a single work-group only.
  source.append("__kernel void sum( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
  source.append("          unsigned int start1, \n");
  source.append("          unsigned int inc1, \n");
  source.append("          unsigned int size1, \n");
  source.append("          unsigned int option,  \n"); //0: use fmax, 1: just sum, 2: sum and return sqrt of sum
  source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * result) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" thread_sum = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" tmp = 0; \n");
  source.append("  for (unsigned int i = get_local_id(0); i<size1; i += get_local_size(0)) \n");
  source.append("  { \n");
  source.append("    if (option > 0) \n");
  source.append("      thread_sum += vec1[i*inc1+start1]; \n");
  source.append("    else \n");
  source.append("    { \n");
  source.append("      tmp = vec1[i*inc1+start1]; \n");
  source.append("      tmp = (tmp < 0) ? -tmp : tmp; \n");
  source.append("      thread_sum = (thread_sum > tmp) ? thread_sum : tmp; \n");
  source.append("    } \n");
  source.append("  } \n");

  source.append("  tmp_buffer[get_local_id(0)] = thread_sum; \n");

  source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) \n");
  source.append("    { \n");
  source.append("      if (option > 0) \n");
  source.append("        tmp_buffer[get_local_id(0)] += tmp_buffer[get_local_id(0) + stride]; \n");
  source.append("      else \n");
  source.append("        tmp_buffer[get_local_id(0)] = (tmp_buffer[get_local_id(0)] > tmp_buffer[get_local_id(0) + stride]) ? tmp_buffer[get_local_id(0)] : tmp_buffer[get_local_id(0) + stride]; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

  source.append("  if (get_global_id(0) == 0) \n");
  source.append("  { \n");
  if (numeric_string == "float" || numeric_string == "double")
  {
    source.append("    if (option == 2) \n");
    source.append("      *result = sqrt(tmp_buffer[0]); \n");
    source.append("    else \n");
  }
  source.append("      *result = tmp_buffer[0]; \n");
  source.append("  } \n");
  source.append("} \n");

}

template <typename StringType>
void generate_index_norm_inf(StringType & source, std::string const & numeric_string)
{
  //index_norm_inf:
  source.append("unsigned int index_norm_inf_impl( \n");
  source.append("          __global const "); source.append(numeric_string); source.append(" * vec, \n");
  source.append("          unsigned int start1, \n");
  source.append("          unsigned int inc1, \n");
  source.append("          unsigned int size1, \n");
  source.append("          __local "); source.append(numeric_string); source.append(" * entry_buffer, \n");
  source.append("          __local unsigned int * index_buffer) \n");
  source.append("{ \n");
  //step 1: fill buffer:
  source.append("  "); source.append(numeric_string); source.append(" cur_max = 0; \n");
  source.append("  "); source.append(numeric_string); source.append(" tmp; \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0)) \n");
  source.append("  { \n");
  if (numeric_string == "float" || numeric_string == "double")
    source.append("    tmp = fabs(vec[i*inc1+start1]); \n");
  else if (numeric_string[0] == 'u') // abs may not be defined for unsigned types
    source.append("    tmp = vec[i*inc1+start1]; \n");
  else
    source.append("    tmp = abs(vec[i*inc1+start1]); \n");
  source.append("    if (cur_max < tmp) \n");
  source.append("    { \n");
  source.append("      entry_buffer[get_global_id(0)] = tmp; \n");
  source.append("      index_buffer[get_global_id(0)] = i; \n");
  source.append("      cur_max = tmp; \n");
  source.append("    } \n");
  source.append("  } \n");

  //step 2: parallel reduction:
  source.append("  for (unsigned int stride = get_global_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_global_id(0) < stride) \n");
  source.append("   { \n");
  //find the first occurring index
  source.append("      if (entry_buffer[get_global_id(0)] < entry_buffer[get_global_id(0)+stride]) \n");
  source.append("      { \n");
  source.append("        index_buffer[get_global_id(0)] = index_buffer[get_global_id(0)+stride]; \n");
  source.append("        entry_buffer[get_global_id(0)] = entry_buffer[get_global_id(0)+stride]; \n");
  source.append("      } \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append(" \n");
  source.append("  return index_buffer[0]; \n");
  source.append("} \n");

  source.append("__kernel void index_norm_inf( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * vec, \n");
  source.append("          unsigned int start1, \n");
  source.append("          unsigned int inc1, \n");
  source.append("          unsigned int size1, \n");
  source.append("          __local "); source.append(numeric_string); source.append(" * entry_buffer, \n");
  source.append("          __local unsigned int * index_buffer, \n");
  source.append("          __global unsigned int * result) \n");
  source.append("{ \n");
  source.append("  entry_buffer[get_global_id(0)] = 0; \n");
  source.append("  index_buffer[get_global_id(0)] = 0; \n");
  source.append("  unsigned int tmp = index_norm_inf_impl(vec, start1, inc1, size1, entry_buffer, index_buffer); \n");
  source.append("  if (get_global_id(0) == 0) *result = tmp; \n");
  source.append("} \n");

}

template <typename StringType>
void generate_maxmin(StringType & source, std::string const & numeric_string, bool is_max)
{
  // sums the array 'vec1' and writes to result. Makes use of a single work-group only.
  if (is_max)
    source.append("__kernel void max_kernel( \n");
  else
    source.append("__kernel void min_kernel( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * vec1, \n");
  source.append("          unsigned int start1, \n");
  source.append("          unsigned int inc1, \n");
  source.append("          unsigned int size1, \n");
  source.append("          __local "); source.append(numeric_string); source.append(" * tmp_buffer, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * result) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" thread_result = vec1[start1]; \n");
  source.append("  for (unsigned int i = get_global_id(0); i<size1; i += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    "); source.append(numeric_string); source.append(" tmp = vec1[i*inc1+start1]; \n");
  if (is_max)
    source.append("      thread_result = thread_result > tmp ? thread_result : tmp; \n");
  else
    source.append("      thread_result = thread_result < tmp ? thread_result : tmp; \n");
  source.append("  } \n");

  source.append("  tmp_buffer[get_local_id(0)] = thread_result; \n");

  source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) \n");
  source.append("  { \n");
  source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
  source.append("    if (get_local_id(0) < stride) \n");
  source.append("    { \n");
  if (is_max)
    source.append("        tmp_buffer[get_local_id(0)] = tmp_buffer[get_local_id(0)] > tmp_buffer[get_local_id(0) + stride] ? tmp_buffer[get_local_id(0)] : tmp_buffer[get_local_id(0) + stride]; \n");
  else
    source.append("        tmp_buffer[get_local_id(0)] = tmp_buffer[get_local_id(0)] < tmp_buffer[get_local_id(0) + stride] ? tmp_buffer[get_local_id(0)] : tmp_buffer[get_local_id(0) + stride]; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("  barrier(CLK_LOCAL_MEM_FENCE); \n");

  source.append("  if (get_local_id(0) == 0) \n");
  source.append("    result[get_group_id(0)] = tmp_buffer[0]; \n");
  source.append("} \n");
}

//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for operations on/with viennacl::vector<> without involving matrices, multiple inner products, or element-wise operations other than addition or subtraction. */
template<typename NumericT>
struct vector
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_vector";
  }

  static void init(viennacl::ocl::context & ctx)
  {
    viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
    std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();

    static std::map<cl_context, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      std::string source;
      source.reserve(8192);

      viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

      // fully parametrized kernels:
      generate_avbv(source, numeric_string);

      // kernels with mostly predetermined skeleton:
      generate_plane_rotation(source, numeric_string);
      generate_vector_swap(source, numeric_string);
      generate_assign_cpu(source, numeric_string);

      generate_inner_prod(source, numeric_string, 1);
      generate_norm(source, numeric_string);
      generate_sum(source, numeric_string);
      generate_index_norm_inf(source, numeric_string);
      generate_maxmin(source, numeric_string, true);
      generate_maxmin(source, numeric_string, false);

      std::string prog_name = program_name();
      #ifdef VIENNACL_BUILD_INFO
      std::cout << "Creating program " << prog_name << std::endl;
      #endif
      ctx.add_program(source, prog_name);
      init_done[ctx.handle().get()] = true;
    } //if
  } //init
};

// class with kernels for multiple inner products.
/** @brief Main kernel class for generating OpenCL kernels for multiple inner products on/with viennacl::vector<>. */
template<typename NumericT>
struct vector_multi_inner_prod
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_vector_multi";
  }

  static void init(viennacl::ocl::context & ctx)
  {
    viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
    std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();

    static std::map<cl_context, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      std::string source;
      source.reserve(8192);

      viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

      generate_inner_prod(source, numeric_string, 2);
      generate_inner_prod(source, numeric_string, 3);
      generate_inner_prod(source, numeric_string, 4);
      generate_inner_prod(source, numeric_string, 8);

      generate_inner_prod_sum(source, numeric_string);

      std::string prog_name = program_name();
      #ifdef VIENNACL_BUILD_INFO
      std::cout << "Creating program " << prog_name << std::endl;
      #endif
      ctx.add_program(source, prog_name);
      init_done[ctx.handle().get()] = true;
    } //if
  } //init
};


template<typename StringT>
void generate_vector_convert(StringT & source, std::string const & dest_type, std::string const & src_type)
{
 source.append(" __kernel void convert_" + dest_type + "_" + src_type + "( \n");
 source.append("  __global " + dest_type + " * dest, \n");
 source.append("  unsigned int start_dest, unsigned int inc_dest, unsigned int size_dest, \n");
 source.append("  __global const " + src_type + " * src, \n");
 source.append("  unsigned int start_src, unsigned int inc_src) \n");
 source.append("  { \n");
 source.append("   for (unsigned int i = get_global_id(0); i < size_dest; i += get_global_size(0)) \n");
 source.append("     dest[start_dest + i * inc_dest] = src[start_src + i * inc_src]; \n");
 source.append("  } \n");
}

/** @brief Main kernel class for vector conversion routines (e.g. convert vector<int> to vector<float>). */
struct vector_convert
{

public:
  static std::string program_name()
  {
    return "vector_convert";
  }

  static void init(viennacl::ocl::context & ctx)
  {
    static std::map<cl_context, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      std::string source;
      source.reserve(4096);

      // int
      generate_vector_convert(source, viennacl::ocl::type_to_string<int>::apply(), viennacl::ocl::type_to_string<int>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<int>::apply(), viennacl::ocl::type_to_string<unsigned int>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<int>::apply(), viennacl::ocl::type_to_string<long>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<int>::apply(), viennacl::ocl::type_to_string<unsigned long>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<int>::apply(), viennacl::ocl::type_to_string<float>::apply());

      // unsigned int
      generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned int>::apply(), viennacl::ocl::type_to_string<int>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned int>::apply(), viennacl::ocl::type_to_string<unsigned int>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned int>::apply(), viennacl::ocl::type_to_string<long>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned int>::apply(), viennacl::ocl::type_to_string<unsigned long>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned int>::apply(), viennacl::ocl::type_to_string<float>::apply());

      // long
      generate_vector_convert(source, viennacl::ocl::type_to_string<long>::apply(), viennacl::ocl::type_to_string<int>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<long>::apply(), viennacl::ocl::type_to_string<unsigned int>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<long>::apply(), viennacl::ocl::type_to_string<long>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<long>::apply(), viennacl::ocl::type_to_string<unsigned long>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<long>::apply(), viennacl::ocl::type_to_string<float>::apply());

      // unsigned long
      generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned long>::apply(), viennacl::ocl::type_to_string<int>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned long>::apply(), viennacl::ocl::type_to_string<unsigned int>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned long>::apply(), viennacl::ocl::type_to_string<long>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned long>::apply(), viennacl::ocl::type_to_string<unsigned long>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned long>::apply(), viennacl::ocl::type_to_string<float>::apply());

      // float
      generate_vector_convert(source, viennacl::ocl::type_to_string<float>::apply(), viennacl::ocl::type_to_string<int>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<float>::apply(), viennacl::ocl::type_to_string<unsigned int>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<float>::apply(), viennacl::ocl::type_to_string<long>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<float>::apply(), viennacl::ocl::type_to_string<unsigned long>::apply());
      generate_vector_convert(source, viennacl::ocl::type_to_string<float>::apply(), viennacl::ocl::type_to_string<float>::apply());

      if (ctx.current_device().double_support())
      {
        viennacl::ocl::append_double_precision_pragma<double>(ctx, source);

        generate_vector_convert(source, viennacl::ocl::type_to_string<int>::apply(),           viennacl::ocl::type_to_string<double>::apply());
        generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned int>::apply(),  viennacl::ocl::type_to_string<double>::apply());
        generate_vector_convert(source, viennacl::ocl::type_to_string<long>::apply(),          viennacl::ocl::type_to_string<double>::apply());
        generate_vector_convert(source, viennacl::ocl::type_to_string<unsigned long>::apply(), viennacl::ocl::type_to_string<double>::apply());
        generate_vector_convert(source, viennacl::ocl::type_to_string<float>::apply(),         viennacl::ocl::type_to_string<double>::apply());

        generate_vector_convert(source, viennacl::ocl::type_to_string<double>::apply(), viennacl::ocl::type_to_string<int>::apply());
        generate_vector_convert(source, viennacl::ocl::type_to_string<double>::apply(), viennacl::ocl::type_to_string<unsigned int>::apply());
        generate_vector_convert(source, viennacl::ocl::type_to_string<double>::apply(), viennacl::ocl::type_to_string<long>::apply());
        generate_vector_convert(source, viennacl::ocl::type_to_string<double>::apply(), viennacl::ocl::type_to_string<unsigned long>::apply());
        generate_vector_convert(source, viennacl::ocl::type_to_string<double>::apply(), viennacl::ocl::type_to_string<float>::apply());
        generate_vector_convert(source, viennacl::ocl::type_to_string<double>::apply(), viennacl::ocl::type_to_string<double>::apply());
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

