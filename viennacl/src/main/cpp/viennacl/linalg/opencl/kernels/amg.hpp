#ifndef VIENNACL_LINALG_OPENCL_KERNELS_AMG_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_AMG_HPP

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

#include "viennacl/linalg/opencl/common.hpp"

/** @file viennacl/linalg/opencl/kernels/amg.hpp
 *  @brief OpenCL kernel file for operations related to algebraic multigrid */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{


template<typename StringT>
void generate_amg_influence_trivial(StringT & source)
{

 source.append("__kernel void amg_influence_trivial( \n");
 source.append("  __global const unsigned int * A_row_indices, \n");
 source.append("  __global const unsigned int * A_col_indices, \n");
 source.append("  unsigned int A_size1, \n");
 source.append("  unsigned int A_nnz, \n");
 source.append("  __global unsigned int * influences_row, \n");
 source.append("  __global unsigned int * influences_id, \n");
 source.append("  __global unsigned int * influences_values) { \n");

 source.append("  for (unsigned int i = get_global_id(0); i < A_size1; i += get_global_size(0)) \n");
 source.append("  { \n");
 source.append("    unsigned int tmp = A_row_indices[i]; \n");
 source.append("    influences_row[i] = tmp; \n");
 source.append("    influences_values[i] = A_row_indices[i+1] - tmp; \n");
 source.append("  } \n");

 source.append("  for (unsigned int i = get_global_id(0); i < A_nnz; i += get_global_size(0)) \n");
 source.append("    influences_id[i] = A_col_indices[i]; \n");

 source.append("  if (get_global_id(0) == 0) \n");
 source.append("    influences_row[A_size1] = A_row_indices[A_size1]; \n");
 source.append("} \n");

}


template<typename StringT>
void generate_amg_pmis2_init_workdata(StringT & source)
{

 source.append("__kernel void amg_pmis2_init_workdata( \n");
 source.append("  __global unsigned int       *work_state, \n");
 source.append("  __global unsigned int       *work_random, \n");
 source.append("  __global unsigned int       *work_index, \n");
 source.append("  __global unsigned int const *point_types, \n");
 source.append("  __global unsigned int const *random_weights, \n");
 source.append("  unsigned int size) { \n");

 source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
 source.append("    switch (point_types[i]) { \n");
 source.append("    case 0:  work_state[i] = 1; break; \n"); //viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED
 source.append("    case 1:  work_state[i] = 2; break; \n"); //viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE
 source.append("    case 2:  work_state[i] = 0; break; \n"); //viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE

 source.append("    default: break; // do nothing \n");
 source.append("    } \n");

 source.append("    work_random[i] = random_weights[i]; \n");
 source.append("    work_index[i]  = i; \n");
 source.append("  } \n");
 source.append("} \n");
}



template<typename StringT>
void generate_amg_pmis2_max_neighborhood(StringT & source)
{

 source.append("__kernel void amg_pmis2_max_neighborhood( \n");
 source.append("  __global unsigned int       *work_state, \n");
 source.append("  __global unsigned int       *work_random, \n");
 source.append("  __global unsigned int       *work_index, \n");
 source.append("  __global unsigned int       *work_state2, \n");
 source.append("  __global unsigned int       *work_random2, \n");
 source.append("  __global unsigned int       *work_index2, \n");
 source.append("  __global unsigned int const *influences_row, \n");
 source.append("  __global unsigned int const *influences_id, \n");
 source.append("  unsigned int size) { \n");

 source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");

 // load
 source.append("    unsigned int state  = work_state[i]; \n");
 source.append("    unsigned int random = work_random[i]; \n");
 source.append("    unsigned int index  = work_index[i]; \n");

 // max
 source.append("    unsigned int j_stop = influences_row[i + 1]; \n");
 source.append("    for (unsigned int j = influences_row[i]; j < j_stop; ++j) { \n");
 source.append("      unsigned int influenced_point_id = influences_id[j]; \n");

 // lexigraphical triple-max (not particularly pretty, but does the job):
 source.append("      if (state < work_state[influenced_point_id]) { \n");
 source.append("        state  = work_state[influenced_point_id]; \n");
 source.append("        random = work_random[influenced_point_id]; \n");
 source.append("        index  = work_index[influenced_point_id]; \n");
 source.append("      } else if (state == work_state[influenced_point_id]) { \n");
 source.append("        if (random < work_random[influenced_point_id]) { \n");
 source.append("          state  = work_state[influenced_point_id]; \n");
 source.append("          random = work_random[influenced_point_id]; \n");
 source.append("          index  = work_index[influenced_point_id]; \n");
 source.append("        } else if (random == work_random[influenced_point_id]) { \n");
 source.append("          if (index < work_index[influenced_point_id]) { \n");
 source.append("            state  = work_state[influenced_point_id]; \n");
 source.append("            random = work_random[influenced_point_id]; \n");
 source.append("            index  = work_index[influenced_point_id]; \n");
 source.append("          } \n");
 source.append("        } \n");
 source.append("      } \n");

 source.append("    }\n"); //for

 // store
 source.append("    work_state2[i]  = state; \n");
 source.append("    work_random2[i] = random; \n");
 source.append("    work_index2[i]  = index; \n");
 source.append("  } \n");
 source.append("} \n");
}



template<typename StringT>
void generate_amg_pmis2_mark_mis_nodes(StringT & source)
{

 source.append("__kernel void amg_pmis2_mark_mis_nodes( \n");
 source.append("  __global unsigned int const *work_state, \n");
 source.append("  __global unsigned int const *work_index, \n");
 source.append("  __global unsigned int       *point_types, \n");
 source.append("  __global unsigned int       *undecided_buffer, \n");
 source.append("  unsigned int size) { \n");

 source.append("  unsigned int num_undecided = 0; \n");
 source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
 source.append("    unsigned int max_state  = work_state[i]; \n");
 source.append("    unsigned int max_index  = work_index[i]; \n");

 source.append("    if (point_types[i] == 0) { \n");                     // viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED
 source.append("      if      (i == max_index) point_types[i] = 1; \n"); // viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE
 source.append("      else if (max_state == 2) point_types[i] = 2; \n"); // viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE
 source.append("      else                     num_undecided += 1; \n");
 source.append("    } \n");
 source.append("  } \n");

 // reduction in shared memory:
 source.append("  __local unsigned int shared_buffer[256]; \n");
 source.append("  shared_buffer[get_local_id(0)] = num_undecided; \n");
 source.append("  for (unsigned int stride = get_local_size(0)/2; stride > 0; stride /= 2) { \n");
 source.append("    barrier(CLK_LOCAL_MEM_FENCE); \n");
 source.append("    if (get_local_id(0) < stride) shared_buffer[get_local_id(0)] += shared_buffer[get_local_id(0)+stride]; \n");
 source.append("  } \n");

 source.append("  if (get_local_id(0) == 0) \n");
 source.append("    undecided_buffer[get_group_id(0)] = shared_buffer[0]; \n");

 source.append("} \n");
}


template<typename StringT>
void generate_amg_pmis2_reset_state(StringT & source)
{

 source.append("__kernel void amg_pmis2_reset_state( \n");
 source.append("  __global unsigned int *point_types, \n");
 source.append("  unsigned int size) { \n");

 source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
 source.append("    if (point_types[i] != 1) point_types[i] = 0;\n"); // mind mapping of POINT_TYPE_COARSE and POINT_TYPE_UNDECIDED
 source.append("  } \n");

 source.append("} \n");
}



//////////////



template<typename StringT>
void generate_amg_agg_propagate_coarse_indices(StringT & source)
{

 source.append(" __kernel void amg_agg_propagate_coarse_indices( \n");
 source.append("  __global unsigned int       *point_types, \n");
 source.append("  __global unsigned int       *coarse_ids, \n");
 source.append("  __global unsigned int const *influences_row, \n");
 source.append("  __global unsigned int const *influences_id, \n");
 source.append("  unsigned int size) { \n");

 source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) \n");
 source.append("  { \n");
 source.append("    if (point_types[i] == 1) { \n"); //viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE
 source.append("      unsigned int coarse_index = coarse_ids[i]; \n");

 source.append("      unsigned int j_stop = influences_row[i + 1]; \n");
 source.append("      for (unsigned int j = influences_row[i]; j < j_stop; ++j) { \n");
 source.append("        unsigned int influenced_point_id = influences_id[j]; \n");
 source.append("        coarse_ids[influenced_point_id] = coarse_index; \n");
 source.append("        if (influenced_point_id != i) point_types[influenced_point_id] = 2; \n"); //viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE
 source.append("      } \n");
 source.append("    } \n");
 source.append("  } \n");
 source.append("} \n");

}



template<typename StringT>
void generate_amg_agg_merge_undecided(StringT & source)
{

 source.append(" __kernel void amg_agg_merge_undecided( \n");
 source.append("  __global unsigned int       *point_types, \n");
 source.append("  __global unsigned int       *coarse_ids, \n");
 source.append("  __global unsigned int const *influences_row, \n");
 source.append("  __global unsigned int const *influences_id, \n");
 source.append("  unsigned int size) { \n");

 source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) \n");
 source.append("  { \n");
 source.append("    if (point_types[i] == 0) { \n"); //viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED

 source.append("      unsigned int j_stop = influences_row[i + 1]; \n");
 source.append("      for (unsigned int j = influences_row[i]; j < j_stop; ++j) { \n");
 source.append("        unsigned int influenced_point_id = influences_id[j]; \n");
 source.append("        if (point_types[influenced_point_id] != 0) { \n");       // viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED
 source.append("          coarse_ids[i] = coarse_ids[influenced_point_id]; \n");
 source.append("          break; \n");
 source.append("        } \n");
 source.append("      } \n");

 source.append("    } \n");
 source.append("  } \n");
 source.append("} \n");

}


template<typename StringT>
void generate_amg_agg_merge_undecided_2(StringT & source)
{

 source.append(" __kernel void amg_agg_merge_undecided_2( \n");
 source.append("  __global unsigned int *point_types, \n");
 source.append("  unsigned int size) { \n");

 source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) \n");
 source.append("    if (point_types[i] == 0) point_types[i] = 2; \n"); // POINT_TYPE_UNDECIDED to POINT_TYPE_FINE

 source.append("} \n");
}

//////////////////////

template<typename StringT>
void generate_amg_interpol_ag(StringT & source, std::string const & numeric_string)
{

 source.append(" __kernel void amg_interpol_ag( \n");
 source.append("  __global unsigned int * P_row_indices, \n");
 source.append("  __global unsigned int * P_column_indices, \n");
 source.append("  __global "); source.append(numeric_string); source.append(" * P_elements, \n");
 source.append("  __global const unsigned int * coarse_agg_ids, \n");
 source.append("  unsigned int size) { \n");

 source.append("   for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) \n");
 source.append("   { \n");
 source.append("     P_row_indices[i] = i; \n");
 source.append("     P_column_indices[i] = coarse_agg_ids[i]; \n");
 source.append("     P_elements[i] = 1; \n");
 source.append("   } \n");
 source.append("   if (get_global_id(0) == 0) P_row_indices[size] = size; \n");
 source.append("  } \n");

}

template<typename StringT>
void generate_amg_interpol_sa(StringT & source, std::string const & numeric_string)
{

 source.append("__kernel void amg_interpol_sa( \n");
 source.append(" __global unsigned int const *A_row_indices, \n");
 source.append(" __global unsigned int const *A_col_indices, \n");
 source.append(" __global "); source.append(numeric_string); source.append(" const *A_elements, \n");
 source.append(" unsigned int A_size1, \n");
 source.append(" unsigned int A_nnz, \n");
 source.append(" __global unsigned int *Jacobi_row_indices, \n");
 source.append(" __global unsigned int *Jacobi_col_indices, \n");
 source.append(" __global "); source.append(numeric_string); source.append(" *Jacobi_elements, \n");
 source.append(" "); source.append(numeric_string); source.append(" omega) { \n");

 source.append("  for (unsigned int row = get_global_id(0); row < A_size1; row += get_global_size(0)) \n");
 source.append("  { \n");
 source.append("    unsigned int row_begin = A_row_indices[row]; \n");
 source.append("    unsigned int row_end   = A_row_indices[row+1]; \n");

 source.append("    Jacobi_row_indices[row] = row_begin; \n");

 // Step 1: Extract diagonal:
 source.append("    "); source.append(numeric_string); source.append(" diag = 0; \n");
 source.append("    for (unsigned int j = row_begin; j < row_end; ++j) { \n");
 source.append("      if (A_col_indices[j] == row) { \n");
 source.append("        diag = A_elements[j]; \n");
 source.append("        break; \n");
 source.append("      } \n");
 source.append("    } \n");

 // Step 2: Write entries:
 source.append("    for (unsigned int j = row_begin; j < row_end; ++j) { \n");
 source.append("      unsigned int col_index = A_col_indices[j]; \n");
 source.append("      Jacobi_col_indices[j] = col_index; \n");
 source.append("      Jacobi_elements[j] = (col_index == row) ? (1 - omega) : (-omega * A_elements[j] / diag); \n");
 source.append("    } \n");

 source.append("  } \n");
 source.append("  if (get_global_id(0) == 0) Jacobi_row_indices[A_size1] = A_nnz; \n");
 source.append("} \n");

}
//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for compressed_matrix. */
template<typename NumericT>
struct amg
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_amg";
  }

  static void init(viennacl::ocl::context & ctx)
  {
    static std::map<cl_context, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
      std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();

      std::string source;
      source.reserve(2048);

      viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

      generate_amg_influence_trivial(source);
      generate_amg_pmis2_init_workdata(source);
      generate_amg_pmis2_max_neighborhood(source);
      generate_amg_pmis2_mark_mis_nodes(source);
      generate_amg_pmis2_reset_state(source);
      generate_amg_agg_propagate_coarse_indices(source);
      generate_amg_agg_merge_undecided(source);
      generate_amg_agg_merge_undecided_2(source);

      generate_amg_interpol_ag(source, numeric_string);
      generate_amg_interpol_sa(source, numeric_string);

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

