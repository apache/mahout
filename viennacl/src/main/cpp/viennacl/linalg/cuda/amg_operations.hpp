#ifndef VIENNACL_LINALG_CUDA_AMG_OPERATIONS_HPP
#define VIENNACL_LINALG_CUDA_AMG_OPERATIONS_HPP

/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file cuda/amg_operations.hpp
    @brief Implementations of routines for AMG in OpenCL.
*/

#include <cstdlib>
#include <cmath>
#include "viennacl/linalg/detail/amg/amg_base.hpp"

#include <map>
#include <set>

namespace viennacl
{
namespace linalg
{
namespace cuda
{
namespace amg
{


///////////////////////////////////////////

__global__ void amg_influence_trivial_kernel(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          unsigned int size1,
          unsigned int nnz,
          unsigned int *influences_row,
          unsigned int *influences_id,
          unsigned int *influences_values
          )
{
  unsigned int global_id   = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int global_size = gridDim.x * blockDim.x;

  for (unsigned int i = global_id; i < size1; i += global_size)
  {
    unsigned int tmp = row_indices[i];
    influences_row[i] = tmp;
    influences_values[i] = row_indices[i+1] - tmp;
  }

  for (unsigned int i = global_id; i < nnz; i += global_size)
    influences_id[i] = column_indices[i];

  if (global_id == 0)
    influences_row[size1] = row_indices[size1];
}


/** @brief Routine for taking all connections in the matrix as strong */
template<typename NumericT>
void amg_influence_trivial(compressed_matrix<NumericT> const & A,
                           viennacl::linalg::detail::amg::amg_level_context & amg_context,
                           viennacl::linalg::amg_tag & tag)
{
  (void)tag;

  amg_influence_trivial_kernel<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                             viennacl::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                             static_cast<unsigned int>(A.size1()),
                                             static_cast<unsigned int>(A.nnz()),
                                             viennacl::cuda_arg(amg_context.influence_jumper_),
                                             viennacl::cuda_arg(amg_context.influence_ids_),
                                             viennacl::cuda_arg(amg_context.influence_values_)
                                            );
  VIENNACL_CUDA_LAST_ERROR_CHECK("amg_influence_trivial_kernel");
}


/** @brief Routine for extracting strongly connected points considering a user-provided threshold value */
template<typename NumericT>
void amg_influence_advanced(compressed_matrix<NumericT> const & A,
                            viennacl::linalg::detail::amg::amg_level_context & amg_context,
                            viennacl::linalg::amg_tag & tag)
{
  throw std::runtime_error("not implemented yet");
}

/** @brief Dispatcher for influence processing */
template<typename NumericT>
void amg_influence(compressed_matrix<NumericT> const & A,
                   viennacl::linalg::detail::amg::amg_level_context & amg_context,
                   viennacl::linalg::amg_tag & tag)
{
  // TODO: dispatch based on influence tolerance provided
  amg_influence_trivial(A, amg_context, tag);
}

/** @brief Assign IDs to coarse points.
*
*  TODO: Use exclusive_scan on GPU for this.
*/
inline void enumerate_coarse_points(viennacl::linalg::detail::amg::amg_level_context & amg_context)
{
  viennacl::backend::typesafe_host_array<unsigned int> point_types(amg_context.point_types_.handle(), amg_context.point_types_.size());
  viennacl::backend::typesafe_host_array<unsigned int> coarse_ids(amg_context.coarse_id_.handle(),    amg_context.coarse_id_.size());
  viennacl::backend::memory_read(amg_context.point_types_.handle(), 0, point_types.raw_size(), point_types.get());
  viennacl::backend::memory_read(amg_context.coarse_id_.handle(),   0, coarse_ids.raw_size(),  coarse_ids.get());

  unsigned int coarse_id = 0;
  for (std::size_t i=0; i<amg_context.point_types_.size(); ++i)
  {
    coarse_ids.set(i, coarse_id);
    if (point_types[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
      ++coarse_id;
  }

  amg_context.num_coarse_ = coarse_id;

  viennacl::backend::memory_write(amg_context.coarse_id_.handle(), 0, coarse_ids.raw_size(), coarse_ids.get());
}

//////////////////////////////////////

/** @brief CUDA kernel initializing the work vectors at each PMIS iteration */
template<typename IndexT>
__global__ void amg_pmis2_init_workdata(IndexT *work_state,
                                        IndexT *work_random,
                                        IndexT *work_index,
                                        IndexT const *point_types,
                                        IndexT const *random_weights,
                                        unsigned int size)
{
  unsigned int global_id   = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int global_size = gridDim.x * blockDim.x;

  for (unsigned int i = global_id; i < size; i += global_size)
  {
    switch (point_types[i])
    {
    case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED: work_state[i] = 1; break;
    case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE:      work_state[i] = 0; break;
    case viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE:    work_state[i] = 2; break;
    default:
      break; // do nothing
    }

    work_random[i] = random_weights[i];
    work_index[i]  = i;
  }
}

/** @brief CUDA kernel propagating the state triple (status, weight, nodeID) to neighbors using a max()-operation */
template<typename IndexT>
__global__ void amg_pmis2_max_neighborhood(IndexT const *work_state,
                                           IndexT const *work_random,
                                           IndexT const *work_index,
                                           IndexT       *work_state2,
                                           IndexT       *work_random2,
                                           IndexT       *work_index2,
                                           IndexT const *influences_row,
                                           IndexT const *influences_id,
                                           unsigned int size)
{
  unsigned int global_id   = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int global_size = gridDim.x * blockDim.x;

  for (unsigned int i = global_id; i < size; i += global_size)
  {
    // load
    unsigned int state  = work_state[i];
    unsigned int random = work_random[i];
    unsigned int index  = work_index[i];

    // max
    unsigned int j_stop = influences_row[i + 1];
    for (unsigned int j = influences_row[i]; j < j_stop; ++j)
    {
      unsigned int influenced_point_id = influences_id[j];

      // lexigraphical triple-max (not particularly pretty, but does the job):
      if (state < work_state[influenced_point_id])
      {
        state  = work_state[influenced_point_id];
        random = work_random[influenced_point_id];
        index  = work_index[influenced_point_id];
      }
      else if (state == work_state[influenced_point_id])
      {
        if (random < work_random[influenced_point_id])
        {
          state  = work_state[influenced_point_id];
          random = work_random[influenced_point_id];
          index  = work_index[influenced_point_id];
        }
        else if (random == work_random[influenced_point_id])
        {
          if (index < work_index[influenced_point_id])
          {
            state  = work_state[influenced_point_id];
            random = work_random[influenced_point_id];
            index  = work_index[influenced_point_id];
          }
        } // max(random)
      } // max(state)
    } // for

    // store
    work_state2[i]  = state;
    work_random2[i] = random;
    work_index2[i]  = index;
  }
}

/** @brief CUDA kernel for marking MIS and non-MIS nodes */
template<typename IndexT>
__global__ void amg_pmis2_mark_mis_nodes(IndexT const *work_state,
                                         IndexT const *work_index,
                                         IndexT *point_types,
                                         IndexT *undecided_buffer,
                                         unsigned int size)
{
  unsigned int global_id   = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int global_size = gridDim.x * blockDim.x;

  unsigned int num_undecided = 0;
  for (unsigned int i = global_id; i < size; i += global_size)
  {
    unsigned int max_state  = work_state[i];
    unsigned int max_index  = work_index[i];

    if (point_types[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
    {
      if (i == max_index) // make this a MIS node
        point_types[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE;
      else if (max_state == 2) // mind the mapping of viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE above!
        point_types[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;
      else
        num_undecided += 1;
    }
  }

  // reduction of the number of undecided nodes:
  __shared__ unsigned int shared_buffer[256];
  shared_buffer[threadIdx.x] = num_undecided;
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
      shared_buffer[threadIdx.x] += shared_buffer[threadIdx.x+stride];
  }

  if (threadIdx.x == 0)
    undecided_buffer[blockIdx.x] = shared_buffer[0];

}

/** @brief CUDA kernel for resetting non-MIS (i.e. coarse) points to undecided so that subsequent kernels work */
__global__ void amg_pmis2_reset_state(unsigned int *point_types, unsigned int size)
{
  unsigned int global_id   = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int global_size = gridDim.x * blockDim.x;

  for (unsigned int i = global_id; i < size; i += global_size)
  {
    if (point_types[i] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
      point_types[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED;
  }
}

/** @brief AG (aggregation based) coarsening, single-threaded version of stage 1
*
* @param A             Operator matrix on all levels
* @param amg_context   AMG hierarchy datastructures
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag_stage1_mis2(compressed_matrix<NumericT> const & A,
                               viennacl::linalg::detail::amg::amg_level_context & amg_context,
                               viennacl::linalg::amg_tag & tag)
{
  viennacl::vector<unsigned int> random_weights(A.size1(), viennacl::context(viennacl::MAIN_MEMORY));
  unsigned int *random_weights_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(random_weights.handle());
  for (std::size_t i=0; i<random_weights.size(); ++i)
    random_weights_ptr[i] = static_cast<unsigned int>(rand()) % static_cast<unsigned int>(A.size1());
  random_weights.switch_memory_context(viennacl::traits::context(A));

  // work vectors:
  viennacl::vector<unsigned int> work_state(A.size1(),  viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_random(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_index(A.size1(),  viennacl::traits::context(A));

  viennacl::vector<unsigned int> work_state2(A.size1(),  viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_random2(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_index2(A.size1(),  viennacl::traits::context(A));

  unsigned int num_undecided = static_cast<unsigned int>(A.size1());
  viennacl::vector<unsigned int> undecided_buffer(256, viennacl::traits::context(A));
  viennacl::backend::typesafe_host_array<unsigned int> undecided_buffer_host(undecided_buffer.handle(), undecided_buffer.size());

  unsigned int pmis_iters = 0;
  while (num_undecided > 0)
  {
    ++pmis_iters;

    //
    // init temporary work data:
    //
    amg_pmis2_init_workdata<<<128, 128>>>(viennacl::cuda_arg(work_state),
                                          viennacl::cuda_arg(work_random),
                                          viennacl::cuda_arg(work_index),
                                          viennacl::cuda_arg(amg_context.point_types_),
                                          viennacl::cuda_arg(random_weights),
                                          static_cast<unsigned int>(A.size1())
                                         );
    VIENNACL_CUDA_LAST_ERROR_CHECK("amg_pmis2_reset_state");


    //
    // Propagate maximum tuple twice
    //
    for (unsigned int r = 0; r < 2; ++r)
    {
      // max operation over neighborhood
      amg_pmis2_max_neighborhood<<<128, 128>>>(viennacl::cuda_arg(work_state),
                                               viennacl::cuda_arg(work_random),
                                               viennacl::cuda_arg(work_index),
                                               viennacl::cuda_arg(work_state2),
                                               viennacl::cuda_arg(work_random2),
                                               viennacl::cuda_arg(work_index2),
                                               viennacl::cuda_arg(amg_context.influence_jumper_),
                                               viennacl::cuda_arg(amg_context.influence_ids_),
                                               static_cast<unsigned int>(A.size1())
                                              );
      VIENNACL_CUDA_LAST_ERROR_CHECK("amg_pmis2_max_neighborhood");

      // copy work array (can be fused into a single kernel if needed. Previous kernel is in most cases sufficiently heavy)
      work_state  = work_state2;
      work_random = work_random2;
      work_index  = work_index2;
    }

    //
    // mark MIS and non-MIS nodes:
    //
    amg_pmis2_mark_mis_nodes<<<128, 128>>>(viennacl::cuda_arg(work_state),
                                           viennacl::cuda_arg(work_index),
                                           viennacl::cuda_arg(amg_context.point_types_),
                                           viennacl::cuda_arg(undecided_buffer),
                                           static_cast<unsigned int>(A.size1())
                                          );
    VIENNACL_CUDA_LAST_ERROR_CHECK("amg_pmis2_reset_state");

    // get number of undecided points on host:
    viennacl::backend::memory_read(undecided_buffer.handle(),   0, undecided_buffer_host.raw_size(),  undecided_buffer_host.get());
    num_undecided = 0;
    for (std::size_t i=0; i<undecided_buffer.size(); ++i)
      num_undecided += undecided_buffer_host[i];

  } //while

  // consistency with sequential MIS: reset state for non-coarse points, so that coarse indices are correctly picked up later
  amg_pmis2_reset_state<<<128, 128>>>(viennacl::cuda_arg(amg_context.point_types_),
                                      static_cast<unsigned int>(amg_context.point_types_.size())
                                     );
  VIENNACL_CUDA_LAST_ERROR_CHECK("amg_pmis2_reset_state");
}





template<typename IndexT>
__global__ void amg_agg_propagate_coarse_indices(IndexT       *point_types,
                                                 IndexT       *coarse_ids,
                                                 IndexT const *influences_row,
                                                 IndexT const *influences_id,
                                                 unsigned int size)
{
  unsigned int global_id   = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int global_size = gridDim.x * blockDim.x;

  for (unsigned int i = global_id; i < size; i += global_size)
  {
    if (point_types[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
    {
      unsigned int coarse_index = coarse_ids[i];

      unsigned int j_stop = influences_row[i + 1];
      for (unsigned int j = influences_row[i]; j < j_stop; ++j)
      {
        unsigned int influenced_point_id = influences_id[j];
        coarse_ids[influenced_point_id] = coarse_index; // Set aggregate index for fine point

        if (influenced_point_id != i) // Note: Any write races between threads are harmless here
          point_types[influenced_point_id] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;
      }
    }
  }
}


template<typename IndexT>
__global__ void amg_agg_merge_undecided(IndexT       *point_types,
                                        IndexT       *coarse_ids,
                                        IndexT const *influences_row,
                                        IndexT const *influences_id,
                                        unsigned int size)
{
  unsigned int global_id   = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int global_size = gridDim.x * blockDim.x;

  for (unsigned int i = global_id; i < size; i += global_size)
  {
    if (point_types[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
    {
      unsigned int j_stop = influences_row[i + 1];
      for (unsigned int j = influences_row[i]; j < j_stop; ++j)
      {
        unsigned int influenced_point_id = influences_id[j];
        if (point_types[influenced_point_id] != viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED) // either coarse or fine point
        {
          //std::cout << "Setting fine node " << i << " to be aggregated with node " << *influence_iter << "/" << pointvector.get_coarse_index(*influence_iter) << std::endl;
          coarse_ids[i] = coarse_ids[influenced_point_id];
          break;
        }
      }
    }
  }
}


__global__ void amg_agg_merge_undecided_2(unsigned int *point_types,
                                          unsigned int size)
{
  unsigned int global_id   = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int global_size = gridDim.x * blockDim.x;

  for (unsigned int i = global_id; i < size; i += global_size)
  {
    if (point_types[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_UNDECIDED)
      point_types[i] = viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_FINE;
  }
}


/** @brief AG (aggregation based) coarsening. Partially single-threaded version (VIENNACL_AMG_COARSE_AG)
*
* @param A             Operator matrix
* @param amg_context   AMG hierarchy datastructures
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag(compressed_matrix<NumericT> const & A,
                   viennacl::linalg::detail::amg::amg_level_context & amg_context,
                   viennacl::linalg::amg_tag & tag)
{

  amg_influence_trivial(A, amg_context, tag);

  //
  // Stage 1: Build aggregates:
  //
  if (tag.get_coarsening_method() == viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION)
    amg_coarse_ag_stage1_mis2(A, amg_context, tag);
  else
    throw std::runtime_error("Only MIS2 coarsening implemented. Selected coarsening not available with CUDA backend!");

  viennacl::linalg::cuda::amg::enumerate_coarse_points(amg_context);

  //
  // Stage 2: Propagate coarse aggregate indices to neighbors:
  //
  amg_agg_propagate_coarse_indices<<<128, 128>>>(viennacl::cuda_arg(amg_context.point_types_),
                                                 viennacl::cuda_arg(amg_context.coarse_id_),
                                                 viennacl::cuda_arg(amg_context.influence_jumper_),
                                                 viennacl::cuda_arg(amg_context.influence_ids_),
                                                 static_cast<unsigned int>(A.size1())
                                                );
  VIENNACL_CUDA_LAST_ERROR_CHECK("amg_agg_propagate_coarse_indices");


  //
  // Stage 3: Merge remaining undecided points (merging to first aggregate found when cycling over the hierarchy
  //
  amg_agg_merge_undecided<<<128, 128>>>(viennacl::cuda_arg(amg_context.point_types_),
                                        viennacl::cuda_arg(amg_context.coarse_id_),
                                        viennacl::cuda_arg(amg_context.influence_jumper_),
                                        viennacl::cuda_arg(amg_context.influence_ids_),
                                        static_cast<unsigned int>(A.size1())
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("amg_agg_merge_undecided");

  //
  // Stage 4: Set undecided points to fine points (coarse ID already set in Stage 3)
  //          Note: Stage 3 and Stage 4 were initially fused, but are now split in order to avoid race conditions (or a fallback to sequential execution).
  //
  amg_agg_merge_undecided_2<<<128, 128>>>(viennacl::cuda_arg(amg_context.point_types_),
                                         static_cast<unsigned int>(A.size1())
                                        );
  VIENNACL_CUDA_LAST_ERROR_CHECK("amg_agg_merge_undecided_2");
}




/** @brief Calls the right coarsening procedure
*
* @param A            Operator matrix on all levels
* @param amg_context  AMG hierarchy datastructures
* @param tag          AMG preconditioner tag
*/
template<typename InternalT1>
void amg_coarse(InternalT1 & A,
                viennacl::linalg::detail::amg::amg_level_context & amg_context,
                viennacl::linalg::amg_tag & tag)
{
  switch (tag.get_coarsening_method())
  {
  case viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION: amg_coarse_ag(A, amg_context, tag); break;
  default: throw std::runtime_error("not implemented yet");
  }
}




////////////////////////////////////// Interpolation /////////////////////////////

template<typename NumericT>
__global__ void amg_interpol_ag_kernel(unsigned int *P_row_buffer,
                                       unsigned int *P_col_buffer,
                                       NumericT *P_elements,
                                       unsigned int *coarse_ids,
                                       unsigned int size)
{
  unsigned int global_id   = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int global_size = gridDim.x * blockDim.x;

  for (unsigned int i = global_id; i < size; i += global_size)
  {
    P_row_buffer[i] = i;
    P_col_buffer[i] = coarse_ids[i];
    P_elements[i]   = NumericT(1);
  }

  // set last entry as well:
  if (global_id == 0)
    P_row_buffer[size] = size;
}

/** @brief AG (aggregation based) interpolation. Multi-Threaded! (VIENNACL_INTERPOL_SA)
 *
 * @param A            Operator matrix
 * @param P            Prolongation matrix
 * @param amg_context  AMG hierarchy datastructures
 * @param tag          AMG configuration tag
*/
template<typename NumericT>
void amg_interpol_ag(compressed_matrix<NumericT> const & A,
                     compressed_matrix<NumericT> & P,
                     viennacl::linalg::detail::amg::amg_level_context & amg_context,
                     viennacl::linalg::amg_tag & tag)
{
  (void)tag;
  P = compressed_matrix<NumericT>(A.size1(), amg_context.num_coarse_, A.size1(), viennacl::traits::context(A));

  amg_interpol_ag_kernel<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(P.handle1().cuda_handle()),
                                       viennacl::cuda_arg<unsigned int>(P.handle2().cuda_handle()),
                                       viennacl::cuda_arg<NumericT>(P.handle().cuda_handle()),
                                       viennacl::cuda_arg(amg_context.coarse_id_),
                                       static_cast<unsigned int>(A.size1())
                                      );
  VIENNACL_CUDA_LAST_ERROR_CHECK("amg_interpol_ag_kernel");

  P.generate_row_block_information();
}



template<typename NumericT>
__global__ void amg_interpol_sa_kernel(
          const unsigned int *A_row_indices,
          const unsigned int *A_col_indices,
          const NumericT     *A_elements,
          unsigned int A_size1,
          unsigned int A_nnz,
          unsigned int *Jacobi_row_indices,
          unsigned int *Jacobi_col_indices,
          NumericT     *Jacobi_elements,
          NumericT     omega
          )
{
  unsigned int global_id   = blockDim.x * blockIdx.x + threadIdx.x;
  unsigned int global_size = gridDim.x * blockDim.x;

  for (unsigned int row = global_id; row < A_size1; row += global_size)
  {
    unsigned int row_begin = A_row_indices[row];
    unsigned int row_end   = A_row_indices[row+1];

    Jacobi_row_indices[row] = row_begin;

    // Step 1: Extract diagonal:
    NumericT diag = 0;
    for (unsigned int j = row_begin; j < row_end; ++j)
    {
      if (A_col_indices[j] == row)
      {
        diag = A_elements[j];
        break;
      }
    }

    // Step 2: Write entries:
    for (unsigned int j = row_begin; j < row_end; ++j)
    {
      unsigned int col_index = A_col_indices[j];
      Jacobi_col_indices[j] = col_index;

      if (col_index == row)
        Jacobi_elements[j] = NumericT(1) - omega;
      else
        Jacobi_elements[j] = - omega * A_elements[j] / diag;
    }
  }

  if (global_id == 0)
    Jacobi_row_indices[A_size1] = A_nnz; // don't forget finalizer
}



/** @brief Smoothed aggregation interpolation. (VIENNACL_INTERPOL_SA)
 *
 * @param A            Operator matrix
 * @param P            Prolongation matrix
 * @param amg_context  AMG hierarchy datastructures
 * @param tag          AMG configuration tag
*/
template<typename NumericT>
void amg_interpol_sa(compressed_matrix<NumericT> const & A,
                     compressed_matrix<NumericT> & P,
                     viennacl::linalg::detail::amg::amg_level_context & amg_context,
                     viennacl::linalg::amg_tag & tag)
{
  (void)tag;
  viennacl::compressed_matrix<NumericT> P_tentative(A.size1(), amg_context.num_coarse_, A.size1(), viennacl::traits::context(A));

  // form tentative operator:
  amg_interpol_ag(A, P_tentative, amg_context, tag);

  viennacl::compressed_matrix<NumericT> Jacobi(A.size1(), A.size1(), A.nnz(), viennacl::traits::context(A));

  amg_interpol_sa_kernel<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                       viennacl::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                       viennacl::cuda_arg<NumericT>(A.handle().cuda_handle()),
                                       static_cast<unsigned int>(A.size1()),
                                       static_cast<unsigned int>(A.nnz()),
                                       viennacl::cuda_arg<unsigned int>(Jacobi.handle1().cuda_handle()),
                                       viennacl::cuda_arg<unsigned int>(Jacobi.handle2().cuda_handle()),
                                       viennacl::cuda_arg<NumericT>(Jacobi.handle().cuda_handle()),
                                       NumericT(tag.get_jacobi_weight())
                                      );
  VIENNACL_CUDA_LAST_ERROR_CHECK("amg_interpol_sa_kernel");

  P = viennacl::linalg::prod(Jacobi, P_tentative);

  P.generate_row_block_information();
}


/** @brief Dispatcher for building the interpolation matrix
 *
 * @param A            Operator matrix
 * @param P            Prolongation matrix
 * @param amg_context  AMG hierarchy datastructures
 * @param tag          AMG configuration tag
*/
template<typename MatrixT>
void amg_interpol(MatrixT const & A,
                  MatrixT & P,
                  viennacl::linalg::detail::amg::amg_level_context & amg_context,
                  viennacl::linalg::amg_tag & tag)
{
  switch (tag.get_interpolation_method())
  {
  case viennacl::linalg::AMG_INTERPOLATION_METHOD_AGGREGATION:          amg_interpol_ag     (A, P, amg_context, tag); break;
  case viennacl::linalg::AMG_INTERPOLATION_METHOD_SMOOTHED_AGGREGATION: amg_interpol_sa     (A, P, amg_context, tag); break;
  default: throw std::runtime_error("Not implemented yet!");
  }
}


template<typename NumericT>
__global__ void compressed_matrix_assign_to_dense(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
          NumericT *B,
          unsigned int B_row_start,     unsigned int B_col_start,
          unsigned int B_row_inc,       unsigned int B_col_inc,
          unsigned int B_row_size,      unsigned int B_col_size,
          unsigned int B_internal_rows, unsigned int B_internal_cols)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < B_row_size;
                    row += gridDim.x * blockDim.x)
  {
    unsigned int row_end = row_indices[row+1];
    for (unsigned int j = row_indices[row]; j<row_end; j++)
      B[(B_row_start + row * B_row_inc) * B_internal_cols + B_col_start + column_indices[j] * B_col_inc] = elements[j];
  }
}


template<typename NumericT, unsigned int AlignmentV>
void assign_to_dense(viennacl::compressed_matrix<NumericT, AlignmentV> const & A,
                     viennacl::matrix_base<NumericT> & B)
{
  compressed_matrix_assign_to_dense<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                                  viennacl::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                                  viennacl::cuda_arg<NumericT>(A.handle().cuda_handle()),
                                                  viennacl::cuda_arg<NumericT>(B),
                                                  static_cast<unsigned int>(viennacl::traits::start1(B)),           static_cast<unsigned int>(viennacl::traits::start2(B)),
                                                  static_cast<unsigned int>(viennacl::traits::stride1(B)),          static_cast<unsigned int>(viennacl::traits::stride2(B)),
                                                  static_cast<unsigned int>(viennacl::traits::size1(B)),            static_cast<unsigned int>(viennacl::traits::size2(B)),
                                                  static_cast<unsigned int>(viennacl::traits::internal_size1(B)),   static_cast<unsigned int>(viennacl::traits::internal_size2(B))
                                                 );
  VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_assign_to_dense");
}




template<typename NumericT>
__global__ void compressed_matrix_smooth_jacobi_kernel(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
          NumericT weight,
          const NumericT * x_old,
          NumericT * x_new,
          const NumericT * rhs,
          unsigned int size)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < size;
                    row += gridDim.x * blockDim.x)
  {
    NumericT sum = NumericT(0);
    NumericT diag = NumericT(1);
    unsigned int row_end = row_indices[row+1];
    for (unsigned int j = row_indices[row]; j < row_end; ++j)
    {
      unsigned int col = column_indices[j];
      if (col == row)
        diag = elements[j];
      else
        sum += elements[j] * x_old[col];
    }
    x_new[row] = weight * (rhs[row] - sum) / diag + (NumericT(1) - weight) * x_old[row];
  }
}




/** @brief Damped Jacobi Smoother (CUDA version)
*
* @param iterations  Number of smoother iterations
* @param A           Operator matrix for the smoothing
* @param x           The vector smoothing is applied to
* @param x_backup    (Different) Vector holding the same values as x
* @param rhs_smooth  The right hand side of the equation for the smoother
* @param weight      Damping factor. 0: No effect of smoother. 1: Undamped Jacobi iteration
*/
template<typename NumericT>
void smooth_jacobi(unsigned int iterations,
                   compressed_matrix<NumericT> const & A,
                   vector<NumericT> & x,
                   vector<NumericT> & x_backup,
                   vector<NumericT> const & rhs_smooth,
                   NumericT weight)
{
  for (unsigned int i=0; i<iterations; ++i)
  {
    x_backup = x;

    compressed_matrix_smooth_jacobi_kernel<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(A.handle1().cuda_handle()),
                                                         viennacl::cuda_arg<unsigned int>(A.handle2().cuda_handle()),
                                                         viennacl::cuda_arg<NumericT>(A.handle().cuda_handle()),
                                                         static_cast<NumericT>(weight),
                                                         viennacl::cuda_arg(x_backup),
                                                         viennacl::cuda_arg(x),
                                                         viennacl::cuda_arg(rhs_smooth),
                                                         static_cast<unsigned int>(rhs_smooth.size())
                                                        );
    VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_smooth_jacobi_kernel");
  }
}


} //namespace amg
} //namespace host_based
} //namespace linalg
} //namespace viennacl

#endif
