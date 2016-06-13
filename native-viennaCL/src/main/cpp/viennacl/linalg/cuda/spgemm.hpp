#ifndef VIENNACL_LINALG_CUDA_SPGEMM_HPP_
#define VIENNACL_LINALG_CUDA_SPGEMM_HPP_

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

/** @file viennacl/linalg/cuda/sparse_matrix_operations.hpp
    @brief Implementations of operations using sparse matrices using CUDA
*/

#include <stdexcept>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/cuda/common.hpp"

#include "viennacl/tools/timer.hpp"

#include "viennacl/linalg/cuda/sparse_matrix_operations_solve.hpp"

namespace viennacl
{
namespace linalg
{
namespace cuda
{

/** @brief Loads a value from the specified address. With CUDA arch 3.5 and above the value is also stored in global constant memory for later reuse */
template<typename NumericT>
static inline __device__ NumericT load_and_cache(const NumericT *address)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
  return __ldg(address);
#else
  return *address;
#endif
}


//
// Stage 1: Obtain upper bound for number of elements per row in C:
//
template<typename IndexT>
__device__ IndexT round_to_next_power_of_2(IndexT val)
{
  if (val > 32)
    return 64; // just to indicate that we need to split/factor the matrix!
  else if (val > 16)
    return 32;
  else if (val > 8)
    return 16;
  else if (val > 4)
    return 8;
  else if (val > 2)
    return 4;
  else if (val > 1)
    return 2;
  else
    return 1;
}

template<typename IndexT>
__global__ void compressed_matrix_gemm_stage_1(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          IndexT A_size1,
          const IndexT * B_row_indices,
          IndexT *subwarpsize_per_group,
          IndexT *max_nnz_row_A_per_group,
          IndexT *max_nnz_row_B_per_group)
{
  unsigned int subwarpsize_in_thread = 0;
  unsigned int max_nnz_row_A = 0;
  unsigned int max_nnz_row_B = 0;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + threadIdx.x; row < row_per_group_end; row += blockDim.x)
  {
    unsigned int A_row_start = A_row_indices[row];
    unsigned int A_row_end   = A_row_indices[row+1];
    unsigned int row_num = A_row_end - A_row_start;
    if (row_num > 32) // too many rows in B need to be merged for a single pass
    {
      unsigned int subwarp_sqrt = (unsigned int)sqrt(double(row_num)) + 1;
      subwarpsize_in_thread = max(subwarp_sqrt, subwarpsize_in_thread);
    }
    else
      subwarpsize_in_thread = max(A_row_end - A_row_start, subwarpsize_in_thread);
    max_nnz_row_A = max(max_nnz_row_A, row_num);
    for (unsigned int j = A_row_start; j < A_row_end; ++j)
    {
      unsigned int col = A_col_indices[j];
      unsigned int row_len_B = B_row_indices[col + 1] - B_row_indices[col];
      max_nnz_row_B = max(row_len_B, max_nnz_row_B);
    }
  }

  // reduction to obtain maximum in thread block
  __shared__ unsigned int shared_subwarpsize[256];
  __shared__ unsigned int shared_max_nnz_row_A[256];
  __shared__ unsigned int shared_max_nnz_row_B[256];

    shared_subwarpsize[threadIdx.x] = subwarpsize_in_thread;
  shared_max_nnz_row_A[threadIdx.x] = max_nnz_row_A;
  shared_max_nnz_row_B[threadIdx.x] = max_nnz_row_B;
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
    {
        shared_subwarpsize[threadIdx.x] = max(  shared_subwarpsize[threadIdx.x],   shared_subwarpsize[threadIdx.x + stride]);
      shared_max_nnz_row_A[threadIdx.x] = max(shared_max_nnz_row_A[threadIdx.x], shared_max_nnz_row_A[threadIdx.x + stride]);
      shared_max_nnz_row_B[threadIdx.x] = max(shared_max_nnz_row_B[threadIdx.x], shared_max_nnz_row_B[threadIdx.x + stride]);
    }
  }

  if (threadIdx.x == 0)
  {
      subwarpsize_per_group[blockIdx.x] = round_to_next_power_of_2(shared_subwarpsize[0]);
    max_nnz_row_A_per_group[blockIdx.x] = shared_max_nnz_row_A[0];
    max_nnz_row_B_per_group[blockIdx.x] = shared_max_nnz_row_B[0];
  }
}

//
// Stage 2: Determine sparsity pattern of C
//
inline __device__ unsigned int merge_subwarp_symbolic(unsigned int row_B_start, unsigned int row_B_end, unsigned int const *B_col_indices, unsigned int B_size2, unsigned int subwarpsize)
{
  unsigned int current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;

  unsigned int num_nnz = 0;
  while (1)
  {
    // determine current minimum (warp shuffle)
    unsigned int min_index = current_front_index;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      min_index = min(min_index, __shfl_xor((int)min_index, (int)i));

    if (min_index == B_size2)
      break;

    // update front:
    current_front_index = (current_front_index == min_index) ? ((++row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2)
                                                             : current_front_index;
    ++num_nnz;
  }

  return num_nnz;
}

inline __device__ unsigned int merge_subwarp_symbolic_double(unsigned int row_B_start, unsigned int row_B_end, unsigned int const *B_col_indices, unsigned int B_size2,
                                                             unsigned int *output_array, unsigned int id_in_warp, unsigned int subwarpsize)
{
  unsigned int current_front_index = (row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2;

  unsigned int num_nnz = 0;
  unsigned int index_buffer = 0;
  unsigned int buffer_size = 0;
  while (1)
  {
    // determine current minimum (warp shuffle)
    unsigned int min_index = current_front_index;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      min_index = min(min_index, __shfl_xor((int)min_index, (int)i));

    if (min_index == B_size2)
      break;

    // update front:
    current_front_index = (current_front_index == min_index) ? ((++row_B_start < row_B_end) ? load_and_cache(B_col_indices + row_B_start) : B_size2)
                                                             : current_front_index;

    // write output
    index_buffer = (id_in_warp == buffer_size) ? min_index : index_buffer;
    ++buffer_size;

    if (buffer_size == subwarpsize) // register buffer full?
    {
      output_array[id_in_warp] = index_buffer;
      output_array += subwarpsize;
      buffer_size = 0;
    }

    ++num_nnz;
  }

  // write remaining entries from register buffer:
  if (id_in_warp < buffer_size)
    output_array[id_in_warp] = index_buffer;

  return num_nnz;
}

template<typename IndexT>
__global__ void compressed_matrix_gemm_stage_2(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          IndexT A_size1,
          const IndexT * B_row_indices,
          const IndexT * B_col_indices,
          IndexT B_size2,
          IndexT * C_row_indices,
          unsigned int *subwarpsize_array,
          unsigned int *max_row_size_A,
          unsigned int *max_row_size_B,
          unsigned int *scratchpad_offsets,
          unsigned int *scratchpad_indices)
{
  unsigned int subwarpsize = subwarpsize_array[blockIdx.x];

  unsigned int num_warps  =  blockDim.x / subwarpsize;
  unsigned int warp_id    = threadIdx.x / subwarpsize;
  unsigned int id_in_warp = threadIdx.x % subwarpsize;

  unsigned int scratchpad_rowlength     = max_row_size_B[blockIdx.x] * subwarpsize;
  unsigned int scratchpad_rows_per_warp = max_row_size_A[blockIdx.x] / subwarpsize + 1;
  unsigned int *subwarp_scratchpad_start = scratchpad_indices + scratchpad_offsets[blockIdx.x] + warp_id * scratchpad_rows_per_warp * scratchpad_rowlength;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + warp_id; row < row_per_group_end; row += num_warps)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    if (row_A_end - row_A_start > subwarpsize)
    {
      unsigned int final_merge_start = 0;
      unsigned int final_merge_end   = 0;

      // merge to temporary scratchpad memory:
      unsigned int *subwarp_scratchpad = subwarp_scratchpad_start;
      unsigned int iter = 0;
      while (row_A_end > row_A_start)
      {
        unsigned int my_row_B = row_A_start + id_in_warp;
        unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
        unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index) : 0;
        unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;

        unsigned int nnz_in_merge = merge_subwarp_symbolic_double(row_B_start, row_B_end, B_col_indices, B_size2,
                                                                  subwarp_scratchpad, id_in_warp, subwarpsize);

        final_merge_start = (iter == id_in_warp) ? subwarp_scratchpad - scratchpad_indices : final_merge_start;
        final_merge_end   = (iter == id_in_warp) ? final_merge_start + nnz_in_merge        : final_merge_end;
        ++iter;

        row_A_start += subwarpsize;
        subwarp_scratchpad += scratchpad_rowlength; // write to next row in scratchpad
      }

      // final merge:
      unsigned int num_nnz = merge_subwarp_symbolic(final_merge_start, final_merge_end, scratchpad_indices, B_size2, subwarpsize);

      if (id_in_warp == 0)
        C_row_indices[row] = num_nnz;
    }
    else
    {
      // single merge
      unsigned int my_row_B = row_A_start + id_in_warp;
      unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
      unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index) : 0;
      unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;

      unsigned int num_nnz = merge_subwarp_symbolic(row_B_start, row_B_end, B_col_indices, B_size2, subwarpsize);

      if (id_in_warp == 0)
        C_row_indices[row] = num_nnz;
    }
  }

}


//
// Stage 3: Fill C with values
//
template<typename NumericT>
__device__ unsigned int merge_subwarp_numeric(NumericT scaling_factor,
                                              unsigned int input_start, unsigned int input_end, const unsigned int *input_indices, const NumericT *input_values, unsigned int invalid_token,
                                              unsigned int *output_indices, NumericT *output_values,
                                              unsigned int id_in_warp, unsigned int subwarpsize)
{
  unsigned int current_front_index = (input_start < input_end) ? load_and_cache(input_indices + input_start) : invalid_token;
  NumericT     current_front_value = (input_start < input_end) ? load_and_cache(input_values  + input_start) : 0;

  unsigned int index_buffer = 0;
  NumericT     value_buffer = 0;
  unsigned int buffer_size = 0;
  unsigned int nnz_written = 0;
  while (1)
  {
    // determine current minimum:
    unsigned int min_index = current_front_index;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      min_index = min(min_index, __shfl_xor((int)min_index, (int)i));

    if (min_index == invalid_token) // done
      break;

    // compute entry in C:
    NumericT output_value = (current_front_index == min_index) ? scaling_factor * current_front_value : 0;
    for (unsigned int i = subwarpsize/2; i >= 1; i /= 2)
      output_value += __shfl_xor((int)output_value, (int)i);

    // update front:
    if (current_front_index == min_index)
    {
      ++input_start;
      current_front_index = (input_start < input_end) ? load_and_cache(input_indices + input_start) : invalid_token;
      current_front_value = (input_start < input_end) ? load_and_cache(input_values  + input_start) : 0;
    }

    // write current front to register buffer:
    index_buffer = (id_in_warp == buffer_size) ? min_index    : index_buffer;
    value_buffer = (id_in_warp == buffer_size) ? output_value : value_buffer;
    ++buffer_size;

    // flush register buffer via a coalesced write once full:
    if (buffer_size == subwarpsize)
    {
      output_indices[id_in_warp] = index_buffer; output_indices += subwarpsize;
      output_values[id_in_warp]  = value_buffer; output_values  += subwarpsize;
      buffer_size = 0;
    }

    ++nnz_written;
  }

  // write remaining entries in register buffer to C:
  if (id_in_warp < buffer_size)
  {
    output_indices[id_in_warp] = index_buffer;
    output_values[id_in_warp]  = value_buffer;
  }

  return nnz_written;
}

template<typename IndexT, typename NumericT>
__global__ void compressed_matrix_gemm_stage_3(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          const NumericT * A_elements,
          IndexT A_size1,
          const IndexT * B_row_indices,
          const IndexT * B_col_indices,
          const NumericT * B_elements,
          IndexT B_size2,
          IndexT const * C_row_indices,
          IndexT * C_col_indices,
          NumericT * C_elements,
          unsigned int *subwarpsize_array,
          unsigned int *max_row_size_A,
          unsigned int *max_row_size_B,
          unsigned int *scratchpad_offsets,
          unsigned int *scratchpad_indices,
          NumericT *scratchpad_values)
{
  unsigned int subwarpsize = subwarpsize_array[blockIdx.x];

  unsigned int num_warps  =  blockDim.x / subwarpsize;
  unsigned int warp_id    = threadIdx.x / subwarpsize;
  unsigned int id_in_warp = threadIdx.x % subwarpsize;

  unsigned int scratchpad_rowlength     = max_row_size_B[blockIdx.x] * subwarpsize;
  unsigned int scratchpad_rows_per_warp = max_row_size_A[blockIdx.x] / subwarpsize + 1;
  unsigned int subwarp_scratchpad_shift = scratchpad_offsets[blockIdx.x] + warp_id * scratchpad_rows_per_warp * scratchpad_rowlength;

  unsigned int rows_per_group = (A_size1 - 1) / gridDim.x + 1;
  unsigned int row_per_group_end = min(A_size1, rows_per_group * (blockIdx.x + 1));

  for (unsigned int row = rows_per_group * blockIdx.x + warp_id; row < row_per_group_end; row += num_warps)
  {
    unsigned int row_A_start = A_row_indices[row];
    unsigned int row_A_end   = A_row_indices[row+1];

    if (row_A_end - row_A_start > subwarpsize)
    {
      // first merge stage:
      unsigned int final_merge_start = 0;
      unsigned int final_merge_end = 0;
      unsigned int iter = 0;
      unsigned int *scratchpad_indices_ptr = scratchpad_indices + subwarp_scratchpad_shift;
      NumericT     *scratchpad_values_ptr  = scratchpad_values  + subwarp_scratchpad_shift;

      while (row_A_start < row_A_end)
      {
        unsigned int my_row_B = row_A_start + id_in_warp;
        unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
        unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index)     : 0;
        unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;
        NumericT val_A = (my_row_B < row_A_end) ? A_elements[my_row_B] : 0;

        unsigned int nnz_written = merge_subwarp_numeric(val_A,
                                                         row_B_start, row_B_end, B_col_indices, B_elements, B_size2,
                                                         scratchpad_indices_ptr, scratchpad_values_ptr,
                                                         id_in_warp, subwarpsize);

        if (iter == id_in_warp)
        {
          final_merge_start = scratchpad_indices_ptr - scratchpad_indices;
          final_merge_end   = final_merge_start + nnz_written;
        }
        ++iter;

        row_A_start += subwarpsize;
        scratchpad_indices_ptr += scratchpad_rowlength;
        scratchpad_values_ptr  += scratchpad_rowlength;
      }

      // second merge stage:
      unsigned int index_in_C = C_row_indices[row];
      merge_subwarp_numeric(NumericT(1),
                            final_merge_start, final_merge_end, scratchpad_indices, scratchpad_values, B_size2,
                            C_col_indices + index_in_C, C_elements + index_in_C,
                            id_in_warp, subwarpsize);
    }
    else
    {
      unsigned int my_row_B = row_A_start + id_in_warp;
      unsigned int row_B_index = (my_row_B < row_A_end) ? A_col_indices[my_row_B] : 0;
      unsigned int row_B_start = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index)     : 0;
      unsigned int row_B_end   = (my_row_B < row_A_end) ? load_and_cache(B_row_indices + row_B_index + 1) : 0;
      NumericT val_A = (my_row_B < row_A_end) ? A_elements[my_row_B] : 0;

      unsigned int index_in_C = C_row_indices[row];

      merge_subwarp_numeric(val_A,
                            row_B_start, row_B_end, B_col_indices, B_elements, B_size2,
                            C_col_indices + index_in_C, C_elements + index_in_C,
                            id_in_warp, subwarpsize);
    }
  }

}




//
// Decomposition kernels:
//
template<typename IndexT>
__global__ void compressed_matrix_gemm_decompose_1(
          const IndexT * A_row_indices,
          IndexT A_size1,
          IndexT max_per_row,
          IndexT *chunks_per_row)
{
  for (IndexT i = blockIdx.x * blockDim.x + threadIdx.x; i < A_size1; i += blockDim.x * gridDim.x)
  {
    IndexT num_entries = A_row_indices[i+1] - A_row_indices[i];
    chunks_per_row[i] = (num_entries < max_per_row) ? 1 : ((num_entries - 1)/ max_per_row + 1);
  }
}


template<typename IndexT, typename NumericT>
__global__ void compressed_matrix_gemm_A2(
          IndexT * A2_row_indices,
          IndexT * A2_col_indices,
          NumericT * A2_elements,
          IndexT A2_size1,
          IndexT *new_row_buffer)
{
  for (IndexT i = blockIdx.x * blockDim.x + threadIdx.x; i < A2_size1; i += blockDim.x * gridDim.x)
  {
    unsigned int index_start = new_row_buffer[i];
    unsigned int index_stop  = new_row_buffer[i+1];

    A2_row_indices[i] = index_start;

    for (IndexT j = index_start; j < index_stop; ++j)
    {
      A2_col_indices[j] = j;
      A2_elements[j] = NumericT(1);
    }
  }

  // write last entry in row_buffer with global thread 0:
  if (threadIdx.x == 0 && blockIdx.x == 0)
    A2_row_indices[A2_size1] = new_row_buffer[A2_size1];
}

template<typename IndexT, typename NumericT>
__global__ void compressed_matrix_gemm_G1(
          IndexT * G1_row_indices,
          IndexT * G1_col_indices,
          NumericT * G1_elements,
          IndexT G1_size1,
          IndexT const *A_row_indices,
          IndexT const *A_col_indices,
          NumericT const *A_elements,
          IndexT A_size1,
          IndexT A_nnz,
          IndexT max_per_row,
          IndexT *new_row_buffer)
{
  // Part 1: Copy column indices and entries:
  for (IndexT i = blockIdx.x * blockDim.x + threadIdx.x; i < A_nnz; i += blockDim.x * gridDim.x)
  {
    G1_col_indices[i] = A_col_indices[i];
    G1_elements[i]    = A_elements[i];
  }

  // Part 2: Derive new row indicies:
  for (IndexT i = blockIdx.x * blockDim.x + threadIdx.x; i < A_size1; i += blockDim.x * gridDim.x)
  {
    unsigned int old_start = A_row_indices[i];
    unsigned int new_start = new_row_buffer[i];
    unsigned int row_chunks = new_row_buffer[i+1] - new_start;

    for (IndexT j=0; j<row_chunks; ++j)
      G1_row_indices[new_start + j] = old_start + j * max_per_row;
  }

  // write last entry in row_buffer with global thread 0:
  if (threadIdx.x == 0 && blockIdx.x == 0)
    G1_row_indices[G1_size1] = A_row_indices[A_size1];
}



/** @brief Carries out sparse_matrix-sparse_matrix multiplication for CSR matrices
*
* Implementation of the convenience expression C = prod(A, B);
* Based on computing C(i, :) = A(i, :) * B via merging the respective rows of B
*
* @param A     Left factor
* @param B     Right factor
* @param C     Result matrix
*/
template<class NumericT, unsigned int AlignmentV>
void prod_impl(viennacl::compressed_matrix<NumericT, AlignmentV> const & A,
               viennacl::compressed_matrix<NumericT, AlignmentV> const & B,
               viennacl::compressed_matrix<NumericT, AlignmentV> & C)
{
  C.resize(A.size1(), B.size2(), false);

  unsigned int blocknum = 256;
  unsigned int threadnum = 128;

  viennacl::vector<unsigned int> subwarp_sizes(blocknum, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group
  viennacl::vector<unsigned int> max_nnz_row_A(blocknum, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group
  viennacl::vector<unsigned int> max_nnz_row_B(blocknum, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

#ifdef VIENNACL_WITH_SPGEMM_CUDA_TIMINGS
  viennacl::tools::timer timer;
#endif

  //
  // Stage 1: Determine upper bound for number of nonzeros
  //
#ifdef VIENNACL_WITH_SPGEMM_CUDA_TIMINGS
  cudaDeviceSynchronize();
  timer.start();
#endif

  compressed_matrix_gemm_stage_1<<<blocknum, threadnum>>>(viennacl::cuda_arg<unsigned int>(A.handle1()),
                                                          viennacl::cuda_arg<unsigned int>(A.handle2()),
                                                          static_cast<unsigned int>(A.size1()),
                                                          viennacl::cuda_arg<unsigned int>(B.handle1()),
                                                          viennacl::cuda_arg(subwarp_sizes),
                                                          viennacl::cuda_arg(max_nnz_row_A),
                                                          viennacl::cuda_arg(max_nnz_row_B)
                                                         );
  VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_1");
#ifdef VIENNACL_WITH_SPGEMM_CUDA_TIMINGS
  cudaDeviceSynchronize();
  std::cout << "Stage 1 device: " << timer.get() << std::endl;
  timer.start();
#endif

  subwarp_sizes.switch_memory_context(viennacl::context(MAIN_MEMORY));
  unsigned int * subwarp_sizes_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(subwarp_sizes.handle());

  max_nnz_row_A.switch_memory_context(viennacl::context(MAIN_MEMORY));
  unsigned int const * max_nnz_row_A_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(max_nnz_row_A.handle());

  max_nnz_row_B.switch_memory_context(viennacl::context(MAIN_MEMORY));
  unsigned int const * max_nnz_row_B_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(max_nnz_row_B.handle());

  //std::cout << "Subwarp sizes: " << subwarp_sizes << std::endl;

  viennacl::vector<unsigned int> scratchpad_offsets(blocknum, viennacl::context(MAIN_MEMORY)); // upper bound for the nonzeros per row encountered for each work group
  unsigned int * scratchpad_offsets_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(scratchpad_offsets.handle());

  unsigned int max_subwarp_size = 0;
  unsigned int A_max_nnz_per_row = 0;
  unsigned int scratchpad_offset = 0;
  //std::cout << "Scratchpad offsets: " << std::endl;
  for (std::size_t i=0; i<subwarp_sizes.size(); ++i)
  {
    max_subwarp_size = std::max(max_subwarp_size, subwarp_sizes_ptr[i]);
    A_max_nnz_per_row = std::max(A_max_nnz_per_row, max_nnz_row_A_ptr[i]);

    scratchpad_offsets_ptr[i] = scratchpad_offset;
    //std::cout << scratchpad_offset << " (with " << (max_nnz_row_A_ptr[i] / subwarp_sizes_ptr[i] + 1) << " warp reloads per group at " << max_nnz_row_A_ptr[i] << " max rows, "
    //                                            << upper_bound_nonzeros_per_row_C_ptr[i] << " row length, "
    //                                            << (256 / subwarp_sizes_ptr[i]) << " warps per group " << std::endl;
    unsigned int max_warp_reloads = max_nnz_row_A_ptr[i] / subwarp_sizes_ptr[i] + 1;
    unsigned int max_row_length_after_warp_merge = subwarp_sizes_ptr[i] * max_nnz_row_B_ptr[i];
    unsigned int warps_in_group = threadnum / subwarp_sizes_ptr[i];
    scratchpad_offset +=  max_warp_reloads
                        * max_row_length_after_warp_merge
                        * warps_in_group;
  }
  //std::cout << "Scratchpad memory for indices: " << scratchpad_offset << " entries (" << scratchpad_offset * sizeof(unsigned int) * 1e-6 << " MB)" << std::endl;

  if (max_subwarp_size > 32)
  {
    // determine augmented size:
    unsigned int max_entries_in_G = 1024;
    if (A_max_nnz_per_row <= 512*512)
      max_entries_in_G = 512;
    if (A_max_nnz_per_row <= 256*256)
      max_entries_in_G = 256;
    if (A_max_nnz_per_row <= 128*128)
      max_entries_in_G = 128;
    if (A_max_nnz_per_row <= 64*64)
      max_entries_in_G = 64;

    viennacl::vector<unsigned int> exclusive_scan_helper(A.size1() + 1, viennacl::traits::context(A));
    compressed_matrix_gemm_decompose_1<<<blocknum, threadnum>>>(viennacl::cuda_arg<unsigned int>(A.handle1()),
                                                                static_cast<unsigned int>(A.size1()),
                                                                static_cast<unsigned int>(max_entries_in_G),
                                                                viennacl::cuda_arg(exclusive_scan_helper)
                                                               );
    VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_decompose_1");

    thrust::exclusive_scan(thrust::device_ptr<unsigned int>(viennacl::cuda_arg(exclusive_scan_helper)),
                           thrust::device_ptr<unsigned int>(viennacl::cuda_arg(exclusive_scan_helper) + exclusive_scan_helper.size()),
                           thrust::device_ptr<unsigned int>(viennacl::cuda_arg(exclusive_scan_helper)));

    unsigned int augmented_size = exclusive_scan_helper[A.size1()];

    // split A = A2 * G1
    viennacl::compressed_matrix<NumericT, AlignmentV> A2(A.size1(), augmented_size, augmented_size, viennacl::traits::context(A));
    viennacl::compressed_matrix<NumericT, AlignmentV> G1(augmented_size, A.size2(),        A.nnz(), viennacl::traits::context(A));

    // fill A2:
    compressed_matrix_gemm_A2<<<blocknum, threadnum>>>(viennacl::cuda_arg<unsigned int>(A2.handle1()),
                                                       viennacl::cuda_arg<unsigned int>(A2.handle2()),
                                                       viennacl::cuda_arg<NumericT>(A2.handle()),
                                                       static_cast<unsigned int>(A2.size1()),
                                                       viennacl::cuda_arg(exclusive_scan_helper)
                                                      );
    VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_A2");

    // fill G1:
    compressed_matrix_gemm_G1<<<blocknum, threadnum>>>(viennacl::cuda_arg<unsigned int>(G1.handle1()),
                                                       viennacl::cuda_arg<unsigned int>(G1.handle2()),
                                                       viennacl::cuda_arg<NumericT>(G1.handle()),
                                                       static_cast<unsigned int>(G1.size1()),
                                                       viennacl::cuda_arg<unsigned int>(A.handle1()),
                                                       viennacl::cuda_arg<unsigned int>(A.handle2()),
                                                       viennacl::cuda_arg<NumericT>(A.handle()),
                                                       static_cast<unsigned int>(A.size1()),
                                                       static_cast<unsigned int>(A.nnz()),
                                                       static_cast<unsigned int>(max_entries_in_G),
                                                       viennacl::cuda_arg(exclusive_scan_helper)
                                                      );
    VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_G1");

    // compute tmp = G1 * B;
    // C = A2 * tmp;
    viennacl::compressed_matrix<NumericT, AlignmentV> tmp(G1.size1(), B.size2(), 0, viennacl::traits::context(A));
    prod_impl(G1, B, tmp); // this runs a standard RMerge without decomposition of G1
    prod_impl(A2, tmp, C); // this may split A2 again
    return;
  }

  subwarp_sizes.switch_memory_context(viennacl::traits::context(A));
  max_nnz_row_A.switch_memory_context(viennacl::traits::context(A));
  max_nnz_row_B.switch_memory_context(viennacl::traits::context(A));
  scratchpad_offsets.switch_memory_context(viennacl::traits::context(A));

  viennacl::vector<unsigned int> scratchpad_indices(scratchpad_offset, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

#ifdef VIENNACL_WITH_SPGEMM_CUDA_TIMINGS
  std::cout << "Intermediate host stage: " << timer.get() << std::endl;
  timer.start();
#endif

  //
  // Stage 2: Determine pattern of C
  //

  compressed_matrix_gemm_stage_2<<<blocknum, threadnum>>>(viennacl::cuda_arg<unsigned int>(A.handle1()),
                                                         viennacl::cuda_arg<unsigned int>(A.handle2()),
                                                         static_cast<unsigned int>(A.size1()),
                                                         viennacl::cuda_arg<unsigned int>(B.handle1()),
                                                         viennacl::cuda_arg<unsigned int>(B.handle2()),
                                                         static_cast<unsigned int>(B.size2()),
                                                         viennacl::cuda_arg<unsigned int>(C.handle1()),
                                                         viennacl::cuda_arg(subwarp_sizes),
                                                         viennacl::cuda_arg(max_nnz_row_A),
                                                         viennacl::cuda_arg(max_nnz_row_B),
                                                         viennacl::cuda_arg(scratchpad_offsets),
                                                         viennacl::cuda_arg(scratchpad_indices)
                                                        );
  VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_2");
#ifdef VIENNACL_WITH_SPGEMM_CUDA_TIMINGS
  cudaDeviceSynchronize();
  std::cout << "Stage 2: " << timer.get() << std::endl;
  timer.start();
#endif


  // exclusive scan on C.handle1(), ultimately allowing to allocate remaining memory for C
  viennacl::backend::typesafe_host_array<unsigned int> row_buffer(C.handle1(), C.size1() + 1);
  viennacl::backend::memory_read(C.handle1(), 0, row_buffer.raw_size(), row_buffer.get());
  unsigned int current_offset = 0;
  for (std::size_t i=0; i<C.size1(); ++i)
  {
    unsigned int tmp = row_buffer[i];
    row_buffer.set(i, current_offset);
    current_offset += tmp;
  }
  row_buffer.set(C.size1(), current_offset);
  viennacl::backend::memory_write(C.handle1(), 0, row_buffer.raw_size(), row_buffer.get());


  //
  // Stage 3: Compute entries in C
  //
  C.reserve(current_offset, false);

  viennacl::vector<NumericT> scratchpad_values(scratchpad_offset, viennacl::traits::context(A)); // upper bound for the nonzeros per row encountered for each work group

#ifdef VIENNACL_WITH_SPGEMM_CUDA_TIMINGS
  std::cout << "Intermediate stage 2->3: " << timer.get() << std::endl;
  timer.start();
#endif

  compressed_matrix_gemm_stage_3<<<blocknum, threadnum>>>(viennacl::cuda_arg<unsigned int>(A.handle1()),
                                                          viennacl::cuda_arg<unsigned int>(A.handle2()),
                                                          viennacl::cuda_arg<NumericT>(A.handle()),
                                                          static_cast<unsigned int>(A.size1()),
                                                          viennacl::cuda_arg<unsigned int>(B.handle1()),
                                                          viennacl::cuda_arg<unsigned int>(B.handle2()),
                                                          viennacl::cuda_arg<NumericT>(B.handle()),
                                                          static_cast<unsigned int>(B.size2()),
                                                          viennacl::cuda_arg<unsigned int>(C.handle1()),
                                                          viennacl::cuda_arg<unsigned int>(C.handle2()),
                                                          viennacl::cuda_arg<NumericT>(C.handle()),
                                                          viennacl::cuda_arg(subwarp_sizes),
                                                          viennacl::cuda_arg(max_nnz_row_A),
                                                          viennacl::cuda_arg(max_nnz_row_B),
                                                          viennacl::cuda_arg(scratchpad_offsets),
                                                          viennacl::cuda_arg(scratchpad_indices),
                                                          viennacl::cuda_arg(scratchpad_values)
                                                         );
  VIENNACL_CUDA_LAST_ERROR_CHECK("compressed_matrix_gemm_stage_3");
#ifdef VIENNACL_WITH_SPGEMM_CUDA_TIMINGS
  cudaDeviceSynchronize();
  std::cout << "Stage 3: " << timer.get() << std::endl;
  std::cout << "----------" << std::endl;
#endif

}

} // namespace cuda
} //namespace linalg
} //namespace viennacl


#endif
