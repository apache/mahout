#ifndef VIENNACL_LINALG_CUDA_SPARSE_MATRIX_OPERATIONS_SOLVE_HPP_
#define VIENNACL_LINALG_CUDA_SPARSE_MATRIX_OPERATIONS_SOLVE_HPP_

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

/** @file viennacl/linalg/cuda/sparse_matrix_operations_solve.hpp
    @brief Implementations of direct triangular solvers for sparse matrices using CUDA
*/

#include "viennacl/forwards.h"

namespace viennacl
{
namespace linalg
{
namespace cuda
{
//
// Compressed matrix
//

//
// non-transposed
//

template<typename NumericT>
__global__ void csr_unit_lu_forward_kernel(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
                NumericT * vector,
          unsigned int size)
{
  __shared__  unsigned int col_index_buffer[128];
  __shared__  NumericT element_buffer[128];
  __shared__  NumericT vector_buffer[128];

  unsigned int nnz = row_indices[size];
  unsigned int current_row = 0;
  unsigned int row_at_window_start = 0;
  NumericT current_vector_entry = vector[0];
  unsigned int loop_end = (nnz / blockDim.x + 1) * blockDim.x;
  unsigned int next_row = row_indices[1];

  for (unsigned int i = threadIdx.x; i < loop_end; i += blockDim.x)
  {
    //load into shared memory (coalesced access):
    if (i < nnz)
    {
      element_buffer[threadIdx.x] = elements[i];
      unsigned int tmp = column_indices[i];
      col_index_buffer[threadIdx.x] = tmp;
      vector_buffer[threadIdx.x] = vector[tmp];
    }

    __syncthreads();

    //now a single thread does the remaining work in shared memory:
    if (threadIdx.x == 0)
    {
      // traverse through all the loaded data:
      for (unsigned int k=0; k<blockDim.x; ++k)
      {
        if (current_row < size && i+k == next_row) //current row is finished. Write back result
        {
          vector[current_row] = current_vector_entry;
          ++current_row;
          if (current_row < size) //load next row's data
          {
            next_row = row_indices[current_row+1];
            current_vector_entry = vector[current_row];
          }
        }

        if (current_row < size && col_index_buffer[k] < current_row) //substitute
        {
          if (col_index_buffer[k] < row_at_window_start) //use recently computed results
            current_vector_entry -= element_buffer[k] * vector_buffer[k];
          else if (col_index_buffer[k] < current_row) //use buffered data
            current_vector_entry -= element_buffer[k] * vector[col_index_buffer[k]];
        }

      } // for k

      row_at_window_start = current_row;
    } // if (get_local_id(0) == 0)

    __syncthreads();
  } //for i
}



template<typename NumericT>
__global__ void csr_lu_forward_kernel(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
                NumericT * vector,
          unsigned int size)
{
  __shared__  unsigned int col_index_buffer[128];
  __shared__  NumericT element_buffer[128];
  __shared__  NumericT vector_buffer[128];

  unsigned int nnz = row_indices[size];
  unsigned int current_row = 0;
  unsigned int row_at_window_start = 0;
  NumericT current_vector_entry = vector[0];
  NumericT diagonal_entry = 0;
  unsigned int loop_end = (nnz / blockDim.x + 1) * blockDim.x;
  unsigned int next_row = row_indices[1];

  for (unsigned int i = threadIdx.x; i < loop_end; i += blockDim.x)
  {
    //load into shared memory (coalesced access):
    if (i < nnz)
    {
      element_buffer[threadIdx.x] = elements[i];
      unsigned int tmp = column_indices[i];
      col_index_buffer[threadIdx.x] = tmp;
      vector_buffer[threadIdx.x] = vector[tmp];
    }

    __syncthreads();

    //now a single thread does the remaining work in shared memory:
    if (threadIdx.x == 0)
    {
      // traverse through all the loaded data:
      for (unsigned int k=0; k<blockDim.x; ++k)
      {
        if (current_row < size && i+k == next_row) //current row is finished. Write back result
        {
          vector[current_row] = current_vector_entry / diagonal_entry;
          ++current_row;
          if (current_row < size) //load next row's data
          {
            next_row = row_indices[current_row+1];
            current_vector_entry = vector[current_row];
          }
        }

        if (current_row < size && col_index_buffer[k] < current_row) //substitute
        {
          if (col_index_buffer[k] < row_at_window_start) //use recently computed results
            current_vector_entry -= element_buffer[k] * vector_buffer[k];
          else if (col_index_buffer[k] < current_row) //use buffered data
            current_vector_entry -= element_buffer[k] * vector[col_index_buffer[k]];
        }
        else if (col_index_buffer[k] == current_row)
          diagonal_entry = element_buffer[k];

      } // for k

      row_at_window_start = current_row;
    } // if (get_local_id(0) == 0)

    __syncthreads();
  } //for i
}


template<typename NumericT>
__global__ void csr_unit_lu_backward_kernel(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
                NumericT * vector,
          unsigned int size)
{
  __shared__  unsigned int col_index_buffer[128];
  __shared__  NumericT element_buffer[128];
  __shared__  NumericT vector_buffer[128];

  unsigned int nnz = row_indices[size];
  unsigned int current_row = size-1;
  unsigned int row_at_window_start = size-1;
  NumericT current_vector_entry = vector[size-1];
  unsigned int loop_end = ( (nnz - 1) / blockDim.x) * blockDim.x;
  unsigned int next_row = row_indices[size-1];

  unsigned int i = loop_end + threadIdx.x;
  while (1)
  {
    //load into shared memory (coalesced access):
    if (i < nnz)
    {
      element_buffer[threadIdx.x] = elements[i];
      unsigned int tmp = column_indices[i];
      col_index_buffer[threadIdx.x] = tmp;
      vector_buffer[threadIdx.x] = vector[tmp];
    }

    __syncthreads();

    //now a single thread does the remaining work in shared memory:
    if (threadIdx.x == 0)
    {
      // traverse through all the loaded data from back to front:
      for (unsigned int k2=0; k2<blockDim.x; ++k2)
      {
        unsigned int k = (blockDim.x - k2) - 1;

        if (i+k >= nnz)
          continue;

        if (col_index_buffer[k] > row_at_window_start) //use recently computed results
          current_vector_entry -= element_buffer[k] * vector_buffer[k];
        else if (col_index_buffer[k] > current_row) //use buffered data
          current_vector_entry -= element_buffer[k] * vector[col_index_buffer[k]];

        if (i+k == next_row) //current row is finished. Write back result
        {
          vector[current_row] = current_vector_entry;
          if (current_row > 0) //load next row's data
          {
            --current_row;
            next_row = row_indices[current_row];
            current_vector_entry = vector[current_row];
          }
        }


      } // for k

      row_at_window_start = current_row;
    } // if (get_local_id(0) == 0)

    __syncthreads();

    if (i < blockDim.x)
      break;

    i -= blockDim.x;
  } //for i
}



template<typename NumericT>
__global__ void csr_lu_backward_kernel(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
                NumericT * vector,
          unsigned int size)
{
  __shared__  unsigned int col_index_buffer[128];
  __shared__  NumericT element_buffer[128];
  __shared__  NumericT vector_buffer[128];

  unsigned int nnz = row_indices[size];
  unsigned int current_row = size-1;
  unsigned int row_at_window_start = size-1;
  NumericT current_vector_entry = vector[size-1];
  NumericT diagonal_entry;
  unsigned int loop_end = ( (nnz - 1) / blockDim.x) * blockDim.x;
  unsigned int next_row = row_indices[size-1];

  unsigned int i = loop_end + threadIdx.x;
  while (1)
  {
    //load into shared memory (coalesced access):
    if (i < nnz)
    {
      element_buffer[threadIdx.x] = elements[i];
      unsigned int tmp = column_indices[i];
      col_index_buffer[threadIdx.x] = tmp;
      vector_buffer[threadIdx.x] = vector[tmp];
    }

    __syncthreads();

    //now a single thread does the remaining work in shared memory:
    if (threadIdx.x == 0)
    {
      // traverse through all the loaded data from back to front:
      for (unsigned int k2=0; k2<blockDim.x; ++k2)
      {
        unsigned int k = (blockDim.x - k2) - 1;

        if (i+k >= nnz)
          continue;

        if (col_index_buffer[k] > row_at_window_start) //use recently computed results
          current_vector_entry -= element_buffer[k] * vector_buffer[k];
        else if (col_index_buffer[k] > current_row) //use buffered data
          current_vector_entry -= element_buffer[k] * vector[col_index_buffer[k]];
        else if (col_index_buffer[k] == current_row)
          diagonal_entry = element_buffer[k];

        if (i+k == next_row) //current row is finished. Write back result
        {
          vector[current_row] = current_vector_entry / diagonal_entry;
          if (current_row > 0) //load next row's data
          {
            --current_row;
            next_row = row_indices[current_row];
            current_vector_entry = vector[current_row];
          }
        }


      } // for k

      row_at_window_start = current_row;
    } // if (get_local_id(0) == 0)

    __syncthreads();

    if (i < blockDim.x)
      break;

    i -= blockDim.x;
  } //for i
}



//
// transposed
//


template<typename NumericT>
__global__ void csr_trans_lu_forward_kernel2(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
                NumericT * vector,
          unsigned int size)
{
  for (unsigned int row = 0; row < size; ++row)
  {
    NumericT result_entry = vector[row];

    unsigned int row_start = row_indices[row];
    unsigned int row_stop  = row_indices[row + 1];
    for (unsigned int entry_index = row_start + threadIdx.x; entry_index < row_stop; entry_index += blockDim.x)
    {
      unsigned int col_index = column_indices[entry_index];
      if (col_index > row)
        vector[col_index] -= result_entry * elements[entry_index];
    }

    __syncthreads();
  }
}

template<typename NumericT>
__global__ void csr_trans_unit_lu_forward_kernel(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
                NumericT * vector,
          unsigned int size)
{
  __shared__  unsigned int row_index_lookahead[256];
  __shared__  unsigned int row_index_buffer[256];

  unsigned int row_index;
  unsigned int col_index;
  NumericT matrix_entry;
  unsigned int nnz = row_indices[size];
  unsigned int row_at_window_start = 0;
  unsigned int row_at_window_end = 0;
  unsigned int loop_end = ( (nnz - 1) / blockDim.x + 1) * blockDim.x;

  for (unsigned int i = threadIdx.x; i < loop_end; i += blockDim.x)
  {
    col_index    = (i < nnz) ? column_indices[i] : 0;
    matrix_entry = (i < nnz) ? elements[i]       : 0;
    row_index_lookahead[threadIdx.x] = (row_at_window_start + threadIdx.x < size) ? row_indices[row_at_window_start + threadIdx.x] : nnz;

    __syncthreads();

    if (i < nnz)
    {
      unsigned int row_index_inc = 0;
      while (i >= row_index_lookahead[row_index_inc + 1])
        ++row_index_inc;
      row_index = row_at_window_start + row_index_inc;
      row_index_buffer[threadIdx.x] = row_index;
    }
    else
    {
      row_index = size+1;
      row_index_buffer[threadIdx.x] = size - 1;
    }

    __syncthreads();

    row_at_window_start = row_index_buffer[0];
    row_at_window_end   = row_index_buffer[blockDim.x - 1];

    //forward elimination
    for (unsigned int row = row_at_window_start; row <= row_at_window_end; ++row)
    {
      NumericT result_entry = vector[row];

      if ( (row_index == row) && (col_index > row) )
        vector[col_index] -= result_entry * matrix_entry;

      __syncthreads();
    }

    row_at_window_start = row_at_window_end;
  }

}

template<typename NumericT>
__global__ void csr_trans_lu_forward_kernel(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
          const NumericT * diagonal_entries,
                NumericT * vector,
          unsigned int size)
{
  __shared__  unsigned int row_index_lookahead[256];
  __shared__  unsigned int row_index_buffer[256];

  unsigned int row_index;
  unsigned int col_index;
  NumericT matrix_entry;
  unsigned int nnz = row_indices[size];
  unsigned int row_at_window_start = 0;
  unsigned int row_at_window_end = 0;
  unsigned int loop_end = ( (nnz - 1) / blockDim.x + 1) * blockDim.x;

  for (unsigned int i = threadIdx.x; i < loop_end; i += blockDim.x)
  {
    col_index    = (i < nnz) ? column_indices[i] : 0;
    matrix_entry = (i < nnz) ? elements[i]       : 0;
    row_index_lookahead[threadIdx.x] = (row_at_window_start + threadIdx.x < size) ? row_indices[row_at_window_start + threadIdx.x] : nnz;

    __syncthreads();

    if (i < nnz)
    {
      unsigned int row_index_inc = 0;
      while (i >= row_index_lookahead[row_index_inc + 1])
        ++row_index_inc;
      row_index = row_at_window_start + row_index_inc;
      row_index_buffer[threadIdx.x] = row_index;
    }
    else
    {
      row_index = size+1;
      row_index_buffer[threadIdx.x] = size - 1;
    }

    __syncthreads();

    row_at_window_start = row_index_buffer[0];
    row_at_window_end   = row_index_buffer[blockDim.x - 1];

    //forward elimination
    for (unsigned int row = row_at_window_start; row <= row_at_window_end; ++row)
    {
      NumericT result_entry = vector[row] / diagonal_entries[row];

      if ( (row_index == row) && (col_index > row) )
        vector[col_index] -= result_entry * matrix_entry;

      __syncthreads();
    }

    row_at_window_start = row_at_window_end;
  }

  // final step: Divide vector by diagonal entries:
  for (unsigned int i = threadIdx.x; i < size; i += blockDim.x)
    vector[i] /= diagonal_entries[i];

}


template<typename NumericT>
__global__ void csr_trans_unit_lu_backward_kernel(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
                NumericT * vector,
          unsigned int size)
{
  __shared__  unsigned int row_index_lookahead[256];
  __shared__  unsigned int row_index_buffer[256];

  unsigned int row_index;
  unsigned int col_index;
  NumericT matrix_entry;
  unsigned int nnz = row_indices[size];
  unsigned int row_at_window_start = size;
  unsigned int row_at_window_end;
  unsigned int loop_end = ( (nnz - 1) / blockDim.x + 1) * blockDim.x;

  for (unsigned int i2 = threadIdx.x; i2 < loop_end; i2 += blockDim.x)
  {
    unsigned int i = (nnz - i2) - 1;
    col_index    = (i2 < nnz) ? column_indices[i] : 0;
    matrix_entry = (i2 < nnz) ? elements[i]       : 0;
    row_index_lookahead[threadIdx.x] = (row_at_window_start >= threadIdx.x) ? row_indices[row_at_window_start - threadIdx.x] : 0;

    __syncthreads();

    if (i2 < nnz)
    {
      unsigned int row_index_dec = 0;
      while (row_index_lookahead[row_index_dec] > i)
        ++row_index_dec;
      row_index = row_at_window_start - row_index_dec;
      row_index_buffer[threadIdx.x] = row_index;
    }
    else
    {
      row_index = size+1;
      row_index_buffer[threadIdx.x] = 0;
    }

    __syncthreads();

    row_at_window_start = row_index_buffer[0];
    row_at_window_end   = row_index_buffer[blockDim.x - 1];

    //backward elimination
    for (unsigned int row2 = 0; row2 <= (row_at_window_start - row_at_window_end); ++row2)
    {
      unsigned int row = row_at_window_start - row2;
      NumericT result_entry = vector[row];

      if ( (row_index == row) && (col_index < row) )
        vector[col_index] -= result_entry * matrix_entry;

      __syncthreads();
    }

    row_at_window_start = row_at_window_end;
  }

}



template<typename NumericT>
__global__ void csr_trans_lu_backward_kernel2(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
          const NumericT * diagonal_entries,
                NumericT * vector,
          unsigned int size)
{
  NumericT result_entry = 0;

  //backward elimination, using U and D:
  for (unsigned int row2 = 0; row2 < size; ++row2)
  {
    unsigned int row = (size - row2) - 1;
    result_entry = vector[row] / diagonal_entries[row];

    unsigned int row_start = row_indices[row];
    unsigned int row_stop  = row_indices[row + 1];
    for (unsigned int entry_index = row_start + threadIdx.x; entry_index < row_stop; ++entry_index)
    {
      unsigned int col_index = column_indices[entry_index];
      if (col_index < row)
        vector[col_index] -= result_entry * elements[entry_index];
    }

    __syncthreads();

    if (threadIdx.x == 0)
      vector[row] = result_entry;
  }
}


template<typename NumericT>
__global__ void csr_trans_lu_backward_kernel(
          const unsigned int * row_indices,
          const unsigned int * column_indices,
          const NumericT * elements,
          const NumericT * diagonal_entries,
                NumericT * vector,
          unsigned int size)
{
  __shared__  unsigned int row_index_lookahead[256];
  __shared__  unsigned int row_index_buffer[256];

  unsigned int row_index;
  unsigned int col_index;
  NumericT matrix_entry;
  unsigned int nnz = row_indices[size];
  unsigned int row_at_window_start = size;
  unsigned int row_at_window_end;
  unsigned int loop_end = ( (nnz - 1) / blockDim.x + 1) * blockDim.x;

  for (unsigned int i2 = threadIdx.x; i2 < loop_end; i2 += blockDim.x)
  {
    unsigned int i = (nnz - i2) - 1;
    col_index    = (i2 < nnz) ? column_indices[i] : 0;
    matrix_entry = (i2 < nnz) ? elements[i]       : 0;
    row_index_lookahead[threadIdx.x] = (row_at_window_start >= threadIdx.x) ? row_indices[row_at_window_start - threadIdx.x] : 0;

    __syncthreads();

    if (i2 < nnz)
    {
      unsigned int row_index_dec = 0;
      while (row_index_lookahead[row_index_dec] > i)
        ++row_index_dec;
      row_index = row_at_window_start - row_index_dec;
      row_index_buffer[threadIdx.x] = row_index;
    }
    else
    {
      row_index = size+1;
      row_index_buffer[threadIdx.x] = 0;
    }

    __syncthreads();

    row_at_window_start = row_index_buffer[0];
    row_at_window_end   = row_index_buffer[blockDim.x - 1];

    //backward elimination
    for (unsigned int row2 = 0; row2 <= (row_at_window_start - row_at_window_end); ++row2)
    {
      unsigned int row = row_at_window_start - row2;
      NumericT result_entry = vector[row] / diagonal_entries[row];

      if ( (row_index == row) && (col_index < row) )
        vector[col_index] -= result_entry * matrix_entry;

      __syncthreads();
    }

    row_at_window_start = row_at_window_end;
  }


  // final step: Divide vector by diagonal entries:
  for (unsigned int i = threadIdx.x; i < size; i += blockDim.x)
    vector[i] /= diagonal_entries[i];

}


template<typename NumericT>
__global__ void csr_block_trans_unit_lu_forward(
          const unsigned int * row_jumper_L,      //L part (note that L is transposed in memory)
          const unsigned int * column_indices_L,
          const NumericT * elements_L,
          const unsigned int * block_offsets,
          NumericT * result,
          unsigned int size)
{
  unsigned int col_start = block_offsets[2*blockIdx.x];
  unsigned int col_stop  = block_offsets[2*blockIdx.x+1];
  unsigned int row_start = row_jumper_L[col_start];
  unsigned int row_stop;
  NumericT result_entry = 0;

  if (col_start >= col_stop)
    return;

  //forward elimination, using L:
  for (unsigned int col = col_start; col < col_stop; ++col)
  {
    result_entry = result[col];
    row_stop = row_jumper_L[col + 1];
    for (unsigned int buffer_index = row_start + threadIdx.x; buffer_index < row_stop; buffer_index += blockDim.x)
      result[column_indices_L[buffer_index]] -= result_entry * elements_L[buffer_index];
    row_start = row_stop; //for next iteration (avoid unnecessary loads from GPU RAM)
    __syncthreads();
  }

}


template<typename NumericT>
__global__ void csr_block_trans_lu_backward(
          const unsigned int * row_jumper_U,      //U part (note that U is transposed in memory)
          const unsigned int * column_indices_U,
          const NumericT * elements_U,
          const NumericT * diagonal_U,
          const unsigned int * block_offsets,
          NumericT * result,
          unsigned int size)
{
  unsigned int col_start = block_offsets[2*blockIdx.x];
  unsigned int col_stop  = block_offsets[2*blockIdx.x+1];
  unsigned int row_start;
  unsigned int row_stop;
  NumericT result_entry = 0;

  if (col_start >= col_stop)
    return;

  //backward elimination, using U and diagonal_U
  for (unsigned int iter = 0; iter < col_stop - col_start; ++iter)
  {
    unsigned int col = (col_stop - iter) - 1;
    result_entry = result[col] / diagonal_U[col];
    row_start = row_jumper_U[col];
    row_stop  = row_jumper_U[col + 1];
    for (unsigned int buffer_index = row_start + threadIdx.x; buffer_index < row_stop; buffer_index += blockDim.x)
      result[column_indices_U[buffer_index]] -= result_entry * elements_U[buffer_index];
    __syncthreads();
  }

  //divide result vector by diagonal:
  for (unsigned int col = col_start + threadIdx.x; col < col_stop; col += blockDim.x)
    result[col] /= diagonal_U[col];
}



//
// Coordinate Matrix
//




//
// ELL Matrix
//



//
// Hybrid Matrix
//



} // namespace opencl
} //namespace linalg
} //namespace viennacl


#endif
