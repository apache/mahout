#ifndef VIENNACL_LINALG_CUDA_BISECT_KERNEL_LARGE_MULTI_HPP_
#define VIENNACL_LINALG_CUDA_BISECT_KERNEL_LARGE_MULTI_HPP_

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


/** @file viennacl/linalg/cuda/bisect_kernel_large_multi.hpp
    @brief Second step of the bisection algorithm for the computation of eigenvalues for large matrices.

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/

/* Perform second step of bisection algorithm for large matrices for
 * intervals that contained after the first step more than one eigenvalue
 */

// includes, project
#include "viennacl/linalg/detail/bisect/config.hpp"
#include "viennacl/linalg/detail/bisect/util.hpp"
// additional kernel
#include "viennacl/linalg/cuda/bisect_util.hpp"

namespace viennacl
{
namespace linalg
{
namespace cuda
{
////////////////////////////////////////////////////////////////////////////////
//! Perform second step of bisection algorithm for large matrices for
//! intervals that after the first step contained more than one eigenvalue
//! @param  g_d  diagonal elements of symmetric, tridiagonal matrix
//! @param  g_s  superdiagonal elements of symmetric, tridiagonal matrix
//! @param  n    matrix size
//! @param  blocks_mult  start addresses of blocks of intervals that are
//!                      processed by one block of threads, each of the
//!                      intervals contains more than one eigenvalue
//! @param  blocks_mult_sum  total number of eigenvalues / singleton intervals
//!                          in one block of intervals
//! @param  g_left  left limits of intervals
//! @param  g_right  right limits of intervals
//! @param  g_left_count  number of eigenvalues less than left limits
//! @param  g_right_count  number of eigenvalues less than right limits
//! @param  g_lambda  final eigenvalue
//! @param  g_pos  index of eigenvalue (in ascending order)
//! @param  precision  desired precision of eigenvalues
////////////////////////////////////////////////////////////////////////////////
template<typename NumericT>
__global__
void
bisectKernelLarge_MultIntervals(const NumericT *g_d, const NumericT *g_s, const unsigned int n,
                                unsigned int *blocks_mult,
                                unsigned int *blocks_mult_sum,
                                NumericT *g_left, NumericT *g_right,
                                unsigned int *g_left_count,
                                unsigned int *g_right_count,
                                NumericT *g_lambda, unsigned int *g_pos,
                                NumericT precision
                               )
{
  const unsigned int tid = threadIdx.x;

    // left and right limits of interval
    __shared__  NumericT  s_left[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK];
    __shared__  NumericT  s_right[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK];

    // number of eigenvalues smaller than interval limits
    __shared__  unsigned int  s_left_count[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK];
    __shared__  unsigned int  s_right_count[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK];

    // helper array for chunk compaction of second chunk
    __shared__  unsigned int  s_compaction_list[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK + 1];
    // compaction list helper for exclusive scan
    unsigned int *s_compaction_list_exc = s_compaction_list + 1;

    // flag if all threads are converged
    __shared__  unsigned int  all_threads_converged;
    // number of active threads
    __shared__  unsigned int  num_threads_active;
    // number of threads to employ for compaction
    __shared__  unsigned int  num_threads_compaction;
    // flag if second chunk has to be compacted
    __shared__  unsigned int compact_second_chunk;

    // parameters of block of intervals processed by this block of threads
    __shared__  unsigned int  c_block_start;
    __shared__  unsigned int  c_block_end;
    __shared__  unsigned int  c_block_offset_output;

    // midpoint of currently active interval of the thread
    NumericT mid = 0.0f;
    // number of eigenvalues smaller than \a mid
    unsigned int  mid_count = 0;
    // current interval parameter
    NumericT  left = 0.0f;
    NumericT  right = 0.0f;
    unsigned int  left_count = 0;
    unsigned int  right_count = 0;
    // helper for compaction, keep track which threads have a second child
    unsigned int  is_active_second = 0;


    __syncthreads();
    // initialize common start conditions
    if (0 == tid)
    {

        c_block_start = blocks_mult[blockIdx.x];
        c_block_end = blocks_mult[blockIdx.x + 1];
        c_block_offset_output = blocks_mult_sum[blockIdx.x];


        num_threads_active = c_block_end - c_block_start;
        s_compaction_list[0] = 0;
        num_threads_compaction = ceilPow2(num_threads_active);

        all_threads_converged = 1;
        compact_second_chunk = 0;
    }

     s_left_count [tid] = 42;
     s_right_count[tid] = 42;
     s_left_count [tid + VIENNACL_BISECT_MAX_THREADS_BLOCK] = 0;
     s_right_count[tid + VIENNACL_BISECT_MAX_THREADS_BLOCK] = 0;

    __syncthreads();


    // read data into shared memory
    if (tid < num_threads_active)
    {
        s_left[tid]  = g_left[c_block_start + tid];
        s_right[tid] = g_right[c_block_start + tid];
        s_left_count[tid]  = g_left_count[c_block_start + tid];
        s_right_count[tid] = g_right_count[c_block_start + tid];
    }

    __syncthreads();
    unsigned int iter = 0;
    // do until all threads converged
    while (true)
    {
        iter++;
        //for (int iter=0; iter < 0; iter++) {
        s_compaction_list[threadIdx.x] = 0;
        s_compaction_list[threadIdx.x + blockDim.x] = 0;
        s_compaction_list[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK] = 0;

        // subdivide interval if currently active and not already converged
        subdivideActiveIntervalMulti(tid, s_left, s_right,
                                s_left_count, s_right_count,
                                num_threads_active,
                                left, right, left_count, right_count,
                                mid, all_threads_converged);
        __syncthreads();

        // stop if all eigenvalues have been found
        if (1 == all_threads_converged)
        {

            break;
        }

        // compute number of eigenvalues smaller than mid for active and not
        // converged intervals, use all threads for loading data from gmem and
        // s_left and s_right as scratch space to store the data load from gmem
        // in shared memory
        mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n,
                                                    mid, tid, num_threads_active,
                                                    s_left, s_right,
                                                    (left == right));

        __syncthreads();

        if (tid < num_threads_active)
        {

            // store intervals
            if (left != right)
            {

                storeNonEmptyIntervals(tid, num_threads_active,
                                       s_left, s_right, s_left_count, s_right_count,
                                       left, mid, right,
                                       left_count, mid_count, right_count,
                                       precision, compact_second_chunk,
                                       s_compaction_list_exc,
                                       is_active_second);

            }
            else
            {

                storeIntervalConverged(s_left, s_right, s_left_count, s_right_count,
                                       left, mid, right,
                                       left_count, mid_count, right_count,
                                       s_compaction_list_exc, compact_second_chunk,
                                       num_threads_active,
                                       is_active_second);

            }
        }

        __syncthreads();

        // compact second chunk of intervals if any of the threads generated
        // two child intervals
        if (1 == compact_second_chunk)
        {

            createIndicesCompaction(s_compaction_list_exc, num_threads_compaction);
            compactIntervals(s_left, s_right, s_left_count, s_right_count,
                             mid, right, mid_count, right_count,
                             s_compaction_list, num_threads_active,
                             is_active_second);
        }

        __syncthreads();

        // update state variables
        if (0 == tid)
        {
            num_threads_active += s_compaction_list[num_threads_active];
            num_threads_compaction = ceilPow2(num_threads_active);

            compact_second_chunk = 0;
            all_threads_converged = 1;
        }

        __syncthreads();

        // clear
        s_compaction_list_exc[threadIdx.x] = 0;
        s_compaction_list_exc[threadIdx.x + blockDim.x] = 0;

        if (num_threads_compaction > blockDim.x)
        {
          break;
        }


        __syncthreads();

    }  // end until all threads converged

    // write data back to global memory
    if (tid < num_threads_active)
    {

        unsigned int addr = c_block_offset_output + tid;

        g_lambda[addr]  = s_left[tid];
        g_pos[addr]   = s_right_count[tid];
    }
}
} // namespace cuda
} // namespace linalg
} // namespace viennacl

#endif // #ifndef VIENNACL_LINALG_CUDA_BISECT_KERNEL_LARGE_MULTI_HPP_
