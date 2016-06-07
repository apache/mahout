#ifndef VIENNACL_LINALG_OPENCL_KERNELS_BISECT_HPP_
#define VIENNACL_LINALG_OPENCL_KERNELS_BISECT_HPP_

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


/** @file viennacl/linalg/opencl/kernels/bisect.hpp
    @brief OpenCL kernels for the bisection algorithm for eigenvalues

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/



#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

#include "viennacl/linalg/opencl/common.hpp"

// declaration, forward

namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{
  template <typename StringType>
  void generate_bisect_kernel_config(StringType & source)
  {
    /* Global configuration parameter */
    source.append("     #define  VIENNACL_BISECT_MAX_THREADS_BLOCK                256\n");
    source.append("     #define  VIENNACL_BISECT_MAX_SMALL_MATRIX                 256\n");
    source.append("     #define  VIENNACL_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX   256\n");
    source.append("     #define  VIENNACL_BISECT_MIN_ABS_INTERVAL                 5.0e-37\n");

  }

  ////////////////////////////////////////////////////////////////////////////////
  // Compute the next lower power of two of n
  // n    number for which next higher power of two is seeked
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_floorPow2(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     inline int  \n");
  source.append("     floorPow2(int n)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0);  \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");


      // early out if already power of two
  source.append("         if (0 == (n & (n-1)))  \n");
  source.append("         {  \n");
  source.append("             return n;  \n");
  source.append("         }  \n");

  source.append("         int exp;  \n");
  source.append("         frexp(( "); source.append(numeric_string); source.append(" )n, &exp);  \n");
  source.append("         return (1 << (exp - 1));  \n");
  source.append("     }  \n");

  }


  ////////////////////////////////////////////////////////////////////////////////
  // Compute the next higher power of two of n
  // n  number for which next higher power of two is seeked
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_ceilPow2(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     inline int  \n");
  source.append("     ceilPow2(int n)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");


      // early out if already power of two
  source.append("         if (0 == (n & (n-1)))  \n");
  source.append("         {  \n");
  source.append("             return n;  \n");
  source.append("         }  \n");

  source.append("         int exp;  \n");
  source.append("         frexp(( "); source.append(numeric_string); source.append(" )n, &exp);  \n");
  source.append("         return (1 << exp);  \n");
  source.append("     }  \n");
  }


  ////////////////////////////////////////////////////////////////////////////////
  // Compute midpoint of interval [\a left, \a right] avoiding overflow if possible
  //
  // left     left  / lower limit of interval
  // right    right / upper limit of interval
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_computeMidpoint(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     inline "); source.append(numeric_string); source.append(" \n");
  source.append("     computeMidpoint(const "); source.append(numeric_string); source.append(" left,\n");
  source.append("       const "); source.append(numeric_string); source.append("  right)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");
  source.append("          "); source.append(numeric_string); source.append("  mid;  \n");

  source.append("         if (sign(left) == sign(right))  \n");
  source.append("         {  \n");
  source.append("             mid = left + (right - left) * 0.5f;  \n");
  source.append("         }  \n");
  source.append("         else  \n");
  source.append("         {  \n");
  source.append("             mid = (left + right) * 0.5f;  \n");
  source.append("         }  \n");

  source.append("         return mid;  \n");
  source.append("     }  \n");

  }


  ////////////////////////////////////////////////////////////////////////////////
  // Check if interval converged and store appropriately
  //
  // addr           address where to store the information of the interval
  // s_left         shared memory storage for left interval limits
  // s_right        shared memory storage for right interval limits
  // s_left_count   shared memory storage for number of eigenvalues less than left interval limits
  // s_right_count  shared memory storage for number of eigenvalues less than right interval limits
  // left           lower limit of interval
  // right          upper limit of interval
  // left_count     eigenvalues less than \a left
  // right_count    eigenvalues less than \a right
  // precision      desired precision for eigenvalues
  ////////////////////////////////////////////////////////////////////////////////

  template<typename StringType>
  void generate_bisect_kernel_storeInterval(StringType & source, std::string const & numeric_string)
  {
  source.append("     \n");
  source.append("     void  \n");
  source.append("     storeInterval(unsigned int addr,  \n");
  source.append("                   __local "); source.append(numeric_string); source.append(" * s_left,   \n");
  source.append("                   __local "); source.append(numeric_string); source.append(" * s_right,  \n");
  source.append("                   __local unsigned int * s_left_count,  \n");
  source.append("                   __local unsigned int * s_right_count,  \n");
  source.append("                    "); source.append(numeric_string); source.append(" left,   \n");
  source.append("                    "); source.append(numeric_string); source.append(" right,  \n");
  source.append("                   unsigned int left_count, \n");
  source.append("                   unsigned int right_count,  \n");
  source.append("                    "); source.append(numeric_string); source.append("  precision)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");

  source.append("         s_left_count[addr] = left_count;  \n");
  source.append("         s_right_count[addr] = right_count;  \n");

      // check if interval converged
  source.append("          "); source.append(numeric_string); source.append(" t0 = fabs(right - left);  \n");
  source.append("          "); source.append(numeric_string); source.append(" t1 = max(fabs(left), fabs(right)) * precision;  \n");

  source.append("         if (t0 <= max(( "); source.append(numeric_string); source.append(" )VIENNACL_BISECT_MIN_ABS_INTERVAL, t1))  \n");
  source.append("         {  \n");
          // compute mid point
  source.append("              "); source.append(numeric_string); source.append(" lambda = computeMidpoint(left, right);  \n");

          // mark as converged
  source.append("             s_left[addr] = lambda;  \n");
  source.append("             s_right[addr] = lambda;  \n");
  source.append("         }  \n");
  source.append("         else  \n");
  source.append("         {  \n");

          // store current limits
  source.append("             s_left[addr] = left;  \n");
  source.append("             s_right[addr] = right;  \n");
  source.append("         }  \n");

  source.append("     }  \n");

  }

  template<typename StringType>
  void generate_bisect_kernel_storeIntervalShort(StringType & source, std::string const & numeric_string)
  {
  source.append("     \n");
  source.append("     void  \n");
  source.append("     storeIntervalShort(unsigned int addr,  \n");
  source.append("                   __local "); source.append(numeric_string); source.append(" * s_left,   \n");
  source.append("                   __local "); source.append(numeric_string); source.append(" * s_right,  \n");
  source.append("                   __local unsigned short * s_left_count,  \n");
  source.append("                   __local unsigned short * s_right_count,  \n");
  source.append("                    "); source.append(numeric_string); source.append(" left,   \n");
  source.append("                    "); source.append(numeric_string); source.append(" right,  \n");
  source.append("                   unsigned int left_count, \n");
  source.append("                   unsigned int right_count,  \n");
  source.append("                    "); source.append(numeric_string); source.append("  precision)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");

  source.append("         s_left_count[addr] = left_count;  \n");
  source.append("         s_right_count[addr] = right_count;  \n");

      // check if interval converged
  source.append("          "); source.append(numeric_string); source.append(" t0 = fabs(right - left);  \n");
  source.append("          "); source.append(numeric_string); source.append(" t1 = max(fabs(left), fabs(right)) * precision;  \n");

  source.append("         if (t0 <= max(( "); source.append(numeric_string); source.append(" )VIENNACL_BISECT_MIN_ABS_INTERVAL, t1))  \n");
  source.append("         {  \n");
          // compute mid point
  source.append("              "); source.append(numeric_string); source.append(" lambda = computeMidpoint(left, right);  \n");

          // mark as converged
  source.append("             s_left[addr] = lambda;  \n");
  source.append("             s_right[addr] = lambda;  \n");
  source.append("         }  \n");
  source.append("         else  \n");
  source.append("         {  \n");

          // store current limits
  source.append("             s_left[addr] = left;  \n");
  source.append("             s_right[addr] = right;  \n");
  source.append("         }  \n");

  source.append("     }  \n");


  }


  ////////////////////////////////////////////////////////////////////////////////
  // Compute number of eigenvalues that are smaller than x given a symmetric,
  // real, and tridiagonal matrix
  //
  // g_d                   diagonal elements stored in global memory
  // g_s                   superdiagonal elements stored in global memory
  // n                     size of matrix
  // x                     value for which the number of eigenvalues that are smaller is sought
  // tid                   thread identified (e.g. threadIdx.x or gtid)
  // num_intervals_active  number of active intervals / threads that currently process an interval
  // s_d                   scratch space to store diagonal entries of the tridiagonal matrix in shared memory
  // s_s                   scratch space to store superdiagonal entries of the tridiagonal matrix in shared memory
  // converged             flag if the current thread is already converged (that is count does not have to be computed)
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_computeNumSmallerEigenvals(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     inline unsigned int  \n");
  source.append("     computeNumSmallerEigenvals(__global "); source.append(numeric_string); source.append(" *g_d,   \n");
  source.append("                                __global "); source.append(numeric_string); source.append(" *g_s,   \n");
  source.append("                                const unsigned int n,  \n");
  source.append("                                const "); source.append(numeric_string); source.append(" x,         \n");
  source.append("                                const unsigned int tid,  \n");
  source.append("                                const unsigned int num_intervals_active,  \n");
  source.append("                                __local "); source.append(numeric_string); source.append(" *s_d,  \n");
  source.append("                                __local "); source.append(numeric_string); source.append(" *s_s,  \n");
  source.append("                                unsigned int converged  \n");
  source.append("                               )  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");


  source.append("          "); source.append(numeric_string); source.append(" delta = 1.0f;  \n");
  source.append("         unsigned int count = 0;  \n");

  source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      // read data into shared memory
  source.append("         if (lcl_id < n)  \n");
  source.append("         {  \n");
  source.append("             s_d[lcl_id] = *(g_d + lcl_id);  \n");
  source.append("             s_s[lcl_id] = *(g_s + lcl_id - 1);  \n");
  source.append("         }  \n");

  source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      // perform loop only for active threads
  source.append("         if ((tid < num_intervals_active) && (0 == converged))  \n");
  source.append("         {  \n");

          // perform (optimized) Gaussian elimination to determine the number
          // of eigenvalues that are smaller than n
  source.append("             for (unsigned int k = 0; k < n; ++k)  \n");
  source.append("             {  \n");
  source.append("                 delta = s_d[k] - x - (s_s[k] * s_s[k]) / delta;  \n");
  source.append("                 count += (delta < 0) ? 1 : 0;  \n");
  source.append("             }  \n");

  source.append("         } \n"); // end if thread currently processing an interval

  source.append("         return count;  \n");
  source.append("     }  \n");

  }


  ////////////////////////////////////////////////////////////////////////////////
  // Compute number of eigenvalues that are smaller than x given a symmetric,
  // real, and tridiagonal matrix
  //
  // g_d                   diagonal elements stored in global memory
  // g_s                   superdiagonal elements stored in global memory
  // n                     size of matrix
  // x                     value for which the number of eigenvalues that are smaller is seeked
  // tid                   thread identified (e.g. threadIdx.x or gtid)
  // num_intervals_active  number of active intervals / threads that currently process an interval
  // s_d                   scratch space to store diagonal entries of the tridiagonal matrix in shared memory
  // s_s                   scratch space to store superdiagonal entries of the tridiagonal matrix in shared memory
  // converged             flag if the current thread is already converged (that is count does not have to be computed)
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_computeNumSmallerEigenvalsLarge(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     inline unsigned int  \n");
  source.append("     computeNumSmallerEigenvalsLarge(__global "); source.append(numeric_string); source.append(" *g_d,   \n");
  source.append("                                __global "); source.append(numeric_string); source.append(" *g_s,   \n");
  source.append("                                const unsigned int n,  \n");
  source.append("                                const "); source.append(numeric_string); source.append(" x,         \n");
  source.append("                                const unsigned int tid,  \n");
  source.append("                                const unsigned int num_intervals_active,  \n");
  source.append("                                __local "); source.append(numeric_string); source.append(" *s_d,  \n");
  source.append("                                __local "); source.append(numeric_string); source.append(" *s_s,  \n");
  source.append("                                unsigned int converged  \n");
  source.append("                               )  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");

  source.append("          "); source.append(numeric_string); source.append(" delta = 1.0f;  \n");
  source.append("         unsigned int count = 0;  \n");

  source.append("         unsigned int rem = n;  \n");

      // do until whole diagonal and superdiagonal has been loaded and processed
  source.append("         for (unsigned int i = 0; i < n; i += lcl_sz)  \n");
  source.append("         {  \n");

  source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // read new chunk of data into shared memory
  source.append("             if ((i + lcl_id) < n)  \n");
  source.append("             {  \n");

  source.append("                 s_d[lcl_id] = *(g_d + i + lcl_id);  \n");
  source.append("                 s_s[lcl_id] = *(g_s + i + lcl_id - 1);  \n");
  source.append("             }  \n");

  source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");


  source.append("             if (tid < num_intervals_active)  \n");
  source.append("             {  \n");

              // perform (optimized) Gaussian elimination to determine the number
              // of eigenvalues that are smaller than n
  source.append("                 for (unsigned int k = 0; k < min(rem,lcl_sz); ++k)  \n");
  source.append("                 {  \n");
  source.append("                     delta = s_d[k] - x - (s_s[k] * s_s[k]) / delta;  \n");
                  // delta = (abs( delta) < (1.0e-10)) ? -(1.0e-10) : delta;
  source.append("                     count += (delta < 0) ? 1 : 0;  \n");
  source.append("                 }  \n");

  source.append("             } \n"); // end if thread currently processing an interval

  source.append("             rem -= lcl_sz;  \n");
  source.append("         }  \n");

  source.append("         return count;  \n");
  source.append("     }  \n");


  }

  ////////////////////////////////////////////////////////////////////////////////
  // Store all non-empty intervals resulting from the subdivision of the interval
  // currently processed by the thread
  //
  // addr                     base address for storing intervals
  // num_threads_active       number of threads / intervals in current sweep
  // s_left                   shared memory storage for left interval limits
  // s_right                  shared memory storage for right interval limits
  // s_left_count             shared memory storage for number of eigenvalues less than left interval limits
  // s_right_count            shared memory storage for number of eigenvalues less than right interval limits
  // left                     lower limit of interval
  // mid                      midpoint of interval
  // right                    upper limit of interval
  // left_count               eigenvalues less than \a left
  // mid_count                eigenvalues less than \a mid
  // right_count              eigenvalues less than \a right
  // precision                desired precision for eigenvalues
  // compact_second_chunk     shared mem flag if second chunk is used and ergo requires compaction
  // s_compaction_list_exc    helper array for stream compaction, s_compaction_list_exc[tid] = 1 when the thread generated two child intervals
  // is_active_interval       mark is thread has a second non-empty child interval
  ////////////////////////////////////////////////////////////////////////////////

  template<typename StringType>
  void generate_bisect_kernel_storeNonEmptyIntervals(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     void  \n");
  source.append("     storeNonEmptyIntervals(unsigned int addr,  \n");
  source.append("                            const unsigned int num_threads_active,  \n");
  source.append("                            __local "); source.append(numeric_string); source.append(" *s_left,   \n");
  source.append("                            __local "); source.append(numeric_string); source.append(" *s_right,  \n");
  source.append("                            __local unsigned int *s_left_count,  \n");
  source.append("                            __local unsigned int *s_right_count,  \n");
  source.append("                             "); source.append(numeric_string); source.append(" left, \n ");
  source.append("                             "); source.append(numeric_string); source.append(" mid,  \n");
  source.append("                             "); source.append(numeric_string); source.append(" right,\n");
  source.append("                            const unsigned int left_count,  \n");
  source.append("                            const unsigned int mid_count,  \n");
  source.append("                            const unsigned int right_count,  \n");
  source.append("                             "); source.append(numeric_string); source.append(" precision,  \n");
  source.append("                            __local unsigned int *compact_second_chunk,  \n");
  source.append("                            __local unsigned int *s_compaction_list_exc,  \n");
  source.append("                            unsigned int *is_active_second)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");

      // check if both child intervals are valid
  source.append("          \n");
  source.append("         if ((left_count != mid_count) && (mid_count != right_count))  \n");
  source.append("         {  \n");

          // store the left interval
  source.append("             storeInterval(addr, s_left, s_right, s_left_count, s_right_count,  \n");
  source.append("                           left, mid, left_count, mid_count, precision);  \n");

          // mark that a second interval has been generated, only stored after
          // stream compaction of second chunk
  source.append("             *is_active_second = 1;  \n");
  source.append("             s_compaction_list_exc[lcl_id] = 1;  \n");
  source.append("             *compact_second_chunk = 1;  \n");
  source.append("         }  \n");
  source.append("         else  \n");
  source.append("         {  \n");

          // only one non-empty child interval

          // mark that no second child
  source.append("             *is_active_second = 0;  \n");
  source.append("             s_compaction_list_exc[lcl_id] = 0;  \n");

          // store the one valid child interval
  source.append("             if (left_count != mid_count)  \n");
  source.append("             {  \n");
  source.append("                 storeInterval(addr, s_left, s_right, s_left_count, s_right_count,  \n");
  source.append("                               left, mid, left_count, mid_count, precision);  \n");
  source.append("             }  \n");
  source.append("             else  \n");
  source.append("             {  \n");
  source.append("                 storeInterval(addr, s_left, s_right, s_left_count, s_right_count,  \n");
  source.append("                               mid, right, mid_count, right_count, precision);  \n");
  source.append("             }  \n");

  source.append("         }  \n");
  source.append("     }  \n");

  }


  ////////////////////////////////////////////////////////////////////////////////
  //! Store all non-empty intervals resulting from the subdivision of the interval
  //! currently processed by the thread
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_storeNonEmptyIntervalsLarge(StringType & source, std::string const & numeric_string)
  {
      source.append("       \n");
      source.append("     void  \n");
      source.append("     storeNonEmptyIntervalsLarge(unsigned int addr,  \n");
      source.append("                            const unsigned int num_threads_active,  \n");
      source.append("                            __local "); source.append(numeric_string); source.append(" *s_left,   \n");
      source.append("                            __local "); source.append(numeric_string); source.append(" *s_right,  \n");
      source.append("                            __local unsigned short *s_left_count,  \n");
      source.append("                            __local unsigned short *s_right_count,  \n");
      source.append("                             "); source.append(numeric_string); source.append(" left, \n ");
      source.append("                             "); source.append(numeric_string); source.append(" mid,  \n");
      source.append("                             "); source.append(numeric_string); source.append(" right,\n");
      source.append("                            const unsigned int left_count,  \n");
      source.append("                            const unsigned int mid_count,  \n");
      source.append("                            const unsigned int right_count,  \n");
      source.append("                             "); source.append(numeric_string); source.append(" epsilon,  \n");
      source.append("                            __local unsigned int *compact_second_chunk,  \n");
      source.append("                            __local unsigned short *s_compaction_list,  \n");
      source.append("                            unsigned int *is_active_second)  \n");
      source.append("     {  \n");
      source.append("         uint glb_id = get_global_id(0); \n");
      source.append("         uint grp_id = get_group_id(0); \n");
      source.append("         uint grp_nm = get_num_groups(0); \n");
      source.append("         uint lcl_id = get_local_id(0); \n");
      source.append("         uint lcl_sz = get_local_size(0); \n");

          // check if both child intervals are valid
      source.append("         if ((left_count != mid_count) && (mid_count != right_count))  \n");
      source.append("         {  \n");

      source.append("             storeIntervalShort(addr, s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                           left, mid, left_count, mid_count, epsilon);  \n");

      source.append("             *is_active_second = 1;  \n");
      source.append("             s_compaction_list[lcl_id] = 1;  \n");
      source.append("             *compact_second_chunk = 1;  \n");
      source.append("         }  \n");
      source.append("         else  \n");
      source.append("         {  \n");

              // only one non-empty child interval

              // mark that no second child
      source.append("             *is_active_second = 0;  \n");
      source.append("             s_compaction_list[lcl_id] = 0;  \n");

              // store the one valid child interval
      source.append("             if (left_count != mid_count)  \n");
      source.append("             {  \n");
      source.append("                 storeIntervalShort(addr, s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                               left, mid, left_count, mid_count, epsilon);  \n");
      source.append("             }  \n");
      source.append("             else  \n");
      source.append("             {  \n");
      source.append("                 storeIntervalShort(addr, s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                               mid, right, mid_count, right_count, epsilon);  \n");
      source.append("             }  \n");
      source.append("         }  \n");
      source.append("     }  \n");
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Create indices for compaction, that is process \a s_compaction_list_exc
  // which is 1 for intervals that generated a second child and 0 otherwise
  // and create for each of the non-zero elements the index where the new
  // interval belongs to in a compact representation of all generated second children
  //
  // s_compaction_list_exc      list containing the flags which threads generated two children
  // num_threads_compaction     number of threads to employ for compaction
  ////////////////////////////////////////////////////////////////////////////////

  template<typename StringType>
  void generate_bisect_kernel_createIndicesCompaction(StringType & source)
  {
  source.append("       \n");
  source.append("     void  \n");
  source.append("     createIndicesCompaction(__local unsigned int *s_compaction_list_exc,  \n");
  source.append("                             unsigned int num_threads_compaction)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");


  source.append("         unsigned int offset = 1;  \n");
  source.append("         const unsigned int tid = lcl_id;  \n");
     // if(tid == 0)
       // printf("num_threads_compaction = %u\n", num_threads_compaction);

      // higher levels of scan tree
  source.append("         for (int d = (num_threads_compaction >> 1); d > 0; d >>= 1)  \n");
  source.append("         {  \n");

  source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

  source.append("             if (tid < d)  \n");
  source.append("             {  \n");

  source.append("                 unsigned int  ai = offset*(2*tid+1)-1;  \n");
  source.append("                 unsigned int  bi = offset*(2*tid+2)-1;  \n");
  source.append("              \n");
  source.append("                 s_compaction_list_exc[bi] =   s_compaction_list_exc[bi]  \n");
  source.append("                                               + s_compaction_list_exc[ai];  \n");
  source.append("             }  \n");

  source.append("             offset <<= 1;  \n");
  source.append("         }  \n");

      // traverse down tree: first down to level 2 across
  source.append("         for (int d = 2; d < num_threads_compaction; d <<= 1)  \n");
  source.append("         {  \n");

  source.append("             offset >>= 1;  \n");
  source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

  source.append("             if (tid < (d-1))  \n");
  source.append("             {  \n");

  source.append("                 unsigned int  ai = offset*(tid+1) - 1;  \n");
  source.append("                 unsigned int  bi = ai + (offset >> 1);  \n");

  source.append("                 s_compaction_list_exc[bi] =   s_compaction_list_exc[bi]  \n");
  source.append("                                               + s_compaction_list_exc[ai];  \n");
  source.append("             }  \n");
  source.append("         }  \n");

  source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

  source.append("     }  \n");
  }


  template<typename StringType>
  void generate_bisect_kernel_createIndicesCompactionShort(StringType & source)
  {
  source.append("       \n");
  source.append("     void  \n");
  source.append("     createIndicesCompactionShort(__local unsigned short *s_compaction_list_exc,  \n");
  source.append("                             unsigned int num_threads_compaction)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");


  source.append("         unsigned int offset = 1;  \n");
  source.append("         const unsigned int tid = lcl_id;  \n");

      // higher levels of scan tree
  source.append("         for (int d = (num_threads_compaction >> 1); d > 0; d >>= 1)  \n");
  source.append("         {  \n");

  source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

  source.append("             if (tid < d)  \n");
  source.append("             {  \n");

  source.append("                 unsigned int  ai = offset*(2*tid+1)-1;  \n");
  source.append("                 unsigned int  bi = offset*(2*tid+2)-1;  \n");
  source.append("              \n");
  source.append("                 s_compaction_list_exc[bi] =   s_compaction_list_exc[bi]  \n");
  source.append("                                               + s_compaction_list_exc[ai];  \n");
  source.append("             }  \n");

  source.append("             offset <<= 1;  \n");
  source.append("         }  \n");

      // traverse down tree: first down to level 2 across
  source.append("         for (int d = 2; d < num_threads_compaction; d <<= 1)  \n");
  source.append("         {  \n");

  source.append("             offset >>= 1;  \n");
  source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

  source.append("             if (tid < (d-1))  \n");
  source.append("             {  \n");

  source.append("                 unsigned int  ai = offset*(tid+1) - 1;  \n");
  source.append("                 unsigned int  bi = ai + (offset >> 1);  \n");

  source.append("                 s_compaction_list_exc[bi] =   s_compaction_list_exc[bi]  \n");
  source.append("                                               + s_compaction_list_exc[ai];  \n");
  source.append("             }  \n");
  source.append("         }  \n");

  source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

  source.append("     }  \n");
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Perform stream compaction for second child intervals
  //
  // s_left              shared memory storage for left interval limits
  // s_right             shared memory storage for right interval limits
  // s_left_count        shared memory storage for number of eigenvalues less than left interval limits
  // s_right_count       shared memory storage for number of eigenvalues less than right interval limits
  // mid                 midpoint of current interval (left of new interval)
  // right               upper limit of interval
  // mid_count           eigenvalues less than \a mid
  // s_compaction_list   list containing the indices where the data has to be stored
  // num_threads_active  number of active threads / intervals
  // is_active_interval  mark is thread has a second non-empty child interval
  ///////////////////////////////////////////////////////////////////////////////


  template<typename StringType>
  void generate_bisect_kernel_compactIntervals(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     void  \n");
  source.append("     compactIntervals(__local "); source.append(numeric_string); source.append(" *s_left,  \n");
  source.append("                      __local "); source.append(numeric_string); source.append(" *s_right, \n");
  source.append("                      __local unsigned int *s_left_count, \n");
  source.append("                      __local unsigned int *s_right_count,  \n");
  source.append("                       "); source.append(numeric_string); source.append(" mid,  \n");
  source.append("                       "); source.append(numeric_string); source.append(" right, \n");
  source.append("                      unsigned int mid_count, unsigned int right_count,  \n");
  source.append("                      __local unsigned int *s_compaction_list,  \n");
  source.append("                      unsigned int num_threads_active,  \n");
  source.append("                      unsigned int is_active_second)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");

  source.append("         const unsigned int tid = lcl_id;  \n");

      // perform compaction / copy data for all threads where the second
      // child is not dead
  source.append("         if ((tid < num_threads_active) && (1 == is_active_second))  \n");
  source.append("         {  \n");
  source.append("             unsigned int addr_w = num_threads_active + s_compaction_list[tid];  \n");
  source.append("             s_left[addr_w] = mid;  \n");
  source.append("             s_right[addr_w] = right;  \n");
  source.append("             s_left_count[addr_w] = mid_count;  \n");
  source.append("             s_right_count[addr_w] = right_count;  \n");
  source.append("         }  \n");
  source.append("     }  \n");
  }




  template<typename StringType>
  void generate_bisect_kernel_compactIntervalsShort(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     void  \n");
  source.append("     compactIntervalsShort(__local "); source.append(numeric_string); source.append(" *s_left,  \n");
  source.append("                      __local "); source.append(numeric_string); source.append(" *s_right,  \n");
  source.append("                      __local unsigned short *s_left_count, \n");
  source.append("                      __local unsigned short *s_right_count,  \n");
  source.append("                      "); source.append(numeric_string); source.append(" mid,   \n");
  source.append("                      "); source.append(numeric_string); source.append(" right, \n");
  source.append("                      unsigned int mid_count, unsigned int right_count,  \n");
  source.append("                      __local unsigned short *s_compaction_list,  \n");
  source.append("                      unsigned int num_threads_active,  \n");
  source.append("                      unsigned int is_active_second)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");

  source.append("         const unsigned int tid = lcl_id;  \n");

      // perform compaction / copy data for all threads where the second
      // child is not dead
  source.append("         if ((tid < num_threads_active) && (1 == is_active_second))  \n");
  source.append("         {  \n");
  source.append("             unsigned int addr_w = num_threads_active + s_compaction_list[tid];  \n");
  source.append("             s_left[addr_w] = mid;  \n");
  source.append("             s_right[addr_w] = right;  \n");
  source.append("             s_left_count[addr_w] = mid_count;  \n");
  source.append("             s_right_count[addr_w] = right_count;  \n");
  source.append("         }  \n");
  source.append("     }  \n");
  }



  template<typename StringType>
  void generate_bisect_kernel_storeIntervalConverged(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     void  \n");
  source.append("     storeIntervalConverged( __local "); source.append(numeric_string); source.append(" *s_left,   \n");
  source.append("                             __local "); source.append(numeric_string); source.append(" *s_right,   \n");
  source.append("                            __local unsigned int *s_left_count, \n");
  source.append("                            __local unsigned int *s_right_count,  \n");
  source.append("                            "); source.append(numeric_string); source.append(" *left,   \n");
  source.append("                            "); source.append(numeric_string); source.append(" *mid,   \n");
  source.append("                            "); source.append(numeric_string); source.append(" *right,   \n");
  source.append("                            unsigned int *left_count,     \n");
  source.append("                            unsigned int *mid_count,      \n");
  source.append("                            unsigned int *right_count,     \n");
  source.append("                            __local unsigned int *s_compaction_list_exc,  \n");
  source.append("                            __local unsigned int *compact_second_chunk,  \n");
  source.append("                            const unsigned int num_threads_active,  \n");
  source.append("                            unsigned int *is_active_second)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");

  source.append("         const unsigned int tid = lcl_id;  \n");
  source.append("         const unsigned int multiplicity = *right_count - *left_count;  \n");
      // check multiplicity of eigenvalue
  source.append("         if (1 == multiplicity)  \n");
  source.append("         {  \n");

          // just re-store intervals, simple eigenvalue
  source.append("             s_left[tid] = *left;  \n");
  source.append("             s_right[tid] = *right;  \n");
  source.append("             s_left_count[tid] = *left_count;  \n");
  source.append("             s_right_count[tid] = *right_count;  \n");
  source.append("             \n");

          // mark that no second child / clear
  source.append("             *is_active_second = 0;  \n");
  source.append("             s_compaction_list_exc[tid] = 0;  \n");
  source.append("         }  \n");
  source.append("         else  \n");
  source.append("         {  \n");

          // number of eigenvalues after the split less than mid
  source.append("             *mid_count = *left_count + (multiplicity >> 1);  \n");

          // store left interval
  source.append("             s_left[tid] = *left;  \n");
  source.append("             s_right[tid] = *right;  \n");
  source.append("             s_left_count[tid] = *left_count;  \n");
  source.append("             s_right_count[tid] = *mid_count;  \n");
  source.append("             *mid = *left;  \n");

          // mark that second child interval exists
  source.append("             *is_active_second = 1;  \n");
  source.append("             s_compaction_list_exc[tid] = 1;  \n");
  source.append("             *compact_second_chunk = 1;  \n");
  source.append("         }  \n");
  source.append("     }  \n");
  }





  template<typename StringType>
  void generate_bisect_kernel_storeIntervalConvergedShort(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     void  \n");
  source.append("     storeIntervalConvergedShort(__local "); source.append(numeric_string); source.append(" *s_left,   \n");
  source.append("                             __local "); source.append(numeric_string); source.append(" *s_right,   \n");
  source.append("                            __local unsigned short *s_left_count, \n");
  source.append("                            __local unsigned short *s_right_count,  \n");
  source.append("                            "); source.append(numeric_string); source.append(" *left,   \n");
  source.append("                            "); source.append(numeric_string); source.append(" *mid,   \n");
  source.append("                            "); source.append(numeric_string); source.append(" *right,   \n");
  source.append("                            unsigned int *left_count,     \n");
  source.append("                            unsigned int *mid_count,      \n");
  source.append("                            unsigned int *right_count,     \n");
  source.append("                            __local unsigned short *s_compaction_list_exc,  \n");
  source.append("                            __local unsigned int *compact_second_chunk,  \n");
  source.append("                            const unsigned int num_threads_active,  \n");
  source.append("                            unsigned int *is_active_second)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");

  source.append("         const unsigned int tid = lcl_id;  \n");
  source.append("         const unsigned int multiplicity = *right_count - *left_count;  \n");
      // check multiplicity of eigenvalue
  source.append("         if (1 == multiplicity)  \n");
  source.append("         {  \n");

          // just re-store intervals, simple eigenvalue
  source.append("             s_left[tid] = *left;  \n");
  source.append("             s_right[tid] = *right;  \n");
  source.append("             s_left_count[tid] = *left_count;  \n");
  source.append("             s_right_count[tid] = *right_count;  \n");
  source.append("             \n");

          // mark that no second child / clear
  source.append("             *is_active_second = 0;  \n");
  source.append("             s_compaction_list_exc[tid] = 0;  \n");
  source.append("         }  \n");
  source.append("         else  \n");
  source.append("         {  \n");

          // number of eigenvalues after the split less than mid
  source.append("             *mid_count = *left_count + (multiplicity >> 1);  \n");

          // store left interval
  source.append("             s_left[tid] = *left;  \n");
  source.append("             s_right[tid] = *right;  \n");
  source.append("             s_left_count[tid] = *left_count;  \n");
  source.append("             s_right_count[tid] = *mid_count;  \n");
  source.append("             *mid = *left;  \n");

          // mark that second child interval exists
  source.append("             *is_active_second = 1;  \n");
  source.append("             s_compaction_list_exc[tid] = 1;  \n");
  source.append("             *compact_second_chunk = 1;  \n");
  source.append("         }  \n");
  source.append("     }  \n");
  }

  ///////////////////////////////////////////////////////////////////////////////
  // Subdivide interval if active and not already converged
  //
  // tid                    id of thread
  // s_left                 shared memory storage for left interval limits
  // s_right                shared memory storage for right interval limits
  // s_left_count           shared memory storage for number of eigenvalues less than left interval limits
  // s_right_count          shared memory storage for number of eigenvalues less than right interval limits
  // num_threads_active     number of active threads in warp
  // left                   lower limit of interval
  // right                  upper limit of interval
  // left_count             eigenvalues less than \a left
  // right_count            eigenvalues less than \a right
  // all_threads_converged  shared memory flag if all threads are converged
  ///////////////////////////////////////////////////////////////////////////////


  template<typename StringType>
  void generate_bisect_kernel_subdivideActiveInterval(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     void  \n");
  source.append("     subdivideActiveIntervalMulti(const unsigned int tid,  \n");
  source.append("                             __local "); source.append(numeric_string); source.append(" *s_left,    \n");
  source.append("                             __local "); source.append(numeric_string); source.append(" *s_right,   \n");
  source.append("                             __local unsigned int *s_left_count,   \n");
  source.append("                             __local unsigned int *s_right_count,  \n");
  source.append("                             const unsigned int num_threads_active,  \n");
  source.append("                              "); source.append(numeric_string); source.append(" *left,   \n");
  source.append("                              "); source.append(numeric_string); source.append(" *right,   \n");
  source.append("                             unsigned int *left_count, unsigned int *right_count,  \n");
  source.append("                              "); source.append(numeric_string); source.append(" *mid,    \n");
  source.append("                              __local unsigned int *all_threads_converged)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");

      // for all active threads
  source.append("         if (tid < num_threads_active)  \n");
  source.append("         {  \n");

  source.append("             *left = s_left[tid];  \n");
  source.append("             *right = s_right[tid];  \n");
  source.append("             *left_count = s_left_count[tid];  \n");
  source.append("             *right_count = s_right_count[tid];  \n");

          // check if thread already converged
  source.append("             if (*left != *right)  \n");
  source.append("             {  \n");

  source.append("                 *mid = computeMidpoint(*left, *right);  \n");
  source.append("                 *all_threads_converged = 0;  \n");
  source.append("             }  \n");
  source.append("             else if ((*right_count - *left_count) > 1)  \n");
  source.append("             {  \n");
              // mark as not converged if multiple eigenvalues enclosed
              // duplicate interval in storeIntervalsConverged()
  source.append("                 *all_threads_converged = 0;  \n");
  source.append("             }  \n");

  source.append("         }    \n");
  // end for all active threads
  source.append("     }  \n");
  }


  template<typename StringType>
  void generate_bisect_kernel_subdivideActiveIntervalShort(StringType & source, std::string const & numeric_string)
  {
  source.append("       \n");
  source.append("     void  \n");
  source.append("     subdivideActiveIntervalShort(const unsigned int tid,  \n");
  source.append("                             __local "); source.append(numeric_string); source.append(" *s_left,    \n");
  source.append("                             __local "); source.append(numeric_string); source.append(" *s_right,   \n");
  source.append("                             __local unsigned short *s_left_count,   \n");
  source.append("                             __local unsigned short *s_right_count,  \n");
  source.append("                             const unsigned int num_threads_active,  \n");
  source.append("                             "); source.append(numeric_string); source.append(" *left,   \n");
  source.append("                             "); source.append(numeric_string); source.append(" *right,   \n");
  source.append("                             unsigned int *left_count, unsigned int *right_count,  \n");
  source.append("                             "); source.append(numeric_string); source.append(" *mid,    \n");
  source.append("                             __local unsigned int *all_threads_converged)  \n");
  source.append("     {  \n");
  source.append("         uint glb_id = get_global_id(0); \n");
  source.append("         uint grp_id = get_group_id(0); \n");
  source.append("         uint grp_nm = get_num_groups(0); \n");
  source.append("         uint lcl_id = get_local_id(0); \n");
  source.append("         uint lcl_sz = get_local_size(0); \n");

      // for all active threads
  source.append("         if (tid < num_threads_active)  \n");
  source.append("         {  \n");

  source.append("             *left = s_left[tid];  \n");
  source.append("             *right = s_right[tid];  \n");
  source.append("             *left_count = s_left_count[tid];  \n");
  source.append("             *right_count = s_right_count[tid];  \n");

          // check if thread already converged
  source.append("             if (*left != *right)  \n");
  source.append("             {  \n");

  source.append("                 *mid = computeMidpoint(*left, *right);  \n");
  source.append("                 *all_threads_converged = 0;  \n");
  source.append("             }  \n");

  source.append("         }    \n");
  // end for all active threads
  source.append("     }  \n");
  }

  // end of utilities
  // start of kernels


  ////////////////////////////////////////////////////////////////////////////////
  // Bisection to find eigenvalues of a real, symmetric, and tridiagonal matrix
  //
  // g_d             diagonal elements in global memory
  // g_s             superdiagonal elements in global elements (stored so that the element *(g_s - 1) can be accessed an equals 0
  // n               size of matrix
  // lg              lower bound of input interval (e.g. Gerschgorin interval)
  // ug              upper bound of input interval (e.g. Gerschgorin interval)
  // lg_eig_count    number of eigenvalues that are smaller than \a lg
  // lu_eig_count    number of eigenvalues that are smaller than \a lu
  // epsilon         desired accuracy of eigenvalues to compute
  ////////////////////////////////////////////////////////////////////////////////
  ///
  template <typename StringType>
  void generate_bisect_kernel_bisectKernel(StringType & source, std::string const & numeric_string)
  {
      source.append("     __kernel  \n");
      source.append("     void  \n");
      source.append("     bisectKernelSmall(__global "); source.append(numeric_string); source.append(" *g_d,   \n");
      source.append("                  __global "); source.append(numeric_string); source.append(" *g_s,   \n");
      source.append("                  const unsigned int n,  \n");
      source.append("                  __global "); source.append(numeric_string); source.append(" *g_left,   \n");
      source.append("                  __global "); source.append(numeric_string); source.append(" *g_right,  \n");
      source.append("                  __global unsigned int *g_left_count, __global unsigned int *g_right_count,  \n");
      source.append("                  const "); source.append(numeric_string); source.append(" lg,  \n");
      source.append("                  const "); source.append(numeric_string); source.append(" ug,  \n");
      source.append("                  const unsigned int lg_eig_count, const unsigned int ug_eig_count, \n");
      source.append("                  "); source.append(numeric_string); source.append(" epsilon  \n");
      source.append("                 )  \n");
      source.append("     {  \n");
      source.append("         g_s = g_s + 1; \n");
      source.append("         uint glb_id = get_global_id(0); \n");
      source.append("         uint grp_id = get_group_id(0); \n");
      source.append("         uint grp_nm = get_num_groups(0); \n");
      source.append("         uint lcl_id = get_local_id(0); \n");
      source.append("         uint lcl_sz = get_local_size(0); \n");

          // intervals (store left and right because the subdivision tree is in general
          // not dense
      source.append("         __local "); source.append(numeric_string); source.append(" s_left[VIENNACL_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX];  \n");
      source.append("         __local "); source.append(numeric_string); source.append(" s_right[VIENNACL_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX];  \n");

          // number of eigenvalues that are smaller than s_left / s_right
          // (correspondence is realized via indices)
      source.append("         __local  unsigned int  s_left_count[VIENNACL_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX];  \n");
      source.append("         __local  unsigned int  s_right_count[VIENNACL_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX];  \n");

          // helper for stream compaction
      source.append("         __local  unsigned int  \n");
      source.append("           s_compaction_list[VIENNACL_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX + 1];  \n");

          // state variables for whole block
          // if 0 then compaction of second chunk of child intervals is not necessary
          // (because all intervals had exactly one non-dead child)
      source.append("         __local  unsigned int compact_second_chunk;  \n");
      source.append("         __local  unsigned int all_threads_converged;  \n");

          // number of currently active threads
      source.append("         __local  unsigned int num_threads_active;  \n");

          // number of threads to use for stream compaction
      source.append("         __local  unsigned int num_threads_compaction;  \n");

          // helper for exclusive scan
      source.append("         __local unsigned int *s_compaction_list_exc = s_compaction_list + 1;  \n");


          // variables for currently processed interval
          // left and right limit of active interval
      source.append("          "); source.append(numeric_string); source.append(" left = 0.0f;  \n");
      source.append("          "); source.append(numeric_string); source.append(" right = 0.0f;  \n");
      source.append("         unsigned int left_count = 0;  \n");
      source.append("         unsigned int right_count = 0;  \n");
          // midpoint of active interval
      source.append("          "); source.append(numeric_string); source.append(" mid = 0.0f;  \n");
          // number of eigenvalues smaller then mid
      source.append("         unsigned int mid_count = 0;  \n");
          // affected from compaction
      source.append("         unsigned int  is_active_second = 0;  \n");

      source.append("         s_compaction_list[lcl_id] = 0;  \n");
      source.append("         s_left[lcl_id] = 0.0;  \n");
      source.append("         s_right[lcl_id] = 0.0;  \n");
      source.append("         s_left_count[lcl_id] = 0;  \n");
      source.append("         s_right_count[lcl_id] = 0;  \n");

      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // set up initial configuration
      source.append("         if (0 == lcl_id)  \n");
      source.append("         {  \n");
      source.append("             s_left[0] = lg;  \n");
      source.append("             s_right[0] = ug;  \n");
      source.append("             s_left_count[0] = lg_eig_count;  \n");
      source.append("             s_right_count[0] = ug_eig_count;  \n");

      source.append("             compact_second_chunk = 0;  \n");
      source.append("             num_threads_active = 1;  \n");

      source.append("             num_threads_compaction = 1;  \n");
      source.append("         }  \n");

          // for all active threads read intervals from the last level
          // the number of (worst case) active threads per level l is 2^l

      source.append("         while (true)  \n");
      source.append("         {  \n");

      source.append("             all_threads_converged = 1;  \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("             is_active_second = 0;  \n");
      source.append("             subdivideActiveIntervalMulti(lcl_id,  \n");
      source.append("                                     s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                                     num_threads_active,  \n");
      source.append("                                     &left, &right, &left_count, &right_count,  \n");
      source.append("                                     &mid, &all_threads_converged);  \n");
   //   source.append("             output[lcl_id] = s_left;  \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // check if done
      source.append("             if (1 == all_threads_converged)  \n");
      source.append("             {  \n");
      source.append("                 break;  \n");
      source.append("             }  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // compute number of eigenvalues smaller than mid
              // use all threads for reading the necessary matrix data from global
              // memory
              // use s_left and s_right as scratch space for diagonal and
              // superdiagonal of matrix
      source.append("             mid_count = computeNumSmallerEigenvals(g_d, g_s, n, mid,  \n");
      source.append("                                                    lcl_id, num_threads_active,  \n");
      source.append("                                                    s_left, s_right,  \n");
      source.append("                                                    (left == right));  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // store intervals
              // for all threads store the first child interval in a continuous chunk of
              // memory, and the second child interval -- if it exists -- in a second
              // chunk; it is likely that all threads reach convergence up to
              // \a epsilon at the same level; furthermore, for higher level most / all
              // threads will have only one child, storing the first child compactly will
              // (first) avoid to perform a compaction step on the first chunk, (second)
              // make it for higher levels (when all threads / intervals have
              // exactly one child)  unnecessary to perform a compaction of the second
              // chunk
      source.append("             if (lcl_id < num_threads_active)  \n");
      source.append("             {  \n");

      source.append("                 if (left != right)  \n");
      source.append("                 {  \n");

                      // store intervals
      source.append("                     storeNonEmptyIntervals(lcl_id, num_threads_active,  \n");
      source.append("                                            s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                                            left, mid, right,  \n");
      source.append("                                            left_count, mid_count, right_count,  \n");
      source.append("                                            epsilon, &compact_second_chunk,  \n");
      source.append("                                            s_compaction_list_exc,  \n");
      source.append("                                            &is_active_second);  \n");
      source.append("                 }  \n");
      source.append("                 else  \n");
      source.append("                 {  \n");

      source.append("                     storeIntervalConverged(s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                                            &left, &mid, &right,  \n");
      source.append("                                            &left_count, &mid_count, &right_count,  \n");
      source.append("                                            s_compaction_list_exc, &compact_second_chunk,  \n");
      source.append("                                            num_threads_active,  \n");
      source.append("                                            &is_active_second);  \n");
      source.append("                 }  \n");
      source.append("             }  \n");

              // necessary so that compact_second_chunk is up-to-date
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // perform compaction of chunk where second children are stored
              // scan of (num_threads_actieigenvaluesve / 2) elements, thus at most
              // (num_threads_active / 4) threads are needed
      source.append("             if (compact_second_chunk > 0)  \n");
      source.append("             {  \n");

      source.append("                 createIndicesCompaction(s_compaction_list_exc, num_threads_compaction);  \n");

      source.append("                 compactIntervals(s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                                  mid, right, mid_count, right_count,  \n");
      source.append("                                  s_compaction_list, num_threads_active,  \n");
      source.append("                                  is_active_second);  \n");
      source.append("             }  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("             if (0 == lcl_id)  \n");
      source.append("             {  \n");

                  // update number of active threads with result of reduction
      source.append("                 num_threads_active += s_compaction_list[num_threads_active];  \n");

      source.append("                 num_threads_compaction = ceilPow2(num_threads_active);  \n");

      source.append("                 compact_second_chunk = 0;  \n");
      source.append("             }  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("         }  \n");

      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // write resulting intervals to global mem
          // for all threads write if they have been converged to an eigenvalue to
          // a separate array

          // at most n valid intervals
      source.append("         if (lcl_id < n)  \n");
      source.append("         {  \n");
              // intervals converged so left and right limit are identical
      source.append("             g_left[lcl_id]  = s_left[lcl_id];  \n");
              // left count is sufficient to have global order
      source.append("             g_left_count[lcl_id]  = s_left_count[lcl_id];  \n");
      source.append("         }  \n");
      source.append("     }  \n");
     }

  ////////////////////////////////////////////////////////////////////////////////
  // Perform second step of bisection algorithm for large matrices for intervals that after the first step contained more than one eigenvalue
  //
  // g_d              diagonal elements of symmetric, tridiagonal matrix
  // g_s              superdiagonal elements of symmetric, tridiagonal matrix
  // n                matrix size
  // blocks_mult      start addresses of blocks of intervals that are processed by one block of threads, each of the intervals contains more than one eigenvalue
  // blocks_mult_sum  total number of eigenvalues / singleton intervals in one block of intervals
  // g_left           left limits of intervals
  // g_right          right limits of intervals
  // g_left_count     number of eigenvalues less than left limits
  // g_right_count    number of eigenvalues less than right limits
  // g_lambda         final eigenvalue
  // g_pos            index of eigenvalue (in ascending order)
  // precision         desired precision of eigenvalues
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_bisectKernelLarge_MultIntervals(StringType & source, std::string const & numeric_string)
  {
      source.append("     __kernel  \n");
      source.append("     void  \n");
      source.append("     bisectKernelLarge_MultIntervals(__global "); source.append(numeric_string); source.append(" *g_d,   \n");
      source.append("                                     __global "); source.append(numeric_string); source.append(" *g_s,   \n");
      source.append("                                     const unsigned int n,  \n");
      source.append("                                     __global unsigned int *blocks_mult,  \n");
      source.append("                                     __global unsigned int *blocks_mult_sum,  \n");
      source.append("                                     __global "); source.append(numeric_string); source.append(" *g_left,   \n");
      source.append("                                     __global "); source.append(numeric_string); source.append(" *g_right,  \n");
      source.append("                                     __global unsigned int *g_left_count,  \n");
      source.append("                                     __global unsigned int *g_right_count,  \n");
      source.append("                                     __global  "); source.append(numeric_string); source.append(" *g_lambda, \n");
      source.append("                                     __global unsigned int *g_pos,  \n");
      source.append("                                     "); source.append(numeric_string); source.append(" precision  \n");
      source.append("                                    )  \n");
      source.append("     {  \n");
      source.append("         g_s = g_s + 1; \n");
      source.append("         uint glb_id = get_global_id(0); \n");
      source.append("         uint grp_id = get_group_id(0); \n");
      source.append("         uint grp_nm = get_num_groups(0); \n");
      source.append("         uint lcl_id = get_local_id(0); \n");
      source.append("         uint lcl_sz = get_local_size(0); \n");

      source.append("       const unsigned int tid = lcl_id;  \n");

          // left and right limits of interval
      source.append("         __local "); source.append(numeric_string); source.append(" s_left[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK];  \n");
      source.append("         __local "); source.append(numeric_string); source.append(" s_right[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK];  \n");

          // number of eigenvalues smaller than interval limits
      source.append("         __local  unsigned int  s_left_count[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK];  \n");
      source.append("         __local  unsigned int  s_right_count[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK];  \n");

          // helper array for chunk compaction of second chunk
      source.append("         __local  unsigned int  s_compaction_list[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK + 1];  \n");
          // compaction list helper for exclusive scan
      source.append("         __local unsigned int *s_compaction_list_exc = s_compaction_list + 1;  \n");

          // flag if all threads are converged
      source.append("         __local  unsigned int  all_threads_converged;  \n");
          // number of active threads
      source.append("         __local  unsigned int  num_threads_active;  \n");
          // number of threads to employ for compaction
      source.append("         __local  unsigned int  num_threads_compaction;  \n");
          // flag if second chunk has to be compacted
      source.append("         __local  unsigned int compact_second_chunk;  \n");

          // parameters of block of intervals processed by this block of threads
      source.append("         __local  unsigned int  c_block_start;  \n");
      source.append("         __local  unsigned int  c_block_end;  \n");
      source.append("         __local  unsigned int  c_block_offset_output;  \n");

          // midpoint of currently active interval of the thread
      source.append("         "); source.append(numeric_string); source.append(" mid = 0.0f;  \n");
          // number of eigenvalues smaller than \a mid
      source.append("         unsigned int  mid_count = 0;  \n");
          // current interval parameter
      source.append("         "); source.append(numeric_string); source.append(" left = 0.0f;  \n");
      source.append("         "); source.append(numeric_string); source.append(" right = 0.0f;  \n");
      source.append("         unsigned int  left_count = 0;  \n");
      source.append("         unsigned int  right_count = 0;  \n");
          // helper for compaction, keep track which threads have a second child
      source.append("         unsigned int  is_active_second = 0;  \n");

      source.append("         barrier(CLK_LOCAL_MEM_FENCE);            \n");

          // initialize common start conditions
      source.append("         if (0 == tid)  \n");
      source.append("         {  \n");

      source.append("             c_block_start = blocks_mult[grp_id];  \n");
      source.append("             c_block_end = blocks_mult[grp_id + 1];  \n");
      source.append("             c_block_offset_output = blocks_mult_sum[grp_id];  \n");
      source.append("               \n");

      source.append("             num_threads_active = c_block_end - c_block_start;  \n");
      source.append("             s_compaction_list[0] = 0;  \n");
      source.append("             num_threads_compaction = ceilPow2(num_threads_active);  \n");

      source.append("             all_threads_converged = 1;  \n");
      source.append("             compact_second_chunk = 0;  \n");
      source.append("         }  \n");
      source.append("          s_left_count [tid] = 42;  \n");
      source.append("          s_right_count[tid] = 42;  \n");
      source.append("          s_left_count [tid + VIENNACL_BISECT_MAX_THREADS_BLOCK] = 0;  \n");
      source.append("          s_right_count[tid + VIENNACL_BISECT_MAX_THREADS_BLOCK] = 0;  \n");
      source.append("           \n");
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
      source.append("           \n");

          // read data into shared memory
      source.append("         if (tid < num_threads_active)  \n");
      source.append("         {  \n");

      source.append("             s_left[tid]  = g_left[c_block_start + tid];  \n");
      source.append("             s_right[tid] = g_right[c_block_start + tid];  \n");
      source.append("             s_left_count[tid]  = g_left_count[c_block_start + tid];  \n");
      source.append("             s_right_count[tid] = g_right_count[c_block_start + tid];  \n");
      source.append("               \n");
      source.append("         }  \n");
      source.append("        \n");
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
      source.append("         unsigned int iter = 0;  \n");
          // do until all threads converged
      source.append("         while (true)  \n");
      source.append("         {  \n");
      source.append("             iter++;  \n");
              //for (int iter=0; iter < 0; iter++) {
      source.append("             s_compaction_list[lcl_id] = 0;  \n");
      source.append("             s_compaction_list[lcl_id + lcl_sz] = 0;  \n");
      source.append("             s_compaction_list[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK] = 0;  \n");

              // subdivide interval if currently active and not already converged
      source.append("             subdivideActiveIntervalMulti(tid, s_left, s_right,  \n");
      source.append("                                     s_left_count, s_right_count,  \n");
      source.append("                                     num_threads_active,  \n");
      source.append("                                     &left, &right, &left_count, &right_count,  \n");
      source.append("                                     &mid, &all_threads_converged);  \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // stop if all eigenvalues have been found
      source.append("             if (1 == all_threads_converged)  \n");
      source.append("             {  \n");
      source.append("                  \n");
      source.append("                 break;  \n");
      source.append("             }  \n");

              // compute number of eigenvalues smaller than mid for active and not
              // converged intervals, use all threads for loading data from gmem and
              // s_left and s_right as scratch space to store the data load from gmem
              // in shared memory
      source.append("             mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n,  \n");
      source.append("                                                         mid, tid, num_threads_active,  \n");
      source.append("                                                         s_left, s_right,  \n");
      source.append("                                                         (left == right));  \n");
      source.append("                                                \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("             if (tid < num_threads_active)  \n");
      source.append("             {  \n");
      source.append("                   \n");
                  // store intervals
      source.append("                 if (left != right)  \n");
      source.append("                 {  \n");

      source.append("                     storeNonEmptyIntervals(tid, num_threads_active,  \n");
      source.append("                                            s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                                            left, mid, right,  \n");
      source.append("                                            left_count, mid_count, right_count,  \n");
      source.append("                                            precision, &compact_second_chunk,  \n");
      source.append("                                            s_compaction_list_exc,  \n");
      source.append("                                            &is_active_second);  \n");
      source.append("                      \n");
      source.append("                 }  \n");
      source.append("                 else  \n");
      source.append("                 {  \n");

      source.append("                     storeIntervalConverged(s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                                            &left, &mid, &right,  \n");
      source.append("                                            &left_count, &mid_count, &right_count,  \n");
      source.append("                                            s_compaction_list_exc, &compact_second_chunk,  \n");
      source.append("                                            num_threads_active,  \n");
      source.append("                                            &is_active_second);  \n");
      source.append("                   \n");
      source.append("                 }  \n");
      source.append("             }  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // compact second chunk of intervals if any of the threads generated
              // two child intervals
      source.append("             if (1 == compact_second_chunk)  \n");
      source.append("             {  \n");

      source.append("                 createIndicesCompaction(s_compaction_list_exc, num_threads_compaction);  \n");
      source.append("                 compactIntervals(s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                                  mid, right, mid_count, right_count,  \n");
      source.append("                                  s_compaction_list, num_threads_active,  \n");
      source.append("                                  is_active_second);  \n");
      source.append("             }  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // update state variables
      source.append("             if (0 == tid)  \n");
      source.append("             {  \n");
      source.append("                 num_threads_active += s_compaction_list[num_threads_active];  \n");
      source.append("                 num_threads_compaction = ceilPow2(num_threads_active);  \n");

      source.append("                 compact_second_chunk = 0;  \n");
      source.append("                 all_threads_converged = 1;  \n");
      source.append("             }  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // clear
      source.append("             s_compaction_list_exc[lcl_id] = 0;  \n");
      source.append("             s_compaction_list_exc[lcl_id + lcl_sz] = 0;   \n");
      source.append("               \n");
      source.append("             if (num_threads_compaction > lcl_sz)              \n");
      source.append("             {  \n");
      source.append("               break;  \n");
      source.append("             }  \n");


      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("    } \n"); // end until all threads converged

          // write data back to global memory
      source.append("         if (tid < num_threads_active)  \n");
      source.append("         {  \n");

      source.append("             unsigned int addr = c_block_offset_output + tid;  \n");
      source.append("               \n");
      source.append("             g_lambda[addr]  = s_left[tid];  \n");
      source.append("             g_pos[addr]   = s_right_count[tid];  \n");
      source.append("         }  \n");
      source.append("     }  \n");
  }


  ////////////////////////////////////////////////////////////////////////////////
  // Determine eigenvalues for large matrices for intervals that after the first step contained one eigenvalue
  //
  // g_d            diagonal elements of symmetric, tridiagonal matrix
  // g_s            superdiagonal elements of symmetric, tridiagonal matrix
  // n              matrix size
  // num_intervals  total number of intervals containing one eigenvalue after the first step
  // g_left         left interval limits
  // g_right        right interval limits
  // g_pos          index of interval / number of intervals that are smaller than right interval limit
  // precision      desired precision of eigenvalues
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_bisectKernelLarge_OneIntervals(StringType & source, std::string const & numeric_string)
  {
      source.append("     __kernel  \n");
      source.append("     void  \n");
      source.append("     bisectKernelLarge_OneIntervals(__global "); source.append(numeric_string); source.append(" *g_d,   \n");
      source.append("                                    __global "); source.append(numeric_string); source.append(" *g_s,    \n");
      source.append("                                    const unsigned int n,  \n");
      source.append("                                    unsigned int num_intervals,  \n");
      source.append("                                    __global "); source.append(numeric_string); source.append(" *g_left,  \n");
      source.append("                                    __global "); source.append(numeric_string); source.append(" *g_right,  \n");
      source.append("                                    __global unsigned int *g_pos,  \n");
      source.append("                                    "); source.append(numeric_string); source.append(" precision)  \n");
      source.append("     {  \n");
      source.append("         g_s = g_s + 1; \n");
      source.append("         uint glb_id = get_global_id(0); \n");
      source.append("         uint grp_id = get_group_id(0); \n");
      source.append("         uint grp_nm = get_num_groups(0); \n");
      source.append("         uint lcl_id = get_local_id(0); \n");
      source.append("         uint lcl_sz = get_local_size(0); \n");
      source.append("         const unsigned int gtid = (lcl_sz * grp_id) + lcl_id;  \n");
      source.append("         __local "); source.append(numeric_string); source.append(" s_left_scratch[VIENNACL_BISECT_MAX_THREADS_BLOCK];  \n");
      source.append("         __local "); source.append(numeric_string); source.append(" s_right_scratch[VIENNACL_BISECT_MAX_THREADS_BLOCK];  \n");
          // active interval of thread
          // left and right limit of current interval
      source.append("          "); source.append(numeric_string); source.append(" left, right;  \n");
          // number of threads smaller than the right limit (also corresponds to the
          // global index of the eigenvalues contained in the active interval)
      source.append("         unsigned int right_count;  \n");
          // flag if current thread converged
      source.append("         unsigned int converged = 0;  \n");
          // midpoint when current interval is subdivided
      source.append("         "); source.append(numeric_string); source.append(" mid = 0.0f;  \n");
          // number of eigenvalues less than mid
      source.append("         unsigned int mid_count = 0;  \n");

          // read data from global memory
      source.append("         if (gtid < num_intervals)  \n");
      source.append("         {  \n");
      source.append("             left = g_left[gtid];  \n");
      source.append("             right = g_right[gtid];  \n");
      source.append("             right_count = g_pos[gtid];  \n");
      source.append("         }  \n");
          // flag to determine if all threads converged to eigenvalue
      source.append("         __local  unsigned int  converged_all_threads;  \n");
          // initialized shared flag
      source.append("         if (0 == lcl_id)  \n");
      source.append("         {  \n");
      source.append("             converged_all_threads = 0;  \n");
      source.append("         }  \n");
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
          // process until all threads converged to an eigenvalue
      source.append("         while (true)  \n");
      source.append("         {  \n");
      source.append("             converged_all_threads = 1;  \n");
              // update midpoint for all active threads
      source.append("             if ((gtid < num_intervals) && (0 == converged))  \n");
      source.append("             {  \n");
      source.append("                 mid = computeMidpoint(left, right);  \n");
      source.append("             }  \n");
              // find number of eigenvalues that are smaller than midpoint
      source.append("             mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n,  \n");
      source.append("                                                         mid, gtid, num_intervals,  \n");
      source.append("                                                         s_left_scratch,  \n");
      source.append("                                                         s_right_scratch,  \n");
      source.append("                                                         converged);  \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
              // for all active threads
      source.append("             if ((gtid < num_intervals) && (0 == converged))  \n");
      source.append("             {  \n");
                  // update intervals -- always one child interval survives
      source.append("                 if (right_count == mid_count)  \n");
      source.append("                 {  \n");
      source.append("                     right = mid;  \n");
      source.append("                 }  \n");
      source.append("                 else  \n");
      source.append("                 {  \n");
      source.append("                     left = mid;  \n");
      source.append("                 }  \n");
                  // check for convergence
      source.append("                 "); source.append(numeric_string); source.append(" t0 = right - left;  \n");
      source.append("                 "); source.append(numeric_string); source.append(" t1 = max(fabs(right), fabs(left)) * precision;  \n");

      source.append("                 if (t0 < min(precision, t1))  \n");
      source.append("                 {  \n");
      source.append("                     "); source.append(numeric_string); source.append(" lambda = computeMidpoint(left, right);  \n");
      source.append("                     left = lambda;  \n");
      source.append("                     right = lambda;  \n");

      source.append("                     converged = 1;  \n");
      source.append("                 }  \n");
      source.append("                 else  \n");
      source.append("                 {  \n");
      source.append("                     converged_all_threads = 0;  \n");
      source.append("                 }  \n");
      source.append("             }  \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
      source.append("             if (1 == converged_all_threads)  \n");
      source.append("             {  \n");
      source.append("                 break;  \n");
      source.append("             }  \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
      source.append("         }  \n");
          // write data back to global memory
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
      source.append("         if (gtid < num_intervals)  \n");
      source.append("         {  \n");
              // intervals converged so left and right interval limit are both identical
              // and identical to the eigenvalue
      source.append("             g_left[gtid] = left;  \n");
      source.append("         }  \n");
      source.append("     }  \n");
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Write data to global memory
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_writeToGmem(StringType & source, std::string const & numeric_string)
  {
      source.append("       \n");
      source.append("     void writeToGmem(const unsigned int tid, const unsigned int tid_2,  \n");
      source.append("                      const unsigned int num_threads_active,  \n");
      source.append("                      const unsigned int num_blocks_mult,  \n");
      source.append("                      __global "); source.append(numeric_string); source.append(" *g_left_one,    \n");
      source.append("                      __global "); source.append(numeric_string); source.append(" *g_right_one,   \n");
      source.append("                      __global unsigned int *g_pos_one,  \n");
      source.append("                      __global "); source.append(numeric_string); source.append(" *g_left_mult,   \n");
      source.append("                      __global "); source.append(numeric_string); source.append(" *g_right_mult,  \n");
      source.append("                      __global unsigned int *g_left_count_mult,  \n");
      source.append("                      __global unsigned int *g_right_count_mult,  \n");
      source.append("                      __local "); source.append(numeric_string); source.append(" *s_left,    \n");
      source.append("                      __local "); source.append(numeric_string); source.append(" *s_right,  \n");
      source.append("                      __local unsigned short *s_left_count, __local unsigned short *s_right_count,  \n");
      source.append("                      __global unsigned int *g_blocks_mult,  \n");
      source.append("                      __global unsigned int *g_blocks_mult_sum,  \n");
      source.append("                      __local unsigned short *s_compaction_list,  \n");
      source.append("                      __local unsigned short *s_cl_helper,  \n");
      source.append("                      unsigned int offset_mult_lambda  \n");
      source.append("                     )  \n");
      source.append("     {  \n");
      source.append("         uint glb_id = get_global_id(0); \n");
      source.append("         uint grp_id = get_group_id(0); \n");
      source.append("         uint grp_nm = get_num_groups(0); \n");
      source.append("         uint lcl_id = get_local_id(0); \n");
      source.append("         uint lcl_sz = get_local_size(0); \n");


      source.append("         if (tid < offset_mult_lambda)  \n");
      source.append("         {  \n");

      source.append("             g_left_one[tid] = s_left[tid];  \n");
      source.append("             g_right_one[tid] = s_right[tid];  \n");
              // right count can be used to order eigenvalues without sorting
      source.append("             g_pos_one[tid] = s_right_count[tid];  \n");
      source.append("         }  \n");
      source.append("         else  \n");
      source.append("         {  \n");

      source.append("               \n");
      source.append("             g_left_mult[tid - offset_mult_lambda] = s_left[tid];  \n");
      source.append("             g_right_mult[tid - offset_mult_lambda] = s_right[tid];  \n");
      source.append("             g_left_count_mult[tid - offset_mult_lambda] = s_left_count[tid];  \n");
      source.append("             g_right_count_mult[tid - offset_mult_lambda] = s_right_count[tid];  \n");
      source.append("         }  \n");

      source.append("         if (tid_2 < num_threads_active)  \n");
      source.append("         {  \n");

      source.append("             if (tid_2 < offset_mult_lambda)  \n");
      source.append("             {  \n");

      source.append("                 g_left_one[tid_2] = s_left[tid_2];  \n");
      source.append("                 g_right_one[tid_2] = s_right[tid_2];  \n");
                  // right count can be used to order eigenvalues without sorting
      source.append("                 g_pos_one[tid_2] = s_right_count[tid_2];  \n");
      source.append("             }  \n");
      source.append("             else  \n");
      source.append("             {  \n");

      source.append("                 g_left_mult[tid_2 - offset_mult_lambda] = s_left[tid_2];  \n");
      source.append("                 g_right_mult[tid_2 - offset_mult_lambda] = s_right[tid_2];  \n");
      source.append("                 g_left_count_mult[tid_2 - offset_mult_lambda] = s_left_count[tid_2];  \n");
      source.append("                 g_right_count_mult[tid_2 - offset_mult_lambda] = s_right_count[tid_2];  \n");
      source.append("             }  \n");

      source.append("    } \n");      // end writing out data

          source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // note that s_cl_blocking = s_compaction_list + 1;, that is by writing out
          // s_compaction_list we write the exclusive scan result
      source.append("         if (tid <= num_blocks_mult)  \n");
      source.append("         {  \n");
      source.append("             g_blocks_mult[tid] = s_compaction_list[tid];  \n");
      source.append("             g_blocks_mult_sum[tid] = s_cl_helper[tid];  \n");
      source.append("         }  \n");
      source.append("         if (tid_2 <= num_blocks_mult)  \n");
      source.append("         {  \n");
      source.append("             g_blocks_mult[tid_2] = s_compaction_list[tid_2];  \n");
      source.append("             g_blocks_mult_sum[tid_2] = s_cl_helper[tid_2];  \n");
      source.append("         }  \n");
      source.append("     }  \n");
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Perform final stream compaction before writing data to global memory
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_compactStreamsFinal(StringType & source, std::string const & numeric_string)
  {
      source.append("       \n");
      source.append("     void  \n");
      source.append("     compactStreamsFinal(const unsigned int tid, const unsigned int tid_2,  \n");
      source.append("                         const unsigned int num_threads_active,  \n");
      source.append("                         __local unsigned int *offset_mult_lambda,  \n");
      source.append("                         __local "); source.append(numeric_string); source.append(" *s_left,    \n");
      source.append("                         __local "); source.append(numeric_string); source.append(" *s_right,   \n");
      source.append("                         __local unsigned short *s_left_count,   __local unsigned short *s_right_count,  \n");
      source.append("                         __local unsigned short *s_cl_one,       __local unsigned short *s_cl_mult,  \n");
      source.append("                         __local unsigned short *s_cl_blocking,  __local unsigned short *s_cl_helper,  \n");
      source.append("                         unsigned int is_one_lambda, unsigned int is_one_lambda_2,  \n");
      source.append("                         "); source.append(numeric_string); source.append(" *left,    \n");
      source.append("                         "); source.append(numeric_string); source.append(" *right,    \n");
      source.append("                         "); source.append(numeric_string); source.append(" *left_2,    \n");
      source.append("                         "); source.append(numeric_string); source.append(" *right_2,    \n");
      source.append("                         unsigned int *left_count, unsigned int *right_count,  \n");
      source.append("                         unsigned int *left_count_2, unsigned int *right_count_2,  \n");
      source.append("                         unsigned int c_block_iend, unsigned int c_sum_block,  \n");
      source.append("                         unsigned int c_block_iend_2, unsigned int c_sum_block_2  \n");
      source.append("                        )  \n");
      source.append("     {  \n");
      source.append("         uint glb_id = get_global_id(0); \n");
      source.append("         uint grp_id = get_group_id(0); \n");
      source.append("         uint grp_nm = get_num_groups(0); \n");
      source.append("         uint lcl_id = get_local_id(0); \n");
      source.append("         uint lcl_sz = get_local_size(0); \n");

          // cache data before performing compaction
      source.append("         *left = s_left[tid];  \n");
      source.append("         *right = s_right[tid];  \n");

      source.append("         if (tid_2 < num_threads_active)  \n");
      source.append("         {  \n");
      source.append("             *left_2 = s_left[tid_2];  \n");
      source.append("             *right_2 = s_right[tid_2];  \n");
      source.append("         }  \n");

      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // determine addresses for intervals containing multiple eigenvalues and
          // addresses for blocks of intervals
      source.append("         unsigned int ptr_w = 0;  \n");
      source.append("         unsigned int ptr_w_2 = 0;  \n");
      source.append("         unsigned int ptr_blocking_w = 0;  \n");
      source.append("         unsigned int ptr_blocking_w_2 = 0;  \n");
      source.append("           \n");
      source.append("          \n");

      source.append("         ptr_w = (1 == is_one_lambda) ? s_cl_one[tid]  \n");
      source.append("                 : s_cl_mult[tid] + *offset_mult_lambda;  \n");

      source.append("         if (0 != c_block_iend)  \n");
      source.append("         {  \n");
      source.append("             ptr_blocking_w = s_cl_blocking[tid];  \n");
      source.append("         }  \n");

      source.append("         if (tid_2 < num_threads_active)  \n");
      source.append("         {  \n");
      source.append("             ptr_w_2 = (1 == is_one_lambda_2) ? s_cl_one[tid_2]  \n");
      source.append("                       : s_cl_mult[tid_2] + *offset_mult_lambda;  \n");

      source.append("             if (0 != c_block_iend_2)  \n");
      source.append("             {  \n");
      source.append("                 ptr_blocking_w_2 = s_cl_blocking[tid_2];  \n");
      source.append("             }  \n");
      source.append("         }  \n");
      source.append("           \n");
      source.append("          \n");
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
      source.append("           \n");
          // store compactly in shared mem
      source.append("           if(tid < num_threads_active) \n");
      source.append("           { \n ");
      source.append("             s_left[ptr_w] = *left;  \n");
      source.append("             s_right[ptr_w] = *right;  \n");
      source.append("             s_left_count[ptr_w] = *left_count;  \n");
      source.append("             s_right_count[ptr_w] = *right_count;  \n");
      source.append("           } \n ");
      source.append("           \n");
      source.append("           \n");
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
      source.append("         if(tid == 1)  \n");
      source.append("         {  \n");
      source.append("           s_left[ptr_w] = *left;  \n");
      source.append("           s_right[ptr_w] = *right;  \n");
      source.append("           s_left_count[ptr_w] = *left_count;  \n");
      source.append("           s_right_count[ptr_w] = *right_count;  \n");
      source.append("           \n");
      source.append("         }  \n");
      source.append("               if (0 != c_block_iend)  \n");
      source.append("         {  \n");
      source.append("             s_cl_blocking[ptr_blocking_w + 1] = c_block_iend - 1;  \n");
      source.append("             s_cl_helper[ptr_blocking_w + 1] = c_sum_block;  \n");
      source.append("         }  \n");
      source.append("           \n");
      source.append("         if (tid_2 < num_threads_active)  \n");
      source.append("         {  \n");
              // store compactly in shared mem
      source.append("             s_left[ptr_w_2] = *left_2;  \n");
      source.append("             s_right[ptr_w_2] = *right_2;  \n");
      source.append("             s_left_count[ptr_w_2] = *left_count_2;  \n");
      source.append("             s_right_count[ptr_w_2] = *right_count_2;  \n");

      source.append("             if (0 != c_block_iend_2)  \n");
      source.append("             {  \n");
      source.append("                 s_cl_blocking[ptr_blocking_w_2 + 1] = c_block_iend_2 - 1;  \n");
      source.append("                 s_cl_helper[ptr_blocking_w_2 + 1] = c_sum_block_2;  \n");
      source.append("             }  \n");
      source.append("         }  \n");

      source.append("     }  \n");
  }



  ////////////////////////////////////////////////////////////////////////////////
  // Compute addresses to obtain compact list of block start addresses
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_scanCompactBlocksStartAddress(StringType & source)
  {
      source.append("       \n");
      source.append("     void  \n");
      source.append("     scanCompactBlocksStartAddress(const unsigned int tid, const unsigned int tid_2,  \n");
      source.append("                                   const unsigned int num_threads_compaction,  \n");
      source.append("                                   __local unsigned short *s_cl_blocking,  \n");
      source.append("                                   __local unsigned short *s_cl_helper  \n");
      source.append("                                  )  \n");
      source.append("     {  \n");
      source.append("         uint glb_id = get_global_id(0); \n");
      source.append("         uint grp_id = get_group_id(0); \n");
      source.append("         uint grp_nm = get_num_groups(0); \n");
      source.append("         uint lcl_id = get_local_id(0); \n");
      source.append("         uint lcl_sz = get_local_size(0); \n");

          // prepare for second step of block generation: compaction of the block
          // list itself to efficiently write out these
      source.append("         s_cl_blocking[tid] = s_cl_helper[tid];  \n");

      source.append("         if (tid_2 < num_threads_compaction)  \n");
      source.append("         {  \n");
      source.append("             s_cl_blocking[tid_2] = s_cl_helper[tid_2];  \n");
      source.append("         }  \n");

      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // additional scan to compact s_cl_blocking that permits to generate a
          // compact list of eigenvalue blocks each one containing about
          // VIENNACL_BISECT_MAX_THREADS_BLOCK eigenvalues (so that each of these blocks may be
          // processed by one thread block in a subsequent processing step

      source.append("         unsigned int offset = 1;  \n");

          // build scan tree
      source.append("         for (int d = (num_threads_compaction >> 1); d > 0; d >>= 1)  \n");
      source.append("         {  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("             if (tid < d)  \n");
      source.append("             {  \n");

      source.append("                 unsigned int  ai = offset*(2*tid+1)-1;  \n");
      source.append("                 unsigned int  bi = offset*(2*tid+2)-1;  \n");
      source.append("                 s_cl_blocking[bi] = s_cl_blocking[bi] + s_cl_blocking[ai];  \n");
      source.append("             }  \n");

      source.append("             offset <<= 1;  \n");
      source.append("         }  \n");

          // traverse down tree: first down to level 2 across
      source.append("         for (int d = 2; d < num_threads_compaction; d <<= 1)  \n");
      source.append("         {  \n");

      source.append("             offset >>= 1;  \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              //
      source.append("             if (tid < (d-1))  \n");
      source.append("             {  \n");

      source.append("                 unsigned int  ai = offset*(tid+1) - 1;  \n");
      source.append("                 unsigned int  bi = ai + (offset >> 1);  \n");
      source.append("                 s_cl_blocking[bi] = s_cl_blocking[bi] + s_cl_blocking[ai];  \n");
      source.append("             }  \n");
      source.append("         }  \n");

      source.append("     }  \n");


  }


  ////////////////////////////////////////////////////////////////////////////////
  // Perform scan to obtain number of eigenvalues before a specific block
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_scanSumBlocks(StringType & source)
  {
      source.append("       \n");
      source.append("     void  \n");
      source.append("     scanSumBlocks(const unsigned int tid, const unsigned int tid_2,  \n");
      source.append("                   const unsigned int num_threads_active,  \n");
      source.append("                   const unsigned int num_threads_compaction,  \n");
      source.append("                   __local unsigned short *s_cl_blocking,  \n");
      source.append("                   __local unsigned short *s_cl_helper)  \n");
      source.append("     {  \n");
      source.append("         uint glb_id = get_global_id(0); \n");
      source.append("         uint grp_id = get_group_id(0); \n");
      source.append("         uint grp_nm = get_num_groups(0); \n");
      source.append("         uint lcl_id = get_local_id(0); \n");
      source.append("         uint lcl_sz = get_local_size(0); \n");

      source.append("         unsigned int offset = 1;  \n");

          // first step of scan to build the sum of elements within each block
          // build up tree
      source.append("         for (int d = num_threads_compaction >> 1; d > 0; d >>= 1)  \n");
      source.append("         {  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("             if (tid < d)  \n");
      source.append("             {  \n");

      source.append("                 unsigned int ai = offset*(2*tid+1)-1;  \n");
      source.append("                 unsigned int bi = offset*(2*tid+2)-1;  \n");

      source.append("                 s_cl_blocking[bi] += s_cl_blocking[ai];  \n");
      source.append("             }  \n");

      source.append("             offset *= 2;  \n");
      source.append("         }  \n");

          // first step of scan to build the sum of elements within each block
          // traverse down tree
      source.append("         for (int d = 2; d < (num_threads_compaction - 1); d <<= 1)  \n");
      source.append("         {  \n");

      source.append("             offset >>= 1;  \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("             if (tid < (d-1))  \n");
      source.append("             {  \n");
      source.append("                 unsigned int ai = offset*(tid+1) - 1;  \n");
      source.append("                 unsigned int bi = ai + (offset >> 1);  \n");
      source.append("                 s_cl_blocking[bi] += s_cl_blocking[ai];  \n");
      source.append("             }  \n");
      source.append("         }  \n");
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("         if (0 == tid)  \n");
      source.append("         {  \n");

              // move last element of scan to last element that is valid
              // necessary because the number of threads employed for scan is a power
              // of two and not necessarily the number of active threasd
      source.append("             s_cl_helper[num_threads_active - 1] =  \n");
      source.append("                 s_cl_helper[num_threads_compaction - 1];  \n");
      source.append("             s_cl_blocking[num_threads_active - 1] =  \n");
      source.append("                 s_cl_blocking[num_threads_compaction - 1];  \n");
      source.append("         }  \n");
      source.append("     }  \n");


  }

  ////////////////////////////////////////////////////////////////////////////////
  // Perform initial scan for compaction of intervals containing one and
  // multiple eigenvalues; also do initial scan to build blocks
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_scanInitial(StringType & source)
  {
      source.append("       \n");
      source.append("     void  \n");
      source.append("     scanInitial(const unsigned int tid, const unsigned int tid_2, const unsigned int n,  \n");
      source.append("                 const unsigned int num_threads_active,  \n");
      source.append("                 const unsigned int num_threads_compaction,  \n");
      source.append("                 __local unsigned short *s_cl_one, __local unsigned short *s_cl_mult,  \n");
      source.append("                 __local unsigned short *s_cl_blocking, __local unsigned short *s_cl_helper  \n");
      source.append("                )  \n");
      source.append("     {  \n");
      source.append("         uint glb_id = get_global_id(0); \n");
      source.append("         uint grp_id = get_group_id(0); \n");
      source.append("         uint grp_nm = get_num_groups(0); \n");
      source.append("         uint lcl_id = get_local_id(0); \n");
      source.append("         uint lcl_sz = get_local_size(0); \n");


          // perform scan to compactly write out the intervals containing one and
          // multiple eigenvalues
          // also generate tree for blocking of intervals containing multiple
          // eigenvalues

      source.append("         unsigned int offset = 1;  \n");

          // build scan tree
      source.append("         for (int d = (num_threads_compaction >> 1); d > 0; d >>= 1)  \n");
      source.append("         {  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("             if (tid < d)  \n");
      source.append("             {  \n");

      source.append("                 unsigned int  ai = offset*(2*tid+1);  \n");
      source.append("                 unsigned int  bi = offset*(2*tid+2)-1;  \n");

      source.append("                 s_cl_one[bi] = s_cl_one[bi] + s_cl_one[ai - 1];  \n");
      source.append("                 s_cl_mult[bi] = s_cl_mult[bi] + s_cl_mult[ai - 1];  \n");

                  // s_cl_helper is binary and zero for an internal node and 1 for a
                  // root node of a tree corresponding to a block
                  // s_cl_blocking contains the number of nodes in each sub-tree at each
                  // iteration, the data has to be kept to compute the total number of
                  // eigenvalues per block that, in turn, is needed to efficiently
                  // write out data in the second step
      source.append("                 if ((s_cl_helper[ai - 1] != 1) || (s_cl_helper[bi] != 1))  \n");
      source.append("                 {  \n");

                      // check how many childs are non terminated
      source.append("                     if (s_cl_helper[ai - 1] == 1)  \n");
      source.append("                     {  \n");
                          // mark as terminated
      source.append("                         s_cl_helper[bi] = 1;  \n");
      source.append("                     }  \n");
      source.append("                     else if (s_cl_helper[bi] == 1)  \n");
      source.append("                     {  \n");
                          // mark as terminated
      source.append("                         s_cl_helper[ai - 1] = 1;  \n");
      source.append("                     }  \n");
      source.append("               else      \n");   // both childs are non-terminated
      source.append("                     {  \n");

      source.append("                         unsigned int temp = s_cl_blocking[bi] + s_cl_blocking[ai - 1];  \n");

      source.append("                         if (temp > (n > 512 ? VIENNACL_BISECT_MAX_THREADS_BLOCK : VIENNACL_BISECT_MAX_THREADS_BLOCK / 2))  \n");
      source.append("                         {  \n");

                              // the two child trees have to form separate blocks, terminate trees
      source.append("                             s_cl_helper[ai - 1] = 1;  \n");
      source.append("                             s_cl_helper[bi] = 1;  \n");
      source.append("                         }  \n");
      source.append("                         else  \n");
      source.append("                         {  \n");
                              // build up tree by joining subtrees
      source.append("                             s_cl_blocking[bi] = temp;  \n");
      source.append("                             s_cl_blocking[ai - 1] = 0;  \n");
      source.append("                         }  \n");
      source.append("                     }  \n");
      source.append("            } \n"); // end s_cl_helper update
      source.append("             }  \n");
      source.append("             offset <<= 1;  \n");
      source.append("         }  \n");


          // traverse down tree, this only for stream compaction, not for block
          // construction
      source.append("         for (int d = 2; d < num_threads_compaction; d <<= 1)  \n");
      source.append("         {  \n");
      source.append("             offset >>= 1;  \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
              //
      source.append("             if (tid < (d-1))  \n");
      source.append("             {  \n");
      source.append("                 unsigned int  ai = offset*(tid+1) - 1;  \n");
      source.append("                 unsigned int  bi = ai + (offset >> 1);  \n");
      source.append("                 s_cl_one[bi] = s_cl_one[bi] + s_cl_one[ai];  \n");
      source.append("                 s_cl_mult[bi] = s_cl_mult[bi] + s_cl_mult[ai];  \n");
      source.append("             }  \n");
      source.append("         }  \n");
      source.append("     }  \n");
  }

  ////////////////////////////////////////////////////////////////////////////////
  // Bisection to find eigenvalues of a real, symmetric, and tridiagonal matrix
  //
  // g_d           diagonal elements in global memory
  // g_s           superdiagonal elements in global elements (stored so that the element *(g_s - 1) can be accessed an equals 0
  // n             size of matrix
  // lg            lower bound of input interval (e.g. Gerschgorin interval)
  // ug            upper bound of input interval (e.g. Gerschgorin interval)
  // lg_eig_count  number of eigenvalues that are smaller than \a lg
  // lu_eig_count  number of eigenvalues that are smaller than \a lu
  // epsilon       desired accuracy of eigenvalues to compute
  ////////////////////////////////////////////////////////////////////////////////

  template <typename StringType>
  void generate_bisect_kernel_bisectKernelLarge(StringType & source, std::string const & numeric_string)
  {
      source.append("     __kernel  \n");
      source.append("     void  \n");
      source.append("     bisectKernelLarge(__global "); source.append(numeric_string); source.append(" *g_d,    \n");
      source.append("                       __global "); source.append(numeric_string); source.append(" *g_s,    \n");
      source.append("                       const unsigned int n,  \n");
      source.append("                       const "); source.append(numeric_string); source.append(" lg,     \n");
      source.append("                       const "); source.append(numeric_string); source.append(" ug,     \n");
      source.append("                       const unsigned int lg_eig_count,  \n");
      source.append("                       const unsigned int ug_eig_count,  \n");
      source.append("                       "); source.append(numeric_string); source.append(" epsilon,  \n");
      source.append("                       __global unsigned int *g_num_one,  \n");
      source.append("                       __global unsigned int *g_num_blocks_mult,  \n");
      source.append("                       __global "); source.append(numeric_string); source.append(" *g_left_one,    \n");
      source.append("                       __global "); source.append(numeric_string); source.append(" *g_right_one,   \n");
      source.append("                       __global unsigned int *g_pos_one,  \n");
      source.append("                       __global "); source.append(numeric_string); source.append(" *g_left_mult,   \n");
      source.append("                       __global "); source.append(numeric_string); source.append(" *g_right_mult,  \n");
      source.append("                       __global unsigned int *g_left_count_mult,  \n");
      source.append("                       __global unsigned int *g_right_count_mult,  \n");
      source.append("                       __global unsigned int *g_blocks_mult,  \n");
      source.append("                       __global unsigned int *g_blocks_mult_sum  \n");
      source.append("                      )  \n");
      source.append("     {  \n");
      source.append("         g_s = g_s + 1; \n");
      source.append("         uint glb_id = get_global_id(0); \n");
      source.append("         uint grp_id = get_group_id(0); \n");
      source.append("         uint grp_nm = get_num_groups(0); \n");
      source.append("         uint lcl_id = get_local_id(0); \n");
      source.append("         uint lcl_sz = get_local_size(0); \n");

      source.append("         const unsigned int tid = lcl_id;  \n");

          // intervals (store left and right because the subdivision tree is in general
          // not dense
      source.append("         __local "); source.append(numeric_string); source.append("  s_left[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK + 1];  \n");
      source.append("         __local "); source.append(numeric_string); source.append("  s_right[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK + 1];  \n");

          // number of eigenvalues that are smaller than s_left / s_right
          // (correspondence is realized via indices)
      source.append("         __local  unsigned short  s_left_count[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK + 1];  \n");
      source.append("         __local  unsigned short  s_right_count[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK + 1];  \n");

          // helper for stream compaction
      source.append("         __local  unsigned short  s_compaction_list[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK + 1];  \n");

          // state variables for whole block
          // if 0 then compaction of second chunk of child intervals is not necessary
          // (because all intervals had exactly one non-dead child)
      source.append("         __local  unsigned int compact_second_chunk;  \n");
          // if 1 then all threads are converged
      source.append("         __local  unsigned int all_threads_converged;  \n");

          // number of currently active threads
      source.append("         __local  unsigned int num_threads_active;  \n");

          // number of threads to use for stream compaction
      source.append("         __local  unsigned int num_threads_compaction;  \n");

          // helper for exclusive scan
      source.append("         __local unsigned short *s_compaction_list_exc = s_compaction_list + 1;  \n");

          // variables for currently processed interval
          // left and right limit of active interval
      source.append("         "); source.append(numeric_string); source.append(" left = 0.0f;  \n");
      source.append("         "); source.append(numeric_string); source.append(" right = 0.0f;  \n");
      source.append("         unsigned int left_count = 0;  \n");
      source.append("         unsigned int right_count = 0;  \n");
          // midpoint of active interval
      source.append("         "); source.append(numeric_string); source.append(" mid = 0.0f;  \n");
          // number of eigenvalues smaller then mid
      source.append("         unsigned int mid_count = 0;  \n");
          // helper for stream compaction (tracking of threads generating second child)
      source.append("         unsigned int is_active_second = 0;  \n");

          // initialize lists
      source.append("         s_compaction_list[tid] = 0;  \n");
      source.append("         s_left[tid] = 0;  \n");
      source.append("         s_right[tid] = 0;  \n");
      source.append("         s_left_count[tid] = 0;  \n");
      source.append("         s_right_count[tid] = 0;  \n");

      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // set up initial configuration
      source.append("         if (0 == tid)  \n");
      source.append("         {  \n");

      source.append("             s_left[0] = lg;  \n");
      source.append("             s_right[0] = ug;  \n");
      source.append("             s_left_count[0] = lg_eig_count;  \n");
      source.append("             s_right_count[0] = ug_eig_count;  \n");

      source.append("             compact_second_chunk = 0;  \n");
      source.append("             num_threads_active = 1;  \n");

      source.append("             num_threads_compaction = 1;  \n");

      source.append("             all_threads_converged = 1;  \n");
      source.append("         }  \n");

      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // for all active threads read intervals from the last level
          // the number of (worst case) active threads per level l is 2^l
          // determine coarse intervals. On these intervals the kernel for one or for multiple eigenvalues
          // will be executed in the second step
      source.append("    while( true )    \n");
      source.append("         {  \n");
      source.append("             s_compaction_list[tid] = 0;  \n");
      source.append("             s_compaction_list[tid + VIENNACL_BISECT_MAX_THREADS_BLOCK] = 0;  \n");
      source.append("             s_compaction_list[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK] = 0;  \n");
      source.append("             subdivideActiveIntervalShort(tid, s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                                     num_threads_active,  \n");
      source.append("                                     &left, &right, &left_count, &right_count,  \n");
      source.append("                                     &mid, &all_threads_converged);  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // check if done
      source.append("             if (1 == all_threads_converged)  \n");
      source.append("             {  \n");
      source.append("                 break;  \n");
      source.append("             }  \n");

              // compute number of eigenvalues smaller than mid
              // use all threads for reading the necessary matrix data from global
              // memory
              // use s_left and s_right as scratch space for diagonal and
              // superdiagonal of matrix
      source.append("             mid_count = computeNumSmallerEigenvalsLarge(g_d, g_s, n,  \n");
      source.append("                                                         mid, lcl_id,  \n");
      source.append("                                                         num_threads_active,  \n");
      source.append("                                                         s_left, s_right,  \n");
      source.append("                                                         (left == right));  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // store intervals
              // for all threads store the first child interval in a continuous chunk of
              // memory, and the second child interval -- if it exists -- in a second
              // chunk; it is likely that all threads reach convergence up to
              // \a epsilon at the same level; furthermore, for higher level most / all
              // threads will have only one child, storing the first child compactly will
              // (first) avoid to perform a compaction step on the first chunk, (second)
              // make it for higher levels (when all threads / intervals have
              // exactly one child)  unnecessary to perform a compaction of the second
              // chunk
      source.append("             if (tid < num_threads_active)  \n");
      source.append("             {  \n");

      source.append("                 if (left != right)  \n");
      source.append("                 {  \n");

                      // store intervals
      source.append("                     storeNonEmptyIntervalsLarge(tid, num_threads_active,  \n");
      source.append("                                                 s_left, s_right,  \n");
      source.append("                                                 s_left_count, s_right_count,  \n");
      source.append("                                                 left, mid, right,  \n");
      source.append("                                                 left_count, mid_count, right_count,  \n");
      source.append("                                                 epsilon, &compact_second_chunk,  \n");
      source.append("                                                 s_compaction_list_exc,  \n");
      source.append("                                                 &is_active_second);  \n");
      source.append("                 }  \n");
      source.append("                 else  \n");
      source.append("                 {  \n");

                      // re-write converged interval (has to be stored again because s_left
                      // and s_right are used as scratch space for
                      // computeNumSmallerEigenvalsLarge()
      source.append("                     s_left[tid] = left;  \n");
      source.append("                     s_right[tid] = left;  \n");
      source.append("                     s_left_count[tid] = left_count;  \n");
      source.append("                     s_right_count[tid] = right_count;  \n");

      source.append("                     is_active_second = 0;  \n");
      source.append("                 }  \n");
      source.append("             }  \n");

              // necessary so that compact_second_chunk is up-to-date
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // perform compaction of chunk where second children are stored
              // scan of (num_threads_active / 2) elements, thus at most
              // (num_threads_active / 4) threads are needed
      source.append("             if (compact_second_chunk > 0)  \n");
      source.append("             {  \n");

                  // create indices for compaction
      source.append("                 createIndicesCompactionShort(s_compaction_list_exc, num_threads_compaction);  \n");
      source.append("             }  \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
      source.append("               \n");
      source.append("        if (compact_second_chunk > 0)               \n");
      source.append("             {  \n");
      source.append("                 compactIntervalsShort(s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                                  mid, right, mid_count, right_count,  \n");
      source.append("                                  s_compaction_list, num_threads_active,  \n");
      source.append("                                  is_active_second);  \n");
      source.append("             }  \n");

      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

              // update state variables
      source.append("             if (0 == tid)  \n");
      source.append("             {  \n");

                  // update number of active threads with result of reduction
      source.append("                 num_threads_active += s_compaction_list[num_threads_active];  \n");
      source.append("                 num_threads_compaction = ceilPow2(num_threads_active);  \n");

      source.append("                 compact_second_chunk = 0;  \n");
      source.append("                 all_threads_converged = 1;  \n");
      source.append("             }  \n");
      source.append("             barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
      source.append("             if (num_threads_compaction > lcl_sz)  \n");
      source.append("             {  \n");
      source.append("                 break;  \n");
      source.append("             }  \n");
      source.append("         }  \n");
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // generate two lists of intervals; one with intervals that contain one
          // eigenvalue (or are converged), and one with intervals that need further
          // subdivision

          // perform two scans in parallel

      source.append("         unsigned int left_count_2;  \n");
      source.append("         unsigned int right_count_2;  \n");

      source.append("         unsigned int tid_2 = tid + lcl_sz;  \n");

          // cache in per thread registers so that s_left_count and s_right_count
          // can be used for scans
      source.append("         left_count = s_left_count[tid];  \n");
      source.append("         right_count = s_right_count[tid];  \n");

          // some threads have to cache data for two intervals
      source.append("         if (tid_2 < num_threads_active)  \n");
      source.append("         {  \n");
      source.append("             left_count_2 = s_left_count[tid_2];  \n");
      source.append("             right_count_2 = s_right_count[tid_2];  \n");
      source.append("         }  \n");

          // compaction list for intervals containing one and multiple eigenvalues
          // do not affect first element for exclusive scan
      source.append("         __local unsigned short  *s_cl_one = s_left_count + 1;  \n");
      source.append("         __local unsigned short  *s_cl_mult = s_right_count + 1;  \n");

          // compaction list for generating blocks of intervals containing multiple
          // eigenvalues
      source.append("         __local unsigned short  *s_cl_blocking = s_compaction_list_exc;  \n");
          // helper compaction list for generating blocks of intervals
      source.append("         __local unsigned short  s_cl_helper[2 * VIENNACL_BISECT_MAX_THREADS_BLOCK + 1];  \n");

      source.append("         if (0 == tid)  \n");
      source.append("         {  \n");
              // set to 0 for exclusive scan
      source.append("             s_left_count[0] = 0;  \n");
      source.append("             s_right_count[0] = 0;  \n");
      source.append("              \n");
      source.append("         }  \n");

      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // flag if interval contains one or multiple eigenvalues
      source.append("         unsigned int is_one_lambda = 0;  \n");
      source.append("         unsigned int is_one_lambda_2 = 0;  \n");

          // number of eigenvalues in the interval
      source.append("         unsigned int multiplicity = right_count - left_count;  \n");
      source.append("         is_one_lambda = (1 == multiplicity);  \n");

      source.append("         s_cl_one[tid] = is_one_lambda;  \n");
      source.append("         s_cl_mult[tid] = (! is_one_lambda);  \n");

          // (note: s_cl_blocking is non-zero only where s_cl_mult[] is non-zero)
      source.append("         s_cl_blocking[tid] = (1 == is_one_lambda) ? 0 : multiplicity;  \n");
      source.append("         s_cl_helper[tid] = 0;  \n");

      source.append("         if (tid_2 < num_threads_active)  \n");
      source.append("         {  \n");

      source.append("             unsigned int multiplicity = right_count_2 - left_count_2;  \n");
      source.append("             is_one_lambda_2 = (1 == multiplicity);  \n");

      source.append("             s_cl_one[tid_2] = is_one_lambda_2;  \n");
      source.append("             s_cl_mult[tid_2] = (! is_one_lambda_2);  \n");

              // (note: s_cl_blocking is non-zero only where s_cl_mult[] is non-zero)
      source.append("             s_cl_blocking[tid_2] = (1 == is_one_lambda_2) ? 0 : multiplicity;  \n");
      source.append("             s_cl_helper[tid_2] = 0;  \n");
      source.append("         }  \n");
      source.append("         else if (tid_2 < (2 * (n > 512 ? VIENNACL_BISECT_MAX_THREADS_BLOCK : VIENNACL_BISECT_MAX_THREADS_BLOCK / 2) + 1))  \n");
      source.append("         {  \n");

              // clear
      source.append("             s_cl_blocking[tid_2] = 0;  \n");
      source.append("             s_cl_helper[tid_2] = 0;  \n");
      source.append("         }  \n");


      source.append("         scanInitial(tid, tid_2, n, num_threads_active, num_threads_compaction,  \n");
      source.append("                     s_cl_one, s_cl_mult, s_cl_blocking, s_cl_helper);  \n");
      source.append("           \n");
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("         scanSumBlocks(tid, tid_2, num_threads_active,  \n");
      source.append("                       num_threads_compaction, s_cl_blocking, s_cl_helper);  \n");

          // end down sweep of scan
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("         unsigned int  c_block_iend = 0;  \n");
      source.append("         unsigned int  c_block_iend_2 = 0;  \n");
      source.append("         unsigned int  c_sum_block = 0;  \n");
      source.append("         unsigned int  c_sum_block_2 = 0;  \n");

          // for each thread / interval that corresponds to root node of interval block
          // store start address of block and total number of eigenvalues in all blocks
          // before this block (particular thread is irrelevant, constraint is to
          // have a subset of threads so that one and only one of them is in each
          // interval)
      source.append("         if (1 == s_cl_helper[tid])  \n");
      source.append("         {  \n");

      source.append("             c_block_iend = s_cl_mult[tid] + 1;  \n");
      source.append("             c_sum_block = s_cl_blocking[tid];  \n");
      source.append("         }  \n");

      source.append("         if (1 == s_cl_helper[tid_2])  \n");
      source.append("         {  \n");

      source.append("             c_block_iend_2 = s_cl_mult[tid_2] + 1;  \n");
      source.append("             c_sum_block_2 = s_cl_blocking[tid_2];  \n");
      source.append("         }  \n");

      source.append("         scanCompactBlocksStartAddress(tid, tid_2, num_threads_compaction,  \n");
      source.append("                                       s_cl_blocking, s_cl_helper);  \n");


          // finished second scan for s_cl_blocking
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // determine the global results
      source.append("         __local  unsigned int num_blocks_mult;  \n");
      source.append("         __local  unsigned int num_mult;  \n");
      source.append("         __local  unsigned int offset_mult_lambda;  \n");

      source.append("         if (0 == tid)  \n");
      source.append("         {  \n");

      source.append("             num_blocks_mult = s_cl_blocking[num_threads_active - 1];  \n");
      source.append("             offset_mult_lambda = s_cl_one[num_threads_active - 1];  \n");
      source.append("             num_mult = s_cl_mult[num_threads_active - 1];  \n");

      source.append("             *g_num_one = offset_mult_lambda;  \n");
      source.append("             *g_num_blocks_mult = num_blocks_mult;  \n");
      source.append("         }  \n");

      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

      source.append("         "); source.append(numeric_string); source.append(" left_2, right_2;  \n");
      source.append("         --s_cl_one;  \n");
      source.append("         --s_cl_mult;  \n");
      source.append("         --s_cl_blocking;  \n");
      source.append("           \n");
      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");
      source.append("         compactStreamsFinal(tid, tid_2, num_threads_active, &offset_mult_lambda,  \n");
      source.append("                             s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                             s_cl_one, s_cl_mult, s_cl_blocking, s_cl_helper,  \n");
      source.append("                             is_one_lambda, is_one_lambda_2,  \n");
      source.append("                             &left, &right, &left_2, &right_2,  \n");
      source.append("                             &left_count, &right_count, &left_count_2, &right_count_2,  \n");
      source.append("                             c_block_iend, c_sum_block, c_block_iend_2, c_sum_block_2  \n");
      source.append("                            );  \n");

      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // final adjustment before writing out data to global memory
      source.append("         if (0 == tid)  \n");
      source.append("         {  \n");
      source.append("             s_cl_blocking[num_blocks_mult] = num_mult;  \n");
      source.append("             s_cl_helper[0] = 0;  \n");
      source.append("         }  \n");

      source.append("         barrier(CLK_LOCAL_MEM_FENCE)  ;  \n");

          // write to global memory
      source.append("         writeToGmem(tid, tid_2, num_threads_active, num_blocks_mult,  \n");
      source.append("                     g_left_one, g_right_one, g_pos_one,  \n");
      source.append("                     g_left_mult, g_right_mult, g_left_count_mult, g_right_count_mult,  \n");
      source.append("                     s_left, s_right, s_left_count, s_right_count,  \n");
      source.append("                     g_blocks_mult, g_blocks_mult_sum,  \n");
      source.append("                     s_compaction_list, s_cl_helper, offset_mult_lambda);  \n");
      source.append("                       \n");

      source.append("     }  \n");
  }

  // main kernel class
  /** @brief Main kernel class for the generation of the bisection kernels and utilities
    *
    */
  template <class NumericT>
  struct bisect_kernel
  {
    static std::string program_name()
    {
      return viennacl::ocl::type_to_string<NumericT>::apply() + "_bisect_kernel";
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

        // only generate for floating points (forces error for integers)
        if (numeric_string == "float" || numeric_string == "double")
        {
          //functions used from bisect_util.cpp
          generate_bisect_kernel_config(source);
          generate_bisect_kernel_floorPow2(source, numeric_string);
          generate_bisect_kernel_ceilPow2(source, numeric_string);
          generate_bisect_kernel_computeMidpoint(source, numeric_string);

          generate_bisect_kernel_storeInterval(source, numeric_string);
          generate_bisect_kernel_storeIntervalShort(source, numeric_string);

          generate_bisect_kernel_computeNumSmallerEigenvals(source, numeric_string);
          generate_bisect_kernel_computeNumSmallerEigenvalsLarge(source, numeric_string);

          generate_bisect_kernel_storeNonEmptyIntervals(source, numeric_string);
          generate_bisect_kernel_storeNonEmptyIntervalsLarge(source, numeric_string);

          generate_bisect_kernel_createIndicesCompaction(source);
          generate_bisect_kernel_createIndicesCompactionShort(source);

          generate_bisect_kernel_compactIntervals(source, numeric_string);
          generate_bisect_kernel_compactIntervalsShort(source, numeric_string);

          generate_bisect_kernel_storeIntervalConverged(source, numeric_string);
          generate_bisect_kernel_storeIntervalConvergedShort(source, numeric_string);

          generate_bisect_kernel_subdivideActiveInterval(source, numeric_string);
          generate_bisect_kernel_subdivideActiveIntervalShort(source, numeric_string);

          generate_bisect_kernel_bisectKernel(source, numeric_string);
          generate_bisect_kernel_bisectKernelLarge_MultIntervals(source, numeric_string);
          generate_bisect_kernel_bisectKernelLarge_OneIntervals(source, numeric_string);


          generate_bisect_kernel_writeToGmem(source, numeric_string);
          generate_bisect_kernel_compactStreamsFinal(source, numeric_string);
          generate_bisect_kernel_scanCompactBlocksStartAddress(source);
          generate_bisect_kernel_scanSumBlocks(source);
          generate_bisect_kernel_scanInitial(source);
          generate_bisect_kernel_bisectKernelLarge(source, numeric_string);


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
}
}
}
}

#endif // #ifndef _BISECT_KERNEL_LARGE_H_
