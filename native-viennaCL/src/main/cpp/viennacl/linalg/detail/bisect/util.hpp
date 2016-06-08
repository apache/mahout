#ifndef VIENNACL_LINALG_DETAIL_BISECT_UTIL_HPP_
#define VIENNACL_LINALG_DETAIL_BISECT_UTIL_HPP_

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


/** @file viennacl/linalg/detail//bisect/util.hpp
    @brief Utility functions

    Implementation based on the sample provided with the CUDA 6.0 SDK, for which
    the creation of derivative works is allowed by including the following statement:
    "This software contains source code provided by NVIDIA Corporation."
*/

namespace viennacl
{
namespace linalg
{
namespace detail
{

////////////////////////////////////////////////////////////////////////////////
//! Minimum
////////////////////////////////////////////////////////////////////////////////
template<class T>
#ifdef __CUDACC__
__host__  __device__
#endif
T
min(const T &lhs, const T &rhs)
{

    return (lhs < rhs) ? lhs : rhs;
}

////////////////////////////////////////////////////////////////////////////////
//! Maximum
////////////////////////////////////////////////////////////////////////////////
template<class T>
#ifdef __CUDACC__
__host__  __device__
#endif
T
max(const T &lhs, const T &rhs)
{

    return (lhs < rhs) ? rhs : lhs;
}

////////////////////////////////////////////////////////////////////////////////
//! Sign of number (float)
////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
__host__  __device__
#endif
inline float
sign_f(const float &val)
{
    return (val < 0.0f) ? -1.0f : 1.0f;
}

////////////////////////////////////////////////////////////////////////////////
//! Sign of number (double)
////////////////////////////////////////////////////////////////////////////////
#ifdef __CUDACC__
__host__  __device__
#endif
inline double
sign_d(const double &val)
{
    return (val < 0.0) ? -1.0 : 1.0;
}

///////////////////////////////////////////////////////////////////////////////
//! Get the number of blocks that are required to process \a num_threads with
//! \a num_threads_blocks threads per block
///////////////////////////////////////////////////////////////////////////////
extern "C"
inline
unsigned int
getNumBlocksLinear(const unsigned int num_threads,
                   const unsigned int num_threads_block)
{
    const unsigned int block_rem =
        ((num_threads % num_threads_block) != 0) ? 1 : 0;
    return (num_threads / num_threads_block) + block_rem;
}
} // namespace detail
} // namespace linalg
} // namespace viennacl
#endif // #ifndef VIENNACL_LINALG_DETAIL_UTIL_HPP_
