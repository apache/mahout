#ifndef VIENNACL_LINALG_DETAIL_BISECT_CONFIG_HPP_
#define VIENNACL_LINALG_DETAIL_BISECT_CONFIG_HPP_

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



/** @file viennacl/linalg/detail//bisect/config.hpp
 *     @brief Global configuration parameters
 *
 *         Implementation based on the sample provided with the CUDA 6.0 SDK, for which
 *             the creation of derivative works is allowed by including the following statement:
 *                 "This software contains source code provided by NVIDIA Corporation."
 *                 */

// should be power of two
#define  VIENNACL_BISECT_MAX_THREADS_BLOCK                256

#ifdef VIENNACL_WITH_OPENCL
#  define VIENNACL_BISECT_MAX_SMALL_MATRIX                 256
#  define VIENNACL_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX   256
#else                                                          // if CUDA is used
#  define VIENNACL_BISECT_MAX_THREADS_BLOCK_SMALL_MATRIX   512 // change to 256 if errors occur
#  define VIENNACL_BISECT_MAX_SMALL_MATRIX                 512 // change to 256 if errors occur
#endif

 #define  VIENNACL_BISECT_MIN_ABS_INTERVAL                 5.0e-37

#endif 
