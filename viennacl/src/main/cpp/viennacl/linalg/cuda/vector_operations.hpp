#ifndef VIENNACL_LINALG_CUDA_VECTOR_OPERATIONS_HPP_
#define VIENNACL_LINALG_CUDA_VECTOR_OPERATIONS_HPP_

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

/** @file viennacl/linalg/cuda/vector_operations.hpp
    @brief Implementations of vector operations using a plain single-threaded execution on CPU
*/

#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/stride.hpp"

#include "viennacl/linalg/cuda/common.hpp"

namespace viennacl
{
namespace linalg
{
namespace cuda
{

//
// Introductory note: By convention, all dimensions are already checked in the dispatcher frontend. No need to double-check again in here!
//
template<typename DestNumericT, typename SrcNumericT>
__global__ void convert_kernel(DestNumericT      * dest, unsigned int start_dest, unsigned int inc_dest, unsigned int size_dest,
                               SrcNumericT const * src,  unsigned int start_src,  unsigned int inc_src)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                    i < size_dest;
                    i += gridDim.x * blockDim.x)
    dest[i*inc_dest+start_dest] = src[i*inc_src+start_src];
}


template<typename DestNumericT, typename SrcNumericT>
void convert(vector_base<DestNumericT> & dest, vector_base<SrcNumericT> const & src)
{
  convert_kernel<<<128, 128>>>(viennacl::cuda_arg(dest),
                              static_cast<unsigned int>(viennacl::traits::start(dest)),
                              static_cast<unsigned int>(viennacl::traits::stride(dest)),
                              static_cast<unsigned int>(viennacl::traits::size(dest)),

                              viennacl::cuda_arg(src),
                              static_cast<unsigned int>(viennacl::traits::start(src)),
                              static_cast<unsigned int>(viennacl::traits::stride(src)) );
  VIENNACL_CUDA_LAST_ERROR_CHECK("convert_kernel");
}


//////////////////////// av /////////////////////////////

// gpu scalar
template<typename NumericT>
__global__ void av_kernel(NumericT * vec1,
                          unsigned int start1,
                          unsigned int inc1,
                          unsigned int size1,

                          const NumericT * fac2,
                          unsigned int options2,
                          const NumericT * vec2,
                          unsigned int start2,
                          unsigned int inc2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  if (options2 & (1 << 1))
  {
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                      i < size1;
                      i += gridDim.x * blockDim.x)
      vec1[i*inc1+start1] = vec2[i*inc2+start2] / alpha;
  }
  else
  {
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                      i < size1;
                      i += gridDim.x * blockDim.x)
      vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha;
  }
}

// cpu scalar
template<typename NumericT>
__global__ void av_kernel(NumericT * vec1,
                          unsigned int start1,
                          unsigned int inc1,
                          unsigned int size1,

                          NumericT fac2,
                          unsigned int options2,
                          const NumericT * vec2,
                          unsigned int start2,
                          unsigned int inc2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  if (options2 & (1 << 1))
  {
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                      i < size1;
                      i += gridDim.x * blockDim.x)
      vec1[i*inc1+start1] = vec2[i*inc2+start2] / alpha;
  }
  else
  {
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                      i < size1;
                      i += gridDim.x * blockDim.x)
      vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha;
  }
}



template<typename NumericT, typename ScalarType1>
void av(vector_base<NumericT> & vec1,
        vector_base<NumericT> const & vec2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
{
  typedef NumericT        value_type;

  unsigned int options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;
  if (reciprocal_alpha)
    data_alpha = static_cast<value_type>(1) / data_alpha;

  value_type temporary_alpha = 0;
  if (viennacl::is_cpu_scalar<ScalarType1>::value)
    temporary_alpha = alpha;

  av_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                          static_cast<unsigned int>(viennacl::traits::start(vec1)),
                          static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                          static_cast<unsigned int>(viennacl::traits::size(vec1)),

                          viennacl::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                          options_alpha,
                          viennacl::cuda_arg(vec2),
                          static_cast<unsigned int>(viennacl::traits::start(vec2)),
                          static_cast<unsigned int>(viennacl::traits::stride(vec2)) );
  VIENNACL_CUDA_LAST_ERROR_CHECK("av_kernel");
}


///////////////////// avbv //////////////////////////////////

// alpha and beta on GPU
template<typename NumericT>
__global__ void avbv_kernel(NumericT * vec1,
                            unsigned int start1,
                            unsigned int inc1,
                            unsigned int size1,

                            const NumericT * fac2,
                            unsigned int options2,
                            const NumericT * vec2,
                            unsigned int start2,
                            unsigned int inc2,

                            const NumericT * fac3,
                            unsigned int options3,
                            const NumericT * vec3,
                            unsigned int start3,
                            unsigned int inc3)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
    }
  }
}

// alpha on CPU, beta on GPU
template<typename NumericT>
__global__ void avbv_kernel(NumericT * vec1,
                            unsigned int start1,
                            unsigned int inc1,
                            unsigned int size1,

                            NumericT fac2,
                            unsigned int options2,
                            const NumericT * vec2,
                            unsigned int start2,
                            unsigned int inc2,

                            const NumericT * fac3,
                            unsigned int options3,
                            const NumericT * vec3,
                            unsigned int start3,
                            unsigned int inc3)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
    }
  }
}

// alpha on GPU, beta on CPU
template<typename NumericT>
__global__ void avbv_kernel(NumericT * vec1,
                            unsigned int start1,
                            unsigned int inc1,
                            unsigned int size1,

                            const NumericT * fac2,
                            unsigned int options2,
                            const NumericT * vec2,
                            unsigned int start2,
                            unsigned int inc2,

                            NumericT fac3,
                            unsigned int options3,
                            const NumericT * vec3,
                            unsigned int start3,
                            unsigned int inc3)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
    }
  }
}

// alpha and beta on CPU
template<typename NumericT>
__global__ void avbv_kernel(NumericT * vec1,
                            unsigned int start1,
                            unsigned int inc1,
                            unsigned int size1,

                            NumericT fac2,
                            unsigned int options2,
                            const NumericT * vec2,
                            unsigned int start2,
                            unsigned int inc2,

                            NumericT fac3,
                            unsigned int options3,
                            const NumericT * vec3,
                            unsigned int start3,
                            unsigned int inc3)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] = vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
    }
  }
}




template<typename NumericT, typename ScalarT1, typename ScalarT2>
void avbv(vector_base<NumericT> & vec1,
          vector_base<NumericT> const & vec2, ScalarT1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
          vector_base<NumericT> const & vec3, ScalarT2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef NumericT        value_type;

  unsigned int options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;
  if (reciprocal_alpha)
    data_alpha = static_cast<value_type>(1) / data_alpha;

  value_type temporary_alpha = 0;
  if (viennacl::is_cpu_scalar<ScalarT1>::value)
    temporary_alpha = alpha;

  unsigned int options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  value_type temporary_beta = 0;
  if (viennacl::is_cpu_scalar<ScalarT2>::value)
    temporary_beta = beta;


  avbv_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                            static_cast<unsigned int>(viennacl::traits::start(vec1)),
                            static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                            static_cast<unsigned int>(viennacl::traits::size(vec1)),

                            viennacl::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                            options_alpha,
                            viennacl::cuda_arg(vec2),
                            static_cast<unsigned int>(viennacl::traits::start(vec2)),
                            static_cast<unsigned int>(viennacl::traits::stride(vec2)),

                            viennacl::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)),
                            options_beta,
                            viennacl::cuda_arg(vec3),
                            static_cast<unsigned int>(viennacl::traits::start(vec3)),
                            static_cast<unsigned int>(viennacl::traits::stride(vec3)) );
  VIENNACL_CUDA_LAST_ERROR_CHECK("avbv_kernel");
}


////////////////////////// avbv_v //////////////////////////////////////


// alpha and beta on GPU
template<typename NumericT>
__global__ void avbv_v_kernel(NumericT * vec1,
                              unsigned int start1,
                              unsigned int inc1,
                              unsigned int size1,

                              const NumericT * fac2,
                              unsigned int options2,
                              const NumericT * vec2,
                              unsigned int start2,
                              unsigned int inc2,

                              const NumericT * fac3,
                              unsigned int options3,
                              const NumericT * vec3,
                              unsigned int start3,
                              unsigned int inc3)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
    }
  }
}

// alpha on CPU, beta on GPU
template<typename NumericT>
__global__ void avbv_v_kernel(NumericT * vec1,
                              unsigned int start1,
                              unsigned int inc1,
                              unsigned int size1,

                              NumericT fac2,
                              unsigned int options2,
                              const NumericT * vec2,
                              unsigned int start2,
                              unsigned int inc2,

                              const NumericT * fac3,
                              unsigned int options3,
                              const NumericT * vec3,
                              unsigned int start3,
                              unsigned int inc3)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
    }
  }
}

// alpha on GPU, beta on CPU
template<typename NumericT>
__global__ void avbv_v_kernel(NumericT * vec1,
                              unsigned int start1,
                              unsigned int inc1,
                              unsigned int size1,

                              const NumericT * fac2,
                              unsigned int options2,
                              const NumericT * vec2,
                              unsigned int start2,
                              unsigned int inc2,

                              NumericT fac3,
                              unsigned int options3,
                              const NumericT * vec3,
                              unsigned int start3,
                              unsigned int inc3)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
    }
  }
}

// alpha and beta on CPU
template<typename NumericT>
__global__ void avbv_v_kernel(NumericT * vec1,
                              unsigned int start1,
                              unsigned int inc1,
                              unsigned int size1,

                              NumericT fac2,
                              unsigned int options2,
                              const NumericT * vec2,
                              unsigned int start2,
                              unsigned int inc2,

                              NumericT fac3,
                              unsigned int options3,
                              const NumericT * vec3,
                              unsigned int start3,
                              unsigned int inc3)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] / alpha + vec3[i*inc3+start3] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] / beta;
    }
    else
    {
      for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                        i < size1;
                        i += gridDim.x * blockDim.x)
        vec1[i*inc1+start1] += vec2[i*inc2+start2] * alpha + vec3[i*inc3+start3] * beta;
    }
  }
}


template<typename NumericT, typename ScalarT1, typename ScalarT2>
void avbv_v(vector_base<NumericT> & vec1,
            vector_base<NumericT> const & vec2, ScalarT1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
            vector_base<NumericT> const & vec3, ScalarT2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
{
  typedef NumericT        value_type;

  unsigned int options_alpha = detail::make_options(len_alpha, reciprocal_alpha, flip_sign_alpha);

  value_type data_alpha = alpha;
  if (flip_sign_alpha)
    data_alpha = -data_alpha;
  if (reciprocal_alpha)
    data_alpha = static_cast<value_type>(1) / data_alpha;

  value_type temporary_alpha = 0;
  if (viennacl::is_cpu_scalar<ScalarT1>::value)
    temporary_alpha = alpha;

  unsigned int options_beta  = detail::make_options(len_beta,  reciprocal_beta,  flip_sign_beta);

  value_type temporary_beta = 0;
  if (viennacl::is_cpu_scalar<ScalarT2>::value)
    temporary_beta = beta;


  avbv_v_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                              static_cast<unsigned int>(viennacl::traits::start(vec1)),
                              static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                              static_cast<unsigned int>(viennacl::traits::size(vec1)),

                              viennacl::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                              options_alpha,
                              viennacl::cuda_arg(vec2),
                              static_cast<unsigned int>(viennacl::traits::start(vec2)),
                              static_cast<unsigned int>(viennacl::traits::stride(vec2)),

                              viennacl::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)),
                              options_beta,
                              viennacl::cuda_arg(vec3),
                              static_cast<unsigned int>(viennacl::traits::start(vec3)),
                              static_cast<unsigned int>(viennacl::traits::stride(vec3)) );
}


//////////////////////////

template<typename NumericT>
__global__ void vector_assign_kernel(NumericT * vec1,
                                     unsigned int start1,
                                     unsigned int inc1,
                                     unsigned int size1,
                                     unsigned int internal_size1,

                                     NumericT alpha)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                    i < size1;
                    i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] =  (i < size1) ? alpha : 0;
}

/** @brief Assign a constant value to a vector (-range/-slice)
*
* @param vec1   The vector to which the value should be assigned
* @param alpha  The value to be assigned
* @param up_to_internal_size  Specifies whether alpha should also be written to padded memory (mostly used for clearing the whole buffer).
*/
template<typename NumericT, typename ScalarT1>
void vector_assign(vector_base<NumericT> & vec1, ScalarT1 const & alpha, bool up_to_internal_size = false)
{
  typedef NumericT        value_type;

  value_type temporary_alpha = 0;
  if (viennacl::is_cpu_scalar<ScalarT1>::value)
    temporary_alpha = alpha;

  unsigned int size = up_to_internal_size ? static_cast<unsigned int>(vec1.internal_size()) : static_cast<unsigned int>(viennacl::traits::size(vec1));

  vector_assign_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                     static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                     static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                     size,
                                     static_cast<unsigned int>(vec1.internal_size()),  //Note: Do NOT use traits::internal_size() here, because vector proxies don't require padding.

                                     viennacl::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)) );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vector_assign_kernel");
}

//////////////////////////

template<typename NumericT>
__global__ void vector_swap_kernel(NumericT * vec1,
                                   unsigned int start1,
                                   unsigned int inc1,
                                   unsigned int size1,

                                   NumericT * vec2,
                                   unsigned int start2,
                                   unsigned int inc2)
{
  NumericT tmp;
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                    i < size1;
                    i += gridDim.x * blockDim.x)
  {
    tmp = vec2[i*inc2+start2];
    vec2[i*inc2+start2] = vec1[i*inc1+start1];
    vec1[i*inc1+start1] = tmp;
  }
}


/** @brief Swaps the contents of two vectors, data is copied
*
* @param vec1   The first vector (or -range, or -slice)
* @param vec2   The second vector (or -range, or -slice)
*/
template<typename NumericT>
void vector_swap(vector_base<NumericT> & vec1, vector_base<NumericT> & vec2)
{
  vector_swap_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                   static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                   static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                   static_cast<unsigned int>(viennacl::traits::size(vec1)),

                                   viennacl::cuda_arg(vec2),
                                   static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                   static_cast<unsigned int>(viennacl::traits::stride(vec2)) );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vector_swap_kernel");
}

///////////////////////// Binary Elementwise operations /////////////

template<typename NumericT>
__global__ void element_op_kernel(NumericT * vec1,
                                   unsigned int start1,
                                   unsigned int inc1,
                                   unsigned int size1,

                                   NumericT const * vec2,
                                   unsigned int start2,
                                   unsigned int inc2,

                                   NumericT const * vec3,
                                   unsigned int start3,
                                   unsigned int inc3,

                                   unsigned int op_type
                                 )
{
  if (op_type == 2)
  {
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                      i < size1;
                      i += gridDim.x * blockDim.x)
    {
      vec1[i*inc1+start1] = pow(vec2[i*inc2+start2], vec3[i*inc3+start3]);
    }
  }
  else if (op_type == 1)
  {
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                      i < size1;
                      i += gridDim.x * blockDim.x)
    {
      vec1[i*inc1+start1] = vec2[i*inc2+start2] / vec3[i*inc3+start3];
    }
  }
  else if (op_type == 0)
  {
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                      i < size1;
                      i += gridDim.x * blockDim.x)
    {
      vec1[i*inc1+start1] = vec2[i*inc2+start2] * vec3[i*inc3+start3];
    }
  }
}

template<typename NumericT>
__global__ void element_op_int_kernel(NumericT * vec1,
                                   unsigned int start1,
                                   unsigned int inc1,
                                   unsigned int size1,

                                   NumericT const * vec2,
                                   unsigned int start2,
                                   unsigned int inc2,

                                   NumericT const * vec3,
                                   unsigned int start3,
                                   unsigned int inc3,

                                   unsigned int op_type
                                 )
{
  if (op_type == 1)
  {
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                      i < size1;
                      i += gridDim.x * blockDim.x)
    {
      vec1[i*inc1+start1] = vec2[i*inc2+start2] / vec3[i*inc3+start3];
    }
  }
  else if (op_type == 0)
  {
    for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
                      i < size1;
                      i += gridDim.x * blockDim.x)
    {
      vec1[i*inc1+start1] = vec2[i*inc2+start2] * vec3[i*inc3+start3];
    }
  }
}

/** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
*
* @param vec1   The result vector (or -range, or -slice)
* @param proxy  The proxy object holding v2, v3 and the operation
*/
template<typename NumericT, typename OpT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_binary<OpT> > const & proxy)
{
  unsigned int op_type = 2; //0: product, 1: division, 2: power
  if (viennacl::is_division<OpT>::value)
    op_type = 1;
  else if (viennacl::is_product<OpT>::value)
    op_type = 0;

  element_op_int_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),

                                  viennacl::cuda_arg(proxy.lhs()),
                                  static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                  static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs())),

                                  viennacl::cuda_arg(proxy.rhs()),
                                  static_cast<unsigned int>(viennacl::traits::start(proxy.rhs())),
                                  static_cast<unsigned int>(viennacl::traits::stride(proxy.rhs())),

                                  op_type
                                 );
  VIENNACL_CUDA_LAST_ERROR_CHECK("element_op_kernel");
}

template<typename OpT>
void element_op(vector_base<float> & vec1,
                vector_expression<const vector_base<float>, const vector_base<float>, op_element_binary<OpT> > const & proxy)
{
  unsigned int op_type = 2; //0: product, 1: division, 2: power
  if (viennacl::is_division<OpT>::value)
    op_type = 1;
  else if (viennacl::is_product<OpT>::value)
    op_type = 0;

  element_op_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),

                                  viennacl::cuda_arg(proxy.lhs()),
                                  static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                  static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs())),

                                  viennacl::cuda_arg(proxy.rhs()),
                                  static_cast<unsigned int>(viennacl::traits::start(proxy.rhs())),
                                  static_cast<unsigned int>(viennacl::traits::stride(proxy.rhs())),

                                  op_type
                                 );
  VIENNACL_CUDA_LAST_ERROR_CHECK("element_op_kernel");
}

template<typename OpT>
void element_op(vector_base<double> & vec1,
                vector_expression<const vector_base<double>, const vector_base<double>, op_element_binary<OpT> > const & proxy)
{
  unsigned int op_type = 2; //0: product, 1: division, 2: power
  if (viennacl::is_division<OpT>::value)
    op_type = 1;
  else if (viennacl::is_product<OpT>::value)
    op_type = 0;

  element_op_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),

                                  viennacl::cuda_arg(proxy.lhs()),
                                  static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                  static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs())),

                                  viennacl::cuda_arg(proxy.rhs()),
                                  static_cast<unsigned int>(viennacl::traits::start(proxy.rhs())),
                                  static_cast<unsigned int>(viennacl::traits::stride(proxy.rhs())),

                                  op_type
                                 );
  VIENNACL_CUDA_LAST_ERROR_CHECK("element_op_kernel");
}

///////////////////////// Unary Elementwise operations /////////////

// Note: Trying to automate things with macros or template metaprogramming failed (preprocessor with nvcc did not work as expected), so this is terribly hand-rolled code
// Question (Karl Rupp): Why is CUDA code always such a hassle when trying to use it in a library context?

// acos
template<typename NumericT>
__global__ void vec_element_acos_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = acos(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_acos> > const & proxy)
{
  typedef NumericT        value_type;

  vec_element_acos_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_acos_kernel");
}

// asin
template<typename NumericT>
__global__ void vec_element_asin_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = asin(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_asin> > const & proxy)
{
  vec_element_asin_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_asin_kernel");
}


// atan
template<typename NumericT>
__global__ void vec_element_atan_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = atan(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_atan> > const & proxy)
{
  vec_element_atan_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_atan_kernel");
}


// ceil
template<typename NumericT>
__global__ void vec_element_ceil_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = ceil(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_ceil> > const & proxy)
{
  vec_element_ceil_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_ceil_kernel");
}


// cos
template<typename NumericT>
__global__ void vec_element_cos_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = cos(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_cos> > const & proxy)
{
  vec_element_cos_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_cos_kernel");
}


// cosh
template<typename NumericT>
__global__ void vec_element_cosh_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = cosh(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_cosh> > const & proxy)
{
  vec_element_cosh_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_cosh_kernel");
}


// exp
template<typename NumericT>
__global__ void vec_element_exp_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = exp(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_exp> > const & proxy)
{
  vec_element_exp_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_exp_kernel");
}


// fabs
template<typename NumericT>
__global__ void vec_element_fabs_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = fabs(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_fabs> > const & proxy)
{
  vec_element_fabs_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_fabs_kernel");
}

// abs
template<typename NumericT>
__global__ void vec_element_abs_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = abs(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_abs> > const & proxy)
{
  vec_element_abs_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                       static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                       static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                       static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                       viennacl::cuda_arg(proxy.lhs()),
                                       static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                       static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                      );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_abs_kernel");
}



// floor
template<typename NumericT>
__global__ void vec_element_floor_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = floor(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_floor> > const & proxy)
{
  vec_element_floor_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_floor_kernel");
}


// log
template<typename NumericT>
__global__ void vec_element_log_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = log(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_log> > const & proxy)
{
  vec_element_log_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_log_kernel");
}


// log10
template<typename NumericT>
__global__ void vec_element_log10_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = log10(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_log10> > const & proxy)
{
  vec_element_log10_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_log10_kernel");
}


// sin
template<typename NumericT>
__global__ void vec_element_sin_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = sin(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_sin> > const & proxy)
{
  vec_element_sin_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_sin_kernel");
}


// sinh
template<typename NumericT>
__global__ void vec_element_sinh_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = sinh(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_sinh> > const & proxy)
{
  vec_element_sinh_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_sinh_kernel");
}


// sqrt
template<typename NumericT>
__global__ void vec_element_sqrt_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = sqrt(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_sqrt> > const & proxy)
{
  vec_element_sqrt_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_sqrt_kernel");
}


// tan
template<typename NumericT>
__global__ void vec_element_tan_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = tan(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_tan> > const & proxy)
{
  vec_element_tan_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_tan_kernel");
}


// tanh
template<typename NumericT>
__global__ void vec_element_tanh_kernel(
    NumericT       * vec1, unsigned int start1, unsigned int inc1, unsigned int size1,
    NumericT const * vec2, unsigned int start2, unsigned int inc2)
{
  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += gridDim.x * blockDim.x)
    vec1[i*inc1+start1] = tanh(vec2[i*inc2+start2]);
}

template<typename NumericT>
void element_op(vector_base<NumericT> & vec1,
                vector_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_element_unary<op_tanh> > const & proxy)
{
  vec_element_tanh_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                        static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                        static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                        viennacl::cuda_arg(proxy.lhs()),
                                        static_cast<unsigned int>(viennacl::traits::start(proxy.lhs())),
                                        static_cast<unsigned int>(viennacl::traits::stride(proxy.lhs()))
                                       );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vec_element_tanh_kernel");
}



///////////////////////// Norms and inner product ///////////////////


template<typename NumericT>
__global__ void inner_prod_kernel(const NumericT * vec1,
                                  unsigned int start1,
                                  unsigned int inc1,
                                  unsigned int size1,
                                  const NumericT * vec2,
                                  unsigned int start2,
                                  unsigned int inc2,
                                  unsigned int size2,
                                  NumericT * group_buffer)
{
  __shared__ NumericT tmp_buffer[128];
  unsigned int group_start1 = (blockIdx.x * size1) / (gridDim.x) * inc1 + start1;
  unsigned int group_start2 = (blockIdx.x * size2) / (gridDim.x) * inc2 + start2;

  unsigned int group_size1 = ((blockIdx.x + 1) * size1) / (gridDim.x)
                               - (  blockIdx.x * size1) / (gridDim.x);


  NumericT tmp = 0;
  for (unsigned int i = threadIdx.x; i < group_size1; i += blockDim.x)
    tmp += vec1[i*inc1+group_start1] * vec2[i*inc2+group_start2];
  tmp_buffer[threadIdx.x] = tmp;

  // parallel reduction
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
      tmp_buffer[threadIdx.x] += tmp_buffer[threadIdx.x+stride];
  }

  if (threadIdx.x == 0)
    group_buffer[blockIdx.x] = tmp_buffer[0];

}



// sums the array 'vec1' and writes to result. Makes use of a single work-group only.
template<typename NumericT>
__global__ void vector_sum_kernel_floats(
          const NumericT * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          unsigned int option, //0: use fmax, 1: just sum, 2: sum and return sqrt of sum
          NumericT * result)
{
  __shared__ NumericT tmp_buffer[128];
  NumericT thread_sum = 0;
  for (unsigned int i = threadIdx.x; i<size1; i += blockDim.x)
  {
    if (option > 0)
      thread_sum += vec1[i*inc1+start1];
    else
      thread_sum = fmax(thread_sum, fabs(vec1[i*inc1+start1]));
  }

  tmp_buffer[threadIdx.x] = thread_sum;

  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
    {
      if (option > 0)
        tmp_buffer[threadIdx.x] += tmp_buffer[threadIdx.x + stride];
      else
        tmp_buffer[threadIdx.x] = fmax(tmp_buffer[threadIdx.x], tmp_buffer[threadIdx.x + stride]);
    }
  }

  if (threadIdx.x == 0)
  {
    if (option == 2)
      *result = sqrt(tmp_buffer[0]);
    else
      *result = tmp_buffer[0];
  }
}

template<typename NumericT>
__global__ void vector_sum_kernel_integers(
          const NumericT * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          unsigned int option, //0: use max, 1: just sum
          NumericT * result)
{
  __shared__ NumericT tmp_buffer[128];
  NumericT thread_sum = 0;
  for (unsigned int i = threadIdx.x; i<size1; i += blockDim.x)
  {
    if (option > 0)
      thread_sum += vec1[i*inc1+start1];
    else
      thread_sum = thread_sum > abs(vec1[i*inc1+start1]) ? thread_sum : abs(vec1[i*inc1+start1]);
  }

  tmp_buffer[threadIdx.x] = thread_sum;

  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
    {
      if (option > 0)
        tmp_buffer[threadIdx.x] += tmp_buffer[threadIdx.x + stride];
      else
        tmp_buffer[threadIdx.x] = tmp_buffer[threadIdx.x] > tmp_buffer[threadIdx.x + stride] ? tmp_buffer[threadIdx.x] : tmp_buffer[threadIdx.x + stride];
    }
  }

  if (threadIdx.x == 0)
    *result = tmp_buffer[0];
}

template<typename NumericT>
__global__ void vector_sum_kernel_unsigned_integers(
          const NumericT * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          unsigned int option, //0: use max, 1: just sum
          NumericT * result)
{
  __shared__ NumericT tmp_buffer[128];
  NumericT thread_sum = 0;
  for (unsigned int i = threadIdx.x; i<size1; i += blockDim.x)
  {
    if (option > 0)
      thread_sum += vec1[i*inc1+start1];
    else
      thread_sum = (thread_sum > vec1[i*inc1+start1]) ? thread_sum : vec1[i*inc1+start1];
  }

  tmp_buffer[threadIdx.x] = thread_sum;

  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
    {
      if (option > 0)
        tmp_buffer[threadIdx.x] += tmp_buffer[threadIdx.x + stride];
      else
        tmp_buffer[threadIdx.x] = tmp_buffer[threadIdx.x] > tmp_buffer[threadIdx.x + stride] ? tmp_buffer[threadIdx.x] : tmp_buffer[threadIdx.x + stride];
    }
  }

  if (threadIdx.x == 0)
    *result = tmp_buffer[0];
}

namespace detail
{
  /** \cond */
  struct vector_sum_kernel_launcher_integers
  {
    template<typename NumericT, typename ScalarT>
    static void apply(vector_base<NumericT> const & temp,
                      unsigned int option,
                      ScalarT & result)
    {
      typedef NumericT        value_type;
      vector_sum_kernel_integers<<<1, 128>>>(viennacl::cuda_arg(temp),
                                            static_cast<unsigned int>(viennacl::traits::start(temp)),
                                            static_cast<unsigned int>(viennacl::traits::stride(temp)),
                                            static_cast<unsigned int>(viennacl::traits::size(temp)),
                                            static_cast<unsigned int>(option),
                                            viennacl::cuda_arg(result) );
      VIENNACL_CUDA_LAST_ERROR_CHECK("vector_sum_kernel");
    }
  };

  struct vector_sum_kernel_launcher_unsigned_integers
  {
    template<typename NumericT, typename ScalarT>
    static void apply(vector_base<NumericT> const & temp,
                      unsigned int option,
                      ScalarT & result)
    {
      typedef NumericT        value_type;
      vector_sum_kernel_unsigned_integers<<<1, 128>>>(viennacl::cuda_arg(temp),
                                                      static_cast<unsigned int>(viennacl::traits::start(temp)),
                                                      static_cast<unsigned int>(viennacl::traits::stride(temp)),
                                                      static_cast<unsigned int>(viennacl::traits::size(temp)),
                                                      static_cast<unsigned int>(option),
                                                      viennacl::cuda_arg(result) );
      VIENNACL_CUDA_LAST_ERROR_CHECK("vector_sum_kernel");
    }
  };

  struct vector_sum_kernel_launcher_floats
  {
    template<typename NumericT, typename ScalarT>
    static void apply(vector_base<NumericT> const & temp,
                      unsigned int option,
                      ScalarT & result)
    {
      typedef NumericT        value_type;
      vector_sum_kernel_floats<<<1, 128>>>(viennacl::cuda_arg(temp),
                                            static_cast<unsigned int>(viennacl::traits::start(temp)),
                                            static_cast<unsigned int>(viennacl::traits::stride(temp)),
                                            static_cast<unsigned int>(viennacl::traits::size(temp)),
                                            static_cast<unsigned int>(option),
                                            viennacl::cuda_arg(result) );
      VIENNACL_CUDA_LAST_ERROR_CHECK("vector_sum_kernel");
    }
  };

  template<typename NumericT>
  struct vector_sum_kernel_launcher : public vector_sum_kernel_launcher_integers {};

  template<>
  struct vector_sum_kernel_launcher<unsigned char>  : public vector_sum_kernel_launcher_unsigned_integers {};

  template<>
  struct vector_sum_kernel_launcher<unsigned short>  : public vector_sum_kernel_launcher_unsigned_integers {};

  template<>
  struct vector_sum_kernel_launcher<unsigned int>  : public vector_sum_kernel_launcher_unsigned_integers {};

  template<>
  struct vector_sum_kernel_launcher<unsigned long>  : public vector_sum_kernel_launcher_unsigned_integers {};

  template<>
  struct vector_sum_kernel_launcher<float>  : public vector_sum_kernel_launcher_floats {};

  template<>
  struct vector_sum_kernel_launcher<double> : public vector_sum_kernel_launcher_floats {};

  /** \endcond */
}


//implementation of inner product:
//namespace {
/** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
*
* @param vec1 The first vector
* @param vec2 The second vector
* @param result The result scalar (on the gpu)
*/
template<typename NumericT, typename ScalarT>
void inner_prod_impl(vector_base<NumericT> const & vec1,
                     vector_base<NumericT> const & vec2,
                     ScalarT & result)
{
  typedef NumericT        value_type;

  static const unsigned int work_groups = 128;
  static viennacl::vector<value_type> temp(work_groups);

  inner_prod_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                  viennacl::cuda_arg(vec2),
                                  static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec2)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec2)),
                                  viennacl::cuda_arg(temp)
                                 );
  VIENNACL_CUDA_LAST_ERROR_CHECK("inner_prod_kernel");

  detail::vector_sum_kernel_launcher<NumericT>::apply(temp, 1, result);
}


/** @brief Computes the inner product of two vectors - implementation. Library users should call inner_prod(vec1, vec2).
*
* @param vec1 The first vector
* @param vec2 The second vector
* @param result The result scalar (on the host)
*/
template<typename NumericT>
void inner_prod_cpu(vector_base<NumericT> const & vec1,
                    vector_base<NumericT> const & vec2,
                    NumericT & result)
{
  typedef NumericT        value_type;

  const unsigned int work_groups = 128;
  viennacl::vector<value_type> temp(work_groups);

  inner_prod_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                  static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                  viennacl::cuda_arg(vec2),
                                  static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                  static_cast<unsigned int>(viennacl::traits::stride(vec2)),
                                  static_cast<unsigned int>(viennacl::traits::size(vec2)),
                                  viennacl::cuda_arg(temp)
                                 );
  VIENNACL_CUDA_LAST_ERROR_CHECK("inner_prod_kernel");

  // Now copy partial results from GPU back to CPU and run reduction there:
  std::vector<value_type> temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = 0;
  for (typename std::vector<value_type>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result += *it;
}

///////////////////////////////////

#define VIENNACL_MDOT_WORKGROUP_SIZE  128
#define VIENNACL_MDOT_WORKGROUP_NUM   128
// M = 2:
template<typename NumericT>
__global__ void inner_prod_2_kernel(const NumericT *x,  unsigned int startx, unsigned int stridex, unsigned int sizex,
                                    const NumericT *y0, unsigned int start0, unsigned int stride0,
                                    const NumericT *y1, unsigned int start1, unsigned int stride1,
                                    NumericT *group_results)
{
  __shared__ NumericT tmp_buffer[2*VIENNACL_MDOT_WORKGROUP_SIZE];
  unsigned int entries_per_thread = (sizex - 1) / (blockDim.x * gridDim.x) + 1;
  unsigned int vec_start_index = blockIdx.x * blockDim.x * entries_per_thread;
  unsigned int vec_stop_index  = min((blockIdx.x + 1) * blockDim.x * entries_per_thread, sizex); // don't go beyond size of x

  NumericT entry_x    = 0;
  NumericT group_sum0 = 0;
  NumericT group_sum1 = 0;
  for (unsigned int i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i * stridex + startx];   // load only once from global memory!
    group_sum0 += entry_x * y0[i * stride0 + start0];
    group_sum1 += entry_x * y1[i * stride1 + start1];
  }
  tmp_buffer[threadIdx.x]              = group_sum0;
  tmp_buffer[threadIdx.x + blockDim.x] = group_sum1;

  // parallel reduction
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x             ] += tmp_buffer[threadIdx.x+stride             ];
      tmp_buffer[threadIdx.x + blockDim.x] += tmp_buffer[threadIdx.x+stride + blockDim.x];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x]             = tmp_buffer[0];
    group_results[blockIdx.x + gridDim.x] = tmp_buffer[blockDim.x];
  }
}

// M = 3:
template<typename NumericT>
__global__ void inner_prod_3_kernel(const NumericT *x,  unsigned int startx, unsigned int stridex, unsigned int sizex,
                                    const NumericT *y0, unsigned int start0, unsigned int stride0,
                                    const NumericT *y1, unsigned int start1, unsigned int stride1,
                                    const NumericT *y2, unsigned int start2, unsigned int stride2,
                                    NumericT *group_results)
{
  __shared__ NumericT tmp_buffer[3*VIENNACL_MDOT_WORKGROUP_SIZE];
  unsigned int entries_per_thread = (sizex - 1) / (blockDim.x * gridDim.x) + 1;
  unsigned int vec_start_index = blockIdx.x * blockDim.x * entries_per_thread;
  unsigned int vec_stop_index  = min((blockIdx.x + 1) * blockDim.x * entries_per_thread, sizex); // don't go beyond vec size

  NumericT entry_x    = 0;
  NumericT group_sum0 = 0;
  NumericT group_sum1 = 0;
  NumericT group_sum2 = 0;
  for (unsigned int i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i * stridex + startx];   // load only once from global memory!
    group_sum0 += entry_x * y0[i * stride0 + start0];
    group_sum1 += entry_x * y1[i * stride1 + start1];
    group_sum2 += entry_x * y2[i * stride2 + start2];
  }
  tmp_buffer[threadIdx.x]                  = group_sum0;
  tmp_buffer[threadIdx.x +     blockDim.x] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * blockDim.x] = group_sum2;

  // parallel reduction
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                 ] += tmp_buffer[threadIdx.x+stride                 ];
      tmp_buffer[threadIdx.x +     blockDim.x] += tmp_buffer[threadIdx.x+stride +     blockDim.x];
      tmp_buffer[threadIdx.x + 2 * blockDim.x] += tmp_buffer[threadIdx.x+stride + 2 * blockDim.x];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    blockDim.x];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * blockDim.x];
  }
}

// M = 4:
template<typename NumericT>
__global__ void inner_prod_4_kernel(const NumericT *x,  unsigned int startx, unsigned int stridex, unsigned int sizex,
                                    const NumericT *y0, unsigned int start0, unsigned int stride0,
                                    const NumericT *y1, unsigned int start1, unsigned int stride1,
                                    const NumericT *y2, unsigned int start2, unsigned int stride2,
                                    const NumericT *y3, unsigned int start3, unsigned int stride3,
                                    NumericT *group_results)
{
  __shared__ NumericT tmp_buffer[4*VIENNACL_MDOT_WORKGROUP_SIZE];
  unsigned int entries_per_thread = (sizex - 1) / (blockDim.x * gridDim.x) + 1;
  unsigned int vec_start_index = blockIdx.x * blockDim.x * entries_per_thread;
  unsigned int vec_stop_index  = min((blockIdx.x + 1) * blockDim.x * entries_per_thread, sizex); // don't go beyond vec size

  NumericT entry_x    = 0;
  NumericT group_sum0 = 0;
  NumericT group_sum1 = 0;
  NumericT group_sum2 = 0;
  NumericT group_sum3 = 0;
  for (unsigned int i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i * stridex + startx];   // load only once from global memory!
    group_sum0 += entry_x * y0[i * stride0 + start0];
    group_sum1 += entry_x * y1[i * stride1 + start1];
    group_sum2 += entry_x * y2[i * stride2 + start2];
    group_sum3 += entry_x * y3[i * stride3 + start3];
  }
  tmp_buffer[threadIdx.x]                  = group_sum0;
  tmp_buffer[threadIdx.x +     blockDim.x] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * blockDim.x] = group_sum2;
  tmp_buffer[threadIdx.x + 3 * blockDim.x] = group_sum3;

  // parallel reduction
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                 ] += tmp_buffer[threadIdx.x+stride                 ];
      tmp_buffer[threadIdx.x +     blockDim.x] += tmp_buffer[threadIdx.x+stride +     blockDim.x];
      tmp_buffer[threadIdx.x + 2 * blockDim.x] += tmp_buffer[threadIdx.x+stride + 2 * blockDim.x];
      tmp_buffer[threadIdx.x + 3 * blockDim.x] += tmp_buffer[threadIdx.x+stride + 3 * blockDim.x];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    blockDim.x];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * blockDim.x];
    group_results[blockIdx.x + 3 * gridDim.x] = tmp_buffer[3 * blockDim.x];
  }
}

// M = 8:
template<typename NumericT>
__global__ void inner_prod_8_kernel(const NumericT *x,  unsigned int startx, unsigned int stridex, unsigned int sizex,
                                    const NumericT *y0, unsigned int start0, unsigned int stride0,
                                    const NumericT *y1, unsigned int start1, unsigned int stride1,
                                    const NumericT *y2, unsigned int start2, unsigned int stride2,
                                    const NumericT *y3, unsigned int start3, unsigned int stride3,
                                    const NumericT *y4, unsigned int start4, unsigned int stride4,
                                    const NumericT *y5, unsigned int start5, unsigned int stride5,
                                    const NumericT *y6, unsigned int start6, unsigned int stride6,
                                    const NumericT *y7, unsigned int start7, unsigned int stride7,
                                    NumericT *group_results)
{
  __shared__ NumericT tmp_buffer[8*VIENNACL_MDOT_WORKGROUP_SIZE];
  unsigned int entries_per_thread = (sizex - 1) / (blockDim.x * gridDim.x) + 1;
  unsigned int vec_start_index = blockIdx.x * blockDim.x * entries_per_thread;
  unsigned int vec_stop_index  = min((blockIdx.x + 1) * blockDim.x * entries_per_thread, sizex); // don't go beyond vec size

  NumericT entry_x    = 0;
  NumericT group_sum0 = 0;
  NumericT group_sum1 = 0;
  NumericT group_sum2 = 0;
  NumericT group_sum3 = 0;
  NumericT group_sum4 = 0;
  NumericT group_sum5 = 0;
  NumericT group_sum6 = 0;
  NumericT group_sum7 = 0;
  for (unsigned int i = vec_start_index + threadIdx.x; i < vec_stop_index; i += blockDim.x) {
    entry_x     = x[i * stridex + startx];   // load only once from global memory!
    group_sum0 += entry_x * y0[i * stride0 + start0];
    group_sum1 += entry_x * y1[i * stride1 + start1];
    group_sum2 += entry_x * y2[i * stride2 + start2];
    group_sum3 += entry_x * y3[i * stride3 + start3];
    group_sum4 += entry_x * y4[i * stride4 + start4];
    group_sum5 += entry_x * y5[i * stride5 + start5];
    group_sum6 += entry_x * y6[i * stride6 + start6];
    group_sum7 += entry_x * y7[i * stride7 + start7];
  }
  tmp_buffer[threadIdx.x]                  = group_sum0;
  tmp_buffer[threadIdx.x +     blockDim.x] = group_sum1;
  tmp_buffer[threadIdx.x + 2 * blockDim.x] = group_sum2;
  tmp_buffer[threadIdx.x + 3 * blockDim.x] = group_sum3;
  tmp_buffer[threadIdx.x + 4 * blockDim.x] = group_sum4;
  tmp_buffer[threadIdx.x + 5 * blockDim.x] = group_sum5;
  tmp_buffer[threadIdx.x + 6 * blockDim.x] = group_sum6;
  tmp_buffer[threadIdx.x + 7 * blockDim.x] = group_sum7;

  // parallel reduction
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2) {
    __syncthreads();
    if (threadIdx.x < stride) {
      tmp_buffer[threadIdx.x                 ] += tmp_buffer[threadIdx.x+stride                 ];
      tmp_buffer[threadIdx.x +     blockDim.x] += tmp_buffer[threadIdx.x+stride +     blockDim.x];
      tmp_buffer[threadIdx.x + 2 * blockDim.x] += tmp_buffer[threadIdx.x+stride + 2 * blockDim.x];
      tmp_buffer[threadIdx.x + 3 * blockDim.x] += tmp_buffer[threadIdx.x+stride + 3 * blockDim.x];
      tmp_buffer[threadIdx.x + 4 * blockDim.x] += tmp_buffer[threadIdx.x+stride + 4 * blockDim.x];
      tmp_buffer[threadIdx.x + 5 * blockDim.x] += tmp_buffer[threadIdx.x+stride + 5 * blockDim.x];
      tmp_buffer[threadIdx.x + 6 * blockDim.x] += tmp_buffer[threadIdx.x+stride + 6 * blockDim.x];
      tmp_buffer[threadIdx.x + 7 * blockDim.x] += tmp_buffer[threadIdx.x+stride + 7 * blockDim.x];
    }
  }

  // write result of group to group_results
  if (threadIdx.x == 0) {
    group_results[blockIdx.x                ] = tmp_buffer[0];
    group_results[blockIdx.x +     gridDim.x] = tmp_buffer[    blockDim.x];
    group_results[blockIdx.x + 2 * gridDim.x] = tmp_buffer[2 * blockDim.x];
    group_results[blockIdx.x + 3 * gridDim.x] = tmp_buffer[3 * blockDim.x];
    group_results[blockIdx.x + 4 * gridDim.x] = tmp_buffer[4 * blockDim.x];
    group_results[blockIdx.x + 5 * gridDim.x] = tmp_buffer[5 * blockDim.x];
    group_results[blockIdx.x + 6 * gridDim.x] = tmp_buffer[6 * blockDim.x];
    group_results[blockIdx.x + 7 * gridDim.x] = tmp_buffer[7 * blockDim.x];
  }
}

// sums the array 'vec1' and writes to result. Makes use of a single work-group only.
template<typename NumericT>
__global__ void vector_multi_sum_kernel(
          NumericT const * vec1,
          NumericT * result,
          unsigned int start_result,
          unsigned int inc_result)
{
  __shared__ NumericT tmp_buffer[VIENNACL_MDOT_WORKGROUP_SIZE];

  tmp_buffer[threadIdx.x] = vec1[threadIdx.x + blockIdx.x * VIENNACL_MDOT_WORKGROUP_SIZE];

  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
      tmp_buffer[threadIdx.x] += tmp_buffer[threadIdx.x + stride];
  }

  if (threadIdx.x == 0)
    result[start_result + inc_result * blockIdx.x] = tmp_buffer[0];
}

template<typename NumericT>
void inner_prod_impl(vector_base<NumericT> const & x,
                     vector_tuple<NumericT> const & vec_tuple,
                     vector_base<NumericT> & result)
{
  typedef NumericT        value_type;

  static viennacl::vector<value_type> temp(8 * VIENNACL_MDOT_WORKGROUP_NUM);

  vcl_size_t current_index = 0;
  while (vec_tuple.const_size() > current_index)
  {
    switch (vec_tuple.const_size() - current_index)
    {
      case 7:
      case 6:
      case 5:
      case 4:
      {
        vector_base<NumericT> const & y0 = vec_tuple.const_at(current_index);
        vector_base<NumericT> const & y1 = vec_tuple.const_at(current_index + 1);
        vector_base<NumericT> const & y2 = vec_tuple.const_at(current_index + 2);
        vector_base<NumericT> const & y3 = vec_tuple.const_at(current_index + 3);

        inner_prod_4_kernel<<<VIENNACL_MDOT_WORKGROUP_NUM,
                              VIENNACL_MDOT_WORKGROUP_SIZE>>>( viennacl::cuda_arg(x),
                                                               static_cast<unsigned int>(viennacl::traits::start(x)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(x)),
                                                               static_cast<unsigned int>(viennacl::traits::size(x)),
                                                               viennacl::cuda_arg(y0),
                                                               static_cast<unsigned int>(viennacl::traits::start(y0)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y0)),
                                                               viennacl::cuda_arg(y1),
                                                               static_cast<unsigned int>(viennacl::traits::start(y1)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y1)),
                                                               viennacl::cuda_arg(y2),
                                                               static_cast<unsigned int>(viennacl::traits::start(y2)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y2)),
                                                               viennacl::cuda_arg(y3),
                                                               static_cast<unsigned int>(viennacl::traits::start(y3)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y3)),
                                                               viennacl::cuda_arg(temp)
                                                              );
        VIENNACL_CUDA_LAST_ERROR_CHECK("inner_prod_4_kernel");
        vector_multi_sum_kernel<<<4, VIENNACL_MDOT_WORKGROUP_NUM>>>(viennacl::cuda_arg(temp),
                                                                    viennacl::cuda_arg(result),
                                                                    static_cast<unsigned int>(viennacl::traits::start(result) + viennacl::traits::stride(result) * current_index),
                                                                    static_cast<unsigned int>(viennacl::traits::stride(result))
                                                                   );
        VIENNACL_CUDA_LAST_ERROR_CHECK("vector_multi_sum_kernel");
      }
        current_index += 4;
        break;
      case 3:
      {
        vector_base<NumericT> const & y0 = vec_tuple.const_at(current_index);
        vector_base<NumericT> const & y1 = vec_tuple.const_at(current_index + 1);
        vector_base<NumericT> const & y2 = vec_tuple.const_at(current_index + 2);

        inner_prod_3_kernel<<<VIENNACL_MDOT_WORKGROUP_NUM,
                              VIENNACL_MDOT_WORKGROUP_SIZE>>>( viennacl::cuda_arg(x),
                                                               static_cast<unsigned int>(viennacl::traits::start(x)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(x)),
                                                               static_cast<unsigned int>(viennacl::traits::size(x)),
                                                               viennacl::cuda_arg(y0),
                                                               static_cast<unsigned int>(viennacl::traits::start(y0)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y0)),
                                                               viennacl::cuda_arg(y1),
                                                               static_cast<unsigned int>(viennacl::traits::start(y1)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y1)),
                                                               viennacl::cuda_arg(y2),
                                                               static_cast<unsigned int>(viennacl::traits::start(y2)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y2)),
                                                               viennacl::cuda_arg(temp)
                                                              );
        VIENNACL_CUDA_LAST_ERROR_CHECK("inner_prod_3_kernel");
        vector_multi_sum_kernel<<<3, VIENNACL_MDOT_WORKGROUP_NUM>>>(viennacl::cuda_arg(temp),
                                                                    viennacl::cuda_arg(result),
                                                                    static_cast<unsigned int>(viennacl::traits::start(result) + viennacl::traits::stride(result) * current_index),
                                                                    static_cast<unsigned int>(viennacl::traits::stride(result))
                                                                   );
        VIENNACL_CUDA_LAST_ERROR_CHECK("vector_multi_sum_kernel");
      }
        current_index += 3;
        break;
      case 2:
      {
        vector_base<NumericT> const & y0 = vec_tuple.const_at(current_index);
        vector_base<NumericT> const & y1 = vec_tuple.const_at(current_index + 1);

        inner_prod_2_kernel<<<VIENNACL_MDOT_WORKGROUP_NUM,
                              VIENNACL_MDOT_WORKGROUP_SIZE>>>( viennacl::cuda_arg(x),
                                                               static_cast<unsigned int>(viennacl::traits::start(x)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(x)),
                                                               static_cast<unsigned int>(viennacl::traits::size(x)),
                                                               viennacl::cuda_arg(y0),
                                                               static_cast<unsigned int>(viennacl::traits::start(y0)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y0)),
                                                               viennacl::cuda_arg(y1),
                                                               static_cast<unsigned int>(viennacl::traits::start(y1)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y1)),
                                                               viennacl::cuda_arg(temp)
                                                              );
        VIENNACL_CUDA_LAST_ERROR_CHECK("inner_prod_2_kernel");
        vector_multi_sum_kernel<<<2, VIENNACL_MDOT_WORKGROUP_NUM>>>(viennacl::cuda_arg(temp),
                                                                    viennacl::cuda_arg(result),
                                                                    static_cast<unsigned int>(viennacl::traits::start(result) + viennacl::traits::stride(result) * current_index),
                                                                    static_cast<unsigned int>(viennacl::traits::stride(result))
                                                                   );
        VIENNACL_CUDA_LAST_ERROR_CHECK("vector_multi_sum_kernel");
      }
        current_index += 2;
        break;
      case 1:
      {
        vector_base<NumericT> const & y0 = vec_tuple.const_at(current_index);
        inner_prod_kernel<<<128, 128>>>(viennacl::cuda_arg(x),
                                        static_cast<unsigned int>(viennacl::traits::start(x)),
                                        static_cast<unsigned int>(viennacl::traits::stride(x)),
                                        static_cast<unsigned int>(viennacl::traits::size(x)),
                                        viennacl::cuda_arg(y0),
                                        static_cast<unsigned int>(viennacl::traits::start(y0)),
                                        static_cast<unsigned int>(viennacl::traits::stride(y0)),
                                        static_cast<unsigned int>(viennacl::traits::size(y0)),
                                        viennacl::cuda_arg(temp)
                                       );
        VIENNACL_CUDA_LAST_ERROR_CHECK("inner_prod_kernel");

        vector_multi_sum_kernel<<<1, 128>>>(viennacl::cuda_arg(temp),
                                            viennacl::cuda_arg(result),
                                            static_cast<unsigned int>(viennacl::traits::start(result) + viennacl::traits::stride(result) * current_index),
                                            static_cast<unsigned int>(viennacl::traits::stride(result))
                                           );
        VIENNACL_CUDA_LAST_ERROR_CHECK("vector_multi_sum_kernel");
      }
        current_index += 1;
        break;

      default:
      {
        vector_base<NumericT> const & y0 = vec_tuple.const_at(current_index);
        vector_base<NumericT> const & y1 = vec_tuple.const_at(current_index + 1);
        vector_base<NumericT> const & y2 = vec_tuple.const_at(current_index + 2);
        vector_base<NumericT> const & y3 = vec_tuple.const_at(current_index + 3);
        vector_base<NumericT> const & y4 = vec_tuple.const_at(current_index + 4);
        vector_base<NumericT> const & y5 = vec_tuple.const_at(current_index + 5);
        vector_base<NumericT> const & y6 = vec_tuple.const_at(current_index + 6);
        vector_base<NumericT> const & y7 = vec_tuple.const_at(current_index + 7);

        inner_prod_8_kernel<<<VIENNACL_MDOT_WORKGROUP_NUM,
                              VIENNACL_MDOT_WORKGROUP_SIZE>>>( viennacl::cuda_arg(x),
                                                               static_cast<unsigned int>(viennacl::traits::start(x)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(x)),
                                                               static_cast<unsigned int>(viennacl::traits::size(x)),
                                                               viennacl::cuda_arg(y0),
                                                               static_cast<unsigned int>(viennacl::traits::start(y0)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y0)),
                                                               viennacl::cuda_arg(y1),
                                                               static_cast<unsigned int>(viennacl::traits::start(y1)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y1)),
                                                               viennacl::cuda_arg(y2),
                                                               static_cast<unsigned int>(viennacl::traits::start(y2)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y2)),
                                                               viennacl::cuda_arg(y3),
                                                               static_cast<unsigned int>(viennacl::traits::start(y3)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y3)),
                                                               viennacl::cuda_arg(y4),
                                                               static_cast<unsigned int>(viennacl::traits::start(y4)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y4)),
                                                               viennacl::cuda_arg(y5),
                                                               static_cast<unsigned int>(viennacl::traits::start(y5)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y5)),
                                                               viennacl::cuda_arg(y6),
                                                               static_cast<unsigned int>(viennacl::traits::start(y6)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y6)),
                                                               viennacl::cuda_arg(y7),
                                                               static_cast<unsigned int>(viennacl::traits::start(y7)),
                                                               static_cast<unsigned int>(viennacl::traits::stride(y7)),
                                                               viennacl::cuda_arg(temp)
                                                              );
        VIENNACL_CUDA_LAST_ERROR_CHECK("inner_prod_8_kernel");
        vector_multi_sum_kernel<<<8, VIENNACL_MDOT_WORKGROUP_NUM>>>(viennacl::cuda_arg(temp),
                                                                    viennacl::cuda_arg(result),
                                                                    static_cast<unsigned int>(viennacl::traits::start(result) + viennacl::traits::stride(result) * current_index),
                                                                    static_cast<unsigned int>(viennacl::traits::stride(result))
                                                                   );
        VIENNACL_CUDA_LAST_ERROR_CHECK("vector_multi_sum_kernel");
      }
        current_index += 8;
        break;
    }
  }
}

#undef VIENNACL_MDOT_WORKGROUP_NUM
#undef VIENNACL_MDOT_WORKGROUP_SIZE

///////////////////////////////////

template<typename NumericT>
__global__ void norm_kernel_floats(
           const NumericT * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          unsigned int norm_selector,
          NumericT * group_buffer)
{
  __shared__ NumericT tmp_buffer[128];

  NumericT tmp = (norm_selector > 2) ? vec[start1] : 0;
  unsigned int work_per_thread = (size1 - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int group_start = blockIdx.x * work_per_thread * blockDim.x;
  unsigned int group_stop  = (blockIdx.x + 1) * work_per_thread * blockDim.x;
  group_stop = (group_stop > size1) ? size1 : group_stop;

  if (norm_selector == 1) //norm_1
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp += fabs(vec[i*inc1 + start1]);
  }
  else if (norm_selector == 2) //norm_2
  {
    NumericT vec_entry = 0;
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
    {
      vec_entry = vec[i*inc1 + start1];
      tmp += vec_entry * vec_entry;
    }
  }
  else if (norm_selector == 0) //norm_inf
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp = fmax(fabs(vec[i*inc1 + start1]), tmp);
  }
  else if (norm_selector == 3) //min
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp = (vec[i*inc1 + start1] < tmp) ? vec[i*inc1 + start1] : tmp;
  }
  else if (norm_selector == 4) //max
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp = (vec[i*inc1 + start1] > tmp) ? vec[i*inc1 + start1] : tmp;
  }

  tmp_buffer[threadIdx.x] = tmp;

  if (norm_selector == 1 || norm_selector == 2) //parallel reduction for norm_1 or norm_2:
  {
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] += tmp_buffer[threadIdx.x+stride];
    }
  }
  else if (norm_selector == 3)
  {
    //min:
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] = (tmp_buffer[threadIdx.x+stride] < tmp_buffer[threadIdx.x]) ? tmp_buffer[threadIdx.x+stride] : tmp_buffer[threadIdx.x];
    }
  }
  else if (norm_selector == 4)
  {
    //max:
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] = (tmp_buffer[threadIdx.x+stride] > tmp_buffer[threadIdx.x]) ? tmp_buffer[threadIdx.x+stride] : tmp_buffer[threadIdx.x];
    }
  }
  else
  {
    //norm_inf:
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] = fmax(tmp_buffer[threadIdx.x], tmp_buffer[threadIdx.x+stride]);
    }
  }

  if (threadIdx.x == 0)
    group_buffer[blockIdx.x] = tmp_buffer[0];
}

template<typename NumericT>
__global__ void norm_kernel_integers(
           const NumericT * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          unsigned int norm_selector,
          NumericT * group_buffer)
{
  __shared__ NumericT tmp_buffer[128];

  NumericT tmp = (norm_selector > 2) ? vec[start1] : 0;
  unsigned int work_per_thread = (size1 - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int group_start = blockIdx.x * work_per_thread * blockDim.x;
  unsigned int group_stop  = (blockIdx.x + 1) * work_per_thread * blockDim.x;
  group_stop = (group_stop > size1) ? size1 : group_stop;

  if (norm_selector == 1) //norm_1
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp += abs(vec[i*inc1 + start1]);
  }
  else if (norm_selector == 0) //norm_inf
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp = (tmp > abs(vec[i*inc1 + start1])) ? tmp : abs(vec[i*inc1 + start1]);
  }
  else if (norm_selector == 3) //min
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp = (vec[i*inc1 + start1] < tmp) ? vec[i*inc1 + start1] : tmp;
  }
  else if (norm_selector == 4) //max
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp = (vec[i*inc1 + start1] > tmp) ? vec[i*inc1 + start1] : tmp;
  }

  tmp_buffer[threadIdx.x] = tmp;

  if (norm_selector == 1 || norm_selector == 2) //parallel reduction for norm_1 or norm_2:
  {
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] += tmp_buffer[threadIdx.x+stride];
    }
  }
  else if (norm_selector == 3)
  {
    //min:
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] = (tmp_buffer[threadIdx.x+stride] < tmp_buffer[threadIdx.x]) ? tmp_buffer[threadIdx.x+stride] : tmp_buffer[threadIdx.x];
    }
  }
  else if (norm_selector == 4)
  {
    //max:
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] = (tmp_buffer[threadIdx.x+stride] > tmp_buffer[threadIdx.x]) ? tmp_buffer[threadIdx.x+stride] : tmp_buffer[threadIdx.x];
    }
  }
  else
  {
    //norm_inf:
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] = (tmp_buffer[threadIdx.x] > tmp_buffer[threadIdx.x+stride]) ? tmp_buffer[threadIdx.x] : tmp_buffer[threadIdx.x+stride];
    }
  }

  if (threadIdx.x == 0)
    group_buffer[blockIdx.x] = tmp_buffer[0];
}

template<typename NumericT>
__global__ void norm_kernel_unsigned_integers(
           const NumericT * vec,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          unsigned int norm_selector,
          NumericT * group_buffer)
{
  __shared__ NumericT tmp_buffer[128];

  NumericT tmp = (norm_selector > 2) ? vec[start1] : 0;
  unsigned int work_per_thread = (size1 - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int group_start = blockIdx.x * work_per_thread * blockDim.x;
  unsigned int group_stop  = (blockIdx.x + 1) * work_per_thread * blockDim.x;
  group_stop = (group_stop > size1) ? size1 : group_stop;

  if (norm_selector == 1) //norm_1
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp += vec[i*inc1 + start1];
  }
  else if (norm_selector == 0) //norm_inf
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp = (tmp > vec[i*inc1 + start1]) ? tmp : vec[i*inc1 + start1];
  }
  else if (norm_selector == 3) //min
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp = (vec[i*inc1 + start1] < tmp) ? vec[i*inc1 + start1] : tmp;
  }
  else if (norm_selector == 4) //max
  {
    for (unsigned int i = group_start + threadIdx.x; i < group_stop; i += blockDim.x)
      tmp = (vec[i*inc1 + start1] > tmp) ? vec[i*inc1 + start1] : tmp;
  }

  tmp_buffer[threadIdx.x] = tmp;

  if (norm_selector == 1 || norm_selector == 2) //parallel reduction for norm_1 or norm_2:
  {
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] += tmp_buffer[threadIdx.x+stride];
    }
  }
  else if (norm_selector == 3)
  {
    //min:
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] = (tmp_buffer[threadIdx.x+stride] < tmp_buffer[threadIdx.x]) ? tmp_buffer[threadIdx.x+stride] : tmp_buffer[threadIdx.x];
    }
  }
  else if (norm_selector == 4)
  {
    //max:
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] = (tmp_buffer[threadIdx.x+stride] > tmp_buffer[threadIdx.x]) ? tmp_buffer[threadIdx.x+stride] : tmp_buffer[threadIdx.x];
    }
  }
  else
  {
    //norm_inf:
    for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
    {
      __syncthreads();
      if (threadIdx.x < stride)
        tmp_buffer[threadIdx.x] = (tmp_buffer[threadIdx.x] > tmp_buffer[threadIdx.x+stride]) ? tmp_buffer[threadIdx.x] : tmp_buffer[threadIdx.x+stride];
    }
  }

  if (threadIdx.x == 0)
    group_buffer[blockIdx.x] = tmp_buffer[0];
}

/** \cond */
namespace detail
{
  struct norm_kernel_launcher_integers
  {
    template<typename NumericT>
    static void apply(vector_base<NumericT> const & vec1,
                      vector_base<NumericT> & temp,
                      unsigned int option)
    {
      norm_kernel_integers<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                         static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                         static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                         static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                         static_cast<unsigned int>(option),
                                         viennacl::cuda_arg(temp)
                                        );
      VIENNACL_CUDA_LAST_ERROR_CHECK("norm_kernel");
    }
  };

  struct norm_kernel_launcher_unsigned_integers
  {
    template<typename NumericT>
    static void apply(vector_base<NumericT> const & vec1,
                      vector_base<NumericT> & temp,
                      unsigned int option)
    {
      norm_kernel_unsigned_integers<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                                 static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                                 static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                                 static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                                 static_cast<unsigned int>(option),
                                                 viennacl::cuda_arg(temp)
                                                );
      VIENNACL_CUDA_LAST_ERROR_CHECK("norm_kernel");
    }
  };


  struct norm_kernel_launcher_floats
  {
    template<typename NumericT>
    static void apply(vector_base<NumericT> const & vec1,
                      vector_base<NumericT> & temp,
                      unsigned int option)
    {
      norm_kernel_floats<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                       static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                       static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                       static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                       static_cast<unsigned int>(option),
                                       viennacl::cuda_arg(temp)
                                      );
      VIENNACL_CUDA_LAST_ERROR_CHECK("norm_kernel");
    }
  };

  template<typename NumericT>
  struct norm_kernel_launcher : public norm_kernel_launcher_integers {};

  template<>
  struct norm_kernel_launcher<unsigned char>  : public norm_kernel_launcher_unsigned_integers {};

  template<>
  struct norm_kernel_launcher<unsigned short>  : public norm_kernel_launcher_unsigned_integers {};

  template<>
  struct norm_kernel_launcher<unsigned int>  : public norm_kernel_launcher_unsigned_integers {};

  template<>
  struct norm_kernel_launcher<unsigned long>  : public norm_kernel_launcher_unsigned_integers {};

  template<>
  struct norm_kernel_launcher<float>  : public norm_kernel_launcher_floats {};

  template<>
  struct norm_kernel_launcher<double> : public norm_kernel_launcher_floats {};

}
/** \endcond */


/** @brief Computes the l^1-norm of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_1_impl(vector_base<NumericT> const & vec1,
                 scalar<NumericT> & result)
{
  typedef NumericT        value_type;

  vcl_size_t work_groups = 128;
  viennacl::vector<value_type> temp(work_groups);

  detail::norm_kernel_launcher<NumericT>::apply(vec1, temp, 1);
  detail::vector_sum_kernel_launcher<NumericT>::apply(temp, 1, result);
}

/** @brief Computes the l^1-norm of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_1_cpu(vector_base<NumericT> const & vec1,
                NumericT & result)
{
  typedef NumericT        value_type;

  vcl_size_t work_groups = 128;
  viennacl::vector<value_type> temp(work_groups);

  detail::norm_kernel_launcher<NumericT>::apply(vec1, temp, 1);

  // Now copy partial results from GPU back to CPU and run reduction there:
  std::vector<value_type> temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = 0;
  for (typename std::vector<value_type>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result += *it;
}

///// norm_2

/** @brief Computes the l^2-norm of a vector - implementation
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_2_impl(vector_base<NumericT> const & vec1,
                 scalar<NumericT> & result)
{
  typedef NumericT       value_type;

  vcl_size_t work_groups = 128;
  viennacl::vector<value_type> temp(work_groups);

  detail::norm_kernel_launcher<NumericT>::apply(vec1, temp, 2);

  detail::vector_sum_kernel_launcher<NumericT>::apply(temp, 2, result);
}

/** @brief Computes the l^2-norm of a vector - implementation
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_2_cpu(vector_base<NumericT> const & vec1,
                NumericT & result)
{
  typedef NumericT        value_type;

  vcl_size_t work_groups = 128;
  viennacl::vector<value_type> temp(work_groups);

  detail::norm_kernel_launcher<NumericT>::apply(vec1, temp, 2);

  std::vector<value_type> temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = 0;
  for (typename std::vector<value_type>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result += *it;
  result = std::sqrt(result);
}


////// norm_inf

/** @brief Computes the supremum-norm of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_inf_impl(vector_base<NumericT> const & vec1,
                   scalar<NumericT> & result)
{
  typedef NumericT      value_type;

  vcl_size_t work_groups = 128;
  viennacl::vector<value_type> temp(work_groups);

  detail::norm_kernel_launcher<NumericT>::apply(vec1, temp, 0);
  detail::vector_sum_kernel_launcher<NumericT>::apply(temp, 0, result);
}



/** @brief Computes the supremum-norm of a vector
*
* @param vec1 The vector
* @param result The result scalar
*/
template<typename NumericT>
void norm_inf_cpu(vector_base<NumericT> const & vec1,
                  NumericT & result)
{
  typedef NumericT        value_type;

  vcl_size_t work_groups = 128;
  viennacl::vector<value_type> temp(work_groups);

  detail::norm_kernel_launcher<NumericT>::apply(vec1, temp, 0);

  std::vector<value_type> temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = 0;
  for (typename std::vector<value_type>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result = std::max(result, *it);
}


////// max

// second reduction stage for min() and max()
template<typename NumericT>
__global__ void vector_maxmin_kernel(
          const NumericT * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          unsigned int option, //0: use max, 1: use min
          NumericT * result)
{
  __shared__ NumericT tmp_buffer[128];
  NumericT thread_minmax = vec1[start1];
  for (unsigned int i = threadIdx.x; i<size1; i += blockDim.x)
  {
    if (option > 0) //min
      thread_minmax = (vec1[i*inc1+start1] < thread_minmax) ? vec1[i*inc1+start1] : thread_minmax;
    else
      thread_minmax = (vec1[i*inc1+start1] > thread_minmax) ? vec1[i*inc1+start1] : thread_minmax;
  }

  tmp_buffer[threadIdx.x] = thread_minmax;

  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
    {
      if (option > 0) //min
        tmp_buffer[threadIdx.x] = (tmp_buffer[threadIdx.x + stride] < tmp_buffer[threadIdx.x]) ? tmp_buffer[threadIdx.x + stride] : tmp_buffer[threadIdx.x];
      else
        tmp_buffer[threadIdx.x] = (tmp_buffer[threadIdx.x + stride] > tmp_buffer[threadIdx.x]) ? tmp_buffer[threadIdx.x + stride] : tmp_buffer[threadIdx.x];
    }
  }

  if (threadIdx.x == 0)
    *result = tmp_buffer[0];
}


/** @brief Computes the maximum of a vector, both reduction stages run on the GPU
*
* @param vec1   The vector
* @param result The result GPU scalar
*/
template<typename NumericT>
void max_impl(vector_base<NumericT> const & vec1,
              scalar<NumericT> & result)
{
  typedef NumericT      value_type;

  vcl_size_t work_groups = 128;
  viennacl::vector<value_type> temp(work_groups, viennacl::traits::context(vec1));

  detail::norm_kernel_launcher<NumericT>::apply(vec1, temp, 4);

  vector_maxmin_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                   static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                   static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                   static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                   static_cast<unsigned int>(0),
                                   viennacl::cuda_arg(result)
                                  );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vector_maxmin_kernel");
}



/** @brief Computes the maximum of a vector, first reduction stage on the GPU, second stage on the CPU
*
* @param vec1   The vector
* @param result The result host scalar
*/
template<typename NumericT>
void max_cpu(vector_base<NumericT> const & vec1,
             NumericT & result)
{
  typedef NumericT        value_type;

  vcl_size_t work_groups = 128;
  viennacl::vector<value_type> temp(work_groups, viennacl::traits::context(vec1));

  detail::norm_kernel_launcher<NumericT>::apply(vec1, temp, 4);

  std::vector<value_type> temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = temp[0];
  for (typename std::vector<value_type>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result = std::max(result, *it);
}

//////////////////

/** @brief Computes the maximum of a vector, both reduction stages run on the GPU
*
* @param vec1   The vector
* @param result The result GPU scalar
*/
template<typename NumericT>
void min_impl(vector_base<NumericT> const & vec1,
              scalar<NumericT> & result)
{
  typedef NumericT      value_type;

  vcl_size_t work_groups = 128;
  viennacl::vector<value_type> temp(work_groups, viennacl::traits::context(vec1));

  detail::norm_kernel_launcher<NumericT>::apply(vec1, temp, 3);

  vector_maxmin_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                   static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                   static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                   static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                   static_cast<unsigned int>(1),
                                   viennacl::cuda_arg(result)
                                  );
  VIENNACL_CUDA_LAST_ERROR_CHECK("vector_maxmin_kernel");
}



/** @brief Computes the maximum of a vector, first reduction stage on the GPU, second stage on the CPU
*
* @param vec1   The vector
* @param result The result host scalar
*/
template<typename NumericT>
void min_cpu(vector_base<NumericT> const & vec1,
             NumericT & result)
{
  typedef NumericT        value_type;

  vcl_size_t work_groups = 128;
  viennacl::vector<value_type> temp(work_groups, viennacl::traits::context(vec1));

  detail::norm_kernel_launcher<NumericT>::apply(vec1, temp, 3);

  std::vector<value_type> temp_cpu(work_groups);
  viennacl::fast_copy(temp.begin(), temp.end(), temp_cpu.begin());

  result = temp[0];
  for (typename std::vector<value_type>::const_iterator it = temp_cpu.begin(); it != temp_cpu.end(); ++it)
    result = std::min(result, *it);
}


//////////////////

/** @brief Computes the maximum of a vector, both reduction stages run on the GPU
*
* @param vec1   The vector
* @param result The result GPU scalar
*/
template<typename NumericT>
void sum_impl(vector_base<NumericT> const & vec1,
              scalar<NumericT> & result)
{
  typedef NumericT      value_type;

  viennacl::vector<NumericT> all_ones = viennacl::scalar_vector<NumericT>(vec1.size(), NumericT(1), viennacl::traits::context(vec1));
  viennacl::linalg::cuda::inner_prod_impl(vec1, all_ones, result);
}



/** @brief Computes the maximum of a vector, first reduction stage on the GPU, second stage on the CPU
*
* @param vec1   The vector
* @param result The result host scalar
*/
template<typename NumericT>
void sum_cpu(vector_base<NumericT> const & vec1,
             NumericT & result)
{
  typedef NumericT        value_type;

  viennacl::vector<NumericT> all_ones = viennacl::scalar_vector<NumericT>(vec1.size(), NumericT(1), viennacl::traits::context(vec1));
  viennacl::linalg::cuda::inner_prod_cpu(vec1, all_ones, result);
}



//////////////////////////////////////



//index_norm_inf:

// fixes the problem of not having (f)abs available in a consistent manner
template<typename NumericT>
__device__ NumericT              cuda_abs(NumericT val) { return (val < 0) ? -val : val; }
__device__ inline unsigned long  cuda_abs(unsigned long  val) { return val; }
__device__ inline unsigned int   cuda_abs(unsigned int   val) { return val; }
__device__ inline unsigned short cuda_abs(unsigned short val) { return val; }
__device__ inline unsigned char  cuda_abs(unsigned char  val) { return val; }

template<typename NumericT>
__global__ void index_norm_inf_kernel(const NumericT * vec,
                                      unsigned int start1,
                                      unsigned int inc1,
                                      unsigned int size1,
                                      unsigned int * result)
{
  __shared__ NumericT float_buffer[128];
  __shared__ unsigned int index_buffer[128];

  float_buffer[threadIdx.x] = 0;
  index_buffer[threadIdx.x] = 0;

  //step 1: fill buffer:
  NumericT cur_max = NumericT(0);
  NumericT tmp;
  for (unsigned int i = threadIdx.x; i < size1; i += blockDim.x)
  {
    tmp = vec[i*inc1+start1];
    tmp = cuda_abs(tmp);
    if (cur_max < tmp)
    {
      float_buffer[threadIdx.x] = tmp;
      index_buffer[threadIdx.x] = i;
      cur_max = tmp;
    }
  }

  //step 2: parallel reduction:
  for (unsigned int stride = blockDim.x/2; stride > 0; stride /= 2)
  {
    __syncthreads();
    if (threadIdx.x < stride)
    {
      //find the first occurring index
      if (float_buffer[threadIdx.x] < float_buffer[threadIdx.x+stride])
      {
        index_buffer[threadIdx.x] = index_buffer[threadIdx.x+stride];
        float_buffer[threadIdx.x] = float_buffer[threadIdx.x+stride];
      }
    }
  }

  if (threadIdx.x == 0)
    *result = index_buffer[0];
}

//This function should return a CPU scalar, otherwise statements like
// vcl_rhs[index_norm_inf(vcl_rhs)]
// are ambiguous
/** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
*
* @param vec1 The vector
* @return The result. Note that the result must be a CPU scalar (unsigned int), since gpu scalars are floating point types.
*/
template<typename NumericT>
vcl_size_t index_norm_inf(vector_base<NumericT> const & vec1)
{
  typedef NumericT       value_type;

  viennacl::backend::mem_handle h;
  viennacl::backend::memory_create(h, sizeof(unsigned int), viennacl::traits::context(vec1));

  index_norm_inf_kernel<<<1, 128>>>(viennacl::cuda_arg(vec1),
                                    static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                    static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                    static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                    viennacl::cuda_arg<unsigned int>(h)
                                    //reinterpret_cast<unsigned int *>(h.cuda_handle().get())
                                  );
  VIENNACL_CUDA_LAST_ERROR_CHECK("index_norm_inf_kernel");

  unsigned int ret = 0;
  viennacl::backend::memory_read(h, 0, sizeof(unsigned int), &ret);
  return static_cast<vcl_size_t>(ret);
}

///////////////////////////////////////////

template<typename NumericT>
__global__ void plane_rotation_kernel(
          NumericT * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,
          NumericT * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2,
          NumericT alpha,
          NumericT beta)
{
  NumericT tmp1 = 0;
  NumericT tmp2 = 0;

  for (unsigned int i = blockDim.x * blockIdx.x + threadIdx.x; i < size1; i += blockDim.x * gridDim.x)
  {
    tmp1 = vec1[i*inc1+start1];
    tmp2 = vec2[i*inc2+start2];

    vec1[i*inc1+start1] = alpha * tmp1 + beta * tmp2;
    vec2[i*inc2+start2] = alpha * tmp2 - beta * tmp1;
  }

}

/** @brief Computes a plane rotation of two vectors.
*
* Computes (x,y) <- (alpha * x + beta * y, -beta * x + alpha * y)
*
* @param vec1   The first vector
* @param vec2   The second vector
* @param alpha  The first transformation coefficient
* @param beta   The second transformation coefficient
*/
template<typename NumericT>
void plane_rotation(vector_base<NumericT> & vec1,
                    vector_base<NumericT> & vec2,
                    NumericT alpha, NumericT beta)
{
  typedef NumericT     value_type;

  value_type temporary_alpha = 0;
  if (viennacl::is_cpu_scalar<value_type>::value)
    temporary_alpha = alpha;

  value_type temporary_beta = 0;
  if (viennacl::is_cpu_scalar<value_type>::value)
    temporary_beta = beta;

  plane_rotation_kernel<<<128, 128>>>(viennacl::cuda_arg(vec1),
                                      static_cast<unsigned int>(viennacl::traits::start(vec1)),
                                      static_cast<unsigned int>(viennacl::traits::stride(vec1)),
                                      static_cast<unsigned int>(viennacl::traits::size(vec1)),
                                      viennacl::cuda_arg(vec2),
                                      static_cast<unsigned int>(viennacl::traits::start(vec2)),
                                      static_cast<unsigned int>(viennacl::traits::stride(vec2)),
                                      static_cast<unsigned int>(viennacl::traits::size(vec2)),
                                      viennacl::cuda_arg<value_type>(detail::arg_reference(alpha, temporary_alpha)),
                                      viennacl::cuda_arg<value_type>(detail::arg_reference(beta, temporary_beta)) );
  VIENNACL_CUDA_LAST_ERROR_CHECK("plane_rotation_kernel");
}

////////////////////////


template<typename NumericT>
__global__ void scan_kernel_1(NumericT const *X,
                              unsigned int startX,
                              unsigned int incX,
                              unsigned int sizeX,

                              NumericT *Y,
                              unsigned int startY,
                              unsigned int incY,

                              unsigned int scan_offset,
                              NumericT *carries) // 0 for inclusive scan, 1 for exclusive
{
  __shared__ NumericT shared_buffer[256];
  NumericT my_value;

  unsigned int work_per_thread = (sizeX - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);
  unsigned int block_offset = 0;

  // run scan on each section
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
  {
    // load data:
    my_value = (i < sizeX) ? X[i * incX + startX] : 0;

    // inclusive scan in shared buffer:
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
    {
      __syncthreads();
      shared_buffer[threadIdx.x] = my_value;
      __syncthreads();
      if (threadIdx.x >= stride)
        my_value += shared_buffer[threadIdx.x - stride];
    }
    __syncthreads();
    shared_buffer[threadIdx.x] = my_value;
    __syncthreads();

    // exclusive scan requires us to write a zero value at the beginning of each block
    if (scan_offset > 0)
      my_value = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;

    // write to output array
    if (i < sizeX)
      Y[i * incY + startY] = block_offset + my_value;

    block_offset += shared_buffer[blockDim.x-1];
  }

  // write carry:
  if (threadIdx.x == 0)
    carries[blockIdx.x] = block_offset;

}

// exclusive-scan of carries
template<typename NumericT>
__global__ void scan_kernel_2(NumericT *carries)
{
  __shared__ NumericT shared_buffer[256];

  // load data:
  NumericT my_carry = carries[threadIdx.x];

  // exclusive scan in shared buffer:

  for(unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    shared_buffer[threadIdx.x] = my_carry;
    __syncthreads();
    if (threadIdx.x >= stride)
      my_carry += shared_buffer[threadIdx.x - stride];
  }
  __syncthreads();
  shared_buffer[threadIdx.x] = my_carry;
  __syncthreads();

  // write to output array
  carries[threadIdx.x] = (threadIdx.x > 0) ? shared_buffer[threadIdx.x - 1] : 0;
}

template<typename NumericT>
__global__ void scan_kernel_3(NumericT *Y,
                              unsigned int startY,
                              unsigned int incY,
                              unsigned int sizeY,

                              NumericT const *carries)
{
  unsigned int work_per_thread = (sizeY - 1) / (gridDim.x * blockDim.x) + 1;
  unsigned int block_start = work_per_thread * blockDim.x *  blockIdx.x;
  unsigned int block_stop  = work_per_thread * blockDim.x * (blockIdx.x + 1);

  __shared__ NumericT shared_offset;

  if (threadIdx.x == 0)
    shared_offset = carries[blockIdx.x];

  __syncthreads();

  // add offset to each element in the block:
  for (unsigned int i = block_start + threadIdx.x; i < block_stop; i += blockDim.x)
    if (i < sizeY)
      Y[i * incY + startY] += shared_offset;
}



namespace detail
{
  /** @brief Worker routine for scan routines
   *
   * Note on performance: For non-in-place scans one could optimize away the temporary 'cuda_carries'-array.
   * This, however, only provides small savings in the latency-dominated regime, yet would effectively double the amount of code to maintain.
   */
  template<typename NumericT>
  void scan_impl(vector_base<NumericT> const & input,
                 vector_base<NumericT>       & output,
                 bool is_inclusive)
  {
    vcl_size_t block_num = 128;
    vcl_size_t threads_per_block = 128;

    viennacl::backend::mem_handle cuda_carries;
    viennacl::backend::memory_create(cuda_carries, sizeof(NumericT)*block_num, viennacl::traits::context(input));

    // First step: Scan within each thread group and write carries
    scan_kernel_1<<<block_num, threads_per_block>>>(viennacl::cuda_arg(input),
                                                    static_cast<unsigned int>(viennacl::traits::start(input)),
                                                    static_cast<unsigned int>(viennacl::traits::stride(input)),
                                                    static_cast<unsigned int>(viennacl::traits::size(input)),

                                                    viennacl::cuda_arg(output),
                                                    static_cast<unsigned int>(viennacl::traits::start(output)),
                                                    static_cast<unsigned int>(viennacl::traits::stride(output)),

                                                    static_cast<unsigned int>(is_inclusive ? 0 : 1),
                                                    viennacl::cuda_arg<NumericT>(cuda_carries)
                                                   );

    // Second step: Compute offset for each thread group (exclusive scan for each thread group)
    scan_kernel_2<<<1, block_num>>>(viennacl::cuda_arg<NumericT>(cuda_carries));

    // Third step: Offset each thread group accordingly
    scan_kernel_3<<<block_num, threads_per_block>>>(viennacl::cuda_arg(output),
                                                    static_cast<unsigned int>(viennacl::traits::start(output)),
                                                    static_cast<unsigned int>(viennacl::traits::stride(output)),
                                                    static_cast<unsigned int>(viennacl::traits::size(output)),

                                                    viennacl::cuda_arg<NumericT>(cuda_carries)
                                                   );
  }
}


/** @brief This function implements an inclusive scan using CUDA.
*
* @param input       Input vector.
* @param output      The output vector. Either idential to input or non-overlapping.
*/
template<typename NumericT>
void inclusive_scan(vector_base<NumericT> const & input,
                    vector_base<NumericT>       & output)
{
  detail::scan_impl(input, output, true);
}


/** @brief This function implements an exclusive scan using CUDA.
*
* @param input       Input vector
* @param output      The output vector. Either idential to input or non-overlapping.
*/
template<typename NumericT>
void exclusive_scan(vector_base<NumericT> const & input,
                    vector_base<NumericT>       & output)
{
  detail::scan_impl(input, output, false);
}



} //namespace cuda
} //namespace linalg
} //namespace viennacl


#endif
