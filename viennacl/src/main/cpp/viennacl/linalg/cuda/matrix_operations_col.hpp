#ifndef VIENNACL_LINALG_CUDA_MATRIX_OPERATIONS_COL_HPP_
#define VIENNACL_LINALG_CUDA_MATRIX_OPERATIONS_COL_HPP_

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

/** @file  viennacl/linalg/cuda/matrix_operations_col.hpp
    @brief Implementations of column-major dense matrix related operations, including matrix-vector products, using CUDA.
*/


namespace viennacl
{
namespace linalg
{
namespace cuda
{

template<typename DestNumericT, typename SrcNumericT>
__global__ void convert_col_kernel(DestNumericT * A,
                                  unsigned int A_start1, unsigned int A_start2,
                                  unsigned int A_inc1,   unsigned int A_inc2,
                                  unsigned int A_size1,  unsigned int A_size2,
                                  unsigned int A_internal_size1,  unsigned int A_internal_size2,

                                  const SrcNumericT * B,
                                  unsigned int B_start1, unsigned int B_start2,
                                  unsigned int B_inc1,   unsigned int B_inc2,
                                  unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1];
}

//
// am
//

// alpha on CPU
template<typename NumericT>
__global__ void am_col_kernel(NumericT * A,
                              unsigned int A_start1, unsigned int A_start2,
                              unsigned int A_inc1,   unsigned int A_inc2,
                              unsigned int A_size1,  unsigned int A_size2,
                              unsigned int A_internal_size1,  unsigned int A_internal_size2,

                              NumericT fac2,
                              unsigned int options2,
                              const NumericT * B,
                              unsigned int B_start1, unsigned int B_start2,
                              unsigned int B_inc1,   unsigned int B_inc2,
                              unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  if (options2 & (1 << 1))
  {
    for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
      for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
        A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha;
  }
  else
  {
    for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
      for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
        A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha;
  }
}

// alpha on GPU
template<typename NumericT>
__global__ void am_col_kernel(NumericT * A,
                              unsigned int A_start1, unsigned int A_start2,
                              unsigned int A_inc1,   unsigned int A_inc2,
                              unsigned int A_size1,  unsigned int A_size2,
                              unsigned int A_internal_size1,  unsigned int A_internal_size2,

                              const NumericT * fac2,
                              unsigned int options2,
                              const NumericT * B,
                              unsigned int B_start1, unsigned int B_start2,
                              unsigned int B_inc1,   unsigned int B_inc2,
                              unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  if (options2 & (1 << 1))
  {
    for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
      for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
        A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha;
  }
  else
  {
    for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
      for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
        A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha;
  }
}


//
// ambm
//

// alpha and beta on CPU
template<typename NumericT>
__global__ void ambm_col_kernel(NumericT * A,
                                unsigned int A_start1, unsigned int A_start2,
                                unsigned int A_inc1,   unsigned int A_inc2,
                                unsigned int A_size1,  unsigned int A_size2,
                                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                                NumericT fac2,
                                unsigned int options2,
                                const NumericT * B,
                                unsigned int B_start1, unsigned int B_start2,
                                unsigned int B_inc1,   unsigned int B_inc2,
                                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                                NumericT fac3,
                                unsigned int options3,
                                const NumericT * C,
                                unsigned int C_start1, unsigned int C_start2,
                                unsigned int C_inc1,   unsigned int C_inc2,
                                unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
}


// alpha on CPU, beta on GPU
template<typename NumericT>
__global__ void ambm_col_kernel(NumericT * A,
                                unsigned int A_start1, unsigned int A_start2,
                                unsigned int A_inc1,   unsigned int A_inc2,
                                unsigned int A_size1,  unsigned int A_size2,
                                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                                NumericT fac2,
                                unsigned int options2,
                                const NumericT * B,
                                unsigned int B_start1, unsigned int B_start2,
                                unsigned int B_inc1,   unsigned int B_inc2,
                                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                                const NumericT * fac3,
                                unsigned int options3,
                                const NumericT * C,
                                unsigned int C_start1, unsigned int C_start2,
                                unsigned int C_inc1,   unsigned int C_inc2,
                                unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
}

// alpha on GPU, beta on CPU
template<typename NumericT>
__global__ void ambm_col_kernel(NumericT * A,
                                unsigned int A_start1, unsigned int A_start2,
                                unsigned int A_inc1,   unsigned int A_inc2,
                                unsigned int A_size1,  unsigned int A_size2,
                                unsigned int A_internal_size1,  unsigned int A_internal_size2,

                                const NumericT * fac2,
                                unsigned int options2,
                                const NumericT * B,
                                unsigned int B_start1, unsigned int B_start2,
                                unsigned int B_inc1,   unsigned int B_inc2,
                                unsigned int B_internal_size1,  unsigned int B_internal_size2,

                                NumericT fac3,
                                unsigned int options3,
                                const NumericT * C,
                                unsigned int C_start1, unsigned int C_start2,
                                unsigned int C_inc1,   unsigned int C_inc2,
                                unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
}


// alpha and beta on GPU
template<typename NumericT>
__global__ void ambm_col_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          const NumericT * fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
}


//
// ambm_m
//

// alpha and beta on CPU
template<typename NumericT>
__global__ void ambm_m_col_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          NumericT fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          NumericT fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
       += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
       += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
       += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
       += B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
}


// alpha on CPU, beta on GPU
template<typename NumericT>
__global__ void ambm_m_col_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          NumericT fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          const NumericT * fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
}

// alpha on GPU, beta on CPU
template<typename NumericT>
__global__ void ambm_m_col_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          NumericT fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
}


// alpha and beta on GPU
template<typename NumericT>
__global__ void ambm_m_col_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * fac2,
          unsigned int options2,
          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          const NumericT * fac3,
          unsigned int options3,
          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2)
{
  NumericT alpha = *fac2;
  if (options2 & (1 << 0))
    alpha = -alpha;

  NumericT beta = *fac3;
  if (options3 & (1 << 0))
    beta = -beta;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  if (options2 & (1 << 1))
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] / alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
  else
  {
    if (options3 & (1 << 1))
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] / beta;
    }
    else
    {
      for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
        for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
          A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
        = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1] * alpha
        + C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1] * beta;
    }
  }
}



//
// assignments
//

template<typename NumericT>
__global__ void matrix_col_assign_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,
          NumericT alpha)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = alpha;
}


template<typename NumericT>
__global__ void matrix_col_diagonal_assign_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,
          NumericT alpha)
{
  unsigned int gid = (blockIdx.x * blockDim.x + threadIdx.x);

  for (unsigned int row = gid; row < A_size1; row += blockDim.x * gridDim.x)
    A[(row * A_inc1 + A_start1) + (row * A_inc2 + A_start2) * A_internal_size1] = alpha;
}

//
// binary element-wise operations
//

template<typename NumericT>
__global__ void element_op_col_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2,

          unsigned int op_type) //0: product, 1: division, 2: pow
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  if (op_type == 2)
  {
    for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
      for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
        A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
      = pow(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1],
            C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1]);
  }
  else if (op_type == 1)
  {
    for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
      for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
        A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
      = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]
      / C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1];
  }
  else if (op_type == 0)
  {
    for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
      for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
        A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
      = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]
      * C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1];
  }
}

template<typename NumericT>
__global__ void element_op_int_col_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2,

          const NumericT * C,
          unsigned int C_start1, unsigned int C_start2,
          unsigned int C_inc1,   unsigned int C_inc2,
          unsigned int C_internal_size1,  unsigned int C_internal_size2,

          unsigned int op_type) //0: product, 1: division, 2: pow
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;

  if (op_type == 1)
  {
    for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
      for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
        A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
      = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]
      / C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1];
  }
  else if (op_type == 0)
  {
    for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
      for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
        A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1]
      = B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]
      * C[(row * C_inc1 + C_start1) + (col * C_inc2 + C_start2) * C_internal_size1];
  }
}


//
// unary element-wise operations
//

// abs
template<typename NumericT>
__global__ void matrix_col_element_abs_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = abs(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// acos
template<typename NumericT>
__global__ void matrix_col_element_acos_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = acos(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// asin
template<typename NumericT>
__global__ void matrix_col_element_asin_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = asin(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// atan
template<typename NumericT>
__global__ void matrix_col_element_atan_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = atan(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// ceil
template<typename NumericT>
__global__ void matrix_col_element_ceil_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = ceil(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// cos
template<typename NumericT>
__global__ void matrix_col_element_cos_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = cos(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// cosh
template<typename NumericT>
__global__ void matrix_col_element_cosh_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = cosh(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// exp
template<typename NumericT>
__global__ void matrix_col_element_exp_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = exp(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// fabs
template<typename NumericT>
__global__ void matrix_col_element_fabs_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = fabs(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// floor
template<typename NumericT>
__global__ void matrix_col_element_floor_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = floor(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// log
template<typename NumericT>
__global__ void matrix_col_element_log_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = log(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// log10
template<typename NumericT>
__global__ void matrix_col_element_log10_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = log10(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// sin
template<typename NumericT>
__global__ void matrix_col_element_sin_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = sin(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// sinh
template<typename NumericT>
__global__ void matrix_col_element_sinh_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = sinh(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// sqrt
template<typename NumericT>
__global__ void matrix_col_element_sqrt_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = sqrt(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// tan
template<typename NumericT>
__global__ void matrix_col_element_tan_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = tan(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}


// tanh
template<typename NumericT>
__global__ void matrix_col_element_tanh_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * B,
          unsigned int B_start1, unsigned int B_start2,
          unsigned int B_inc1,   unsigned int B_inc2,
          unsigned int B_internal_size1,  unsigned int B_internal_size2)
{
  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int col = col_gid; col < A_size2; col += gridDim.x)
    for (unsigned int row = row_gid; row < A_size1; row += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] = tanh(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]);
}



//
// matrix-vector product
//

template<typename NumericT>
__global__ void vec_mul_col_kernel(
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * v,
          unsigned int v_start,
          unsigned int v_inc,
          unsigned int v_size,
          NumericT * result,
          unsigned int result_start,
          unsigned int result_inc,
          unsigned int result_size)
{

  for (unsigned int row = blockIdx.x * blockDim.x + threadIdx.x; row < A_row_size; row += gridDim.x * blockDim.x)
  {
    NumericT dot_prod = 0;
    for (unsigned int col = 0; col < A_col_size; ++col)
      dot_prod += A[(row * A_row_inc + A_row_start) + (col * A_col_inc + A_col_start) * A_internal_rows] * v[v_start + v_inc * col];
    result[row * result_inc + result_start] = dot_prod;
  }
}


template<typename NumericT>
__global__ void trans_vec_mul_col_kernel(
          const NumericT * A,
          unsigned int A_row_start,
          unsigned int A_col_start,
          unsigned int A_row_inc,
          unsigned int A_col_inc,
          unsigned int A_row_size,
          unsigned int A_col_size,
          unsigned int A_internal_rows,
          unsigned int A_internal_cols,
          const NumericT * v,
          unsigned int v_start,
          unsigned int v_inc,
          unsigned int v_size,
          NumericT * result,
          unsigned int result_start,
          unsigned int result_inc,
          unsigned int result_size)
{
  __shared__ NumericT work[128];

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;
  unsigned int lid = threadIdx.x;

  for (unsigned int row = row_gid; row < A_col_size; row += gridDim.x)
  {
    NumericT dot_prod = 0;
    for (unsigned int col = col_gid; col < A_row_size; col += blockDim.x)
      dot_prod += A[(row * A_col_inc + A_col_start) * A_internal_rows + col * A_row_inc + A_row_start] * v[v_start + v_inc * col];
    work[lid] = dot_prod;

    for (unsigned int stride = blockDim.x/2; stride>0; stride>>=1){
      __syncthreads();
      if (lid < stride)
        work[lid] += work[lid+stride];
    }

    if (lid == 0)
      result[row * result_inc + result_start] = work[0];
  }
}


//
// matrix-matrix products
//




//
// scaled rank-1-update
//

// alpha on CPU
template<typename NumericT>
__global__ void scaled_rank1_update_col_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          NumericT val,
          unsigned int options2,

          const NumericT * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,

          const NumericT * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2)
{
  NumericT alpha = val;
  if (options2 & (1 << 0))
    alpha = -alpha;
  if (options2 & (1 << 1))
    alpha = NumericT(1) / alpha;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
  {
    NumericT tmp = alpha * vec1[row * inc1 + start1];
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] += tmp * vec2[col * inc2 + start2];
  }
}


// alpha on GPU
template<typename NumericT>
__global__ void scaled_rank1_update_col_kernel(
          NumericT * A,
          unsigned int A_start1, unsigned int A_start2,
          unsigned int A_inc1,   unsigned int A_inc2,
          unsigned int A_size1,  unsigned int A_size2,
          unsigned int A_internal_size1,  unsigned int A_internal_size2,

          const NumericT * val,
          unsigned int options2,

          const NumericT * vec1,
          unsigned int start1,
          unsigned int inc1,
          unsigned int size1,

          const NumericT * vec2,
          unsigned int start2,
          unsigned int inc2,
          unsigned int size2)
{
  NumericT alpha = *val;
  if (options2 & (1 << 0))
    alpha = -alpha;
  if (options2 & (1 << 1))
    alpha = NumericT(1) / alpha;

  unsigned int row_gid = (blockIdx.x * blockDim.x + threadIdx.x) / blockDim.x;
  unsigned int col_gid = (blockIdx.x * blockDim.x + threadIdx.x) % blockDim.x;

  for (unsigned int row = row_gid; row < A_size1; row += gridDim.x)
  {
    NumericT tmp = alpha * vec1[row * inc1 + start1];
    for (unsigned int col = col_gid; col < A_size2; col += blockDim.x)
      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] += tmp * vec2[col * inc2 + start2];
  }
}


template <typename T>
__global__ void bidiag_pack_row_major_kernel(
            T * A,
            T * D,
            T * S,
            unsigned int size1,
            unsigned int size2,
            unsigned int stride)
{
  unsigned int size = min(size1, size2);
  if(blockIdx.x * blockDim.x + threadIdx.x == 0)
    S[0] = 0;

  for(unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
           i < size;
           i += gridDim.x * blockDim.x)
    {
      D[i] = A[i*stride + i];
      S[i+1] = (i + 1 < size2) ? A[i*stride + (i + 1)] : 0;
    }
}

template <typename T>
__global__ void bidiag_pack_column_major_kernel(
            T * A,
            T * D,
            T * S,
            unsigned int size1,
            unsigned int size2,
            unsigned int stride)
{
  unsigned int size = min(size1, size2);
  if(blockIdx.x * blockDim.x + threadIdx.x == 0)
    S[0] = 0;

  for(unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
           i < size;
           i += gridDim.x * blockDim.x)
    {
      D[i] = A[i*stride + i];
      S[i+1] = (i + 1 < size2) ? A[i + (i + 1) * stride] : 0;
    }
}



template<typename T>
__global__ void copy_col_row_major_kernel(
        T * A,
        T * V,
        unsigned int row_start,
        unsigned int col_start,
        unsigned int size,
        unsigned int stride)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sz = gridDim.x * blockDim.x;

    for(unsigned int i = row_start + x; i < size; i += sz)
    {
        V[i - row_start] = A[i * stride + col_start];
    }
}

template<typename T>
__global__ void copy_col_column_major_kernel(
        T * A,
        T * V,
        unsigned int row_start,
        unsigned int col_start,
        unsigned int size,
        unsigned int stride)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sz = gridDim.x * blockDim.x;

    for(unsigned int i = row_start + x; i < size; i += sz)
    {
        V[i - row_start] = A[i + col_start * stride];
    }
}

template<typename T>
__global__ void copy_row_row_major_kernel(
        T * A,
        T * V,
        unsigned int row_start,
        unsigned int col_start,
        unsigned int size,
        unsigned int stride)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sz = gridDim.x * blockDim.x;

    for(unsigned int i = col_start + x; i < size; i += sz)
    {
        V[i - col_start] = A[row_start * stride + i];
    }

}

template<typename T>
__global__ void copy_row_column_major_kernel(
        T * A,
        T * V,
        unsigned int row_start,
        unsigned int col_start,
        unsigned int size,
        unsigned int stride)
{
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int sz = gridDim.x * blockDim.x;

    for(unsigned int i = col_start + x; i < size; i += sz)
    {
        V[i - col_start] = A[row_start + i * stride];
    }

}



template<typename T>
__global__ void house_update_A_left_row_major_kernel(
        T * A,
        T * V,        //householder vector
        unsigned int row_start,
        unsigned int col_start,
        unsigned int size1,
        unsigned int size2,
        unsigned int stride)
{
    T ss = 0;

    for(unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + col_start;
        i < size2;
        i += gridDim.x * blockDim.x)
    {
        ss = 0;
        for(unsigned int j = row_start; j < size1; j++)
            ss = ss +(V[j] * A[j * stride + i]);

        for(unsigned int j = row_start; j < size1; j++)
            A[j * stride + i] = A[j * stride + i] - (2 * V[j] * ss);
    }
}

template<typename T>
__global__ void house_update_A_left_column_major_kernel(
        T * A,
        T * V,        //householder vector
        unsigned int row_start,
        unsigned int col_start,
        unsigned int size1,
        unsigned int size2,
        unsigned int stride)
{
    T ss = 0;

    for(unsigned int i = blockIdx.x * blockDim.x + threadIdx.x + col_start;
        i < size2;
        i += gridDim.x * blockDim.x)
    {
        ss = 0;
        for(unsigned int j = row_start; j < size1; j++)
            ss = ss +(V[j] * A[j + i * stride]);

        for(unsigned int j = row_start; j < size1; j++)
            A[j + i * stride] = A[j + i * stride] - (2 * V[j] * ss);
    }
}



template<typename T>
__global__ void house_update_A_right_row_major_kernel(
        T * A,
        T * V,  //householder vector
        unsigned int row_start,
        unsigned int col_start,
        unsigned int size1,
        unsigned int size2,
        unsigned int stride)
{
    __shared__ T sums[128];
    T ss = 0;

    for(unsigned int i = blockIdx.x + row_start; i < size1; i+= gridDim.x)
    {
        ss = 0;
        for(unsigned int j = threadIdx.x; j < size2; j+= blockDim.x)
            ss = ss + (V[j] * A[i * stride + j]);
        sums[threadIdx.x] = ss;

        __syncthreads();
        col_reduce_lcl_array(sums, threadIdx.x, blockDim.x);
        __syncthreads();

        T sum_Av = sums[0];

        for(unsigned int j = threadIdx.x; j < size2; j+= blockDim.x)
            A[i * stride + j] = A[i * stride + j] - (2 * V[j] * sum_Av);
    }
}

template<typename T>
__global__ void house_update_A_right_column_major_kernel(
        T * A,
        T * V,  //householder vector
        unsigned int row_start,
        unsigned int col_start,
        unsigned int size1,
        unsigned int size2,
        unsigned int stride)
{
    __shared__ T sums[128];
    T ss = 0;

    for(unsigned int i = blockIdx.x + row_start; i < size1; i+= gridDim.x)
    {
        ss = 0;
        for(unsigned int j = threadIdx.x; j < size2; j+= blockDim.x)
            ss = ss + (V[j] * A[i + j * stride]);
        sums[threadIdx.x] = ss;

        __syncthreads();
        col_reduce_lcl_array(sums, threadIdx.x, blockDim.x);
        __syncthreads();

        T sum_Av = sums[0];

        for(unsigned int j = threadIdx.x; j < size2; j+= blockDim.x)
            A[i + j * stride] = A[i + j * stride] - (2 * V[j] * sum_Av);
    }
}



template<typename T>
__device__ void col_reduce_lcl_array(
        T * sums,
        unsigned int th_Idx,
        unsigned int bl_Dim)
{
    unsigned int step = bl_Dim >> 1;

    while(step > 0)
    {
        if(th_Idx < step)
            sums[th_Idx] += sums[th_Idx + step];
        step >>= 1;
        __syncthreads();
    }
}


template <typename T>
__global__ void house_update_QL_row_major_kernel(
        T * QL,
        T * V,
        unsigned int size1,
        unsigned int strideQ)
{
  __shared__ T sums[128];
  T ss = 0;
  for(unsigned int i = blockIdx.x; i < size1; i += gridDim.x)
  {
    ss = 0;
    for(unsigned int j = threadIdx.x; j < size1; j += blockDim.x)
      ss = ss + (V[j] * QL[i * strideQ + j]);
    sums[threadIdx.x] = ss;

    __syncthreads();
    col_reduce_lcl_array(sums, threadIdx.x, blockDim.x);
    __syncthreads();

    T sum_Qv = sums[0];

    for(unsigned int j = threadIdx.x; j < size1; j += blockDim.x)
      QL[i * strideQ + j] = QL[i * strideQ + j] - (2 * V[j] * sum_Qv);
  }
}

template <typename T>
__global__ void house_update_QL_column_major_kernel(
        T * QL,
        T * V,
        unsigned int size1,
        unsigned int strideQ)
{
  __shared__ T sums[128];
  T ss = 0;
  for(unsigned int i = blockIdx.x; i < size1; i += gridDim.x)
  {
    ss = 0;
    for(unsigned int j = threadIdx.x; j < size1; j += blockDim.x)
      ss = ss + (V[j] * QL[i + j * strideQ]);
    sums[threadIdx.x] = ss;

    __syncthreads();
    col_reduce_lcl_array(sums, threadIdx.x, blockDim.x);
    __syncthreads();

    T sum_Qv = sums[0];

    for(unsigned int j = threadIdx.x; j < size1; j += blockDim.x)
      QL[i + j * strideQ] = QL[i + j * strideQ] - (2 * V[j] * sum_Qv);
  }
}


template <typename T>
__global__ void givens_next_row_major_kernel(
        T * matr,
        T * cs,
        T * ss,
        unsigned int size,
        unsigned int stride,
        unsigned int start_i,
        unsigned int end_i)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ T cs_lcl[256];
    __shared__ T ss_lcl[256];

    T x = (j < size) ? matr[(end_i + 1) + j * stride] : 0;

    unsigned int elems_num = end_i - start_i + 1;
    unsigned int block_num = (elems_num + blockDim.x - 1) / blockDim.x;

    for(unsigned int block_id = 0; block_id < block_num; block_id++)
    {
        unsigned int to = min(elems_num - block_id * blockDim.x, blockDim.x);

        if(threadIdx.x < to)
        {
            cs_lcl[threadIdx.x] = cs[end_i - (threadIdx.x + block_id * blockDim.x)];
            ss_lcl[threadIdx.x] = ss[end_i - (threadIdx.x + block_id * blockDim.x)];
        }
        __syncthreads();
        if(j < size)
        {
            for(unsigned int ind = 0; ind < to; ind++)
            {
                unsigned int i = end_i - (ind + block_id * blockDim.x);
                T z = matr[i + j * stride];
                T cs_val = cs_lcl[ind];
                T ss_val = ss_lcl[ind];
                matr[(i + 1) + j * stride] = x * cs_val + z * ss_val;
                x = -x * ss_val + z * cs_val;
            }
        }
        __syncthreads();
     }
     if(j < size)
       matr[(start_i) + j * stride] = x;
}

template <typename T>
__global__ void givens_next_column_major_kernel(
        T * matr,
        T * cs,
        T * ss,
        unsigned int size,
        unsigned int stride,
        unsigned int start_i,
        unsigned int end_i)
{
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ T cs_lcl[256];
    __shared__ T ss_lcl[256];

    T x = (j < size) ? matr[(end_i + 1) *stride + j] : 0;

    unsigned int elems_num = end_i - start_i + 1;
    unsigned int block_num = (elems_num + blockDim.x - 1) / blockDim.x;

    for(unsigned int block_id = 0; block_id < block_num; block_id++)
    {
        unsigned int to = min(elems_num - block_id * blockDim.x, blockDim.x);

        if(threadIdx.x < to)
        {
            cs_lcl[threadIdx.x] = cs[end_i - (threadIdx.x + block_id * blockDim.x)];
            ss_lcl[threadIdx.x] = ss[end_i - (threadIdx.x + block_id * blockDim.x)];
        }
        __syncthreads();
        if(j < size)
        {
            for(unsigned int ind = 0; ind < to; ind++)
            {
                unsigned int i = end_i - (ind + block_id * blockDim.x);
                T z = matr[i *stride + j];
                T cs_val = cs_lcl[ind];
                T ss_val = ss_lcl[ind];
                matr[(i + 1) * stride + j] = x * cs_val + z * ss_val;
                x = -x * ss_val + z * cs_val;
            }
        }
        __syncthreads();
     }
     if(j < size)
       matr[(start_i) * stride + j] = x;
}





} // namespace cuda
} //namespace linalg
} //namespace viennacl


#endif
