#ifndef VIENNACL_LINALG_CUDA_ILU_OPERATIONS_HPP_
#define VIENNACL_LINALG_CUDA_ILU_OPERATIONS_HPP_

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

/** @file viennacl/linalg/cuda/ilu_operations.hpp
    @brief Implementations of specialized routines for the Chow-Patel parallel ILU preconditioner using CUDA
*/

#include <cmath>
#include <algorithm>  //for std::max and std::min

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/linalg/vector_operations.hpp"
#include "viennacl/traits/stride.hpp"


namespace viennacl
{
namespace linalg
{
namespace cuda
{

template<typename IndexT> // to control external linkage
__global__ void extract_L_kernel_1(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          unsigned int A_size1,
          unsigned int * L_row_indices)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < A_size1;
                    row += gridDim.x * blockDim.x)
  {
    unsigned int row_begin = A_row_indices[row];
    unsigned int row_end   = A_row_indices[row+1];

    unsigned int num_entries_L = 0;
    for (unsigned int j=row_begin; j<row_end; ++j)
    {
      unsigned int col = A_col_indices[j];
      if (col <= row)
        ++num_entries_L;
    }

    L_row_indices[row] = num_entries_L;
  }
}

template<typename NumericT>
__global__ void extract_L_kernel_2(
          unsigned int const *A_row_indices,
          unsigned int const *A_col_indices,
          NumericT     const *A_elements,
          unsigned int A_size1,

          unsigned int const *L_row_indices,
          unsigned int       *L_col_indices,
          NumericT           *L_elements)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < A_size1;
                    row += gridDim.x * blockDim.x)
  {
    unsigned int row_begin = A_row_indices[row];
    unsigned int row_end   = A_row_indices[row+1];

    unsigned int index_L = L_row_indices[row];
    for (unsigned int j = row_begin; j < row_end; ++j)
    {
      unsigned int col = A_col_indices[j];
      NumericT value = A_elements[j];

      if (col <= row)
      {
        L_col_indices[index_L] = col;
        L_elements[index_L]    = value;
        ++index_L;
      }
    }
  }
}

template<typename NumericT>
void extract_L(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L)
{
  //
  // Step 1: Count elements in L and U:
  //
  extract_L_kernel_1<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(A.handle1()),
                                   viennacl::cuda_arg<unsigned int>(A.handle2()),
                                   static_cast<unsigned int>(A.size1()),
                                   viennacl::cuda_arg<unsigned int>(L.handle1())
                                  );
  VIENNACL_CUDA_LAST_ERROR_CHECK("extract_L_kernel_1");

  //
  // Step 2: Exclusive scan on row_buffers:
  //
  viennacl::vector<unsigned int> wrapped_L_row_buffer(viennacl::cuda_arg<unsigned int>(L.handle1().cuda_handle()), viennacl::CUDA_MEMORY, A.size1() + 1);
  viennacl::linalg::exclusive_scan(wrapped_L_row_buffer, wrapped_L_row_buffer);
  L.reserve(wrapped_L_row_buffer[L.size1()], false);

  //
  // Step 3: Write entries
  //
  extract_L_kernel_2<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(A.handle1()),
                                   viennacl::cuda_arg<unsigned int>(A.handle2()),
                                   viennacl::cuda_arg<NumericT>(A.handle()),
                                   static_cast<unsigned int>(A.size1()),
                                   viennacl::cuda_arg<unsigned int>(L.handle1()),
                                   viennacl::cuda_arg<unsigned int>(L.handle2()),
                                   viennacl::cuda_arg<NumericT>(L.handle())
                                  );
  VIENNACL_CUDA_LAST_ERROR_CHECK("extract_L_kernel_2");

  L.generate_row_block_information();

} // extract_L

///////////////////////////////////////////////


template<typename NumericT>
__global__ void ilu_scale_kernel_1(
          unsigned int const *A_row_indices,
          unsigned int const *A_col_indices,
          NumericT     const *A_elements,
          unsigned int A_size1,

          NumericT           *D_elements)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < A_size1;
                    row += gridDim.x * blockDim.x)
  {
    unsigned int row_begin = A_row_indices[row];
    unsigned int row_end   = A_row_indices[row+1];

    for (unsigned int j = row_begin; j < row_end; ++j)
    {
      unsigned int col = A_col_indices[j];
      if (row == col)
      {
        D_elements[row] = NumericT(1) / sqrt(fabs(A_elements[j]));
        break;
      }
    }
  }
}

/** @brief Scales values in a matrix such that output = D * input * D, where D is a diagonal matrix (only the diagonal is provided) */
template<typename NumericT>
__global__ void ilu_scale_kernel_2(
          unsigned int const *R_row_indices,
          unsigned int const *R_col_indices,
          NumericT           *R_elements,
          unsigned int R_size1,

          NumericT           *D_elements)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < R_size1;
                    row += gridDim.x * blockDim.x)
  {
    unsigned int row_begin = R_row_indices[row];
    unsigned int row_end   = R_row_indices[row+1];

    NumericT D_row = D_elements[row];

    for (unsigned int j = row_begin; j < row_end; ++j)
      R_elements[j] *= D_row * D_elements[R_col_indices[j]];
  }
}



/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L and U accordingly. */
template<typename NumericT>
void icc_scale(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L)
{
  viennacl::vector<NumericT> D(A.size1(), viennacl::traits::context(A));

  // fill D:
  ilu_scale_kernel_1<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(A.handle1()),
                                   viennacl::cuda_arg<unsigned int>(A.handle2()),
                                   viennacl::cuda_arg<NumericT>(A.handle()),
                                   static_cast<unsigned int>(A.size1()),
                                   viennacl::cuda_arg(D)
                                  );
  VIENNACL_CUDA_LAST_ERROR_CHECK("ilu_scale_kernel_1");

  // scale L:
  ilu_scale_kernel_2<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(L.handle1()),
                                   viennacl::cuda_arg<unsigned int>(L.handle2()),
                                   viennacl::cuda_arg<NumericT>(L.handle()),
                                   static_cast<unsigned int>(L.size1()),
                                   viennacl::cuda_arg(D)
                                  );
  VIENNACL_CUDA_LAST_ERROR_CHECK("ilu_scale_kernel_1");
}

/////////////////////////////////////

/** @brief CUDA kernel for one Chow-Patel-ICC sweep */
template<typename NumericT>
__global__ void icc_chow_patel_sweep_kernel(
          unsigned int const *L_row_indices,
          unsigned int const *L_col_indices,
          NumericT           *L_elements,
          NumericT     const *L_backup,
          unsigned int L_size1,
          NumericT     const *aij_L)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < L_size1;
                    row += gridDim.x * blockDim.x)
  {
    //
    // update L:
    //
    unsigned int row_Li_start = L_row_indices[row];
    unsigned int row_Li_end   = L_row_indices[row + 1];

    for (unsigned int i = row_Li_start; i < row_Li_end; ++i)
    {
      unsigned int col = L_col_indices[i];

      unsigned int row_Lj_start = L_row_indices[col];
      unsigned int row_Lj_end   = L_row_indices[col + 1];

      // compute \sum_{k=1}^{j-1} l_ik u_kj
      unsigned int index_Lj = row_Lj_start;
      unsigned int col_Lj = L_col_indices[index_Lj];
      NumericT s = aij_L[i];
      for (unsigned int index_Li = row_Li_start; index_Li < i; ++index_Li)
      {
        unsigned int col_Li = L_col_indices[index_Li];

        // find element in U
        while (col_Lj < col_Li)
        {
          ++index_Lj;
          col_Lj = L_col_indices[index_Lj];
        }

        if (col_Lj == col_Li)
          s -= L_backup[index_Li] * L_backup[index_Lj];
      }

      // update l_ij:
      L_elements[i] = (row == col) ? sqrt(s) : (s / L_backup[row_Lj_end - 1]);  // diagonal element is last entry in U
    }

  }
}


/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ILU using OpenMP (cf. Algorithm 2 in paper) */
template<typename NumericT>
void icc_chow_patel_sweep(compressed_matrix<NumericT>       & L,
                          vector<NumericT>            const & aij_L)
{
  viennacl::backend::mem_handle L_backup;
  viennacl::backend::memory_create(L_backup, L.handle().raw_size(), viennacl::traits::context(L));
  viennacl::backend::memory_copy(L.handle(), L_backup, 0, 0, L.handle().raw_size());

  icc_chow_patel_sweep_kernel<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(L.handle1()),
                                            viennacl::cuda_arg<unsigned int>(L.handle2()),
                                            viennacl::cuda_arg<NumericT>(L.handle()),
                                            viennacl::cuda_arg<NumericT>(L_backup),
                                            static_cast<unsigned int>(L.size1()),

                                            viennacl::cuda_arg<NumericT>(aij_L.handle())
                                           );
  VIENNACL_CUDA_LAST_ERROR_CHECK("icc_chow_patel_sweep_kernel");

}


////////////////////////////// ILU ///////////////////////////

template<typename IndexT> // to control external linkage
__global__ void extract_LU_kernel_1(
          const IndexT * A_row_indices,
          const IndexT * A_col_indices,
          unsigned int A_size1,

          unsigned int * L_row_indices,

          unsigned int * U_row_indices)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < A_size1;
                    row += gridDim.x * blockDim.x)
  {
    unsigned int row_begin = A_row_indices[row];
    unsigned int row_end   = A_row_indices[row+1];

    unsigned int num_entries_L = 0;
    unsigned int num_entries_U = 0;
    for (unsigned int j=row_begin; j<row_end; ++j)
    {
      unsigned int col = A_col_indices[j];
      if (col <= row)
        ++num_entries_L;
      if (col >= row)
        ++num_entries_U;
    }

    L_row_indices[row] = num_entries_L;
    U_row_indices[row] = num_entries_U;
  }
}

template<typename NumericT>
__global__ void extract_LU_kernel_2(
          unsigned int const *A_row_indices,
          unsigned int const *A_col_indices,
          NumericT     const *A_elements,
          unsigned int A_size1,

          unsigned int const *L_row_indices,
          unsigned int       *L_col_indices,
          NumericT           *L_elements,

          unsigned int const *U_row_indices,
          unsigned int       *U_col_indices,
          NumericT           *U_elements)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < A_size1;
                    row += gridDim.x * blockDim.x)
  {
    unsigned int row_begin = A_row_indices[row];
    unsigned int row_end   = A_row_indices[row+1];

    unsigned int index_L = L_row_indices[row];
    unsigned int index_U = U_row_indices[row];
    for (unsigned int j = row_begin; j < row_end; ++j)
    {
      unsigned int col = A_col_indices[j];
      NumericT value = A_elements[j];

      if (col <= row)
      {
        L_col_indices[index_L] = col;
        L_elements[index_L]    = value;
        ++index_L;
      }

      if (col >= row)
      {
        U_col_indices[index_U] = col;
        U_elements[index_U]    = value;
        ++index_U;
      }
    }
  }
}

template<typename NumericT>
void extract_LU(compressed_matrix<NumericT> const & A,
                compressed_matrix<NumericT>       & L,
                compressed_matrix<NumericT>       & U)
{
  //
  // Step 1: Count elements in L and U:
  //
  extract_LU_kernel_1<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(A.handle1()),
                                    viennacl::cuda_arg<unsigned int>(A.handle2()),
                                    static_cast<unsigned int>(A.size1()),
                                    viennacl::cuda_arg<unsigned int>(L.handle1()),
                                    viennacl::cuda_arg<unsigned int>(U.handle1())
                                   );
  VIENNACL_CUDA_LAST_ERROR_CHECK("extract_LU_kernel_1");

  //
  // Step 2: Exclusive scan on row_buffers:
  //
  viennacl::vector<unsigned int> wrapped_L_row_buffer(viennacl::cuda_arg<unsigned int>(L.handle1()), viennacl::CUDA_MEMORY, A.size1() + 1);
  viennacl::linalg::exclusive_scan(wrapped_L_row_buffer, wrapped_L_row_buffer);
  L.reserve(wrapped_L_row_buffer[L.size1()], false);

  viennacl::vector<unsigned int> wrapped_U_row_buffer(viennacl::cuda_arg<unsigned int>(U.handle1()), viennacl::CUDA_MEMORY, A.size1() + 1);
  viennacl::linalg::exclusive_scan(wrapped_U_row_buffer, wrapped_U_row_buffer);
  U.reserve(wrapped_U_row_buffer[U.size1()], false);

  //
  // Step 3: Write entries
  //
  extract_LU_kernel_2<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(A.handle1()),
                                    viennacl::cuda_arg<unsigned int>(A.handle2()),
                                    viennacl::cuda_arg<NumericT>(A.handle()),
                                    static_cast<unsigned int>(A.size1()),
                                    viennacl::cuda_arg<unsigned int>(L.handle1()),
                                    viennacl::cuda_arg<unsigned int>(L.handle2()),
                                    viennacl::cuda_arg<NumericT>(L.handle()),
                                    viennacl::cuda_arg<unsigned int>(U.handle1()),
                                    viennacl::cuda_arg<unsigned int>(U.handle2()),
                                    viennacl::cuda_arg<NumericT>(U.handle())
                                   );
  VIENNACL_CUDA_LAST_ERROR_CHECK("extract_LU_kernel_2");

  L.generate_row_block_information();
  // Note: block information for U will be generated after transposition

} // extract_LU

///////////////////////////////////////////////

/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L and U accordingly. */
template<typename NumericT>
void ilu_scale(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L,
               compressed_matrix<NumericT>       & U)
{
  viennacl::vector<NumericT> D(A.size1(), viennacl::traits::context(A));

  // fill D:
  ilu_scale_kernel_1<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(A.handle1()),
                                   viennacl::cuda_arg<unsigned int>(A.handle2()),
                                   viennacl::cuda_arg<NumericT>(A.handle()),
                                   static_cast<unsigned int>(A.size1()),
                                   viennacl::cuda_arg<NumericT>(D.handle())
                                  );
  VIENNACL_CUDA_LAST_ERROR_CHECK("ilu_scale_kernel_1");

  // scale L:
  ilu_scale_kernel_2<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(L.handle1()),
                                   viennacl::cuda_arg<unsigned int>(L.handle2()),
                                   viennacl::cuda_arg<NumericT>(L.handle()),
                                   static_cast<unsigned int>(L.size1()),
                                   viennacl::cuda_arg<NumericT>(D.handle())
                                  );
  VIENNACL_CUDA_LAST_ERROR_CHECK("ilu_scale_kernel_2");

  // scale U:
  ilu_scale_kernel_2<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(U.handle1()),
                                   viennacl::cuda_arg<unsigned int>(U.handle2()),
                                   viennacl::cuda_arg<NumericT>(U.handle()),
                                   static_cast<unsigned int>(U.size1()),
                                   viennacl::cuda_arg<NumericT>(D.handle())
                                  );
  VIENNACL_CUDA_LAST_ERROR_CHECK("ilu_scale_kernel_2");
}

/////////////////////////////////////

/** @brief CUDA kernel for one Chow-Patel-ILU sweep */
template<typename NumericT>
__global__ void ilu_chow_patel_sweep_kernel(
          unsigned int const *L_row_indices,
          unsigned int const *L_col_indices,
          NumericT           *L_elements,
          NumericT     const *L_backup,
          unsigned int L_size1,

          NumericT     const *aij_L,

          unsigned int const *U_trans_row_indices,
          unsigned int const *U_trans_col_indices,
          NumericT           *U_trans_elements,
          NumericT     const *U_trans_backup,

          NumericT     const *aij_U_trans)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < L_size1;
                    row += gridDim.x * blockDim.x)
  {
    //
    // update L:
    //
    unsigned int row_L_start = L_row_indices[row];
    unsigned int row_L_end   = L_row_indices[row + 1];

    for (unsigned int j = row_L_start; j < row_L_end; ++j)
    {
      unsigned int col = L_col_indices[j];

      if (col == row)
        continue;

      unsigned int row_U_start = U_trans_row_indices[col];
      unsigned int row_U_end   = U_trans_row_indices[col + 1];

      // compute \sum_{k=1}^{j-1} l_ik u_kj
      unsigned int index_U = row_U_start;
      unsigned int col_U = (index_U < row_U_end) ? U_trans_col_indices[index_U] : L_size1;
      NumericT sum = 0;
      for (unsigned int k = row_L_start; k < j; ++k)
      {
        unsigned int col_L = L_col_indices[k];

        // find element in U
        while (col_U < col_L)
        {
          ++index_U;
          col_U = U_trans_col_indices[index_U];
        }

        if (col_U == col_L)
          sum += L_backup[k] * U_trans_backup[index_U];
      }

      // update l_ij:
      L_elements[j] = (aij_L[j] - sum) / U_trans_backup[row_U_end - 1];  // diagonal element is last entry in U
    }


    //
    // update U:
    //
    unsigned int row_U_start = U_trans_row_indices[row];
    unsigned int row_U_end   = U_trans_row_indices[row + 1];
    for (unsigned int j = row_U_start; j < row_U_end; ++j)
    {
      unsigned int col = U_trans_col_indices[j];

      row_L_start = L_row_indices[col];
      row_L_end   = L_row_indices[col + 1];

      // compute \sum_{k=1}^{j-1} l_ik u_kj
      unsigned int index_L = row_L_start;
      unsigned int col_L = (index_L < row_L_end) ? L_col_indices[index_L] : L_size1;
      NumericT sum = 0;
      for (unsigned int k = row_U_start; k < j; ++k)
      {
        unsigned int col_U = U_trans_col_indices[k];

        // find element in L
        while (col_L < col_U)
        {
          ++index_L;
          col_L = L_col_indices[index_L];
        }

        if (col_U == col_L)
          sum += L_backup[index_L] * U_trans_backup[k];
      }

      // update u_ij:
      U_trans_elements[j] = aij_U_trans[j] - sum;
    }
  }
}


/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ILU using OpenMP (cf. Algorithm 2 in paper) */
template<typename NumericT>
void ilu_chow_patel_sweep(compressed_matrix<NumericT>       & L,
                          vector<NumericT>            const & aij_L,
                          compressed_matrix<NumericT>       & U_trans,
                          vector<NumericT>            const & aij_U_trans)
{
  viennacl::backend::mem_handle L_backup;
  viennacl::backend::memory_create(L_backup, L.handle().raw_size(), viennacl::traits::context(L));
  viennacl::backend::memory_copy(L.handle(), L_backup, 0, 0, L.handle().raw_size());

  viennacl::backend::mem_handle U_backup;
  viennacl::backend::memory_create(U_backup, U_trans.handle().raw_size(), viennacl::traits::context(U_trans));
  viennacl::backend::memory_copy(U_trans.handle(), U_backup, 0, 0, U_trans.handle().raw_size());

  ilu_chow_patel_sweep_kernel<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(L.handle1()),
                                            viennacl::cuda_arg<unsigned int>(L.handle2()),
                                            viennacl::cuda_arg<NumericT>(L.handle()),
                                            viennacl::cuda_arg<NumericT>(L_backup),
                                            static_cast<unsigned int>(L.size1()),

                                            viennacl::cuda_arg<NumericT>(aij_L.handle()),

                                            viennacl::cuda_arg<unsigned int>(U_trans.handle1()),
                                            viennacl::cuda_arg<unsigned int>(U_trans.handle2()),
                                            viennacl::cuda_arg<NumericT>(U_trans.handle()),
                                            viennacl::cuda_arg<NumericT>(U_backup),

                                            viennacl::cuda_arg<NumericT>(aij_U_trans.handle())
                                           );
  VIENNACL_CUDA_LAST_ERROR_CHECK("ilu_chow_patel_sweep_kernel");

}

//////////////////////////////////////

template<typename NumericT>
__global__ void ilu_form_neumann_matrix_kernel(
          unsigned int const *R_row_indices,
          unsigned int const *R_col_indices,
          NumericT           *R_elements,
          unsigned int R_size1,

          NumericT           *D_elements)
{
  for (unsigned int row  = blockDim.x * blockIdx.x + threadIdx.x;
                    row  < R_size1;
                    row += gridDim.x * blockDim.x)
  {
    unsigned int row_begin = R_row_indices[row];
    unsigned int row_end   = R_row_indices[row+1];

    // part 1: extract diagonal entry
    NumericT diag = 0;
    for (unsigned int j = row_begin; j < row_end; ++j)
    {
      unsigned int col = R_col_indices[j];
      if (col == row)
      {
        diag = R_elements[j];
        R_elements[j] = 0; // (I - D^{-1}R)
        break;
      }
    }
    D_elements[row] = diag;

    // part2: scale
    for (unsigned int j = row_begin; j < row_end; ++j)
      R_elements[j] /= -diag;
  }
}



template<typename NumericT>
void ilu_form_neumann_matrix(compressed_matrix<NumericT> & R,
                             vector<NumericT> & diag_R)
{
  ilu_form_neumann_matrix_kernel<<<128, 128>>>(viennacl::cuda_arg<unsigned int>(R.handle1()),
                                               viennacl::cuda_arg<unsigned int>(R.handle2()),
                                               viennacl::cuda_arg<NumericT>(R.handle()),
                                               static_cast<unsigned int>(R.size1()),
                                               viennacl::cuda_arg<NumericT>(diag_R.handle())
                                              );
  VIENNACL_CUDA_LAST_ERROR_CHECK("ilu_form_neumann_matrix_kernel");
}

} //namespace host_based
} //namespace linalg
} //namespace viennacl


#endif
