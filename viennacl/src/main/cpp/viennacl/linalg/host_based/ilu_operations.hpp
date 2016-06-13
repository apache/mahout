#ifndef VIENNACL_LINALG_HOST_BASED_ILU_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_ILU_OPERATIONS_HPP_

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

/** @file viennacl/linalg/host_based/ilu_operations.hpp
    @brief Implementations of specialized routines for the Chow-Patel parallel ILU preconditioner using the host (OpenMP)
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
#include "viennacl/linalg/host_based/common.hpp"
#include "viennacl/linalg/vector_operations.hpp"
#include "viennacl/traits/stride.hpp"


// Minimum vector size for using OpenMP on vector operations:
#ifndef VIENNACL_OPENMP_ILU_MIN_SIZE
  #define VIENNACL_OPENMP_ILU_MIN_SIZE  5000
#endif

namespace viennacl
{
namespace linalg
{
namespace host_based
{

template<typename NumericT>
void extract_L(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L)
{
  // L is known to have correct dimensions

  unsigned int const *A_row_buffer = detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const *A_col_buffer = detail::extract_raw_pointer<unsigned int>(A.handle2());
  NumericT     const *A_elements   = detail::extract_raw_pointer<NumericT>(A.handle());

  unsigned int       *L_row_buffer = detail::extract_raw_pointer<unsigned int>(L.handle1());

  //
  // Step 1: Count elements in L
  //
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (A.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(A.size1()); ++row)
  {
    unsigned int col_begin = A_row_buffer[row];
    unsigned int col_end   = A_row_buffer[row+1];

    for (unsigned int j = col_begin; j < col_end; ++j)
    {
      unsigned int col = A_col_buffer[j];
      if (long(col) <= row)
        ++L_row_buffer[row];
    }
  }

  //
  // Step 2: Exclusive scan on row_buffer arrays to get correct starting indices
  //
  viennacl::vector_base<unsigned int> wrapped_L_row_buffer(L.handle1(), L.size1() + 1, 0, 1);
  viennacl::linalg::exclusive_scan(wrapped_L_row_buffer);
  L.reserve(wrapped_L_row_buffer[L.size1()], false);

  unsigned int       *L_col_buffer = detail::extract_raw_pointer<unsigned int>(L.handle2());
  NumericT           *L_elements   = detail::extract_raw_pointer<NumericT>(L.handle());

  //
  // Step 3: Write entries:
  //
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (A.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(A.size1()); ++row)
  {
    unsigned int col_begin = A_row_buffer[row];
    unsigned int col_end   = A_row_buffer[row+1];

    unsigned int index_L = L_row_buffer[row];
    for (unsigned int j = col_begin; j < col_end; ++j)
    {
      unsigned int col = A_col_buffer[j];
      NumericT value = A_elements[j];

      if (long(col) <= row)
      {
        L_col_buffer[index_L] = col;
        L_elements[index_L]   = value;
        ++index_L;
      }
    }
  }

} // extract_L


/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L and U accordingly. */
template<typename NumericT>
void icc_scale(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L)
{
  viennacl::vector<NumericT> D(A.size1(), viennacl::traits::context(A));

  unsigned int const *A_row_buffer = detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const *A_col_buffer = detail::extract_raw_pointer<unsigned int>(A.handle2());
  NumericT     const *A_elements   = detail::extract_raw_pointer<NumericT>(A.handle());

  NumericT           *D_elements   = detail::extract_raw_pointer<NumericT>(D.handle());

  //
  // Step 1: Determine D
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for if (A.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(A.size1()); ++row)
  {
    unsigned int col_begin = A_row_buffer[row];
    unsigned int col_end   = A_row_buffer[row+1];

    for (unsigned int j = col_begin; j < col_end; ++j)
    {
      unsigned int col = A_col_buffer[j];
      if (row == col)
      {
        D_elements[row] = NumericT(1) / std::sqrt(std::fabs(A_elements[j]));
        break;
      }
    }
  }

  //
  // Step 2: Scale values in L:
  //
  unsigned int const *L_row_buffer = detail::extract_raw_pointer<unsigned int>(L.handle1());
  unsigned int const *L_col_buffer = detail::extract_raw_pointer<unsigned int>(L.handle2());
  NumericT           *L_elements   = detail::extract_raw_pointer<NumericT>(L.handle());

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for if (A.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(A.size1()); ++row)
  {
    unsigned int col_begin = L_row_buffer[row];
    unsigned int col_end   = L_row_buffer[row+1];

    NumericT D_row = D_elements[row];

    for (unsigned int j = col_begin; j < col_end; ++j)
      L_elements[j] *= D_row * D_elements[L_col_buffer[j]];
  }

  L.generate_row_block_information();
}



/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ICC using OpenMP (cf. Algorithm 3 in paper, but for L rather than U) */
template<typename NumericT>
void icc_chow_patel_sweep(compressed_matrix<NumericT> & L,
                          vector<NumericT>            & aij_L)
{
  unsigned int const *L_row_buffer = detail::extract_raw_pointer<unsigned int>(L.handle1());
  unsigned int const *L_col_buffer = detail::extract_raw_pointer<unsigned int>(L.handle2());
  NumericT           *L_elements   = detail::extract_raw_pointer<NumericT>(L.handle());

  NumericT           *aij_ptr   = detail::extract_raw_pointer<NumericT>(aij_L.handle());

  // temporary workspace
  NumericT *L_backup = (NumericT *)malloc(sizeof(NumericT) * L.nnz());

  // backup:
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (L.nnz() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long i = 0; i < static_cast<long>(L.nnz()); ++i)
    L_backup[i] = L_elements[i];


  // sweep
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (L.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(L.size1()); ++row)
  {
    //
    // update L:
    //
    unsigned int row_Li_start = L_row_buffer[row];
    unsigned int row_Li_end   = L_row_buffer[row + 1];

    for (unsigned int i = row_Li_start; i < row_Li_end; ++i)
    {
      unsigned int col = L_col_buffer[i];

      unsigned int row_Lj_start = L_row_buffer[col];
      unsigned int row_Lj_end   = L_row_buffer[col+1];

      // compute \sum_{k=1}^{j-1} l_ik l_jk
      unsigned int index_Lj = row_Lj_start;
      unsigned int col_Lj = L_col_buffer[index_Lj];
      NumericT s = aij_ptr[i];
      for (unsigned int index_Li = row_Li_start; index_Li < i; ++index_Li)
      {
        unsigned int col_Li = L_col_buffer[index_Li];

        // find element in row j
        while (col_Lj < col_Li)
        {
          ++index_Lj;
          col_Lj = L_col_buffer[index_Lj];
        }

        if (col_Lj == col_Li)
          s -= L_backup[index_Li] * L_backup[index_Lj];
      }

      if (row != col)
        L_elements[i] = s / L_backup[row_Lj_end - 1]; // diagonal element is last in row!
      else
        L_elements[i] = std::sqrt(s);
    }
  }

  free(L_backup);
}



//////////////////////// ILU ////////////////////////

template<typename NumericT>
void extract_LU(compressed_matrix<NumericT> const & A,
                compressed_matrix<NumericT>       & L,
                compressed_matrix<NumericT>       & U)
{
  // L and U are known to have correct dimensions

  unsigned int const *A_row_buffer = detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const *A_col_buffer = detail::extract_raw_pointer<unsigned int>(A.handle2());
  NumericT     const *A_elements   = detail::extract_raw_pointer<NumericT>(A.handle());

  unsigned int       *L_row_buffer = detail::extract_raw_pointer<unsigned int>(L.handle1());
  unsigned int       *U_row_buffer = detail::extract_raw_pointer<unsigned int>(U.handle1());

  //
  // Step 1: Count elements in L and U
  //
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (A.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(A.size1()); ++row)
  {
    unsigned int col_begin = A_row_buffer[row];
    unsigned int col_end   = A_row_buffer[row+1];

    for (unsigned int j = col_begin; j < col_end; ++j)
    {
      unsigned int col = A_col_buffer[j];
      if (long(col) <= row)
        ++L_row_buffer[row];
      if (long(col) >= row)
        ++U_row_buffer[row];
    }
  }

  //
  // Step 2: Exclusive scan on row_buffer arrays to get correct starting indices
  //
  viennacl::vector_base<unsigned int> wrapped_L_row_buffer(L.handle1(), L.size1() + 1, 0, 1);
  viennacl::linalg::exclusive_scan(wrapped_L_row_buffer);
  L.reserve(wrapped_L_row_buffer[L.size1()], false);

  viennacl::vector_base<unsigned int> wrapped_U_row_buffer(U.handle1(), U.size1() + 1, 0, 1);
  viennacl::linalg::exclusive_scan(wrapped_U_row_buffer);
  U.reserve(wrapped_U_row_buffer[U.size1()], false);

  unsigned int       *L_col_buffer = detail::extract_raw_pointer<unsigned int>(L.handle2());
  NumericT           *L_elements   = detail::extract_raw_pointer<NumericT>(L.handle());

  unsigned int       *U_col_buffer = detail::extract_raw_pointer<unsigned int>(U.handle2());
  NumericT           *U_elements   = detail::extract_raw_pointer<NumericT>(U.handle());

  //
  // Step 3: Write entries:
  //
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (A.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(A.size1()); ++row)
  {
    unsigned int col_begin = A_row_buffer[row];
    unsigned int col_end   = A_row_buffer[row+1];

    unsigned int index_L = L_row_buffer[row];
    unsigned int index_U = U_row_buffer[row];
    for (unsigned int j = col_begin; j < col_end; ++j)
    {
      unsigned int col = A_col_buffer[j];
      NumericT value = A_elements[j];

      if (long(col) <= row)
      {
        L_col_buffer[index_L] = col;
        L_elements[index_L]   = value;
        ++index_L;
      }

      if (long(col) >= row)
      {
        U_col_buffer[index_U] = col;
        U_elements[index_U]   = value;
        ++index_U;
      }
    }
  }

} // extract_LU



/** @brief Scales the values extracted from A such that A' = DAD has unit diagonal. Updates values from A in L and U accordingly. */
template<typename NumericT>
void ilu_scale(compressed_matrix<NumericT> const & A,
               compressed_matrix<NumericT>       & L,
               compressed_matrix<NumericT>       & U)
{
  viennacl::vector<NumericT> D(A.size1(), viennacl::traits::context(A));

  unsigned int const *A_row_buffer = detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const *A_col_buffer = detail::extract_raw_pointer<unsigned int>(A.handle2());
  NumericT     const *A_elements   = detail::extract_raw_pointer<NumericT>(A.handle());

  NumericT           *D_elements   = detail::extract_raw_pointer<NumericT>(D.handle());

  //
  // Step 1: Determine D
  //
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for if (A.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(A.size1()); ++row)
  {
    unsigned int col_begin = A_row_buffer[row];
    unsigned int col_end   = A_row_buffer[row+1];

    for (unsigned int j = col_begin; j < col_end; ++j)
    {
      unsigned int col = A_col_buffer[j];
      if (row == col)
      {
        D_elements[row] = NumericT(1) / std::sqrt(std::fabs(A_elements[j]));
        break;
      }
    }
  }

  //
  // Step 2: Scale values in L:
  //
  unsigned int const *L_row_buffer = detail::extract_raw_pointer<unsigned int>(L.handle1());
  unsigned int const *L_col_buffer = detail::extract_raw_pointer<unsigned int>(L.handle2());
  NumericT           *L_elements   = detail::extract_raw_pointer<NumericT>(L.handle());

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for if (A.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(A.size1()); ++row)
  {
    unsigned int col_begin = L_row_buffer[row];
    unsigned int col_end   = L_row_buffer[row+1];

    NumericT D_row = D_elements[row];

    for (unsigned int j = col_begin; j < col_end; ++j)
      L_elements[j] *= D_row * D_elements[L_col_buffer[j]];
  }

  //
  // Step 3: Scale values in U:
  //
  unsigned int const *U_row_buffer = detail::extract_raw_pointer<unsigned int>(U.handle1());
  unsigned int const *U_col_buffer = detail::extract_raw_pointer<unsigned int>(U.handle2());
  NumericT           *U_elements   = detail::extract_raw_pointer<NumericT>(U.handle());

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for if (A.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(A.size1()); ++row)
  {
    unsigned int col_begin = U_row_buffer[row];
    unsigned int col_end   = U_row_buffer[row+1];

    NumericT D_row = D_elements[row];

    for (unsigned int j = col_begin; j < col_end; ++j)
      U_elements[j] *= D_row * D_elements[U_col_buffer[j]];
  }

  L.generate_row_block_information();
  // Note: block information for U will be generated after transposition

}

template<typename NumericT>
void ilu_transpose(compressed_matrix<NumericT> const & A,
                   compressed_matrix<NumericT>       & B)
{
  NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  // initialize datastructures for B:
  B = compressed_matrix<NumericT>(A.size2(), A.size1(), A.nnz(), viennacl::traits::context(A));

  NumericT     * B_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(B.handle());
  unsigned int * B_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(B.handle1());
  unsigned int * B_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(B.handle2());

  // prepare uninitialized B_row_buffer:
  for (std::size_t i = 0; i < B.size1(); ++i)
    B_row_buffer[i] = 0;

  //
  // Stage 1: Compute pattern for B
  //
  for (std::size_t row = 0; row < A.size1(); ++row)
  {
    unsigned int row_start = A_row_buffer[row];
    unsigned int row_stop  = A_row_buffer[row+1];

    for (unsigned int nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
      B_row_buffer[A_col_buffer[nnz_index]] += 1;
  }

  // Bring row-start array in place using exclusive-scan:
  unsigned int offset = B_row_buffer[0];
  B_row_buffer[0] = 0;
  for (std::size_t row = 1; row < B.size1(); ++row)
  {
    unsigned int tmp = B_row_buffer[row];
    B_row_buffer[row] = offset;
    offset += tmp;
  }
  B_row_buffer[B.size1()] = offset;

  //
  // Stage 2: Fill with data
  //

  std::vector<unsigned int> B_row_offsets(B.size1()); //number of elements already written per row

  for (unsigned int row = 0; row < static_cast<unsigned int>(A.size1()); ++row)
  {
    //std::cout << "Row " << row << ": ";
    unsigned int row_start = A_row_buffer[row];
    unsigned int row_stop  = A_row_buffer[row+1];

    for (unsigned int nnz_index = row_start; nnz_index < row_stop; ++nnz_index)
    {
      unsigned int col_in_A = A_col_buffer[nnz_index];
      unsigned int B_nnz_index = B_row_buffer[col_in_A] + B_row_offsets[col_in_A];
      B_col_buffer[B_nnz_index] = row;
      B_elements[B_nnz_index] = A_elements[nnz_index];
      ++B_row_offsets[col_in_A];
      //B_temp.at(A_col_buffer[nnz_index])[row] = A_elements[nnz_index];
    }
  }

  // Step 3: Make datastructure consistent (row blocks!)
  B.generate_row_block_information();
}



/** @brief Performs one nonlinear relaxation step in the Chow-Patel-ILU using OpenMP (cf. Algorithm 2 in paper) */
template<typename NumericT>
void ilu_chow_patel_sweep(compressed_matrix<NumericT>       & L,
                          vector<NumericT>            const & aij_L,
                          compressed_matrix<NumericT>       & U_trans,
                          vector<NumericT>            const & aij_U_trans)
{
  unsigned int const *L_row_buffer = detail::extract_raw_pointer<unsigned int>(L.handle1());
  unsigned int const *L_col_buffer = detail::extract_raw_pointer<unsigned int>(L.handle2());
  NumericT           *L_elements   = detail::extract_raw_pointer<NumericT>(L.handle());

  NumericT     const *aij_L_ptr    = detail::extract_raw_pointer<NumericT>(aij_L.handle());

  unsigned int const *U_row_buffer = detail::extract_raw_pointer<unsigned int>(U_trans.handle1());
  unsigned int const *U_col_buffer = detail::extract_raw_pointer<unsigned int>(U_trans.handle2());
  NumericT           *U_elements   = detail::extract_raw_pointer<NumericT>(U_trans.handle());

  NumericT     const *aij_U_trans_ptr = detail::extract_raw_pointer<NumericT>(aij_U_trans.handle());

  // temporary workspace
  NumericT *L_backup = new NumericT[L.nnz()];
  NumericT *U_backup = new NumericT[U_trans.nnz()];

  // backup:
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (L.nnz() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long i = 0; i < static_cast<long>(L.nnz()); ++i)
    L_backup[i] = L_elements[i];

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (U_trans.nnz() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long i = 0; i < static_cast<long>(U_trans.nnz()); ++i)
    U_backup[i] = U_elements[i];

  // sweep
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (L.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(L.size1()); ++row)
  {
    //
    // update L:
    //
    unsigned int row_L_start = L_row_buffer[row];
    unsigned int row_L_end   = L_row_buffer[row + 1];

    for (unsigned int j = row_L_start; j < row_L_end; ++j)
    {
      unsigned int col = L_col_buffer[j];

      if (col == row)
        continue;

      unsigned int row_U_start = U_row_buffer[col];
      unsigned int row_U_end   = U_row_buffer[col + 1];

      // compute \sum_{k=1}^{j-1} l_ik u_kj
      unsigned int index_U = row_U_start;
      unsigned int col_U = (index_U < row_U_end) ? U_col_buffer[index_U] : static_cast<unsigned int>(U_trans.size2());
      NumericT sum = 0;
      for (unsigned int k = row_L_start; k < j; ++k)
      {
        unsigned int col_L = L_col_buffer[k];

        // find element in U
        while (col_U < col_L)
        {
          ++index_U;
          col_U = U_col_buffer[index_U];
        }

        if (col_U == col_L)
          sum += L_backup[k] * U_backup[index_U];
      }

      // update l_ij:
      assert(U_col_buffer[row_U_end - 1] == col && bool("Accessing U element which is not a diagonal element!"));
      L_elements[j] = (aij_L_ptr[j] - sum) / U_backup[row_U_end - 1];  // diagonal element is last entry in U
    }


    //
    // update U:
    //
    unsigned int row_U_start = U_row_buffer[row];
    unsigned int row_U_end   = U_row_buffer[row + 1];
    for (unsigned int j = row_U_start; j < row_U_end; ++j)
    {
      unsigned int col = U_col_buffer[j];

      row_L_start = L_row_buffer[col];
      row_L_end   = L_row_buffer[col + 1];

      // compute \sum_{k=1}^{j-1} l_ik u_kj
      unsigned int index_L = row_L_start;
      unsigned int col_L = (index_L < row_L_end) ? L_col_buffer[index_L] : static_cast<unsigned int>(L.size1());
      NumericT sum = 0;
      for (unsigned int k = row_U_start; k < j; ++k)
      {
        unsigned int col_U = U_col_buffer[k];

        // find element in L
        while (col_L < col_U)
        {
          ++index_L;
          col_L = L_col_buffer[index_L];
        }

        if (col_U == col_L)
          sum += L_backup[index_L] * U_backup[k];
      }

      // update u_ij:
      U_elements[j] = aij_U_trans_ptr[j] - sum;
    }
  }

  delete[] L_backup;
  delete[] U_backup;
}


template<typename NumericT>
void ilu_form_neumann_matrix(compressed_matrix<NumericT> & R,
                             vector<NumericT> & diag_R)
{
  unsigned int *R_row_buffer = detail::extract_raw_pointer<unsigned int>(R.handle1());
  unsigned int *R_col_buffer = detail::extract_raw_pointer<unsigned int>(R.handle2());
  NumericT     *R_elements   = detail::extract_raw_pointer<NumericT>(R.handle());

  NumericT     *diag_R_ptr   = detail::extract_raw_pointer<NumericT>(diag_R.handle());

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for if (R.size1() > VIENNACL_OPENMP_ILU_MIN_SIZE)
#endif
  for (long row = 0; row < static_cast<long>(R.size1()); ++row)
  {
    unsigned int col_begin = R_row_buffer[row];
    unsigned int col_end   = R_row_buffer[row+1];

    // part 1: extract diagonal entry
    NumericT diag = 0;
    for (unsigned int j = col_begin; j < col_end; ++j)
    {
      unsigned int col = R_col_buffer[j];
      if (col == row)
      {
        diag = R_elements[j];
        R_elements[j] = 0; // (I - D^{-1}R)
        break;
      }
    }
    diag_R_ptr[row] = diag;

    assert((diag > 0 || diag < 0) && bool("Zero diagonal detected!"));

    // part2: scale
    for (unsigned int j = col_begin; j < col_end; ++j)
      R_elements[j] /= -diag;
  }

  //std::cout << "diag_R: " << diag_R << std::endl;
}

} //namespace host_based
} //namespace linalg
} //namespace viennacl


#endif
