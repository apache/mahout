#ifndef VIENNACL_LINALG_DETAIL_BLOCK_ILU_HPP_
#define VIENNACL_LINALG_DETAIL_BLOCK_ILU_HPP_

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

/** @file viennacl/linalg/detail/ilu/block_ilu.hpp
    @brief Implementations of incomplete block factorization preconditioners
*/

#include <vector>
#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/detail/ilu/common.hpp"
#include "viennacl/linalg/detail/ilu/ilu0.hpp"
#include "viennacl/linalg/detail/ilu/ilut.hpp"

#include <map>

namespace viennacl
{
namespace linalg
{
namespace detail
{
  /** @brief Helper range class for representing a subvector of a larger buffer. */
  template<typename VectorT, typename NumericT, typename SizeT = vcl_size_t>
  class ilu_vector_range
  {
  public:
    ilu_vector_range(VectorT & v,
                     SizeT start_index,
                     SizeT vec_size
                    ) : vec_(v), start_(start_index), size_(vec_size) {}

    NumericT & operator()(SizeT index)
    {
      assert(index < size_ && bool("Index out of bounds!"));
      return vec_[start_ + index];
    }

    NumericT & operator[](SizeT index)
    {
      assert(index < size_ && bool("Index out of bounds!"));
      return vec_[start_ + index];
    }

    SizeT size() const { return size_; }

  private:
    VectorT & vec_;
    SizeT start_;
    SizeT size_;
  };

  /** @brief Extracts a diagonal block from a larger system matrix
    *
    * @param A                   The full matrix
    * @param diagonal_block_A    The output matrix, to which the extracted block is written to
    * @param start_index         First row- and column-index of the block
    * @param stop_index          First row- and column-index beyond the block
    */
  template<typename NumericT>
  void extract_block_matrix(viennacl::compressed_matrix<NumericT> const & A,
                            viennacl::compressed_matrix<NumericT>       & diagonal_block_A,
                            vcl_size_t start_index,
                            vcl_size_t stop_index
                            )
  {
    assert( (A.handle1().get_active_handle_id() == viennacl::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
    assert( (A.handle2().get_active_handle_id() == viennacl::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
    assert( (A.handle().get_active_handle_id() == viennacl::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );

    NumericT     const * A_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
    unsigned int const * A_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
    unsigned int const * A_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

    NumericT     * output_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(diagonal_block_A.handle());
    unsigned int * output_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(diagonal_block_A.handle1());
    unsigned int * output_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(diagonal_block_A.handle2());

    vcl_size_t output_counter = 0;
    for (vcl_size_t row = start_index; row < stop_index; ++row)
    {
      unsigned int buffer_col_start = A_row_buffer[row];
      unsigned int buffer_col_end   = A_row_buffer[row+1];

      output_row_buffer[row - start_index] = static_cast<unsigned int>(output_counter);

      for (unsigned int buf_index = buffer_col_start; buf_index < buffer_col_end; ++buf_index)
      {
        unsigned int col = A_col_buffer[buf_index];
        if (col < start_index)
          continue;

        if (col >= static_cast<unsigned int>(stop_index))
          continue;

        output_col_buffer[output_counter] = static_cast<unsigned int>(col - start_index);
        output_elements[output_counter] = A_elements[buf_index];
        ++output_counter;
      }
      output_row_buffer[row - start_index + 1] = static_cast<unsigned int>(output_counter);
    }
  }

} // namespace detail



/** @brief A block ILU preconditioner class, can be supplied to solve()-routines
 *
 * @tparam MatrixType   Type of the system matrix
 * @tparam ILUTag       Type of the tag identifiying the ILU preconditioner to be used on each block.
*/
template<typename MatrixT, typename ILUTag>
class block_ilu_precond
{
typedef typename MatrixT::value_type      ScalarType;

public:
  typedef std::vector<std::pair<vcl_size_t, vcl_size_t> >    index_vector_type;   //the pair refers to index range [a, b) of each block


  block_ilu_precond(MatrixT const & mat,
                    ILUTag const & tag,
                    vcl_size_t num_blocks = 8
                   ) : tag_(tag), L_blocks(num_blocks), U_blocks(num_blocks)
  {
    // Set up vector of block indices:
    block_indices_.resize(num_blocks);
    for (vcl_size_t i=0; i<num_blocks; ++i)
    {
      vcl_size_t start_index = (   i  * mat.size1()) / num_blocks;
      vcl_size_t stop_index  = ((i+1) * mat.size1()) / num_blocks;

      block_indices_[i] = std::pair<vcl_size_t, vcl_size_t>(start_index, stop_index);
    }

    //initialize preconditioner:
    //std::cout << "Start CPU precond" << std::endl;
    init(mat);
    //std::cout << "End CPU precond" << std::endl;
  }

  block_ilu_precond(MatrixT const & mat,
                    ILUTag const & tag,
                    index_vector_type const & block_boundaries
                   ) : tag_(tag), block_indices_(block_boundaries), L_blocks(block_boundaries.size()), U_blocks(block_boundaries.size())
  {
    //initialize preconditioner:
    //std::cout << "Start CPU precond" << std::endl;
    init(mat);
    //std::cout << "End CPU precond" << std::endl;
  }


  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    for (vcl_size_t i=0; i<block_indices_.size(); ++i)
      apply_dispatch(vec, i, ILUTag());
  }

private:
  void init(MatrixT const & A)
  {
    viennacl::context host_context(viennacl::MAIN_MEMORY);
    viennacl::compressed_matrix<ScalarType> mat(host_context);

    viennacl::copy(A, mat);

    unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(mat.handle1());

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
    for (long i2=0; i2<static_cast<long>(block_indices_.size()); ++i2)
    {
      vcl_size_t i = static_cast<vcl_size_t>(i2);
      // Step 1: Extract blocks
      vcl_size_t block_size = block_indices_[i].second - block_indices_[i].first;
      vcl_size_t block_nnz  = row_buffer[block_indices_[i].second] - row_buffer[block_indices_[i].first];
      viennacl::compressed_matrix<ScalarType> mat_block(block_size, block_size, block_nnz, host_context);

      detail::extract_block_matrix(mat, mat_block, block_indices_[i].first, block_indices_[i].second);

      // Step 2: Precondition blocks:
      viennacl::switch_memory_context(L_blocks[i], host_context);
      viennacl::switch_memory_context(U_blocks[i], host_context);
      init_dispatch(mat_block, L_blocks[i], U_blocks[i], tag_);
    }

  }

  void init_dispatch(viennacl::compressed_matrix<ScalarType> const & mat_block,
                     viennacl::compressed_matrix<ScalarType> & L,
                     viennacl::compressed_matrix<ScalarType> & U,
                     viennacl::linalg::ilu0_tag)
  {
    (void)U;
    L = mat_block;
    viennacl::linalg::precondition(L, tag_);
  }

  void init_dispatch(viennacl::compressed_matrix<ScalarType> const & mat_block,
                     viennacl::compressed_matrix<ScalarType> & L,
                     viennacl::compressed_matrix<ScalarType> & U,
                     viennacl::linalg::ilut_tag)
  {
    L.resize(mat_block.size1(), mat_block.size2());
    U.resize(mat_block.size1(), mat_block.size2());
    viennacl::linalg::precondition(mat_block, L, U, tag_);
  }

  template<typename VectorT>
  void apply_dispatch(VectorT & vec, vcl_size_t i, viennacl::linalg::ilu0_tag) const
  {
    detail::ilu_vector_range<VectorT, ScalarType> vec_range(vec, block_indices_[i].first, L_blocks[i].size2());

    unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L_blocks[i].handle1());
    unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L_blocks[i].handle2());
    ScalarType   const * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<ScalarType>(L_blocks[i].handle());

    viennacl::linalg::host_based::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, L_blocks[i].size2(), unit_lower_tag());
    viennacl::linalg::host_based::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, L_blocks[i].size2(), upper_tag());
  }

  template<typename VectorT>
  void apply_dispatch(VectorT & vec, vcl_size_t i, viennacl::linalg::ilut_tag) const
  {
    detail::ilu_vector_range<VectorT, ScalarType> vec_range(vec, block_indices_[i].first, L_blocks[i].size2());

    {
      unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L_blocks[i].handle1());
      unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L_blocks[i].handle2());
      ScalarType   const * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<ScalarType>(L_blocks[i].handle());

      viennacl::linalg::host_based::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, L_blocks[i].size2(), unit_lower_tag());
    }

    {
      unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(U_blocks[i].handle1());
      unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(U_blocks[i].handle2());
      ScalarType   const * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<ScalarType>(U_blocks[i].handle());

      viennacl::linalg::host_based::detail::csr_inplace_solve<ScalarType>(row_buffer, col_buffer, elements, vec_range, U_blocks[i].size2(), upper_tag());
    }
  }

  ILUTag tag_;
  index_vector_type block_indices_;
  std::vector< viennacl::compressed_matrix<ScalarType> > L_blocks;
  std::vector< viennacl::compressed_matrix<ScalarType> > U_blocks;
};





/** @brief ILUT preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV, typename ILUTagT>
class block_ilu_precond< compressed_matrix<NumericT, AlignmentV>, ILUTagT>
{
  typedef compressed_matrix<NumericT, AlignmentV>        MatrixType;

public:
  typedef std::vector<std::pair<vcl_size_t, vcl_size_t> >    index_vector_type;   //the pair refers to index range [a, b) of each block


  block_ilu_precond(MatrixType const & mat,
                    ILUTagT const & tag,
                    vcl_size_t num_blocks = 8
                   ) : tag_(tag),
                       block_indices_(num_blocks),
                       gpu_block_indices_(),
                       gpu_L_trans_(0, 0, viennacl::context(viennacl::MAIN_MEMORY)),
                       gpu_U_trans_(0, 0, viennacl::context(viennacl::MAIN_MEMORY)),
                       gpu_D_(mat.size1(), viennacl::context(viennacl::MAIN_MEMORY)),
                       L_blocks_(num_blocks),
                       U_blocks_(num_blocks)
  {
    // Set up vector of block indices:
    block_indices_.resize(num_blocks);
    for (vcl_size_t i=0; i<num_blocks; ++i)
    {
      vcl_size_t start_index = (   i  * mat.size1()) / num_blocks;
      vcl_size_t stop_index  = ((i+1) * mat.size1()) / num_blocks;

      block_indices_[i] = std::pair<vcl_size_t, vcl_size_t>(start_index, stop_index);
    }

    //initialize preconditioner:
    //std::cout << "Start CPU precond" << std::endl;
    init(mat);
    //std::cout << "End CPU precond" << std::endl;
  }

  block_ilu_precond(MatrixType const & mat,
                    ILUTagT const & tag,
                    index_vector_type const & block_boundaries
                   ) : tag_(tag),
                       block_indices_(block_boundaries),
                       gpu_block_indices_(),
                       gpu_L_trans_(0, 0, viennacl::context(viennacl::MAIN_MEMORY)),
                       gpu_U_trans_(0, 0, viennacl::context(viennacl::MAIN_MEMORY)),
                       gpu_D_(mat.size1(), viennacl::context(viennacl::MAIN_MEMORY)),
                       L_blocks_(block_boundaries.size()),
                       U_blocks_(block_boundaries.size())
  {
    //initialize preconditioner:
    //std::cout << "Start CPU precond" << std::endl;
    init(mat);
    //std::cout << "End CPU precond" << std::endl;
  }


  void apply(vector<NumericT> & vec) const
  {
    viennacl::linalg::detail::block_inplace_solve(trans(gpu_L_trans_), gpu_block_indices_, block_indices_.size(), gpu_D_,
                                                  vec,
                                                  viennacl::linalg::unit_lower_tag());

    viennacl::linalg::detail::block_inplace_solve(trans(gpu_U_trans_), gpu_block_indices_, block_indices_.size(), gpu_D_,
                                                  vec,
                                                  viennacl::linalg::upper_tag());

    //apply_cpu(vec);
  }


private:

  void init(MatrixType const & A)
  {
    viennacl::context host_context(viennacl::MAIN_MEMORY);
    viennacl::compressed_matrix<NumericT> mat(host_context);

    mat = A;

    unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(mat.handle1());

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
    for (long i=0; i<static_cast<long>(block_indices_.size()); ++i)
    {
      // Step 1: Extract blocks
      vcl_size_t block_size = block_indices_[static_cast<vcl_size_t>(i)].second - block_indices_[static_cast<vcl_size_t>(i)].first;
      vcl_size_t block_nnz  = row_buffer[block_indices_[static_cast<vcl_size_t>(i)].second] - row_buffer[block_indices_[static_cast<vcl_size_t>(i)].first];
      viennacl::compressed_matrix<NumericT> mat_block(block_size, block_size, block_nnz, host_context);

      detail::extract_block_matrix(mat, mat_block, block_indices_[static_cast<vcl_size_t>(i)].first, block_indices_[static_cast<vcl_size_t>(i)].second);

      // Step 2: Precondition blocks:
      viennacl::switch_memory_context(L_blocks_[static_cast<vcl_size_t>(i)], host_context);
      viennacl::switch_memory_context(U_blocks_[static_cast<vcl_size_t>(i)], host_context);
      init_dispatch(mat_block, L_blocks_[static_cast<vcl_size_t>(i)], U_blocks_[static_cast<vcl_size_t>(i)], tag_);
    }

    /*
     * copy resulting preconditioner back to GPU:
     */
    viennacl::backend::typesafe_host_array<unsigned int> block_indices_uint(gpu_block_indices_, 2 * block_indices_.size());
    for (vcl_size_t i=0; i<block_indices_.size(); ++i)
    {
      block_indices_uint.set(2*i,     block_indices_[i].first);
      block_indices_uint.set(2*i + 1, block_indices_[i].second);
    }

    viennacl::backend::memory_create(gpu_block_indices_, block_indices_uint.raw_size(), viennacl::traits::context(A), block_indices_uint.get());

    blocks_to_device(A);

  }

  // Copy computed preconditioned blocks to OpenCL device
  void blocks_to_device(MatrixType const & A)
  {
    gpu_L_trans_.resize(A.size1(), A.size2());
    gpu_U_trans_.resize(A.size1(), A.size2());
    gpu_D_.resize(A.size1());

    unsigned int * L_trans_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(gpu_L_trans_.handle1());
    unsigned int * U_trans_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(gpu_U_trans_.handle1());

    //
    // Count elements per row
    //
#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
    for (long block_index2 = 0; block_index2 < static_cast<long>(L_blocks_.size()); ++block_index2)
    {
      vcl_size_t block_index = vcl_size_t(block_index2);

      unsigned int block_start = static_cast<unsigned int>(block_indices_[block_index].first);
      unsigned int block_stop  = static_cast<unsigned int>(block_indices_[block_index].second);

      unsigned int const * L_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L_blocks_[block_index].handle1());
      unsigned int const * L_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L_blocks_[block_index].handle2());

      // zero row array of L:
      std::fill(L_trans_row_buffer + block_start,
                L_trans_row_buffer + block_stop,
                static_cast<unsigned int>(0));

      // count number of elements per row:
      for (vcl_size_t row = 0; row < L_blocks_[block_index].size1(); ++row)
      {
        unsigned int col_start = L_row_buffer[row];
        unsigned int col_end   = L_row_buffer[row+1];

        for (unsigned int j = col_start; j < col_end; ++j)
        {
          unsigned int col = L_col_buffer[j];
          if (col < static_cast<unsigned int>(row))
            L_trans_row_buffer[col + block_start] += 1;
        }
      }

      ////// same for U

      unsigned int const * U_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(U_blocks_[block_index].handle1());
      unsigned int const * U_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(U_blocks_[block_index].handle2());

      // zero row array of U:
      std::fill(U_trans_row_buffer + block_start,
                U_trans_row_buffer + block_stop,
                static_cast<unsigned int>(0));

      // count number of elements per row:
      for (vcl_size_t row = 0; row < U_blocks_[block_index].size1(); ++row)
      {
        unsigned int col_start = U_row_buffer[row];
        unsigned int col_end   = U_row_buffer[row+1];

        for (unsigned int j = col_start; j < col_end; ++j)
        {
          unsigned int col = U_col_buffer[j];
          if (col > row)
            U_trans_row_buffer[col + block_start] += 1;
        }
      }
    }


    //
    // Exclusive scan on row buffer (feel free to add parallelization here)
    //
    unsigned int current_value = 0;
    for (vcl_size_t i=0; i<gpu_L_trans_.size1(); ++i)
    {
      unsigned int tmp = L_trans_row_buffer[i];
      L_trans_row_buffer[i] = current_value;
      current_value += tmp;
    }
    gpu_L_trans_.reserve(current_value);

    current_value = 0;
    for (vcl_size_t i=0; i<gpu_U_trans_.size1(); ++i)
    {
      unsigned int tmp = U_trans_row_buffer[i];
      U_trans_row_buffer[i] = current_value;
      current_value += tmp;
    }
    gpu_U_trans_.reserve(current_value);


    //
    // Fill with data
    //
    unsigned int       * L_trans_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(gpu_L_trans_.handle2());
    NumericT           * L_trans_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(gpu_L_trans_.handle());

    unsigned int       * U_trans_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(gpu_U_trans_.handle2());
    NumericT           * U_trans_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(gpu_U_trans_.handle());

    NumericT           * D_elements         = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(gpu_D_.handle());

    std::vector<unsigned int> offset_L(gpu_L_trans_.size1());
    std::vector<unsigned int> offset_U(gpu_U_trans_.size1());

#ifdef VIENNACL_WITH_OPENMP
    #pragma omp parallel for
#endif
    for (long block_index2 = 0; block_index2 < static_cast<long>(L_blocks_.size()); ++block_index2)
    {
      vcl_size_t   block_index = vcl_size_t(block_index2);
      unsigned int block_start = static_cast<unsigned int>(block_indices_[block_index].first);

      unsigned int const * L_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L_blocks_[block_index].handle1());
      unsigned int const * L_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(L_blocks_[block_index].handle2());
      NumericT     const * L_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT    >(L_blocks_[block_index].handle());


      // write L_trans:
      for (vcl_size_t row = 0; row < L_blocks_[block_index].size1(); ++row)
      {
        unsigned int col_start = L_row_buffer[row];
        unsigned int col_end   = L_row_buffer[row+1];

        for (unsigned int j = col_start; j < col_end; ++j)
        {
          unsigned int col = L_col_buffer[j];
          if (col < row)
          {
            unsigned int row_trans = col + block_start;
            unsigned int k = L_trans_row_buffer[row_trans] + offset_L[row_trans];
            offset_L[row_trans] += 1;

            L_trans_col_buffer[k] = static_cast<unsigned int>(row) + block_start;
            L_trans_elements[k]   = L_elements[j];
          }
        }
      }

      unsigned int const * U_row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(U_blocks_[block_index].handle1());
      unsigned int const * U_col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(U_blocks_[block_index].handle2());
      NumericT     const * U_elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT    >(U_blocks_[block_index].handle());

      // write U_trans and D:
      for (vcl_size_t row = 0; row < U_blocks_[block_index].size1(); ++row)
      {
        unsigned int col_start = U_row_buffer[row];
        unsigned int col_end   = U_row_buffer[row+1];

        for (unsigned int j = col_start; j < col_end; ++j)
        {
          unsigned int row_trans = U_col_buffer[j] + block_start;
          unsigned int k = U_trans_row_buffer[row_trans] + offset_U[row_trans];

          if (row_trans == row + block_start) // entry for D
          {
            D_elements[row_trans] = U_elements[j];
          }
          else if (row_trans > row + block_start) //entry for U
          {
            offset_U[row_trans] += 1;

            U_trans_col_buffer[k] = static_cast<unsigned int>(row) + block_start;
            U_trans_elements[k]   = U_elements[j];
          }
        }
      }

    }

    //
    // Send to destination device:
    //
    viennacl::switch_memory_context(gpu_L_trans_, viennacl::traits::context(A));
    viennacl::switch_memory_context(gpu_U_trans_, viennacl::traits::context(A));
    viennacl::switch_memory_context(gpu_D_,       viennacl::traits::context(A));
  }

  void init_dispatch(viennacl::compressed_matrix<NumericT> const & mat_block,
                     viennacl::compressed_matrix<NumericT> & L,
                     viennacl::compressed_matrix<NumericT> & U,
                     viennacl::linalg::ilu0_tag)
  {
    L = mat_block;
    viennacl::linalg::precondition(L, tag_);
    U = L; // fairly poor workaround...
  }

  void init_dispatch(viennacl::compressed_matrix<NumericT> const & mat_block,
                     viennacl::compressed_matrix<NumericT> & L,
                     viennacl::compressed_matrix<NumericT> & U,
                     viennacl::linalg::ilut_tag)
  {
    L.resize(mat_block.size1(), mat_block.size2());
    U.resize(mat_block.size1(), mat_block.size2());
    viennacl::linalg::precondition(mat_block, L, U, tag_);
  }


  ILUTagT                               tag_;
  index_vector_type                     block_indices_;
  viennacl::backend::mem_handle         gpu_block_indices_;
  viennacl::compressed_matrix<NumericT> gpu_L_trans_;
  viennacl::compressed_matrix<NumericT> gpu_U_trans_;
  viennacl::vector<NumericT>            gpu_D_;

  std::vector<MatrixType> L_blocks_;
  std::vector<MatrixType> U_blocks_;
};


}
}




#endif



