#ifndef VIENNACL_LINALG_DETAIL_ILU0_HPP_
#define VIENNACL_LINALG_DETAIL_ILU0_HPP_

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

/** @file viennacl/linalg/detail/ilu/ilu0.hpp
  @brief Implementations of incomplete factorization preconditioners with static nonzero pattern.

  Contributed by Evan Bollig.

  ILU0 (Incomplete LU with zero fill-in)
  - All preconditioner nonzeros exist at locations that were nonzero in the input matrix.
  - The number of nonzeros in the output preconditioner are exactly the same number as the input matrix

 Evan Bollig 3/30/12

 Adapted from viennacl/linalg/detail/ilut.hpp

 Low-level reimplementation by Karl Rupp in Nov 2012, increasing performance substantially. Also added level-scheduling.

*/

#include <vector>
#include <cmath>
#include <iostream>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/detail/ilu/common.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/backend/memory.hpp"

#include "viennacl/linalg/host_based/common.hpp"

#include <map>

namespace viennacl
{
namespace linalg
{

/** @brief A tag for incomplete LU factorization with static pattern (ILU0)
*/
class ilu0_tag
{
public:
  ilu0_tag(bool with_level_scheduling = false) : use_level_scheduling_(with_level_scheduling) {}

  bool use_level_scheduling() const { return use_level_scheduling_; }
  void use_level_scheduling(bool b) { use_level_scheduling_ = b; }

private:
  bool use_level_scheduling_;
};


/** @brief Implementation of a ILU-preconditioner with static pattern. Optimized version for CSR matrices.
  *
  * refer to the Algorithm in Saad's book (1996 edition)
  *
  *  @param A       The sparse matrix matrix. The result is directly written to A.
  */
template<typename NumericT>
void precondition(viennacl::compressed_matrix<NumericT> & A, ilu0_tag const & /* tag */)
{
  assert( (A.handle1().get_active_handle_id() == viennacl::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
  assert( (A.handle2().get_active_handle_id() == viennacl::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );
  assert( (A.handle().get_active_handle_id()  == viennacl::MAIN_MEMORY) && bool("System matrix must reside in main memory for ILU0") );

  NumericT           * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericT>(A.handle());
  unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle1());
  unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(A.handle2());

  // Note: Line numbers in the following refer to the algorithm in Saad's book

  for (vcl_size_t i=1; i<A.size1(); ++i)  // Line 1
  {
    unsigned int row_i_begin = row_buffer[i];
    unsigned int row_i_end   = row_buffer[i+1];
    for (unsigned int buf_index_k = row_i_begin; buf_index_k < row_i_end; ++buf_index_k) //Note: We do not assume that the column indices within a row are sorted
    {
      unsigned int k = col_buffer[buf_index_k];
      if (k >= i)
        continue; //Note: We do not assume that the column indices within a row are sorted

      unsigned int row_k_begin = row_buffer[k];
      unsigned int row_k_end   = row_buffer[k+1];

      // get a_kk:
      NumericT a_kk = 0;
      for (unsigned int buf_index_akk = row_k_begin; buf_index_akk < row_k_end; ++buf_index_akk)
      {
        if (col_buffer[buf_index_akk] == k)
        {
          a_kk = elements[buf_index_akk];
          break;
        }
      }

      NumericT & a_ik = elements[buf_index_k];
      a_ik /= a_kk;                                 //Line 3

      for (unsigned int buf_index_j = row_i_begin; buf_index_j < row_i_end; ++buf_index_j) //Note: We do not assume that the column indices within a row are sorted
      {
        unsigned int j = col_buffer[buf_index_j];
        if (j <= k)
          continue;

        // determine a_kj:
        NumericT a_kj = 0;
        for (unsigned int buf_index_akj = row_k_begin; buf_index_akj < row_k_end; ++buf_index_akj)
        {
          if (col_buffer[buf_index_akj] == j)
          {
            a_kj = elements[buf_index_akj];
            break;
          }
        }

        //a_ij -= a_ik * a_kj
        elements[buf_index_j] -= a_ik * a_kj;  //Line 5
      }
    }
  }

}


/** @brief ILU0 preconditioner class, can be supplied to solve()-routines
*/
template<typename MatrixT>
class ilu0_precond
{
  typedef typename MatrixT::value_type      NumericType;

public:
  ilu0_precond(MatrixT const & mat, ilu0_tag const & tag) : tag_(tag), LU_()
  {
    //initialize preconditioner:
    //std::cout << "Start CPU precond" << std::endl;
    init(mat);
    //std::cout << "End CPU precond" << std::endl;
  }

  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    unsigned int const * row_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(LU_.handle1());
    unsigned int const * col_buffer = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(LU_.handle2());
    NumericType  const * elements   = viennacl::linalg::host_based::detail::extract_raw_pointer<NumericType>(LU_.handle());

    viennacl::linalg::host_based::detail::csr_inplace_solve<NumericType>(row_buffer, col_buffer, elements, vec, LU_.size2(), unit_lower_tag());
    viennacl::linalg::host_based::detail::csr_inplace_solve<NumericType>(row_buffer, col_buffer, elements, vec, LU_.size2(), upper_tag());
  }

private:
  void init(MatrixT const & mat)
  {
    viennacl::context host_context(viennacl::MAIN_MEMORY);
    viennacl::switch_memory_context(LU_, host_context);

    viennacl::copy(mat, LU_);
    viennacl::linalg::precondition(LU_, tag_);
  }

  ilu0_tag                                   tag_;
  viennacl::compressed_matrix<NumericType>   LU_;
};


/** @brief ILU0 preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV>
class ilu0_precond< viennacl::compressed_matrix<NumericT, AlignmentV> >
{
  typedef viennacl::compressed_matrix<NumericT, AlignmentV>   MatrixType;

public:
  ilu0_precond(MatrixType const & mat, ilu0_tag const & tag)
    : tag_(tag),
      LU_(mat.size1(), mat.size2(), viennacl::traits::context(mat))
  {
    //initialize preconditioner:
    //std::cout << "Start GPU precond" << std::endl;
    init(mat);
    //std::cout << "End GPU precond" << std::endl;
  }

  void apply(viennacl::vector<NumericT> & vec) const
  {
    viennacl::context host_context(viennacl::MAIN_MEMORY);
    if (vec.handle().get_active_handle_id() != viennacl::MAIN_MEMORY)
    {
      if (tag_.use_level_scheduling())
      {
        //std::cout << "Using multifrontal on GPU..." << std::endl;
        detail::level_scheduling_substitute(vec,
                                            multifrontal_L_row_index_arrays_,
                                            multifrontal_L_row_buffers_,
                                            multifrontal_L_col_buffers_,
                                            multifrontal_L_element_buffers_,
                                            multifrontal_L_row_elimination_num_list_);

        vec = viennacl::linalg::element_div(vec, multifrontal_U_diagonal_);

        detail::level_scheduling_substitute(vec,
                                            multifrontal_U_row_index_arrays_,
                                            multifrontal_U_row_buffers_,
                                            multifrontal_U_col_buffers_,
                                            multifrontal_U_element_buffers_,
                                            multifrontal_U_row_elimination_num_list_);
      }
      else
      {
        viennacl::context old_context = viennacl::traits::context(vec);
        viennacl::switch_memory_context(vec, host_context);
        viennacl::linalg::inplace_solve(LU_, vec, unit_lower_tag());
        viennacl::linalg::inplace_solve(LU_, vec, upper_tag());
        viennacl::switch_memory_context(vec, old_context);
      }
    }
    else //apply ILU0 directly on CPU
    {
      if (tag_.use_level_scheduling())
      {
        //std::cout << "Using multifrontal..." << std::endl;
        detail::level_scheduling_substitute(vec,
                                            multifrontal_L_row_index_arrays_,
                                            multifrontal_L_row_buffers_,
                                            multifrontal_L_col_buffers_,
                                            multifrontal_L_element_buffers_,
                                            multifrontal_L_row_elimination_num_list_);

        vec = viennacl::linalg::element_div(vec, multifrontal_U_diagonal_);

        detail::level_scheduling_substitute(vec,
                                            multifrontal_U_row_index_arrays_,
                                            multifrontal_U_row_buffers_,
                                            multifrontal_U_col_buffers_,
                                            multifrontal_U_element_buffers_,
                                            multifrontal_U_row_elimination_num_list_);
      }
      else
      {
        viennacl::linalg::inplace_solve(LU_, vec, unit_lower_tag());
        viennacl::linalg::inplace_solve(LU_, vec, upper_tag());
      }
    }
  }

  vcl_size_t levels() const { return multifrontal_L_row_index_arrays_.size(); }

private:
  void init(MatrixType const & mat)
  {
    viennacl::context host_context(viennacl::MAIN_MEMORY);
    viennacl::switch_memory_context(LU_, host_context);
    LU_ = mat;
    viennacl::linalg::precondition(LU_, tag_);

    if (!tag_.use_level_scheduling())
      return;

    // multifrontal part:
    viennacl::switch_memory_context(multifrontal_U_diagonal_, host_context);
    multifrontal_U_diagonal_.resize(LU_.size1(), false);
    host_based::detail::row_info(LU_, multifrontal_U_diagonal_, viennacl::linalg::detail::SPARSE_ROW_DIAGONAL);

    detail::level_scheduling_setup_L(LU_,
                                     multifrontal_U_diagonal_, //dummy
                                     multifrontal_L_row_index_arrays_,
                                     multifrontal_L_row_buffers_,
                                     multifrontal_L_col_buffers_,
                                     multifrontal_L_element_buffers_,
                                     multifrontal_L_row_elimination_num_list_);


    detail::level_scheduling_setup_U(LU_,
                                     multifrontal_U_diagonal_,
                                     multifrontal_U_row_index_arrays_,
                                     multifrontal_U_row_buffers_,
                                     multifrontal_U_col_buffers_,
                                     multifrontal_U_element_buffers_,
                                     multifrontal_U_row_elimination_num_list_);

    //
    // Bring to device if necessary:
    //

    // L:
    for (typename std::list< viennacl::backend::mem_handle >::iterator it  = multifrontal_L_row_index_arrays_.begin();
                                                                       it != multifrontal_L_row_index_arrays_.end();
                                                                     ++it)
      viennacl::backend::switch_memory_context<unsigned int>(*it, viennacl::traits::context(mat));

    for (typename std::list< viennacl::backend::mem_handle >::iterator it  = multifrontal_L_row_buffers_.begin();
                                                                       it != multifrontal_L_row_buffers_.end();
                                                                     ++it)
      viennacl::backend::switch_memory_context<unsigned int>(*it, viennacl::traits::context(mat));

    for (typename std::list< viennacl::backend::mem_handle >::iterator it  = multifrontal_L_col_buffers_.begin();
                                                                       it != multifrontal_L_col_buffers_.end();
                                                                     ++it)
      viennacl::backend::switch_memory_context<unsigned int>(*it, viennacl::traits::context(mat));

    for (typename std::list< viennacl::backend::mem_handle >::iterator it  = multifrontal_L_element_buffers_.begin();
                                                                       it != multifrontal_L_element_buffers_.end();
                                                                     ++it)
      viennacl::backend::switch_memory_context<NumericT>(*it, viennacl::traits::context(mat));


    // U:

    viennacl::switch_memory_context(multifrontal_U_diagonal_, viennacl::traits::context(mat));

    for (typename std::list< viennacl::backend::mem_handle >::iterator it  = multifrontal_U_row_index_arrays_.begin();
                                                                       it != multifrontal_U_row_index_arrays_.end();
                                                                     ++it)
      viennacl::backend::switch_memory_context<unsigned int>(*it, viennacl::traits::context(mat));

    for (typename std::list< viennacl::backend::mem_handle >::iterator it  = multifrontal_U_row_buffers_.begin();
                                                                       it != multifrontal_U_row_buffers_.end();
                                                                     ++it)
      viennacl::backend::switch_memory_context<unsigned int>(*it, viennacl::traits::context(mat));

    for (typename std::list< viennacl::backend::mem_handle >::iterator it  = multifrontal_U_col_buffers_.begin();
                                                                       it != multifrontal_U_col_buffers_.end();
                                                                     ++it)
      viennacl::backend::switch_memory_context<unsigned int>(*it, viennacl::traits::context(mat));

    for (typename std::list< viennacl::backend::mem_handle >::iterator it  = multifrontal_U_element_buffers_.begin();
                                                                       it != multifrontal_U_element_buffers_.end();
                                                                     ++it)
      viennacl::backend::switch_memory_context<NumericT>(*it, viennacl::traits::context(mat));

  }

  ilu0_tag tag_;
  viennacl::compressed_matrix<NumericT> LU_;

  std::list<viennacl::backend::mem_handle> multifrontal_L_row_index_arrays_;
  std::list<viennacl::backend::mem_handle> multifrontal_L_row_buffers_;
  std::list<viennacl::backend::mem_handle> multifrontal_L_col_buffers_;
  std::list<viennacl::backend::mem_handle> multifrontal_L_element_buffers_;
  std::list<vcl_size_t>                    multifrontal_L_row_elimination_num_list_;

  viennacl::vector<NumericT> multifrontal_U_diagonal_;
  std::list<viennacl::backend::mem_handle> multifrontal_U_row_index_arrays_;
  std::list<viennacl::backend::mem_handle> multifrontal_U_row_buffers_;
  std::list<viennacl::backend::mem_handle> multifrontal_U_col_buffers_;
  std::list<viennacl::backend::mem_handle> multifrontal_U_element_buffers_;
  std::list<vcl_size_t>                    multifrontal_U_row_elimination_num_list_;

};

} // namespace linalg
} // namespace viennacl


#endif



