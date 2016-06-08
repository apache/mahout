#ifndef VIENNACL_SLICED_ELL_MATRIX_HPP_
#define VIENNACL_SLICED_ELL_MATRIX_HPP_

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

/** @file viennacl/sliced_ell_matrix.hpp
    @brief Implementation of the sliced_ell_matrix class

    Based on the SELL-C-sigma format provided by Kreutzer et al., 2014
*/


#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"

#include "viennacl/tools/tools.hpp"

#include "viennacl/linalg/sparse_matrix_operations.hpp"

namespace viennacl
{
/** @brief Sparse matrix class using the sliced ELLPACK with parameters C, \f$ \sigma \f$
  *
  * Based on the SELL-C-sigma format provided by Kreutzer et al., 2014
  * Can be seen as a block-wise ELLPACK format, where C rows are accumulated into the same block
  * for which a column-wise storage is used. Enables fully-coalesced reads from global memory.
  *
  * Note: Currently \f$ \sigma \f$ is fixed to 1, since the sorting of rows
  */
template<typename ScalarT, typename IndexT /* see forwards.h = unsigned int */>
class sliced_ell_matrix
{
public:
  typedef viennacl::backend::mem_handle                                                           handle_type;
  typedef scalar<typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<ScalarT>::ResultType>   value_type;
  typedef vcl_size_t                                                                              size_type;

  explicit sliced_ell_matrix() : rows_(0), cols_(0), rows_per_block_(0) {}

  /** @brief Standard constructor for setting the row and column sizes as well as the block size.
    *
    * Supported values for num_rows_per_block_ are 32, 64, 128, 256. Other values may work, but are unlikely to yield good performance.
    **/
  sliced_ell_matrix(size_type num_rows,
                    size_type num_cols,
                    size_type num_rows_per_block_ = 0)
    : rows_(num_rows),
      cols_(num_cols),
      rows_per_block_(num_rows_per_block_) {}

  explicit sliced_ell_matrix(viennacl::context ctx) : rows_(0), cols_(0), rows_per_block_(0)
  {
    columns_per_block_.switch_active_handle_id(ctx.memory_type());
    column_indices_.switch_active_handle_id(ctx.memory_type());
    block_start_.switch_active_handle_id(ctx.memory_type());
    elements_.switch_active_handle_id(ctx.memory_type());

#ifdef VIENNACL_WITH_OPENCL
    if (ctx.memory_type() == OPENCL_MEMORY)
    {
      columns_per_block_.opencl_handle().context(ctx.opencl_context());
      column_indices_.opencl_handle().context(ctx.opencl_context());
      block_start_.opencl_handle().context(ctx.opencl_context());
      elements_.opencl_handle().context(ctx.opencl_context());
    }
#endif
  }

  /** @brief Resets all entries in the matrix back to zero without changing the matrix size. Resets the sparsity pattern. */
  void clear()
  {
    viennacl::backend::typesafe_host_array<IndexT> host_columns_per_block_buffer(columns_per_block_, rows_ / rows_per_block_ + 1);
    viennacl::backend::typesafe_host_array<IndexT> host_column_buffer(column_indices_, internal_size1());
    viennacl::backend::typesafe_host_array<IndexT> host_block_start_buffer(block_start_, (rows_ - 1) / rows_per_block_ + 1);
    std::vector<ScalarT> host_elements(1);

    viennacl::backend::memory_create(columns_per_block_, host_columns_per_block_buffer.element_size() * (rows_ / rows_per_block_ + 1), viennacl::traits::context(columns_per_block_), host_columns_per_block_buffer.get());
    viennacl::backend::memory_create(column_indices_,    host_column_buffer.element_size() * internal_size1(),                         viennacl::traits::context(column_indices_),    host_column_buffer.get());
    viennacl::backend::memory_create(block_start_,       host_block_start_buffer.element_size() * ((rows_ - 1) / rows_per_block_ + 1), viennacl::traits::context(block_start_),       host_block_start_buffer.get());
    viennacl::backend::memory_create(elements_,          sizeof(ScalarT) * 1,                                                          viennacl::traits::context(elements_),          &(host_elements[0]));
  }

  vcl_size_t internal_size1() const { return viennacl::tools::align_to_multiple<vcl_size_t>(rows_, rows_per_block_); }
  vcl_size_t internal_size2() const { return cols_; }

  vcl_size_t size1() const { return rows_; }
  vcl_size_t size2() const { return cols_; }

  vcl_size_t rows_per_block() const { return rows_per_block_; }

  //vcl_size_t nnz() const { return rows_ * maxnnz_; }
  //vcl_size_t internal_nnz() const { return internal_size1() * internal_maxnnz(); }

  handle_type & handle1()       { return columns_per_block_; }
  const handle_type & handle1() const { return columns_per_block_; }

  handle_type & handle2()       { return column_indices_; }
  const handle_type & handle2() const { return column_indices_; }

  handle_type & handle3()       { return block_start_; }
  const handle_type & handle3() const { return block_start_; }

  handle_type & handle()       { return elements_; }
  const handle_type & handle() const { return elements_; }

#if defined(_MSC_VER) && _MSC_VER < 1500          //Visual Studio 2005 needs special treatment
  template<typename CPUMatrixT>
  friend void copy(CPUMatrixT const & cpu_matrix, sliced_ell_matrix & gpu_matrix );
#else
  template<typename CPUMatrixT, typename ScalarT2, typename IndexT2>
  friend void copy(CPUMatrixT const & cpu_matrix, sliced_ell_matrix<ScalarT2, IndexT2> & gpu_matrix );
#endif

private:
  vcl_size_t rows_;
  vcl_size_t cols_;
  vcl_size_t rows_per_block_; //parameter C in the paper by Kreutzer et al.

  handle_type columns_per_block_;
  handle_type column_indices_;
  handle_type block_start_;
  handle_type elements_;
};

template<typename CPUMatrixT, typename ScalarT, typename IndexT>
void copy(CPUMatrixT const & cpu_matrix, sliced_ell_matrix<ScalarT, IndexT> & gpu_matrix )
{
  assert( (gpu_matrix.size1() == 0 || viennacl::traits::size1(cpu_matrix) == gpu_matrix.size1()) && bool("Size mismatch") );
  assert( (gpu_matrix.size2() == 0 || viennacl::traits::size2(cpu_matrix) == gpu_matrix.size2()) && bool("Size mismatch") );

  if (gpu_matrix.rows_per_block() == 0) // not yet initialized by user. Set default: 32 is perfect for NVIDIA GPUs and older AMD GPUs. Still okay for newer AMD GPUs.
    gpu_matrix.rows_per_block_ = 32;

  if (viennacl::traits::size1(cpu_matrix) > 0 && viennacl::traits::size2(cpu_matrix) > 0)
  {
    //determine max capacity for row
    IndexT columns_in_current_block = 0;
    vcl_size_t total_element_buffer_size = 0;
    viennacl::backend::typesafe_host_array<IndexT> columns_in_block_buffer(gpu_matrix.handle1(), (viennacl::traits::size1(cpu_matrix) - 1) / gpu_matrix.rows_per_block() + 1);
    for (typename CPUMatrixT::const_iterator1 row_it = cpu_matrix.begin1(); row_it != cpu_matrix.end1(); ++row_it)
    {
      vcl_size_t entries_in_row = 0;
      for (typename CPUMatrixT::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
        ++entries_in_row;

      columns_in_current_block = std::max(columns_in_current_block, static_cast<IndexT>(entries_in_row));

      // check for end of block
      if ( (row_it.index1() % gpu_matrix.rows_per_block() == gpu_matrix.rows_per_block() - 1)
           || row_it.index1() == viennacl::traits::size1(cpu_matrix) - 1)
      {
        total_element_buffer_size += columns_in_current_block * gpu_matrix.rows_per_block();
        columns_in_block_buffer.set(row_it.index1() / gpu_matrix.rows_per_block(), columns_in_current_block);
        columns_in_current_block = 0;
      }
    }

    //setup GPU matrix
    gpu_matrix.rows_ = cpu_matrix.size1();
    gpu_matrix.cols_ = cpu_matrix.size2();

    viennacl::backend::typesafe_host_array<IndexT> coords(gpu_matrix.handle2(), total_element_buffer_size);
    viennacl::backend::typesafe_host_array<IndexT> block_start(gpu_matrix.handle3(), (viennacl::traits::size1(cpu_matrix) - 1) / gpu_matrix.rows_per_block() + 1);
    std::vector<ScalarT> elements(total_element_buffer_size, 0);

    vcl_size_t block_offset = 0;
    vcl_size_t block_index  = 0;
    vcl_size_t row_in_block = 0;
    for (typename CPUMatrixT::const_iterator1 row_it = cpu_matrix.begin1(); row_it != cpu_matrix.end1(); ++row_it)
    {
      vcl_size_t entry_in_row = 0;

      for (typename CPUMatrixT::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
      {
        vcl_size_t buffer_index = block_offset + entry_in_row * gpu_matrix.rows_per_block() + row_in_block;
        coords.set(buffer_index, col_it.index2());
        elements[buffer_index] = *col_it;
        entry_in_row++;
      }

      ++row_in_block;

      // check for end of block:
      if ( (row_it.index1() % gpu_matrix.rows_per_block() == gpu_matrix.rows_per_block() - 1)
           || row_it.index1() == viennacl::traits::size1(cpu_matrix) - 1)
      {
        block_start.set(block_index, static_cast<IndexT>(block_offset));
        block_offset += columns_in_block_buffer[block_index] * gpu_matrix.rows_per_block();
        ++block_index;
        row_in_block = 0;
      }
    }

    viennacl::backend::memory_create(gpu_matrix.handle1(), columns_in_block_buffer.raw_size(), traits::context(gpu_matrix.handle1()), columns_in_block_buffer.get());
    viennacl::backend::memory_create(gpu_matrix.handle2(), coords.raw_size(),                  traits::context(gpu_matrix.handle2()), coords.get());
    viennacl::backend::memory_create(gpu_matrix.handle3(), block_start.raw_size(),             traits::context(gpu_matrix.handle3()), block_start.get());
    viennacl::backend::memory_create(gpu_matrix.handle(),  sizeof(ScalarT) * elements.size(),  traits::context(gpu_matrix.handle()), &(elements[0]));
  }
}



/** @brief Copies a sparse matrix from the host to the compute device. The host type is the std::vector< std::map < > > format .
  *
  * @param cpu_matrix   A sparse matrix on the host composed of an STL vector and an STL map.
  * @param gpu_matrix   The sparse ell_matrix from ViennaCL
  */
template<typename IndexT, typename NumericT, typename IndexT2>
void copy(std::vector< std::map<IndexT, NumericT> > const & cpu_matrix,
          sliced_ell_matrix<NumericT, IndexT2> & gpu_matrix)
{
  vcl_size_t max_col = 0;
  for (vcl_size_t i=0; i<cpu_matrix.size(); ++i)
  {
    if (cpu_matrix[i].size() > 0)
      max_col = std::max<vcl_size_t>(max_col, (cpu_matrix[i].rbegin())->first);
  }

  viennacl::copy(tools::const_sparse_matrix_adapter<NumericT, IndexT>(cpu_matrix, cpu_matrix.size(), max_col + 1), gpu_matrix);
}


/*
template<typename CPUMatrixT, typename ScalarT, typename IndexT>
void copy(sliced_ell_matrix<ScalarT, IndexT> const & gpu_matrix, CPUMatrixT & cpu_matrix )
{
  assert( (viennacl::traits::size1(cpu_matrix) == gpu_matrix.size1()) && bool("Size mismatch") );
  assert( (viennacl::traits::size2(cpu_matrix) == gpu_matrix.size2()) && bool("Size mismatch") );

  if (gpu_matrix.size1() > 0 && gpu_matrix.size2() > 0)
  {
    std::vector<NumericT> elements(gpu_matrix.internal_nnz());
    viennacl::backend::typesafe_host_array<unsigned int> coords(gpu_matrix.handle2(), gpu_matrix.internal_nnz());

    viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(NumericT) * elements.size(), &(elements[0]));
    viennacl::backend::memory_read(gpu_matrix.handle2(), 0, coords.raw_size(), coords.get());

    for (vcl_size_t row = 0; row < gpu_matrix.size1(); row++)
    {
      for (vcl_size_t ind = 0; ind < gpu_matrix.internal_maxnnz(); ind++)
      {
        vcl_size_t offset = gpu_matrix.internal_size1() * ind + row;

        if (elements[offset] == static_cast<NumericT>(0.0))
            continue;

        if (coords[offset] >= gpu_matrix.size2())
        {
            std::cerr << "ViennaCL encountered invalid data " << offset << " " << ind << " " << row << " " << coords[offset] << " " << gpu_matrix.size2() << std::endl;
            return;
        }

        cpu_matrix(row, coords[offset]) = elements[offset];
      }
    }
  }
} */


//
// Specify available operations:
//

/** \cond */

namespace linalg
{
namespace detail
{
  // x = A * y
  template<typename ScalarT, typename IndexT>
  struct op_executor<vector_base<ScalarT>, op_assign, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_base<ScalarT>, op_prod> >
  {
    static void apply(vector_base<ScalarT> & lhs, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_base<ScalarT>, op_prod> const & rhs)
    {
      // check for the special case x = A * x
      if (viennacl::traits::handle(lhs) == viennacl::traits::handle(rhs.rhs()))
      {
        viennacl::vector<ScalarT> temp(lhs);
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), ScalarT(1), temp, ScalarT(0));
        lhs = temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), ScalarT(1), lhs, ScalarT(0));
    }
  };

  template<typename ScalarT, typename IndexT>
  struct op_executor<vector_base<ScalarT>, op_inplace_add, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_base<ScalarT>, op_prod> >
  {
    static void apply(vector_base<ScalarT> & lhs, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_base<ScalarT>, op_prod> const & rhs)
    {
      // check for the special case x += A * x
      if (viennacl::traits::handle(lhs) == viennacl::traits::handle(rhs.rhs()))
      {
        viennacl::vector<ScalarT> temp(lhs);
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), ScalarT(1), temp, ScalarT(0));
        lhs += temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), ScalarT(1), lhs, ScalarT(1));
    }
  };

  template<typename ScalarT, typename IndexT>
  struct op_executor<vector_base<ScalarT>, op_inplace_sub, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_base<ScalarT>, op_prod> >
  {
    static void apply(vector_base<ScalarT> & lhs, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_base<ScalarT>, op_prod> const & rhs)
    {
      // check for the special case x -= A * x
      if (viennacl::traits::handle(lhs) == viennacl::traits::handle(rhs.rhs()))
      {
        viennacl::vector<ScalarT> temp(lhs);
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), ScalarT(1), temp, ScalarT(0));
        lhs -= temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), ScalarT(-1), lhs, ScalarT(1));
    }
  };


  // x = A * vec_op
  template<typename ScalarT, typename IndexT, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<ScalarT>, op_assign, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<ScalarT> & lhs, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<ScalarT> temp(rhs.rhs(), viennacl::traits::context(rhs));
      viennacl::linalg::prod_impl(rhs.lhs(), temp, lhs);
    }
  };

  // x = A * vec_op
  template<typename ScalarT, typename IndexT, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<ScalarT>, op_inplace_add, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<ScalarT> & lhs, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<ScalarT> temp(rhs.rhs(), viennacl::traits::context(rhs));
      viennacl::vector<ScalarT> temp_result(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), temp, temp_result);
      lhs += temp_result;
    }
  };

  // x = A * vec_op
  template<typename ScalarT, typename IndexT, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<ScalarT>, op_inplace_sub, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<ScalarT> & lhs, vector_expression<const sliced_ell_matrix<ScalarT, IndexT>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<ScalarT> temp(rhs.rhs(), viennacl::traits::context(rhs));
      viennacl::vector<ScalarT> temp_result(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), temp, temp_result);
      lhs -= temp_result;
    }
  };

} // namespace detail
} // namespace linalg

/** \endcond */
}

#endif


