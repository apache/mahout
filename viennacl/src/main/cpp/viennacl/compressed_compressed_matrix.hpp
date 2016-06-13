#ifndef VIENNACL_COMPRESSED_compressed_compressed_matrix_HPP_
#define VIENNACL_COMPRESSED_compressed_compressed_matrix_HPP_

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

/** @file viennacl/compressed_compressed_matrix.hpp
    @brief Implementation of the compressed_compressed_matrix class (CSR format with a relatively small number of nonzero rows)
*/

#include <vector>
#include <list>
#include <map>
#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"

#include "viennacl/linalg/sparse_matrix_operations.hpp"

#include "viennacl/tools/tools.hpp"
#include "viennacl/tools/entry_proxy.hpp"

namespace viennacl
{
namespace detail
{
  template<typename CPUMatrixT, typename NumericT>
  void copy_impl(const CPUMatrixT & cpu_matrix,
                 compressed_compressed_matrix<NumericT> & gpu_matrix,
                 vcl_size_t nonzero_rows,
                 vcl_size_t nonzeros)
  {
    assert( (gpu_matrix.size1() == 0 || viennacl::traits::size1(cpu_matrix) == gpu_matrix.size1()) && bool("Size mismatch") );
    assert( (gpu_matrix.size2() == 0 || viennacl::traits::size2(cpu_matrix) == gpu_matrix.size2()) && bool("Size mismatch") );

    viennacl::backend::typesafe_host_array<unsigned int> row_buffer(gpu_matrix.handle1(), nonzero_rows + 1);
    viennacl::backend::typesafe_host_array<unsigned int> row_indices(gpu_matrix.handle3(), nonzero_rows);
    viennacl::backend::typesafe_host_array<unsigned int> col_buffer(gpu_matrix.handle2(), nonzeros);
    std::vector<NumericT> elements(nonzeros);

    vcl_size_t row_index  = 0;
    vcl_size_t data_index = 0;

    for (typename CPUMatrixT::const_iterator1 row_it = cpu_matrix.begin1();
         row_it != cpu_matrix.end1();
         ++row_it)
    {
      bool row_empty = true;

      for (typename CPUMatrixT::const_iterator2 col_it = row_it.begin();
           col_it != row_it.end();
           ++col_it)
      {
        NumericT entry = *col_it;
        if (entry < 0 || entry > 0)  // entry != 0 without compiler warnings
        {
          if (row_empty)
          {
            assert(row_index < nonzero_rows && bool("Provided count of nonzero rows exceeded!"));

            row_empty = false;
            row_buffer.set(row_index, data_index);
            row_indices.set(row_index, col_it.index1());
            ++row_index;
          }

          col_buffer.set(data_index, col_it.index2());
          elements[data_index] = entry;
          ++data_index;
        }
      }
    }
    row_buffer.set(row_index, data_index);

    gpu_matrix.set(row_buffer.get(),
                   row_indices.get(),
                   col_buffer.get(),
                   &elements[0],
        cpu_matrix.size1(),
        cpu_matrix.size2(),
        nonzero_rows,
        nonzeros);
  }
}

//provide copy-operation:
/** @brief Copies a sparse matrix from the host to the OpenCL device (either GPU or multi-core CPU)
  *
  * There are some type requirements on the CPUMatrixT type (fulfilled by e.g. boost::numeric::ublas):
  * - .size1() returns the number of rows
  * - .size2() returns the number of columns
  * - const_iterator1    is a type definition for an iterator along increasing row indices
  * - const_iterator2    is a type definition for an iterator along increasing columns indices
  * - The const_iterator1 type provides an iterator of type const_iterator2 via members .begin() and .end() that iterates along column indices in the current row.
  * - The types const_iterator1 and const_iterator2 provide members functions .index1() and .index2() that return the current row and column indices respectively.
  * - Dereferenciation of an object of type const_iterator2 returns the entry.
  *
  * @param cpu_matrix   A sparse matrix on the host.
  * @param gpu_matrix   A compressed_compressed_matrix from ViennaCL
  */
template<typename CPUMatrixT, typename NumericT>
void copy(const CPUMatrixT & cpu_matrix,
          compressed_compressed_matrix<NumericT> & gpu_matrix )
{
  //std::cout << "copy for (" << cpu_matrix.size1() << ", " << cpu_matrix.size2() << ", " << cpu_matrix.nnz() << ")" << std::endl;

  if ( cpu_matrix.size1() > 0 && cpu_matrix.size2() > 0 )
  {
    //determine nonzero rows and total nonzeros:
    vcl_size_t num_entries = 0;
    vcl_size_t nonzero_rows = 0;
    for (typename CPUMatrixT::const_iterator1 row_it = cpu_matrix.begin1();
         row_it != cpu_matrix.end1();
         ++row_it)
    {
      bool row_empty = true;
      for (typename CPUMatrixT::const_iterator2 col_it = row_it.begin();
           col_it != row_it.end();
           ++col_it)
      {
        NumericT val = *col_it;
        if (val < 0 || val > 0) // val != 0 without compiler warnings
        {
          ++num_entries;

          if (row_empty)
          {
            row_empty = false;
            ++nonzero_rows;
          }
        }
      }
    }

    if (num_entries == 0) //we copy an empty matrix
      num_entries = 1;

    //set up matrix entries:
    viennacl::detail::copy_impl(cpu_matrix, gpu_matrix, nonzero_rows, num_entries);
  }
}


//adapted for std::vector< std::map < > > argument:
/** @brief Copies a sparse square matrix in the std::vector< std::map < > > format to an OpenCL device. Use viennacl::tools::sparse_matrix_adapter for non-square matrices.
  *
  * @param cpu_matrix   A sparse square matrix on the host using STL types
  * @param gpu_matrix   A compressed_compressed_matrix from ViennaCL
  */
template<typename SizeT, typename NumericT>
void copy(const std::vector< std::map<SizeT, NumericT> > & cpu_matrix,
          compressed_compressed_matrix<NumericT> & gpu_matrix )
{
  vcl_size_t nonzero_rows = 0;
  vcl_size_t nonzeros = 0;
  vcl_size_t max_col = 0;
  for (vcl_size_t i=0; i<cpu_matrix.size(); ++i)
  {
    if (cpu_matrix[i].size() > 0)
      ++nonzero_rows;
    nonzeros += cpu_matrix[i].size();
    if (cpu_matrix[i].size() > 0)
      max_col = std::max<vcl_size_t>(max_col, (cpu_matrix[i].rbegin())->first);
  }

  viennacl::detail::copy_impl(tools::const_sparse_matrix_adapter<NumericT, SizeT>(cpu_matrix, cpu_matrix.size(), max_col + 1),
                              gpu_matrix,
                              nonzero_rows,
                              nonzeros);
}


//
// gpu to cpu:
//
/** @brief Copies a sparse matrix from the OpenCL device (either GPU or multi-core CPU) to the host.
  *
  * There are two type requirements on the CPUMatrixT type (fulfilled by e.g. boost::numeric::ublas):
  * - resize(rows, cols)  A resize function to bring the matrix into the correct size
  * - operator(i,j)       Write new entries via the parenthesis operator
  *
  * @param gpu_matrix   A compressed_compressed_matrix from ViennaCL
  * @param cpu_matrix   A sparse matrix on the host.
  */
template<typename CPUMatrixT, typename NumericT>
void copy(const compressed_compressed_matrix<NumericT> & gpu_matrix,
          CPUMatrixT & cpu_matrix )
{
  assert( (cpu_matrix.size1() == gpu_matrix.size1()) && bool("Size mismatch") );
  assert( (cpu_matrix.size2() == gpu_matrix.size2()) && bool("Size mismatch") );

  if ( gpu_matrix.size1() > 0 && gpu_matrix.size2() > 0 )
  {
    //get raw data from memory:
    viennacl::backend::typesafe_host_array<unsigned int> row_buffer(gpu_matrix.handle1(), gpu_matrix.nnz1() + 1);
    viennacl::backend::typesafe_host_array<unsigned int> row_indices(gpu_matrix.handle1(), gpu_matrix.nnz1());
    viennacl::backend::typesafe_host_array<unsigned int> col_buffer(gpu_matrix.handle2(), gpu_matrix.nnz());
    std::vector<NumericT> elements(gpu_matrix.nnz());

    //std::cout << "GPU->CPU, nonzeros: " << gpu_matrix.nnz() << std::endl;

    viennacl::backend::memory_read(gpu_matrix.handle1(), 0, row_buffer.raw_size(), row_buffer.get());
    viennacl::backend::memory_read(gpu_matrix.handle3(), 0, row_indices.raw_size(), row_indices.get());
    viennacl::backend::memory_read(gpu_matrix.handle2(), 0, col_buffer.raw_size(), col_buffer.get());
    viennacl::backend::memory_read(gpu_matrix.handle(),  0, sizeof(NumericT)* gpu_matrix.nnz(), &(elements[0]));

    //fill the cpu_matrix:
    vcl_size_t data_index = 0;
    for (vcl_size_t i = 1; i < row_buffer.size(); ++i)
    {
      while (data_index < row_buffer[i])
      {
        if (col_buffer[data_index] >= gpu_matrix.size2())
        {
          std::cerr << "ViennaCL encountered invalid data at colbuffer[" << data_index << "]: " << col_buffer[data_index] << std::endl;
          return;
        }

        NumericT val = elements[data_index];
        if (val < 0 || val > 0) // val != 0 without compiler warning
          cpu_matrix(row_indices[i-1], col_buffer[data_index]) = val;
        ++data_index;
      }
    }
  }
}


/** @brief Copies a sparse matrix from an OpenCL device to the host. The host type is the std::vector< std::map < > > format .
  *
  * @param gpu_matrix   A compressed_compressed_matrix from ViennaCL
  * @param cpu_matrix   A sparse matrix on the host.
  */
template<typename NumericT>
void copy(const compressed_compressed_matrix<NumericT> & gpu_matrix,
          std::vector< std::map<unsigned int, NumericT> > & cpu_matrix)
{
  tools::sparse_matrix_adapter<NumericT> temp(cpu_matrix, cpu_matrix.size(), cpu_matrix.size());
  copy(gpu_matrix, temp);
}


//////////////////////// compressed_compressed_matrix //////////////////////////
/** @brief A sparse square matrix in compressed sparse rows format optimized for the case that only a few rows carry nonzero entries.
  *
  * The difference to the 'standard' CSR format is that there is an additional array 'row_indices' so that the i-th set of indices in the CSR-layout refers to row_indices[i].
  *
  * @tparam NumericT    The floating point type (either float or double, checked at compile time)
  * @tparam AlignmentV     The internal memory size for the entries in each row is given by (size()/AlignmentV + 1) * AlignmentV. AlignmentV must be a power of two. Best values or usually 4, 8 or 16, higher values are usually a waste of memory.
  */
template<class NumericT>
class compressed_compressed_matrix
{
public:
  typedef viennacl::backend::mem_handle                                                              handle_type;
  typedef scalar<typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<NumericT>::ResultType>   value_type;
  typedef vcl_size_t                                                                                 size_type;

  /** @brief Default construction of a compressed matrix. No memory is allocated */
  compressed_compressed_matrix() : rows_(0), cols_(0), nonzero_rows_(0), nonzeros_(0) {}

  /** @brief Construction of a compressed matrix with the supplied number of rows and columns. If the number of nonzeros is positive, memory is allocated
      *
      * @param rows         Number of rows
      * @param cols         Number of columns
      * @param nonzero_rows Optional number of nonzero rows for memory preallocation
      * @param nonzeros     Optional number of nonzeros for memory preallocation
      * @param ctx          Context in which to create the matrix. Uses the default context if omitted
      */
  explicit compressed_compressed_matrix(vcl_size_t rows, vcl_size_t cols, vcl_size_t nonzero_rows = 0, vcl_size_t nonzeros = 0, viennacl::context ctx = viennacl::context())
    : rows_(rows), cols_(cols), nonzero_rows_(nonzero_rows), nonzeros_(nonzeros)
  {
    row_buffer_.switch_active_handle_id(ctx.memory_type());
    row_indices_.switch_active_handle_id(ctx.memory_type());
    col_buffer_.switch_active_handle_id(ctx.memory_type());
    elements_.switch_active_handle_id(ctx.memory_type());

#ifdef VIENNACL_WITH_OPENCL
    if (ctx.memory_type() == OPENCL_MEMORY)
    {
      row_buffer_.opencl_handle().context(ctx.opencl_context());
      row_indices_.opencl_handle().context(ctx.opencl_context());
      col_buffer_.opencl_handle().context(ctx.opencl_context());
      elements_.opencl_handle().context(ctx.opencl_context());
    }
#endif
    if (rows > 0)
    {
      viennacl::backend::memory_create(row_buffer_, viennacl::backend::typesafe_host_array<unsigned int>().element_size() * (rows + 1), ctx);
    }
    if (nonzeros > 0)
    {
      viennacl::backend::memory_create(col_buffer_, viennacl::backend::typesafe_host_array<unsigned int>().element_size() * nonzeros, ctx);
      viennacl::backend::memory_create(elements_, sizeof(NumericT) * nonzeros, ctx);
    }
  }

  /** @brief Construction of a compressed matrix with the supplied number of rows and columns. If the number of nonzeros is positive, memory is allocated
      *
      * @param rows     Number of rows
      * @param cols     Number of columns
      * @param ctx      Context in which to create the matrix
      */
  explicit compressed_compressed_matrix(vcl_size_t rows, vcl_size_t cols, viennacl::context ctx)
    : rows_(rows), cols_(cols), nonzeros_(0)
  {
    row_buffer_.switch_active_handle_id(ctx.memory_type());
    col_buffer_.switch_active_handle_id(ctx.memory_type());
    elements_.switch_active_handle_id(ctx.memory_type());

#ifdef VIENNACL_WITH_OPENCL
    if (ctx.memory_type() == OPENCL_MEMORY)
    {
      row_buffer_.opencl_handle().context(ctx.opencl_context());
      col_buffer_.opencl_handle().context(ctx.opencl_context());
      elements_.opencl_handle().context(ctx.opencl_context());
    }
#endif
    if (rows > 0)
    {
      viennacl::backend::memory_create(row_buffer_, viennacl::backend::typesafe_host_array<unsigned int>().element_size() * (rows + 1), ctx);
    }
  }

  explicit compressed_compressed_matrix(viennacl::context ctx) : rows_(0), cols_(0), nonzero_rows_(0), nonzeros_(0)
  {
    row_buffer_.switch_active_handle_id(ctx.memory_type());
    row_indices_.switch_active_handle_id(ctx.memory_type());
    col_buffer_.switch_active_handle_id(ctx.memory_type());
    elements_.switch_active_handle_id(ctx.memory_type());

#ifdef VIENNACL_WITH_OPENCL
    if (ctx.memory_type() == OPENCL_MEMORY)
    {
      row_buffer_.opencl_handle().context(ctx.opencl_context());
      row_indices_.opencl_handle().context(ctx.opencl_context());
      col_buffer_.opencl_handle().context(ctx.opencl_context());
      elements_.opencl_handle().context(ctx.opencl_context());
    }
#endif
  }


#ifdef VIENNACL_WITH_OPENCL
  explicit compressed_compressed_matrix(cl_mem mem_row_buffer, cl_mem mem_row_indices, cl_mem mem_col_buffer, cl_mem mem_elements,
                                        vcl_size_t rows, vcl_size_t cols, vcl_size_t nonzero_rows, vcl_size_t nonzeros) :
    rows_(rows), cols_(cols), nonzero_rows_(nonzero_rows), nonzeros_(nonzeros)
  {
    row_buffer_.switch_active_handle_id(viennacl::OPENCL_MEMORY);
    row_buffer_.opencl_handle() = mem_row_buffer;
    row_buffer_.opencl_handle().inc();             //prevents that the user-provided memory is deleted once the matrix object is destroyed.
    row_buffer_.raw_size(sizeof(cl_uint) * (nonzero_rows + 1));

    row_indices_.switch_active_handle_id(viennacl::OPENCL_MEMORY);
    row_indices_.opencl_handle() = mem_row_indices;
    row_indices_.opencl_handle().inc();             //prevents that the user-provided memory is deleted once the matrix object is destroyed.
    row_indices_.raw_size(sizeof(cl_uint) * nonzero_rows);

    col_buffer_.switch_active_handle_id(viennacl::OPENCL_MEMORY);
    col_buffer_.opencl_handle() = mem_col_buffer;
    col_buffer_.opencl_handle().inc();             //prevents that the user-provided memory is deleted once the matrix object is destroyed.
    col_buffer_.raw_size(sizeof(cl_uint) * nonzeros);

    elements_.switch_active_handle_id(viennacl::OPENCL_MEMORY);
    elements_.opencl_handle() = mem_elements;
    elements_.opencl_handle().inc();               //prevents that the user-provided memory is deleted once the matrix object is destroyed.
    elements_.raw_size(sizeof(NumericT) * nonzeros);
  }
#endif


  /** @brief Assignment a compressed matrix from possibly another memory domain. */
  compressed_compressed_matrix & operator=(compressed_compressed_matrix const & other)
  {
    assert( (rows_ == 0 || rows_ == other.size1()) && bool("Size mismatch") );
    assert( (cols_ == 0 || cols_ == other.size2()) && bool("Size mismatch") );

    rows_ = other.size1();
    cols_ = other.size2();
    nonzero_rows_ = other.nnz1();
    nonzeros_ = other.nnz();

    viennacl::backend::typesafe_memory_copy<unsigned int>(other.row_buffer_,  row_buffer_);
    viennacl::backend::typesafe_memory_copy<unsigned int>(other.row_indices_, row_indices_);
    viennacl::backend::typesafe_memory_copy<unsigned int>(other.col_buffer_,  col_buffer_);
    viennacl::backend::typesafe_memory_copy<NumericT>(other.elements_, elements_);

    return *this;
  }


  /** @brief Sets the row, column and value arrays of the compressed matrix
      *
      * @param row_jumper     Pointer to an array holding the indices of the first element of each row (starting with zero). E.g. row_jumper[10] returns the index of the first entry of the 11th row. The array length is 'cols + 1'
      * @param row_indices    Array holding the indices of the nonzero rows
      * @param col_buffer     Pointer to an array holding the column index of each entry. The array length is 'nonzeros'
      * @param elements       Pointer to an array holding the entries of the sparse matrix. The array length is 'elements'
      * @param rows           Number of rows of the sparse matrix
      * @param cols           Number of columns of the sparse matrix
      * @param nonzero_rows   Number of nonzero rows
      * @param nonzeros       Total number of nonzero entries
      */
  void set(const void * row_jumper,
           const void * row_indices,
           const void * col_buffer,
           const NumericT * elements,
           vcl_size_t rows,
           vcl_size_t cols,
           vcl_size_t nonzero_rows,
           vcl_size_t nonzeros)
  {
    assert( (rows > 0)         && bool("Error in compressed_compressed_matrix::set(): Number of rows must be larger than zero!"));
    assert( (cols > 0)         && bool("Error in compressed_compressed_matrix::set(): Number of columns must be larger than zero!"));
    assert( (nonzero_rows > 0) && bool("Error in compressed_compressed_matrix::set(): Number of nonzero rows must be larger than zero!"));
    assert( (nonzeros > 0)     && bool("Error in compressed_compressed_matrix::set(): Number of nonzeros must be larger than zero!"));
    //std::cout << "Setting memory: " << cols + 1 << ", " << nonzeros << std::endl;

    viennacl::backend::memory_create(row_buffer_,  viennacl::backend::typesafe_host_array<unsigned int>(row_buffer_).element_size() * (nonzero_rows + 1),  viennacl::traits::context(row_buffer_),  row_jumper);
    viennacl::backend::memory_create(row_indices_, viennacl::backend::typesafe_host_array<unsigned int>(row_indices_).element_size() * nonzero_rows, viennacl::traits::context(row_indices_), row_indices);
    viennacl::backend::memory_create(col_buffer_,  viennacl::backend::typesafe_host_array<unsigned int>(col_buffer_).element_size() * nonzeros,    viennacl::traits::context(col_buffer_),  col_buffer);
    viennacl::backend::memory_create(elements_, sizeof(NumericT) * nonzeros, viennacl::traits::context(elements_), elements);

    nonzeros_ = nonzeros;
    nonzero_rows_ = nonzero_rows;
    rows_ = rows;
    cols_ = cols;
  }

  /** @brief Resets all entries in the matrix back to zero without changing the matrix size. Resets the sparsity pattern. */
  void clear()
  {
    viennacl::backend::typesafe_host_array<unsigned int> host_row_buffer(row_buffer_, rows_ + 1);
    viennacl::backend::typesafe_host_array<unsigned int> host_row_indices(row_indices_, rows_ + 1);
    viennacl::backend::typesafe_host_array<unsigned int> host_col_buffer(col_buffer_, 1);
    std::vector<NumericT> host_elements(1);

    viennacl::backend::memory_create(row_buffer_,  host_row_buffer.element_size() * (rows_ + 1),  viennacl::traits::context(row_buffer_),  host_row_buffer.get());
    viennacl::backend::memory_create(row_indices_, host_row_indices.element_size() * (rows_ + 1), viennacl::traits::context(row_indices_), host_row_indices.get());
    viennacl::backend::memory_create(col_buffer_,  host_col_buffer.element_size() * 1,            viennacl::traits::context(col_buffer_),  host_col_buffer.get());
    viennacl::backend::memory_create(elements_,    sizeof(NumericT) * 1,                          viennacl::traits::context(elements_),    &(host_elements[0]));

    nonzeros_ = 0;
    nonzero_rows_ = 0;
  }

  /** @brief  Returns the number of rows */
  const vcl_size_t & size1() const { return rows_; }
  /** @brief  Returns the number of columns */
  const vcl_size_t & size2() const { return cols_; }
  /** @brief  Returns the number of nonzero entries */
  const vcl_size_t & nnz1() const { return nonzero_rows_; }
  /** @brief  Returns the number of nonzero entries */
  const vcl_size_t & nnz() const { return nonzeros_; }

  /** @brief  Returns the OpenCL handle to the row index array */
  const handle_type & handle1() const { return row_buffer_; }
  /** @brief  Returns the OpenCL handle to the column index array */
  const handle_type & handle2() const { return col_buffer_; }
  /** @brief  Returns the OpenCL handle to the row index array */
  const handle_type & handle3() const { return row_indices_; }
  /** @brief  Returns the OpenCL handle to the matrix entry array */
  const handle_type & handle() const { return elements_; }

  /** @brief  Returns the OpenCL handle to the row index array */
  handle_type & handle1() { return row_buffer_; }
  /** @brief  Returns the OpenCL handle to the column index array */
  handle_type & handle2() { return col_buffer_; }
  /** @brief  Returns the OpenCL handle to the row index array */
  handle_type & handle3() { return row_indices_; }
  /** @brief  Returns the OpenCL handle to the matrix entry array */
  handle_type & handle() { return elements_; }

  void switch_memory_context(viennacl::context new_ctx)
  {
    viennacl::backend::switch_memory_context<unsigned int>(row_buffer_, new_ctx);
    viennacl::backend::switch_memory_context<unsigned int>(row_indices_, new_ctx);
    viennacl::backend::switch_memory_context<unsigned int>(col_buffer_, new_ctx);
    viennacl::backend::switch_memory_context<NumericT>(elements_, new_ctx);
  }

  viennacl::memory_types memory_context() const
  {
    return row_buffer_.get_active_handle_id();
  }

private:

  vcl_size_t rows_;
  vcl_size_t cols_;
  vcl_size_t nonzero_rows_;
  vcl_size_t nonzeros_;
  handle_type row_buffer_;
  handle_type row_indices_;
  handle_type col_buffer_;
  handle_type elements_;
};



//
// Specify available operations:
//

/** \cond */

namespace linalg
{
namespace detail
{
  // x = A * y
  template<typename T>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const compressed_compressed_matrix<T>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_compressed_matrix<T>, const vector_base<T>, op_prod> const & rhs)
    {
      // check for the special case x = A * x
      if (viennacl::traits::handle(lhs) == viennacl::traits::handle(rhs.rhs()))
      {
        viennacl::vector<T> temp(lhs);
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), T(1), temp, T(0));
        lhs = temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), T(1), lhs, T(0));
    }
  };

  template<typename T>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const compressed_compressed_matrix<T>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_compressed_matrix<T>, const vector_base<T>, op_prod> const & rhs)
    {
      // check for the special case x += A * x
      if (viennacl::traits::handle(lhs) == viennacl::traits::handle(rhs.rhs()))
      {
        viennacl::vector<T> temp(lhs);
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), T(1), temp, T(0));
        lhs += temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), T(1), lhs, T(1));
    }
  };

  template<typename T>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const compressed_compressed_matrix<T>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_compressed_matrix<T>, const vector_base<T>, op_prod> const & rhs)
    {
      // check for the special case x -= A * x
      if (viennacl::traits::handle(lhs) == viennacl::traits::handle(rhs.rhs()))
      {
        viennacl::vector<T> temp(lhs);
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), T(1), temp, T(0));
        lhs -= temp;
      }
      else
        viennacl::linalg::prod_impl(rhs.lhs(), rhs.rhs(), T(-1), lhs, T(1));
    }
  };


  // x = A * vec_op
  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const compressed_compressed_matrix<T>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_compressed_matrix<T>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs());
      viennacl::linalg::prod_impl(rhs.lhs(), temp, lhs);
    }
  };

  // x = A * vec_op
  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const compressed_compressed_matrix<T>, vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_compressed_matrix<T>, vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs(), viennacl::traits::context(rhs));
      viennacl::vector<T> temp_result(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), temp, temp_result);
      lhs += temp_result;
    }
  };

  // x = A * vec_op
  template<typename T, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const compressed_compressed_matrix<T>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const compressed_compressed_matrix<T>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs(), viennacl::traits::context(rhs));
      viennacl::vector<T> temp_result(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), temp, temp_result);
      lhs -= temp_result;
    }
  };

} // namespace detail
} // namespace linalg

/** \endcond */
}

#endif
