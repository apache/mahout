#ifndef VIENNACL_COORDINATE_MATRIX_HPP_
#define VIENNACL_COORDINATE_MATRIX_HPP_

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

/** @file viennacl/coordinate_matrix.hpp
    @brief Implementation of the coordinate_matrix class
*/

#include <map>
#include <vector>
#include <list>

#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"

#include "viennacl/linalg/sparse_matrix_operations.hpp"

namespace viennacl
{


//provide copy-operation:
/** @brief Copies a sparse matrix from the host to the OpenCL device (either GPU or multi-core CPU)
  *
  * For the requirements on the CPUMatrixT type, see the documentation of the function copy(CPUMatrixT, compressed_matrix<>)
  *
  * @param cpu_matrix   A sparse matrix on the host.
  * @param gpu_matrix   A compressed_matrix from ViennaCL
  */
template<typename CPUMatrixT, typename NumericT, unsigned int AlignmentV>
void copy(const CPUMatrixT & cpu_matrix,
          coordinate_matrix<NumericT, AlignmentV> & gpu_matrix )
{
  assert( (gpu_matrix.size1() == 0 || viennacl::traits::size1(cpu_matrix) == gpu_matrix.size1()) && bool("Size mismatch") );
  assert( (gpu_matrix.size2() == 0 || viennacl::traits::size2(cpu_matrix) == gpu_matrix.size2()) && bool("Size mismatch") );

  vcl_size_t group_num = 64;

  // Step 1: Determine nonzeros:
  if ( cpu_matrix.size1() > 0 && cpu_matrix.size2() > 0 )
  {
    vcl_size_t num_entries = 0;
    for (typename CPUMatrixT::const_iterator1 row_it = cpu_matrix.begin1(); row_it != cpu_matrix.end1(); ++row_it)
      for (typename CPUMatrixT::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
        ++num_entries;

    // Step 2: Set up matrix data:
    gpu_matrix.nonzeros_ = num_entries;
    gpu_matrix.rows_ = cpu_matrix.size1();
    gpu_matrix.cols_ = cpu_matrix.size2();

    viennacl::backend::typesafe_host_array<unsigned int> group_boundaries(gpu_matrix.handle3(), group_num + 1);
    viennacl::backend::typesafe_host_array<unsigned int> coord_buffer(gpu_matrix.handle12(), 2*gpu_matrix.internal_nnz());
    std::vector<NumericT> elements(gpu_matrix.internal_nnz());

    vcl_size_t data_index = 0;
    vcl_size_t current_fraction = 0;

    group_boundaries.set(0, 0);
    for (typename CPUMatrixT::const_iterator1 row_it = cpu_matrix.begin1();  row_it != cpu_matrix.end1(); ++row_it)
    {
      for (typename CPUMatrixT::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
      {
        coord_buffer.set(2*data_index, col_it.index1());
        coord_buffer.set(2*data_index + 1, col_it.index2());
        elements[data_index] = *col_it;
        ++data_index;
      }

      while (data_index > vcl_size_t(static_cast<double>(current_fraction + 1) / static_cast<double>(group_num)) * num_entries)    //split data equally over 64 groups
        group_boundaries.set(++current_fraction, data_index);
    }

    //write end of last group:
    group_boundaries.set(group_num, data_index);
    //group_boundaries[1] = data_index; //for one compute unit

    //std::cout << "Group boundaries: " << std::endl;
    //for (vcl_size_t i=0; i<group_boundaries.size(); ++i)
    //  std::cout << group_boundaries[i] << std::endl;

    viennacl::backend::memory_create(gpu_matrix.group_boundaries_, group_boundaries.raw_size(), traits::context(gpu_matrix.group_boundaries_), group_boundaries.get());
    viennacl::backend::memory_create(gpu_matrix.coord_buffer_,         coord_buffer.raw_size(), traits::context(gpu_matrix.coord_buffer_),     coord_buffer.get());
    viennacl::backend::memory_create(gpu_matrix.elements_,  sizeof(NumericT)*elements.size(), traits::context(gpu_matrix.elements_),         &(elements[0]));
  }
}

/** @brief Copies a sparse matrix in the std::vector< std::map < > > format to an OpenCL device.
  *
  * @param cpu_matrix   A sparse square matrix on the host.
  * @param gpu_matrix   A coordinate_matrix from ViennaCL
  */
template<typename NumericT, unsigned int AlignmentV>
void copy(const std::vector< std::map<unsigned int, NumericT> > & cpu_matrix,
          coordinate_matrix<NumericT, AlignmentV> & gpu_matrix )
{
  vcl_size_t max_col = 0;
  for (vcl_size_t i=0; i<cpu_matrix.size(); ++i)
  {
    if (cpu_matrix[i].size() > 0)
      max_col = std::max<vcl_size_t>(max_col, (cpu_matrix[i].rbegin())->first);
  }

  viennacl::copy(tools::const_sparse_matrix_adapter<NumericT>(cpu_matrix, cpu_matrix.size(), max_col + 1), gpu_matrix);
}

//gpu to cpu:
/** @brief Copies a sparse matrix from the OpenCL device (either GPU or multi-core CPU) to the host.
  *
  * There are two type requirements on the CPUMatrixT type (fulfilled by e.g. boost::numeric::ublas):
  * - resize(rows, cols)  A resize function to bring the matrix into the correct size
  * - operator(i,j)       Write new entries via the parenthesis operator
  *
  * @param gpu_matrix   A coordinate_matrix from ViennaCL
  * @param cpu_matrix   A sparse matrix on the host.
  */
template<typename CPUMatrixT, typename NumericT, unsigned int AlignmentV>
void copy(const coordinate_matrix<NumericT, AlignmentV> & gpu_matrix,
          CPUMatrixT & cpu_matrix )
{
  assert( (viennacl::traits::size1(cpu_matrix) == gpu_matrix.size1()) && bool("Size mismatch") );
  assert( (viennacl::traits::size2(cpu_matrix) == gpu_matrix.size2()) && bool("Size mismatch") );

  if ( gpu_matrix.size1() > 0 && gpu_matrix.size2() > 0 )
  {
    //get raw data from memory:
    viennacl::backend::typesafe_host_array<unsigned int> coord_buffer(gpu_matrix.handle12(), 2*gpu_matrix.nnz());
    std::vector<NumericT> elements(gpu_matrix.nnz());

    //std::cout << "GPU nonzeros: " << gpu_matrix.nnz() << std::endl;

    viennacl::backend::memory_read(gpu_matrix.handle12(), 0, coord_buffer.raw_size(), coord_buffer.get());
    viennacl::backend::memory_read(gpu_matrix.handle(),   0, sizeof(NumericT) * elements.size(), &(elements[0]));

    //fill the cpu_matrix:
    for (vcl_size_t index = 0; index < gpu_matrix.nnz(); ++index)
      cpu_matrix(coord_buffer[2*index], coord_buffer[2*index+1]) = elements[index];

  }
}

/** @brief Copies a sparse matrix from an OpenCL device to the host. The host type is the std::vector< std::map < > > format .
  *
  * @param gpu_matrix   A coordinate_matrix from ViennaCL
  * @param cpu_matrix   A sparse matrix on the host.
  */
template<typename NumericT, unsigned int AlignmentV>
void copy(const coordinate_matrix<NumericT, AlignmentV> & gpu_matrix,
          std::vector< std::map<unsigned int, NumericT> > & cpu_matrix)
{
  if (cpu_matrix.size() == 0)
    cpu_matrix.resize(gpu_matrix.size1());

  assert(cpu_matrix.size() == gpu_matrix.size1() && bool("Matrix dimension mismatch!"));

  tools::sparse_matrix_adapter<NumericT> temp(cpu_matrix, gpu_matrix.size1(), gpu_matrix.size2());
  copy(gpu_matrix, temp);
}


//////////////////////// coordinate_matrix //////////////////////////
/** @brief A sparse square matrix, where entries are stored as triplets (i,j, val), where i and j are the row and column indices and val denotes the entry.
  *
  * The present implementation of coordinate_matrix suffers from poor runtime efficiency. Users are adviced to use compressed_matrix in the meanwhile.
  *
  * @tparam NumericT    The floating point type (either float or double, checked at compile time)
  * @tparam AlignmentV     The internal memory size for the arrays, given by (size()/AlignmentV + 1) * AlignmentV. AlignmentV must be a power of two.
  */
template<class NumericT, unsigned int AlignmentV /* see forwards.h */ >
class coordinate_matrix
{
public:
  typedef viennacl::backend::mem_handle                                                              handle_type;
  typedef scalar<typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<NumericT>::ResultType>   value_type;
  typedef vcl_size_t                                                                                 size_type;

  /** @brief Default construction of a coordinate matrix. No memory is allocated */
  coordinate_matrix() : rows_(0), cols_(0), nonzeros_(0), group_num_(64) {}

  explicit coordinate_matrix(viennacl::context ctx) : rows_(0), cols_(0), nonzeros_(0), group_num_(64)
  {
    group_boundaries_.switch_active_handle_id(ctx.memory_type());
    coord_buffer_.switch_active_handle_id(ctx.memory_type());
    elements_.switch_active_handle_id(ctx.memory_type());

#ifdef VIENNACL_WITH_OPENCL
    if (ctx.memory_type() == OPENCL_MEMORY)
    {
      group_boundaries_.opencl_handle().context(ctx.opencl_context());
      coord_buffer_.opencl_handle().context(ctx.opencl_context());
      elements_.opencl_handle().context(ctx.opencl_context());
    }
#endif
  }

  /** @brief Construction of a coordinate matrix with the supplied number of rows and columns. If the number of nonzeros is positive, memory is allocated
      *
      * @param rows     Number of rows
      * @param cols     Number of columns
      * @param nonzeros Optional number of nonzeros for memory preallocation
      * @param ctx      Optional context in which the matrix is created (one out of multiple OpenCL contexts, CUDA, host)
      */
  coordinate_matrix(vcl_size_t rows, vcl_size_t cols, vcl_size_t nonzeros = 0, viennacl::context ctx = viennacl::context()) :
    rows_(rows), cols_(cols), nonzeros_(nonzeros)
  {
    if (nonzeros > 0)
    {
      viennacl::backend::memory_create(group_boundaries_, viennacl::backend::typesafe_host_array<unsigned int>().element_size() * (group_num_ + 1), ctx);
      viennacl::backend::memory_create(coord_buffer_,     viennacl::backend::typesafe_host_array<unsigned int>().element_size() * 2 * internal_nnz(), ctx);
      viennacl::backend::memory_create(elements_,         sizeof(NumericT) * internal_nnz(), ctx);
    }
    else
    {
      group_boundaries_.switch_active_handle_id(ctx.memory_type());
      coord_buffer_.switch_active_handle_id(ctx.memory_type());
      elements_.switch_active_handle_id(ctx.memory_type());

#ifdef VIENNACL_WITH_OPENCL
      if (ctx.memory_type() == OPENCL_MEMORY)
      {
        group_boundaries_.opencl_handle().context(ctx.opencl_context());
        coord_buffer_.opencl_handle().context(ctx.opencl_context());
        elements_.opencl_handle().context(ctx.opencl_context());
      }
#endif
    }
  }

  /** @brief Construction of a coordinate matrix with the supplied number of rows and columns in the supplied context. Does not yet allocate memory.
      *
      * @param rows     Number of rows
      * @param cols     Number of columns
      * @param ctx      Context in which to create the matrix
      */
  explicit coordinate_matrix(vcl_size_t rows, vcl_size_t cols, viennacl::context ctx)
    : rows_(rows), cols_(cols), nonzeros_(0)
  {
    group_boundaries_.switch_active_handle_id(ctx.memory_type());
    coord_buffer_.switch_active_handle_id(ctx.memory_type());
    elements_.switch_active_handle_id(ctx.memory_type());

#ifdef VIENNACL_WITH_OPENCL
    if (ctx.memory_type() == OPENCL_MEMORY)
    {
      group_boundaries_.opencl_handle().context(ctx.opencl_context());
      coord_buffer_.opencl_handle().context(ctx.opencl_context());
      elements_.opencl_handle().context(ctx.opencl_context());
    }
#endif
  }


  /** @brief Allocate memory for the supplied number of nonzeros in the matrix. Old values are preserved. */
  void reserve(vcl_size_t new_nonzeros)
  {
    if (new_nonzeros > nonzeros_)  //TODO: Do we need to initialize new memory with zero?
    {
      handle_type coord_buffer_old;
      handle_type elements_old;
      viennacl::backend::memory_shallow_copy(coord_buffer_, coord_buffer_old);
      viennacl::backend::memory_shallow_copy(elements_, elements_old);

      vcl_size_t internal_new_nnz = viennacl::tools::align_to_multiple<vcl_size_t>(new_nonzeros, AlignmentV);
      viennacl::backend::typesafe_host_array<unsigned int> size_deducer(coord_buffer_);
      viennacl::backend::memory_create(coord_buffer_, size_deducer.element_size() * 2 * internal_new_nnz, viennacl::traits::context(coord_buffer_));
      viennacl::backend::memory_create(elements_,     sizeof(NumericT)  * internal_new_nnz,             viennacl::traits::context(elements_));

      viennacl::backend::memory_copy(coord_buffer_old, coord_buffer_, 0, 0, size_deducer.element_size() * 2 * nonzeros_);
      viennacl::backend::memory_copy(elements_old,     elements_,     0, 0, sizeof(NumericT)  * nonzeros_);

      nonzeros_ = new_nonzeros;
    }
  }

  /** @brief Resize the matrix.
      *
      * @param new_size1    New number of rows
      * @param new_size2    New number of columns
      * @param preserve     If true, the old values are preserved. At present, old values are always discarded.
      */
  void resize(vcl_size_t new_size1, vcl_size_t new_size2, bool preserve = true)
  {
    assert (new_size1 > 0 && new_size2 > 0);

    if (new_size1 < rows_ || new_size2 < cols_) //enlarge buffer
    {
      std::vector<std::map<unsigned int, NumericT> > stl_sparse_matrix;
      if (rows_ > 0)
        stl_sparse_matrix.resize(rows_);

      if (preserve && rows_ > 0)
        viennacl::copy(*this, stl_sparse_matrix);

      stl_sparse_matrix.resize(new_size1);

      //std::cout << "Cropping STL matrix of size " << stl_sparse_matrix.size() << std::endl;
      if (new_size2 < cols_ && rows_ > 0)
      {
        for (vcl_size_t i=0; i<stl_sparse_matrix.size(); ++i)
        {
          std::list<unsigned int> to_delete;
          for (typename std::map<unsigned int, NumericT>::iterator it = stl_sparse_matrix[i].begin();
               it != stl_sparse_matrix[i].end();
               ++it)
          {
            if (it->first >= new_size2)
              to_delete.push_back(it->first);
          }

          for (std::list<unsigned int>::iterator it = to_delete.begin(); it != to_delete.end(); ++it)
            stl_sparse_matrix[i].erase(*it);
        }
        //std::cout << "Cropping done..." << std::endl;
      }

      rows_ = new_size1;
      cols_ = new_size2;
      viennacl::copy(stl_sparse_matrix, *this);
    }

    rows_ = new_size1;
    cols_ = new_size2;
  }

  /** @brief Resets all entries in the matrix back to zero without changing the matrix size. Resets the sparsity pattern. */
  void clear()
  {
    viennacl::backend::typesafe_host_array<unsigned int> host_group_buffer(group_boundaries_, 65);
    viennacl::backend::typesafe_host_array<unsigned int> host_coord_buffer(coord_buffer_, 2);
    std::vector<NumericT> host_elements(1);

    viennacl::backend::memory_create(group_boundaries_, host_group_buffer.element_size() * 65, viennacl::traits::context(group_boundaries_), host_group_buffer.get());
    viennacl::backend::memory_create(coord_buffer_,     host_coord_buffer.element_size() * 2,   viennacl::traits::context(coord_buffer_),     host_coord_buffer.get());
    viennacl::backend::memory_create(elements_,         sizeof(NumericT) * 1,                   viennacl::traits::context(elements_),         &(host_elements[0]));

    nonzeros_ = 0;
    group_num_ = 64;
  }

  /** @brief  Returns the number of rows */
  vcl_size_t size1() const { return rows_; }
  /** @brief  Returns the number of columns */
  vcl_size_t size2() const { return cols_; }
  /** @brief  Returns the number of nonzero entries */
  vcl_size_t nnz() const { return nonzeros_; }
  /** @brief  Returns the number of internal nonzero entries */
  vcl_size_t internal_nnz() const { return viennacl::tools::align_to_multiple<vcl_size_t>(nonzeros_, AlignmentV); }

  /** @brief  Returns the OpenCL handle to the (row, column) index array */
  const handle_type & handle12() const { return coord_buffer_; }
  /** @brief  Returns the OpenCL handle to the matrix entry array */
  const handle_type & handle() const { return elements_; }
  /** @brief  Returns the OpenCL handle to the group start index array */
  const handle_type & handle3() const { return group_boundaries_; }

  vcl_size_t groups() const { return group_num_; }

#if defined(_MSC_VER) && _MSC_VER < 1500      //Visual Studio 2005 needs special treatment
  template<typename CPUMatrixT>
  friend void copy(const CPUMatrixT & cpu_matrix, coordinate_matrix & gpu_matrix );
#else
  template<typename CPUMatrixT, typename NumericT2, unsigned int AlignmentV2>
  friend void copy(const CPUMatrixT & cpu_matrix, coordinate_matrix<NumericT2, AlignmentV2> & gpu_matrix );
#endif

private:
  /** @brief Copy constructor is by now not available. */
  coordinate_matrix(coordinate_matrix const &);

  /** @brief Assignment is by now not available. */
  coordinate_matrix & operator=(coordinate_matrix const &);


  vcl_size_t rows_;
  vcl_size_t cols_;
  vcl_size_t nonzeros_;
  vcl_size_t group_num_;
  handle_type coord_buffer_;
  handle_type elements_;
  handle_type group_boundaries_;
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
  template<typename T, unsigned int A>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const coordinate_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const coordinate_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
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

  template<typename T, unsigned int A>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const coordinate_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const coordinate_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
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

  template<typename T, unsigned int A>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const coordinate_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const coordinate_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
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
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_assign, vector_expression<const coordinate_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const coordinate_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs(), viennacl::traits::context(rhs));
      viennacl::linalg::prod_impl(rhs.lhs(), temp, lhs);
    }
  };

  // x += A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const coordinate_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const coordinate_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs(), viennacl::traits::context(rhs));
      viennacl::vector<T> temp_result(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), temp, temp_result);
      lhs += temp_result;
    }
  };

  // x -= A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const coordinate_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const coordinate_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
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
