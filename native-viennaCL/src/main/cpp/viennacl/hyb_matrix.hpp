#ifndef VIENNACL_HYB_MATRIX_HPP_
#define VIENNACL_HYB_MATRIX_HPP_

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

/** @file viennacl/hyb_matrix.hpp
    @brief Implementation of the hyb_matrix class

    Contributed by Volodymyr Kysenko.
*/

#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"

#include "viennacl/tools/tools.hpp"

#include "viennacl/linalg/sparse_matrix_operations.hpp"

namespace viennacl
{
/** @brief Sparse matrix class using a hybrid format composed of the ELL and CSR format for storing the nonzeros. */
template<typename NumericT, unsigned int AlignmentV  /* see forwards.h for default argument */>
class hyb_matrix
{
public:
  typedef viennacl::backend::mem_handle                                                              handle_type;
  typedef scalar<typename viennacl::tools::CHECK_SCALAR_TEMPLATE_ARGUMENT<NumericT>::ResultType>   value_type;

  hyb_matrix() : csr_threshold_(NumericT(0.8)), rows_(0), cols_(0) {}

  hyb_matrix(viennacl::context ctx) : csr_threshold_(NumericT(0.8)), rows_(0), cols_(0)
  {
    ell_coords_.switch_active_handle_id(ctx.memory_type());
    ell_elements_.switch_active_handle_id(ctx.memory_type());

    csr_rows_.switch_active_handle_id(ctx.memory_type());
    csr_cols_.switch_active_handle_id(ctx.memory_type());
    csr_elements_.switch_active_handle_id(ctx.memory_type());

#ifdef VIENNACL_WITH_OPENCL
    if (ctx.memory_type() == OPENCL_MEMORY)
    {
      ell_coords_.opencl_handle().context(ctx.opencl_context());
      ell_elements_.opencl_handle().context(ctx.opencl_context());

      csr_rows_.opencl_handle().context(ctx.opencl_context());
      csr_cols_.opencl_handle().context(ctx.opencl_context());
      csr_elements_.opencl_handle().context(ctx.opencl_context());
    }
#endif
  }

  /** @brief Resets all entries in the matrix back to zero without changing the matrix size. Resets the sparsity pattern. */
  void clear()
  {
    // ELL part:
    ellnnz_ = 0;

    viennacl::backend::typesafe_host_array<unsigned int> host_coords_buffer(ell_coords_, internal_size1());
    std::vector<NumericT> host_elements(internal_size1());

    viennacl::backend::memory_create(ell_coords_,   host_coords_buffer.element_size() * internal_size1(), viennacl::traits::context(ell_coords_),   host_coords_buffer.get());
    viennacl::backend::memory_create(ell_elements_, sizeof(NumericT) * internal_size1(),                  viennacl::traits::context(ell_elements_), &(host_elements[0]));

    // CSR part:
    csrnnz_ = 0;

    viennacl::backend::typesafe_host_array<unsigned int> host_row_buffer(csr_rows_, rows_ + 1);
    viennacl::backend::typesafe_host_array<unsigned int> host_col_buffer(csr_cols_, 1);
    host_elements.resize(1);

    viennacl::backend::memory_create(csr_rows_,     host_row_buffer.element_size() * (rows_ + 1), viennacl::traits::context(csr_rows_),     host_row_buffer.get());
    viennacl::backend::memory_create(csr_cols_,     host_col_buffer.element_size() * 1,           viennacl::traits::context(csr_cols_),     host_col_buffer.get());
    viennacl::backend::memory_create(csr_elements_, sizeof(NumericT) * 1,                         viennacl::traits::context(csr_elements_), &(host_elements[0]));
  }

  NumericT  csr_threshold()  const { return csr_threshold_; }
  void csr_threshold(NumericT thr) { csr_threshold_ = thr; }

  vcl_size_t internal_size1() const { return viennacl::tools::align_to_multiple<vcl_size_t>(rows_, AlignmentV); }
  vcl_size_t internal_size2() const { return viennacl::tools::align_to_multiple<vcl_size_t>(cols_, AlignmentV); }

  vcl_size_t size1() const { return rows_; }
  vcl_size_t size2() const { return cols_; }

  vcl_size_t internal_ellnnz() const {return viennacl::tools::align_to_multiple<vcl_size_t>(ellnnz_, AlignmentV); }
  vcl_size_t ell_nnz() const { return ellnnz_; }
  vcl_size_t csr_nnz() const { return csrnnz_; }

  const handle_type & handle() const { return ell_elements_; }
  const handle_type & handle2() const { return ell_coords_; }
  const handle_type & handle3() const { return csr_rows_; }
  const handle_type & handle4() const { return csr_cols_; }
  const handle_type & handle5() const { return csr_elements_; }

public:
#if defined(_MSC_VER) && _MSC_VER < 1500          //Visual Studio 2005 needs special treatment
  template<typename CPUMatrixT>
  friend void copy(const CPUMatrixT & cpu_matrix, hyb_matrix & gpu_matrix );
#else
  template<typename CPUMatrixT, typename T, unsigned int ALIGN>
  friend void copy(const CPUMatrixT & cpu_matrix, hyb_matrix<T, ALIGN> & gpu_matrix );
#endif

private:
  NumericT  csr_threshold_;
  vcl_size_t rows_;
  vcl_size_t cols_;
  vcl_size_t ellnnz_;
  vcl_size_t csrnnz_;

  handle_type ell_coords_; // ell coords
  handle_type ell_elements_; // ell elements

  handle_type csr_rows_;
  handle_type csr_cols_;
  handle_type csr_elements_;
};

template<typename CPUMatrixT, typename NumericT, unsigned int AlignmentV>
void copy(const CPUMatrixT& cpu_matrix, hyb_matrix<NumericT, AlignmentV>& gpu_matrix )
{
  assert( (gpu_matrix.size1() == 0 || viennacl::traits::size1(cpu_matrix) == gpu_matrix.size1()) && bool("Size mismatch") );
  assert( (gpu_matrix.size2() == 0 || viennacl::traits::size2(cpu_matrix) == gpu_matrix.size2()) && bool("Size mismatch") );

  if (cpu_matrix.size1() > 0 && cpu_matrix.size2() > 0)
  {
    //determine max capacity for row
    vcl_size_t max_entries_per_row = 0;
    std::vector<vcl_size_t> hist_entries(cpu_matrix.size2() + 1, 0);

    for (typename CPUMatrixT::const_iterator1 row_it = cpu_matrix.begin1(); row_it != cpu_matrix.end1(); ++row_it)
    {
      vcl_size_t num_entries = 0;
      for (typename CPUMatrixT::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
      {
        ++num_entries;
      }

      hist_entries[num_entries] += 1;
      max_entries_per_row = std::max(max_entries_per_row, num_entries);
    }

    vcl_size_t sum = 0;
    for (vcl_size_t ind = 0; ind <= max_entries_per_row; ind++)
    {
      sum += hist_entries[ind];

      if (NumericT(sum) >= NumericT(gpu_matrix.csr_threshold()) * NumericT(cpu_matrix.size1()))
      {
        max_entries_per_row = ind;
        break;
      }
    }

    //setup GPU matrix
    gpu_matrix.ellnnz_ = max_entries_per_row;
    gpu_matrix.rows_ = cpu_matrix.size1();
    gpu_matrix.cols_ = cpu_matrix.size2();

    vcl_size_t nnz = gpu_matrix.internal_size1() * gpu_matrix.internal_ellnnz();

    viennacl::backend::typesafe_host_array<unsigned int>  ell_coords(gpu_matrix.ell_coords_, nnz);
    viennacl::backend::typesafe_host_array<unsigned int>  csr_rows(gpu_matrix.csr_rows_, cpu_matrix.size1() + 1);
    std::vector<unsigned int> csr_cols;

    std::vector<NumericT> ell_elements(nnz);
    std::vector<NumericT> csr_elements;

    vcl_size_t csr_index = 0;

    for (typename CPUMatrixT::const_iterator1 row_it = cpu_matrix.begin1(); row_it != cpu_matrix.end1(); ++row_it)
    {
      vcl_size_t data_index = 0;

      csr_rows.set(row_it.index1(), csr_index);

      for (typename CPUMatrixT::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
      {
        if (data_index < max_entries_per_row)
        {
          ell_coords.set(gpu_matrix.internal_size1() * data_index + col_it.index1(), col_it.index2());
          ell_elements[gpu_matrix.internal_size1() * data_index + col_it.index1()] = *col_it;
        }
        else
        {
          csr_cols.push_back(static_cast<unsigned int>(col_it.index2()));
          csr_elements.push_back(*col_it);

          csr_index++;
        }

        data_index++;
      }

    }

    if (csr_cols.empty())
    {
      csr_cols.push_back(0);
      csr_elements.push_back(0);
    }

    csr_rows.set(csr_rows.size() - 1, csr_index);

    gpu_matrix.csrnnz_ = csr_cols.size();

    viennacl::backend::typesafe_host_array<unsigned int> csr_cols_for_gpu(gpu_matrix.csr_cols_, csr_cols.size());
    for (vcl_size_t i=0; i<csr_cols.size(); ++i)
      csr_cols_for_gpu.set(i, csr_cols[i]);

    viennacl::backend::memory_create(gpu_matrix.ell_coords_,   ell_coords.raw_size(),                    traits::context(gpu_matrix.ell_coords_), ell_coords.get());
    viennacl::backend::memory_create(gpu_matrix.ell_elements_, sizeof(NumericT) * ell_elements.size(), traits::context(gpu_matrix.ell_elements_), &(ell_elements[0]));

    viennacl::backend::memory_create(gpu_matrix.csr_rows_,     csr_rows.raw_size(),                      traits::context(gpu_matrix.csr_rows_), csr_rows.get());
    viennacl::backend::memory_create(gpu_matrix.csr_cols_,     csr_cols_for_gpu.raw_size(),              traits::context(gpu_matrix.csr_cols_), csr_cols_for_gpu.get());
    viennacl::backend::memory_create(gpu_matrix.csr_elements_, sizeof(NumericT) * csr_elements.size(), traits::context(gpu_matrix.csr_elements_), &(csr_elements[0]));
  }
}


/** @brief Copies a sparse matrix from the host to the compute device. The host type is the std::vector< std::map < > > format .
  *
  * @param cpu_matrix   A sparse matrix on the host composed of an STL vector and an STL map.
  * @param gpu_matrix   The sparse hyb_matrix from ViennaCL
  */
template<typename IndexT, typename NumericT, unsigned int AlignmentV>
void copy(std::vector< std::map<IndexT, NumericT> > const & cpu_matrix,
          hyb_matrix<NumericT, AlignmentV> & gpu_matrix)
{
  vcl_size_t max_col = 0;
  for (vcl_size_t i=0; i<cpu_matrix.size(); ++i)
  {
    if (cpu_matrix[i].size() > 0)
      max_col = std::max<vcl_size_t>(max_col, (cpu_matrix[i].rbegin())->first);
  }

  viennacl::copy(tools::const_sparse_matrix_adapter<NumericT, IndexT>(cpu_matrix, cpu_matrix.size(), max_col + 1), gpu_matrix);
}




template<typename CPUMatrixT, typename NumericT, unsigned int AlignmentV>
void copy(const hyb_matrix<NumericT, AlignmentV>& gpu_matrix, CPUMatrixT& cpu_matrix)
{
  assert( (viennacl::traits::size1(cpu_matrix) == gpu_matrix.size1()) && bool("Size mismatch") );
  assert( (viennacl::traits::size2(cpu_matrix) == gpu_matrix.size2()) && bool("Size mismatch") );

  if (gpu_matrix.size1() > 0 && gpu_matrix.size2() > 0)
  {
    std::vector<NumericT> ell_elements(gpu_matrix.internal_size1() * gpu_matrix.internal_ellnnz());
    viennacl::backend::typesafe_host_array<unsigned int> ell_coords(gpu_matrix.handle2(), gpu_matrix.internal_size1() * gpu_matrix.internal_ellnnz());

    std::vector<NumericT> csr_elements(gpu_matrix.csr_nnz());
    viennacl::backend::typesafe_host_array<unsigned int> csr_rows(gpu_matrix.handle3(), gpu_matrix.size1() + 1);
    viennacl::backend::typesafe_host_array<unsigned int> csr_cols(gpu_matrix.handle4(), gpu_matrix.csr_nnz());

    viennacl::backend::memory_read(gpu_matrix.handle(), 0, sizeof(NumericT) * ell_elements.size(), &(ell_elements[0]));
    viennacl::backend::memory_read(gpu_matrix.handle2(), 0, ell_coords.raw_size(), ell_coords.get());
    viennacl::backend::memory_read(gpu_matrix.handle3(), 0, csr_rows.raw_size(),   csr_rows.get());
    viennacl::backend::memory_read(gpu_matrix.handle4(), 0, csr_cols.raw_size(),   csr_cols.get());
    viennacl::backend::memory_read(gpu_matrix.handle5(), 0, sizeof(NumericT) * csr_elements.size(), &(csr_elements[0]));


    for (vcl_size_t row = 0; row < gpu_matrix.size1(); row++)
    {
      for (vcl_size_t ind = 0; ind < gpu_matrix.internal_ellnnz(); ind++)
      {
        vcl_size_t offset = gpu_matrix.internal_size1() * ind + row;

        NumericT val = ell_elements[offset];
        if (val <= 0 && val >= 0) // val == 0 without compiler warnings
          continue;

        if (ell_coords[offset] >= gpu_matrix.size2())
        {
          std::cerr << "ViennaCL encountered invalid data " << offset << " " << ind << " " << row << " " << ell_coords[offset] << " " << gpu_matrix.size2() << std::endl;
          return;
        }

        cpu_matrix(row, ell_coords[offset]) = val;
      }

      for (vcl_size_t ind = csr_rows[row]; ind < csr_rows[row+1]; ind++)
      {
        NumericT val = csr_elements[ind];
        if (val <= 0 && val >= 0) // val == 0 without compiler warnings
          continue;

        if (csr_cols[ind] >= gpu_matrix.size2())
        {
          std::cerr << "ViennaCL encountered invalid data " << std::endl;
          return;
        }

        cpu_matrix(row, csr_cols[ind]) = val;
      }
    }
  }
}

/** @brief Copies a sparse matrix from the compute device to the host. The host type is the std::vector< std::map < > > format .
  *
  * @param gpu_matrix   The sparse hyb_matrix from ViennaCL
  * @param cpu_matrix   A sparse matrix on the host composed of an STL vector and an STL map.
  */
template<typename NumericT, unsigned int AlignmentV, typename IndexT>
void copy(const hyb_matrix<NumericT, AlignmentV> & gpu_matrix,
          std::vector< std::map<IndexT, NumericT> > & cpu_matrix)
{
  if (cpu_matrix.size() == 0)
    cpu_matrix.resize(gpu_matrix.size1());

  assert(cpu_matrix.size() == gpu_matrix.size1() && bool("Matrix dimension mismatch!"));

  tools::sparse_matrix_adapter<NumericT, IndexT> temp(cpu_matrix, cpu_matrix.size(), gpu_matrix.size2());
  viennacl::copy(gpu_matrix, temp);
}

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
  struct op_executor<vector_base<T>, op_assign, vector_expression<const hyb_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hyb_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
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
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const hyb_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hyb_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
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
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const hyb_matrix<T, A>, const vector_base<T>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hyb_matrix<T, A>, const vector_base<T>, op_prod> const & rhs)
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
  struct op_executor<vector_base<T>, op_assign, vector_expression<const hyb_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hyb_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs(), viennacl::traits::context(rhs));
      viennacl::linalg::prod_impl(rhs.lhs(), temp, lhs);
    }
  };

  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_add, vector_expression<const hyb_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hyb_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
    {
      viennacl::vector<T> temp(rhs.rhs(), viennacl::traits::context(rhs));
      viennacl::vector<T> temp_result(lhs);
      viennacl::linalg::prod_impl(rhs.lhs(), temp, temp_result);
      lhs += temp_result;
    }
  };

  // x = A * vec_op
  template<typename T, unsigned int A, typename LHS, typename RHS, typename OP>
  struct op_executor<vector_base<T>, op_inplace_sub, vector_expression<const hyb_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> >
  {
    static void apply(vector_base<T> & lhs, vector_expression<const hyb_matrix<T, A>, const vector_expression<const LHS, const RHS, OP>, op_prod> const & rhs)
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
