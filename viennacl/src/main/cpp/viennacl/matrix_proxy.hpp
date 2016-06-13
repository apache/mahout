#ifndef VIENNACL_MATRIX_PROXY_HPP_
#define VIENNACL_MATRIX_PROXY_HPP_

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

/** @file matrix_proxy.hpp
    @brief Proxy classes for matrices.
*/

#include "viennacl/forwards.h"
#include "viennacl/range.hpp"
#include "viennacl/slice.hpp"
#include "viennacl/detail/matrix_def.hpp"
#include "viennacl/traits/size.hpp"

namespace viennacl
{

namespace detail
{
  ///////// const access

  template<typename NumericT, typename MatrixT>
  NumericT const & matrix_access(MatrixT const & A, vcl_size_t i, vcl_size_t j)
  {
    return A(i, j);
  }

  template<typename NumericT>
  NumericT const & matrix_access(std::vector< std::vector<NumericT> > const & A, vcl_size_t i, vcl_size_t j)
  {
    return A[i][j];
  }

  //////// non-const access

  template<typename NumericT, typename MatrixT>
  NumericT & matrix_access(MatrixT & A, vcl_size_t i, vcl_size_t j)
  {
    return A(i, j);
  }

  template<typename NumericT>
  NumericT & matrix_access(std::vector< std::vector<NumericT> > & A, vcl_size_t i, vcl_size_t j)
  {
    return A[i][j];
  }

}

/** @brief Class for representing non-strided submatrices of a bigger matrix A.
  *
  * In MATLAB notation, this could for example refer to the submatrix A(3:8, 6:10) of a matrix A.
  */
template<typename MatrixType>
class matrix_range : public matrix_base<typename MatrixType::cpu_value_type>
{
  typedef matrix_base<typename MatrixType::cpu_value_type>    base_type;
  typedef matrix_range<MatrixType>                            self_type;

public:
  typedef typename MatrixType::value_type     value_type;
  typedef typename MatrixType::handle_type    handle_type;
  typedef typename viennacl::result_of::cpu_value_type<value_type>::type    cpu_value_type;
  typedef range::size_type                    size_type;
  typedef range::difference_type              difference_type;
  typedef value_type                          reference;
  typedef const value_type &                  const_reference;


  matrix_range(MatrixType const & A,
               range const & row_range,
               range const & col_range) : base_type(const_cast<handle_type &>(A.handle()),
                                                    row_range.size(), row_range.start() * A.stride1() + A.start1(), A.stride1(), A.internal_size1(),
                                                    col_range.size(), col_range.start() * A.stride2() + A.start2(), A.stride2(), A.internal_size2(),
                                                    A.row_major()) {}

  matrix_range(self_type const & A,
               range const & row_range,
               range const & col_range) : base_type(const_cast<handle_type &>(A.handle()),
                                                    row_range.size(), row_range.start() * A.stride1() + A.start1(), A.stride1(), A.internal_size1(),
                                                    col_range.size(), col_range.start() * A.stride2() + A.start2(), A.stride2(), A.internal_size2(),
                                                    A.row_major()) {}


  matrix_range(self_type const & other) : base_type(const_cast<handle_type &>(other.handle()),
                                                    other.size1(), other.start1(), other.stride1(), other.internal_size1(),
                                                    other.size2(), other.start2(), other.stride2(), other.internal_size2(),
                                                    other.row_major()) {}

  using base_type::operator=;

  // the following are needed for Visual Studio:
  template<typename OtherNumericT, typename F>
  base_type & operator=(viennacl::matrix<OtherNumericT, F> const & B)                          { return base_type::operator=(static_cast<viennacl::matrix_base<OtherNumericT> const &>(B)); }

  template<typename OtherNumericT, typename F>
  base_type & operator=(viennacl::matrix_range<viennacl::matrix<OtherNumericT, F> > const & B) { return base_type::operator=(static_cast<viennacl::matrix_base<OtherNumericT> const &>(B)); }

  template<typename OtherNumericT, typename F>
  base_type & operator=(viennacl::matrix_slice<viennacl::matrix<OtherNumericT, F> > const & B) { return base_type::operator=(static_cast<viennacl::matrix_base<OtherNumericT> const &>(B)); }
};

template<typename MatrixType>
class matrix_range<matrix_range<MatrixType> > : public matrix_base<typename MatrixType::cpu_value_type>
{
  typedef matrix_base<typename MatrixType::cpu_value_type>    base_type;
public:
  typedef typename MatrixType::handle_type    handle_type;

  matrix_range(MatrixType const & A,
               range const & row_range,
               range const & col_range) : base_type(const_cast<handle_type &>(A.handle()),
                                                    row_range.size(), row_range.start() * A.stride1() + A.start1(), A.stride1(), A.internal_size1(),
                                                    col_range.size(), col_range.start() * A.stride2() + A.start2(), A.stride2(), A.internal_size2(),
                                                    A.row_major()) {}

  matrix_range(matrix_range<MatrixType> const & A,
               range const & row_range,
               range const & col_range) : base_type(const_cast<handle_type &>(A.handle()),
                                                    row_range.size(), row_range.start() * A.stride1() + A.start1(), A.stride1(), A.internal_size1(),
                                                    col_range.size(), col_range.start() * A.stride2() + A.start2(), A.stride2(), A.internal_size2(),
                                                    A.row_major()) {}
};

/////////////////////////////////////////////////////////////
///////////////////////// CPU to GPU ////////////////////////
/////////////////////////////////////////////////////////////

//row_major:
template<typename CPUMatrixT, typename NumericT>
void copy(const CPUMatrixT & cpu_matrix,
          matrix_range<matrix<NumericT, row_major, 1> > & gpu_matrix_range )
{
  assert(    (viennacl::traits::size1(cpu_matrix) == gpu_matrix_range.size1())
          && (viennacl::traits::size2(cpu_matrix) == gpu_matrix_range.size2())
          && bool("Matrix size mismatch!"));

  if ( gpu_matrix_range.start2() != 0)
  {
    std::vector<NumericT> entries(gpu_matrix_range.size2());

    //copy each stride separately:
    for (vcl_size_t i=0; i < gpu_matrix_range.size1(); ++i)
    {
      for (vcl_size_t j=0; j < gpu_matrix_range.size2(); ++j)
        entries[j] = detail::matrix_access<NumericT>(cpu_matrix, i, j);

      vcl_size_t start_offset = (gpu_matrix_range.start1() + i) * gpu_matrix_range.internal_size2() + gpu_matrix_range.start2();
      vcl_size_t num_entries = gpu_matrix_range.size2();
      viennacl::backend::memory_write(gpu_matrix_range.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));
      //std::cout << "Strided copy worked!" << std::endl;
    }
  }
  else
  {
    //full block can be copied:
    std::vector<NumericT> entries(gpu_matrix_range.size1()*gpu_matrix_range.internal_size2());

    //copy each stride separately:
    for (vcl_size_t i=0; i < gpu_matrix_range.size1(); ++i)
      for (vcl_size_t j=0; j < gpu_matrix_range.size2(); ++j)
        entries[i*gpu_matrix_range.internal_size2() + j] = detail::matrix_access<NumericT>(cpu_matrix, i, j);

    vcl_size_t start_offset = gpu_matrix_range.start1() * gpu_matrix_range.internal_size2();
    vcl_size_t num_entries = gpu_matrix_range.size1() * gpu_matrix_range.internal_size2();
    viennacl::backend::memory_write(gpu_matrix_range.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));
    //std::cout << "Block copy worked!" << std::endl;
  }
}

//column_major:
template<typename CPUMatrixT, typename NumericT>
void copy(const CPUMatrixT & cpu_matrix,
          matrix_range<matrix<NumericT, column_major, 1> > & gpu_matrix_range )
{
  assert(    (viennacl::traits::size1(cpu_matrix) == gpu_matrix_range.size1())
          && (viennacl::traits::size2(cpu_matrix) == gpu_matrix_range.size2())
          && bool("Matrix size mismatch!"));

  if ( gpu_matrix_range.start1() != 0 ||  gpu_matrix_range.size1() != gpu_matrix_range.size1())
  {
    std::vector<NumericT> entries(gpu_matrix_range.size1());

    //copy each stride separately:
    for (vcl_size_t j=0; j < gpu_matrix_range.size2(); ++j)
    {
      for (vcl_size_t i=0; i < gpu_matrix_range.size1(); ++i)
        entries[i] = detail::matrix_access<NumericT>(cpu_matrix, i, j);

      vcl_size_t start_offset = (gpu_matrix_range.start2() + j) * gpu_matrix_range.internal_size1() + gpu_matrix_range.start1();
      vcl_size_t num_entries = gpu_matrix_range.size1();
      viennacl::backend::memory_write(gpu_matrix_range.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));
      //std::cout << "Strided copy worked!" << std::endl;
    }
  }
  else
  {
    //full block can be copied:
    std::vector<NumericT> entries(gpu_matrix_range.internal_size1()*gpu_matrix_range.size2());

    //copy each stride separately:
    for (vcl_size_t i=0; i < gpu_matrix_range.size1(); ++i)
      for (vcl_size_t j=0; j < gpu_matrix_range.size2(); ++j)
        entries[i + j*gpu_matrix_range.internal_size1()] = detail::matrix_access<NumericT>(cpu_matrix, i, j);

    vcl_size_t start_offset = gpu_matrix_range.start2() * gpu_matrix_range.internal_size1();
    vcl_size_t num_entries = gpu_matrix_range.internal_size1() * gpu_matrix_range.size2();
    viennacl::backend::memory_write(gpu_matrix_range.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));
    //std::cout << "Block copy worked!" << std::endl;
  }

}


/////////////////////////////////////////////////////////////
///////////////////////// GPU to CPU ////////////////////////
/////////////////////////////////////////////////////////////


//row_major:
template<typename CPUMatrixT, typename NumericT>
void copy(matrix_range<matrix<NumericT, row_major, 1> > const & gpu_matrix_range,
          CPUMatrixT & cpu_matrix)
{
  assert(    (viennacl::traits::size1(cpu_matrix) == gpu_matrix_range.size1())
          && (viennacl::traits::size2(cpu_matrix) == gpu_matrix_range.size2())
          && bool("Matrix size mismatch!"));

  if ( gpu_matrix_range.start2() != 0)
  {
    std::vector<NumericT> entries(gpu_matrix_range.size2());

    //copy each stride separately:
    for (vcl_size_t i=0; i < gpu_matrix_range.size1(); ++i)
    {
      vcl_size_t start_offset = (gpu_matrix_range.start1() + i) * gpu_matrix_range.internal_size2() + gpu_matrix_range.start2();
      vcl_size_t num_entries = gpu_matrix_range.size2();
      viennacl::backend::memory_read(gpu_matrix_range.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));
      //std::cout << "Strided copy worked!" << std::endl;

      for (vcl_size_t j=0; j < gpu_matrix_range.size2(); ++j)
        detail::matrix_access<NumericT>(cpu_matrix, i, j) = entries[j];
    }
  }
  else
  {
    //full block can be copied:
    std::vector<NumericT> entries(gpu_matrix_range.size1()*gpu_matrix_range.internal_size2());

    vcl_size_t start_offset = gpu_matrix_range.start1() * gpu_matrix_range.internal_size2();
    viennacl::backend::memory_read(gpu_matrix_range.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*entries.size(), &(entries[0]));
    //std::cout << "Block copy worked!" << std::endl;

    for (vcl_size_t i=0; i < gpu_matrix_range.size1(); ++i)
      for (vcl_size_t j=0; j < gpu_matrix_range.size2(); ++j)
        detail::matrix_access<NumericT>(cpu_matrix, i, j) = entries[i*gpu_matrix_range.internal_size2() + j];
  }

}


//column_major:
template<typename CPUMatrixT, typename NumericT>
void copy(matrix_range<matrix<NumericT, column_major, 1> > const & gpu_matrix_range,
          CPUMatrixT & cpu_matrix)
{
  assert(    (viennacl::traits::size1(cpu_matrix) == gpu_matrix_range.size1())
          && (viennacl::traits::size2(cpu_matrix) == gpu_matrix_range.size2())
          && bool("Matrix size mismatch!"));

  if ( gpu_matrix_range.start1() != 0)
  {
    std::vector<NumericT> entries(gpu_matrix_range.size1());

    //copy each stride separately:
    for (vcl_size_t j=0; j < gpu_matrix_range.size2(); ++j)
    {
      vcl_size_t start_offset = (gpu_matrix_range.start2() + j) * gpu_matrix_range.internal_size1() + gpu_matrix_range.start1();
      vcl_size_t num_entries = gpu_matrix_range.size1();
      viennacl::backend::memory_read(gpu_matrix_range.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));
      //std::cout << "Strided copy worked!" << std::endl;

      for (vcl_size_t i=0; i < gpu_matrix_range.size1(); ++i)
        detail::matrix_access<NumericT>(cpu_matrix, i, j) = entries[i];
    }
  }
  else
  {
    //full block can be copied:
    std::vector<NumericT> entries(gpu_matrix_range.internal_size1()*gpu_matrix_range.size2());

    //copy each stride separately:
    vcl_size_t start_offset = gpu_matrix_range.start2() * gpu_matrix_range.internal_size1();
    vcl_size_t num_entries = gpu_matrix_range.internal_size1() * gpu_matrix_range.size2();
    viennacl::backend::memory_read(gpu_matrix_range.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));
    //std::cout << "Block copy worked!" << std::endl;

    for (vcl_size_t i=0; i < gpu_matrix_range.size1(); ++i)
      for (vcl_size_t j=0; j < gpu_matrix_range.size2(); ++j)
        detail::matrix_access<NumericT>(cpu_matrix, i, j) = entries[i + j*gpu_matrix_range.internal_size1()];
  }

}


//
// Convenience function
//
template<typename MatrixType>
matrix_range<MatrixType> project(MatrixType const & A, viennacl::range const & r1, viennacl::range const & r2)
{
  assert(r1.size() <= A.size1() && r2.size() <= A.size2() && bool("Size of range invalid!"));

  return matrix_range<MatrixType>(A, r1, r2);
}


template<typename MatrixType>
matrix_range<MatrixType> project(matrix_range<MatrixType> const & A, viennacl::range const & r1, viennacl::range const & r2)
{
  assert(r1.size() <= A.size1() && r2.size() <= A.size2() && bool("Size of range invalid!"));

  return matrix_range<MatrixType>(A, r1, r2);
}




//
//
//
/////////////////////////////// Slice /////////////////////////////////////////////
//
//
//





/** @brief Class for representing strided submatrices of a bigger matrix A.
  *
  * In MATLAB notation, this could for example refer to the submatrix A(3:2:8, 6:3:16) of a matrix A.
  */
template<typename MatrixType>
class matrix_slice : public matrix_base<typename MatrixType::cpu_value_type>
{
  typedef matrix_base<typename MatrixType::cpu_value_type>    base_type;
  typedef matrix_slice<MatrixType>                            self_type;

public:

  typedef typename MatrixType::value_type     value_type;
  typedef typename MatrixType::handle_type    handle_type;
  typedef typename viennacl::result_of::cpu_value_type<value_type>::type    cpu_value_type;
  typedef range::size_type                    size_type;
  typedef range::difference_type              difference_type;
  typedef value_type                          reference;
  typedef const value_type &                  const_reference;

  matrix_slice(MatrixType const & A,
               slice const & row_slice,
               slice const & col_slice) : base_type(const_cast<handle_type &>(A.handle()),
                                                    row_slice.size(), row_slice.start() * A.stride1() + A.start1(), row_slice.stride() * A.stride1(), A.internal_size1(),
                                                    col_slice.size(), col_slice.start() * A.stride2() + A.start2(), col_slice.stride() * A.stride2(), A.internal_size2(),
                                                    A.row_major()) {}

  matrix_slice(self_type const & A,
               slice const & row_slice,
               slice const & col_slice) : base_type(const_cast<handle_type &>(A.handle()),
                                                    row_slice.size(), row_slice.start() * A.stride1() + A.start1(), row_slice.stride() * A.stride1(), A.internal_size1(),
                                                    col_slice.size(), col_slice.start() * A.stride2() + A.start2(), col_slice.stride() * A.stride2(), A.internal_size2(),
                                                    A.row_major()) {}


  matrix_slice(self_type const & other) : base_type(const_cast<handle_type &>(other.handle()),
                                                    other.size1(), other.start1(), other.stride1(), other.internal_size1(),
                                                    other.size2(), other.start2(), other.stride2(), other.internal_size2(),
                                                    other.row_major()) {}

  using base_type::operator=;

  // the following are needed for Visual Studio:
  template<typename OtherNumericT, typename F>
  base_type & operator=(viennacl::matrix<OtherNumericT, F> const & B)                          { return base_type::operator=(static_cast<viennacl::matrix_base<OtherNumericT> const &>(B)); }

  template<typename OtherNumericT, typename F>
  base_type & operator=(viennacl::matrix_range<viennacl::matrix<OtherNumericT, F> > const & B) { return base_type::operator=(static_cast<viennacl::matrix_base<OtherNumericT> const &>(B)); }

  template<typename OtherNumericT, typename F>
  base_type & operator=(viennacl::matrix_slice<viennacl::matrix<OtherNumericT, F> > const & B) { return base_type::operator=(static_cast<viennacl::matrix_base<OtherNumericT> const &>(B)); }
};

template<typename MatrixType>
class matrix_slice<matrix_range<MatrixType> > : public matrix_base<typename MatrixType::cpu_value_type>
{
  typedef matrix_base<typename MatrixType::cpu_value_type>    base_type;
public:
  typedef typename MatrixType::handle_type    handle_type;

  matrix_slice(MatrixType const & A,
               slice const & row_slice,
               slice const & col_slice) : base_type(const_cast<handle_type &>(A.handle()),
                                                    row_slice.size(), row_slice.start() * A.stride1() + A.start1(), row_slice.stride() * A.stride1(), A.internal_size1(),
                                                    col_slice.size(), col_slice.start() * A.stride2() + A.start2(), col_slice.stride() * A.stride2(), A.internal_size2(),
                                                    A.row_major()) {}

  matrix_slice(matrix_slice<MatrixType> const & A,
               slice const & row_slice,
               slice const & col_slice) : base_type(const_cast<handle_type &>(A.handle()),
                                                    row_slice.size(), row_slice.start() * A.stride1() + A.start1(), row_slice.stride() * A.stride1(), A.internal_size1(),
                                                    col_slice.size(), col_slice.start() * A.stride2() + A.start2(), col_slice.stride() * A.stride2(), A.internal_size2(),
                                                    A.row_major()) {}
};


/////////////////////////////////////////////////////////////
///////////////////////// CPU to GPU ////////////////////////
/////////////////////////////////////////////////////////////

//row_major:
template<typename CPUMatrixT, typename NumericT>
void copy(const CPUMatrixT & cpu_matrix,
          matrix_slice<matrix<NumericT, row_major, 1> > & gpu_matrix_slice )
{
  assert(    (viennacl::traits::size1(cpu_matrix) == gpu_matrix_slice.size1())
          && (viennacl::traits::size2(cpu_matrix) == gpu_matrix_slice.size2())
          && bool("Matrix size mismatch!"));

  if ( (gpu_matrix_slice.size1() > 0) && (gpu_matrix_slice.size1() > 0) )
  {
    vcl_size_t num_entries = gpu_matrix_slice.size2() * gpu_matrix_slice.stride2(); //no. of entries per stride

    std::vector<NumericT> entries(num_entries);

    //copy each stride separately:
    for (vcl_size_t i=0; i < gpu_matrix_slice.size1(); ++i)
    {
      vcl_size_t start_offset = (gpu_matrix_slice.start1() + i * gpu_matrix_slice.stride1()) * gpu_matrix_slice.internal_size2() + gpu_matrix_slice.start2();
      viennacl::backend::memory_read(gpu_matrix_slice.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));

      for (vcl_size_t j=0; j < gpu_matrix_slice.size2(); ++j)
        entries[j * gpu_matrix_slice.stride2()] = detail::matrix_access<NumericT>(cpu_matrix, i, j);

      viennacl::backend::memory_write(gpu_matrix_slice.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));
    }
  }
}

//column_major:
template<typename CPUMatrixT, typename NumericT>
void copy(const CPUMatrixT & cpu_matrix,
          matrix_slice<matrix<NumericT, column_major, 1> > & gpu_matrix_slice )
{
  assert(    (viennacl::traits::size1(cpu_matrix) == gpu_matrix_slice.size1())
          && (viennacl::traits::size2(cpu_matrix) == gpu_matrix_slice.size2())
          && bool("Matrix size mismatch!"));


  if ( (gpu_matrix_slice.size1() > 0) && (gpu_matrix_slice.size1() > 0) )
  {
    vcl_size_t num_entries = gpu_matrix_slice.size1() * gpu_matrix_slice.stride1(); //no. of entries per stride

    std::vector<NumericT> entries(num_entries);

    //copy each column stride separately:
    for (vcl_size_t j=0; j < gpu_matrix_slice.size2(); ++j)
    {
      vcl_size_t start_offset = gpu_matrix_slice.start1() + (gpu_matrix_slice.start2() + j * gpu_matrix_slice.stride2()) * gpu_matrix_slice.internal_size1();

      viennacl::backend::memory_read(gpu_matrix_slice.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));

      for (vcl_size_t i=0; i < gpu_matrix_slice.size1(); ++i)
        entries[i * gpu_matrix_slice.stride1()] = detail::matrix_access<NumericT>(cpu_matrix, i, j);

      viennacl::backend::memory_write(gpu_matrix_slice.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));
    }
  }

}


/////////////////////////////////////////////////////////////
///////////////////////// GPU to CPU ////////////////////////
/////////////////////////////////////////////////////////////


//row_major:
template<typename CPUMatrixT, typename NumericT>
void copy(matrix_slice<matrix<NumericT, row_major, 1> > const & gpu_matrix_slice,
          CPUMatrixT & cpu_matrix)
{
  assert(    (viennacl::traits::size1(cpu_matrix) == gpu_matrix_slice.size1())
          && (viennacl::traits::size2(cpu_matrix) == gpu_matrix_slice.size2())
          && bool("Matrix size mismatch!"));

  if ( (gpu_matrix_slice.size1() > 0) && (gpu_matrix_slice.size1() > 0) )
  {
    vcl_size_t num_entries = gpu_matrix_slice.size2() * gpu_matrix_slice.stride2(); //no. of entries per stride

    std::vector<NumericT> entries(num_entries);

    //copy each stride separately:
    for (vcl_size_t i=0; i < gpu_matrix_slice.size1(); ++i)
    {
      vcl_size_t start_offset = (gpu_matrix_slice.start1() + i * gpu_matrix_slice.stride1()) * gpu_matrix_slice.internal_size2() + gpu_matrix_slice.start2();

      viennacl::backend::memory_read(gpu_matrix_slice.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));

      for (vcl_size_t j=0; j < gpu_matrix_slice.size2(); ++j)
        detail::matrix_access<NumericT>(cpu_matrix, i, j) = entries[j * gpu_matrix_slice.stride2()];
    }
  }

}


//column_major:
template<typename CPUMatrixT, typename NumericT>
void copy(matrix_slice<matrix<NumericT, column_major, 1> > const & gpu_matrix_slice,
          CPUMatrixT & cpu_matrix)
{
  assert(    (viennacl::traits::size1(cpu_matrix) == gpu_matrix_slice.size1())
          && (viennacl::traits::size2(cpu_matrix) == gpu_matrix_slice.size2())
          && bool("Matrix size mismatch!"));

  if ( (gpu_matrix_slice.size1() > 0) && (gpu_matrix_slice.size1() > 0) )
  {
    vcl_size_t num_entries = gpu_matrix_slice.size1() * gpu_matrix_slice.stride1(); //no. of entries per stride

    std::vector<NumericT> entries(num_entries);

    //copy each column stride separately:
    for (vcl_size_t j=0; j < gpu_matrix_slice.size2(); ++j)
    {
      vcl_size_t start_offset = gpu_matrix_slice.start1() + (gpu_matrix_slice.start2() + j * gpu_matrix_slice.stride2()) * gpu_matrix_slice.internal_size1();

      viennacl::backend::memory_read(gpu_matrix_slice.handle(), sizeof(NumericT)*start_offset, sizeof(NumericT)*num_entries, &(entries[0]));

      for (vcl_size_t i=0; i < gpu_matrix_slice.size1(); ++i)
        detail::matrix_access<NumericT>(cpu_matrix, i, j) = entries[i * gpu_matrix_slice.stride1()];
    }
  }

}


//
// Convenience function
//
template<typename MatrixType>
matrix_slice<MatrixType> project(MatrixType const & A, viennacl::slice const & r1, viennacl::slice const & r2)
{
  assert(r1.size() <= A.size1() && r2.size() <= A.size2() && bool("Size of slice invalid!"));

  return matrix_slice<MatrixType>(A, r1, r2);
}

template<typename MatrixType>
matrix_slice<MatrixType> project(matrix_range<MatrixType> const & A, viennacl::slice const & r1, viennacl::slice const & r2)
{
  assert(r1.size() <= A.size1() && r2.size() <= A.size2() && bool("Size of slice invalid!"));

  return matrix_slice<MatrixType>(A, r1, r2);
}

template<typename MatrixType>
matrix_slice<MatrixType> project(matrix_slice<MatrixType> const & A, viennacl::slice const & r1, viennacl::slice const & r2)
{
  assert(r1.size() <= A.size1() && r2.size() <= A.size2() && bool("Size of slice invalid!"));

  return matrix_slice<MatrixType>(A, r1, r2);
}

// TODO: Allow mix of range/slice

}

#endif
