#ifndef VIENNACL_IO_MATRIX_MARKET_HPP
#define VIENNACL_IO_MATRIX_MARKET_HPP

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


/** @file matrix_market.hpp
    @brief A reader and writer for the matrix market format is implemented here
*/

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <cctype>
#include "viennacl/tools/adapter.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/fill.hpp"

namespace viennacl
{
namespace io
{
//helper
namespace detail
{
  inline void trim(char * buffer, long max_size)
  {
    //trim at beginning of string
    long start = 0;
    for (long i=0; i<max_size; ++i)
    {
      if (buffer[i] == ' ')
        ++start;
      else
        break;
    }

    //trim at end of string
    long stop = start;
    for (long i=stop; i<max_size; ++i)
    {
      if (buffer[i] == 0)   //end of string
        break;

      if (buffer[i] != ' ')
        stop = i;
    }

    for (long i=0; i<=stop - start; ++i)
    {
      buffer[i] = buffer[start + i];
    }

    if (buffer[0] != ' ')
      buffer[stop - start + 1] = 0; //terminate string
    else
      buffer[0] = 0;
  }

  inline std::string tolower(std::string & s)
  {
    std::transform(s.begin(), s.end(), s.begin(), static_cast < int(*)(int) > (std::tolower));
    return s;
  }



} //namespace

///////// reader ////////////

/** @brief Reads a sparse or dense matrix from a file (MatrixMarket format)
*
* Note: If the matrix in the MatrixMarket file is complex, only the real-valued part is loaded!
*
* @param mat The matrix that is to be read
* @param file Filename from which the matrix should be read
* @param index_base The index base, typically 1
* @tparam MatrixT A generic matrix type. Type requirements: size1() returns number of rows, size2() returns number columns, operator() writes array entries, resize() allows resizing the matrix.
* @return Returns nonzero if file is read correctly
*/
template<typename MatrixT>
long read_matrix_market_file_impl(MatrixT & mat,
                                  const char * file,
                                  long index_base)
{
  typedef typename viennacl::result_of::cpu_value_type<typename viennacl::result_of::value_type<MatrixT>::type>::type    ScalarT;

  //std::cout << "Reading matrix market file" << std::endl;
  char buffer[1025];
  std::ifstream reader(file);
  std::string token;
  long linenum = 0;
  bool symmetric = false;
  bool dense_format = false;
  bool is_header = true;
  bool pattern_matrix = false;
  //bool is_complex = false;
  long cur_row = 0;
  long cur_col = 0;
  long valid_entries = 0;
  long nnz = 0;


  if (!reader){
    std::cerr << "ViennaCL: Matrix Market Reader: Cannot open file " << file << std::endl;
    return EXIT_FAILURE;
  }

  while (reader.good())
  {
    // get a non-empty line
    do
    {
      reader.getline(buffer, 1024);
      ++linenum;
      detail::trim(buffer, 1024);
    }
    while (reader.good() && buffer[0] == 0);

    if (buffer[0] == '%')
    {
      if (buffer[1] == '%')
      {
        //parse header:
        std::stringstream line(std::string(buffer + 2));
        line >> token;
        if (detail::tolower(token) != "matrixmarket")
        {
          std::cerr << "Error in file " << file << " at line " << linenum << " in file " << file << ": Expected 'MatrixMarket', got '" << token << "'" << std::endl;
          return 0;
        }

        line >> token;
        if (detail::tolower(token) != "matrix")
        {
          std::cerr << "Error in file " << file << " at line " << linenum << " in file " << file << ": Expected 'matrix', got '" << token << "'" << std::endl;
          return 0;
        }

        line >> token;
        if (detail::tolower(token) != "coordinate")
        {
          if (detail::tolower(token) == "array")
          {
            dense_format = true;
            std::cerr << "Error in file " << file << " at line " << linenum << " in file " << file << ": 'array' type is not supported yet!" << std::endl;
            return 0;
          }
          else
          {
            std::cerr << "Error in file " << file << " at line " << linenum << " in file " << file << ": Expected 'array' or 'coordinate', got '" << token << "'" << std::endl;
            return 0;
          }
        }

        line >> token;
        if (detail::tolower(token) == "pattern")
        {
          pattern_matrix = true;
        }
        else if (detail::tolower(token) == "complex")
        {
          //is_complex = true;
        }
        else if (detail::tolower(token) != "real")
        {
          std::cerr << "Error in file " << file << ": The MatrixMarket reader provided with ViennaCL supports only real valued floating point arithmetic or pattern type matrices." << std::endl;
          return 0;
        }

        line >> token;
        if (detail::tolower(token) == "general"){ }
        else if (detail::tolower(token) == "symmetric"){ symmetric = true; }
        else
        {
          std::cerr << "Error in file " << file << ": The MatrixMarket reader provided with ViennaCL supports only general or symmetric matrices." << std::endl;
          return 0;
        }

      }
    }
    else
    {
      std::stringstream line(std::stringstream::in | std::stringstream::out);
      line << std::string(buffer);

      if (is_header)
      {
        //read header line
        vcl_size_t rows;
        vcl_size_t cols;

        if (line.good())
          line >> rows;
        else
        {
          std::cerr << "Error in file " << file << ": Could not get matrix dimensions (rows) in line " << linenum << std::endl;
          return 0;
        }

        if (line.good())
          line >> cols;
        else
        {
          std::cerr << "Error in file " << file << ": Could not get matrix dimensions (columns) in line " << linenum << std::endl;
          return 0;
        }
        if (!dense_format)
        {
          if (line.good())
            line >> nnz;
          else
          {
            std::cerr << "Error in file " << file << ": Could not get matrix dimensions (columns) in line " << linenum << std::endl;
            return 0;
          }
        }

        if (rows > 0 && cols > 0)
          viennacl::traits::resize(mat, rows, cols);

        is_header = false;
      }
      else
      {
        //read data
        if (dense_format)
        {
          ScalarT value;
          line >> value;
          viennacl::traits::fill(mat, static_cast<vcl_size_t>(cur_row), static_cast<vcl_size_t>(cur_col), value);

          if (++cur_row == static_cast<long>(viennacl::traits::size1(mat)))
          {
            //next column
            ++cur_col;
            cur_row = 0;
          }
        }
        else //sparse format
        {
          long row;
          long col;
          ScalarT value = ScalarT(1);

          //parse data:
          if (line.good())
            line >> row;
          else
          {
            std::cerr << "Error in file " << file << ": Parse error for matrix row entry in line " << linenum << std::endl;
            return 0;
          }

          if (line.good())
            line >> col;
          else
          {
            std::cerr << "Error in file " << file << ": Parse error for matrix col entry in line " << linenum << std::endl;
            return 0;
          }

          //take index_base base into account:
          row -= index_base;
          col -= index_base;

          if (!pattern_matrix) // value for pattern matrix is implicitly 1, so we only need to read data for 'normal' matrices
          {
            if (line.good())
            {
                line >> value;
            }
            else
            {
              std::cerr << "Error in file " << file << ": Parse error for matrix entry in line " << linenum << std::endl;
              return 0;
            }
          }

          if (row >= static_cast<long>(viennacl::traits::size1(mat)) || row < 0)
          {
            std::cerr << "Error in file " << file << " at line " << linenum << ": Row index out of bounds: " << row << " (matrix dim: " << viennacl::traits::size1(mat) << " x " << viennacl::traits::size2(mat) << ")" << std::endl;
            return 0;
          }

          if (col >= static_cast<long>(viennacl::traits::size2(mat)) || col < 0)
          {
            std::cerr << "Error in file " << file << " at line " << linenum << ": Column index out of bounds: " << col << " (matrix dim: " << viennacl::traits::size1(mat) << " x " << viennacl::traits::size2(mat) << ")" << std::endl;
            return 0;
          }

          viennacl::traits::fill(mat, static_cast<vcl_size_t>(row), static_cast<vcl_size_t>(col), value); //basically equivalent to mat(row, col) = value;
          if (symmetric)
            viennacl::traits::fill(mat, static_cast<vcl_size_t>(col), static_cast<vcl_size_t>(row), value); //basically equivalent to mat(col, row) = value;

          if (++valid_entries == nnz)
            break;

        } //else dense_format
      }
    }
  }

  //std::cout << linenum << " lines read." << std::endl;
  reader.close();
  return linenum;
}


/** @brief Reads a sparse matrix from a file (MatrixMarket format)
*
* @param mat The matrix that is to be read (ublas-types and std::vector< std::map <unsigned int, ScalarT> > are supported)
* @param file The filename
* @param index_base The index base, typically 1
* @tparam MatrixT A generic matrix type. Type requirements: size1() returns number of rows, size2() returns number columns, operator() writes array entries, resize() allows resizing the matrix.
* @return Returns nonzero if file is read correctly
*/
template<typename MatrixT>
long read_matrix_market_file(MatrixT & mat,
                             const char * file,
                             long index_base = 1)
{
  return read_matrix_market_file_impl(mat, file, index_base);
}

template<typename MatrixT>
long read_matrix_market_file(MatrixT & mat,
                             const std::string & file,
                             long index_base = 1)
{
  return read_matrix_market_file_impl(mat, file.c_str(), index_base);
}

template<typename ScalarT>
long read_matrix_market_file(std::vector< std::map<unsigned int, ScalarT> > & mat,
                             const char * file,
                             long index_base = 1)
{
  viennacl::tools::sparse_matrix_adapter<ScalarT> adapted_matrix(mat);
  return read_matrix_market_file_impl(adapted_matrix, file, index_base);
}

template<typename ScalarT>
long read_matrix_market_file(std::vector< std::map<unsigned int, ScalarT> > & mat,
                             const std::string & file,
                             long index_base = 1)
{
  viennacl::tools::sparse_matrix_adapter<ScalarT> adapted_matrix(mat);
  return read_matrix_market_file_impl(adapted_matrix, file.c_str(), index_base);
}


////////// writer /////////////
template<typename MatrixT>
void write_matrix_market_file_impl(MatrixT const & mat, const char * file, long index_base)
{
  std::ofstream writer(file);

  long num_entries = 0;
  for (typename MatrixT::const_iterator1 row_it = mat.begin1();
       row_it != mat.end1();
       ++row_it)
    for (typename MatrixT::const_iterator2 col_it = row_it.begin();
         col_it != row_it.end();
         ++col_it)
      ++num_entries;

  writer << "%%MatrixMarket matrix coordinate real general" << std::endl;
  writer << mat.size1() << " " << mat.size2() << " " << num_entries << std::endl;

  for (typename MatrixT::const_iterator1 row_it = mat.begin1();
       row_it != mat.end1();
       ++row_it)
    for (typename MatrixT::const_iterator2 col_it = row_it.begin();
         col_it != row_it.end();
         ++col_it)
      writer << col_it.index1() + index_base << " " << col_it.index2() + index_base << " " << *col_it << std::endl;

  writer.close();
}

template<typename ScalarT>
void write_matrix_market_file(std::vector< std::map<unsigned int, ScalarT> > const & mat,
                              const char * file,
                              long index_base = 1)
{
  viennacl::tools::const_sparse_matrix_adapter<ScalarT> adapted_matrix(mat);
  return write_matrix_market_file_impl(adapted_matrix, file, index_base);
}

template<typename ScalarT>
void write_matrix_market_file(std::vector< std::map<unsigned int, ScalarT> > const & mat,
                              const std::string & file,
                              long index_base = 1)
{
  viennacl::tools::const_sparse_matrix_adapter<ScalarT> adapted_matrix(mat);
  return write_matrix_market_file_impl(adapted_matrix, file.c_str(), index_base);
}

/** @brief Writes a sparse matrix to a file (MatrixMarket format)
*
* @param mat The matrix that is to be read (ublas-types and std::vector< std::map <unsigned int, ScalarT> > are supported)
* @param file The filename
* @param index_base The index base, typically 1
* @tparam MatrixT A generic matrix type. Type requirements: size1() returns number of rows, size2() returns number columns, operator() writes array entries, resize() allows resizing the matrix.
* @return Returns nonzero if file is read correctly
*/
template<typename MatrixT>
void write_matrix_market_file(MatrixT const & mat,
                              const std::string & file,
                              long index_base = 1)
{
  write_matrix_market_file_impl(mat, file.c_str(), index_base);
}


} //namespace io
} //namespace viennacl

#endif
