#ifndef VIENNACL_FORWARDS_H
#define VIENNACL_FORWARDS_H

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


/** @file viennacl/forwards.h
    @brief This file provides the forward declarations for the main types used within ViennaCL
*/

/**
 @mainpage Main Page

 Here you can find all the documentation on how to use the GPU-accelerated linear algebra library ViennaCL.
 The formerly separate \ref usermanual "user manual" is no longer available as a standalone PDF, but all integrated into the HTML-based documentation.
 Please use the navigation panel on the left to access the desired information.

 Quick links:
     - \ref manual-installation "Installation and building the examples"
     - \ref manual-types        "Basic types"
     - \ref manual-operations   "Basic operations"
     - \ref manual-algorithms   "Algorithms"


 -----------------------------------
 \htmlonly
 <div style="align: right; width: 100%">
 <a href="http://www.tuwien.ac.at/"><img src="tuwien.png"></a>
 <a href="http://www.iue.tuwien.ac.at/"><img src="iue.png"></a>
 <a href="http://www.asc.tuwien.ac.at/"><img src="asc.png"></a>
 </div>
 \endhtmlonly
*/


//compatibility defines:
#ifdef VIENNACL_HAVE_UBLAS
  #define VIENNACL_WITH_UBLAS
#endif

#ifdef VIENNACL_HAVE_EIGEN
  #define VIENNACL_WITH_EIGEN
#endif

#ifdef VIENNACL_HAVE_MTL4
  #define VIENNACL_WITH_MTL4
#endif

#include <cstddef>
#include <cassert>
#include <string>
#include <stdexcept>

#include "viennacl/meta/enable_if.hpp"
#include "viennacl/version.hpp"

/** @brief Main namespace in ViennaCL. Holds all the basic types such as vector, matrix, etc. and defines operations upon them. */
namespace viennacl
{
  typedef std::size_t                                       vcl_size_t;
  typedef std::ptrdiff_t                                    vcl_ptrdiff_t;



  /** @brief A tag class representing assignment */
  struct op_assign {};
  /** @brief A tag class representing inplace addition */
  struct op_inplace_add {};
  /** @brief A tag class representing inplace subtraction */
  struct op_inplace_sub {};

  /** @brief A tag class representing addition */
  struct op_add {};
  /** @brief A tag class representing subtraction */
  struct op_sub {};
  /** @brief A tag class representing multiplication by a scalar */
  struct op_mult {};
  /** @brief A tag class representing matrix-vector products and element-wise multiplications*/
  struct op_prod {};
  /** @brief A tag class representing matrix-matrix products */
  struct op_mat_mat_prod {};
  /** @brief A tag class representing division */
  struct op_div {};
  /** @brief A tag class representing the power function */
  struct op_pow {};

  /** @brief A tag class representing equality */
 struct op_eq {};
 /** @brief A tag class representing inequality */
 struct op_neq {};
 /** @brief A tag class representing greater-than */
 struct op_greater {};
 /** @brief A tag class representing less-than */
 struct op_less {};
 /** @brief A tag class representing greater-than-or-equal-to */
 struct op_geq {};
 /** @brief A tag class representing less-than-or-equal-to */
 struct op_leq {};

  /** @brief A tag class representing the summation of a vector */
  struct op_sum {};

  /** @brief A tag class representing the summation of all rows of a matrix */
  struct op_row_sum {};

  /** @brief A tag class representing the summation of all columns of a matrix */
  struct op_col_sum {};

  /** @brief A tag class representing element-wise casting operations on vectors and matrices */
  template<typename OP>
  struct op_element_cast {};

  /** @brief A tag class representing element-wise binary operations (like multiplication) on vectors or matrices */
  template<typename OP>
  struct op_element_binary {};

  /** @brief A tag class representing element-wise unary operations (like sin()) on vectors or matrices */
  template<typename OP>
  struct op_element_unary {};

  /** @brief A tag class representing the modulus function for integers */
  struct op_abs {};
  /** @brief A tag class representing the acos() function */
  struct op_acos {};
  /** @brief A tag class representing the asin() function */
  struct op_asin {};
  /** @brief A tag class for representing the argmax() function */
  struct op_argmax {};
  /** @brief A tag class for representing the argmin() function */
  struct op_argmin {};
  /** @brief A tag class representing the atan() function */
  struct op_atan {};
  /** @brief A tag class representing the atan2() function */
  struct op_atan2 {};
  /** @brief A tag class representing the ceil() function */
  struct op_ceil {};
  /** @brief A tag class representing the cos() function */
  struct op_cos {};
  /** @brief A tag class representing the cosh() function */
  struct op_cosh {};
  /** @brief A tag class representing the exp() function */
  struct op_exp {};
  /** @brief A tag class representing the fabs() function */
  struct op_fabs {};
  /** @brief A tag class representing the fdim() function */
  struct op_fdim {};
  /** @brief A tag class representing the floor() function */
  struct op_floor {};
  /** @brief A tag class representing the fmax() function */
  struct op_fmax {};
  /** @brief A tag class representing the fmin() function */
  struct op_fmin {};
  /** @brief A tag class representing the fmod() function */
  struct op_fmod {};
  /** @brief A tag class representing the log() function */
  struct op_log {};
  /** @brief A tag class representing the log10() function */
  struct op_log10 {};
  /** @brief A tag class representing the sin() function */
  struct op_sin {};
  /** @brief A tag class representing the sinh() function */
  struct op_sinh {};
  /** @brief A tag class representing the sqrt() function */
  struct op_sqrt {};
  /** @brief A tag class representing the tan() function */
  struct op_tan {};
  /** @brief A tag class representing the tanh() function */
  struct op_tanh {};

  /** @brief A tag class representing the (off-)diagonal of a matrix */
  struct op_matrix_diag {};

  /** @brief A tag class representing a matrix given by a vector placed on a certain (off-)diagonal */
  struct op_vector_diag {};

  /** @brief A tag class representing the extraction of a matrix row to a vector */
  struct op_row {};

  /** @brief A tag class representing the extraction of a matrix column to a vector */
  struct op_column {};

  /** @brief A tag class representing inner products of two vectors */
  struct op_inner_prod {};

  /** @brief A tag class representing the 1-norm of a vector */
  struct op_norm_1 {};

  /** @brief A tag class representing the 2-norm of a vector */
  struct op_norm_2 {};

  /** @brief A tag class representing the inf-norm of a vector */
  struct op_norm_inf {};

  /** @brief A tag class representing the maximum of a vector */
  struct op_max {};

  /** @brief A tag class representing the minimum of a vector */
  struct op_min {};


  /** @brief A tag class representing the Frobenius-norm of a matrix */
  struct op_norm_frobenius {};

  /** @brief A tag class representing transposed matrices */
  struct op_trans {};

  /** @brief A tag class representing sign flips (for scalars only. Vectors and matrices use the standard multiplication by the scalar -1.0) */
  struct op_flip_sign {};

  //forward declaration of basic types:
  template<class TYPE>
  class scalar;

  template<typename LHS, typename RHS, typename OP>
  class scalar_expression;

  template<typename SCALARTYPE>
  class entry_proxy;

  template<typename SCALARTYPE>
  class const_entry_proxy;

  template<typename LHS, typename RHS, typename OP>
  class vector_expression;

  template<class SCALARTYPE, unsigned int ALIGNMENT>
  class vector_iterator;

  template<class SCALARTYPE, unsigned int ALIGNMENT>
  class const_vector_iterator;

  template<typename SCALARTYPE>
  class implicit_vector_base;

  template<typename SCALARTYPE>
  struct zero_vector;

  template<typename SCALARTYPE>
  struct unit_vector;

  template<typename SCALARTYPE>
  struct one_vector;

  template<typename SCALARTYPE>
  struct scalar_vector;

  template<class SCALARTYPE, typename SizeType = vcl_size_t, typename DistanceType = vcl_ptrdiff_t>
  class vector_base;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class vector;

  template<typename ScalarT>
  class vector_tuple;

  //the following forwards are needed for GMRES
  template<typename SCALARTYPE, unsigned int ALIGNMENT, typename CPU_ITERATOR>
  void copy(CPU_ITERATOR const & cpu_begin,
            CPU_ITERATOR const & cpu_end,
            vector_iterator<SCALARTYPE, ALIGNMENT> gpu_begin);

  template<typename SCALARTYPE, unsigned int ALIGNMENT_SRC, unsigned int ALIGNMENT_DEST>
  void copy(const_vector_iterator<SCALARTYPE, ALIGNMENT_SRC> const & gpu_src_begin,
            const_vector_iterator<SCALARTYPE, ALIGNMENT_SRC> const & gpu_src_end,
            vector_iterator<SCALARTYPE, ALIGNMENT_DEST> gpu_dest_begin);

  template<typename SCALARTYPE, unsigned int ALIGNMENT_SRC, unsigned int ALIGNMENT_DEST>
  void copy(const_vector_iterator<SCALARTYPE, ALIGNMENT_SRC> const & gpu_src_begin,
            const_vector_iterator<SCALARTYPE, ALIGNMENT_SRC> const & gpu_src_end,
            const_vector_iterator<SCALARTYPE, ALIGNMENT_DEST> gpu_dest_begin);

  template<typename SCALARTYPE, unsigned int ALIGNMENT, typename CPU_ITERATOR>
  void fast_copy(const const_vector_iterator<SCALARTYPE, ALIGNMENT> & gpu_begin,
                 const const_vector_iterator<SCALARTYPE, ALIGNMENT> & gpu_end,
                 CPU_ITERATOR cpu_begin );

  template<typename CPU_ITERATOR, typename SCALARTYPE, unsigned int ALIGNMENT>
  void fast_copy(CPU_ITERATOR const & cpu_begin,
                  CPU_ITERATOR const & cpu_end,
                  vector_iterator<SCALARTYPE, ALIGNMENT> gpu_begin);


  /** @brief Tag class for indicating row-major layout of a matrix. Not passed to the matrix directly, see row_major type. */
  struct row_major_tag {};
  /** @brief Tag class for indicating column-major layout of a matrix. Not passed to the matrix directly, see row_major type. */
  struct column_major_tag {};

  /** @brief A tag for row-major storage of a dense matrix. */
  struct row_major
  {
    typedef row_major_tag         orientation_category;

    /** @brief Returns the memory offset for entry (i,j) of a dense matrix.
    *
    * @param i   row index
    * @param j   column index
    * @param num_cols  number of entries per column (including alignment)
    */
    static vcl_size_t mem_index(vcl_size_t i, vcl_size_t j, vcl_size_t /* num_rows */, vcl_size_t num_cols)
    {
      return i * num_cols + j;
    }
  };

  /** @brief A tag for column-major storage of a dense matrix. */
  struct column_major
  {
    typedef column_major_tag         orientation_category;

    /** @brief Returns the memory offset for entry (i,j) of a dense matrix.
    *
    * @param i   row index
    * @param j   column index
    * @param num_rows  number of entries per row (including alignment)
    */
    static vcl_size_t mem_index(vcl_size_t i, vcl_size_t j, vcl_size_t num_rows, vcl_size_t /* num_cols */)
    {
      return i + j * num_rows;
    }
  };

  struct row_iteration;
  struct col_iteration;

  template<typename LHS, typename RHS, typename OP>
  class matrix_expression;

  class context;

  enum memory_types
  {
    MEMORY_NOT_INITIALIZED
    , MAIN_MEMORY
    , OPENCL_MEMORY
    , CUDA_MEMORY
  };

  namespace backend
  {
    class mem_handle;
  }

  //
  // Matrix types:
  //
  static const vcl_size_t dense_padding_size = 128;

  /** @brief A dense matrix class
  *
  * @tparam SCALARTYPE   The underlying scalar type (either float or double)
  * @tparam ALIGNMENT   The internal memory size is given by (size()/ALIGNMENT + 1) * ALIGNMENT. ALIGNMENT must be a power of two. Best values or usually 4, 8 or 16, higher values are usually a waste of memory.
  */
  template<typename ROWCOL, typename MATRIXTYPE>
  class matrix_iterator;

  template<class SCALARTYPE, typename SizeType = vcl_size_t, typename DistanceType = vcl_ptrdiff_t>
  class matrix_base;

  template<class SCALARTYPE, typename F = row_major, unsigned int ALIGNMENT = 1>
  class matrix;

  template<typename SCALARTYPE>
  class implicit_matrix_base;

  template<class SCALARTYPE>
  class identity_matrix;

  template<class SCALARTYPE>
  class zero_matrix;

  template<class SCALARTYPE>
  class scalar_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class compressed_matrix;

  template<class SCALARTYPE>
  class compressed_compressed_matrix;


  template<class SCALARTYPE, unsigned int ALIGNMENT = 128>
  class coordinate_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class ell_matrix;

  template<typename ScalarT, typename IndexT = unsigned int>
  class sliced_ell_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class hyb_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class circulant_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class hankel_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class toeplitz_matrix;

  template<class SCALARTYPE, unsigned int ALIGNMENT = 1>
  class vandermonde_matrix;

  //
  // Proxies:
  //
  template<typename SizeType = vcl_size_t, typename DistanceType = std::ptrdiff_t>
  class basic_range;

  typedef basic_range<>  range;

  template<typename SizeType = vcl_size_t, typename DistanceType = std::ptrdiff_t>
  class basic_slice;

  typedef basic_slice<>  slice;

  template<typename VectorType>
  class vector_range;

  template<typename VectorType>
  class vector_slice;

  template<typename MatrixType>
  class matrix_range;

  template<typename MatrixType>
  class matrix_slice;


  /** @brief Helper struct for checking whether a type is a host scalar type (e.g. float, double) */
  template<typename T>
  struct is_cpu_scalar
  {
    enum { value = false };
  };

  /** @brief Helper struct for checking whether a type is a viennacl::scalar<> */
  template<typename T>
  struct is_scalar
  {
    enum { value = false };
  };

  /** @brief Helper struct for checking whether a type represents a sign flip on a viennacl::scalar<> */
  template<typename T>
  struct is_flip_sign_scalar
  {
    enum { value = false };
  };

  /** @brief Helper struct for checking whether the provided type represents a scalar (either host, from ViennaCL, or a flip-sign proxy) */
  template<typename T>
  struct is_any_scalar
  {
    enum { value = (is_scalar<T>::value || is_cpu_scalar<T>::value || is_flip_sign_scalar<T>::value )};
  };

  /** @brief Checks for a type being either vector_base or implicit_vector_base */
  template<typename T>
  struct is_any_vector { enum { value = 0 }; };

  /** @brief Checks for either matrix_base or implicit_matrix_base */
  template<typename T>
  struct is_any_dense_matrix { enum { value = 0 }; };

  /** @brief Helper class for checking whether a matrix has a row-major layout. */
  template<typename T>
  struct is_row_major
  {
    enum { value = false };
  };

  /** @brief Helper class for checking whether a matrix is a compressed_matrix (CSR format) */
  template<typename T>
  struct is_compressed_matrix
  {
    enum { value = false };
  };

  /** @brief Helper class for checking whether a matrix is a coordinate_matrix (COO format) */
  template<typename T>
  struct is_coordinate_matrix
  {
    enum { value = false };
  };

  /** @brief Helper class for checking whether a matrix is an ell_matrix (ELL format) */
  template<typename T>
  struct is_ell_matrix
  {
    enum { value = false };
  };

  /** @brief Helper class for checking whether a matrix is a sliced_ell_matrix (SELL-C-\f$ \sigma \f$ format) */
  template<typename T>
  struct is_sliced_ell_matrix
  {
    enum { value = false };
  };


  /** @brief Helper class for checking whether a matrix is a hyb_matrix (hybrid format: ELL plus CSR) */
  template<typename T>
  struct is_hyb_matrix
  {
    enum { value = false };
  };

  /** @brief Helper class for checking whether the provided type is one of the sparse matrix types (compressed_matrix, coordinate_matrix, etc.) */
  template<typename T>
  struct is_any_sparse_matrix
  {
    enum { value = false };
  };


  /** @brief Helper class for checking whether a matrix is a circulant matrix */
  template<typename T>
  struct is_circulant_matrix
  {
    enum { value = false };
  };

  /** @brief Helper class for checking whether a matrix is a Hankel matrix */
  template<typename T>
  struct is_hankel_matrix
  {
    enum { value = false };
  };

  /** @brief Helper class for checking whether a matrix is a Toeplitz matrix */
  template<typename T>
  struct is_toeplitz_matrix
  {
    enum { value = false };
  };

  /** @brief Helper class for checking whether a matrix is a Vandermonde matrix */
  template<typename T>
  struct is_vandermonde_matrix
  {
    enum { value = false };
  };

  /** @brief Helper class for checking whether the provided type is any of the dense structured matrix types (circulant, Hankel, etc.) */
  template<typename T>
  struct is_any_dense_structured_matrix
  {
    enum { value = viennacl::is_circulant_matrix<T>::value || viennacl::is_hankel_matrix<T>::value || viennacl::is_toeplitz_matrix<T>::value || viennacl::is_vandermonde_matrix<T>::value };
  };




  /** @brief Exception class in case of memory errors */
  class memory_exception : public std::exception
  {
  public:
    memory_exception() : message_() {}
    memory_exception(std::string message) : message_("ViennaCL: Internal memory error: " + message) {}

    virtual const char* what() const throw() { return message_.c_str(); }

    virtual ~memory_exception() throw() {}
  private:
    std::string message_;
  };

  class cuda_not_available_exception : public std::exception
  {
  public:
    cuda_not_available_exception() : message_("ViennaCL was compiled without CUDA support, but CUDA functionality required for this operation.") {}

    virtual const char* what() const throw() { return message_.c_str(); }

    virtual ~cuda_not_available_exception() throw() {}
  private:
    std::string message_;
  };

  class zero_on_diagonal_exception : public std::runtime_error
  {
  public:
    zero_on_diagonal_exception(std::string const & what_arg) : std::runtime_error(what_arg) {}
  };

  class unknown_norm_exception : public std::runtime_error
  {
  public:
    unknown_norm_exception(std::string const & what_arg) : std::runtime_error(what_arg) {}
  };



  namespace tools
  {
    //helper for matrix row/col iterators
    //must be specialized for every viennacl matrix type
    /** @brief Helper class for incrementing an iterator in a dense matrix. */
    template<typename ROWCOL, typename MATRIXTYPE>
    struct MATRIX_ITERATOR_INCREMENTER
    {
      typedef typename MATRIXTYPE::ERROR_SPECIALIZATION_FOR_THIS_MATRIX_TYPE_MISSING          ErrorIndicator;

      static void apply(const MATRIXTYPE & /*mat*/, unsigned int & /*row*/, unsigned int & /*col*/) {}
    };
  }

  namespace linalg
  {
#if !defined(_MSC_VER) || defined(__CUDACC__)

    template<class SCALARTYPE, unsigned int ALIGNMENT>
    void convolve_i(viennacl::vector<SCALARTYPE, ALIGNMENT>& input1,
                    viennacl::vector<SCALARTYPE, ALIGNMENT>& input2,
                    viennacl::vector<SCALARTYPE, ALIGNMENT>& output);

    template<typename T>
    viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<op_prod> >
    element_prod(vector_base<T> const & v1, vector_base<T> const & v2);

    template<typename T>
    viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<op_div> >
    element_div(vector_base<T> const & v1, vector_base<T> const & v2);



    template<typename T>
    void inner_prod_impl(vector_base<T> const & vec1,
                         vector_base<T> const & vec2,
                         scalar<T> & result);

    template<typename LHS, typename RHS, typename OP, typename T>
    void inner_prod_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec1,
                         vector_base<T> const & vec2,
                         scalar<T> & result);

    template<typename T, typename LHS, typename RHS, typename OP>
    void inner_prod_impl(vector_base<T> const & vec1,
                         viennacl::vector_expression<LHS, RHS, OP> const & vec2,
                         scalar<T> & result);

    template<typename LHS1, typename RHS1, typename OP1,
              typename LHS2, typename RHS2, typename OP2, typename T>
    void inner_prod_impl(viennacl::vector_expression<LHS1, RHS1, OP1> const & vec1,
                         viennacl::vector_expression<LHS2, RHS2, OP2> const & vec2,
                         scalar<T> & result);

    ///////////////////////////

    template<typename T>
    void inner_prod_cpu(vector_base<T> const & vec1,
                        vector_base<T> const & vec2,
                        T & result);

    template<typename LHS, typename RHS, typename OP, typename T>
    void inner_prod_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec1,
                        vector_base<T> const & vec2,
                        T & result);

    template<typename T, typename LHS, typename RHS, typename OP>
    void inner_prod_cpu(vector_base<T> const & vec1,
                        viennacl::vector_expression<LHS, RHS, OP> const & vec2,
                        T & result);

    template<typename LHS1, typename RHS1, typename OP1,
              typename LHS2, typename RHS2, typename OP2, typename S3>
    void inner_prod_cpu(viennacl::vector_expression<LHS1, RHS1, OP1> const & vec1,
                        viennacl::vector_expression<LHS2, RHS2, OP2> const & vec2,
                        S3 & result);



    //forward definition of norm_1_impl function
    template<typename T>
    void norm_1_impl(vector_base<T> const & vec, scalar<T> & result);

    template<typename LHS, typename RHS, typename OP, typename T>
    void norm_1_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                     scalar<T> & result);


    template<typename T>
    void norm_1_cpu(vector_base<T> const & vec,
                    T & result);

    template<typename LHS, typename RHS, typename OP, typename S2>
    void norm_1_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                    S2 & result);

    //forward definition of norm_2_impl function
    template<typename T>
    void norm_2_impl(vector_base<T> const & vec, scalar<T> & result);

    template<typename LHS, typename RHS, typename OP, typename T>
    void norm_2_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                     scalar<T> & result);

    template<typename T>
    void norm_2_cpu(vector_base<T> const & vec, T & result);

    template<typename LHS, typename RHS, typename OP, typename S2>
    void norm_2_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                    S2 & result);


    //forward definition of norm_inf_impl function
    template<typename T>
    void norm_inf_impl(vector_base<T> const & vec, scalar<T> & result);

    template<typename LHS, typename RHS, typename OP, typename T>
    void norm_inf_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                      scalar<T> & result);


    template<typename T>
    void norm_inf_cpu(vector_base<T> const & vec, T & result);

    template<typename LHS, typename RHS, typename OP, typename S2>
    void norm_inf_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                      S2 & result);

    //forward definition of max()-related functions
    template<typename T>
    void max_impl(vector_base<T> const & vec, scalar<T> & result);

    template<typename LHS, typename RHS, typename OP, typename T>
    void max_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                  scalar<T> & result);


    template<typename T>
    void max_cpu(vector_base<T> const & vec, T & result);

    template<typename LHS, typename RHS, typename OP, typename S2>
    void max_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                 S2 & result);

    //forward definition of min()-related functions
    template<typename T>
    void min_impl(vector_base<T> const & vec, scalar<T> & result);

    template<typename LHS, typename RHS, typename OP, typename T>
    void min_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                  scalar<T> & result);


    template<typename T>
    void min_cpu(vector_base<T> const & vec, T & result);

    template<typename LHS, typename RHS, typename OP, typename S2>
    void min_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                 S2 & result);

    //forward definition of sum()-related functions
    template<typename T>
    void sum_impl(vector_base<T> const & vec, scalar<T> & result);

    template<typename LHS, typename RHS, typename OP, typename T>
    void sum_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                  scalar<T> & result);


    template<typename T>
    void sum_cpu(vector_base<T> const & vec, T & result);

    template<typename LHS, typename RHS, typename OP, typename S2>
    void sum_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                 S2 & result);


    // forward definition of frobenius norm:
    template<typename T>
    void norm_frobenius_impl(matrix_base<T> const & vec, scalar<T> & result);

    template<typename T>
    void norm_frobenius_cpu(matrix_base<T> const & vec, T & result);


    template<typename T>
    vcl_size_t index_norm_inf(vector_base<T> const & vec);

    template<typename LHS, typename RHS, typename OP>
    vcl_size_t index_norm_inf(viennacl::vector_expression<LHS, RHS, OP> const & vec);

    //forward definition of prod_impl functions

    template<typename NumericT>
    void prod_impl(const matrix_base<NumericT> & mat,
                   const vector_base<NumericT> & vec,
                         vector_base<NumericT> & result);

    template<typename NumericT>
    void prod_impl(const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & mat_trans,
                   const vector_base<NumericT> & vec,
                         vector_base<NumericT> & result);

    template<typename SparseMatrixType, class SCALARTYPE, unsigned int ALIGNMENT>
    typename viennacl::enable_if< viennacl::is_any_sparse_matrix<SparseMatrixType>::value,
                                  vector_expression<const SparseMatrixType,
                                                    const vector<SCALARTYPE, ALIGNMENT>,
                                                    op_prod >
                                 >::type
    prod_impl(const SparseMatrixType & mat,
              const vector<SCALARTYPE, ALIGNMENT> & vec);

    // forward definition of summation routines for matrices:

    template<typename NumericT>
    void row_sum_impl(const matrix_base<NumericT> & A,
                            vector_base<NumericT> & result);

    template<typename NumericT>
    void column_sum_impl(const matrix_base<NumericT> & A,
                               vector_base<NumericT> & result);

#endif

    namespace detail
    {
      enum row_info_types
      {
        SPARSE_ROW_NORM_INF = 0,
        SPARSE_ROW_NORM_1,
        SPARSE_ROW_NORM_2,
        SPARSE_ROW_DIAGONAL
      };

    }


    /** @brief A tag class representing a lower triangular matrix */
    struct lower_tag
    {
      static const char * name() { return "lower"; }
    };      //lower triangular matrix
    /** @brief A tag class representing an upper triangular matrix */
    struct upper_tag
    {
      static const char * name() { return "upper"; }
    };      //upper triangular matrix
    /** @brief A tag class representing a lower triangular matrix with unit diagonal*/
    struct unit_lower_tag
    {
      static const char * name() { return "unit_lower"; }
    }; //unit lower triangular matrix
    /** @brief A tag class representing an upper triangular matrix with unit diagonal*/
    struct unit_upper_tag
    {
      static const char * name() { return "unit_upper"; }
    }; //unit upper triangular matrix

    //preconditioner tags
    class ilut_tag;

    /** @brief A tag class representing the use of no preconditioner */
    class no_precond
    {
      public:
        template<typename VectorType>
        void apply(VectorType &) const {}
    };


  } //namespace linalg

  //
  // More namespace comments to follow:
  //

  /** @brief Namespace providing routines for handling the different memory domains. */
  namespace backend
  {
    /** @brief Provides implementations for handling memory buffers in CPU RAM. */
    namespace cpu_ram
    {
      /** @brief Holds implementation details for handling memory buffers in CPU RAM. Not intended for direct use by library users. */
      namespace detail {}
    }

    /** @brief Provides implementations for handling CUDA memory buffers. */
    namespace cuda
    {
      /** @brief Holds implementation details for handling CUDA memory buffers. Not intended for direct use by library users. */
      namespace detail {}
    }

    /** @brief Implementation details for the generic memory backend interface. */
    namespace detail {}

    /** @brief Provides implementations for handling OpenCL memory buffers. */
    namespace opencl
    {
      /** @brief Holds implementation details for handling OpenCL memory buffers. Not intended for direct use by library users. */
      namespace detail {}
    }
  }


  /** @brief Holds implementation details for functionality in the main viennacl-namespace. Not intended for direct use by library users. */
  namespace detail
  {
    /** @brief Helper namespace for fast Fourier transforms. Not to be used directly by library users. */
    namespace fft
    {
      /** @brief Helper namespace for fast-Fourier transformation. Deprecated. */
      namespace FFT_DATA_ORDER {}
    }
  }


  /** @brief Provides an OpenCL kernel generator. */
  namespace device_specific
  {
    /** @brief Provides the implementation for tuning the kernels for a particular device. */
    namespace autotune {}

    /** @brief Contains implementation details of the kernel generator. */
    namespace detail {}

    /** @brief Namespace holding the various device-specific parameters for generating the best kernels. */
    namespace profiles {}

    /** @brief Contains various helper routines for kernel generation. */
    namespace utils {}
  }

  /** @brief Provides basic input-output functionality. */
  namespace io
  {
    /** @brief Implementation details for IO functionality. Usually not of interest for a library user. */
    namespace detail {}

    /** @brief Namespace holding the various XML tag definitions for the kernel parameter tuning facility. */
    namespace tag {}

    /** @brief Namespace holding the various XML strings for the kernel parameter tuning facility. */
    namespace val {}
  }

  /** @brief Provides all linear algebra operations which are not covered by operator overloads. */
  namespace linalg
  {
    /** @brief Holds all CUDA compute kernels used by ViennaCL. */
    namespace cuda
    {
      /** @brief Helper functions for the CUDA linear algebra backend. */
      namespace detail {}
    }

    /** @brief Namespace holding implementation details for linear algebra routines. Usually not of interest for a library user. */
    namespace detail
    {
      /** @brief Implementation namespace for algebraic multigrid preconditioner. */
      namespace amg {}

      /** @brief Implementation namespace for sparse approximate inverse preconditioner. */
      namespace spai {}
    }

    /** @brief Holds all compute kernels with conventional host-based execution (buffers in CPU RAM). */
    namespace host_based
    {
      /** @brief Helper functions for the host-based linear algebra backend. */
      namespace detail {}
    }

    /** @brief Namespace containing the OpenCL kernels. Deprecated, will be moved to viennacl::linalg::opencl in future releases. */
    namespace kernels {}

    /** @brief Holds all routines providing OpenCL linear algebra operations. */
    namespace opencl
    {
      /** @brief Helper functions for OpenCL-accelerated linear algebra operations. */
      namespace detail {}

      /** @brief Contains the OpenCL kernel generation functions for a predefined set of functionality. */
      namespace kernels
      {
        /** @brief Implementation details for the predefined OpenCL kernels. */
        namespace detail {}
      }
    }
  }

  /** @brief OpenCL backend. Manages platforms, contexts, buffers, kernels, etc. */
  namespace ocl {}

  /** @brief Namespace containing many meta-functions. */
  namespace result_of {}

  /** @brief Namespace for various tools used within ViennaCL. */
  namespace tools
  {
    /** @brief Contains implementation details for the tools. Usually not of interest for the library user. */
    namespace detail {}
  }

  /** @brief Namespace providing traits-information as well as generic wrappers to common routines for vectors and matrices such as size() or clear() */
  namespace traits {}

  /** @brief Contains the scheduling functionality which allows for dynamic kernel generation as well as the fusion of multiple statements into a single kernel. */
  namespace scheduler
  {
    /** @brief Implementation details for the scheduler */
    namespace detail {}

    /** @brief Helper metafunctions used for the scheduler */
    namespace result_of {}
  }

} //namespace viennacl

#endif

/*@}*/
