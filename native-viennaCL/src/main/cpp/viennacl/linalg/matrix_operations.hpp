#ifndef VIENNACL_LINALG_MATRIX_OPERATIONS_HPP_
#define VIENNACL_LINALG_MATRIX_OPERATIONS_HPP_

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

/** @file viennacl/linalg/matrix_operations.hpp
    @brief Implementations of dense matrix related operations including matrix-vector products.
*/

#include "viennacl/forwards.h"
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/vector_proxy.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/vector.hpp"
#include "viennacl/linalg/host_based/matrix_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/matrix_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/matrix_operations.hpp"
#endif

namespace viennacl
{
  namespace linalg
  {

    template<typename DestNumericT, typename SrcNumericT>
    void convert(matrix_base<DestNumericT> & dest, matrix_base<SrcNumericT> const & src)
    {
      assert(viennacl::traits::size1(dest) == viennacl::traits::size1(src) && bool("Incompatible matrix sizes in m1 = m2 (convert): size1(m1) != size1(m2)"));
      assert(viennacl::traits::size2(dest) == viennacl::traits::size2(src) && bool("Incompatible matrix sizes in m1 = m2 (convert): size2(m1) != size2(m2)"));

      switch (viennacl::traits::handle(dest).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::convert(dest, src);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::convert(dest, src);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::convert(dest, src);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    template<typename NumericT,
              typename SizeT, typename DistanceT>
    void trans(const matrix_expression<const matrix_base<NumericT, SizeT, DistanceT>,const matrix_base<NumericT, SizeT, DistanceT>, op_trans> & proxy,
              matrix_base<NumericT> & temp_trans)
    {
      switch (viennacl::traits::handle(proxy).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::trans(proxy, temp_trans);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::trans(proxy,temp_trans);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::trans(proxy,temp_trans);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename NumericT,
              typename ScalarType1>
    void am(matrix_base<NumericT> & mat1,
            matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
    {
      switch (viennacl::traits::handle(mat1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::am(mat1, mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::am(mat1, mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::am(mat1, mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename NumericT,
              typename ScalarType1, typename ScalarType2>
    void ambm(matrix_base<NumericT> & mat1,
              matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
              matrix_base<NumericT> const & mat3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
    {
      switch (viennacl::traits::handle(mat1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::ambm(mat1,
                                             mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                             mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::ambm(mat1,
                                         mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::ambm(mat1,
                                       mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                       mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename NumericT,
              typename ScalarType1, typename ScalarType2>
    void ambm_m(matrix_base<NumericT> & mat1,
                matrix_base<NumericT> const & mat2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                matrix_base<NumericT> const & mat3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
    {
      switch (viennacl::traits::handle(mat1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::ambm_m(mat1,
                                               mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                               mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::ambm_m(mat1,
                                           mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                           mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::ambm_m(mat1,
                                         mat2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         mat3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename NumericT>
    void matrix_assign(matrix_base<NumericT> & mat, NumericT s, bool clear = false)
    {
      switch (viennacl::traits::handle(mat).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::matrix_assign(mat, s, clear);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::matrix_assign(mat, s, clear);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::matrix_assign(mat, s, clear);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename NumericT>
    void matrix_diagonal_assign(matrix_base<NumericT> & mat, NumericT s)
    {
      switch (viennacl::traits::handle(mat).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::matrix_diagonal_assign(mat, s);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::matrix_diagonal_assign(mat, s);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::matrix_diagonal_assign(mat, s);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Dispatcher interface for A = diag(v, k) */
    template<typename NumericT>
    void matrix_diag_from_vector(const vector_base<NumericT> & v, int k, matrix_base<NumericT> & A)
    {
      switch (viennacl::traits::handle(v).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::matrix_diag_from_vector(v, k, A);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::matrix_diag_from_vector(v, k, A);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::matrix_diag_from_vector(v, k, A);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Dispatcher interface for v = diag(A, k) */
    template<typename NumericT>
    void matrix_diag_to_vector(const matrix_base<NumericT> & A, int k, vector_base<NumericT> & v)
    {
      switch (viennacl::traits::handle(A).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::matrix_diag_to_vector(A, k, v);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::matrix_diag_to_vector(A, k, v);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::matrix_diag_to_vector(A, k, v);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    template<typename NumericT>
    void matrix_row(const matrix_base<NumericT> & A, unsigned int i, vector_base<NumericT> & v)
    {
      switch (viennacl::traits::handle(A).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::matrix_row(A, i, v);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::matrix_row(A, i, v);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::matrix_row(A, i, v);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    template<typename NumericT>
    void matrix_column(const matrix_base<NumericT> & A, unsigned int j, vector_base<NumericT> & v)
    {
      switch (viennacl::traits::handle(A).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::matrix_column(A, j, v);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::matrix_column(A, j, v);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::matrix_column(A, j, v);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the Frobenius norm of a matrix - dispatcher interface
    *
    * @param A      The matrix
    * @param result The result scalar
    *
    * Note that if A is strided or off-set, then a copy will be created.
    */
    template<typename T>
    void norm_frobenius_impl(matrix_base<T> const & A,
                             scalar<T> & result)
    {
      typedef typename matrix_base<T>::handle_type  HandleType;

      if ((A.start1() > 0) || (A.start2() > 0) || (A.stride1() > 1) || (A.stride2() > 1)) {
        if (A.row_major()) {
          viennacl::matrix<T, viennacl::row_major> temp_A(A);
          viennacl::vector_base<T> temp(const_cast<HandleType &>(temp_A.handle()), temp_A.internal_size(), 0, 1);
          norm_2_impl(temp, result);
        } else {
          viennacl::matrix<T, viennacl::column_major> temp_A(A);
          viennacl::vector_base<T> temp(const_cast<HandleType &>(temp_A.handle()), temp_A.internal_size(), 0, 1);
          norm_2_impl(temp, result);
        }
      } else {
        viennacl::vector_base<T> temp(const_cast<HandleType &>(A.handle()), A.internal_size(), 0, 1);
        norm_2_impl(temp, result);
      }

    }

    /** @brief Computes the Frobenius norm of a vector with final reduction on the CPU
    *
    * @param A      The matrix
    * @param result The result scalar
    *
    * Note that if A is strided or off-set, then a copy will be created.
    */
    template<typename T>
    void norm_frobenius_cpu(matrix_base<T> const & A,
                            T & result)
    {
      typedef typename matrix_base<T>::handle_type  HandleType;

      if ((A.start1() > 0) || (A.start2() > 0) || (A.stride1() > 1) || (A.stride2() > 1)) {
        if (A.row_major()) {
          viennacl::matrix<T, viennacl::row_major> temp_A(A);
          viennacl::vector_base<T> temp(const_cast<HandleType &>(temp_A.handle()), temp_A.internal_size(), 0, 1);
          norm_2_cpu(temp, result);
        } else {
          viennacl::matrix<T, viennacl::column_major> temp_A(A);
          viennacl::vector_base<T> temp(const_cast<HandleType &>(temp_A.handle()), temp_A.internal_size(), 0, 1);
          norm_2_cpu(temp, result);
        }
      } else {
        viennacl::vector_base<T> temp(const_cast<HandleType &>(A.handle()), A.internal_size(), 0, 1);
        norm_2_cpu(temp, result);
      }

    }

    //
    /////////////////////////   matrix-vector products /////////////////////////////////
    //



    // A * x

    /** @brief Carries out matrix-vector multiplication
    *
    * Implementation of the convenience expression result = prod(mat, vec);
    *
    * @param mat    The matrix
    * @param vec    The vector
    * @param result The result vector
    */
    template<typename NumericT>
    void prod_impl(const matrix_base<NumericT> & mat,
                   const vector_base<NumericT> & vec,
                         vector_base<NumericT> & result)
    {
      assert( (viennacl::traits::size1(mat) == viennacl::traits::size(result)) && bool("Size check failed at v1 = prod(A, v2): size1(A) != size(v1)"));
      assert( (viennacl::traits::size2(mat) == viennacl::traits::size(vec))    && bool("Size check failed at v1 = prod(A, v2): size2(A) != size(v2)"));

      switch (viennacl::traits::handle(mat).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(mat, false, vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(mat, false, vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(mat, false, vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    // trans(A) * x

    /** @brief Carries out matrix-vector multiplication with a transposed matrix
    *
    * Implementation of the convenience expression result = trans(mat) * vec;
    *
    * @param mat_trans  The transposed matrix proxy
    * @param vec        The vector
    * @param result     The result vector
    */
    template<typename NumericT>
    void prod_impl(const matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & mat_trans,
                   const vector_base<NumericT> & vec,
                         vector_base<NumericT> & result)
    {
      assert( (viennacl::traits::size1(mat_trans.lhs()) == viennacl::traits::size(vec))    && bool("Size check failed at v1 = trans(A) * v2: size1(A) != size(v2)"));
      assert( (viennacl::traits::size2(mat_trans.lhs()) == viennacl::traits::size(result)) && bool("Size check failed at v1 = trans(A) * v2: size2(A) != size(v1)"));

      switch (viennacl::traits::handle(mat_trans.lhs()).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(mat_trans.lhs(), true, vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(mat_trans.lhs(), true, vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(mat_trans.lhs(), true, vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    //
    /////////////////////////   matrix-matrix products /////////////////////////////////
    //

    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(A, B);
    *
    */
    template<typename NumericT, typename ScalarType >
    void prod_impl(const matrix_base<NumericT> & A,
                   const matrix_base<NumericT> & B,
                         matrix_base<NumericT> & C,
                   ScalarType alpha,
                   ScalarType beta)
    {
      assert( (viennacl::traits::size1(A) == viennacl::traits::size1(C)) && bool("Size check failed at C = prod(A, B): size1(A) != size1(C)"));
      assert( (viennacl::traits::size2(A) == viennacl::traits::size1(B)) && bool("Size check failed at C = prod(A, B): size2(A) != size1(B)"));
      assert( (viennacl::traits::size2(B) == viennacl::traits::size2(C)) && bool("Size check failed at C = prod(A, B): size2(B) != size2(C)"));


      switch (viennacl::traits::handle(A).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(A, false, B, false, C, alpha, beta);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(A, false, B, false, C, alpha, beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(A, false, B, false, C, alpha, beta);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }



    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(trans(A), B);
    *
    */
    template<typename NumericT, typename ScalarType >
    void prod_impl(const viennacl::matrix_expression< const matrix_base<NumericT>,
                                                      const matrix_base<NumericT>,
                                                      op_trans> & A,
                   const matrix_base<NumericT> & B,
                         matrix_base<NumericT> & C,
                   ScalarType alpha,
                   ScalarType beta)
    {
      assert(viennacl::traits::size2(A.lhs()) == viennacl::traits::size1(C) && bool("Size check failed at C = prod(trans(A), B): size2(A) != size1(C)"));
      assert(viennacl::traits::size1(A.lhs()) == viennacl::traits::size1(B) && bool("Size check failed at C = prod(trans(A), B): size1(A) != size1(B)"));
      assert(viennacl::traits::size2(B)       == viennacl::traits::size2(C) && bool("Size check failed at C = prod(trans(A), B): size2(B) != size2(C)"));

      switch (viennacl::traits::handle(A.lhs()).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(A.lhs(), true, B, false, C, alpha, beta);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(A.lhs(), true, B, false, C, alpha, beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(A.lhs(), true, B, false, C, alpha, beta);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }




    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(A, trans(B));
    *
    */
    template<typename NumericT, typename ScalarType >
    void prod_impl(const matrix_base<NumericT> & A,
                   const viennacl::matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & B,
                         matrix_base<NumericT> & C,
                   ScalarType alpha,
                   ScalarType beta)
    {
      assert(viennacl::traits::size1(A)       == viennacl::traits::size1(C)       && bool("Size check failed at C = prod(A, trans(B)): size1(A) != size1(C)"));
      assert(viennacl::traits::size2(A)       == viennacl::traits::size2(B.lhs()) && bool("Size check failed at C = prod(A, trans(B)): size2(A) != size2(B)"));
      assert(viennacl::traits::size1(B.lhs()) == viennacl::traits::size2(C)       && bool("Size check failed at C = prod(A, trans(B)): size1(B) != size2(C)"));

      switch (viennacl::traits::handle(A).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(A, false, B.lhs(), true, C, alpha, beta);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(A, false, B.lhs(), true, C, alpha, beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(A, false, B.lhs(), true, C, alpha, beta);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }



    /** @brief Carries out matrix-matrix multiplication
    *
    * Implementation of C = prod(trans(A), trans(B));
    *
    */
    template<typename NumericT, typename ScalarType >
    void prod_impl(const viennacl::matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & A,
                   const viennacl::matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans> & B,
                   matrix_base<NumericT> & C,
                   ScalarType alpha,
                   ScalarType beta)
    {
      assert(viennacl::traits::size2(A.lhs()) == viennacl::traits::size1(C)       && bool("Size check failed at C = prod(trans(A), trans(B)): size2(A) != size1(C)"));
      assert(viennacl::traits::size1(A.lhs()) == viennacl::traits::size2(B.lhs()) && bool("Size check failed at C = prod(trans(A), trans(B)): size1(A) != size2(B)"));
      assert(viennacl::traits::size1(B.lhs()) == viennacl::traits::size2(C)       && bool("Size check failed at C = prod(trans(A), trans(B)): size1(B) != size2(C)"));

      switch (viennacl::traits::handle(A.lhs()).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::prod_impl(A.lhs(), true, B.lhs(), true, C, alpha, beta);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::prod_impl(A.lhs(), true, B.lhs(), true, C, alpha, beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::prod_impl(A.lhs(), true, B.lhs(), true, C, alpha, beta);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    ///////////////////////// summation operations /////////////

    template<typename NumericT>
    void row_sum_impl(matrix_base<NumericT> const & A, vector_base<NumericT> & result)
    {
      viennacl::vector<NumericT> all_ones = viennacl::scalar_vector<NumericT>(A.size2(), NumericT(1), viennacl::traits::context(A));
      viennacl::linalg::prod_impl(A, all_ones, result);
    }

    template<typename NumericT>
    void column_sum_impl(matrix_base<NumericT> const & A, vector_base<NumericT> & result)
    {
      viennacl::vector<NumericT> all_ones = viennacl::scalar_vector<NumericT>(A.size1(), NumericT(1), viennacl::traits::context(A));
      viennacl::linalg::prod_impl(matrix_expression< const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>(A, A), all_ones, result);
    }

    ///////////////////////// Elementwise operations /////////////



    /** @brief Implementation of the element-wise operation A = B .* C and A = B ./ C for matrices (using MATLAB syntax). Don't use this function directly, use element_prod() and element_div().
    *
    * @param A      The result matrix (or -range, or -slice)
    * @param proxy  The proxy object holding B, C, and the operation
    */
    template<typename T, typename OP>
    void element_op(matrix_base<T> & A,
                    matrix_expression<const matrix_base<T>, const matrix_base<T>, OP> const & proxy)
    {
      assert( (viennacl::traits::size1(A) == viennacl::traits::size1(proxy)) && bool("Size check failed at A = element_op(B): size1(A) != size1(B)"));
      assert( (viennacl::traits::size2(A) == viennacl::traits::size2(proxy)) && bool("Size check failed at A = element_op(B): size2(A) != size2(B)"));

      switch (viennacl::traits::handle(A).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::element_op(A, proxy);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::element_op(A, proxy);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::element_op(A, proxy);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


#define VIENNACL_MAKE_BINARY_OP(OPNAME)\
    template<typename T>\
    viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<op_##OPNAME> >\
    element_##OPNAME(matrix_base<T> const & A, matrix_base<T> const & B)\
    {\
      return viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_binary<op_##OPNAME> >(A, B);\
    }\
\
    template<typename M1, typename M2, typename OP, typename T>\
    viennacl::matrix_expression<const matrix_expression<const M1, const M2, OP>,\
                                const matrix_base<T>,\
                                op_element_binary<op_##OPNAME> >\
    element_##OPNAME(matrix_expression<const M1, const M2, OP> const & proxy, matrix_base<T> const & B)\
    {\
      return viennacl::matrix_expression<const matrix_expression<const M1, const M2, OP>,\
                                         const matrix_base<T>,\
                                         op_element_binary<op_##OPNAME> >(proxy, B);\
    }\
\
    template<typename T, typename M2, typename M3, typename OP>\
    viennacl::matrix_expression<const matrix_base<T>,\
                                const matrix_expression<const M2, const M3, OP>,\
                                op_element_binary<op_##OPNAME> >\
    element_##OPNAME(matrix_base<T> const & A, matrix_expression<const M2, const M3, OP> const & proxy)\
    {\
      return viennacl::matrix_expression<const matrix_base<T>,\
                                         const matrix_expression<const M2, const M3, OP>,\
                                         op_element_binary<op_##OPNAME> >(A, proxy);\
    }\
\
    template<typename M1, typename M2, typename OP1,\
              typename M3, typename M4, typename OP2>\
    viennacl::matrix_expression<const matrix_expression<const M1, const M2, OP1>,\
                                const matrix_expression<const M3, const M4, OP2>,\
                                op_element_binary<op_##OPNAME> >\
    element_##OPNAME(matrix_expression<const M1, const M2, OP1> const & proxy1,\
                 matrix_expression<const M3, const M4, OP2> const & proxy2)\
    {\
      return viennacl::matrix_expression<const matrix_expression<const M1, const M2, OP1>,\
                                         const matrix_expression<const M3, const M4, OP2>,\
                                         op_element_binary<op_##OPNAME> >(proxy1, proxy2);\
    }

    VIENNACL_MAKE_BINARY_OP(prod)
    VIENNACL_MAKE_BINARY_OP(div)
    VIENNACL_MAKE_BINARY_OP(pow)

    VIENNACL_MAKE_BINARY_OP(eq)
    VIENNACL_MAKE_BINARY_OP(neq)
    VIENNACL_MAKE_BINARY_OP(greater)
    VIENNACL_MAKE_BINARY_OP(less)
    VIENNACL_MAKE_BINARY_OP(geq)
    VIENNACL_MAKE_BINARY_OP(leq)

#undef VIENNACL_GENERATE_BINARY_OP_OVERLOADS



#define VIENNACL_MAKE_UNARY_ELEMENT_OP(funcname) \
    template<typename T> \
    viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<op_##funcname> > \
    element_##funcname(matrix_base<T> const & A) \
    { \
      return viennacl::matrix_expression<const matrix_base<T>, const matrix_base<T>, op_element_unary<op_##funcname> >(A, A); \
    } \
    template<typename LHS, typename RHS, typename OP> \
    viennacl::matrix_expression<const matrix_expression<const LHS, const RHS, OP>, \
                                const matrix_expression<const LHS, const RHS, OP>, \
                                op_element_unary<op_##funcname> > \
    element_##funcname(matrix_expression<const LHS, const RHS, OP> const & proxy) \
    { \
      return viennacl::matrix_expression<const matrix_expression<const LHS, const RHS, OP>, \
                                         const matrix_expression<const LHS, const RHS, OP>, \
                                         op_element_unary<op_##funcname> >(proxy, proxy); \
    } \

    VIENNACL_MAKE_UNARY_ELEMENT_OP(abs)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(acos)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(asin)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(atan)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(ceil)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(cos)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(cosh)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(exp)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(fabs)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(floor)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(log)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(log10)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(sin)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(sinh)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(sqrt)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(tan)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(tanh)

#undef VIENNACL_MAKE_UNARY_ELEMENT_OP


    //
    /////////////////////////   miscellaneous operations /////////////////////////////////
    //


    /** @brief Returns a proxy class for the operation mat += vec1 * vec2^T, i.e. a rank 1 update
    *
    * @param vec1    The first vector
    * @param vec2    The second vector
    */
    template<typename NumericT>
    viennacl::matrix_expression<const vector_base<NumericT>, const vector_base<NumericT>, op_prod>
    outer_prod(const vector_base<NumericT> & vec1, const vector_base<NumericT> & vec2)
    {
      return viennacl::matrix_expression< const vector_base<NumericT>, const vector_base<NumericT>, op_prod>(vec1, vec2);
    }


    /** @brief The implementation of the operation mat += alpha * vec1 * vec2^T, i.e. a scaled rank 1 update
    *
    * Implementation of the convenience expression result += alpha * outer_prod(vec1, vec2);
    *
    * @param mat1             The matrix to be updated
    * @param alpha            The scaling factor (either a viennacl::scalar<>, float, or double)
    * @param len_alpha        Length of the buffer for an eventual final reduction step (currently always '1')
    * @param reciprocal_alpha Use 1/alpha instead of alpha
    * @param flip_sign_alpha  Use -alpha instead of alpha
    * @param vec1             The first vector
    * @param vec2             The second vector
    */
    template<typename NumericT, typename S1>
    void scaled_rank_1_update(matrix_base<NumericT> & mat1,
                              S1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                              const vector_base<NumericT> & vec1,
                              const vector_base<NumericT> & vec2)
    {
      switch (viennacl::traits::handle(mat1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::scaled_rank_1_update(mat1,
                                                             alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                             vec1, vec2);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::scaled_rank_1_update(mat1,
                                                         alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                         vec1, vec2);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::scaled_rank_1_update(mat1,
                                                       alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                       vec1, vec2);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief This function stores the diagonal and the superdiagonal of a matrix in two vectors.
    *
    *
    * @param A     The matrix from which the vectors will be extracted of.
    * @param dh    The vector in which the diagonal of the matrix will be stored in.
    * @param sh    The vector in which the superdiagonal of the matrix will be stored in.
    */

    template <typename NumericT, typename VectorType>
    void bidiag_pack(matrix_base<NumericT> & A,
                     VectorType & dh,
                     VectorType & sh
                    )
    {
      switch (viennacl::traits::handle(A).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::bidiag_pack(A, dh, sh);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::bidiag_pack(A, dh, sh);
          break;
#endif

#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::bidiag_pack(A, dh, sh);
          break;
#endif

        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }


    }
    /** @brief This function copies a row or a column from a matrix to a vector.
    *
    *
    * @param A          The matrix where to copy from.
    * @param V          The vector to fill with data.
    * @param row_start  The number of the first row to copy.
    * @param col_start  The number of the first column to copy.
    * @param copy_col   Set to TRUE to copy a column, FALSE to copy a row.
    */

    template <typename SCALARTYPE>
    void copy_vec(matrix_base<SCALARTYPE>& A,
                  vector_base<SCALARTYPE>& V,
                  vcl_size_t row_start,
                  vcl_size_t col_start,
                  bool copy_col
    )
    {
      switch (viennacl::traits::handle(A).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::copy_vec(A, V, row_start, col_start, copy_col);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::copy_vec(A, V, row_start, col_start, copy_col);
          break;
#endif

#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::copy_vec(A, V, row_start, col_start, copy_col);
          break;
#endif

        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }

    }

    /** @brief This function applies a householder transformation to a matrix. A <- P * A with a householder reflection P
    *
    * @param A       The matrix to be updated.
    * @param D       The normalized householder vector.
    * @param start   The repetition counter.
    */
  template <typename NumericT>
  void house_update_A_left(matrix_base<NumericT> & A,
                           vector_base<NumericT>    & D,
                           vcl_size_t start)
  {
    switch (viennacl::traits::handle(A).get_active_handle_id())
    {
      case viennacl::MAIN_MEMORY:
        viennacl::linalg::host_based::house_update_A_left(A, D, start);
        break;
#ifdef VIENNACL_WITH_OPENCL
      case viennacl::OPENCL_MEMORY:
        viennacl::linalg::opencl::house_update_A_left(A, D, start);
        break;
#endif

#ifdef VIENNACL_WITH_CUDA
      case viennacl::CUDA_MEMORY:
        viennacl::linalg::cuda::house_update_A_left(A, D, start);
        break;
#endif

      case viennacl::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }


  /** @brief This function applies a householder transformation to a matrix: A <- A * P with a householder reflection P
  *
  *
  * @param A        The matrix to be updated.
  * @param D        The normalized householder vector.
  */

  template <typename NumericT>
  void house_update_A_right(matrix_base<NumericT>& A,
                            vector_base<NumericT>   & D)
  {
    switch (viennacl::traits::handle(A).get_active_handle_id())
    {
      case viennacl::MAIN_MEMORY:
        viennacl::linalg::host_based::house_update_A_right(A, D);
        break;
#ifdef VIENNACL_WITH_OPENCL
      case viennacl::OPENCL_MEMORY:
        viennacl::linalg::opencl::house_update_A_right(A, D);
        break;
#endif

#ifdef VIENNACL_WITH_CUDA
      case viennacl::CUDA_MEMORY:
        viennacl::linalg::cuda::house_update_A_right(A, D);
        break;
#endif

      case viennacl::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }

  /** @brief This function updates the matrix Q, which is needed for the computation of the eigenvectors.
  *
  * @param Q        The matrix to be updated.
  * @param D        The householder vector.
  * @param A_size1  size1 of matrix A
  */

  template <typename NumericT>
  void house_update_QL(matrix_base<NumericT> & Q,
                       vector_base<NumericT>    & D,
                       vcl_size_t A_size1)
  {
    switch (viennacl::traits::handle(Q).get_active_handle_id())
    {
      case viennacl::MAIN_MEMORY:
        viennacl::linalg::host_based::house_update_QL(Q, D, A_size1);
        break;
#ifdef VIENNACL_WITH_OPENCL
      case viennacl::OPENCL_MEMORY:
        viennacl::linalg::opencl::house_update_QL(Q, D, A_size1);
        break;
#endif

#ifdef VIENNACL_WITH_CUDA
      case viennacl::CUDA_MEMORY:
        viennacl::linalg::cuda::house_update_QL(Q, D, A_size1);
        break;
#endif

      case viennacl::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }


  /** @brief This function updates the matrix Q. It is part of the tql2 algorithm.
  *
  *
  * @param Q       The matrix to be updated.
  * @param tmp1    Vector with data from the tql2 algorithm.
  * @param tmp2    Vector with data from the tql2 algorithm.
  * @param l       Data from the tql2 algorithm.
  * @param m       Data from the tql2 algorithm.
  */
  template<typename NumericT>
  void givens_next(matrix_base<NumericT> & Q,
                   vector_base<NumericT> & tmp1,
                   vector_base<NumericT> & tmp2,
                   int l,
                   int m
                )
  {
    switch (viennacl::traits::handle(Q).get_active_handle_id())
    {
      case viennacl::MAIN_MEMORY:
        viennacl::linalg::host_based::givens_next(Q, tmp1, tmp2, l, m);
        break;
#ifdef VIENNACL_WITH_OPENCL
      case viennacl::OPENCL_MEMORY:
        viennacl::linalg::opencl::givens_next(Q, tmp1, tmp2, l, m);
        break;
#endif

#ifdef VIENNACL_WITH_CUDA
      case viennacl::CUDA_MEMORY:
        viennacl::linalg::cuda::givens_next(Q, tmp1, tmp2, l, m);
        break;
#endif

      case viennacl::MEMORY_NOT_INITIALIZED:
        throw memory_exception("not initialised!");
      default:
        throw memory_exception("not implemented");
    }
  }

  } //namespace linalg




  //
  /////////////////////////  Operator overloads /////////////////////////////////
  //


  //v += A * x
  /** @brief Implementation of the operation v1 += A * v2, where A is a matrix
  *
  * @param v1     The result vector v1 where A * v2 is added to
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT>
  operator+=(vector_base<NumericT> & v1,
             const viennacl::vector_expression< const matrix_base<NumericT>, const vector_base<NumericT>, viennacl::op_prod> & proxy)
  {
    assert(viennacl::traits::size1(proxy.lhs()) == v1.size() && bool("Size check failed for v1 += A * v2: size1(A) != size(v1)"));

    vector<NumericT> result(viennacl::traits::size1(proxy.lhs()));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 += result;
    return v1;
  }

  /** @brief Implementation of the operation v1 -= A * v2, where A is a matrix
  *
  * @param v1     The result vector v1 where A * v2 is subtracted from
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT>
  operator-=(vector_base<NumericT> & v1,
             const viennacl::vector_expression< const matrix_base<NumericT>, const vector_base<NumericT>, viennacl::op_prod> & proxy)
  {
    assert(viennacl::traits::size1(proxy.lhs()) == v1.size() && bool("Size check failed for v1 -= A * v2: size1(A) != size(v1)"));

    vector<NumericT> result(viennacl::traits::size1(proxy.lhs()));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 -= result;
    return v1;
  }





  //free functions:
  /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  viennacl::vector<NumericT>
  operator+(const vector_base<NumericT> & v1,
            const vector_expression< const matrix_base<NumericT>, const vector_base<NumericT>, op_prod> & proxy)
  {
    assert(viennacl::traits::size1(proxy.lhs()) == viennacl::traits::size(v1) && bool("Size check failed for v1 + A * v2: size1(A) != size(v1)"));

    vector<NumericT> result(viennacl::traits::size(v1));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result += v1;
    return result;
  }

  /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  viennacl::vector<NumericT>
  operator-(const vector_base<NumericT> & v1,
            const vector_expression< const matrix_base<NumericT>, const vector_base<NumericT>, op_prod> & proxy)
  {
    assert(viennacl::traits::size1(proxy.lhs()) == viennacl::traits::size(v1) && bool("Size check failed for v1 - A * v2: size1(A) != size(v1)"));

    vector<NumericT> result(viennacl::traits::size(v1));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result = v1 - result;
    return result;
  }


  ////////// transposed_matrix_proxy


  //v += A^T * x
  /** @brief Implementation of the operation v1 += A * v2, where A is a matrix
  *
  * @param v1     The addend vector where the result is written to.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT>
  operator+=(vector_base<NumericT> & v1,
             const vector_expression< const matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>,
                                                              const vector_base<NumericT>,
                                                              op_prod> & proxy)
  {
    assert(viennacl::traits::size2(proxy.lhs()) == v1.size() && bool("Size check failed in v1 += trans(A) * v2: size2(A) != size(v1)"));

    vector<NumericT> result(viennacl::traits::size2(proxy.lhs()));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 += result;
    return v1;
  }

  //v -= A^T * x
  /** @brief Implementation of the operation v1 -= A * v2, where A is a matrix
  *
  * @param v1     The addend vector where the result is written to.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT>
  operator-=(vector_base<NumericT> & v1,
             const vector_expression< const matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>,
                                                              const vector_base<NumericT>,
                                                              op_prod> & proxy)
  {
    assert(viennacl::traits::size2(proxy.lhs()) == v1.size() && bool("Size check failed in v1 += trans(A) * v2: size2(A) != size(v1)"));

    vector<NumericT> result(viennacl::traits::size2(proxy.lhs()));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    v1 -= result;
    return v1;
  }


  //free functions:
  /** @brief Implementation of the operation 'result = v1 + A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT>
  operator+(const vector_base<NumericT> & v1,
            const vector_expression< const matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>,
                                     const vector_base<NumericT>,
                                     op_prod> & proxy)
  {
    assert(viennacl::traits::size2(proxy.lhs()) == viennacl::traits::size(v1) && bool("Size check failed in v1 + trans(A) * v2: size2(A) != size(v1)"));

    vector<NumericT> result(viennacl::traits::size(v1));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result += v1;
    return result;
  }

  /** @brief Implementation of the operation 'result = v1 - A * v2', where A is a matrix
  *
  * @param v1     The addend vector.
  * @param proxy  An expression template proxy class.
  */
  template<typename NumericT>
  vector<NumericT>
  operator-(const vector_base<NumericT> & v1,
            const vector_expression< const matrix_expression<const matrix_base<NumericT>, const matrix_base<NumericT>, op_trans>,
                                     const vector_base<NumericT>,
                                     op_prod> & proxy)
  {
    assert(viennacl::traits::size2(proxy.lhs()) == viennacl::traits::size(v1) && bool("Size check failed in v1 - trans(A) * v2: size2(A) != size(v1)"));

    vector<NumericT> result(viennacl::traits::size(v1));
    viennacl::linalg::prod_impl(proxy.lhs(), proxy.rhs(), result);
    result = v1 - result;
    return result;
  }


} //namespace viennacl


#endif
