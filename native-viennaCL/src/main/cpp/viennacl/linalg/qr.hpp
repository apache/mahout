#ifndef VIENNACL_LINALG_QR_HPP
#define VIENNACL_LINALG_QR_HPP

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

/** @file viennacl/linalg/qr.hpp
    @brief Provides a QR factorization using a block-based approach.
*/

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <math.h>
#include <cmath>
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/matrix_expression.hpp"

#include "viennacl/matrix.hpp"
#include "viennacl/matrix_proxy.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/range.hpp"

namespace viennacl
{
  namespace linalg
  {
    namespace detail
    {

      template<typename MatrixType, typename VectorType>
      typename MatrixType::value_type setup_householder_vector_ublas(MatrixType const & A, VectorType & v, MatrixType & matrix_1x1, vcl_size_t j)
      {
        using boost::numeric::ublas::range;
        using boost::numeric::ublas::project;

        typedef typename MatrixType::value_type   ScalarType;

        //compute norm of column below diagonal:
        matrix_1x1 = boost::numeric::ublas::prod( trans(project(A, range(j+1, A.size1()), range(j, j+1))),
                                                        project(A, range(j+1, A.size1()), range(j, j+1))
                                                );
        ScalarType sigma = matrix_1x1(0,0);
        ScalarType beta = 0;
        ScalarType A_jj = A(j,j);

        assert( sigma >= 0.0  && bool("sigma must be non-negative!"));

        //get v from A:
        v(j,0) = 1.0;
        project(v, range(j+1, A.size1()), range(0,1)) = project(A, range(j+1, A.size1()), range(j,j+1));

        if (sigma <= 0)
          return 0;
        else
        {
          ScalarType mu = std::sqrt(sigma + A_jj*A_jj);

          ScalarType v1 = (A_jj <= 0) ? (A_jj - mu) : (-sigma / (A_jj + mu));
          beta = static_cast<ScalarType>(2.0) * v1 * v1 / (sigma + v1 * v1);

          //divide v by its diagonal element v[j]
          project(v, range(j+1, A.size1()), range(0,1)) /= v1;
        }

        return beta;
      }


      template<typename MatrixType, typename VectorType>
      typename viennacl::result_of::cpu_value_type< typename MatrixType::value_type >::type
      setup_householder_vector_viennacl(MatrixType const & A, VectorType & v, MatrixType & matrix_1x1, vcl_size_t j)
      {
        using viennacl::range;
        using viennacl::project;

        typedef typename viennacl::result_of::cpu_value_type< typename MatrixType::value_type >::type   ScalarType;

        //compute norm of column below diagonal:
        matrix_1x1 = viennacl::linalg::prod( trans(project(A, range(j+1, A.size1()), range(j, j+1))),
                                                   project(A, range(j+1, A.size1()), range(j, j+1))
                                           );
        ScalarType sigma = matrix_1x1(0,0);
        ScalarType beta = 0;
        ScalarType A_jj = A(j,j);

        assert( sigma >= 0.0  && bool("sigma must be non-negative!"));

        //get v from A:
        v(j,0) = 1.0;
        project(v, range(j+1, A.size1()), range(0,1)) = project(A, range(j+1, A.size1()), range(j,j+1));

        if (sigma == 0)
          return 0;
        else
        {
          ScalarType mu = std::sqrt(sigma + A_jj*A_jj);

          ScalarType v1 = (A_jj <= 0) ? (A_jj - mu) : (-sigma / (A_jj + mu));

          beta = 2.0 * v1 * v1 / (sigma + v1 * v1);

          //divide v by its diagonal element v[j]
          project(v, range(j+1, A.size1()), range(0,1)) /= v1;
        }

        return beta;
      }


      // Apply (I - beta v v^T) to the k-th column of A, where v is the reflector starting at j-th row/column
      template<typename MatrixType, typename VectorType, typename ScalarType>
      void householder_reflect(MatrixType & A, VectorType & v, ScalarType beta, vcl_size_t j, vcl_size_t k)
      {
        ScalarType v_in_col = A(j,k);
        for (vcl_size_t i=j+1; i<A.size1(); ++i)
          v_in_col += v[i] * A(i,k);

        //assert(v[j] == 1.0);

        for (vcl_size_t i=j; i<A.size1(); ++i)
          A(i,k) -= beta * v_in_col * v[i];
      }

      template<typename MatrixType, typename VectorType, typename ScalarType>
      void householder_reflect_ublas(MatrixType & A, VectorType & v, MatrixType & matrix_1x1, ScalarType beta, vcl_size_t j, vcl_size_t k)
      {
        using boost::numeric::ublas::range;
        using boost::numeric::ublas::project;

        ScalarType v_in_col = A(j,k);
        matrix_1x1 = boost::numeric::ublas::prod(trans(project(v, range(j+1, A.size1()), range(0, 1))),
                                                       project(A, range(j+1, A.size1()), range(k,k+1)));
        v_in_col += matrix_1x1(0,0);

        project(A, range(j, A.size1()), range(k, k+1)) -= (beta * v_in_col) * project(v, range(j, A.size1()), range(0, 1));
      }

      template<typename MatrixType, typename VectorType, typename ScalarType>
      void householder_reflect_viennacl(MatrixType & A, VectorType & v, MatrixType & matrix_1x1, ScalarType beta, vcl_size_t j, vcl_size_t k)
      {
        using viennacl::range;
        using viennacl::project;

        ScalarType v_in_col = A(j,k);

        matrix_1x1 = viennacl::linalg::prod(trans(project(v, range(j+1, A.size1()), range(0, 1))),
                                                  project(A, range(j+1, A.size1()), range(k,k+1)));
        v_in_col += matrix_1x1(0,0);

        if ( beta * v_in_col != 0.0)
        {
          VectorType temp = project(v, range(j, A.size1()), range(0, 1));
          project(v, range(j, A.size1()), range(0, 1)) *= (beta * v_in_col);
          project(A, range(j, A.size1()), range(k, k+1)) -= project(v, range(j, A.size1()), range(0, 1));
          project(v, range(j, A.size1()), range(0, 1)) = temp;
        }
      }


      // Apply (I - beta v v^T) to A, where v is the reflector starting at j-th row/column
      template<typename MatrixType, typename VectorType, typename ScalarType>
      void householder_reflect(MatrixType & A, VectorType & v, ScalarType beta, vcl_size_t j)
      {
        vcl_size_t column_end = A.size2();

        for (vcl_size_t k=j; k<column_end; ++k) //over columns
          householder_reflect(A, v, beta, j, k);
      }


      template<typename MatrixType, typename VectorType>
      void write_householder_to_A(MatrixType & A, VectorType const & v, vcl_size_t j)
      {
        for (vcl_size_t i=j+1; i<A.size1(); ++i)
          A(i,j) = v[i];
      }

      template<typename MatrixType, typename VectorType>
      void write_householder_to_A_ublas(MatrixType & A, VectorType const & v, vcl_size_t j)
      {
        using boost::numeric::ublas::range;
        using boost::numeric::ublas::project;

        //VectorType temp = project(v, range(j+1, A.size1()));
        project( A, range(j+1, A.size1()), range(j, j+1) ) = project(v, range(j+1, A.size1()), range(0, 1) );;
      }

      template<typename MatrixType, typename VectorType>
      void write_householder_to_A_viennacl(MatrixType & A, VectorType const & v, vcl_size_t j)
      {
        using viennacl::range;
        using viennacl::project;

        //VectorType temp = project(v, range(j+1, A.size1()));
        project( A, range(j+1, A.size1()), range(j, j+1) ) = project(v, range(j+1, A.size1()), range(0, 1) );;
      }



      /** @brief Implementation of inplace-QR factorization for a general Boost.uBLAS compatible matrix A
      *
      * @param A            A dense compatible to Boost.uBLAS
      * @param block_size   The block size to be used. The number of columns of A must be a multiple of block_size
      */
      template<typename MatrixType>
      std::vector<typename MatrixType::value_type> inplace_qr_ublas(MatrixType & A, vcl_size_t block_size = 32)
      {
        typedef typename MatrixType::value_type   ScalarType;
        typedef boost::numeric::ublas::matrix_range<MatrixType>  MatrixRange;

        using boost::numeric::ublas::range;
        using boost::numeric::ublas::project;

        std::vector<ScalarType> betas(A.size2());
        MatrixType v(A.size1(), 1);
        MatrixType matrix_1x1(1,1);

        MatrixType Y(A.size1(), block_size); Y.clear(); Y.resize(A.size1(), block_size);
        MatrixType W(A.size1(), block_size); W.clear(); W.resize(A.size1(), block_size);

        //run over A in a block-wise manner:
        for (vcl_size_t j = 0; j < std::min(A.size1(), A.size2()); j += block_size)
        {
          vcl_size_t effective_block_size = std::min(std::min(A.size1(), A.size2()), j+block_size) - j;

          //determine Householder vectors:
          for (vcl_size_t k = 0; k < effective_block_size; ++k)
          {
            betas[j+k] = detail::setup_householder_vector_ublas(A, v, matrix_1x1, j+k);

            for (vcl_size_t l = k; l < effective_block_size; ++l)
              detail::householder_reflect_ublas(A, v, matrix_1x1, betas[j+k], j+k, j+l);

            detail::write_householder_to_A_ublas(A, v, j+k);
          }

          //
          // Setup Y:
          //
          Y.clear();  Y.resize(A.size1(), block_size);
          for (vcl_size_t k = 0; k < effective_block_size; ++k)
          {
            //write Householder to Y:
            Y(j+k,k) = 1.0;
            project(Y, range(j+k+1, A.size1()), range(k, k+1)) = project(A, range(j+k+1, A.size1()), range(j+k, j+k+1));
          }

          //
          // Setup W:
          //

          //first vector:
          W.clear();  W.resize(A.size1(), block_size);
          W(j, 0) = -betas[j];
          project(W, range(j+1, A.size1()), range(0, 1)) = -betas[j] * project(A, range(j+1, A.size1()), range(j, j+1));


          //k-th column of W is given by -beta * (Id + W*Y^T) v_k, where W and Y have k-1 columns
          for (vcl_size_t k = 1; k < effective_block_size; ++k)
          {
            MatrixRange Y_old = project(Y, range(j, A.size1()), range(0, k));
            MatrixRange v_k   = project(Y, range(j, A.size1()), range(k, k+1));
            MatrixRange W_old = project(W, range(j, A.size1()), range(0, k));
            MatrixRange z     = project(W, range(j, A.size1()), range(k, k+1));

            MatrixType YT_prod_v = boost::numeric::ublas::prod(boost::numeric::ublas::trans(Y_old), v_k);
            z = - betas[j+k] * (v_k + prod(W_old, YT_prod_v));
          }

          //
          //apply (I+WY^T)^T = I + Y W^T to the remaining columns of A:
          //

          if (A.size2() - j - effective_block_size > 0)
          {

            MatrixRange A_part(A, range(j, A.size1()), range(j+effective_block_size, A.size2()));
            MatrixRange W_part(W, range(j, A.size1()), range(0, effective_block_size));
            MatrixType temp = boost::numeric::ublas::prod(trans(W_part), A_part);

            A_part += prod(project(Y, range(j, A.size1()), range(0, effective_block_size)),
                          temp);
          }
        }

        return betas;
      }


      /** @brief Implementation of a OpenCL-only QR factorization for GPUs (or multi-core CPU). DEPRECATED! Use only if you're curious and interested in playing a bit with a GPU-only implementation.
      *
      * Performance is rather poor at small matrix sizes.
      * Prefer the use of the hybrid version, which is automatically chosen using the interface function inplace_qr()
      *
      * @param A            A dense ViennaCL matrix to be factored
      * @param block_size   The block size to be used. The number of columns of A must be a multiple of block_size
      */
      template<typename MatrixType>
      std::vector< typename viennacl::result_of::cpu_value_type< typename MatrixType::value_type >::type >
      inplace_qr_viennacl(MatrixType & A, vcl_size_t block_size = 16)
      {
        typedef typename viennacl::result_of::cpu_value_type< typename MatrixType::value_type >::type   ScalarType;
        typedef viennacl::matrix_range<MatrixType>  MatrixRange;

        using viennacl::range;
        using viennacl::project;

        std::vector<ScalarType> betas(A.size2());
        MatrixType v(A.size1(), 1);
        MatrixType matrix_1x1(1,1);

        MatrixType Y(A.size1(), block_size); Y.clear();
        MatrixType W(A.size1(), block_size); W.clear();

        MatrixType YT_prod_v(block_size, 1);
        MatrixType z(A.size1(), 1);

        //run over A in a block-wise manner:
        for (vcl_size_t j = 0; j < std::min(A.size1(), A.size2()); j += block_size)
        {
          vcl_size_t effective_block_size = std::min(std::min(A.size1(), A.size2()), j+block_size) - j;

          //determine Householder vectors:
          for (vcl_size_t k = 0; k < effective_block_size; ++k)
          {
            betas[j+k] = detail::setup_householder_vector_viennacl(A, v, matrix_1x1, j+k);
            for (vcl_size_t l = k; l < effective_block_size; ++l)
              detail::householder_reflect_viennacl(A, v, matrix_1x1, betas[j+k], j+k, j+l);

            detail::write_householder_to_A_viennacl(A, v, j+k);
          }

          //
          // Setup Y:
          //
          Y.clear();
          for (vcl_size_t k = 0; k < effective_block_size; ++k)
          {
            //write Householder to Y:
            Y(j+k,k) = 1.0;
            project(Y, range(j+k+1, A.size1()), range(k, k+1)) = project(A, range(j+k+1, A.size1()), range(j+k, j+k+1));
          }

          //
          // Setup W:
          //

          //first vector:
          W.clear();
          W(j, 0) = -betas[j];
          //project(W, range(j+1, A.size1()), range(0, 1)) = -betas[j] * project(A, range(j+1, A.size1()), range(j, j+1));
          project(W, range(j+1, A.size1()), range(0, 1)) = project(A, range(j+1, A.size1()), range(j, j+1));
          project(W, range(j+1, A.size1()), range(0, 1)) *= -betas[j];


          //k-th column of W is given by -beta * (Id + W*Y^T) v_k, where W and Y have k-1 columns
          for (vcl_size_t k = 1; k < effective_block_size; ++k)
          {
            MatrixRange Y_old = project(Y, range(j, A.size1()), range(0, k));
            MatrixRange v_k   = project(Y, range(j, A.size1()), range(k, k+1));
            MatrixRange W_old = project(W, range(j, A.size1()), range(0, k));

            project(YT_prod_v, range(0, k), range(0,1)) = prod(trans(Y_old), v_k);
            project(z, range(j, A.size1()), range(0,1)) = prod(W_old, project(YT_prod_v, range(0, k), range(0,1)));
            project(W, range(j, A.size1()), range(k, k+1)) = project(z, range(j, A.size1()), range(0,1));
            project(W, range(j, A.size1()), range(k, k+1)) += v_k;
            project(W, range(j, A.size1()), range(k, k+1)) *= - betas[j+k];
          }

          //
          //apply (I+WY^T)^T = I + Y W^T to the remaining columns of A:
          //

          if (A.size2() > j + effective_block_size)
          {

            MatrixRange A_part(A, range(j, A.size1()), range(j+effective_block_size, A.size2()));
            MatrixRange W_part(W, range(j, A.size1()), range(0, effective_block_size));
            MatrixType temp = prod(trans(W_part), A_part);

            A_part += prod(project(Y, range(j, A.size1()), range(0, effective_block_size)),
                          temp);
          }
        }

        return betas;
      }






      //MatrixType is ViennaCL-matrix
      /** @brief Implementation of a hybrid QR factorization using uBLAS on the CPU and ViennaCL for GPUs (or multi-core CPU)
      *
      * Prefer the use of the convenience interface inplace_qr()
      *
      * @param A            A dense ViennaCL matrix to be factored
      * @param block_size   The block size to be used. The number of columns of A must be a multiple of block_size
      */
      template<typename MatrixType>
      std::vector< typename viennacl::result_of::cpu_value_type< typename MatrixType::value_type >::type >
      inplace_qr_hybrid(MatrixType & A, vcl_size_t block_size = 16)
      {
        typedef typename viennacl::result_of::cpu_value_type< typename MatrixType::value_type >::type   ScalarType;

        typedef viennacl::matrix_range<MatrixType>                    VCLMatrixRange;
        typedef boost::numeric::ublas::matrix<ScalarType>             UblasMatrixType;
        typedef boost::numeric::ublas::matrix_range<UblasMatrixType>  UblasMatrixRange;

        std::vector<ScalarType> betas(A.size2());
        UblasMatrixType v(A.size1(), 1);
        UblasMatrixType matrix_1x1(1,1);

        UblasMatrixType ublasW(A.size1(), block_size); ublasW.clear(); ublasW.resize(A.size1(), block_size);
        UblasMatrixType ublasY(A.size1(), block_size); ublasY.clear(); ublasY.resize(A.size1(), block_size);

        UblasMatrixType ublasA(A.size1(), A.size1());

        MatrixType vclW(ublasW.size1(), ublasW.size2());
        MatrixType vclY(ublasY.size1(), ublasY.size2());


        //run over A in a block-wise manner:
        for (vcl_size_t j = 0; j < std::min(A.size1(), A.size2()); j += block_size)
        {
          vcl_size_t effective_block_size = std::min(std::min(A.size1(), A.size2()), j+block_size) - j;
          UblasMatrixRange ublasA_part = boost::numeric::ublas::project(ublasA,
                                                                        boost::numeric::ublas::range(0, A.size1()),
                                                                        boost::numeric::ublas::range(j, j + effective_block_size));
          viennacl::copy(viennacl::project(A,
                                          viennacl::range(0, A.size1()),
                                          viennacl::range(j, j+effective_block_size)),
                         ublasA_part
                        );

          //determine Householder vectors:
          for (vcl_size_t k = 0; k < effective_block_size; ++k)
          {
            betas[j+k] = detail::setup_householder_vector_ublas(ublasA, v, matrix_1x1, j+k);

            for (vcl_size_t l = k; l < effective_block_size; ++l)
              detail::householder_reflect_ublas(ublasA, v, matrix_1x1, betas[j+k], j+k, j+l);

            detail::write_householder_to_A_ublas(ublasA, v, j+k);
          }

          //
          // Setup Y:
          //
          ublasY.clear();  ublasY.resize(A.size1(), block_size);
          for (vcl_size_t k = 0; k < effective_block_size; ++k)
          {
            //write Householder to Y:
            ublasY(j+k,k) = 1.0;
            boost::numeric::ublas::project(ublasY,
                                           boost::numeric::ublas::range(j+k+1, A.size1()),
                                           boost::numeric::ublas::range(k, k+1))
              = boost::numeric::ublas::project(ublasA,
                                               boost::numeric::ublas::range(j+k+1, A.size1()),
                                               boost::numeric::ublas::range(j+k, j+k+1));
          }

          //
          // Setup W:
          //

          //first vector:
          ublasW.clear();  ublasW.resize(A.size1(), block_size);
          ublasW(j, 0) = -betas[j];
          boost::numeric::ublas::project(ublasW,
                                        boost::numeric::ublas::range(j+1, A.size1()),
                                        boost::numeric::ublas::range(0, 1))
            = -betas[j] * boost::numeric::ublas::project(ublasA,
                                                          boost::numeric::ublas::range(j+1, A.size1()),
                                                          boost::numeric::ublas::range(j, j+1));


          //k-th column of W is given by -beta * (Id + W*Y^T) v_k, where W and Y have k-1 columns
          for (vcl_size_t k = 1; k < effective_block_size; ++k)
          {
            UblasMatrixRange Y_old = boost::numeric::ublas::project(ublasY,
                                                                    boost::numeric::ublas::range(j, A.size1()),
                                                                    boost::numeric::ublas::range(0, k));
            UblasMatrixRange v_k   = boost::numeric::ublas::project(ublasY,
                                                                    boost::numeric::ublas::range(j, A.size1()),
                                                                    boost::numeric::ublas::range(k, k+1));
            UblasMatrixRange W_old = boost::numeric::ublas::project(ublasW,
                                                                    boost::numeric::ublas::range(j, A.size1()),
                                                                    boost::numeric::ublas::range(0, k));
            UblasMatrixRange z     = boost::numeric::ublas::project(ublasW,
                                                                    boost::numeric::ublas::range(j, A.size1()),
                                                                    boost::numeric::ublas::range(k, k+1));

            UblasMatrixType YT_prod_v = boost::numeric::ublas::prod(boost::numeric::ublas::trans(Y_old), v_k);
            z = - betas[j+k] * (v_k + prod(W_old, YT_prod_v));
          }



          //
          //apply (I+WY^T)^T = I + Y W^T to the remaining columns of A:
          //

          VCLMatrixRange A_part = viennacl::project(A,
                                                    viennacl::range(0, A.size1()),
                                                    viennacl::range(j, j+effective_block_size));

          viennacl::copy(boost::numeric::ublas::project(ublasA,
                                                        boost::numeric::ublas::range(0, A.size1()),
                                                        boost::numeric::ublas::range(j, j+effective_block_size)),
                        A_part);

          viennacl::copy(ublasW, vclW);
          viennacl::copy(ublasY, vclY);

          if (A.size2() > j + effective_block_size)
          {

            VCLMatrixRange A_part2(A, viennacl::range(j, A.size1()), viennacl::range(j+effective_block_size, A.size2()));
            VCLMatrixRange W_part(vclW, viennacl::range(j, A.size1()), viennacl::range(0, effective_block_size));
            MatrixType temp = viennacl::linalg::prod(trans(W_part), A_part2);

            A_part2 += viennacl::linalg::prod(viennacl::project(vclY, viennacl::range(j, A.size1()), viennacl::range(0, effective_block_size)),
                                              temp);
          }
        }

        return betas;
      }



    } //namespace detail




    //takes an inplace QR matrix A and generates Q and R explicitly
    template<typename MatrixType, typename VectorType>
    void recoverQ(MatrixType const & A, VectorType const & betas, MatrixType & Q, MatrixType & R)
    {
      typedef typename MatrixType::value_type   ScalarType;

      std::vector<ScalarType> v(A.size1());

      Q.clear();
      R.clear();

      //
      // Recover R from upper-triangular part of A:
      //
      vcl_size_t i_max = std::min(R.size1(), R.size2());
      for (vcl_size_t i=0; i<i_max; ++i)
        for (vcl_size_t j=i; j<R.size2(); ++j)
          R(i,j) = A(i,j);

      //
      // Recover Q by applying all the Householder reflectors to the identity matrix:
      //
      for (vcl_size_t i=0; i<Q.size1(); ++i)
        Q(i,i) = 1.0;

      vcl_size_t j_max = std::min(A.size1(), A.size2());
      for (vcl_size_t j=0; j<j_max; ++j)
      {
        vcl_size_t col_index = j_max - j - 1;
        v[col_index] = 1.0;
        for (vcl_size_t i=col_index+1; i<A.size1(); ++i)
          v[i] = A(i, col_index);

        if (betas[col_index] > 0 || betas[col_index] < 0)
          detail::householder_reflect(Q, v, betas[col_index], col_index);
      }
    }


    /** @brief Computes Q^T b, where Q is an implicit orthogonal matrix defined via its Householder reflectors stored in A.
     *
     *  @param A      A matrix holding the Householder reflectors in the lower triangular part. Typically obtained from calling inplace_qr() on the original matrix
     *  @param betas  The scalars beta_i for each Householder reflector (I - beta_i v_i v_i^T)
     *  @param b      The vector b to which the result Q^T b is directly written to
     */
    template<typename MatrixType, typename VectorType1, typename VectorType2>
    void inplace_qr_apply_trans_Q(MatrixType const & A, VectorType1 const & betas, VectorType2 & b)
    {
      typedef typename viennacl::result_of::cpu_value_type<typename MatrixType::value_type>::type   ScalarType;

      //
      // Apply Q^T = (I - beta_m v_m v_m^T) \times ... \times (I - beta_0 v_0 v_0^T) by applying all the Householder reflectors to b:
      //
      for (vcl_size_t col_index=0; col_index<std::min(A.size1(), A.size2()); ++col_index)
      {
        ScalarType v_in_b = b[col_index];
        for (vcl_size_t i=col_index+1; i<A.size1(); ++i)
          v_in_b += A(i, col_index) * b[i];

        b[col_index] -= betas[col_index] * v_in_b;
        for (vcl_size_t i=col_index+1; i<A.size1(); ++i)
          b[i] -= betas[col_index] * A(i, col_index) * v_in_b;
      }
    }

    template<typename T, typename F, unsigned int ALIGNMENT, typename VectorType1, unsigned int A2>
    void inplace_qr_apply_trans_Q(viennacl::matrix<T, F, ALIGNMENT> const & A, VectorType1 const & betas, viennacl::vector<T, A2> & b)
    {
      boost::numeric::ublas::matrix<T> ublas_A(A.size1(), A.size2());
      viennacl::copy(A, ublas_A);

      std::vector<T> stl_b(b.size());
      viennacl::copy(b, stl_b);

      inplace_qr_apply_trans_Q(ublas_A, betas, stl_b);

      viennacl::copy(stl_b, b);
    }

    /** @brief Overload of inplace-QR factorization of a ViennaCL matrix A
     *
     * @param A            A dense ViennaCL matrix to be factored
     * @param block_size   The block size to be used.
     */
    template<typename T, typename F, unsigned int ALIGNMENT>
    std::vector<T> inplace_qr(viennacl::matrix<T, F, ALIGNMENT> & A, vcl_size_t block_size = 16)
    {
      return detail::inplace_qr_hybrid(A, block_size);
    }

    /** @brief Overload of inplace-QR factorization for a general Boost.uBLAS compatible matrix A
     *
     * @param A            A dense compatible to Boost.uBLAS
     * @param block_size   The block size to be used.
     */
    template<typename MatrixType>
    std::vector<typename MatrixType::value_type> inplace_qr(MatrixType & A, vcl_size_t block_size = 16)
    {
      return detail::inplace_qr_ublas(A, block_size);
    }



  } //linalg
} //viennacl


#endif
