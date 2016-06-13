#ifndef VIENNACL_LINALG_SVD_HPP
#define VIENNACL_LINALG_SVD_HPP

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

/** @file viennacl/linalg/svd.hpp
    @brief Provides singular value decomposition using a block-based approach.  Experimental.

    Contributed by Volodymyr Kysenko.
*/


// Note: Boost.uBLAS is required at the moment
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>


#include <cmath>

#include "viennacl/matrix.hpp"
#include "viennacl/linalg/opencl/kernels/svd.hpp"
#include "viennacl/linalg/qr-method-common.hpp"

namespace viennacl
{
  namespace linalg
  {

    namespace detail
    {

      template<typename MatrixType, typename VectorType>
      void givens_prev(MatrixType & matrix,
                       VectorType & tmp1,
                       VectorType & tmp2,
                       int n,
                       int l,
                       int k
                      )
      {
        typedef typename MatrixType::value_type                                   ScalarType;
        typedef typename viennacl::result_of::cpu_value_type<ScalarType>::type    CPU_ScalarType;

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(matrix).context());
        viennacl::ocl::kernel & kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<CPU_ScalarType>::program_name(), SVD_GIVENS_PREV_KERNEL);

        kernel.global_work_size(0, viennacl::tools::align_to_multiple<vcl_size_t>(viennacl::traits::size1(matrix), 256));
        kernel.local_work_size(0, 256);

        viennacl::ocl::enqueue(kernel(
                                      matrix,
                                      tmp1,
                                      tmp2,
                                      static_cast<cl_uint>(n),
                                      static_cast<cl_uint>(matrix.internal_size1()),
                                      static_cast<cl_uint>(l + 1),
                                      static_cast<cl_uint>(k + 1)
                              ));
      }


      template<typename MatrixType, typename VectorType>
      void change_signs(MatrixType& matrix, VectorType& signs, int n)
      {
        typedef typename MatrixType::value_type                                   ScalarType;
        typedef typename viennacl::result_of::cpu_value_type<ScalarType>::type    CPU_ScalarType;

        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(matrix).context());
        viennacl::ocl::kernel & kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<CPU_ScalarType>::program_name(), SVD_INVERSE_SIGNS_KERNEL);

        kernel.global_work_size(0, viennacl::tools::align_to_multiple<vcl_size_t>(viennacl::traits::size1(matrix), 16));
        kernel.global_work_size(1, viennacl::tools::align_to_multiple<vcl_size_t>(viennacl::traits::size2(matrix), 16));

        kernel.local_work_size(0, 16);
        kernel.local_work_size(1, 16);

        viennacl::ocl::enqueue(kernel(
                                      matrix,
                                      signs,
                                      static_cast<cl_uint>(n),
                                      static_cast<cl_uint>(matrix.internal_size1())
                              ));
      }

      template<typename MatrixType, typename CPU_VectorType>
      void svd_qr_shift(MatrixType & vcl_u,
                        MatrixType & vcl_v,
                        CPU_VectorType & q,
                        CPU_VectorType & e)
      {
        typedef typename MatrixType::value_type                                   ScalarType;
        typedef typename viennacl::result_of::cpu_value_type<ScalarType>::type    CPU_ScalarType;

        vcl_size_t n = q.size();
        int m = static_cast<int>(vcl_u.size1());

        detail::transpose(vcl_u);
        detail::transpose(vcl_v);

        std::vector<CPU_ScalarType> signs_v(n, 1);
        std::vector<CPU_ScalarType> cs1(n), ss1(n), cs2(n), ss2(n);

        viennacl::vector<CPU_ScalarType> tmp1(n, viennacl::traits::context(vcl_u)), tmp2(n, viennacl::traits::context(vcl_u));

        bool goto_test_conv = false;

        for (int k = static_cast<int>(n) - 1; k >= 0; k--)
        {
          // std::cout << "K = " << k << std::endl;

          vcl_size_t iter = 0;
          for (iter = 0; iter < detail::ITER_MAX; iter++)
          {
            // test for split
            int l;
            for (l = k; l >= 0; l--)
            {
              goto_test_conv = false;
              if (std::fabs(e[vcl_size_t(l)]) <= detail::EPS)
              {
                // set it
                goto_test_conv = true;
                break;
              }

              if (std::fabs(q[vcl_size_t(l) - 1]) <= detail::EPS)
              {
                // goto
                break;
              }
            }

            if (!goto_test_conv)
            {
              CPU_ScalarType c = 0.0;
              CPU_ScalarType s = 1.0;

              //int l1 = l - 1;
              //int l2 = k;

              for (int i = l; i <= k; i++)
              {
                CPU_ScalarType f = s * e[vcl_size_t(i)];
                e[vcl_size_t(i)] = c * e[vcl_size_t(i)];

                if (std::fabs(f) <= detail::EPS)
                {
                  //l2 = i - 1;
                  break;
                }

                CPU_ScalarType g = q[vcl_size_t(i)];
                CPU_ScalarType h = detail::pythag(f, g);
                q[vcl_size_t(i)] = h;
                c = g / h;
                s = -f / h;

                cs1[vcl_size_t(i)] = c;
                ss1[vcl_size_t(i)] = s;
              }

              // std::cout << "Hitted!" << l1 << " " << l2 << "\n";

              // for (int i = l; i <= l2; i++)
              // {
              //   for (int j = 0; j < m; j++)
              //   {
              //     CPU_ScalarType y = u(j, l1);
              //     CPU_ScalarType z = u(j, i);
              //     u(j, l1) = y * cs1[i] + z * ss1[i];
              //     u(j, i) = -y * ss1[i] + z * cs1[i];
              //   }
              // }
            }

            CPU_ScalarType z = q[vcl_size_t(k)];

            if (l == k)
            {
              if (z < 0)
              {
                q[vcl_size_t(k)] = -z;

                signs_v[vcl_size_t(k)] *= -1;
              }

              break;
            }

            if (iter >= detail::ITER_MAX - 1)
              break;

            CPU_ScalarType x = q[vcl_size_t(l)];
            CPU_ScalarType y = q[vcl_size_t(k) - 1];
            CPU_ScalarType g = e[vcl_size_t(k) - 1];
            CPU_ScalarType h = e[vcl_size_t(k)];
            CPU_ScalarType f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2 * h * y);

            g = detail::pythag<CPU_ScalarType>(f, 1);

            if (f < 0) {
              f = ((x - z) * (x + z) + h * (y / (f - g) - h)) / x;
            } else {
              f = ((x - z) * (x + z) + h * (y / (f + g) - h)) / x;
            }

            CPU_ScalarType c = 1;
            CPU_ScalarType s = 1;

            for (vcl_size_t i = static_cast<vcl_size_t>(l) + 1; i <= static_cast<vcl_size_t>(k); i++)
            {
              g = e[i];
              y = q[i];
              h = s * g;
              g = c * g;
              CPU_ScalarType z2 = detail::pythag(f, h);
              e[i - 1] = z2;
              c = f / z2;
              s = h / z2;
              f = x * c + g * s;
              g = -x * s + g * c;
              h = y * s;
              y = y * c;

              cs1[i] = c;
              ss1[i] = s;

              z2 = detail::pythag(f, h);
              q[i - 1] = z2;
              c = f / z2;
              s = h / z2;
              f = c * g + s * y;
              x = -s * g + c * y;

              cs2[i] = c;
              ss2[i] = s;
            }

            {
              viennacl::copy(cs1, tmp1);
              viennacl::copy(ss1, tmp2);

              givens_prev(vcl_v, tmp1, tmp2, static_cast<int>(n), l, k);
            }

            {
              viennacl::copy(cs2, tmp1);
              viennacl::copy(ss2, tmp2);

              givens_prev(vcl_u, tmp1, tmp2, m, l, k);
            }

            e[vcl_size_t(l)] = 0.0;
            e[vcl_size_t(k)] = f;
            q[vcl_size_t(k)] = x;
          }

        }


        viennacl::copy(signs_v, tmp1);
        change_signs(vcl_v, tmp1, static_cast<int>(n));

        // transpose singular matrices again
        detail::transpose(vcl_u);
        detail::transpose(vcl_v);
      }


      /*template<typename SCALARTYPE, unsigned int ALIGNMENT>
      bool householder_c(viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT> & A,
                          viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT> & Q,
                          viennacl::vector<SCALARTYPE, ALIGNMENT> & D,
                          vcl_size_t start)
      {

        vcl_size_t row_start = start;
        vcl_size_t col_start = start;

        if (row_start + 1 >= A.size1())
          return false;

        std::vector<SCALARTYPE> tmp(A.size1(), 0);

        copy_vec(A, D, row_start, col_start, true);
        fast_copy(D.begin(), D.begin() + (A.size1() - row_start), tmp.begin() + row_start);

        detail::householder_vector(tmp, row_start);

        fast_copy(tmp, D);

        viennacl::ocl::kernel & kernel = viennacl::ocl::get_kernel(viennacl::linalg::opencl::kernels::svd<SCALARTYPE>::program_name(), SVD_HOUSEHOLDER_COL_KERNEL);

        //kernel.global_work_size(0, A.size1() << 1);

        viennacl::ocl::enqueue(kernel(
                                      A,
                                      Q,
                                      D,
                                      static_cast<cl_uint>(row_start),
                                      static_cast<cl_uint>(col_start),
                                      static_cast<cl_uint>(A.size1()),
                                      static_cast<cl_uint>(A.size2()),
                                      static_cast<cl_uint>(A.internal_size2()),
                                      static_cast<cl_uint>(Q.internal_size2()),
                                      viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(SCALARTYPE)))
                              ));

        return true;
      }*/

      template<typename SCALARTYPE, unsigned int ALIGNMENT>
      bool householder_c(viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT>& A,
                          viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT>& Q,
                          viennacl::vector<SCALARTYPE, ALIGNMENT>& D,
                          vcl_size_t row_start, vcl_size_t col_start)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

        if (row_start + 1 >= A.size1())
          return false;

        prepare_householder_vector(A, D, A.size1(), row_start, col_start, row_start, true);

        {
          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<SCALARTYPE>::program_name(), SVD_HOUSEHOLDER_UPDATE_A_LEFT_KERNEL);

          viennacl::ocl::enqueue(kernel(
                                        A,
                                        D,
                                        static_cast<cl_uint>(row_start),
                                        static_cast<cl_uint>(col_start),
                                        static_cast<cl_uint>(A.size1()),
                                        static_cast<cl_uint>(A.size2()),
                                        static_cast<cl_uint>(A.internal_size2()),
                                        viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(SCALARTYPE)))
                                ));

        }

        {
          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<SCALARTYPE>::program_name(), SVD_HOUSEHOLDER_UPDATE_QL_KERNEL);

          viennacl::ocl::enqueue(kernel(
                                        Q,
                                        D,
                                        static_cast<cl_uint>(A.size1()),
                                      //  static_cast<cl_uint>(A.size2()),
                                        static_cast<cl_uint>(Q.internal_size2()),
                                        viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(SCALARTYPE)))
                                ));

        }

        return true;
      }

      /*
      template<typename SCALARTYPE, unsigned int ALIGNMENT>
      bool householder_r(viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT>& A,
                          viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT>& Q,
                          viennacl::vector<SCALARTYPE, ALIGNMENT>& S,
                          vcl_size_t start)
      {

        vcl_size_t row_start = start;
        vcl_size_t col_start = start + 1;

        if (col_start + 1 >= A.size2())
          return false;

        std::vector<SCALARTYPE> tmp(A.size2(), 0);

        copy_vec(A, S, row_start, col_start, false);
        fast_copy(S.begin(),
                  S.begin() + (A.size2() - col_start),
                  tmp.begin() + col_start);

        detail::householder_vector(tmp, col_start);
        fast_copy(tmp, S);

        viennacl::ocl::kernel& kernel = viennacl::ocl::get_kernel(viennacl::linalg::opencl::kernels::svd<SCALARTYPE>::program_name(), SVD_HOUSEHOLDER_ROW_KERNEL);

        viennacl::ocl::enqueue(kernel(
                                      A,
                                      Q,
                                      S,
                                      static_cast<cl_uint>(row_start),
                                      static_cast<cl_uint>(col_start),
                                      static_cast<cl_uint>(A.size1()),
                                      static_cast<cl_uint>(A.size2()),
                                      static_cast<cl_uint>(A.internal_size2()),
                                      static_cast<cl_uint>(Q.internal_size2()),
                                      viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(SCALARTYPE)))
                                ));
        return true;
      } */

      template<typename SCALARTYPE, unsigned int ALIGNMENT>
      bool householder_r(viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT> & A,
                          viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT> & Q,
                          viennacl::vector<SCALARTYPE, ALIGNMENT>& D,
                          vcl_size_t row_start, vcl_size_t col_start)
      {
        viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());

        if (col_start + 1 >= A.size2())
          return false;

        prepare_householder_vector(A, D, A.size2(), row_start, col_start, col_start, false);

        {
          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<SCALARTYPE>::program_name(), SVD_HOUSEHOLDER_UPDATE_A_RIGHT_KERNEL);

          viennacl::ocl::enqueue(kernel(
                                        A,
                                        D,
                                        static_cast<cl_uint>(row_start),
                                        static_cast<cl_uint>(col_start),
                                        static_cast<cl_uint>(A.size1()),
                                        static_cast<cl_uint>(A.size2()),
                                        static_cast<cl_uint>(A.internal_size2()),
                                        viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(SCALARTYPE)))
                                ));
        }

        {
          viennacl::ocl::kernel& kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::svd<SCALARTYPE>::program_name(), SVD_HOUSEHOLDER_UPDATE_QR_KERNEL);

          viennacl::ocl::enqueue(kernel(
                                        Q,
                                        D,
                                        static_cast<cl_uint>(A.size1()),
                                        static_cast<cl_uint>(A.size2()),
                                        static_cast<cl_uint>(Q.internal_size2()),
                                        viennacl::ocl::local_mem(static_cast<cl_uint>(128 * sizeof(SCALARTYPE)))
                                ));
        }

        return true;
      }

      template<typename SCALARTYPE, unsigned int ALIGNMENT>
      void bidiag(viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT> & Ai,
                  viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT> & QL,
                  viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT> & QR)
      {
        vcl_size_t row_num = Ai.size1();
        vcl_size_t col_num = Ai.size2();

        vcl_size_t to = std::min(row_num, col_num);
        vcl_size_t big_to = std::max(row_num, col_num);

        //for storing householder vector
        viennacl::vector<SCALARTYPE, ALIGNMENT> hh_vector(big_to, viennacl::traits::context(Ai));

        QL = viennacl::identity_matrix<SCALARTYPE>(QL.size1(), viennacl::traits::context(QL));
        QR = viennacl::identity_matrix<SCALARTYPE>(QR.size1(), viennacl::traits::context(QR));

        for (vcl_size_t i = 0; i < to; i++)
        {
          householder_c(Ai, QL, hh_vector, i, i);
          householder_r(Ai, QR, hh_vector, i, i+1);
        }
      }

    } // namespace detail


    /** @brief Computes the singular value decomposition of a matrix A. Experimental in 1.3.x
     *
     * @param A     The input matrix. Will be overwritten with a diagonal matrix containing the singular values on return
     * @param QL    The left orthogonal matrix
     * @param QR    The right orthogonal matrix
     */
    template<typename SCALARTYPE, unsigned int ALIGNMENT>
    void svd(viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT> & A,
              viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT> & QL,
              viennacl::matrix<SCALARTYPE, row_major, ALIGNMENT> & QR)
    {
      viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
      viennacl::linalg::opencl::kernels::svd<SCALARTYPE>::init(ctx);

      vcl_size_t row_num = A.size1();
      vcl_size_t col_num = A.size2();

      vcl_size_t to = std::min(row_num, col_num);


      //viennacl::vector<SCALARTYPE, ALIGNMENT> d(to);
      //viennacl::vector<SCALARTYPE, ALIGNMENT> s(to + 1);

      // first stage
      detail::bidiag(A, QL, QR);

      // second stage
      //std::vector<SCALARTYPE> dh(to, 0);
      //std::vector<SCALARTYPE> sh(to + 1, 0);
      boost::numeric::ublas::vector<SCALARTYPE> dh = boost::numeric::ublas::scalar_vector<SCALARTYPE>(to, 0);
      boost::numeric::ublas::vector<SCALARTYPE> sh = boost::numeric::ublas::scalar_vector<SCALARTYPE>(to + 1, 0);


      viennacl::linalg::opencl::bidiag_pack_svd(A, dh, sh);

      detail::svd_qr_shift( QL, QR, dh, sh);

      // Write resulting diagonal matrix with singular values to A:
      boost::numeric::ublas::matrix<SCALARTYPE> h_Sigma(row_num, col_num);
      h_Sigma.clear();

      for (vcl_size_t i = 0; i < to; i++)
        h_Sigma(i, i) = dh[i];

      copy(h_Sigma, A);
    }
  }
}
#endif
