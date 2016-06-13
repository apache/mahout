#ifndef VIENNACL_LINALG_HOST_BASED_NMF_OPERATIONS_HPP_
#define VIENNACL_LINALG_HOST_BASED_NMF_OPERATIONS_HPP_

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

/** @file viennacl/linalg/host_based/vector_operations.hpp
 @brief Implementations of NMF operations using a plain single-threaded or OpenMP-enabled execution on CPU
 */

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/norm_frobenius.hpp"

#include "viennacl/linalg/host_based/common.hpp"

namespace viennacl
{
namespace linalg
{

/** @brief Configuration class for the nonnegative-matrix-factorization algorithm. Specify tolerances, maximum iteration counts, etc., here. */
class nmf_config
{
public:
  nmf_config(double val_epsilon = 1e-4, double val_epsilon_stagnation = 1e-5,
      vcl_size_t num_max_iters = 10000, vcl_size_t num_check_iters = 100) :
      eps_(val_epsilon), stagnation_eps_(val_epsilon_stagnation), max_iters_(num_max_iters), check_after_steps_(
          (num_check_iters > 0) ? num_check_iters : 1), print_relative_error_(false), iters_(0)
  {
  }

  /** @brief Returns the relative tolerance for convergence */
  double tolerance() const
  {
    return eps_;
  }

  /** @brief Sets the relative tolerance for convergence, i.e. norm(V - W * H) / norm(V - W_init * H_init) */
  void tolerance(double e)
  {
    eps_ = e;
  }

  /** @brief Relative tolerance for the stagnation check */
  double stagnation_tolerance() const
  {
    return stagnation_eps_;
  }

  /** @brief Sets the tolerance for the stagnation check (i.e. the minimum required relative change of the residual between two iterations) */
  void stagnation_tolerance(double e)
  {
    stagnation_eps_ = e;
  }

  /** @brief Returns the maximum number of iterations for the NMF algorithm */
  vcl_size_t max_iterations() const
  {
    return max_iters_;
  }
  /** @brief Sets the maximum number of iterations for the NMF algorithm */
  void max_iterations(vcl_size_t m)
  {
    max_iters_ = m;
  }

  /** @brief Returns the number of iterations of the last NMF run using this configuration object */
  vcl_size_t iters() const
  {
    return iters_;
  }

  /** @brief Number of steps after which the convergence of NMF should be checked (again) */
  vcl_size_t check_after_steps() const
  {
    return check_after_steps_;
  }
  /** @brief Set the number of steps after which the convergence of NMF should be checked (again) */
  void check_after_steps(vcl_size_t c)
  {
    if (c > 0)
      check_after_steps_ = c;
  }

  /** @brief Returns the flag specifying whether the relative tolerance should be printed in each iteration */
  bool print_relative_error() const
  {
    return print_relative_error_;
  }
  /** @brief Specify whether the relative error should be printed at each convergence check after 'num_check_iters' steps */
  void print_relative_error(bool b)
  {
    print_relative_error_ = b;
  }

  template<typename ScalarType>
  friend void nmf(viennacl::matrix_base<ScalarType> const & V,
      viennacl::matrix_base<ScalarType> & W, viennacl::matrix_base<ScalarType> & H,
      nmf_config const & conf);

private:
  double eps_;
  double stagnation_eps_;
  vcl_size_t max_iters_;
  vcl_size_t check_after_steps_;
  bool print_relative_error_;
public:
  mutable vcl_size_t iters_;
};

namespace host_based
{
  /** @brief Missing OpenMP kernel for nonnegative matrix factorization of a dense matrices. */
  template<typename NumericT>
  void el_wise_mul_div(NumericT       * matrix1,
                       NumericT const * matrix2,
                       NumericT const * matrix3, vcl_size_t size)
  {
#ifdef VIENNACL_WITH_OPENMP
#pragma omp parallel for
#endif
    for (long i2 = 0; i2 < long(size); i2++)
    {
      vcl_size_t i = vcl_size_t(i2);
      NumericT val     = matrix1[i] * matrix2[i];
      NumericT divisor = matrix3[i];
      matrix1[i] = (divisor > (NumericT) 0.00001) ? (val / divisor) : (NumericT) 0;
    }
  }

  /** @brief The nonnegative matrix factorization (approximation) algorithm as suggested by Lee and Seung. Factorizes a matrix V with nonnegative entries into matrices W and H such that ||V - W*H|| is minimized.
   *
   * @param V     Input matrix
   * @param W     First factor
   * @param H     Second factor
   * @param conf  A configuration object holding tolerances and the like
   */
  template<typename NumericT>
  void nmf(viennacl::matrix_base<NumericT> const & V,
           viennacl::matrix_base<NumericT> & W,
           viennacl::matrix_base<NumericT> & H,
           viennacl::linalg::nmf_config const & conf)
  {
    vcl_size_t k = W.size2();
    conf.iters_ = 0;

    if (viennacl::linalg::norm_frobenius(W) <= 0)
      W = viennacl::scalar_matrix<NumericT>(W.size1(), W.size2(), NumericT(1.0));

    if (viennacl::linalg::norm_frobenius(H) <= 0)
      H = viennacl::scalar_matrix<NumericT>(H.size1(), H.size2(), NumericT(1.0));

    viennacl::matrix_base<NumericT> wn(V.size1(), k, W.row_major());
    viennacl::matrix_base<NumericT> wd(V.size1(), k, W.row_major());
    viennacl::matrix_base<NumericT> wtmp(V.size1(), V.size2(), W.row_major());

    viennacl::matrix_base<NumericT> hn(k, V.size2(), H.row_major());
    viennacl::matrix_base<NumericT> hd(k, V.size2(), H.row_major());
    viennacl::matrix_base<NumericT> htmp(k, k, H.row_major());

    viennacl::matrix_base<NumericT> appr(V.size1(), V.size2(), V.row_major());

    NumericT last_diff = 0;
    NumericT diff_init = 0;
    bool stagnation_flag = false;

    for (vcl_size_t i = 0; i < conf.max_iterations(); i++)
    {
      conf.iters_ = i + 1;

      hn   = viennacl::linalg::prod(trans(W), V);
      htmp = viennacl::linalg::prod(trans(W), W);
      hd   = viennacl::linalg::prod(htmp, H);

      NumericT * data_H  = detail::extract_raw_pointer<NumericT>(H);
      NumericT * data_hn = detail::extract_raw_pointer<NumericT>(hn);
      NumericT * data_hd = detail::extract_raw_pointer<NumericT>(hd);

      viennacl::linalg::host_based::el_wise_mul_div(data_H, data_hn, data_hd, H.internal_size1() * H.internal_size2());

      wn   = viennacl::linalg::prod(V, trans(H));
      wtmp = viennacl::linalg::prod(W, H);
      wd   = viennacl::linalg::prod(wtmp, trans(H));

      NumericT * data_W  = detail::extract_raw_pointer<NumericT>(W);
      NumericT * data_wn = detail::extract_raw_pointer<NumericT>(wn);
      NumericT * data_wd = detail::extract_raw_pointer<NumericT>(wd);

      viennacl::linalg::host_based::el_wise_mul_div(data_W, data_wn, data_wd, W.internal_size1() * W.internal_size2());

      if (i % conf.check_after_steps() == 0)  //check for convergence
      {
        appr = viennacl::linalg::prod(W, H);

        appr -= V;
        NumericT diff_val = viennacl::linalg::norm_frobenius(appr);

        if (i == 0)
          diff_init = diff_val;

        if (conf.print_relative_error())
          std::cout << diff_val / diff_init << std::endl;

        // Approximation check
        if (diff_val / diff_init < conf.tolerance())
          break;

        // Stagnation check
        if (std::fabs(diff_val - last_diff) / (diff_val * NumericT(conf.check_after_steps())) < conf.stagnation_tolerance()) //avoid situations where convergence stagnates
        {
          if (stagnation_flag)    // iteration stagnates (two iterates with no notable progress)
            break;
          else
            // record stagnation in this iteration
            stagnation_flag = true;
        } else
          // good progress in this iteration, so unset stagnation flag
          stagnation_flag = false;

        // prepare for next iterate:
        last_diff = diff_val;
      }
    }
  }

} //namespace host_based
} //namespace linalg
} //namespace viennacl

#endif /* VIENNACL_LINALG_HOST_BASED_NMF_OPERATIONS_HPP_ */
