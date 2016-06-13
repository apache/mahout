#ifndef VIENNACL_GMRES_HPP_
#define VIENNACL_GMRES_HPP_

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

/** @file viennacl/linalg/gmres.hpp
    @brief Implementations of the generalized minimum residual method are in this file.
*/

#include <vector>
#include <cmath>
#include <limits>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/traits/clear.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/context.hpp"
#include "viennacl/meta/result_of.hpp"

#include "viennacl/linalg/iterative_operations.hpp"
#include "viennacl/vector_proxy.hpp"


namespace viennacl
{
namespace linalg
{

/** @brief A tag for the solver GMRES. Used for supplying solver parameters and for dispatching the solve() function
*/
class gmres_tag       //generalized minimum residual
{
public:
  /** @brief The constructor
  *
  * @param tol            Relative tolerance for the residual (solver quits if ||r|| < tol * ||r_initial||)
  * @param max_iterations The maximum number of iterations (including restarts
  * @param krylov_dim     The maximum dimension of the Krylov space before restart (number of restarts is found by max_iterations / krylov_dim)
  */
  gmres_tag(double tol = 1e-10, unsigned int max_iterations = 300, unsigned int krylov_dim = 20)
   : tol_(tol), abs_tol_(0), iterations_(max_iterations), krylov_dim_(krylov_dim), iters_taken_(0) {}

  /** @brief Returns the relative tolerance */
  double tolerance() const { return tol_; }

  /** @brief Returns the absolute tolerance */
  double abs_tolerance() const { return abs_tol_; }
  /** @brief Sets the absolute tolerance */
  void abs_tolerance(double new_tol) { if (new_tol >= 0) abs_tol_ = new_tol; }

  /** @brief Returns the maximum number of iterations */
  unsigned int max_iterations() const { return iterations_; }
  /** @brief Returns the maximum dimension of the Krylov space before restart */
  unsigned int krylov_dim() const { return krylov_dim_; }
  /** @brief Returns the maximum number of GMRES restarts */
  unsigned int max_restarts() const
  {
    unsigned int ret = iterations_ / krylov_dim_;
    if (ret > 0 && (ret * krylov_dim_ == iterations_) )
      return ret - 1;
    return ret;
  }

  /** @brief Return the number of solver iterations: */
  unsigned int iters() const { return iters_taken_; }
  /** @brief Set the number of solver iterations (should only be modified by the solver) */
  void iters(unsigned int i) const { iters_taken_ = i; }

  /** @brief Returns the estimated relative error at the end of the solver run */
  double error() const { return last_error_; }
  /** @brief Sets the estimated relative error at the end of the solver run */
  void error(double e) const { last_error_ = e; }

private:
  double tol_;
  double abs_tol_;
  unsigned int iterations_;
  unsigned int krylov_dim_;

  //return values from solver
  mutable unsigned int iters_taken_;
  mutable double last_error_;
};

namespace detail
{

  template<typename SrcVectorT, typename DestVectorT>
  void gmres_copy_helper(SrcVectorT const & src, DestVectorT & dest, vcl_size_t len, vcl_size_t start = 0)
  {
    for (vcl_size_t i=0; i<len; ++i)
      dest[start+i] = src[start+i];
  }

  template<typename NumericT, typename DestVectorT>
  void gmres_copy_helper(viennacl::vector<NumericT> const & src, DestVectorT & dest, vcl_size_t len, vcl_size_t start = 0)
  {
    typedef typename viennacl::vector<NumericT>::difference_type   difference_type;
    viennacl::copy( src.begin() + static_cast<difference_type>(start),
                    src.begin() + static_cast<difference_type>(start + len),
                   dest.begin() + static_cast<difference_type>(start));
  }

  /** @brief Computes the householder vector 'hh_vec' which rotates 'input_vec' such that all entries below the j-th entry of 'v' become zero.
    *
    * @param input_vec       The input vector
    * @param hh_vec          The householder vector defining the relection (I - beta * hh_vec * hh_vec^T)
    * @param beta            The coefficient beta in (I - beta  * hh_vec * hh_vec^T)
    * @param mu              The norm of the input vector part relevant for the reflection: norm_2(input_vec[j:size])
    * @param j               Index of the last nonzero index in 'input_vec' after applying the reflection
  */
  template<typename VectorT, typename NumericT>
  void gmres_setup_householder_vector(VectorT const & input_vec, VectorT & hh_vec, NumericT & beta, NumericT & mu, vcl_size_t j)
  {
    NumericT input_j = input_vec(j);

    // copy entries from input vector to householder vector:
    detail::gmres_copy_helper(input_vec, hh_vec, viennacl::traits::size(hh_vec) - (j+1), j+1);

    NumericT sigma = viennacl::linalg::norm_2(hh_vec);
    sigma *= sigma;

    if (sigma <= 0)
    {
      beta = 0;
      mu = input_j;
    }
    else
    {
      mu = std::sqrt(sigma + input_j*input_j);

      NumericT hh_vec_0 = (input_j <= 0) ? (input_j - mu) : (-sigma / (input_j + mu));

      beta = NumericT(2) * hh_vec_0 * hh_vec_0 / (sigma + hh_vec_0 * hh_vec_0);

      //divide hh_vec by its diagonal element hh_vec_0
      hh_vec /= hh_vec_0;
      hh_vec[j] = NumericT(1);
    }
  }

  // Apply (I - beta h h^T) to x (Householder reflection with Householder vector h)
  template<typename VectorT, typename NumericT>
  void gmres_householder_reflect(VectorT & x, VectorT const & h, NumericT beta)
  {
    NumericT hT_in_x = viennacl::linalg::inner_prod(h, x);
    x -= (beta * hT_in_x) * h;
  }


  /** @brief Implementation of a pipelined GMRES solver without preconditioner
  *
  * Following algorithm 2.1 proposed by Walker in "A Simpler GMRES", but uses classical Gram-Schmidt instead of modified Gram-Schmidt for better parallelization.
  * Uses some pipelining techniques for minimizing host-device transfer
  *
  * @param A            The system matrix
  * @param rhs          The load vector
  * @param tag          Solver configuration tag
  * @param monitor      A callback routine which is called at each GMRES restart
  * @param monitor_data Data pointer to be passed to the callback routine to pass on user-specific data
  * @return The result vector
  */
  template <typename MatrixType, typename ScalarType>
  viennacl::vector<ScalarType> pipelined_solve(MatrixType const & A,
                                               viennacl::vector<ScalarType> const & rhs,
                                               gmres_tag const & tag,
                                               viennacl::linalg::no_precond,
                                               bool (*monitor)(viennacl::vector<ScalarType> const &, ScalarType, void*) = NULL,
                                               void *monitor_data = NULL)
  {
    viennacl::vector<ScalarType> residual(rhs);
    viennacl::vector<ScalarType> result = viennacl::zero_vector<ScalarType>(rhs.size(), viennacl::traits::context(rhs));

    viennacl::vector<ScalarType> device_krylov_basis(rhs.internal_size() * tag.krylov_dim(), viennacl::traits::context(rhs)); // not using viennacl::matrix here because of spurious padding in column number
    viennacl::vector<ScalarType> device_buffer_R(tag.krylov_dim()*tag.krylov_dim(), viennacl::traits::context(rhs));
    std::vector<ScalarType>      host_buffer_R(device_buffer_R.size());

    vcl_size_t buffer_size_per_vector = 128;
    vcl_size_t num_buffer_chunks      = 3;
    viennacl::vector<ScalarType> device_inner_prod_buffer = viennacl::zero_vector<ScalarType>(num_buffer_chunks*buffer_size_per_vector, viennacl::traits::context(rhs)); // temporary buffer
    viennacl::vector<ScalarType> device_r_dot_vk_buffer   = viennacl::zero_vector<ScalarType>(buffer_size_per_vector * tag.krylov_dim(), viennacl::traits::context(rhs)); // holds result of first reduction stage for <r, v_k> on device
    viennacl::vector<ScalarType> device_vi_in_vk_buffer   = viennacl::zero_vector<ScalarType>(buffer_size_per_vector * tag.krylov_dim(), viennacl::traits::context(rhs)); // holds <v_i, v_k> for i=0..k-1 on device
    viennacl::vector<ScalarType> device_values_xi_k       = viennacl::zero_vector<ScalarType>(tag.krylov_dim(), viennacl::traits::context(rhs)); // holds values \xi_k = <r, v_k> on device
    std::vector<ScalarType>      host_r_dot_vk_buffer(device_r_dot_vk_buffer.size());
    std::vector<ScalarType>      host_values_xi_k(tag.krylov_dim());
    std::vector<ScalarType>      host_values_eta_k_buffer(tag.krylov_dim());
    std::vector<ScalarType>      host_update_coefficients(tag.krylov_dim());

    ScalarType norm_rhs = viennacl::linalg::norm_2(residual);
    ScalarType rho_0 = norm_rhs;
    ScalarType rho = ScalarType(1);

    tag.iters(0);

    for (unsigned int restart_count = 0; restart_count <= tag.max_restarts(); ++restart_count)
    {
      //
      // prepare restart:
      //
      if (restart_count > 0)
      {
        // compute new residual without introducing a temporary for A*x:
        residual = viennacl::linalg::prod(A, result);
        residual = rhs - residual;

        rho_0 = viennacl::linalg::norm_2(residual);
      }

      if (rho_0 <= ScalarType(tag.abs_tolerance()))  // trivial right hand side?
        break;

      residual /= rho_0;
      rho = ScalarType(1);

      // check for convergence:
      if (rho_0 / norm_rhs < tag.tolerance() || rho_0 < tag.abs_tolerance())
        break;

      //
      // minimize in Krylov basis:
      //
      vcl_size_t k = 0;
      for (k = 0; k < static_cast<vcl_size_t>(tag.krylov_dim()); ++k)
      {
        if (k == 0)
        {
          // compute v0 = A*r and perform first reduction stage for ||v0||
          viennacl::vector_range<viennacl::vector<ScalarType> > v0(device_krylov_basis, viennacl::range(0, rhs.size()));
          viennacl::linalg::pipelined_gmres_prod(A, residual, v0, device_inner_prod_buffer);

          // Normalize v_1 and compute first reduction stage for <r, v_0> in device_r_dot_vk_buffer:
          viennacl::linalg::pipelined_gmres_normalize_vk(v0, residual,
                                                         device_buffer_R, k*tag.krylov_dim() + k,
                                                         device_inner_prod_buffer, device_r_dot_vk_buffer,
                                                         buffer_size_per_vector, k*buffer_size_per_vector);
        }
        else
        {
          // compute v0 = A*r and perform first reduction stage for ||v0||
          viennacl::vector_range<viennacl::vector<ScalarType> > vk        (device_krylov_basis, viennacl::range( k   *rhs.internal_size(),  k   *rhs.internal_size() + rhs.size()));
          viennacl::vector_range<viennacl::vector<ScalarType> > vk_minus_1(device_krylov_basis, viennacl::range((k-1)*rhs.internal_size(), (k-1)*rhs.internal_size() + rhs.size()));
          viennacl::linalg::pipelined_gmres_prod(A, vk_minus_1, vk, device_inner_prod_buffer);

          //
          // Gram-Schmidt, stage 1: compute first reduction stage of <v_i, v_k>
          //
          viennacl::linalg::pipelined_gmres_gram_schmidt_stage1(device_krylov_basis, rhs.size(), rhs.internal_size(), k, device_vi_in_vk_buffer, buffer_size_per_vector);

          //
          // Gram-Schmidt, stage 2: compute second reduction stage of <v_i, v_k> and use that to compute v_k -= sum_i <v_i, v_k> v_i.
          //                        Store <v_i, v_k> in R-matrix and compute first reduction stage for ||v_k||
          //
          viennacl::linalg::pipelined_gmres_gram_schmidt_stage2(device_krylov_basis, rhs.size(), rhs.internal_size(), k,
                                                                device_vi_in_vk_buffer,
                                                                device_buffer_R, tag.krylov_dim(),
                                                                device_inner_prod_buffer, buffer_size_per_vector);

          //
          // Normalize v_k and compute first reduction stage for <r, v_k> in device_r_dot_vk_buffer:
          //
          viennacl::linalg::pipelined_gmres_normalize_vk(vk, residual,
                                                         device_buffer_R, k*tag.krylov_dim() + k,
                                                         device_inner_prod_buffer, device_r_dot_vk_buffer,
                                                         buffer_size_per_vector, k*buffer_size_per_vector);
        }
      }

      //
      // Run reduction to obtain the values \xi_k = <r, v_k>.
      // Note that unlike Algorithm 2.1 in Walker: "A Simpler GMRES", we do not update the residual
      //
      viennacl::fast_copy(device_r_dot_vk_buffer.begin(), device_r_dot_vk_buffer.end(), host_r_dot_vk_buffer.begin());
      for (std::size_t i=0; i<k; ++i)
      {
        host_values_xi_k[i] = ScalarType(0);
        for (std::size_t j=0; j<buffer_size_per_vector; ++j)
          host_values_xi_k[i] += host_r_dot_vk_buffer[i*buffer_size_per_vector + j];
      }

      //
      // Bring values in R  back to host:
      //
      viennacl::fast_copy(device_buffer_R.begin(), device_buffer_R.end(), host_buffer_R.begin());

      //
      // Check for premature convergence: If the diagonal element drops too far below the first norm, we're done and restrict the Krylov size accordingly.
      //
      vcl_size_t full_krylov_dim = k; //needed for proper access to R
      for (std::size_t i=0; i<k; ++i)
      {
        if (std::fabs(host_buffer_R[i + i*k]) < tag.tolerance() * host_buffer_R[0])
        {
          k = i;
          break;
        }
      }


      // Compute error estimator:
      for (std::size_t i=0; i<k; ++i)
      {
        tag.iters( tag.iters() + 1 ); //increase iteration counter

        // check for accumulation of round-off errors for poorly conditioned systems
        if (host_values_xi_k[i] >= rho || host_values_xi_k[i] <= -rho)
        {
          k = i;
          break;  // restrict Krylov space at this point. No gain from using additional basis vectors, since orthogonality is lost.
        }

        // update error estimator
        rho *= std::sin( std::acos(host_values_xi_k[i] / rho) );
      }

      //
      // Solve minimization problem:
      //
      host_values_eta_k_buffer = host_values_xi_k;

      for (int i2=static_cast<int>(k)-1; i2>-1; --i2)
      {
        vcl_size_t i = static_cast<vcl_size_t>(i2);
        for (vcl_size_t j=static_cast<vcl_size_t>(i)+1; j<k; ++j)
          host_values_eta_k_buffer[i] -= host_buffer_R[i + j*full_krylov_dim] * host_values_eta_k_buffer[j];

        host_values_eta_k_buffer[i] /= host_buffer_R[i + i*full_krylov_dim];
      }

      //
      // Update x += rho * z with z = \eta_0 * residual + sum_{i=0}^{k-1} \eta_{i+1} v_i
      // Note that we have not updated the residual yet, hence this slightly modified as compared to the form given in Algorithm 2.1 in Walker: "A Simpler GMRES"
      //
      for (vcl_size_t i=0; i<k; ++i)
        host_update_coefficients[i] = rho_0 * host_values_eta_k_buffer[i];

      viennacl::fast_copy(host_update_coefficients.begin(), host_update_coefficients.end(), device_values_xi_k.begin()); //reuse device_values_xi_k_buffer here for simplicity

      viennacl::linalg::pipelined_gmres_update_result(result, residual,
                                                      device_krylov_basis, rhs.size(), rhs.internal_size(),
                                                      device_values_xi_k, k);

      tag.error( std::fabs(rho*rho_0 / norm_rhs) );

      if (monitor && monitor(result, std::fabs(rho*rho_0 / norm_rhs), monitor_data))
        break;
    }

    return result;
  }

  /** @brief Overload for the pipelined CG implementation for the ViennaCL sparse matrix types */
  template<typename NumericT>
  viennacl::vector<NumericT> solve_impl(viennacl::compressed_matrix<NumericT> const & A,
                                        viennacl::vector<NumericT> const & rhs,
                                        gmres_tag const & tag,
                                        viennacl::linalg::no_precond,
                                        bool (*monitor)(viennacl::vector<NumericT> const &, NumericT, void*) = NULL,
                                        void *monitor_data = NULL)
  {
    return pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond(), monitor, monitor_data);
  }


  /** @brief Overload for the pipelined CG implementation for the ViennaCL sparse matrix types */
  template<typename NumericT>
  viennacl::vector<NumericT> solve_impl(viennacl::coordinate_matrix<NumericT> const & A,
                                        viennacl::vector<NumericT> const & rhs,
                                        gmres_tag const & tag,
                                        viennacl::linalg::no_precond,
                                        bool (*monitor)(viennacl::vector<NumericT> const &, NumericT, void*) = NULL,
                                        void *monitor_data = NULL)
  {
    return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond(), monitor, monitor_data);
  }



  /** @brief Overload for the pipelined CG implementation for the ViennaCL sparse matrix types */
  template<typename NumericT>
  viennacl::vector<NumericT> solve_impl(viennacl::ell_matrix<NumericT> const & A,
                                        viennacl::vector<NumericT> const & rhs,
                                        gmres_tag const & tag,
                                        viennacl::linalg::no_precond,
                                        bool (*monitor)(viennacl::vector<NumericT> const &, NumericT, void*) = NULL,
                                        void *monitor_data = NULL)
  {
    return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond(), monitor, monitor_data);
  }



  /** @brief Overload for the pipelined CG implementation for the ViennaCL sparse matrix types */
  template<typename NumericT>
  viennacl::vector<NumericT> solve_impl(viennacl::sliced_ell_matrix<NumericT> const & A,
                                        viennacl::vector<NumericT> const & rhs,
                                        gmres_tag const & tag,
                                        viennacl::linalg::no_precond,
                                        bool (*monitor)(viennacl::vector<NumericT> const &, NumericT, void*) = NULL,
                                        void *monitor_data = NULL)
  {
    return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond(), monitor, monitor_data);
  }


  /** @brief Overload for the pipelined CG implementation for the ViennaCL sparse matrix types */
  template<typename NumericT>
  viennacl::vector<NumericT> solve_impl(viennacl::hyb_matrix<NumericT> const & A,
                                        viennacl::vector<NumericT> const & rhs,
                                        gmres_tag const & tag,
                                        viennacl::linalg::no_precond,
                                        bool (*monitor)(viennacl::vector<NumericT> const &, NumericT, void*) = NULL,
                                        void *monitor_data = NULL)
  {
    return detail::pipelined_solve(A, rhs, tag, viennacl::linalg::no_precond(), monitor, monitor_data);
  }


  /** @brief Implementation of the GMRES solver.
  *
  * Following the algorithm proposed by Walker in "A Simpler GMRES"
  *
  * @param matrix       The system matrix
  * @param rhs          The load vector
  * @param tag          Solver configuration tag
  * @param precond      A preconditioner. Precondition operation is done via member function apply()
  * @param monitor      A callback routine which is called at each GMRES restart
  * @param monitor_data Data pointer to be passed to the callback routine to pass on user-specific data
  *
  * @return The result vector
  */
  template<typename MatrixT, typename VectorT, typename PreconditionerT>
  VectorT solve_impl(MatrixT const & matrix,
                     VectorT const & rhs,
                     gmres_tag const & tag,
                     PreconditionerT const & precond,
                     bool (*monitor)(VectorT const &, typename viennacl::result_of::cpu_value_type<typename viennacl::result_of::value_type<VectorT>::type>::type, void*) = NULL,
                     void *monitor_data = NULL)
  {
    typedef typename viennacl::result_of::value_type<VectorT>::type            NumericType;
    typedef typename viennacl::result_of::cpu_value_type<NumericType>::type    CPU_NumericType;

    unsigned int problem_size = static_cast<unsigned int>(viennacl::traits::size(rhs));
    VectorT result = rhs;
    viennacl::traits::clear(result);

    vcl_size_t krylov_dim = static_cast<vcl_size_t>(tag.krylov_dim());
    if (problem_size < krylov_dim)
      krylov_dim = problem_size; //A Krylov space larger than the matrix would lead to seg-faults (mathematically, error is certain to be zero already)

    VectorT res = rhs;
    VectorT v_k_tilde = rhs;
    VectorT v_k_tilde_temp = rhs;

    std::vector< std::vector<CPU_NumericType> > R(krylov_dim, std::vector<CPU_NumericType>(tag.krylov_dim()));
    std::vector<CPU_NumericType> projection_rhs(krylov_dim);

    std::vector<VectorT>          householder_reflectors(krylov_dim, rhs);
    std::vector<CPU_NumericType>  betas(krylov_dim);

    CPU_NumericType norm_rhs = viennacl::linalg::norm_2(rhs);

    if (norm_rhs <= tag.abs_tolerance()) //solution is zero if RHS norm is zero
      return result;

    tag.iters(0);

    for (unsigned int it = 0; it <= tag.max_restarts(); ++it)
    {
      //
      // (Re-)Initialize residual: r = b - A*x (without temporary for the result of A*x)
      //
      res = viennacl::linalg::prod(matrix, result);  //initial guess zero
      res = rhs - res;
      precond.apply(res);

      CPU_NumericType rho_0 = viennacl::linalg::norm_2(res);

      //
      // Check for premature convergence
      //
      if (rho_0 / norm_rhs < tag.tolerance() || rho_0 < tag.abs_tolerance()) // norm_rhs is known to be nonzero here
      {
        tag.error(rho_0 / norm_rhs);
        return result;
      }

      //
      // Normalize residual and set 'rho' to 1 as requested in 'A Simpler GMRES' by Walker and Zhou.
      //
      res /= rho_0;
      CPU_NumericType rho = static_cast<CPU_NumericType>(1.0);


      //
      // Iterate up until maximal Krylove space dimension is reached:
      //
      vcl_size_t k = 0;
      for (k = 0; k < krylov_dim; ++k)
      {
        tag.iters( tag.iters() + 1 ); //increase iteration counter

        // prepare storage:
        viennacl::traits::clear(R[k]);
        viennacl::traits::clear(householder_reflectors[k]);

        //compute v_k = A * v_{k-1} via Householder matrices
        if (k == 0)
        {
          v_k_tilde = viennacl::linalg::prod(matrix, res);
          precond.apply(v_k_tilde);
        }
        else
        {
          viennacl::traits::clear(v_k_tilde);
          v_k_tilde[k-1] = CPU_NumericType(1);

          //Householder rotations, part 1: Compute P_1 * P_2 * ... * P_{k-1} * e_{k-1}
          for (int i = static_cast<int>(k)-1; i > -1; --i)
            detail::gmres_householder_reflect(v_k_tilde, householder_reflectors[vcl_size_t(i)], betas[vcl_size_t(i)]);

          v_k_tilde_temp = viennacl::linalg::prod(matrix, v_k_tilde);
          precond.apply(v_k_tilde_temp);
          v_k_tilde = v_k_tilde_temp;

          //Householder rotations, part 2: Compute P_{k-1} * ... * P_{1} * v_k_tilde
          for (vcl_size_t i = 0; i < k; ++i)
            detail::gmres_householder_reflect(v_k_tilde, householder_reflectors[i], betas[i]);
        }

        //
        // Compute Householder reflection for v_k_tilde such that all entries below k-th entry are zero:
        //
        CPU_NumericType rho_k_k = 0;
        detail::gmres_setup_householder_vector(v_k_tilde, householder_reflectors[k], betas[k], rho_k_k, k);

        //
        // copy first k entries from v_k_tilde to R[k] in order to fill k-th column with result of
        // P_k * v_k_tilde = (v[0], ... , v[k-1], norm(v), 0, 0, ...) =: (rho_{1,k}, rho_{2,k}, ..., rho_{k,k}, 0, ..., 0);
        //
        detail::gmres_copy_helper(v_k_tilde, R[k], k);
        R[k][k] = rho_k_k;

        //
        // Update residual: r = P_k r
        // Set zeta_k = r[k] including machine precision considerations: mathematically we have |r[k]| <= rho
        // Set rho *= sin(acos(r[k] / rho))
        //
        detail::gmres_householder_reflect(res, householder_reflectors[k], betas[k]);

        if (res[k] > rho) //machine precision reached
          res[k] = rho;
        if (res[k] < -rho) //machine precision reached
          res[k] = -rho;
        projection_rhs[k] = res[k];

        rho *= std::sin( std::acos(projection_rhs[k] / rho) );

        if (std::fabs(rho * rho_0 / norm_rhs) < tag.tolerance())  // Residual is sufficiently reduced, stop here
        {
          tag.error( std::fabs(rho*rho_0 / norm_rhs) );
          ++k;
          break;
        }
      } // for k

      //
      // Triangular solver stage:
      //

      for (int i2=static_cast<int>(k)-1; i2>-1; --i2)
      {
        vcl_size_t i = static_cast<vcl_size_t>(i2);
        for (vcl_size_t j=i+1; j<k; ++j)
          projection_rhs[i] -= R[j][i] * projection_rhs[j];     //R is transposed

        projection_rhs[i] /= R[i][i];
      }

      //
      // Note: 'projection_rhs' now holds the solution (eta_1, ..., eta_k)
      //

      res *= projection_rhs[0];

      if (k > 0)
      {
        for (unsigned int i = 0; i < k-1; ++i)
          res[i] += projection_rhs[i+1];
      }

      //
      // Form z inplace in 'res' by applying P_1 * ... * P_{k}
      //
      for (int i=static_cast<int>(k)-1; i>=0; --i)
        detail::gmres_householder_reflect(res, householder_reflectors[vcl_size_t(i)], betas[vcl_size_t(i)]);

      res *= rho_0;
      result += res;  // x += rho_0 * z    in the paper

      //
      // Check for convergence:
      //
      tag.error(std::fabs(rho*rho_0 / norm_rhs));

      if (monitor && monitor(result, std::fabs(rho*rho_0 / norm_rhs), monitor_data))
        break;

      if ( tag.error() < tag.tolerance() )
        return result;
    }

    return result;
  }

}

template<typename MatrixT, typename VectorT, typename PreconditionerT>
VectorT solve(MatrixT const & matrix, VectorT const & rhs, gmres_tag const & tag, PreconditionerT const & precond)
{
  return detail::solve_impl(matrix, rhs, tag, precond);
}

/** @brief Convenience overload for calling the preconditioned BiCGStab solver using types from the C++ STL.
  *
  * A std::vector<std::map<T, U> > matrix is convenient for e.g. finite element assembly.
  * It is not the fastest option for setting up a system, but often it is fast enough - particularly for just trying things out.
  */
template<typename IndexT, typename NumericT, typename PreconditionerT>
std::vector<NumericT> solve(std::vector< std::map<IndexT, NumericT> > const & A, std::vector<NumericT> const & rhs, gmres_tag const & tag, PreconditionerT const & precond)
{
  viennacl::compressed_matrix<NumericT> vcl_A;
  viennacl::copy(A, vcl_A);

  viennacl::vector<NumericT> vcl_rhs(rhs.size());
  viennacl::copy(rhs, vcl_rhs);

  viennacl::vector<NumericT> vcl_result = solve(vcl_A, vcl_rhs, tag, precond);

  std::vector<NumericT> result(vcl_result.size());
  viennacl::copy(vcl_result, result);
  return result;
}

/** @brief Entry point for the unpreconditioned GMRES method.
 *
 *  @param A         The system matrix
 *  @param rhs       Right hand side vector (load vector)
 *  @param tag       A BiCGStab tag providing relative tolerances, etc.
 */

template<typename MatrixT, typename VectorT>
VectorT solve(MatrixT const & A, VectorT const & rhs, gmres_tag const & tag)
{
  return solve(A, rhs, tag, no_precond());
}



template<typename VectorT>
class gmres_solver
{
public:
  typedef typename viennacl::result_of::cpu_value_type<VectorT>::type   numeric_type;

  gmres_solver(gmres_tag const & tag) : tag_(tag), monitor_callback_(NULL), user_data_(NULL) {}

  template<typename MatrixT, typename PreconditionerT>
  VectorT operator()(MatrixT const & A, VectorT const & b, PreconditionerT const & precond) const
  {
    if (viennacl::traits::size(init_guess_) > 0) // take initial guess into account
    {
      VectorT mod_rhs = viennacl::linalg::prod(A, init_guess_);
      mod_rhs = b - mod_rhs;
      VectorT y = detail::solve_impl(A, mod_rhs, tag_, precond, monitor_callback_, user_data_);
      return init_guess_ + y;
    }
    return detail::solve_impl(A, b, tag_, precond, monitor_callback_, user_data_);
  }


  template<typename MatrixT>
  VectorT operator()(MatrixT const & A, VectorT const & b) const
  {
    return operator()(A, b, viennacl::linalg::no_precond());
  }

  /** @brief Specifies an initial guess for the iterative solver.
    *
    * An iterative solver for Ax = b with initial guess x_0 is equivalent to an iterative solver for Ay = b' := b - Ax_0, where x = x_0 + y.
    */
  void set_initial_guess(VectorT const & x) { init_guess_ = x; }

  /** @brief Sets a monitor function pointer to be called in each iteration. Set to NULL to run without monitor.
   *
   *  The monitor function is called with the current guess for the result as first argument and the current relative residual estimate as second argument.
   *  The third argument is a pointer to user-defined data, through which additional information can be passed.
   *  This pointer needs to be set with set_monitor_data. If not set, NULL is passed.
   *  If the montior function returns true, the solver terminates (either convergence or divergence).
   */
  void set_monitor(bool (*monitor_fun)(VectorT const &, numeric_type, void *), void *user_data)
  {
    monitor_callback_ = monitor_fun;
    user_data_ = user_data;
  }

  /** @brief Returns the solver tag containing basic configuration such as tolerances, etc. */
  gmres_tag const & tag() const { return tag_; }

private:
  gmres_tag  tag_;
  VectorT    init_guess_;
  bool       (*monitor_callback_)(VectorT const &, numeric_type, void *);
  void       *user_data_;
};


}
}

#endif
