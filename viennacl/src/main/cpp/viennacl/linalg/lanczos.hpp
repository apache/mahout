#ifndef VIENNACL_LINALG_LANCZOS_HPP_
#define VIENNACL_LINALG_LANCZOS_HPP_

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

/** @file viennacl/linalg/lanczos.hpp
*   @brief Generic interface for the Lanczos algorithm.
*
*   Contributed by Guenther Mader and Astrid Rupp.
*/

#include <cmath>
#include <vector>
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/norm_2.hpp"
#include "viennacl/io/matrix_market.hpp"
#include "viennacl/linalg/bisect.hpp"
#include "viennacl/tools/random.hpp"

namespace viennacl
{
namespace linalg
{

/** @brief A tag for the lanczos algorithm.
*/
class lanczos_tag
{
public:

  enum
  {
    partial_reorthogonalization = 0,
    full_reorthogonalization,
    no_reorthogonalization
  };

  /** @brief The constructor
  *
  * @param factor                 Exponent of epsilon - tolerance for batches of Reorthogonalization
  * @param numeig                 Number of eigenvalues to be returned
  * @param met                    Method for Lanczos-Algorithm: 0 for partial Reorthogonalization, 1 for full Reorthogonalization and 2 for Lanczos without Reorthogonalization
  * @param krylov                 Maximum krylov-space size
  */

  lanczos_tag(double factor = 0.75,
              vcl_size_t numeig = 10,
              int met = 0,
              vcl_size_t krylov = 100) : factor_(factor), num_eigenvalues_(numeig), method_(met), krylov_size_(krylov) {}

  /** @brief Sets the number of eigenvalues */
  void num_eigenvalues(vcl_size_t numeig){ num_eigenvalues_ = numeig; }

    /** @brief Returns the number of eigenvalues */
  vcl_size_t num_eigenvalues() const { return num_eigenvalues_; }

    /** @brief Sets the exponent of epsilon. Values between 0.6 and 0.9 usually give best results. */
  void factor(double fct) { factor_ = fct; }

  /** @brief Returns the exponent */
  double factor() const { return factor_; }

  /** @brief Sets the size of the kylov space. Must be larger than number of eigenvalues to compute. */
  void krylov_size(vcl_size_t max) { krylov_size_ = max; }

  /** @brief Returns the size of the kylov space */
  vcl_size_t  krylov_size() const { return krylov_size_; }

  /** @brief Sets the reorthogonalization method */
  void method(int met){ method_ = met; }

  /** @brief Returns the reorthogonalization method */
  int method() const { return method_; }


private:
  double factor_;
  vcl_size_t num_eigenvalues_;
  int method_; // see enum defined above for possible values
  vcl_size_t krylov_size_;
};


namespace detail
{
  /** @brief Inverse iteration for finding an eigenvector for an eigenvalue.
   *
   *  beta[0] to be ignored for consistency.
   */
  template<typename NumericT>
  void inverse_iteration(std::vector<NumericT> const & alphas, std::vector<NumericT> const & betas,
                         NumericT & eigenvalue, std::vector<NumericT> & eigenvector)
  {
    std::vector<NumericT> alpha_sweeped = alphas;
    for (vcl_size_t i=0; i<alpha_sweeped.size(); ++i)
      alpha_sweeped[i] -= eigenvalue;
    for (vcl_size_t row=1; row < alpha_sweeped.size(); ++row)
      alpha_sweeped[row] -= betas[row] * betas[row] / alpha_sweeped[row-1];

    // starting guess: ignore last equation
    eigenvector[alphas.size() - 1] = 1.0;

    for (vcl_size_t iter=0; iter<1; ++iter)
    {
      // solve first n-1 equations (A - \lambda I) y = -beta[n]
      eigenvector[alphas.size() - 1] /= alpha_sweeped[alphas.size() - 1];
      for (vcl_size_t row2=1; row2 < alphas.size(); ++row2)
      {
        vcl_size_t row = alphas.size() - row2 - 1;
        eigenvector[row] -= eigenvector[row+1] * betas[row+1];
        eigenvector[row] /= alpha_sweeped[row];
      }

      // normalize eigenvector:
      NumericT norm_vector = 0;
      for (vcl_size_t i=0; i<eigenvector.size(); ++i)
        norm_vector += eigenvector[i] * eigenvector[i];
      norm_vector = std::sqrt(norm_vector);
      for (vcl_size_t i=0; i<eigenvector.size(); ++i)
        eigenvector[i] /= norm_vector;
    }

    //eigenvalue = (alphas[0] * eigenvector[0] + betas[1] * eigenvector[1]) / eigenvector[0];
  }

  /**
  *   @brief Implementation of the Lanczos PRO algorithm (partial reorthogonalization)
  *
  *   @param A              The system matrix
  *   @param r              Random start vector
  *   @param eigenvectors_A Dense matrix holding the eigenvectors of A (one eigenvector per column)
  *   @param size           Size of krylov-space
  *   @param tag            Lanczos_tag with several options for the algorithm
  *   @param compute_eigenvectors   Boolean flag. If true, eigenvectors are computed. Otherwise the routine returns after calculating eigenvalues.
  *   @return               Returns the eigenvalues (number of eigenvalues equals size of krylov-space)
  */

  template<typename MatrixT, typename DenseMatrixT, typename NumericT>
  std::vector<NumericT>
  lanczosPRO (MatrixT const& A, vector_base<NumericT> & r, DenseMatrixT & eigenvectors_A, vcl_size_t size, lanczos_tag const & tag, bool compute_eigenvectors)
  {
    // generation of some random numbers, used for lanczos PRO algorithm
    viennacl::tools::normal_random_numbers<NumericT> get_N;

    std::vector<vcl_size_t> l_bound(size/2), u_bound(size/2);
    vcl_size_t n = r.size();
    std::vector<NumericT> w(size), w_old(size); //w_k, w_{k-1}

    NumericT inner_rt;
    std::vector<NumericT> alphas, betas;
    viennacl::matrix<NumericT, viennacl::column_major> Q(n, size); //column-major matrix holding the Krylov basis vectors

    bool second_step = false;
    NumericT eps = std::numeric_limits<NumericT>::epsilon();
    NumericT squ_eps = std::sqrt(eps);
    NumericT eta = std::exp(std::log(eps) * tag.factor());

    NumericT beta = viennacl::linalg::norm_2(r);

    r /= beta;

    viennacl::vector_base<NumericT> q_0(Q.handle(), Q.size1(), 0, 1);
    q_0 = r;

    viennacl::vector<NumericT> u = viennacl::linalg::prod(A, r);
    NumericT alpha = viennacl::linalg::inner_prod(u, r);
    alphas.push_back(alpha);
    w[0] = 1;
    betas.push_back(beta);

    vcl_size_t batches = 0;
    for (vcl_size_t i = 1; i < size; i++) // Main loop for setting up the Krylov space
    {
      viennacl::vector_base<NumericT> q_iminus1(Q.handle(), Q.size1(), (i-1) * Q.internal_size1(), 1);
      r = u - alpha * q_iminus1;
      beta = viennacl::linalg::norm_2(r);

      betas.push_back(beta);
      r = r / beta;

      //
      // Update recurrence relation for estimating orthogonality loss
      //
      w_old = w;
      w[0] = (betas[1] * w_old[1] + (alphas[0] - alpha) * w_old[0] - betas[i - 1] * w_old[0]) / beta + eps * 0.3 * get_N() * (betas[1] + beta);
      for (vcl_size_t j = 1; j < i - 1; j++)
        w[j] = (betas[j + 1] * w_old[j + 1] + (alphas[j] - alpha) * w_old[j] + betas[j] * w_old[j - 1] - betas[i - 1] * w_old[j]) / beta + eps * 0.3 * get_N() * (betas[j + 1] + beta);
      w[i-1] = 0.6 * eps * NumericT(n) * get_N() * betas[1] / beta;

      //
      // Check whether there has been a need for reorthogonalization detected in the previous iteration.
      // If so, run the reorthogonalization for each batch
      //
      if (second_step)
      {
        for (vcl_size_t j = 0; j < batches; j++)
        {
          for (vcl_size_t k = l_bound[j] + 1; k < u_bound[j] - 1; k++)
          {
            viennacl::vector_base<NumericT> q_k(Q.handle(), Q.size1(), k * Q.internal_size1(), 1);
            inner_rt = viennacl::linalg::inner_prod(r, q_k);
            r = r - inner_rt * q_k;
            w[k] = 1.5 * eps * get_N();
          }
        }
        NumericT temp = viennacl::linalg::norm_2(r);
        r = r / temp;
        beta = beta * temp;
        second_step = false;
      }
      batches = 0;

      //
      // Check for semiorthogonality
      //
      for (vcl_size_t j = 0; j < i; j++)
      {
        if (std::fabs(w[j]) >= squ_eps) // tentative loss of orthonormality, hence reorthonomalize
        {
          viennacl::vector_base<NumericT> q_j(Q.handle(), Q.size1(), j * Q.internal_size1(), 1);
          inner_rt = viennacl::linalg::inner_prod(r, q_j);
          r = r - inner_rt * q_j;
          w[j] = 1.5 * eps * get_N();
          vcl_size_t k = j - 1;

          // orthogonalization with respect to earlier basis vectors
          while (std::fabs(w[k]) > eta)
          {
            viennacl::vector_base<NumericT> q_k(Q.handle(), Q.size1(), k * Q.internal_size1(), 1);
            inner_rt = viennacl::linalg::inner_prod(r, q_k);
            r = r - inner_rt * q_k;
            w[k] = 1.5 * eps * get_N();
            if (k == 0) break;
            k--;
          }
          l_bound[batches] = k;

          // orthogonalization with respect to later basis vectors
          k = j + 1;
          while (k < i && std::fabs(w[k]) > eta)
          {
            viennacl::vector_base<NumericT> q_k(Q.handle(), Q.size1(), k * Q.internal_size1(), 1);
            inner_rt = viennacl::linalg::inner_prod(r, q_k);
            r = r - inner_rt * q_k;
            w[k] = 1.5 * eps * get_N();
            k++;
          }
          u_bound[batches] = k - 1;
          batches++;

          j = k-1; // go to end of current batch
        }
      }

      //
      // Normalize basis vector and reorthogonalize as needed
      //
      if (batches > 0)
      {
        NumericT temp = viennacl::linalg::norm_2(r);
        r = r / temp;
        beta = beta * temp;
        second_step = true;
      }

      // store Krylov vector in Q:
      viennacl::vector_base<NumericT> q_i(Q.handle(), Q.size1(), i * Q.internal_size1(), 1);
      q_i = r;

      //
      // determine and store alpha = <r, u> with u = A q_i - beta q_{i-1}
      //
      u = viennacl::linalg::prod(A, r);
      u += (-beta) * q_iminus1;
      alpha = viennacl::linalg::inner_prod(u, r);
      alphas.push_back(alpha);
    }

    //
    // Step 2: Compute eigenvalues of tridiagonal matrix obtained during Lanczos iterations:
    //
    std::vector<NumericT> eigenvalues = bisect(alphas, betas);

    //
    // Step 3: Compute eigenvectors via inverse iteration. Does not update eigenvalues, so only approximate by nature.
    //
    if (compute_eigenvectors)
    {
      std::vector<NumericT> eigenvector_tridiag(alphas.size());
      for (std::size_t i=0; i < tag.num_eigenvalues(); ++i)
      {
        // compute eigenvector of tridiagonal matrix via inverse:
        inverse_iteration(alphas, betas, eigenvalues[eigenvalues.size() - i - 1], eigenvector_tridiag);

        // eigenvector w of full matrix A. Given as w = Q * u, where u is the eigenvector of the tridiagonal matrix
        viennacl::vector<NumericT> eigenvector_u(eigenvector_tridiag.size());
        viennacl::copy(eigenvector_tridiag, eigenvector_u);

        viennacl::vector_base<NumericT> eigenvector_A(eigenvectors_A.handle(),
                                                      eigenvectors_A.size1(),
                                                      eigenvectors_A.row_major() ? i : i * eigenvectors_A.internal_size1(),
                                                      eigenvectors_A.row_major() ? eigenvectors_A.internal_size2() : 1);
        eigenvector_A = viennacl::linalg::prod(project(Q,
                                                       range(0, Q.size1()),
                                                       range(0, eigenvector_u.size())),
                                               eigenvector_u);
      }
    }

    return eigenvalues;
  }


  /**
  *   @brief Implementation of the Lanczos FRO algorithm
  *
  *   @param A            The system matrix
  *   @param r            Random start vector
  *   @param eigenvectors_A  A dense matrix in which the eigenvectors of A will be stored. Both row- and column-major matrices are supported.
  *   @param krylov_dim   Size of krylov-space
  *   @param tag          The Lanczos tag holding tolerances, etc.
  *   @param compute_eigenvectors   Boolean flag. If true, eigenvectors are computed. Otherwise the routine returns after calculating eigenvalues.
  *   @return             Returns the eigenvalues (number of eigenvalues equals size of krylov-space)
  */
  template< typename MatrixT, typename DenseMatrixT, typename NumericT>
  std::vector<NumericT>
  lanczos(MatrixT const& A, vector_base<NumericT> & r, DenseMatrixT & eigenvectors_A, vcl_size_t krylov_dim, lanczos_tag const & tag, bool compute_eigenvectors)
  {
    std::vector<NumericT> alphas, betas;
    viennacl::vector<NumericT> Aq(r.size());
    viennacl::matrix<NumericT, viennacl::column_major> Q(r.size(), krylov_dim + 1);  // Krylov basis (each Krylov vector is one column)

    NumericT norm_r = norm_2(r);
    NumericT beta = norm_r;
    r /= norm_r;

    // first Krylov vector:
    viennacl::vector_base<NumericT> q0(Q.handle(), Q.size1(), 0, 1);
    q0 = r;

    //
    // Step 1: Run Lanczos' method to obtain tridiagonal matrix
    //
    for (vcl_size_t i = 0; i < krylov_dim; i++)
    {
      betas.push_back(beta);
      // last available vector from Krylov basis:
      viennacl::vector_base<NumericT> q_i(Q.handle(), Q.size1(), i * Q.internal_size1(), 1);

      // Lanczos algorithm:
      // - Compute A * q:
      Aq = viennacl::linalg::prod(A, q_i);

      // - Form Aq <- Aq - <Aq, q_i> * q_i - beta * q_{i-1}, where beta is ||q_i|| before normalization in previous iteration
      NumericT alpha = viennacl::linalg::inner_prod(Aq, q_i);
      Aq -= alpha * q_i;

      if (i > 0)
      {
        viennacl::vector_base<NumericT> q_iminus1(Q.handle(), Q.size1(), (i-1) * Q.internal_size1(), 1);
        Aq -= beta * q_iminus1;

        // Extra measures for improved numerical stability?
        if (tag.method() == lanczos_tag::full_reorthogonalization)
        {
          // Gram-Schmidt (re-)orthogonalization:
          // TODO: Reuse fast (pipelined) routines from GMRES or GEMV
          for (vcl_size_t j = 0; j < i; j++)
          {
            viennacl::vector_base<NumericT> q_j(Q.handle(), Q.size1(), j * Q.internal_size1(), 1);
            NumericT inner_rq = viennacl::linalg::inner_prod(Aq, q_j);
            Aq -= inner_rq * q_j;
          }
        }
      }

      // normalize Aq and add to Krylov basis at column i+1 in Q:
      beta = viennacl::linalg::norm_2(Aq);
      viennacl::vector_base<NumericT> q_iplus1(Q.handle(), Q.size1(), (i+1) * Q.internal_size1(), 1);
      q_iplus1 = Aq / beta;

      alphas.push_back(alpha);
    }

    //
    // Step 2: Compute eigenvalues of tridiagonal matrix obtained during Lanczos iterations:
    //
    std::vector<NumericT> eigenvalues = bisect(alphas, betas);

    //
    // Step 3: Compute eigenvectors via inverse iteration. Does not update eigenvalues, so only approximate by nature.
    //
    if (compute_eigenvectors)
    {
      std::vector<NumericT> eigenvector_tridiag(alphas.size());
      for (std::size_t i=0; i < tag.num_eigenvalues(); ++i)
      {
        // compute eigenvector of tridiagonal matrix via inverse:
        inverse_iteration(alphas, betas, eigenvalues[eigenvalues.size() - i - 1], eigenvector_tridiag);

        // eigenvector w of full matrix A. Given as w = Q * u, where u is the eigenvector of the tridiagonal matrix
        viennacl::vector<NumericT> eigenvector_u(eigenvector_tridiag.size());
        viennacl::copy(eigenvector_tridiag, eigenvector_u);

        viennacl::vector_base<NumericT> eigenvector_A(eigenvectors_A.handle(),
                                                      eigenvectors_A.size1(),
                                                      eigenvectors_A.row_major() ? i : i * eigenvectors_A.internal_size1(),
                                                      eigenvectors_A.row_major() ? eigenvectors_A.internal_size2() : 1);
        eigenvector_A = viennacl::linalg::prod(project(Q,
                                                       range(0, Q.size1()),
                                                       range(0, eigenvector_u.size())),
                                               eigenvector_u);
      }
    }

    return eigenvalues;
  }

} // end namespace detail

/**
*   @brief Implementation of the calculation of eigenvalues using lanczos (with and without reorthogonalization).
*
*   Implementation of Lanczos with partial reorthogonalization is implemented separately.
*
*   @param matrix          The system matrix
*   @param eigenvectors_A  A dense matrix in which the eigenvectors of A will be stored. Both row- and column-major matrices are supported.
*   @param tag             Tag with several options for the lanczos algorithm
*   @param compute_eigenvectors   Boolean flag. If true, eigenvectors are computed. Otherwise the routine returns after calculating eigenvalues.
*   @return                Returns the n largest eigenvalues (n defined in the lanczos_tag)
*/
template<typename MatrixT, typename DenseMatrixT>
std::vector< typename viennacl::result_of::cpu_value_type<typename MatrixT::value_type>::type >
eig(MatrixT const & matrix, DenseMatrixT & eigenvectors_A, lanczos_tag const & tag, bool compute_eigenvectors = true)
{
  typedef typename viennacl::result_of::value_type<MatrixT>::type           NumericType;
  typedef typename viennacl::result_of::cpu_value_type<NumericType>::type   CPU_NumericType;
  typedef typename viennacl::result_of::vector_for_matrix<MatrixT>::type    VectorT;

  viennacl::tools::uniform_random_numbers<CPU_NumericType> random_gen;

  std::vector<CPU_NumericType> eigenvalues;
  vcl_size_t matrix_size = matrix.size1();
  VectorT r(matrix_size);
  std::vector<CPU_NumericType> s(matrix_size);

  for (vcl_size_t i=0; i<s.size(); ++i)
    s[i] = CPU_NumericType(0.5) + random_gen();

  detail::copy_vec_to_vec(s,r);

  vcl_size_t size_krylov = (matrix_size < tag.krylov_size()) ? matrix_size
                                                              : tag.krylov_size();

  switch (tag.method())
  {
  case lanczos_tag::partial_reorthogonalization:
    eigenvalues = detail::lanczosPRO(matrix, r, eigenvectors_A, size_krylov, tag, compute_eigenvectors);
    break;
  case lanczos_tag::full_reorthogonalization:
  case lanczos_tag::no_reorthogonalization:
    eigenvalues = detail::lanczos(matrix, r, eigenvectors_A, size_krylov, tag, compute_eigenvectors);
    break;
  }

  std::vector<CPU_NumericType> largest_eigenvalues;

  for (vcl_size_t i = 1; i<=tag.num_eigenvalues(); i++)
    largest_eigenvalues.push_back(eigenvalues[size_krylov-i]);


  return largest_eigenvalues;
}


/**
*   @brief Implementation of the calculation of eigenvalues using lanczos (with and without reorthogonalization).
*
*   Implementation of Lanczos with partial reorthogonalization is implemented separately.
*
*   @param matrix        The system matrix
*   @param tag           Tag with several options for the lanczos algorithm
*   @return              Returns the n largest eigenvalues (n defined in the lanczos_tag)
*/
template<typename MatrixT>
std::vector< typename viennacl::result_of::cpu_value_type<typename MatrixT::value_type>::type >
eig(MatrixT const & matrix, lanczos_tag const & tag)
{
  typedef typename viennacl::result_of::cpu_value_type<typename MatrixT::value_type>::type  NumericType;

  viennacl::matrix<NumericType> eigenvectors(matrix.size1(), tag.num_eigenvalues());
  return eig(matrix, eigenvectors, tag, false);
}

} // end namespace linalg
} // end namespace viennacl
#endif
