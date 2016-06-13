#ifndef VIENNACL_LINALG_MIXED_PRECISION_CG_HPP_
#define VIENNACL_LINALG_MIXED_PRECISION_CG_HPP_

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

/** @file viennacl/linalg/mixed_precision_cg.hpp
    @brief The conjugate gradient method using mixed precision is implemented here. Experimental.
*/

#include <vector>
#include <map>
#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/traits/clear.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/meta/result_of.hpp"
#include "viennacl/backend/memory.hpp"

#include "viennacl/vector_proxy.hpp"

namespace viennacl
{
  namespace linalg
  {

    /** @brief A tag for the conjugate gradient Used for supplying solver parameters and for dispatching the solve() function
    */
    class mixed_precision_cg_tag
    {
      public:
        /** @brief The constructor
        *
        * @param tol              Relative tolerance for the residual (solver quits if ||r|| < tol * ||r_initial||)
        * @param max_iterations   The maximum number of iterations
        * @param inner_tol        Inner tolerance for the low-precision iterations
        */
        mixed_precision_cg_tag(double tol = 1e-8, unsigned int max_iterations = 300, float inner_tol = 1e-2f) : tol_(tol), iterations_(max_iterations), inner_tol_(inner_tol) {}

        /** @brief Returns the relative tolerance */
        double tolerance() const { return tol_; }
        /** @brief Returns the relative tolerance */
        float inner_tolerance() const { return inner_tol_; }
        /** @brief Returns the maximum number of iterations */
        unsigned int max_iterations() const { return iterations_; }

        /** @brief Return the number of solver iterations: */
        unsigned int iters() const { return iters_taken_; }
        void iters(unsigned int i) const { iters_taken_ = i; }

        /** @brief Returns the estimated relative error at the end of the solver run */
        double error() const { return last_error_; }
        /** @brief Sets the estimated relative error at the end of the solver run */
        void error(double e) const { last_error_ = e; }


      private:
        double tol_;
        unsigned int iterations_;
        float inner_tol_;

        //return values from solver
        mutable unsigned int iters_taken_;
        mutable double last_error_;
    };


    /** @brief Implementation of the conjugate gradient solver without preconditioner
    *
    * Following the algorithm in the book by Y. Saad "Iterative Methods for sparse linear systems"
    *
    * @param matrix     The system matrix
    * @param rhs        The load vector
    * @param tag        Solver configuration tag
    * @return The result vector
    */
    template<typename MatrixType, typename VectorType>
    VectorType solve(const MatrixType & matrix, VectorType const & rhs, mixed_precision_cg_tag const & tag)
    {
      //typedef typename VectorType::value_type      ScalarType;
      typedef typename viennacl::result_of::cpu_value_type<VectorType>::type    CPU_ScalarType;

      //std::cout << "Starting CG" << std::endl;
      vcl_size_t problem_size = viennacl::traits::size(rhs);
      VectorType result(rhs);
      viennacl::traits::clear(result);

      VectorType residual = rhs;

      CPU_ScalarType ip_rr = viennacl::linalg::inner_prod(rhs, rhs);
      CPU_ScalarType new_ip_rr = 0;
      CPU_ScalarType norm_rhs_squared = ip_rr;

      if (norm_rhs_squared <= 0) //solution is zero if RHS norm is zero
        return result;

      viennacl::vector<float> residual_low_precision(problem_size, viennacl::traits::context(rhs));
      viennacl::vector<float> result_low_precision(problem_size, viennacl::traits::context(rhs));
      viennacl::vector<float> p_low_precision(problem_size, viennacl::traits::context(rhs));
      viennacl::vector<float> tmp_low_precision(problem_size, viennacl::traits::context(rhs));
      float inner_ip_rr = static_cast<float>(ip_rr);
      float new_inner_ip_rr = 0;
      float initial_inner_rhs_norm_squared = static_cast<float>(ip_rr);
      float alpha;
      float beta;

      // transfer rhs to single precision:
      p_low_precision = rhs;
      residual_low_precision = p_low_precision;

      // transfer matrix to single precision:
      viennacl::compressed_matrix<float> matrix_low_precision(matrix.size1(), matrix.size2(), matrix.nnz(), viennacl::traits::context(rhs));
      viennacl::backend::memory_copy(matrix.handle1(), const_cast<viennacl::backend::mem_handle &>(matrix_low_precision.handle1()), 0, 0, matrix_low_precision.handle1().raw_size() );
      viennacl::backend::memory_copy(matrix.handle2(), const_cast<viennacl::backend::mem_handle &>(matrix_low_precision.handle2()), 0, 0, matrix_low_precision.handle2().raw_size() );

      viennacl::vector_base<CPU_ScalarType> matrix_elements_high_precision(const_cast<viennacl::backend::mem_handle &>(matrix.handle()), matrix.nnz(), 0, 1);
      viennacl::vector_base<float>          matrix_elements_low_precision(matrix_low_precision.handle(), matrix.nnz(), 0, 1);
      matrix_elements_low_precision = matrix_elements_high_precision;
      matrix_low_precision.generate_row_block_information();

      for (unsigned int i = 0; i < tag.max_iterations(); ++i)
      {
        tag.iters(i+1);

        // lower precision 'inner iteration'
        tmp_low_precision = viennacl::linalg::prod(matrix_low_precision, p_low_precision);

        alpha = inner_ip_rr / viennacl::linalg::inner_prod(tmp_low_precision, p_low_precision);
        result_low_precision += alpha * p_low_precision;
        residual_low_precision -= alpha * tmp_low_precision;

        new_inner_ip_rr = viennacl::linalg::inner_prod(residual_low_precision, residual_low_precision);

        beta = new_inner_ip_rr / inner_ip_rr;
        inner_ip_rr = new_inner_ip_rr;

        p_low_precision = residual_low_precision + beta * p_low_precision;

        //
        // If enough progress has been achieved, update current residual with high precision evaluation
        // This is effectively a restart of the CG method
        //
        if (new_inner_ip_rr < tag.inner_tolerance() * initial_inner_rhs_norm_squared || i == tag.max_iterations()-1)
        {
          residual = result_low_precision; // reusing residual vector as temporary buffer for conversion. Overwritten below anyway
          result += residual;

          // residual = b - Ax  (without introducing a temporary)
          residual = viennacl::linalg::prod(matrix, result);
          residual = rhs - residual;

          new_ip_rr = viennacl::linalg::inner_prod(residual, residual);
          if (new_ip_rr / norm_rhs_squared < tag.tolerance() *  tag.tolerance())//squared norms involved here
            break;

          p_low_precision = residual;

          result_low_precision.clear();
          residual_low_precision = p_low_precision;
          initial_inner_rhs_norm_squared = static_cast<float>(new_ip_rr);
          inner_ip_rr = static_cast<float>(new_ip_rr);
        }
      }

      //store last error estimate:
      tag.error(std::sqrt(new_ip_rr / norm_rhs_squared));

      return result;
    }

    template<typename MatrixType, typename VectorType>
    VectorType solve(const MatrixType & matrix, VectorType const & rhs, mixed_precision_cg_tag const & tag, viennacl::linalg::no_precond)
    {
      return solve(matrix, rhs, tag);
    }


  }
}

#endif
