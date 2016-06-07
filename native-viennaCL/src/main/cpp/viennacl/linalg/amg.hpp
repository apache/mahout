#ifndef VIENNACL_LINALG_AMG_HPP_
#define VIENNACL_LINALG_AMG_HPP_

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

/** @file viennacl/linalg/amg.hpp
    @brief Main include file for algebraic multigrid (AMG) preconditioners.  Experimental.

    Implementation contributed by Markus Wagner
*/

#include <vector>
#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/compressed_matrix.hpp"

#include "viennacl/linalg/detail/amg/amg_base.hpp"
#include "viennacl/linalg/sparse_matrix_operations.hpp"
#include "viennacl/linalg/amg_operations.hpp"
#include "viennacl/tools/timer.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/lu.hpp"

#include <map>

#ifdef VIENNACL_WITH_OPENMP
 #include <omp.h>
#endif

#define VIENNACL_AMG_MAX_LEVELS 20

namespace viennacl
{
namespace linalg
{

class amg_coarse_problem_too_large_exception : public std::runtime_error
{
public:
  amg_coarse_problem_too_large_exception(std::string const & msg, vcl_size_t num_points) : std::runtime_error(msg), c_points_(num_points) {}

  /** @brief Returns the number of coarse points for which no further coarsening could be applied */
  vcl_size_t coarse_points() const { return c_points_; }

private:
  vcl_size_t c_points_;
};


namespace detail
{
  /** @brief Sparse Galerkin product: Calculates A_coarse = trans(P)*A_fine*P = R*A_fine*P
    *
    * @param A_fine    Operator matrix on fine grid (quadratic)
    * @param P         Prolongation/Interpolation matrix
    * @param R         Restriction matrix
    * @param A_coarse  Result matrix on coarse grid (Galerkin operator)
    */
  template<typename NumericT>
  void amg_galerkin_prod(compressed_matrix<NumericT> & A_fine,
                         compressed_matrix<NumericT> & P,
                         compressed_matrix<NumericT> & R, //P^T
                         compressed_matrix<NumericT> & A_coarse)
  {

    compressed_matrix<NumericT> A_fine_times_P(viennacl::traits::context(A_fine));

    // transpose P in memory (no known way of efficiently multiplying P^T * B for CSR-matrices P and B):
    viennacl::linalg::detail::amg::amg_transpose(P, R);

    // compute Galerkin product using a temporary for the result of A_fine * P
    A_fine_times_P = viennacl::linalg::prod(A_fine, P);
    A_coarse = viennacl::linalg::prod(R, A_fine_times_P);

  }


  /** @brief Setup AMG preconditioner
  *
  * @param list_of_A                  Operator matrices on all levels
  * @param list_of_P                  Prolongation/Interpolation operators on all levels
  * @param list_of_R                  Restriction operators on all levels
  * @param list_of_amg_level_context  Auxiliary datastructures for managing the grid hierarchy (coarse nodes, etc.)
  * @param tag                        AMG preconditioner tag
  */
  template<typename NumericT, typename AMGContextListT>
  vcl_size_t amg_setup(std::vector<compressed_matrix<NumericT> > & list_of_A,
                       std::vector<compressed_matrix<NumericT> > & list_of_P,
                       std::vector<compressed_matrix<NumericT> > & list_of_R,
                       AMGContextListT & list_of_amg_level_context,
                       amg_tag & tag)
  {
    // Set number of iterations. If automatic coarse grid construction is chosen (0), then set a maximum size and stop during the process.
    vcl_size_t iterations = tag.get_coarse_levels();
    if (iterations == 0)
      iterations = VIENNACL_AMG_MAX_LEVELS;

    for (vcl_size_t i=0; i<iterations; ++i)
    {
      list_of_amg_level_context[i].switch_context(tag.get_setup_context());
      list_of_amg_level_context[i].resize(list_of_A[i].size1(), list_of_A[i].nnz());

      // Construct C and F points on coarse level (i is fine level, i+1 coarse level).
      detail::amg::amg_coarse(list_of_A[i], list_of_amg_level_context[i], tag);

      // Calculate number of C and F points on level i.
      unsigned int c_points = list_of_amg_level_context[i].num_coarse_;
      unsigned int f_points = static_cast<unsigned int>(list_of_A[i].size1()) - c_points;

      if (f_points == 0 && c_points > tag.get_coarsening_cutoff())
      {
        std::stringstream ss;
        ss << "No further coarsening possible (" << c_points << " coarse points). Consider changing the strong connection threshold or increasing the coarsening cutoff." << std::endl;
        throw amg_coarse_problem_too_large_exception(ss.str(), c_points);
      }

      // Stop routine when the maximal coarse level is found (no C or F point). Coarsest level is level i.
      if (c_points == 0 || f_points == 0)
        break;

      // Construct interpolation matrix for level i.
      detail::amg::amg_interpol(list_of_A[i], list_of_P[i], list_of_amg_level_context[i], tag);

      // Compute coarse grid operator (A[i+1] = R * A[i] * P) with R = trans(P).
      amg_galerkin_prod(list_of_A[i], list_of_P[i], list_of_R[i], list_of_A[i+1]);

      // send matrices to target context:
      list_of_A[i].switch_memory_context(tag.get_target_context());
      list_of_P[i].switch_memory_context(tag.get_target_context());
      list_of_R[i].switch_memory_context(tag.get_target_context());

      // If Limit of coarse points is reached then stop. Coarsest level is level i+1.
      if (tag.get_coarse_levels() == 0 && c_points <= tag.get_coarsening_cutoff())
        return i+1;
    }

    return iterations;
  }


  /** @brief Initialize AMG preconditioner
  *
  * @param mat                        System matrix
  * @param list_of_A                  Operator matrices on all levels
  * @param list_of_P                  Prolongation/Interpolation operators on all levels
  * @param list_of_R                  Restriction operators on all levels
  * @param list_of_amg_level_context  Auxiliary datastructures for managing the grid hierarchy (coarse nodes, etc.)
  * @param tag                        AMG preconditioner tag
  */
  template<typename MatrixT, typename InternalT1, typename InternalT2>
  void amg_init(MatrixT const & mat, InternalT1 & list_of_A, InternalT1 & list_of_P, InternalT1 & list_of_R, InternalT2 & list_of_amg_level_context, amg_tag & tag)
  {
    typedef typename InternalT1::value_type SparseMatrixType;

    vcl_size_t num_levels = (tag.get_coarse_levels() > 0) ? tag.get_coarse_levels() : VIENNACL_AMG_MAX_LEVELS;

    list_of_A.resize(num_levels+1, SparseMatrixType(tag.get_setup_context()));
    list_of_P.resize(num_levels,   SparseMatrixType(tag.get_setup_context()));
    list_of_R.resize(num_levels,   SparseMatrixType(tag.get_setup_context()));
    list_of_amg_level_context.resize(num_levels);

    // Insert operator matrix as operator for finest level.
    //SparseMatrixType A0(mat);
    //A.insert_element(0, A0);
    list_of_A[0].switch_memory_context(viennacl::traits::context(mat));
    list_of_A[0] = mat;
    list_of_A[0].switch_memory_context(tag.get_setup_context());
  }

  /** @brief Setup data structures for precondition phase for later use on the GPU
  *
  * @param result          Result vector on all levels
  * @param result_backup   Copy of result vector on all levels
  * @param rhs             RHS vector on all levels
  * @param residual        Residual vector on all levels
  * @param A               Operators matrices on all levels from setup phase
  * @param coarse_levels   Number of coarse levels for which the datastructures should be set up.
  * @param tag             AMG preconditioner tag
  */
  template<typename InternalVectorT, typename SparseMatrixT>
  void amg_setup_apply(InternalVectorT & result,
                       InternalVectorT & result_backup,
                       InternalVectorT & rhs,
                       InternalVectorT & residual,
                       SparseMatrixT const & A,
                       vcl_size_t coarse_levels,
                       amg_tag const & tag)
  {
    typedef typename InternalVectorT::value_type VectorType;

    result.resize(coarse_levels + 1);
    result_backup.resize(coarse_levels + 1);
    rhs.resize(coarse_levels + 1);
    residual.resize(coarse_levels);

    for (vcl_size_t level=0; level <= coarse_levels; ++level)
    {
             result[level] = VectorType(A[level].size1(), tag.get_target_context());
      result_backup[level] = VectorType(A[level].size1(), tag.get_target_context());
                rhs[level] = VectorType(A[level].size1(), tag.get_target_context());
    }
    for (vcl_size_t level=0; level < coarse_levels; ++level)
    {
      residual[level] = VectorType(A[level].size1(), tag.get_target_context());
    }
  }


  /** @brief Pre-compute LU factorization for direct solve (ublas library).
  *
  * Speeds up precondition phase as this is computed only once overall instead of once per iteration.
  *
  * @param op           Operator matrix for direct solve
  * @param A            Operator matrix on coarsest level
  * @param tag          AMG preconditioner tag
  */
  template<typename NumericT, typename SparseMatrixT>
  void amg_lu(viennacl::matrix<NumericT> & op,
              SparseMatrixT const & A,
              amg_tag const & tag)
  {
    op.switch_memory_context(tag.get_setup_context());
    op.resize(A.size1(), A.size2(), false);
    viennacl::linalg::detail::amg::assign_to_dense(A, op);

    viennacl::linalg::lu_factorize(op);
    op.switch_memory_context(tag.get_target_context());
  }

}

/** @brief AMG preconditioner class, can be supplied to solve()-routines
*/
template<typename MatrixT>
class amg_precond;


/** @brief AMG preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename NumericT, unsigned int AlignmentV>
class amg_precond< compressed_matrix<NumericT, AlignmentV> >
{
  typedef viennacl::compressed_matrix<NumericT, AlignmentV> SparseMatrixType;
  typedef viennacl::vector<NumericT>                        VectorType;
  typedef detail::amg::amg_level_context                    AMGContextType;

public:

  amg_precond() {}

  /** @brief The constructor. Builds data structures.
  *
  * @param mat  System matrix
  * @param tag  The AMG tag
  */
  amg_precond(compressed_matrix<NumericT, AlignmentV> const & mat,
              amg_tag const & tag)
  {
    tag_ = tag;

    // Initialize data structures.
    detail::amg_init(mat, A_list_, P_list_, R_list_, amg_context_list_, tag_);
  }

  /** @brief Start setup phase for this class and copy data structures.
  */
  void setup()
  {
    // Start setup phase.
    vcl_size_t num_coarse_levels = detail::amg_setup(A_list_, P_list_, R_list_, amg_context_list_, tag_);

    // Setup precondition phase (Data structures).
    detail::amg_setup_apply(result_list_, result_backup_list_, rhs_list_, residual_list_, A_list_, num_coarse_levels, tag_);

    // LU factorization for direct solve.
    detail::amg_lu(coarsest_op_, A_list_[num_coarse_levels], tag_);
  }


  /** @brief Precondition Operation
  *
  * @param vec       The vector to which preconditioning is applied to
  */
  template<typename VectorT>
  void apply(VectorT & vec) const
  {
    vcl_size_t level;

    // Precondition operation (Yang, p.3).
    rhs_list_[0] = vec;

    // Part 1: Restrict down to coarsest level
    for (level=0; level < residual_list_.size(); level++)
    {
      result_list_[level].clear();

      // Apply Smoother presmooth_ times.
      viennacl::linalg::detail::amg::smooth_jacobi(static_cast<unsigned int>(tag_.get_presmooth_steps()),
                                                   A_list_[level],
                                                   result_list_[level],
                                                   result_backup_list_[level],
                                                   rhs_list_[level],
                                                   static_cast<NumericT>(tag_.get_jacobi_weight()));

      // Compute residual.
      //residual[level] = rhs_[level] - viennacl::linalg::prod(A_[level], result_[level]);
      residual_list_[level] = viennacl::linalg::prod(A_list_[level], result_list_[level]);
      residual_list_[level] = rhs_list_[level] - residual_list_[level];

      // Restrict to coarse level. Result is RHS of coarse level equation.
      //residual_coarse[level] = viennacl::linalg::prod(R[level],residual[level]);
      rhs_list_[level+1] = viennacl::linalg::prod(R_list_[level], residual_list_[level]);
    }

    // Part 2: On highest level use direct solve to solve equation (on the CPU)
    result_list_[level] = rhs_list_[level];
    viennacl::linalg::lu_substitute(coarsest_op_, result_list_[level]);

    // Part 3: Prolongation to finest level
    for (int level2 = static_cast<int>(residual_list_.size()-1); level2 >= 0; level2--)
    {
      level = static_cast<vcl_size_t>(level2);

      // Interpolate error to fine level and correct solution.
      result_backup_list_[level] = viennacl::linalg::prod(P_list_[level], result_list_[level+1]);
      result_list_[level] += result_backup_list_[level];

      // Apply Smoother postsmooth_ times.
      viennacl::linalg::detail::amg::smooth_jacobi(static_cast<unsigned int>(tag_.get_postsmooth_steps()),
                                                   A_list_[level],
                                                   result_list_[level],
                                                   result_backup_list_[level],
                                                   rhs_list_[level],
                                                   static_cast<NumericT>(tag_.get_jacobi_weight()));
    }
    vec = result_list_[0];
  }

  /** @brief Returns the total number of multigrid levels in the hierarchy including the finest level. */
  vcl_size_t levels() const { return residual_list_.size(); }


  /** @brief Returns the problem/operator size at the respective multigrid level
    *
    * @param level     Index of the multigrid level. 0 is the finest level, levels() - 1 is the coarsest level.
    */
  vcl_size_t size(vcl_size_t level) const
  {
    assert(level < levels() && bool("Level index out of bounds!"));
    return residual_list_[level].size();
  }

  /** @brief Returns the associated preconditioner tag containing the configuration for the multigrid preconditioner. */
  amg_tag const & tag() const { return tag_; }

private:
  std::vector<SparseMatrixType> A_list_;
  std::vector<SparseMatrixType> P_list_;
  std::vector<SparseMatrixType> R_list_;
  std::vector<AMGContextType>   amg_context_list_;

  viennacl::matrix<NumericT>        coarsest_op_;

  mutable std::vector<VectorType> result_list_;
  mutable std::vector<VectorType> result_backup_list_;
  mutable std::vector<VectorType> rhs_list_;
  mutable std::vector<VectorType> residual_list_;

  amg_tag tag_;
};

}
}



#endif

