#ifndef VIENNACL_LINALG_DETAIL_AMG_AMG_BASE_HPP_
#define VIENNACL_LINALG_DETAIL_AMG_AMG_BASE_HPP_

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

/** @file amg_base.hpp
    @brief Helper classes and functions for the AMG preconditioner. Experimental.

    AMG code contributed by Markus Wagner
*/

#include <cmath>
#include <set>
#include <list>
#include <stdexcept>
#include <algorithm>

#include <map>
#ifdef VIENNACL_WITH_OPENMP
#include <omp.h>
#endif

#include "viennacl/context.hpp"

namespace viennacl
{
namespace linalg
{

/** @brief Enumeration of coarsening methods for algebraic multigrid. */
enum amg_coarsening_method
{
  AMG_COARSENING_METHOD_ONEPASS = 1,
  AMG_COARSENING_METHOD_AGGREGATION,
  AMG_COARSENING_METHOD_MIS2_AGGREGATION
};

/** @brief Enumeration of interpolation methods for algebraic multigrid. */
enum amg_interpolation_method
{
  AMG_INTERPOLATION_METHOD_DIRECT = 1,
  AMG_INTERPOLATION_METHOD_AGGREGATION,
  AMG_INTERPOLATION_METHOD_SMOOTHED_AGGREGATION
};


/** @brief A tag for algebraic multigrid (AMG). Used to transport information from the user to the implementation.
*/
class amg_tag
{
public:
  /** @brief The constructor, setting default values for the various parameters.
    *
    * Default coarsening routine: Aggreggation based on maximum independent sets of distance (MIS-2)
    * Default interpolation routine: Smoothed aggregation
    * Default threshold for strong connections: 0.1 (customizations are recommeded!)
    * Default weight for Jacobi smoother: 1.0
    * Default number of pre-smooth operations: 2
    * Default number of post-smooth operations: 2
    * Default number of coarse levels: 0 (this indicates that as many coarse levels as needed are constructed until the cutoff is reached)
    * Default coarse grid size for direct solver (coarsening cutoff): 50
    */
  amg_tag()
  : coarsening_method_(AMG_COARSENING_METHOD_MIS2_AGGREGATION), interpolation_method_(AMG_INTERPOLATION_METHOD_AGGREGATION),
    strong_connection_threshold_(0.1), jacobi_weight_(1.0),
    presmooth_steps_(2), postsmooth_steps_(2),
    coarse_levels_(0), coarse_cutoff_(50) {}

  // Getter-/Setter-Functions
  /** @brief Sets the strategy used for constructing coarse grids  */
  void set_coarsening_method(amg_coarsening_method s) { coarsening_method_ = s; }
  /** @brief Returns the current coarsening strategy */
  amg_coarsening_method get_coarsening_method() const { return coarsening_method_; }

  /** @brief Sets the interpolation method to the provided method */
  void set_interpolation_method(amg_interpolation_method interpol) { interpolation_method_ = interpol; }
  /** @brief Returns the current interpolation method */
  amg_interpolation_method get_interpolation_method() const { return interpolation_method_; }

  /** @brief Sets the strong connection threshold. Customizations by the user essential for best results!
    *
    * With classical interpolation, a connection is considered strong if |a_ij| >= threshold * max_k(|a_ik|)
    * Strength of connection currently ignored for aggregation-based coarsening (to be added in the future).
    */
  void set_strong_connection_threshold(double threshold) { if (threshold > 0) strong_connection_threshold_ = threshold; }
  /** @brief Returns the strong connection threshold parameter.
    *
    * @see set_strong_connection_threshold() for an explanation of the threshold parameter
    */
  double get_strong_connection_threshold() const { return strong_connection_threshold_; }

  /** @brief Sets the weight (damping) for the Jacobi smoother.
    *
    * The optimal value depends on the problem at hand. Values of 0.67 or 1.0 are usually a good starting point for further experiments.
    */
  void set_jacobi_weight(double w) { if (w > 0) jacobi_weight_ = w; }
  /** @brief Returns the Jacobi smoother weight (damping). */
  double get_jacobi_weight() const { return jacobi_weight_; }

  /** @brief Sets the number of smoother applications on the fine level before restriction to the coarser level. */
  void set_presmooth_steps(vcl_size_t steps) { presmooth_steps_ = steps; }
  /** @brief Returns the number of smoother applications on the fine level before restriction to the coarser level. */
  vcl_size_t get_presmooth_steps() const { return presmooth_steps_; }

  /** @brief Sets the number of smoother applications on the coarse level before interpolation to the finer level. */
  void set_postsmooth_steps(vcl_size_t steps) { postsmooth_steps_ = steps; }
  /** @brief Returns the number of smoother applications on the coarse level before interpolation to the finer level. */
  vcl_size_t get_postsmooth_steps() const { return postsmooth_steps_; }

  /** @brief Sets the number of coarse levels. If set to zero, then coarse levels are constructed until the cutoff size is reached. */
  void set_coarse_levels(vcl_size_t levels)  { coarse_levels_ = levels; }
  /** @brief Returns the number of coarse levels. If zero, then coarse levels are constructed until the cutoff size is reached. */
  vcl_size_t get_coarse_levels() const { return coarse_levels_; }

  /** @brief Sets the coarse grid size for which the recursive multigrid scheme is stopped and a direct solver is used. */
  void set_coarsening_cutoff(vcl_size_t size)  { coarse_cutoff_ = size; }
  /** @brief Returns the coarse grid size for which the recursive multigrid scheme is stopped and a direct solver is used. */
  vcl_size_t get_coarsening_cutoff() const { return coarse_cutoff_; }

  /** @brief Sets the ViennaCL context for the setup stage. Set this to a host context if you want to run the setup on the host.
    *
    * Set the ViennaCL context for the solver application via set_target_context().
    * Target and setup context can be different.
    */
  void set_setup_context(viennacl::context ctx)  { setup_ctx_ = ctx; }
  /** @brief Returns the ViennaCL context for the preconditioenr setup. */
  viennacl::context const & get_setup_context() const { return setup_ctx_; }

  /** @brief Sets the ViennaCL context for the solver cycle stage (i.e. preconditioner applications).
    *
    * Since the cycle stage easily benefits from accelerators, you usually want to set this to a CUDA or OpenCL-enabled context.
    */
  void set_target_context(viennacl::context ctx)  { target_ctx_ = ctx; }
  /** @brief Returns the ViennaCL context for the solver cycle stage (i.e. preconditioner applications). */
  viennacl::context const & get_target_context() const { return target_ctx_; }

private:
  amg_coarsening_method coarsening_method_;
  amg_interpolation_method interpolation_method_;
  double strong_connection_threshold_, jacobi_weight_;
  vcl_size_t presmooth_steps_, postsmooth_steps_, coarse_levels_, coarse_cutoff_;
  viennacl::context setup_ctx_, target_ctx_;
};


namespace detail
{
namespace amg
{


  struct amg_level_context
  {
    void resize(vcl_size_t num_points, vcl_size_t max_nnz)
    {
      influence_jumper_.resize(num_points + 1, false);
      influence_ids_.resize(max_nnz, false);
      influence_values_.resize(num_points, false);
      point_types_.resize(num_points, false);
      coarse_id_.resize(num_points, false);
    }

    void switch_context(viennacl::context ctx)
    {
      influence_jumper_.switch_memory_context(ctx);
      influence_ids_.switch_memory_context(ctx);
      influence_values_.switch_memory_context(ctx);
      point_types_.switch_memory_context(ctx);
      coarse_id_.switch_memory_context(ctx);
    }

    enum
    {
      POINT_TYPE_UNDECIDED = 0,
      POINT_TYPE_COARSE,
      POINT_TYPE_FINE
    } amg_point_types;

    viennacl::vector<unsigned int> influence_jumper_; // similar to row_buffer for CSR matrices
    viennacl::vector<unsigned int> influence_ids_;    // IDs of influencing points
    viennacl::vector<unsigned int> influence_values_; // Influence measure for each point
    viennacl::vector<unsigned int> point_types_;      // 0: undecided, 1: coarse point, 2: fine point. Using char here because type for enum might be a larger type
    viennacl::vector<unsigned int> coarse_id_;        // coarse ID used on the next level. Only valid for coarse points. Fine points may (ab)use their entry for something else.
    unsigned int num_coarse_;
  };


} //namespace amg
}
}
}

#endif
