#ifndef VIENNACL_LINALG_OPENCL_AMG_OPERATIONS_HPP
#define VIENNACL_LINALG_OPENCL_AMG_OPERATIONS_HPP

/* =========================================================================
   Copyright (c) 2010-2016, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

/** @file opencl/amg_operations.hpp
    @brief Implementations of routines for AMG in OpenCL.
*/

#include <cstdlib>
#include <cmath>
#include <map>

#include "viennacl/linalg/detail/amg/amg_base.hpp"
#include "viennacl/linalg/opencl/common.hpp"
#include "viennacl/linalg/opencl/kernels/amg.hpp"


namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace amg
{


///////////////////////////////////////////

/** @brief Routine for taking all connections in the matrix as strong */
template<typename NumericT>
void amg_influence_trivial(compressed_matrix<NumericT> const & A,
                           viennacl::linalg::detail::amg::amg_level_context & amg_context,
                           viennacl::linalg::amg_tag & tag)
{
  (void)tag;

  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::amg<NumericT>::init(ctx);
  viennacl::ocl::kernel & influence_kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::amg<NumericT>::program_name(), "amg_influence_trivial");

  viennacl::ocl::enqueue(influence_kernel(A.handle1().opencl_handle(), A.handle2().opencl_handle(),
                                          cl_uint(A.size1()),
                                          cl_uint(A.nnz()),
                                          viennacl::traits::opencl_handle(amg_context.influence_jumper_),
                                          viennacl::traits::opencl_handle(amg_context.influence_ids_),
                                          viennacl::traits::opencl_handle(amg_context.influence_values_)
                                         )
                         );
}


/** @brief Routine for extracting strongly connected points considering a user-provided threshold value */
template<typename NumericT>
void amg_influence_advanced(compressed_matrix<NumericT> const & A,
                            viennacl::linalg::detail::amg::amg_level_context & amg_context,
                            viennacl::linalg::amg_tag & tag)
{
  (void)A; (void)amg_context; (void)tag;
  throw std::runtime_error("amg_influence_advanced() not implemented for OpenCL yet");
}


/** @brief Dispatcher for influence processing */
template<typename NumericT>
void amg_influence(compressed_matrix<NumericT> const & A,
                   viennacl::linalg::detail::amg::amg_level_context & amg_context,
                   viennacl::linalg::amg_tag & tag)
{
  // TODO: dispatch based on influence tolerance provided
  amg_influence_trivial(A, amg_context, tag);
}



/** @brief Assign IDs to coarse points.
*
*  TODO: Use exclusive_scan on GPU for this.
*/
inline void enumerate_coarse_points(viennacl::linalg::detail::amg::amg_level_context & amg_context)
{
  viennacl::backend::typesafe_host_array<unsigned int> point_types(amg_context.point_types_.handle(), amg_context.point_types_.size());
  viennacl::backend::typesafe_host_array<unsigned int> coarse_ids(amg_context.coarse_id_.handle(),    amg_context.coarse_id_.size());
  viennacl::backend::memory_read(amg_context.point_types_.handle(), 0, point_types.raw_size(), point_types.get());
  viennacl::backend::memory_read(amg_context.coarse_id_.handle(),   0, coarse_ids.raw_size(),  coarse_ids.get());

  unsigned int coarse_id = 0;
  for (std::size_t i=0; i<amg_context.point_types_.size(); ++i)
  {
    coarse_ids.set(i, coarse_id);
    if (point_types[i] == viennacl::linalg::detail::amg::amg_level_context::POINT_TYPE_COARSE)
      ++coarse_id;
  }

  amg_context.num_coarse_ = coarse_id;

  viennacl::backend::memory_write(amg_context.coarse_id_.handle(), 0, coarse_ids.raw_size(), coarse_ids.get());
}


//////////////////////////////////////



/** @brief AG (aggregation based) coarsening, single-threaded version of stage 1
*
* @param A             Operator matrix on all levels
* @param amg_context   AMG hierarchy datastructures
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag_stage1_mis2(compressed_matrix<NumericT> const & A,
                               viennacl::linalg::detail::amg::amg_level_context & amg_context,
                               viennacl::linalg::amg_tag & tag)
{
  (void)tag;
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::amg<NumericT>::init(ctx);

  viennacl::vector<unsigned int> random_weights(A.size1(), viennacl::context(viennacl::MAIN_MEMORY));
  unsigned int *random_weights_ptr = viennacl::linalg::host_based::detail::extract_raw_pointer<unsigned int>(random_weights.handle());
  for (std::size_t i=0; i<random_weights.size(); ++i)
    random_weights_ptr[i] = static_cast<unsigned int>(rand()) % static_cast<unsigned int>(A.size1());
  random_weights.switch_memory_context(viennacl::traits::context(A));

  // work vectors:
  viennacl::vector<unsigned int> work_state(A.size1(),  viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_random(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_index(A.size1(),  viennacl::traits::context(A));

  viennacl::vector<unsigned int> work_state2(A.size1(),  viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_random2(A.size1(), viennacl::traits::context(A));
  viennacl::vector<unsigned int> work_index2(A.size1(),  viennacl::traits::context(A));

  unsigned int num_undecided = static_cast<unsigned int>(A.size1());
  viennacl::vector<unsigned int> undecided_buffer(256, viennacl::traits::context(A));
  viennacl::backend::typesafe_host_array<unsigned int> undecided_buffer_host(undecided_buffer.handle(), undecided_buffer.size());

  viennacl::ocl::kernel & init_workdata_kernel    = ctx.get_kernel(viennacl::linalg::opencl::kernels::amg<NumericT>::program_name(), "amg_pmis2_init_workdata");
  viennacl::ocl::kernel & max_neighborhood_kernel = ctx.get_kernel(viennacl::linalg::opencl::kernels::amg<NumericT>::program_name(), "amg_pmis2_max_neighborhood");
  viennacl::ocl::kernel & mark_mis_nodes_kernel   = ctx.get_kernel(viennacl::linalg::opencl::kernels::amg<NumericT>::program_name(), "amg_pmis2_mark_mis_nodes");
  viennacl::ocl::kernel & reset_state_kernel      = ctx.get_kernel(viennacl::linalg::opencl::kernels::amg<NumericT>::program_name(), "amg_pmis2_reset_state");

  unsigned int pmis_iters = 0;
  while (num_undecided > 0)
  {
    ++pmis_iters;

    //
    // init temporary work data:
    //
    viennacl::ocl::enqueue(init_workdata_kernel(work_state,  work_random,  work_index,
                                                amg_context.point_types_,
                                                random_weights,
                                                cl_uint(A.size1())
                                               )
                          );

    //
    // Propagate maximum tuple twice
    //
    for (unsigned int r = 0; r < 2; ++r)
    {
      // max operation
      viennacl::ocl::enqueue(max_neighborhood_kernel(work_state,  work_random,  work_index,
                                                     work_state2, work_random2, work_index2,
                                                     amg_context.influence_jumper_, amg_context.influence_ids_,
                                                     cl_uint(A.size1())
                                                    )
                            );

      // copy work array (can be fused into a single kernel if needed. Previous kernel is in most cases sufficiently heavy)
      work_state  = work_state2;
      work_random = work_random2;
      work_index  = work_index2;
    }

    //
    // mark MIS and non-MIS nodes:
    //
    viennacl::ocl::enqueue(mark_mis_nodes_kernel(work_state, work_index,
                                                 amg_context.point_types_,
                                                 undecided_buffer,
                                                 cl_uint(A.size1())
                                                )
                          );

    // get number of undecided points on host:
    viennacl::backend::memory_read(undecided_buffer.handle(), 0, undecided_buffer_host.raw_size(), undecided_buffer_host.get());
    num_undecided = 0;
    for (std::size_t i=0; i<undecided_buffer.size(); ++i)
      num_undecided += undecided_buffer_host[i];

  } //while

  viennacl::ocl::enqueue(reset_state_kernel(amg_context.point_types_, cl_uint(amg_context.point_types_.size()) ) );
}



/** @brief AG (aggregation based) coarsening. Partially single-threaded version (VIENNACL_AMG_COARSE_AG)
*
* @param A             Operator matrix
* @param amg_context   AMG hierarchy datastructures
* @param tag           AMG preconditioner tag
*/
template<typename NumericT>
void amg_coarse_ag(compressed_matrix<NumericT> const & A,
                   viennacl::linalg::detail::amg::amg_level_context & amg_context,
                   viennacl::linalg::amg_tag & tag)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::amg<NumericT>::init(ctx);

  amg_influence_trivial(A, amg_context, tag);

  //
  // Stage 1: Build aggregates:
  //
  if (tag.get_coarsening_method() == viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION)
    amg_coarse_ag_stage1_mis2(A, amg_context, tag);
  else
    throw std::runtime_error("Only MIS2 coarsening implemented. Selected coarsening not available with OpenCL backend!");

  viennacl::linalg::opencl::amg::enumerate_coarse_points(amg_context);

  //
  // Stage 2: Propagate coarse aggregate indices to neighbors:
  //
  viennacl::ocl::kernel & propagate_coarse_indices = ctx.get_kernel(viennacl::linalg::opencl::kernels::amg<NumericT>::program_name(), "amg_agg_propagate_coarse_indices");
  viennacl::ocl::enqueue(propagate_coarse_indices(amg_context.point_types_,
                                                  amg_context.coarse_id_,
                                                  amg_context.influence_jumper_,
                                                  amg_context.influence_ids_,
                                                  cl_uint(A.size1())
                                                 )
                        );

  //
  // Stage 3: Merge remaining undecided points (merging to first aggregate found when cycling over the hierarchy
  //
  viennacl::ocl::kernel & merge_undecided = ctx.get_kernel(viennacl::linalg::opencl::kernels::amg<NumericT>::program_name(), "amg_agg_merge_undecided");
  viennacl::ocl::enqueue(merge_undecided(amg_context.point_types_,
                                         amg_context.coarse_id_,
                                         amg_context.influence_jumper_,
                                         amg_context.influence_ids_,
                                         cl_uint(A.size1())
                                        )
                         );

  //
  // Stage 4: Set undecided points to fine points (coarse ID already set in Stage 3)
  //          Note: Stage 3 and Stage 4 were initially fused, but are now split in order to avoid race conditions (or a fallback to sequential execution).
  //
  viennacl::ocl::kernel & merge_undecided_2 = ctx.get_kernel(viennacl::linalg::opencl::kernels::amg<NumericT>::program_name(), "amg_agg_merge_undecided_2");
  viennacl::ocl::enqueue(merge_undecided_2(amg_context.point_types_, cl_uint(A.size1()) ) );

}




/** @brief Calls the right coarsening procedure
*
* @param A            Operator matrix on all levels
* @param amg_context  AMG hierarchy datastructures
* @param tag          AMG preconditioner tag
*/
template<typename InternalT1>
void amg_coarse(InternalT1 & A,
                viennacl::linalg::detail::amg::amg_level_context & amg_context,
                viennacl::linalg::amg_tag & tag)
{
  switch (tag.get_coarsening_method())
  {
  case viennacl::linalg::AMG_COARSENING_METHOD_MIS2_AGGREGATION: amg_coarse_ag(A, amg_context, tag); break;
  default: throw std::runtime_error("not implemented yet");
  }
}




////////////////////////////////////// Interpolation /////////////////////////////


/** @brief AG (aggregation based) interpolation. Multi-Threaded! (VIENNACL_INTERPOL_SA)
 *
 * @param A            Operator matrix
 * @param P            Prolongation matrix
 * @param amg_context  AMG hierarchy datastructures
 * @param tag          AMG configuration tag
*/
template<typename NumericT>
void amg_interpol_ag(compressed_matrix<NumericT> const & A,
                     compressed_matrix<NumericT> & P,
                     viennacl::linalg::detail::amg::amg_level_context & amg_context,
                     viennacl::linalg::amg_tag & tag)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::amg<NumericT>::init(ctx);

  (void)tag;
  P = compressed_matrix<NumericT>(A.size1(), amg_context.num_coarse_, A.size1(), viennacl::traits::context(A));

  // build matrix here
  viennacl::ocl::kernel & interpolate_ag = ctx.get_kernel(viennacl::linalg::opencl::kernels::amg<NumericT>::program_name(), "amg_interpol_ag");
  viennacl::ocl::enqueue(interpolate_ag(P.handle1().opencl_handle(),
                                        P.handle2().opencl_handle(),
                                        P.handle().opencl_handle(),
                                        amg_context.coarse_id_,
                                        cl_uint(A.size1())
                                        )
                         );

  P.generate_row_block_information();
}

/** @brief Smoothed aggregation interpolation. (VIENNACL_INTERPOL_SA)
 *
 * @param A            Operator matrix
 * @param P            Prolongation matrix
 * @param amg_context  AMG hierarchy datastructures
 * @param tag          AMG configuration tag
*/
template<typename NumericT>
void amg_interpol_sa(compressed_matrix<NumericT> const & A,
                     compressed_matrix<NumericT> & P,
                     viennacl::linalg::detail::amg::amg_level_context & amg_context,
                     viennacl::linalg::amg_tag & tag)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::amg<NumericT>::init(ctx);

  (void)tag;
  viennacl::compressed_matrix<NumericT> P_tentative(A.size1(), amg_context.num_coarse_, A.size1(), viennacl::traits::context(A));

  // form tentative operator:
  amg_interpol_ag(A, P_tentative, amg_context, tag);

  viennacl::compressed_matrix<NumericT> Jacobi(A.size1(), A.size1(), A.nnz(), viennacl::traits::context(A));

  viennacl::ocl::kernel & interpol_sa = ctx.get_kernel(viennacl::linalg::opencl::kernels::amg<NumericT>::program_name(), "amg_interpol_sa");
  viennacl::ocl::enqueue(interpol_sa(A.handle1().opencl_handle(),
                                     A.handle2().opencl_handle(),
                                     A.handle().opencl_handle(),
                                     cl_uint(A.size1()),
                                     cl_uint(A.nnz()),
                                     Jacobi.handle1().opencl_handle(),
                                     Jacobi.handle2().opencl_handle(),
                                     Jacobi.handle().opencl_handle(),
                                     NumericT(tag.get_jacobi_weight())
                                    )
                         );

  P = viennacl::linalg::prod(Jacobi, P_tentative);

  P.generate_row_block_information();
}

/** @brief Dispatcher for building the interpolation matrix
 *
 * @param A            Operator matrix
 * @param P            Prolongation matrix
 * @param amg_context  AMG hierarchy datastructures
 * @param tag          AMG configuration tag
*/
template<typename MatrixT>
void amg_interpol(MatrixT const & A,
                  MatrixT & P,
                  viennacl::linalg::detail::amg::amg_level_context & amg_context,
                  viennacl::linalg::amg_tag & tag)
{
  switch (tag.get_interpolation_method())
  {
  case viennacl::linalg::AMG_INTERPOLATION_METHOD_AGGREGATION:           amg_interpol_ag     (A, P, amg_context, tag); break;
  case viennacl::linalg::AMG_INTERPOLATION_METHOD_SMOOTHED_AGGREGATION:  amg_interpol_sa     (A, P, amg_context, tag); break;
  default: throw std::runtime_error("Not implemented yet!");
  }
}

/** Assign sparse matrix A to dense matrix B */
template<typename NumericT, unsigned int AlignmentV>
void assign_to_dense(viennacl::compressed_matrix<NumericT, AlignmentV> const & A,
                     viennacl::matrix_base<NumericT> & B)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
  viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(),
                                             "assign_to_dense");

  viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(),
                           viennacl::traits::opencl_handle(B),
                           cl_uint(viennacl::traits::start1(B)),         cl_uint(viennacl::traits::start2(B)),
                           cl_uint(viennacl::traits::stride1(B)),        cl_uint(viennacl::traits::stride2(B)),
                           cl_uint(viennacl::traits::size1(B)),          cl_uint(viennacl::traits::size2(B)),
                           cl_uint(viennacl::traits::internal_size1(B)), cl_uint(viennacl::traits::internal_size2(B)) ));

}

/** @brief Jacobi Smoother (OpenCL version)
*
* @param iterations  Number of smoother iterations
* @param A           Operator matrix for the smoothing
* @param x           The vector smoothing is applied to
* @param x_backup    (Different) Vector holding the same values as x
* @param rhs_smooth  The right hand side of the equation for the smoother
* @param weight      Damping factor. 0: No effect of smoother. 1: Undamped Jacobi iteration
*/
template<typename NumericT>
void smooth_jacobi(unsigned int iterations,
                   compressed_matrix<NumericT> const & A,
                   vector<NumericT> & x,
                   vector<NumericT> & x_backup,
                   vector<NumericT> const & rhs_smooth,
                   NumericT weight)
{
  viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(x).context());
  viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::init(ctx);
  viennacl::ocl::kernel & k = ctx.get_kernel(viennacl::linalg::opencl::kernels::compressed_matrix<NumericT>::program_name(), "jacobi");

  for (unsigned int i=0; i<iterations; ++i)
  {
    x_backup = x;

    viennacl::ocl::enqueue(k(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(),
                            static_cast<NumericT>(weight),
                            viennacl::traits::opencl_handle(x_backup),
                            viennacl::traits::opencl_handle(x),
                            viennacl::traits::opencl_handle(rhs_smooth),
                            static_cast<cl_uint>(rhs_smooth.size())));

  }
}


} //namespace amg
} //namespace host_based
} //namespace linalg
} //namespace viennacl

#endif
