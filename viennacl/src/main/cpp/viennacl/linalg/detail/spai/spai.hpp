#ifndef VIENNACL_LINALG_DETAIL_SPAI_SPAI_HPP
#define VIENNACL_LINALG_DETAIL_SPAI_SPAI_HPP

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

/** @file viennacl/linalg/detail/spai/spai.hpp
    @brief Main implementation of SPAI (not FSPAI). Experimental.
*/

#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <math.h>
#include <map>

//local includes
#include "viennacl/linalg/detail/spai/spai_tag.hpp"
#include "viennacl/linalg/qr.hpp"
#include "viennacl/linalg/detail/spai/spai-dynamic.hpp"
#include "viennacl/linalg/detail/spai/spai-static.hpp"
#include "viennacl/linalg/detail/spai/sparse_vector.hpp"
#include "viennacl/linalg/detail/spai/block_matrix.hpp"
#include "viennacl/linalg/detail/spai/block_vector.hpp"

//boost includes
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/storage.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/lu.hpp"
#include "boost/numeric/ublas/triangular.hpp"
#include "boost/numeric/ublas/matrix_expression.hpp"

// ViennaCL includes
#include "viennacl/linalg/prod.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/linalg/sparse_matrix_operations.hpp"
#include "viennacl/linalg/matrix_operations.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/linalg/inner_prod.hpp"
#include "viennacl/linalg/ilu.hpp"
#include "viennacl/ocl/backend.hpp"
#include "viennacl/linalg/opencl/kernels/spai.hpp"



#define VIENNACL_SPAI_K_b 20

namespace viennacl
{
namespace linalg
{
namespace detail
{
namespace spai
{

//debug function for print
template<typename SparseVectorT>
void print_sparse_vector(SparseVectorT const & v)
{
  for (typename SparseVectorT::const_iterator vec_it = v.begin(); vec_it!= v.end(); ++vec_it)
    std::cout << "[ " << vec_it->first << " ]:" << vec_it->second << std::endl;
}

template<typename DenseMatrixT>
void print_matrix(DenseMatrixT & m)
{
  for (int i = 0; i < m.size2(); ++i)
  {
    for (int j = 0; j < m.size1(); ++j)
      std::cout<<m(j, i)<<" ";
    std::cout<<std::endl;
  }
}

/** @brief Add two sparse vectors res_v = b*v
 *
 * @param v      initial sparse vector
 * @param b      scalar
 * @param res_v  output vector
 */
template<typename SparseVectorT, typename NumericT>
void add_sparse_vectors(SparseVectorT const & v, NumericT b,  SparseVectorT & res_v)
{
  for (typename SparseVectorT::const_iterator v_it = v.begin(); v_it != v.end(); ++v_it)
    res_v[v_it->first] += b*v_it->second;
}

//sparse-matrix - vector product
/** @brief Computation of residual res = A*v - e
 *
 * @param A_v_c   column major vectorized input sparse matrix
 * @param v       sparse vector, in this case new column of preconditioner matrix
 * @param ind     index for current column
 * @param res     residual
 */
template<typename SparseVectorT, typename NumericT>
void compute_spai_residual(std::vector<SparseVectorT> const & A_v_c,
                           SparseVectorT const & v,
                           unsigned int ind,
                           SparseVectorT & res)
{
  for (typename SparseVectorT::const_iterator v_it = v.begin(); v_it != v.end(); ++v_it)
    add_sparse_vectors(A_v_c[v_it->first], v_it->second, res);

  res[ind] -= NumericT(1);
}

/** @brief Setting up index set of columns and rows for certain column
 *
 * @param A_v_c   column major vectorized initial sparse matrix
 * @param v       current column of preconditioner matrix
 * @param J       set of column indices
 * @param I       set of row indices
 */
template<typename SparseVectorT>
void build_index_set(std::vector<SparseVectorT> const & A_v_c,
                     SparseVectorT const & v,
                     std::vector<unsigned int> & J,
                     std::vector<unsigned int> & I)
{
  buildColumnIndexSet(v, J);
  projectRows(A_v_c, J, I);
}

/** @brief Initializes a dense matrix from a sparse one
 *
 * @param A_in    Oiginal sparse matrix
 * @param J       Set of column indices
 * @param I       Set of row indices
 * @param A_out   dense matrix output
 */
template<typename SparseMatrixT, typename DenseMatrixT>
void initProjectSubMatrix(SparseMatrixT const & A_in,
                          std::vector<unsigned int> const & J,
                          std::vector<unsigned int> & I,
                          DenseMatrixT & A_out)
{
  A_out.resize(I.size(), J.size(), false);
  for (vcl_size_t j = 0; j < J.size(); ++j)
    for (vcl_size_t i = 0; i < I.size(); ++i)
      A_out(i,j) = A_in(I[i],J[j]);
}


/************************************************** CPU BLOCK SET UP ***************************************/

/** @brief Setting up blocks and QR factorizing them on CPU
 *
 * @param A        initial sparse matrix
 * @param A_v_c    column major vectorized initial sparse matrix
 * @param M_v      initialized preconditioner
 * @param g_I      container of row indices
 * @param g_J      container of column indices
 * @param g_A_I_J  container of dense matrices -> R matrices after QR factorization
 * @param g_b_v    container of vectors beta, necessary for Q recovery
 */
template<typename SparseMatrixT, typename DenseMatrixT, typename SparseVectorT, typename VectorT>
void block_set_up(SparseMatrixT const & A,
                  std::vector<SparseVectorT> const & A_v_c,
                  std::vector<SparseVectorT> const & M_v,
                  std::vector<std::vector<unsigned int> >& g_I,
                  std::vector<std::vector<unsigned int> >& g_J,
                  std::vector<DenseMatrixT>& g_A_I_J,
                  std::vector<VectorT>& g_b_v)
{
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i2 = 0; i2 < static_cast<long>(M_v.size()); ++i2)
  {
    vcl_size_t i = static_cast<vcl_size_t>(i2);
    build_index_set(A_v_c, M_v[i], g_J[i], g_I[i]);
    initProjectSubMatrix(A, g_J[i], g_I[i], g_A_I_J[i]);
    //print_matrix(g_A_I_J[i]);
    single_qr(g_A_I_J[i], g_b_v[i]);
    //print_matrix(g_A_I_J[i]);
  }
}

/** @brief Setting up index set of columns and rows for all columns
 *
 * @param A_v_c   column major vectorized initial sparse matrix
 * @param M_v     initialized preconditioner
 * @param g_J     container of column indices
 * @param g_I     container of row indices
 */
template<typename SparseVectorT>
void index_set_up(std::vector<SparseVectorT> const & A_v_c,
                  std::vector<SparseVectorT> const & M_v,
                  std::vector<std::vector<unsigned int> > & g_J,
                  std::vector<std::vector<unsigned int> > & g_I)
{
#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i2 = 0; i2 < static_cast<long>(M_v.size()); ++i2)
  {
    vcl_size_t i = static_cast<vcl_size_t>(i2);
    build_index_set(A_v_c, M_v[i], g_J[i], g_I[i]);
  }
}

/************************************************** GPU BLOCK SET UP ***************************************/

/** @brief Setting up blocks and QR factorizing them on GPU
 *
 * @param A            initial sparse matrix
 * @param A_v_c        column major vectorized initial sparse matrix
 * @param M_v          initialized preconditioner
 * @param g_is_update  container that indicates which blocks are active
 * @param g_I          container of row indices
 * @param g_J          container of column indices
 * @param g_A_I_J      container of dense matrices -> R matrices after QR factorization
 * @param g_bv         container of vectors beta, necessary for Q recovery
 */
template<typename NumericT, unsigned int AlignmentV, typename SparseVectorT>
void block_set_up(viennacl::compressed_matrix<NumericT, AlignmentV> const & A,
                  std::vector<SparseVectorT> const & A_v_c,
                  std::vector<SparseVectorT> const & M_v,
                  std::vector<cl_uint> g_is_update,
                  std::vector<std::vector<unsigned int> > & g_I,
                  std::vector<std::vector<unsigned int> > & g_J,
                  block_matrix & g_A_I_J,
                  block_vector & g_bv)
{
  viennacl::context ctx = viennacl::traits::context(A);
  bool is_empty_block;

  //build index set
  index_set_up(A_v_c, M_v, g_J, g_I);
  block_assembly(A, g_J, g_I, g_A_I_J, g_is_update, is_empty_block);
  block_qr<NumericT>(g_I, g_J, g_A_I_J, g_bv, g_is_update, ctx);
}


/***************************************************************************************************/
/******************************** SOLVING LS PROBLEMS ON GPU ***************************************/
/***************************************************************************************************/

/** @brief Elicitation of sparse vector m for particular column from m_in - contigious vector for all columns
 *
 * @param m_in          contigious sparse vector for all columns
 * @param start_m_ind   start index of particular vector
 * @param J             column index set
 * @param m             sparse vector for particular column
 */
template<typename NumericT, typename SparseVectorT>
void custom_fan_out(std::vector<NumericT> const & m_in,
                    unsigned int start_m_ind,
                    std::vector<unsigned int> const & J,
                    SparseVectorT & m)
{
  unsigned int  cnt = 0;
  for (vcl_size_t i = 0; i < J.size(); ++i)
    m[J[i]] = m_in[start_m_ind + cnt++];
}



//GPU based least square problem
/** @brief Solution of Least square problem on GPU
 *
 * @param A_v_c        column-major vectorized initial sparse matrix
 * @param M_v          column-major vectorized sparse preconditioner matrix
 * @param g_I          container of row set indices
 * @param g_J          container of column set indices
 * @param g_A_I_J_vcl  contigious matrix that consists of blocks A(I_k, J_k)
 * @param g_bv_vcl     contigious vector that consists of betas, necessary for Q recovery
 * @param g_res        container of residuals
 * @param g_is_update  container with indicators which blocks are active
 * @param tag          spai tag
 * @param ctx          Optional context in which the auxiliary data is created (one out of multiple OpenCL contexts, CUDA, host)
 */
template<typename SparseVectorT, typename NumericT>
void least_square_solve(std::vector<SparseVectorT> & A_v_c,
                        std::vector<SparseVectorT> & M_v,
                        std::vector<std::vector<unsigned int> >& g_I,
                        std::vector<std::vector<unsigned int> > & g_J,
                        block_matrix & g_A_I_J_vcl,
                        block_vector & g_bv_vcl,
                        std::vector<SparseVectorT> & g_res,
                        std::vector<cl_uint> & g_is_update,
                        const spai_tag & tag,
                        viennacl::context ctx)
{
  viennacl::ocl::context & opencl_ctx = const_cast<viennacl::ocl::context &>(ctx.opencl_context());
  unsigned int y_sz, m_sz;
  std::vector<cl_uint> y_inds(M_v.size() + 1, static_cast<cl_uint>(0));
  std::vector<cl_uint> m_inds(M_v.size() + 1, static_cast<cl_uint>(0));

  get_size(g_I, y_sz);
  init_start_inds(g_I, y_inds);
  init_start_inds(g_J, m_inds);

  //create y_v
  std::vector<NumericT> y_v(y_sz, NumericT(0));
  for (vcl_size_t i = 0; i < M_v.size(); ++i)
  {
    for (vcl_size_t j = 0; j < g_I[i].size(); ++j)
    {
      if (g_I[i][j] == i)
        y_v[y_inds[i] + j] = NumericT(1.0);
    }
  }
  //compute m_v
  get_size(g_J, m_sz);
  std::vector<NumericT> m_v(m_sz, static_cast<cl_uint>(0));

  block_vector y_v_vcl;
  block_vector m_v_vcl;
  //prepearing memory for least square problem on GPU
  y_v_vcl.handle() = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                              static_cast<unsigned int>(sizeof(NumericT)*y_v.size()),
                                              &(y_v[0]));
  m_v_vcl.handle() = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                              static_cast<unsigned int>(sizeof(NumericT)*m_v.size()),
                                              &(m_v[0]));
  y_v_vcl.handle1() = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                               static_cast<unsigned int>(sizeof(cl_uint)*(g_I.size() + 1)),
                                               &(y_inds[0]));
  viennacl::ocl::handle<cl_mem> g_is_update_vcl = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                                                           static_cast<unsigned int>(sizeof(cl_uint)*(g_is_update.size())),
                                                                           &(g_is_update[0]));
  viennacl::linalg::opencl::kernels::spai<NumericT>::init(opencl_ctx);
  viennacl::ocl::kernel & ls_kernel = opencl_ctx.get_kernel(viennacl::linalg::opencl::kernels::spai<NumericT>::program_name(), "block_least_squares");
  ls_kernel.local_work_size(0, 1);
  ls_kernel.global_work_size(0, 256);
  viennacl::ocl::enqueue(ls_kernel(g_A_I_J_vcl.handle(), g_A_I_J_vcl.handle2(), g_bv_vcl.handle(), g_bv_vcl.handle1(), m_v_vcl.handle(),
                                   y_v_vcl.handle(), y_v_vcl.handle1(),
                                   g_A_I_J_vcl.handle1(), g_is_update_vcl,
                                   //viennacl::ocl::local_mem(static_cast<unsigned int>(sizeof(ScalarType)*(local_r_n*local_c_n))),
                                   static_cast<unsigned int>(M_v.size())));
  //copy vector m_v back from GPU to CPU
  cl_int vcl_err = clEnqueueReadBuffer(opencl_ctx.get_queue().handle().get(),
                                       m_v_vcl.handle().get(), CL_TRUE, 0,
                                       sizeof(NumericT)*(m_v.size()),
                                       &(m_v[0]), 0, NULL, NULL);
  VIENNACL_ERR_CHECK(vcl_err);

  //fan out vector in parallel
  //#pragma omp parallel for
  for (long i = 0; i < static_cast<long>(M_v.size()); ++i)
  {
    if (g_is_update[static_cast<vcl_size_t>(i)])
    {
      //faned out onto sparse vector
      custom_fan_out(m_v, m_inds[static_cast<vcl_size_t>(i)], g_J[static_cast<vcl_size_t>(i)], M_v[static_cast<vcl_size_t>(i)]);
      g_res[static_cast<vcl_size_t>(i)].clear();
      compute_spai_residual<SparseVectorT, NumericT>(A_v_c,  M_v[static_cast<vcl_size_t>(i)], static_cast<unsigned int>(i), g_res[static_cast<vcl_size_t>(i)]);
      NumericT res_norm = 0;
      //compute norm of res - just to make sure that this implementatino works correct
      sparse_norm_2(g_res[static_cast<vcl_size_t>(i)], res_norm);
      //std::cout<<"Residual norm of column #: "<<i<<std::endl;
      //std::cout<<res_norm<<std::endl;
      //std::cout<<"************************"<<std::endl;
      g_is_update[static_cast<vcl_size_t>(i)] = (res_norm > tag.getResidualNormThreshold())&& (!tag.getIsStatic())?(1):(0);
    }
  }
}

//CPU based least square problems
/** @brief Solution of Least square problem on CPU
 *
 * @param A_v_c        column-major vectorized initial sparse matrix
 * @param g_R          blocks for least square solution
 * @param g_b_v        vectors beta, necessary for Q recovery
 * @param g_I          container of row index set for all columns of matrix M
 * @param g_J          container of column index set for all columns of matrix M
 * @param g_res        container of residuals
 * @param g_is_update  container with indicators which blocks are active
 * @param M_v          column-major vectorized sparse matrix, final preconditioner
 * @param tag          spai tag
 */
template<typename SparseVectorT, typename DenseMatrixT, typename VectorT>
void least_square_solve(std::vector<SparseVectorT> const & A_v_c,
                        std::vector<DenseMatrixT> & g_R,
                        std::vector<VectorT> & g_b_v,
                        std::vector<std::vector<unsigned int> > & g_I,
                        std::vector<std::vector<unsigned int> > & g_J,
                        std::vector<SparseVectorT> & g_res,
                        std::vector<bool> & g_is_update,
                        std::vector<SparseVectorT> & M_v,
                        spai_tag const & tag)
{
  typedef typename DenseMatrixT::value_type       NumericType;

#ifdef VIENNACL_WITH_OPENMP
  #pragma omp parallel for
#endif
  for (long i2 = 0; i2 < static_cast<long>(M_v.size()); ++i2)
  {
    vcl_size_t i = static_cast<vcl_size_t>(i2);
    if (g_is_update[i])
    {
      VectorT y = boost::numeric::ublas::zero_vector<NumericType>(g_I[i].size());

      projectI<VectorT, NumericType>(g_I[i], y, static_cast<unsigned int>(tag.getBegInd() + long(i)));
      apply_q_trans_vec(g_R[i], g_b_v[i], y);

      VectorT m_new =  boost::numeric::ublas::zero_vector<NumericType>(g_R[i].size2());
      backwardSolve(g_R[i], y, m_new);
      fanOutVector(m_new, g_J[i], M_v[i]);
      g_res[i].clear();

      compute_spai_residual<SparseVectorT, NumericType>(A_v_c,  M_v[i], static_cast<unsigned int>(tag.getBegInd() + long(i)), g_res[i]);

      NumericType res_norm = 0;
      sparse_norm_2(g_res[i], res_norm);
//                    std::cout<<"Residual norm of column #: "<<i<<std::endl;
//                    std::cout<<res_norm<<std::endl;
//                    std::cout<<"************************"<<std::endl;
      g_is_update[i] = (res_norm > tag.getResidualNormThreshold())&& (!tag.getIsStatic());
    }
  }
}

//************************************ UPDATE CHECK ***************************************************//

template<typename VectorType>
bool is_all_update(VectorType& parallel_is_update)
{
  for (unsigned int i = 0; i < parallel_is_update.size(); ++i)
  {
    if (parallel_is_update[i])
      return true;
  }
  return false;
}

//********************************** MATRIX VECTORIZATION ***********************************************//

//Matrix vectorization, column based approach
/** @brief Solution of Least square problem on CPU
 *
 * @param M_in   input sparse, boost::numeric::ublas::compressed_matrix
 * @param M_v    array of sparse vectors
 */
template<typename SparseMatrixT, typename SparseVectorT>
void vectorize_column_matrix(SparseMatrixT const & M_in,
                             std::vector<SparseVectorT> & M_v)
{
  for (typename SparseMatrixT::const_iterator1 row_it = M_in.begin1(); row_it!= M_in.end1(); ++row_it)
    for (typename SparseMatrixT::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
        M_v[static_cast<unsigned int>(col_it.index2())][static_cast<unsigned int>(col_it.index1())] = *col_it;
}

//Matrix vectorization row based approach
template<typename SparseMatrixT, typename SparseVectorT>
void vectorize_row_matrix(SparseMatrixT const & M_in,
                          std::vector<SparseVectorT> & M_v)
{
  for (typename SparseMatrixT::const_iterator1 row_it = M_in.begin1(); row_it!= M_in.end1(); ++row_it)
    for (typename SparseMatrixT::const_iterator2 col_it = row_it.begin(); col_it != row_it.end(); ++col_it)
      M_v[static_cast<unsigned int>(col_it.index1())][static_cast<unsigned int>(col_it.index2())] = *col_it;
}

//************************************* BLOCK ASSEMBLY CODE *********************************************//


template<typename SizeT>
void write_set_to_array(std::vector<std::vector<SizeT> > const & ind_set,
                        std::vector<cl_uint> & a)
{
  vcl_size_t cnt = 0;

  for (vcl_size_t i = 0; i < ind_set.size(); ++i)
    for (vcl_size_t j = 0; j < ind_set[i].size(); ++j)
      a[cnt++] = static_cast<cl_uint>(ind_set[i][j]);
}



//assembling blocks on GPU
/** @brief Assembly of blocks on GPU by a gived set of row indices: g_I and column indices: g_J
 *
 * @param A               intial sparse matrix
 * @param g_J             container of column index set
 * @param g_I             container of row index set
 * @param g_A_I_J_vcl     contigious blocks A(I, J) using GPU memory
 * @param g_is_update     container with indicators which blocks are active
 * @param is_empty_block  parameter that indicates if no block were assembled
 */
template<typename NumericT, unsigned int AlignmentV>
void block_assembly(viennacl::compressed_matrix<NumericT, AlignmentV> const & A,
                    std::vector<std::vector<unsigned int> > const & g_J,
                    std::vector<std::vector<unsigned int> > const & g_I,
                    block_matrix & g_A_I_J_vcl,
                    std::vector<cl_uint> & g_is_update,
                    bool & is_empty_block)
{
  //computing start indices for index sets and start indices for block matrices
  unsigned int sz_I, sz_J, sz_blocks;
  std::vector<cl_uint> matrix_dims(g_I.size()*2, static_cast<cl_uint>(0));
  std::vector<cl_uint> i_ind(g_I.size() + 1, static_cast<cl_uint>(0));
  std::vector<cl_uint> j_ind(g_I.size() + 1, static_cast<cl_uint>(0));
  std::vector<cl_uint> blocks_ind(g_I.size() + 1, static_cast<cl_uint>(0));
  //
  init_start_inds(g_J, j_ind);
  init_start_inds(g_I, i_ind);
  //
  get_size(g_J, sz_J);
  get_size(g_I, sz_I);
  std::vector<cl_uint> I_set(sz_I, static_cast<cl_uint>(0));
  //
  std::vector<cl_uint> J_set(sz_J, static_cast<cl_uint>(0));

  // computing size for blocks
  // writing set to arrays
  write_set_to_array(g_I, I_set);
  write_set_to_array(g_J, J_set);

  // if block for assembly does exist
  if (I_set.size() > 0 && J_set.size() > 0)
  {
    viennacl::context ctx = viennacl::traits::context(A);
    viennacl::ocl::context & opencl_ctx = const_cast<viennacl::ocl::context &>(ctx.opencl_context());
    compute_blocks_size(g_I, g_J, sz_blocks, blocks_ind, matrix_dims);
    std::vector<NumericT> con_A_I_J(sz_blocks, NumericT(0));

    block_vector set_I_vcl, set_J_vcl;
    //init memory on GPU
    //contigious g_A_I_J
    g_A_I_J_vcl.handle() = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                                    static_cast<unsigned int>(sizeof(NumericT)*(sz_blocks)),
                                                    &(con_A_I_J[0]));
    g_A_I_J_vcl.handle().context(opencl_ctx);

    //matrix_dimensions
    g_A_I_J_vcl.handle1() = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                                     static_cast<unsigned int>(sizeof(cl_uint)*2*static_cast<cl_uint>(g_I.size())),
                                                     &(matrix_dims[0]));
    g_A_I_J_vcl.handle1().context(opencl_ctx);

    //start_block inds
    g_A_I_J_vcl.handle2() = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                                     static_cast<unsigned int>(sizeof(cl_uint)*(g_I.size() + 1)),
                                                     &(blocks_ind[0]));
    g_A_I_J_vcl.handle2().context(opencl_ctx);

    //set_I
    set_I_vcl.handle() = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                                  static_cast<unsigned int>(sizeof(cl_uint)*sz_I),
                                                  &(I_set[0]));
    set_I_vcl.handle().context(opencl_ctx);

    //set_J
    set_J_vcl.handle() = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                                  static_cast<unsigned int>(sizeof(cl_uint)*sz_J),
                                                  &(J_set[0]));
    set_J_vcl.handle().context(opencl_ctx);

    //i_ind
    set_I_vcl.handle1() = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                                   static_cast<unsigned int>(sizeof(cl_uint)*(g_I.size() + 1)),
                                                   &(i_ind[0]));
    set_I_vcl.handle().context(opencl_ctx);

    //j_ind
    set_J_vcl.handle1() = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                                   static_cast<unsigned int>(sizeof(cl_uint)*(g_I.size() + 1)),
                                                   &(j_ind[0]));
    set_J_vcl.handle().context(opencl_ctx);

    viennacl::ocl::handle<cl_mem> g_is_update_vcl = opencl_ctx.create_memory(CL_MEM_READ_WRITE,
                                                                             static_cast<unsigned int>(sizeof(cl_uint)*g_is_update.size()),
                                                                             &(g_is_update[0]));

    viennacl::linalg::opencl::kernels::spai<NumericT>::init(opencl_ctx);
    viennacl::ocl::kernel& assembly_kernel = opencl_ctx.get_kernel(viennacl::linalg::opencl::kernels::spai<NumericT>::program_name(), "assemble_blocks");
    assembly_kernel.local_work_size(0, 1);
    assembly_kernel.global_work_size(0, 256);
    viennacl::ocl::enqueue(assembly_kernel(A.handle1().opencl_handle(), A.handle2().opencl_handle(), A.handle().opencl_handle(),
                                           set_I_vcl.handle(), set_J_vcl.handle(), set_I_vcl.handle1(),
                                           set_J_vcl.handle1(),
                                           g_A_I_J_vcl.handle2(), g_A_I_J_vcl.handle1(), g_A_I_J_vcl.handle(),
                                           g_is_update_vcl,
                                           static_cast<unsigned int>(g_I.size())));
    is_empty_block = false;
  }
  else
    is_empty_block = true;
}

/************************************************************************************************************************/

/** @brief Insertion of vectorized matrix column into original sparse matrix
 *
 * @param M_v       column-major vectorized matrix
 * @param M         original sparse matrix
 * @param is_right  indicates if matrix should be transposed in the output
 */
template<typename SparseMatrixT, typename SparseVectorT>
void insert_sparse_columns(std::vector<SparseVectorT> const & M_v,
                           SparseMatrixT& M,
                           bool is_right)
{
  if (is_right)
  {
    for (unsigned int i = 0; i < M_v.size(); ++i)
      for (typename SparseVectorT::const_iterator vec_it = M_v[i].begin(); vec_it!=M_v[i].end(); ++vec_it)
        M(vec_it->first, i) = vec_it->second;
  }
  else  //transposed fill of M
  {
    for (unsigned int i = 0; i < M_v.size(); ++i)
      for (typename SparseVectorT::const_iterator vec_it = M_v[i].begin(); vec_it!=M_v[i].end(); ++vec_it)
        M(i, vec_it->first) = vec_it->second;
  }
}

/** @brief Transposition of sparse matrix
 *
 * @param A_in      intial sparse matrix
 * @param A output  transposed matrix
 */
template<typename MatrixT>
void sparse_transpose(MatrixT const & A_in, MatrixT & A)
{
  typedef typename MatrixT::value_type         NumericType;

  std::vector<std::map<vcl_size_t, NumericType> >   temp_A(A_in.size2());
  A.resize(A_in.size2(), A_in.size1(), false);

  for (typename MatrixT::const_iterator1 row_it = A_in.begin1();
       row_it != A_in.end1();
       ++row_it)
  {
    for (typename MatrixT::const_iterator2 col_it = row_it.begin();
         col_it != row_it.end();
         ++col_it)
    {
      temp_A[col_it.index2()][col_it.index1()] = *col_it;
    }
  }

  for (vcl_size_t i=0; i<temp_A.size(); ++i)
  {
    for (typename std::map<vcl_size_t, NumericType>::const_iterator it = temp_A[i].begin();
         it != temp_A[i].end();
         ++it)
      A(i, it->first) = it->second;
  }
}




//        template<typename SparseVectorType>
//        void custom_copy(std::vector<SparseVectorType> & M_v, std::vector<SparseVectorType> & l_M_v, const unsigned int beg_ind){
//            for (int i = 0; i < l_M_v.size(); ++i){
//                l_M_v[i] = M_v[i + beg_ind];
//            }
//        }

//CPU version
/** @brief Construction of SPAI preconditioner on CPU
 *
 * @param A     initial sparse matrix
 * @param M     output preconditioner
 * @param tag   spai tag
 */
template<typename MatrixT>
void computeSPAI(MatrixT const & A,
                 MatrixT & M,
                 spai_tag & tag)
{
  typedef typename MatrixT::value_type                                       NumericT;
  typedef typename boost::numeric::ublas::vector<NumericT>                   VectorType;
  typedef typename viennacl::linalg::detail::spai::sparse_vector<NumericT>   SparseVectorType;
  typedef typename boost::numeric::ublas::matrix<NumericT>                   DenseMatrixType;

  //sparse matrix transpose...
  unsigned int cur_iter = 0;
  tag.setBegInd(0); tag.setEndInd(VIENNACL_SPAI_K_b);
  bool go_on = true;
  std::vector<SparseVectorType> A_v_c(M.size2());
  std::vector<SparseVectorType> M_v(M.size2());
  vectorize_column_matrix(A, A_v_c);
  vectorize_column_matrix(M, M_v);


  while (go_on)
  {
    go_on = (tag.getEndInd() < static_cast<long>(M.size2()));
    cur_iter = 0;
    unsigned int l_sz = static_cast<unsigned int>(tag.getEndInd() - tag.getBegInd());
    //std::vector<bool> g_is_update(M.size2(), true);
    std::vector<bool> g_is_update(l_sz, true);

    //init is update
    //init_parallel_is_update(g_is_update);
    //std::vector<SparseVectorType> A_v_c(K);
    //std::vector<SparseVectorType> M_v(K);
    //vectorization of marices
    //print_matrix(M_v);

    std::vector<SparseVectorType> l_M_v(l_sz);
    //custom_copy(M_v, l_M_v, beg_ind);
    std::copy(M_v.begin() + tag.getBegInd(), M_v.begin() + tag.getEndInd(), l_M_v.begin());

    //print_matrix(l_M_v);
    //std::vector<SparseVectorType> l_A_v_c(K);
    //custom_copy(A_v_c, l_A_v_c, beg_ind);
    //std::copy(A_v_c.begin() + beg_ind, A_v_c.begin() + end_ind, l_A_v_c.begin());
    //print_matrix(l_A_v_c);
    //vectorize_row_matrix(A, A_v_r);
    //working blocks

    std::vector<DenseMatrixType> g_A_I_J(l_sz);
    std::vector<VectorType> g_b_v(l_sz);
    std::vector<SparseVectorType> g_res(l_sz);
    std::vector<std::vector<unsigned int> > g_I(l_sz);
    std::vector<std::vector<unsigned int> > g_J(l_sz);

    while ((cur_iter < tag.getIterationLimit())&&is_all_update(g_is_update))
    {
      // SET UP THE BLOCKS..
      // PHASE ONE
      if (cur_iter == 0)
        block_set_up(A, A_v_c, l_M_v,  g_I, g_J, g_A_I_J, g_b_v);
      else
        block_update(A, A_v_c, g_res, g_is_update, g_I, g_J, g_b_v, g_A_I_J, tag);

      //PHASE TWO, LEAST SQUARE SOLUTION
      least_square_solve(A_v_c, g_A_I_J, g_b_v, g_I, g_J, g_res, g_is_update, l_M_v, tag);

      if (tag.getIsStatic()) break;
      cur_iter++;
    }

    std::copy(l_M_v.begin(), l_M_v.end(), M_v.begin() + tag.getBegInd());
    tag.setBegInd(tag.getEndInd());//beg_ind = end_ind;
    tag.setEndInd(std::min(static_cast<long>(tag.getBegInd() + VIENNACL_SPAI_K_b), static_cast<long>(M.size2())));
    //std::copy(l_M_v.begin(), l_M_v.end(), M_v.begin() + tag.getBegInd());
  }

  M.resize(M.size1(), M.size2(), false);
  insert_sparse_columns(M_v, M, tag.getIsRight());
}


//GPU - based version
/** @brief Construction of SPAI preconditioner on GPU
 *
 * @param A      initial sparse matrix
 * @param cpu_A  copy of initial matrix on CPU
 * @param cpu_M  output preconditioner on CPU
 * @param M      output preconditioner
 * @param tag    SPAI tag class with parameters
 */
template<typename NumericT, unsigned int AlignmentV>
void computeSPAI(viennacl::compressed_matrix<NumericT, AlignmentV> const & A, //input
                 boost::numeric::ublas::compressed_matrix<NumericT> const & cpu_A,
                 boost::numeric::ublas::compressed_matrix<NumericT> & cpu_M, //output
                 viennacl::compressed_matrix<NumericT, AlignmentV> & M,
                 spai_tag const & tag)
{
  typedef typename viennacl::linalg::detail::spai::sparse_vector<NumericT>        SparseVectorType;

  //typedef typename viennacl::compressed_matrix<ScalarType> GPUSparseMatrixType;
  //sparse matrix transpose...
  unsigned int cur_iter = 0;
  std::vector<cl_uint> g_is_update(cpu_M.size2(), static_cast<cl_uint>(1));
  //init is update
  //init_parallel_is_update(g_is_update);
  std::vector<SparseVectorType> A_v_c(cpu_M.size2());
  std::vector<SparseVectorType> M_v(cpu_M.size2());
  vectorize_column_matrix(cpu_A, A_v_c);
  vectorize_column_matrix(cpu_M, M_v);
  std::vector<SparseVectorType> g_res(cpu_M.size2());
  std::vector<std::vector<unsigned int> > g_I(cpu_M.size2());
  std::vector<std::vector<unsigned int> > g_J(cpu_M.size2());

  //OpenCL variables
  block_matrix g_A_I_J_vcl;
  block_vector g_bv_vcl;
  while ((cur_iter < tag.getIterationLimit())&&is_all_update(g_is_update))
  {
    // SET UP THE BLOCKS..
    // PHASE ONE..
    //timer.start();
    //index set up on CPU
    if (cur_iter == 0)
      block_set_up(A, A_v_c, M_v, g_is_update, g_I, g_J, g_A_I_J_vcl, g_bv_vcl);
    else
      block_update(A, A_v_c, g_is_update, g_res, g_J, g_I, g_A_I_J_vcl, g_bv_vcl, tag);
    //std::cout<<"Phase 2 timing: "<<timer.get()<<std::endl;
    //PERFORM LEAST SQUARE problems solution
    //PHASE TWO
    //timer.start();
    least_square_solve<SparseVectorType, NumericT>(A_v_c, M_v, g_I, g_J, g_A_I_J_vcl, g_bv_vcl, g_res, g_is_update, tag, viennacl::traits::context(A));
    //std::cout<<"Phase 3 timing: "<<timer.get()<<std::endl;
    if (tag.getIsStatic())
      break;
    cur_iter++;
  }

  cpu_M.resize(cpu_M.size1(), cpu_M.size2(), false);
  insert_sparse_columns(M_v, cpu_M, tag.getIsRight());
  //copy back to GPU
  M.resize(static_cast<unsigned int>(cpu_M.size1()), static_cast<unsigned int>(cpu_M.size2()));
  viennacl::copy(cpu_M, M);
}

}
}
}
}
#endif
