#ifndef VIENNACL_LINALG_VECTOR_OPERATIONS_HPP_
#define VIENNACL_LINALG_VECTOR_OPERATIONS_HPP_

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

/** @file viennacl/linalg/vector_operations.hpp
    @brief Implementations of vector operations.
*/

#include "viennacl/forwards.h"
#include "viennacl/range.hpp"
#include "viennacl/scalar.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/meta/predicate.hpp"
#include "viennacl/meta/enable_if.hpp"
#include "viennacl/traits/size.hpp"
#include "viennacl/traits/start.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/stride.hpp"
#include "viennacl/linalg/detail/op_executor.hpp"
#include "viennacl/linalg/host_based/vector_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
  #include "viennacl/linalg/opencl/vector_operations.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
  #include "viennacl/linalg/cuda/vector_operations.hpp"
#endif

namespace viennacl
{
  namespace linalg
  {
    template<typename DestNumericT, typename SrcNumericT>
    void convert(vector_base<DestNumericT> & dest, vector_base<SrcNumericT> const & src)
    {
      assert(viennacl::traits::size(dest) == viennacl::traits::size(src) && bool("Incompatible vector sizes in v1 = v2 (convert): size(v1) != size(v2)"));

      switch (viennacl::traits::handle(dest).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::convert(dest, src);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::convert(dest, src);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::convert(dest, src);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    template<typename T, typename ScalarType1>
    void av(vector_base<T> & vec1,
            vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha)
    {
      assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2) && bool("Incompatible vector sizes in v1 = v2 @ alpha: size(v1) != size(v2)"));

      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::av(vec1, vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::av(vec1, vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::av(vec1, vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename T, typename ScalarType1, typename ScalarType2>
    void avbv(vector_base<T> & vec1,
              vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
              vector_base<T> const & vec3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
    {
      assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2) && bool("Incompatible vector sizes in v1 = v2 @ alpha + v3 @ beta: size(v1) != size(v2)"));
      assert(viennacl::traits::size(vec2) == viennacl::traits::size(vec3) && bool("Incompatible vector sizes in v1 = v2 @ alpha + v3 @ beta: size(v2) != size(v3)"));

      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::avbv(vec1,
                                                  vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                  vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::avbv(vec1,
                                         vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::avbv(vec1,
                                       vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                       vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    template<typename T, typename ScalarType1, typename ScalarType2>
    void avbv_v(vector_base<T> & vec1,
                vector_base<T> const & vec2, ScalarType1 const & alpha, vcl_size_t len_alpha, bool reciprocal_alpha, bool flip_sign_alpha,
                vector_base<T> const & vec3, ScalarType2 const & beta,  vcl_size_t len_beta,  bool reciprocal_beta,  bool flip_sign_beta)
    {
      assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2) && bool("Incompatible vector sizes in v1 += v2 @ alpha + v3 @ beta: size(v1) != size(v2)"));
      assert(viennacl::traits::size(vec2) == viennacl::traits::size(vec3) && bool("Incompatible vector sizes in v1 += v2 @ alpha + v3 @ beta: size(v2) != size(v3)"));

      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::avbv_v(vec1,
                                                    vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                                    vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::avbv_v(vec1,
                                           vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                           vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::avbv_v(vec1,
                                         vec2, alpha, len_alpha, reciprocal_alpha, flip_sign_alpha,
                                         vec3,  beta, len_beta,  reciprocal_beta,  flip_sign_beta);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Assign a constant value to a vector (-range/-slice)
    *
    * @param vec1   The vector to which the value should be assigned
    * @param alpha  The value to be assigned
    * @param up_to_internal_size    Whether 'alpha' should be written to padded memory as well. This is used for setting all entries to zero, including padded memory.
    */
    template<typename T>
    void vector_assign(vector_base<T> & vec1, const T & alpha, bool up_to_internal_size = false)
    {
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::vector_assign(vec1, alpha, up_to_internal_size);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::vector_assign(vec1, alpha, up_to_internal_size);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::vector_assign(vec1, alpha, up_to_internal_size);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Swaps the contents of two vectors, data is copied
    *
    * @param vec1   The first vector (or -range, or -slice)
    * @param vec2   The second vector (or -range, or -slice)
    */
    template<typename T>
    void vector_swap(vector_base<T> & vec1, vector_base<T> & vec2)
    {
      assert(viennacl::traits::size(vec1) == viennacl::traits::size(vec2) && bool("Incompatible vector sizes in vector_swap()"));

      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::vector_swap(vec1, vec2);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::vector_swap(vec1, vec2);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::vector_swap(vec1, vec2);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    ///////////////////////// Elementwise operations /////////////



    /** @brief Implementation of the element-wise operation v1 = v2 .* v3 and v1 = v2 ./ v3    (using MATLAB syntax)
    *
    * @param vec1   The result vector (or -range, or -slice)
    * @param proxy  The proxy object holding v2, v3 and the operation
    */
    template<typename T, typename OP>
    void element_op(vector_base<T> & vec1,
                    vector_expression<const vector_base<T>, const vector_base<T>, OP> const & proxy)
    {
      assert(viennacl::traits::size(vec1) == viennacl::traits::size(proxy) && bool("Incompatible vector sizes in element_op()"));

      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::element_op(vec1, proxy);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::element_op(vec1, proxy);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::element_op(vec1, proxy);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** \cond */

// Helper macro for generating binary element-wise operations such as element_prod(), element_div(), element_pow() without unnecessary code duplication */
#define VIENNACL_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(OPNAME) \
    template<typename T> \
    viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<op_##OPNAME> > \
    element_##OPNAME(vector_base<T> const & v1, vector_base<T> const & v2) \
    { \
      return viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_binary<op_##OPNAME> >(v1, v2); \
    } \
\
    template<typename V1, typename V2, typename OP, typename T> \
    viennacl::vector_expression<const vector_expression<const V1, const V2, OP>, const vector_base<T>, op_element_binary<op_##OPNAME> > \
    element_##OPNAME(vector_expression<const V1, const V2, OP> const & proxy, vector_base<T> const & v2) \
    { \
      return viennacl::vector_expression<const vector_expression<const V1, const V2, OP>, const vector_base<T>, op_element_binary<op_##OPNAME> >(proxy, v2); \
    } \
\
    template<typename T, typename V2, typename V3, typename OP> \
    viennacl::vector_expression<const vector_base<T>, const vector_expression<const V2, const V3, OP>, op_element_binary<op_##OPNAME> > \
    element_##OPNAME(vector_base<T> const & v1, vector_expression<const V2, const V3, OP> const & proxy) \
    { \
      return viennacl::vector_expression<const vector_base<T>, const vector_expression<const V2, const V3, OP>, op_element_binary<op_##OPNAME> >(v1, proxy); \
    } \
\
    template<typename V1, typename V2, typename OP1, \
              typename V3, typename V4, typename OP2> \
    viennacl::vector_expression<const vector_expression<const V1, const V2, OP1>, \
                                const vector_expression<const V3, const V4, OP2>, \
                                op_element_binary<op_##OPNAME> > \
    element_##OPNAME(vector_expression<const V1, const V2, OP1> const & proxy1, \
                     vector_expression<const V3, const V4, OP2> const & proxy2) \
    {\
      return viennacl::vector_expression<const vector_expression<const V1, const V2, OP1>, \
                                         const vector_expression<const V3, const V4, OP2>, \
                                         op_element_binary<op_##OPNAME> >(proxy1, proxy2); \
    }

    VIENNACL_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(prod)  //for element_prod()
    VIENNACL_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(div)   //for element_div()
    VIENNACL_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(pow)   //for element_pow()

    VIENNACL_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(eq)
    VIENNACL_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(neq)
    VIENNACL_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(greater)
    VIENNACL_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(less)
    VIENNACL_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(geq)
    VIENNACL_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS(leq)

#undef VIENNACL_GENERATE_BINARY_ELEMENTOPERATION_OVERLOADS

// Helper macro for generating unary element-wise operations such as element_exp(), element_sin(), etc. without unnecessary code duplication */
#define VIENNACL_MAKE_UNARY_ELEMENT_OP(funcname) \
    template<typename T> \
    viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<op_##funcname> > \
    element_##funcname(vector_base<T> const & v) \
    { \
      return viennacl::vector_expression<const vector_base<T>, const vector_base<T>, op_element_unary<op_##funcname> >(v, v); \
    } \
    template<typename LHS, typename RHS, typename OP> \
    viennacl::vector_expression<const vector_expression<const LHS, const RHS, OP>, \
                                const vector_expression<const LHS, const RHS, OP>, \
                                op_element_unary<op_##funcname> > \
    element_##funcname(vector_expression<const LHS, const RHS, OP> const & proxy) \
    { \
      return viennacl::vector_expression<const vector_expression<const LHS, const RHS, OP>, \
                                         const vector_expression<const LHS, const RHS, OP>, \
                                         op_element_unary<op_##funcname> >(proxy, proxy); \
    } \

    VIENNACL_MAKE_UNARY_ELEMENT_OP(abs)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(acos)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(asin)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(atan)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(ceil)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(cos)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(cosh)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(exp)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(fabs)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(floor)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(log)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(log10)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(sin)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(sinh)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(sqrt)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(tan)
    VIENNACL_MAKE_UNARY_ELEMENT_OP(tanh)

#undef VIENNACL_MAKE_UNARY_ELEMENT_OP

    /** \endcond */

    ///////////////////////// Norms and inner product ///////////////////


    //implementation of inner product:
    //namespace {

    /** @brief Computes the inner product of two vectors - dispatcher interface
     *
     * @param vec1 The first vector
     * @param vec2 The second vector
     * @param result The result scalar (on the gpu)
     */
    template<typename T>
    void inner_prod_impl(vector_base<T> const & vec1,
                         vector_base<T> const & vec2,
                         scalar<T> & result)
    {
      assert( vec1.size() == vec2.size() && bool("Size mismatch") );

      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::inner_prod_impl(vec1, vec2, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::inner_prod_impl(vec1, vec2, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::inner_prod_impl(vec1, vec2, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    // vector expression on lhs
    template<typename LHS, typename RHS, typename OP, typename T>
    void inner_prod_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec1,
                         vector_base<T> const & vec2,
                         scalar<T> & result)
    {
      viennacl::vector<T> temp = vec1;
      inner_prod_impl(temp, vec2, result);
    }


    // vector expression on rhs
    template<typename T, typename LHS, typename RHS, typename OP>
    void inner_prod_impl(vector_base<T> const & vec1,
                         viennacl::vector_expression<LHS, RHS, OP> const & vec2,
                         scalar<T> & result)
    {
      viennacl::vector<T> temp = vec2;
      inner_prod_impl(vec1, temp, result);
    }


    // vector expression on lhs and rhs
    template<typename LHS1, typename RHS1, typename OP1,
              typename LHS2, typename RHS2, typename OP2, typename T>
    void inner_prod_impl(viennacl::vector_expression<LHS1, RHS1, OP1> const & vec1,
                         viennacl::vector_expression<LHS2, RHS2, OP2> const & vec2,
                         scalar<T> & result)
    {
      viennacl::vector<T> temp1 = vec1;
      viennacl::vector<T> temp2 = vec2;
      inner_prod_impl(temp1, temp2, result);
    }




    /** @brief Computes the inner product of two vectors with the final reduction step on the CPU - dispatcher interface
     *
     * @param vec1 The first vector
     * @param vec2 The second vector
     * @param result The result scalar (on the gpu)
     */
    template<typename T>
    void inner_prod_cpu(vector_base<T> const & vec1,
                        vector_base<T> const & vec2,
                        T & result)
    {
      assert( vec1.size() == vec2.size() && bool("Size mismatch") );

      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::inner_prod_impl(vec1, vec2, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::inner_prod_cpu(vec1, vec2, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::inner_prod_cpu(vec1, vec2, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    // vector expression on lhs
    template<typename LHS, typename RHS, typename OP, typename T>
    void inner_prod_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec1,
                        vector_base<T> const & vec2,
                        T & result)
    {
      viennacl::vector<T> temp = vec1;
      inner_prod_cpu(temp, vec2, result);
    }


    // vector expression on rhs
    template<typename T, typename LHS, typename RHS, typename OP>
    void inner_prod_cpu(vector_base<T> const & vec1,
                        viennacl::vector_expression<LHS, RHS, OP> const & vec2,
                        T & result)
    {
      viennacl::vector<T> temp = vec2;
      inner_prod_cpu(vec1, temp, result);
    }


    // vector expression on lhs and rhs
    template<typename LHS1, typename RHS1, typename OP1,
              typename LHS2, typename RHS2, typename OP2, typename S3>
    void inner_prod_cpu(viennacl::vector_expression<LHS1, RHS1, OP1> const & vec1,
                        viennacl::vector_expression<LHS2, RHS2, OP2> const & vec2,
                        S3 & result)
    {
      viennacl::vector<S3> temp1 = vec1;
      viennacl::vector<S3> temp2 = vec2;
      inner_prod_cpu(temp1, temp2, result);
    }



    /** @brief Computes the inner products <x, y1>, <x, y2>, ..., <x, y_N> and writes the result to a (sub-)vector
     *
     * @param x       The common vector
     * @param y_tuple A collection of vector, all of the same size.
     * @param result  The result scalar (on the gpu). Needs to match the number of elements in y_tuple
     */
    template<typename T>
    void inner_prod_impl(vector_base<T> const & x,
                         vector_tuple<T> const & y_tuple,
                         vector_base<T> & result)
    {
      assert( x.size() == y_tuple.const_at(0).size() && bool("Size mismatch") );
      assert( result.size() == y_tuple.const_size() && bool("Number of elements does not match result size") );

      switch (viennacl::traits::handle(x).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::inner_prod_impl(x, y_tuple, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::inner_prod_impl(x, y_tuple, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::inner_prod_impl(x, y_tuple, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Computes the l^1-norm of a vector - dispatcher interface
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_1_impl(vector_base<T> const & vec,
                     scalar<T> & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::norm_1_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::norm_1_impl(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::norm_1_impl(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }


    /** @brief Computes the l^1-norm of a vector - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void norm_1_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                     S2 & result)
    {
      viennacl::vector<typename viennacl::result_of::cpu_value_type<S2>::type> temp = vec;
      norm_1_impl(temp, result);
    }



    /** @brief Computes the l^1-norm of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_1_cpu(vector_base<T> const & vec,
                    T & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::norm_1_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::norm_1_cpu(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::norm_1_cpu(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the l^1-norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void norm_1_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                    S2 & result)
    {
      viennacl::vector<typename viennacl::result_of::cpu_value_type<LHS>::type> temp = vec;
      norm_1_cpu(temp, result);
    }




    /** @brief Computes the l^2-norm of a vector - dispatcher interface
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_2_impl(vector_base<T> const & vec,
                     scalar<T> & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::norm_2_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::norm_2_impl(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::norm_2_impl(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the l^2-norm of a vector - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename T>
    void norm_2_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                     scalar<T> & result)
    {
      viennacl::vector<T> temp = vec;
      norm_2_impl(temp, result);
    }


    /** @brief Computes the l^2-norm of a vector with final reduction on the CPU - dispatcher interface
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_2_cpu(vector_base<T> const & vec,
                    T & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::norm_2_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::norm_2_cpu(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::norm_2_cpu(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the l^2-norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void norm_2_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                    S2 & result)
    {
      viennacl::vector<typename viennacl::result_of::cpu_value_type<LHS>::type> temp = vec;
      norm_2_cpu(temp, result);
    }




    /** @brief Computes the supremum-norm of a vector
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_inf_impl(vector_base<T> const & vec,
                       scalar<T> & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::norm_inf_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::norm_inf_impl(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::norm_inf_impl(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename T>
    void norm_inf_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                       scalar<T> & result)
    {
      viennacl::vector<T> temp = vec;
      norm_inf_impl(temp, result);
    }


    /** @brief Computes the supremum-norm of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void norm_inf_cpu(vector_base<T> const & vec,
                      T & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::norm_inf_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::norm_inf_cpu(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::norm_inf_cpu(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void norm_inf_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec,
                      S2 & result)
    {
      viennacl::vector<typename viennacl::result_of::cpu_value_type<LHS>::type> temp = vec;
      norm_inf_cpu(temp, result);
    }


    //This function should return a CPU scalar, otherwise statements like
    // vcl_rhs[index_norm_inf(vcl_rhs)]
    // are ambiguous
    /** @brief Computes the index of the first entry that is equal to the supremum-norm in modulus.
    *
    * @param vec The vector
    * @return The result. Note that the result must be a CPU scalar
    */
    template<typename T>
    vcl_size_t index_norm_inf(vector_base<T> const & vec)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          return viennacl::linalg::host_based::index_norm_inf(vec);
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          return viennacl::linalg::opencl::index_norm_inf(vec);
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          return viennacl::linalg::cuda::index_norm_inf(vec);
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    */
    template<typename LHS, typename RHS, typename OP>
    vcl_size_t index_norm_inf(viennacl::vector_expression<LHS, RHS, OP> const & vec)
    {
      viennacl::vector<typename viennacl::result_of::cpu_value_type<LHS>::type> temp = vec;
      return index_norm_inf(temp);
    }

///////////////////

    /** @brief Computes the maximum of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename NumericT>
    void max_impl(vector_base<NumericT> const & vec, viennacl::scalar<NumericT> & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::max_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::max_impl(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::max_impl(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename NumericT>
    void max_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec, viennacl::scalar<NumericT> & result)
    {
      viennacl::vector<NumericT> temp = vec;
      max_impl(temp, result);
    }


    /** @brief Computes the maximum of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void max_cpu(vector_base<T> const & vec, T & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::max_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::max_cpu(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::max_cpu(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void max_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec, S2 & result)
    {
      viennacl::vector<typename viennacl::result_of::cpu_value_type<LHS>::type> temp = vec;
      max_cpu(temp, result);
    }

///////////////////

    /** @brief Computes the minimum of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename NumericT>
    void min_impl(vector_base<NumericT> const & vec, viennacl::scalar<NumericT> & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::min_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::min_impl(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::min_impl(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename NumericT>
    void min_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec, viennacl::scalar<NumericT> & result)
    {
      viennacl::vector<NumericT> temp = vec;
      min_impl(temp, result);
    }


    /** @brief Computes the minimum of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void min_cpu(vector_base<T> const & vec, T & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::min_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::min_cpu(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::min_cpu(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the supremum norm of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void min_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec, S2 & result)
    {
      viennacl::vector<typename viennacl::result_of::cpu_value_type<LHS>::type> temp = vec;
      min_cpu(temp, result);
    }

///////////////////

    /** @brief Computes the sum of a vector with final reduction on the device (GPU, etc.)
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename NumericT>
    void sum_impl(vector_base<NumericT> const & vec, viennacl::scalar<NumericT> & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::sum_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::sum_impl(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::sum_impl(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the sum of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename NumericT>
    void sum_impl(viennacl::vector_expression<LHS, RHS, OP> const & vec, viennacl::scalar<NumericT> & result)
    {
      viennacl::vector<NumericT> temp = vec;
      sum_impl(temp, result);
    }


    /** @brief Computes the sum of a vector with final reduction on the CPU
    *
    * @param vec The vector
    * @param result The result scalar
    */
    template<typename T>
    void sum_cpu(vector_base<T> const & vec, T & result)
    {
      switch (viennacl::traits::handle(vec).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::sum_impl(vec, result);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::sum_cpu(vec, result);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::sum_cpu(vec, result);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Computes the sum of a vector with final reduction on the CPU - interface for a vector expression. Creates a temporary.
    *
    * @param vec    The vector expression
    * @param result The result scalar
    */
    template<typename LHS, typename RHS, typename OP, typename S2>
    void sum_cpu(viennacl::vector_expression<LHS, RHS, OP> const & vec, S2 & result)
    {
      viennacl::vector<typename viennacl::result_of::cpu_value_type<LHS>::type> temp = vec;
      sum_cpu(temp, result);
    }





    /** @brief Computes a plane rotation of two vectors.
    *
    * Computes (x,y) <- (alpha * x + beta * y, -beta * x + alpha * y)
    *
    * @param vec1   The first vector
    * @param vec2   The second vector
    * @param alpha  The first transformation coefficient (CPU scalar)
    * @param beta   The second transformation coefficient (CPU scalar)
    */
    template<typename T>
    void plane_rotation(vector_base<T> & vec1,
                        vector_base<T> & vec2,
                        T alpha, T beta)
    {
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::plane_rotation(vec1, vec2, alpha, beta);
          break;
#ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::plane_rotation(vec1, vec2, alpha, beta);
          break;
#endif
#ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::plane_rotation(vec1, vec2, alpha, beta);
          break;
#endif
        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief This function implements an inclusive scan.
    *
    * Given an element vector (x_0, x_1, ..., x_{n-1}),
    * this routine computes (x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-1})
    *
    * The two vectors either need to be the same (in-place), or reside in distinct memory regions.
    * Partial overlaps of vec1 and vec2 are not allowed.
    *
    * @param vec1       Input vector.
    * @param vec2       The output vector.
    */
    template<typename NumericT>
    void inclusive_scan(vector_base<NumericT> & vec1,
                        vector_base<NumericT> & vec2)
    {
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::inclusive_scan(vec1, vec2);
          break;
  #ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::inclusive_scan(vec1, vec2);
          break;
  #endif

  #ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::inclusive_scan(vec1, vec2);
          break;
  #endif

        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Implements an in-place inclusive scan of a vector.
    *
    * Given an input element vector (x_0, x_1, ..., x_{n-1}),
    * this routine overwrites the vector with (x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-1})
    */
    template<typename NumericT>
    void inclusive_scan(vector_base<NumericT> & vec)
    {
      inclusive_scan(vec, vec);
    }

    /** @brief This function implements an exclusive scan.
    *
    * Given an element vector (x_0, x_1, ..., x_{n-1}),
    * this routine computes (0, x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-2})
    *
    * The two vectors either need to be the same (in-place), or reside in distinct memory regions.
    * Partial overlaps of vec1 and vec2 are not allowed.
    *
    * @param vec1       Input vector.
    * @param vec2       The output vector.
    */
    template<typename NumericT>
    void exclusive_scan(vector_base<NumericT> & vec1,
                        vector_base<NumericT> & vec2)
    {
      switch (viennacl::traits::handle(vec1).get_active_handle_id())
      {
        case viennacl::MAIN_MEMORY:
          viennacl::linalg::host_based::exclusive_scan(vec1, vec2);
          break;
  #ifdef VIENNACL_WITH_OPENCL
        case viennacl::OPENCL_MEMORY:
          viennacl::linalg::opencl::exclusive_scan(vec1, vec2);
          break;
  #endif

  #ifdef VIENNACL_WITH_CUDA
        case viennacl::CUDA_MEMORY:
          viennacl::linalg::cuda::exclusive_scan(vec1, vec2);
          break;
  #endif

        case viennacl::MEMORY_NOT_INITIALIZED:
          throw memory_exception("not initialised!");
        default:
          throw memory_exception("not implemented");
      }
    }

    /** @brief Inplace exclusive scan of a vector
    *
    * Given an element vector (x_0, x_1, ..., x_{n-1}),
    * this routine overwrites the input vector with (0, x_0, x_0 + x_1, ..., x_0 + x_1 + ... + x_{n-2})
    */
    template<typename NumericT>
    void exclusive_scan(vector_base<NumericT> & vec)
    {
      exclusive_scan(vec, vec);
    }
  } //namespace linalg

  template<typename T, typename LHS, typename RHS, typename OP>
  vector_base<T> & operator += (vector_base<T> & v1, const vector_expression<const LHS, const RHS, OP> & proxy)
  {
    assert( (viennacl::traits::size(proxy) == v1.size()) && bool("Incompatible vector sizes!"));
    assert( (v1.size() > 0) && bool("Vector not yet initialized!") );

    linalg::detail::op_executor<vector_base<T>, op_inplace_add, vector_expression<const LHS, const RHS, OP> >::apply(v1, proxy);

    return v1;
  }

  template<typename T, typename LHS, typename RHS, typename OP>
  vector_base<T> & operator -= (vector_base<T> & v1, const vector_expression<const LHS, const RHS, OP> & proxy)
  {
    assert( (viennacl::traits::size(proxy) == v1.size()) && bool("Incompatible vector sizes!"));
    assert( (v1.size() > 0) && bool("Vector not yet initialized!") );

    linalg::detail::op_executor<vector_base<T>, op_inplace_sub, vector_expression<const LHS, const RHS, OP> >::apply(v1, proxy);

    return v1;
  }

} //namespace viennacl


#endif
