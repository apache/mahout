#ifndef VIENNACL_TRAITS_HANDLE_HPP_
#define VIENNACL_TRAITS_HANDLE_HPP_

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

/** @file viennacl/traits/handle.hpp
    @brief Extracts the underlying OpenCL handle from a vector, a matrix, an expression etc.
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"

#include "viennacl/backend/mem_handle.hpp"

namespace viennacl
{
namespace traits
{
//
// Generic memory handle
//
/** @brief Returns the generic memory handle of an object. Non-const version. */
template<typename T>
viennacl::backend::mem_handle & handle(T & obj)
{
  return obj.handle();
}

/** @brief Returns the generic memory handle of an object. Const-version. */
template<typename T>
viennacl::backend::mem_handle const & handle(T const & obj)
{
  return obj.handle();
}

/** \cond */
inline char   handle(char val)   { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline short  handle(short val)  { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline int    handle(int val)    { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline long   handle(long val)   { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline float  handle(float val)  { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline double handle(double val) { return val; }  //for unification purposes when passing CPU-scalars to kernels

template<typename LHS, typename RHS, typename OP>
viennacl::backend::mem_handle       & handle(viennacl::scalar_expression< const LHS, const RHS, OP> & obj)
{
  return handle(obj.lhs());
}

template<typename LHS, typename RHS, typename OP>
viennacl::backend::mem_handle const & handle(viennacl::matrix_expression<LHS, RHS, OP> const & obj);

template<typename LHS, typename RHS, typename OP>
viennacl::backend::mem_handle const & handle(viennacl::vector_expression<LHS, RHS, OP> const & obj);

template<typename LHS, typename RHS, typename OP>
viennacl::backend::mem_handle const & handle(viennacl::scalar_expression< const LHS, const RHS, OP> const & obj)
{
  return handle(obj.lhs());
}

// proxy objects require extra care (at the moment)
template<typename T>
viennacl::backend::mem_handle       & handle(viennacl::vector_base<T>       & obj)
{
  return obj.handle();
}

template<typename T>
viennacl::backend::mem_handle const & handle(viennacl::vector_base<T> const & obj)
{
  return obj.handle();
}



template<typename T>
viennacl::backend::mem_handle       & handle(viennacl::matrix_range<T>       & obj)
{
  return obj.get().handle();
}

template<typename T>
viennacl::backend::mem_handle const & handle(viennacl::matrix_range<T> const & obj)
{
  return obj.get().handle();
}


template<typename T>
viennacl::backend::mem_handle       & handle(viennacl::matrix_slice<T>      & obj)
{
  return obj.get().handle();
}

template<typename T>
viennacl::backend::mem_handle const & handle(viennacl::matrix_slice<T> const & obj)
{
  return obj.get().handle();
}

template<typename LHS, typename RHS, typename OP>
viennacl::backend::mem_handle const & handle(viennacl::vector_expression<LHS, RHS, OP> const & obj)
{
  return handle(obj.lhs());
}

template<typename LHS, typename RHS, typename OP>
viennacl::backend::mem_handle const & handle(viennacl::matrix_expression<LHS, RHS, OP> const & obj)
{
  return handle(obj.lhs());
}

/** \endcond */

//
// RAM handle extraction
//
/** @brief Generic helper routine for extracting the RAM handle of a ViennaCL object. Non-const version. */
template<typename T>
typename viennacl::backend::mem_handle::ram_handle_type & ram_handle(T & obj)
{
  return viennacl::traits::handle(obj).ram_handle();
}

/** @brief Generic helper routine for extracting the RAM handle of a ViennaCL object. Const version. */
template<typename T>
typename viennacl::backend::mem_handle::ram_handle_type const & ram_handle(T const & obj)
{
  return viennacl::traits::handle(obj).ram_handle();
}

/** \cond */
inline viennacl::backend::mem_handle::ram_handle_type & ram_handle(viennacl::backend::mem_handle & h)
{
  return h.ram_handle();
}

inline viennacl::backend::mem_handle::ram_handle_type const & ram_handle(viennacl::backend::mem_handle const & h)
{
  return h.ram_handle();
}
/** \endcond */

//
// OpenCL handle extraction
//
#ifdef VIENNACL_WITH_OPENCL
/** @brief Generic helper routine for extracting the OpenCL handle of a ViennaCL object. Non-const version. */
template<typename T>
viennacl::ocl::handle<cl_mem> & opencl_handle(T & obj)
{
  return viennacl::traits::handle(obj).opencl_handle();
}

/** @brief Generic helper routine for extracting the OpenCL handle of a ViennaCL object. Const version. */
template<typename T>
viennacl::ocl::handle<cl_mem> const & opencl_handle(T const & obj)
{
  return viennacl::traits::handle(obj).opencl_handle();
}

inline cl_char   opencl_handle(char            val) { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline cl_short  opencl_handle(short           val) { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline cl_int    opencl_handle(int             val) { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline cl_long   opencl_handle(long            val) { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline cl_uchar  opencl_handle(unsigned char   val) { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline cl_ushort opencl_handle(unsigned short  val) { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline cl_uint   opencl_handle(unsigned int    val) { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline cl_ulong  opencl_handle(unsigned long   val) { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline float     opencl_handle(float           val) { return val; }  //for unification purposes when passing CPU-scalars to kernels
inline double    opencl_handle(double          val) { return val; }  //for unification purposes when passing CPU-scalars to kernels


// for user-provided matrix-vector routines:
template<typename LHS, typename NumericT>
viennacl::ocl::handle<cl_mem> const & opencl_handle(viennacl::vector_expression<LHS, const vector_base<NumericT>, op_prod> const & obj)
{
  return viennacl::traits::handle(obj.rhs()).opencl_handle();
}

template<typename T>
viennacl::ocl::context & opencl_context(T const & obj)
{
  return const_cast<viennacl::ocl::context &>(opencl_handle(obj).context());
}
#endif

//
// OpenCL context extraction
//




//
// Active handle ID
//
/** @brief Returns an ID for the currently active memory domain of an object */
template<typename T>
viennacl::memory_types active_handle_id(T const & obj)
{
  return handle(obj).get_active_handle_id();
}

/** \cond */
template<typename T>
viennacl::memory_types active_handle_id(circulant_matrix<T> const &) { return OPENCL_MEMORY; }

template<typename T>
viennacl::memory_types active_handle_id(hankel_matrix<T> const &) { return OPENCL_MEMORY; }

template<typename T>
viennacl::memory_types active_handle_id(toeplitz_matrix<T> const &) { return OPENCL_MEMORY; }

template<typename T>
viennacl::memory_types active_handle_id(vandermonde_matrix<T> const &) { return OPENCL_MEMORY; }

template<typename LHS, typename RHS, typename OP>
viennacl::memory_types active_handle_id(viennacl::vector_expression<LHS, RHS, OP> const &);

template<typename LHS, typename RHS, typename OP>
viennacl::memory_types active_handle_id(viennacl::scalar_expression<LHS, RHS, OP> const & obj)
{
  return active_handle_id(obj.lhs());
}

template<typename LHS, typename RHS, typename OP>
viennacl::memory_types active_handle_id(viennacl::vector_expression<LHS, RHS, OP> const & obj)
{
  return active_handle_id(obj.lhs());
}

template<typename LHS, typename RHS, typename OP>
viennacl::memory_types active_handle_id(viennacl::matrix_expression<LHS, RHS, OP> const & obj)
{
  return active_handle_id(obj.lhs());
}

// for user-provided matrix-vector routines:
template<typename LHS, typename NumericT>
viennacl::memory_types active_handle_id(viennacl::vector_expression<LHS, const vector_base<NumericT>, op_prod> const & obj)
{
  return active_handle_id(obj.rhs());
}

/** \endcond */

} //namespace traits
} //namespace viennacl


#endif
