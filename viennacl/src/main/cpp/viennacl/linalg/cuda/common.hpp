#ifndef VIENNACL_LINALG_CUDA_COMMON_HPP_
#define VIENNACL_LINALG_CUDA_COMMON_HPP_

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

/** @file viennacl/linalg/cuda/common.hpp
    @brief Common routines for CUDA execution
*/

#include <sstream>
#include <cuda_runtime.h>
#include "viennacl/backend/cuda.hpp"
#include "viennacl/traits/handle.hpp"

#define VIENNACL_CUDA_LAST_ERROR_CHECK(message)  detail::cuda_last_error_check (message, __FILE__, __LINE__)

namespace viennacl
{

////// scalar

/** @brief Convenience helper function for extracting the CUDA handle from a ViennaCL scalar. Non-const version. */
template<typename NumericT>
NumericT * cuda_arg(scalar<NumericT> & obj)
{
  return reinterpret_cast<NumericT *>(viennacl::traits::handle(obj).cuda_handle().get());
}

/** @brief Convenience helper function for extracting the CUDA handle from a ViennaCL scalar. Const version. */
template<typename NumericT>
const NumericT * cuda_arg(scalar<NumericT> const & obj)
{
  return reinterpret_cast<const NumericT *>(viennacl::traits::handle(obj).cuda_handle().get());
}


////// vector_base

/** @brief Convenience helper function for extracting the CUDA handle from a ViennaCL vector (through the base class vector_base) with implicit return type deduction. Non-const version. */
template<typename NumericT>
NumericT * cuda_arg(vector_base<NumericT> & obj)
{
  return reinterpret_cast<NumericT *>(viennacl::traits::handle(obj).cuda_handle().get());
}

/** @brief Convenience helper function for extracting the CUDA handle from a ViennaCL vector (through the base class vector_base) with implicit return type deduction. Const version. */
template<typename NumericT>
const NumericT * cuda_arg(vector_base<NumericT> const & obj)
{
  return reinterpret_cast<const NumericT *>(viennacl::traits::handle(obj).cuda_handle().get());
}

/** @brief Convenience helper function for extracting the CUDA handle from a ViennaCL vector (through the base class vector_base). Return type needs to be explicitly provided as first template argument. Non-const version. */
template<typename ReturnT, typename NumericT>
ReturnT * cuda_arg(vector_base<NumericT> & obj)
{
  return reinterpret_cast<ReturnT *>(viennacl::traits::handle(obj).cuda_handle().get());
}

/** @brief Convenience helper function for extracting the CUDA handle from a ViennaCL vector (through the base class vector_base). Return type needs to be explicitly provided as first template argument. Const version. */
template<typename ReturnT, typename NumericT>
const ReturnT * cuda_arg(vector_base<NumericT> const & obj)
{
  return reinterpret_cast<const ReturnT *>(viennacl::traits::handle(obj).cuda_handle().get());
}


////// matrix_base

/** @brief Convenience helper function for extracting the CUDA handle from a ViennaCL matrix (through the base class matrix_base). Non-const version. */
template<typename NumericT>
NumericT * cuda_arg(matrix_base<NumericT> & obj)
{
  return reinterpret_cast<NumericT *>(viennacl::traits::handle(obj).cuda_handle().get());
}

/** @brief Convenience helper function for extracting the CUDA handle from a ViennaCL matrix (through the base class matrix_base). Const version. */
template<typename NumericT>
const NumericT * cuda_arg(matrix_base<NumericT> const & obj)
{
  return reinterpret_cast<const NumericT *>(viennacl::traits::handle(obj).cuda_handle().get());
}



////// mem_handle


/** @brief Convenience helper function for extracting the CUDA handle from a generic memory handle. Non-const version. */
template<typename ReturnT>
ReturnT * cuda_arg(viennacl::backend::mem_handle & h)
{
  return reinterpret_cast<ReturnT *>(h.cuda_handle().get());
}

/** @brief Convenience helper function for extracting the CUDA handle from a generic memory handle. Const-version. */
template<typename ReturnT>
ReturnT const * cuda_arg(viennacl::backend::mem_handle const & h)
{
  return reinterpret_cast<const ReturnT *>(h.cuda_handle().get());
}

/** \cond **/
template<typename ReturnT>
ReturnT * cuda_arg(viennacl::backend::mem_handle::cuda_handle_type & h)
{
  return reinterpret_cast<ReturnT *>(h.get());
}

template<typename ReturnT>
ReturnT const *  cuda_arg(viennacl::backend::mem_handle::cuda_handle_type const & h)
{
  return reinterpret_cast<const ReturnT *>(h.get());
}

inline unsigned int cuda_arg(unsigned int val)  { return val; }

template<typename NumericT> char           cuda_arg(char val)           { return val; }
template<typename NumericT> unsigned char  cuda_arg(unsigned char val)  { return val; }

template<typename NumericT> short          cuda_arg(short val)          { return val; }
template<typename NumericT> unsigned short cuda_arg(unsigned short val) { return val; }

template<typename NumericT> int            cuda_arg(int val)            { return val; }
template<typename NumericT> unsigned int   cuda_arg(unsigned int val)   { return val; }

template<typename NumericT> long           cuda_arg(long val)           { return val; }
template<typename NumericT> unsigned long  cuda_arg(unsigned long val)  { return val; }

template<typename NumericT> float          cuda_arg(float val)          { return val; }
template<typename NumericT> double         cuda_arg(double val)         { return val; }

/** \endcond */


namespace linalg
{
namespace cuda
{


namespace detail
{

inline unsigned int make_options(vcl_size_t length, bool reciprocal, bool flip_sign)
{
  return static_cast<unsigned int>( ((length > 1) ? (static_cast<unsigned int>(length) << 2) : 0) + (reciprocal ? 2 : 0) + (flip_sign ? 1 : 0) );
}

inline void cuda_last_error_check(const char * message, const char * file, const int line )
{
  cudaError_t error_code = cudaGetLastError();

  if (cudaSuccess != error_code)
  {
    std::stringstream ss;
    ss << file << "(" << line << "): " << ": getLastCudaError() CUDA error " << error_code << ": " << cudaGetErrorString( error_code ) << " @ " << message << std::endl;
    throw viennacl::backend::cuda::cuda_exception(ss.str(), error_code);
  }
}

template<typename NumericT>
struct type_to_type2;

template<>
struct type_to_type2<float> { typedef float2  type; };

template<>
struct type_to_type2<double> { typedef double2  type; };


template<typename NumericT, typename OtherT>
typename viennacl::backend::mem_handle::cuda_handle_type & arg_reference(viennacl::scalar<NumericT> & s, OtherT) { return s.handle().cuda_handle(); }

template<typename NumericT, typename OtherT>
typename viennacl::backend::mem_handle::cuda_handle_type const & arg_reference(viennacl::scalar<NumericT> const & s, OtherT) { return s.handle().cuda_handle(); }

// all other cases where T is not a ViennaCL scalar
template<typename ArgT>
typename viennacl::enable_if< viennacl::is_cpu_scalar<ArgT>::value,
                              char const &>::type
arg_reference(ArgT, char const & val)  { return val; }

template<typename ArgT>
typename viennacl::enable_if< viennacl::is_cpu_scalar<ArgT>::value,
                              unsigned char const &>::type
arg_reference(ArgT, unsigned char const & val)  { return val; }

template<typename ArgT>
typename viennacl::enable_if< viennacl::is_cpu_scalar<ArgT>::value,
                              short const &>::type
arg_reference(ArgT, short const & val)  { return val; }

template<typename ArgT>
typename viennacl::enable_if< viennacl::is_cpu_scalar<ArgT>::value,
                              unsigned short const &>::type
arg_reference(ArgT, unsigned short const & val)  { return val; }

template<typename ArgT>
typename viennacl::enable_if< viennacl::is_cpu_scalar<ArgT>::value,
                              int const &>::type
arg_reference(ArgT, int const & val)  { return val; }

template<typename ArgT>
typename viennacl::enable_if< viennacl::is_cpu_scalar<ArgT>::value,
                              unsigned int const &>::type
arg_reference(ArgT, unsigned int const & val)  { return val; }

template<typename ArgT>
typename viennacl::enable_if< viennacl::is_cpu_scalar<ArgT>::value,
                              long const &>::type
arg_reference(ArgT, long const & val)  { return val; }

template<typename ArgT>
typename viennacl::enable_if< viennacl::is_cpu_scalar<ArgT>::value,
                              unsigned long const &>::type
arg_reference(ArgT, unsigned long const & val)  { return val; }

template<typename ArgT>
typename viennacl::enable_if< viennacl::is_cpu_scalar<ArgT>::value,
                              float const &>::type
arg_reference(ArgT, float const & val)  { return val; }

template<typename ArgT>
typename viennacl::enable_if< viennacl::is_cpu_scalar<ArgT>::value,
                              double const &>::type
arg_reference(ArgT, double const & val)  { return val; }

} //namespace detail
} //namespace cuda
} //namespace linalg
} //namespace viennacl


#endif
