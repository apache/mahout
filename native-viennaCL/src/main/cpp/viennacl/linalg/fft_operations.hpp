#ifndef VIENNACL_LINALG_FFT_OPERATIONS_HPP_
#define VIENNACL_LINALG_FFT_OPERATIONS_HPP_

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

/**  @file viennacl/linalg/fft_operations.hpp
 @brief Implementations of Fast Furier Transformation.
 */

#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>

#include "viennacl/linalg/host_based/fft_operations.hpp"

#ifdef VIENNACL_WITH_OPENCL
#include "viennacl/linalg/opencl/fft_operations.hpp"
#include "viennacl/linalg/opencl/kernels/fft.hpp"
#endif

#ifdef VIENNACL_WITH_CUDA
#include "viennacl/linalg/cuda/fft_operations.hpp"
#endif

namespace viennacl
{
namespace linalg
{

/**
 * @brief Direct 1D algorithm for computing Fourier transformation.
 *
 * Works on any sizes of data.
 * Serial implementation has o(n^2) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void direct(viennacl::vector<NumericT, AlignmentV> const & in,
            viennacl::vector<NumericT, AlignmentV>       & out, vcl_size_t size, vcl_size_t stride,
            vcl_size_t batch_num, NumericT sign = NumericT(-1),
            viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{

  switch (viennacl::traits::handle(in).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::direct(in, out, size, stride, batch_num, sign, data_order);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::direct(viennacl::traits::opencl_handle(in), viennacl::traits::opencl_handle(out), size, stride, batch_num, sign,data_order);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::direct(in, out, size, stride, batch_num,sign,data_order);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");

  }
}

/**
 * @brief Direct 2D algorithm for computing Fourier transformation.
 *
 * Works on any sizes of data.
 * Serial implementation has o(n^2) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void direct(viennacl::matrix<NumericT, viennacl::row_major, AlignmentV> const & in,
            viennacl::matrix<NumericT, viennacl::row_major, AlignmentV>& out, vcl_size_t size,
            vcl_size_t stride, vcl_size_t batch_num, NumericT sign = NumericT(-1),
            viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{

  switch (viennacl::traits::handle(in).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::direct(in, out, size, stride, batch_num, sign, data_order);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::direct(viennacl::traits::opencl_handle(in), viennacl::traits::opencl_handle(out), size, stride, batch_num, sign,data_order);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::direct(in, out, size, stride, batch_num,sign,data_order);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");

  }
}

/*
 * This function performs reorder of input data. Indexes are sorted in bit-reversal order.
 * Such reordering should be done before in-place FFT.
 */
template<typename NumericT, unsigned int AlignmentV>
void reorder(viennacl::vector<NumericT, AlignmentV>& in, vcl_size_t size, vcl_size_t stride,
             vcl_size_t bits_datasize, vcl_size_t batch_num,
             viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  switch (viennacl::traits::handle(in).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::reorder(in, size, stride, bits_datasize, batch_num, data_order);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::reorder<NumericT>(viennacl::traits::opencl_handle(in), size, stride, bits_datasize, batch_num, data_order);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::reorder(in, size, stride, bits_datasize, batch_num, data_order);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");

  }
}

/**
 * @brief Radix-2 1D algorithm for computing Fourier transformation.
 *
 * Works only on power-of-two sizes of data.
 * Serial implementation has o(n * lg n) complexity.
 * This is a Cooley-Tukey algorithm
 */
template<typename NumericT, unsigned int AlignmentV>
void radix2(viennacl::matrix<NumericT, viennacl::row_major, AlignmentV> & in, vcl_size_t size,
            vcl_size_t stride, vcl_size_t batch_num, NumericT sign = NumericT(-1),
            viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{
  switch (viennacl::traits::handle(in).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::radix2(in, size, stride, batch_num, sign, data_order);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::radix2(viennacl::traits::opencl_handle(in), size, stride, batch_num, sign,data_order);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::radix2(in, size, stride, batch_num, sign, data_order);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Radix-2 2D algorithm for computing Fourier transformation.
 *
 * Works only on power-of-two sizes of data.
 * Serial implementation has o(n * lg n) complexity.
 * This is a Cooley-Tukey algorithm
 */
template<typename NumericT, unsigned int AlignmentV>
void radix2(viennacl::vector<NumericT, AlignmentV>& in, vcl_size_t size, vcl_size_t stride,
            vcl_size_t batch_num, NumericT sign = NumericT(-1),
            viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::DATA_ORDER data_order = viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR)
{

  switch (viennacl::traits::handle(in).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::radix2(in, size, stride, batch_num, sign, data_order);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::radix2(viennacl::traits::opencl_handle(in), size, stride, batch_num, sign,data_order);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::radix2(in, size, stride, batch_num, sign,data_order);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Bluestein's algorithm for computing Fourier transformation.
 *
 * Currently,  Works only for sizes of input data which less than 2^16.
 * Uses a lot of additional memory, but should be fast for any size of data.
 * Serial implementation has something about o(n * lg n) complexity
 */
template<typename NumericT, unsigned int AlignmentV>
void bluestein(viennacl::vector<NumericT, AlignmentV> & in,
               viennacl::vector<NumericT, AlignmentV> & out, vcl_size_t /*batch_num*/)
{

  switch (viennacl::traits::handle(in).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::bluestein(in, out, 1);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::bluestein(in, out, 1);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::bluestein(in, out, 1);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Mutiply two complex vectors and store result in output
 */
template<typename NumericT, unsigned int AlignmentV>
void multiply_complex(viennacl::vector<NumericT, AlignmentV> const & input1,
                      viennacl::vector<NumericT, AlignmentV> const & input2,
                      viennacl::vector<NumericT, AlignmentV>       & output)
{
  switch (viennacl::traits::handle(input1).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::multiply_complex(input1, input2, output);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::multiply_complex(input1, input2, output);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::multiply_complex(input1, input2, output);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Normalize vector on with his own size
 */
template<typename NumericT, unsigned int AlignmentV>
void normalize(viennacl::vector<NumericT, AlignmentV> & input)
{
  switch (viennacl::traits::handle(input).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::normalize(input);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::normalize(input);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::normalize(input);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Inplace_transpose matrix
 */
template<typename NumericT, unsigned int AlignmentV>
void transpose(viennacl::matrix<NumericT, viennacl::row_major, AlignmentV> & input)
{
  switch (viennacl::traits::handle(input).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::transpose(input);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::transpose(input);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::transpose(input);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Transpose matrix
 */
template<typename NumericT, unsigned int AlignmentV>
void transpose(viennacl::matrix<NumericT, viennacl::row_major, AlignmentV> const & input,
               viennacl::matrix<NumericT, viennacl::row_major, AlignmentV>       & output)
{
  switch (viennacl::traits::handle(input).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::transpose(input, output);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::transpose(input, output);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::transpose(input, output);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Create complex vector from real vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
 */
template<typename NumericT>
void real_to_complex(viennacl::vector_base<NumericT> const & in,
                     viennacl::vector_base<NumericT>       & out, vcl_size_t size)
{
  switch (viennacl::traits::handle(in).get_active_handle_id())
  {
    case viennacl::MAIN_MEMORY:
      viennacl::linalg::host_based::real_to_complex(in, out, size);
      break;
#ifdef VIENNACL_WITH_OPENCL
      case viennacl::OPENCL_MEMORY:
      viennacl::linalg::opencl::real_to_complex(in,out,size);
      break;
#endif

#ifdef VIENNACL_WITH_CUDA
      case viennacl::CUDA_MEMORY:
      viennacl::linalg::cuda::real_to_complex(in,out,size);
      break;
#endif

    case viennacl::MEMORY_NOT_INITIALIZED:
      throw memory_exception("not initialised!");
    default:
      throw memory_exception("not implemented");
  }
}

/**
 * @brief Create real vector from complex vector (even elements(2*k) = real part, odd elements(2*k+1) = imaginary part)
 */
template<typename NumericT>
void complex_to_real(viennacl::vector_base<NumericT> const & in,
                     viennacl::vector_base<NumericT>       & out, vcl_size_t size)
{
  switch (viennacl::traits::handle(in).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::complex_to_real(in, out, size);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::complex_to_real(in, out, size);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::complex_to_real(in, out, size);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

/**
 * @brief Reverse vector to oposite order and save it in input vector
 */
template<typename NumericT>
void reverse(viennacl::vector_base<NumericT> & in)
{
  switch (viennacl::traits::handle(in).get_active_handle_id())
  {
  case viennacl::MAIN_MEMORY:
    viennacl::linalg::host_based::reverse(in);
    break;
#ifdef VIENNACL_WITH_OPENCL
  case viennacl::OPENCL_MEMORY:
    viennacl::linalg::opencl::reverse(in);
    break;
#endif

#ifdef VIENNACL_WITH_CUDA
  case viennacl::CUDA_MEMORY:
    viennacl::linalg::cuda::reverse(in);
    break;
#endif

  case viennacl::MEMORY_NOT_INITIALIZED:
    throw memory_exception("not initialised!");
  default:
    throw memory_exception("not implemented");
  }
}

}
}

#endif /* FFT_OPERATIONS_HPP_ */
