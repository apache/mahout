#ifndef VIENNACL_FFT_HPP
#define VIENNACL_FFT_HPP

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

/** @file viennacl/fft.hpp
 @brief All routines related to the Fast Fourier Transform. Experimental.
 */

#include <viennacl/vector.hpp>
#include <viennacl/matrix.hpp>

#include "viennacl/linalg/fft_operations.hpp"
#include "viennacl/traits/handle.hpp"

#include <cmath>

#include <stdexcept>
/// @cond
namespace viennacl
{
namespace detail
{
namespace fft
{
  inline bool is_radix2(vcl_size_t data_size)
  {
    return !((data_size > 2) && (data_size & (data_size - 1)));
  }
} //namespace fft
} //namespace detail

/**
 * @brief Generic inplace version of 1-D Fourier transformation.
 *
 * @param input       Input vector, result will be stored here.
 * @param batch_num   Number of items in batch
 * @param sign        Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void inplace_fft(viennacl::vector<NumericT, AlignmentV>& input, vcl_size_t batch_num = 1,
                 NumericT sign = -1.0)
{
  vcl_size_t size = (input.size() >> 1) / batch_num;

  if (!viennacl::detail::fft::is_radix2(size))
  {
    viennacl::vector<NumericT, AlignmentV> output(input.size());
    viennacl::linalg::direct(input, output, size, size, batch_num, sign);
    viennacl::copy(output, input);
  }
  else
    viennacl::linalg::radix2(input, size, size, batch_num, sign);
}

/**
 * @brief Generic version of 1-D Fourier transformation.
 *
 * @param input      Input vector.
 * @param output     Output vector.
 * @param batch_num  Number of items in batch.
 * @param sign       Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void fft(viennacl::vector<NumericT, AlignmentV>& input,
         viennacl::vector<NumericT, AlignmentV>& output, vcl_size_t batch_num = 1, NumericT sign = -1.0)
{
  vcl_size_t size = (input.size() >> 1) / batch_num;
  if (viennacl::detail::fft::is_radix2(size))
  {
    viennacl::copy(input, output);
    viennacl::linalg::radix2(output, size, size, batch_num, sign);
  }
  else
    viennacl::linalg::direct(input, output, size, size, batch_num, sign);
}

/**
 * @brief Generic inplace version of 2-D Fourier transformation.
 *
 * @param input       Input matrix, result will be stored here.
 * @param sign        Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void inplace_fft(viennacl::matrix<NumericT, viennacl::row_major, AlignmentV>& input,
                 NumericT sign = -1.0)
{
  vcl_size_t rows_num = input.size1();
  vcl_size_t cols_num = input.size2() >> 1;

  vcl_size_t cols_int = input.internal_size2() >> 1;

  // batch with rows
  if (viennacl::detail::fft::is_radix2(cols_num))
    viennacl::linalg::radix2(input, cols_num, cols_int, rows_num, sign,
                             viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR);
  else
  {
    viennacl::matrix<NumericT, viennacl::row_major, AlignmentV> output(input.size1(),
                                                                       input.size2());

    viennacl::linalg::direct(input, output, cols_num, cols_int, rows_num, sign,
                             viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR);

    input = output;
  }

  // batch with cols
  if (viennacl::detail::fft::is_radix2(rows_num))
    viennacl::linalg::radix2(input, rows_num, cols_int, cols_num, sign,
                             viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR);
  else
  {
    viennacl::matrix<NumericT, viennacl::row_major, AlignmentV> output(input.size1(),
                                                                       input.size2());

    viennacl::linalg::direct(input, output, rows_num, cols_int, cols_num, sign,
                             viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR);

    input = output;
  }

}

/**
 * @brief Generic version of 2-D Fourier transformation.
 *
 * @param input      Input vector.
 * @param output     Output vector.
 * @param sign       Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void fft(viennacl::matrix<NumericT, viennacl::row_major, AlignmentV>& input, //TODO
         viennacl::matrix<NumericT, viennacl::row_major, AlignmentV>& output, NumericT sign = -1.0)
{

  vcl_size_t rows_num = input.size1();
  vcl_size_t cols_num = input.size2() >> 1;
  vcl_size_t cols_int = input.internal_size2() >> 1;

  // batch with rows
  if (viennacl::detail::fft::is_radix2(cols_num))
  {
    output = input;
    viennacl::linalg::radix2(output, cols_num, cols_int, rows_num, sign,
                             viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR);
  }
  else
    viennacl::linalg::direct(input, output, cols_num, cols_int, rows_num, sign,
                             viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::ROW_MAJOR);

  // batch with cols
  if (viennacl::detail::fft::is_radix2(rows_num))
  {
    //std::cout<<"output"<<output<<std::endl;

    viennacl::linalg::radix2(output, rows_num, cols_int, cols_num, sign,
                             viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR);
  }
  else
  {
    viennacl::matrix<NumericT, viennacl::row_major, AlignmentV> tmp(output.size1(),
                                                                    output.size2());
    tmp = output;
    //std::cout<<"tmp"<<tmp<<std::endl;
    viennacl::linalg::direct(tmp, output, rows_num, cols_int, cols_num, sign,
                             viennacl::linalg::host_based::detail::fft::FFT_DATA_ORDER::COL_MAJOR);
  }
}

/**
 * @brief Generic inplace version of inverse 1-D Fourier transformation.
 *
 * Shorthand function for fft(sign = 1.0)
 *
 * @param input      Input vector.
 * @param batch_num  Number of items in batch.
 * @param sign       Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void inplace_ifft(viennacl::vector<NumericT, AlignmentV>& input, vcl_size_t batch_num = 1)
{
  viennacl::inplace_fft(input, batch_num, NumericT(1.0));
  viennacl::linalg::normalize(input);
}

/**
 * @brief Generic version of inverse 1-D Fourier transformation.
 *
 * Shorthand function for fft(sign = 1.0)
 *
 * @param input      Input vector.
 * @param output     Output vector.
 * @param batch_num  Number of items in batch.
 * @param sign       Sign of exponent, default is -1.0
 */
template<class NumericT, unsigned int AlignmentV>
void ifft(viennacl::vector<NumericT, AlignmentV>& input,
          viennacl::vector<NumericT, AlignmentV>& output, vcl_size_t batch_num = 1)
{
  viennacl::fft(input, output, batch_num, NumericT(1.0));
  viennacl::linalg::normalize(output);
}

namespace linalg
{
  /**
   * @brief 1-D convolution of two vectors.
   *
   * This function does not make any changes to input vectors
   *
   * @param input1     Input vector #1.
   * @param input2     Input vector #2.
   * @param output     Output vector.
   */
  template<class NumericT, unsigned int AlignmentV>
  void convolve(viennacl::vector<NumericT, AlignmentV>& input1,
                viennacl::vector<NumericT, AlignmentV>& input2,
                viennacl::vector<NumericT, AlignmentV>& output)
  {
    assert(input1.size() == input2.size());
    assert(input1.size() == output.size());
    //temporal arrays
    viennacl::vector<NumericT, AlignmentV> tmp1(input1.size());
    viennacl::vector<NumericT, AlignmentV> tmp2(input2.size());
    viennacl::vector<NumericT, AlignmentV> tmp3(output.size());

    // align input arrays to equal size
    // FFT of input data
    viennacl::fft(input1, tmp1);
    viennacl::fft(input2, tmp2);

    // multiplication of input data
    viennacl::linalg::multiply_complex(tmp1, tmp2, tmp3);
    // inverse FFT of input data
    viennacl::ifft(tmp3, output);
  }

  /**
   * @brief 1-D convolution of two vectors.
   *
   * This function can make changes to input vectors to avoid additional memory allocations.
   *
   * @param input1     Input vector #1.
   * @param input2     Input vector #2.
   * @param output     Output vector.
   */
  template<class NumericT, unsigned int AlignmentV>
  void convolve_i(viennacl::vector<NumericT, AlignmentV>& input1,
                  viennacl::vector<NumericT, AlignmentV>& input2,
                  viennacl::vector<NumericT, AlignmentV>& output)
  {
    assert(input1.size() == input2.size());
    assert(input1.size() == output.size());

    viennacl::inplace_fft(input1);
    viennacl::inplace_fft(input2);

    viennacl::linalg::multiply_complex(input1, input2, output);

    viennacl::inplace_ifft(output);
  }
}      //namespace linalg
}      //namespace viennacl

/// @endcond
#endif
