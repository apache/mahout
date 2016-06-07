#ifndef VIENNACL_LINALG_OPENCL_KERNELS_FFT_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_FFT_HPP

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

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

/** @file viennacl/linalg/opencl/kernels/fft.hpp
 *  @brief OpenCL kernel file for FFT operations */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{

//////////////////////////// Part 1: Kernel generation routines ////////////////////////////////////


// Postprocessing phase of Bluestein algorithm
template<typename StringT>
void generate_fft_bluestein_post(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bluestein_post(__global "); source.append(numeric_string); source.append("2 *Z, \n");
  source.append("                             __global "); source.append(numeric_string); source.append("2 *out, \n");
  source.append("                             unsigned int size) \n");
  source.append("{ \n");
  source.append("  unsigned int glb_id = get_global_id(0); \n");
  source.append("  unsigned int glb_sz = get_global_size(0); \n");

  source.append("  unsigned int double_size = size << 1; \n");
  source.append("  "); source.append(numeric_string); source.append(" sn_a, cs_a; \n");
  source.append("  const "); source.append(numeric_string); source.append(" NUM_PI = 3.14159265358979323846; \n");

  source.append("  for (unsigned int i = glb_id; i < size; i += glb_sz) { \n");
  source.append("    unsigned int rm = i * i % (double_size); \n");
  source.append("    "); source.append(numeric_string); source.append(" angle = ("); source.append(numeric_string); source.append(")rm / size * (-NUM_PI); \n");

  source.append("    sn_a = sincos(angle, &cs_a); \n");

  source.append("    "); source.append(numeric_string); source.append("2 b_i = ("); source.append(numeric_string); source.append("2)(cs_a, sn_a); \n");
  source.append("    out[i] = ("); source.append(numeric_string); source.append("2)(Z[i].x * b_i.x - Z[i].y * b_i.y, Z[i].x * b_i.y + Z[i].y * b_i.x); \n");
  source.append("  } \n");
  source.append("} \n");
}

// Preprocessing phase of Bluestein algorithm
template<typename StringT>
void generate_fft_bluestein_pre(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void bluestein_pre(__global "); source.append(numeric_string); source.append("2 *input, \n");
  source.append("  __global "); source.append(numeric_string); source.append("2 *A, \n");
  source.append("  __global "); source.append(numeric_string); source.append("2 *B, \n");
  source.append("  unsigned int size, \n");
  source.append("  unsigned int ext_size \n");
  source.append("  ) { \n");
  source.append("  unsigned int glb_id = get_global_id(0); \n");
  source.append("  unsigned int glb_sz = get_global_size(0); \n");

  source.append("  unsigned int double_size = size << 1; \n");

  source.append("  "); source.append(numeric_string); source.append(" sn_a, cs_a; \n");
  source.append("  const "); source.append(numeric_string); source.append(" NUM_PI = 3.14159265358979323846; \n");

  source.append("  for (unsigned int i = glb_id; i < size; i += glb_sz) { \n");
  source.append("    unsigned int rm = i * i % (double_size); \n");
  source.append("    "); source.append(numeric_string); source.append(" angle = ("); source.append(numeric_string); source.append(")rm / size * NUM_PI; \n");

  source.append("    sn_a = sincos(-angle, &cs_a); \n");

  source.append("    "); source.append(numeric_string); source.append("2 a_i = ("); source.append(numeric_string); source.append("2)(cs_a, sn_a); \n");
  source.append("    "); source.append(numeric_string); source.append("2 b_i = ("); source.append(numeric_string); source.append("2)(cs_a, -sn_a); \n");

  source.append("    A[i] = ("); source.append(numeric_string); source.append("2)(input[i].x * a_i.x - input[i].y * a_i.y, input[i].x * a_i.y + input[i].y * a_i.x); \n");
  source.append("    B[i] = b_i; \n");

          // very bad instruction, to be fixed
  source.append("    if (i) \n");
  source.append("      B[ext_size - i] = b_i; \n");
  source.append("  } \n");
  source.append("} \n");
}

/** @brief Extract real part of a complex number array */
template<typename StringT>
void generate_fft_complex_to_real(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void complex_to_real(__global "); source.append(numeric_string); source.append("2 *in, \n");
  source.append("  __global "); source.append(numeric_string); source.append("  *out, \n");
  source.append("  unsigned int size) { \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))  \n");
  source.append("    out[i] = in[i].x; \n");
  source.append("} \n");
}

/** @brief OpenCL kernel generation code for dividing a complex number by a real number */
template<typename StringT>
void generate_fft_div_vec_scalar(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void fft_div_vec_scalar(__global "); source.append(numeric_string); source.append("2 *input1, \n");
  source.append("  unsigned int size, \n");
  source.append("  "); source.append(numeric_string); source.append(" factor) { \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0))  \n");
  source.append("    input1[i] /= factor; \n");
  source.append("} \n");
}

/** @brief Elementwise product of two complex vectors */
template<typename StringT>
void generate_fft_mult_vec(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void fft_mult_vec(__global const "); source.append(numeric_string); source.append("2 *input1, \n");
  source.append("  __global const "); source.append(numeric_string); source.append("2 *input2, \n");
  source.append("  __global "); source.append(numeric_string); source.append("2 *output, \n");
  source.append("  unsigned int size) { \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append("2 in1 = input1[i]; \n");
  source.append("    "); source.append(numeric_string); source.append("2 in2 = input2[i]; \n");

  source.append("    output[i] = ("); source.append(numeric_string); source.append("2)(in1.x * in2.x - in1.y * in2.y, in1.x * in2.y + in1.y * in2.x); \n");
  source.append("  } \n");
  source.append("} \n");
}

/** @brief Embedds a real-valued vector into a complex one */
template<typename StringT>
void generate_fft_real_to_complex(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void real_to_complex(__global "); source.append(numeric_string); source.append(" *in, \n");
  source.append("  __global "); source.append(numeric_string); source.append("2 *out, \n");
  source.append("  unsigned int size) { \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append("2 val = 0; \n");
  source.append("    val.x = in[i]; \n");
  source.append("    out[i] = val; \n");
  source.append("  } \n");
  source.append("} \n");
}

/** @brief Reverses the entries in a vector */
template<typename StringT>
void generate_fft_reverse_inplace(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void reverse_inplace(__global "); source.append(numeric_string); source.append(" *vec, uint size) { \n");
  source.append("  for (uint i = get_global_id(0); i < (size >> 1); i+=get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" val1 = vec[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" val2 = vec[size - i - 1]; \n");

  source.append("    vec[i] = val2; \n");
  source.append("    vec[size - i - 1] = val1; \n");
  source.append("  } \n");
  source.append("} \n");
}

/** @brief Simplistic matrix transpose function */
template<typename StringT>
void generate_fft_transpose(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void transpose(__global "); source.append(numeric_string); source.append("2 *input, \n");
  source.append("  __global "); source.append(numeric_string); source.append("2 *output, \n");
  source.append("  unsigned int row_num, \n");
  source.append("  unsigned int col_num) { \n");
  source.append("  unsigned int size = row_num * col_num; \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i+= get_global_size(0)) { \n");
  source.append("    unsigned int row = i / col_num; \n");
  source.append("    unsigned int col = i - row*col_num; \n");

  source.append("    unsigned int new_pos = col * row_num + row; \n");

  source.append("    output[new_pos] = input[i]; \n");
  source.append("  } \n");
  source.append("} \n");
}

/** @brief Simplistic inplace matrix transpose function */
template<typename StringT>
void generate_fft_transpose_inplace(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void transpose_inplace(__global "); source.append(numeric_string); source.append("2* input, \n");
  source.append("  unsigned int row_num, \n");
  source.append("  unsigned int col_num) { \n");
  source.append("  unsigned int size = row_num * col_num; \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i+= get_global_size(0)) { \n");
  source.append("    unsigned int row = i / col_num; \n");
  source.append("    unsigned int col = i - row*col_num; \n");

  source.append("    unsigned int new_pos = col * row_num + row; \n");

  source.append("    if (i < new_pos) { \n");
  source.append("      "); source.append(numeric_string); source.append("2 val = input[i]; \n");
  source.append("      input[i] = input[new_pos]; \n");
  source.append("      input[new_pos] = val; \n");
  source.append("    } \n");
  source.append("  } \n");
  source.append("} \n");
}

/** @brief Computes the matrix vector product with a Vandermonde matrix */
template<typename StringT>
void generate_fft_vandermonde_prod(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void vandermonde_prod(__global "); source.append(numeric_string); source.append(" *vander, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" *vector, \n");
  source.append("  __global "); source.append(numeric_string); source.append(" *result, \n");
  source.append("  uint size) { \n");
  source.append("  for (uint i = get_global_id(0); i < size; i+= get_global_size(0)) { \n");
  source.append("    "); source.append(numeric_string); source.append(" mul = vander[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" pwr = 1; \n");
  source.append("    "); source.append(numeric_string); source.append(" val = 0; \n");

  source.append("    for (uint j = 0; j < size; j++) { \n");
  source.append("      val = val + pwr * vector[j]; \n");
  source.append("      pwr *= mul; \n");
  source.append("    } \n");

  source.append("    result[i] = val; \n");
  source.append("  } \n");
  source.append("} \n");
}

/** @brief Zero two complex vectors (to avoid kernel launch overhead) */
template<typename StringT>
void generate_fft_zero2(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void zero2(__global "); source.append(numeric_string); source.append("2 *input1, \n");
  source.append("  __global "); source.append(numeric_string); source.append("2 *input2, \n");
  source.append("  unsigned int size) { \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) { \n");
  source.append("    input1[i] = 0; \n");
  source.append("    input2[i] = 0; \n");
  source.append("  } \n");
  source.append("} \n");
}

//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for the fast Fourier transform. */
template<typename NumericT>
struct fft
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_fft";
  }

  static void init(viennacl::ocl::context & ctx)
  {
    static std::map<cl_context, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
      std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();

      std::string source;
      source.reserve(8192);

      viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

      // unary operations
      if (numeric_string == "float" || numeric_string == "double")
      {
        generate_fft_bluestein_post(source, numeric_string);
        generate_fft_bluestein_pre(source, numeric_string);
        generate_fft_complex_to_real(source, numeric_string);
        generate_fft_div_vec_scalar(source, numeric_string);
        generate_fft_mult_vec(source, numeric_string);
        generate_fft_real_to_complex(source, numeric_string);
        generate_fft_reverse_inplace(source, numeric_string);
        generate_fft_transpose(source, numeric_string);
        generate_fft_transpose_inplace(source, numeric_string);
        generate_fft_vandermonde_prod(source, numeric_string);
        generate_fft_zero2(source, numeric_string);
      }

      std::string prog_name = program_name();
      #ifdef VIENNACL_BUILD_INFO
      std::cout << "Creating program " << prog_name << std::endl;
      #endif
      ctx.add_program(source, prog_name);
      init_done[ctx.handle().get()] = true;
    } //if
  } //init
};

}  // namespace kernels
}  // namespace opencl
}  // namespace linalg
}  // namespace viennacl
#endif

