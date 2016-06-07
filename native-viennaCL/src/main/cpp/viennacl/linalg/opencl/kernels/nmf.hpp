#ifndef VIENNACL_LINALG_OPENCL_KERNELS_NMF_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_NMF_HPP

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

/** @file viennacl/linalg/opencl/kernels/nmf.hpp
 *  @brief OpenCL kernel file for nonnegative matrix factorization */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{

template<typename StringT>
void generate_nmf_el_wise_mul_div(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void el_wise_mul_div( \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * matrix1, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * matrix2, \n");
  source.append("  __global const "); source.append(numeric_string); source.append(" * matrix3, \n");
  source.append("  unsigned int size) \n");
  source.append("{ \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size; i += get_global_size(0)) \n");
  source.append("  { \n");
  source.append("    "); source.append(numeric_string); source.append(" val = matrix1[i] * matrix2[i]; \n");
  source.append("    "); source.append(numeric_string); source.append(" divisor = matrix3[i]; \n");
  source.append("    matrix1[i] = (divisor > ("); source.append(numeric_string); source.append(")0.00001) ? (val / divisor) : ("); source.append(numeric_string); source.append(")0; \n");
  source.append("  } \n");
  source.append("} \n");
}

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for nonnegative matrix factorization of a dense matrices. */
template<typename NumericT>
struct nmf
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_nmf";
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

      // only generate for floating points (forces error for integers)
      if (numeric_string == "float" || numeric_string == "double")
      {
        generate_nmf_el_wise_mul_div(source, numeric_string);
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

