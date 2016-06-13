#ifndef VIENNACL_LINALG_OPENCL_KERNELS_SCALAR_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_SCALAR_HPP

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

/** @file viennacl/linalg/opencl/kernels/scalar.hpp
 *  @brief OpenCL kernel file for scalar operations */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{

//////////////////////////// Part 1: Kernel generation routines ////////////////////////////////////

/** @brief Enumeration for the scalar type in avbv-like operations */
enum asbs_scalar_type
{
  VIENNACL_ASBS_NONE = 0, // scalar does not exist/contribute
  VIENNACL_ASBS_CPU,
  VIENNACL_ASBS_GPU
};

/** @brief Configuration struct for generating OpenCL kernels for linear combinations of viennacl::scalar<> objects */
struct asbs_config
{
  asbs_config() : with_stride_and_range(true), a(VIENNACL_ASBS_CPU), b(VIENNACL_ASBS_NONE) {}

  bool with_stride_and_range;
  std::string      assign_op;
  asbs_scalar_type a;
  asbs_scalar_type b;
};

// just returns the assignment string
template<typename StringT>
void generate_asbs_impl3(StringT & source, char sign_a, char sign_b, asbs_config const & cfg, bool mult_alpha, bool mult_beta)
{
  source.append("      *s1 "); source.append(cfg.assign_op); source.append(1, sign_a); source.append(" *s2 ");
  if (mult_alpha)
    source.append("* alpha ");
  else
    source.append("/ alpha ");
  if (cfg.b != VIENNACL_ASBS_NONE)
  {
    source.append(1, sign_b); source.append(" *s3 ");
    if (mult_beta)
      source.append("* beta");
    else
      source.append("/ beta");
  }
  source.append("; \n");
}

template<typename StringT>
void generate_asbs_impl2(StringT & source, char sign_a, char sign_b, asbs_config const & cfg)
{
  source.append("    if (options2 & (1 << 1)) { \n");
  if (cfg.b != VIENNACL_ASBS_NONE)
  {
    source.append("     if (options3 & (1 << 1)) \n");
    generate_asbs_impl3(source, sign_a, sign_b, cfg, false, false);
    source.append("     else \n");
    generate_asbs_impl3(source, sign_a, sign_b, cfg, false, true);
  }
  else
    generate_asbs_impl3(source, sign_a, sign_b, cfg, false, true);
  source.append("    } else { \n");
  if (cfg.b != VIENNACL_ASBS_NONE)
  {
    source.append("     if (options3 & (1 << 1)) \n");
    generate_asbs_impl3(source, sign_a, sign_b, cfg, true, false);
    source.append("     else \n");
    generate_asbs_impl3(source, sign_a, sign_b, cfg, true, true);
  }
  else
    generate_asbs_impl3(source, sign_a, sign_b, cfg, true, true);
  source.append("    } \n");

}

template<typename StringT>
void generate_asbs_impl(StringT & source, std::string const & numeric_string, asbs_config const & cfg)
{
  source.append("__kernel void as");
  if (cfg.b != VIENNACL_ASBS_NONE)
    source.append("bs");
  if (cfg.assign_op != "=")
    source.append("_s");

  if (cfg.a == VIENNACL_ASBS_CPU)
    source.append("_cpu");
  else if (cfg.a == VIENNACL_ASBS_GPU)
    source.append("_gpu");

  if (cfg.b == VIENNACL_ASBS_CPU)
    source.append("_cpu");
  else if (cfg.b == VIENNACL_ASBS_GPU)
    source.append("_gpu");
  source.append("( \n");
  source.append("  __global "); source.append(numeric_string); source.append(" * s1, \n");
  source.append(" \n");
  if (cfg.a == VIENNACL_ASBS_CPU)
  {
    source.append("  "); source.append(numeric_string); source.append(" fac2, \n");
  }
  else if (cfg.a == VIENNACL_ASBS_GPU)
  {
    source.append("  __global "); source.append(numeric_string); source.append(" * fac2, \n");
  }
  source.append("  unsigned int options2, \n");  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse
  source.append("  __global const "); source.append(numeric_string); source.append(" * s2");

  if (cfg.b != VIENNACL_ASBS_NONE)
  {
    source.append(", \n\n");
    if (cfg.b == VIENNACL_ASBS_CPU)
    {
      source.append("  "); source.append(numeric_string); source.append(" fac3, \n");
    }
    else if (cfg.b == VIENNACL_ASBS_GPU)
    {
      source.append("  __global "); source.append(numeric_string); source.append(" * fac3, \n");
    }
    source.append("  unsigned int options3, \n");  // 0: no action, 1: flip sign, 2: take inverse, 3: flip sign and take inverse
    source.append("  __global const "); source.append(numeric_string); source.append(" * s3");
  }
  source.append(") \n{ \n");

  if (cfg.a == VIENNACL_ASBS_CPU)
  {
    source.append("  "); source.append(numeric_string); source.append(" alpha = fac2; \n");
  }
  else if (cfg.a == VIENNACL_ASBS_GPU)
  {
    source.append("  "); source.append(numeric_string); source.append(" alpha = fac2[0]; \n");
  }
  source.append(" \n");

  if (cfg.b == VIENNACL_ASBS_CPU)
  {
    source.append("  "); source.append(numeric_string); source.append(" beta = fac3; \n");
  }
  else if (cfg.b == VIENNACL_ASBS_GPU)
  {
    source.append("  "); source.append(numeric_string); source.append(" beta = fac3[0]; \n");
  }

  source.append("  if (options2 & (1 << 0)) { \n");
  if (cfg.b != VIENNACL_ASBS_NONE)
  {
    source.append("   if (options3 & (1 << 0)) { \n");
    generate_asbs_impl2(source, '-', '-', cfg);
    source.append("   } else { \n");
    generate_asbs_impl2(source, '-', '+', cfg);
    source.append("   } \n");
  }
  else
    generate_asbs_impl2(source, '-', '+', cfg);
  source.append("  } else { \n");
  if (cfg.b != VIENNACL_ASBS_NONE)
  {
    source.append("   if (options3 & (1 << 0)) { \n");
    generate_asbs_impl2(source, '+', '-', cfg);
    source.append("   } else { \n");
    generate_asbs_impl2(source, '+', '+', cfg);
    source.append("   } \n");
  }
  else
    generate_asbs_impl2(source, '+', '+', cfg);

  source.append("  } \n");
  source.append("} \n");
}

template<typename StringT>
void generate_asbs(StringT & source, std::string const & numeric_string)
{
  asbs_config cfg;
  cfg.assign_op = "=";
  cfg.with_stride_and_range = true;

  // as
  cfg.b = VIENNACL_ASBS_NONE; cfg.a = VIENNACL_ASBS_CPU; generate_asbs_impl(source, numeric_string, cfg);
  cfg.b = VIENNACL_ASBS_NONE; cfg.a = VIENNACL_ASBS_GPU; generate_asbs_impl(source, numeric_string, cfg);

  // asbs
  cfg.a = VIENNACL_ASBS_CPU; cfg.b = VIENNACL_ASBS_CPU; generate_asbs_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_ASBS_CPU; cfg.b = VIENNACL_ASBS_GPU; generate_asbs_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_ASBS_GPU; cfg.b = VIENNACL_ASBS_CPU; generate_asbs_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_ASBS_GPU; cfg.b = VIENNACL_ASBS_GPU; generate_asbs_impl(source, numeric_string, cfg);

  // asbs
  cfg.assign_op = "+=";

  cfg.a = VIENNACL_ASBS_CPU; cfg.b = VIENNACL_ASBS_CPU; generate_asbs_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_ASBS_CPU; cfg.b = VIENNACL_ASBS_GPU; generate_asbs_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_ASBS_GPU; cfg.b = VIENNACL_ASBS_CPU; generate_asbs_impl(source, numeric_string, cfg);
  cfg.a = VIENNACL_ASBS_GPU; cfg.b = VIENNACL_ASBS_GPU; generate_asbs_impl(source, numeric_string, cfg);
}

template<typename StringT>
void generate_scalar_swap(StringT & source, std::string const & numeric_string)
{
  source.append("__kernel void swap( \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * s1, \n");
  source.append("          __global "); source.append(numeric_string); source.append(" * s2) \n");
  source.append("{ \n");
  source.append("  "); source.append(numeric_string); source.append(" tmp = *s2; \n");
  source.append("  *s2 = *s1; \n");
  source.append("  *s1 = tmp; \n");
  source.append("} \n");
}

//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for operations involving viennacl::scalar<>, but not viennacl::vector<> or viennacl::matrix<>. */
template<typename NumericT>
struct scalar
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_scalar";
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

      // fully parametrized kernels:
      generate_asbs(source, numeric_string);
      generate_scalar_swap(source, numeric_string);


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

