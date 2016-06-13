#ifndef VIENNACL_LINALG_OPENCL_KERNELS_MATRIX_ELEMENT_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_MATRIX_ELEMENT_HPP

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"
#include "viennacl/linalg/opencl/kernels/matrix.hpp"

/** @file viennacl/linalg/opencl/kernels/matrix_element.hpp
 *  @brief OpenCL kernel file for element-wise matrix operations */
namespace viennacl
{
namespace linalg
{
namespace opencl
{
namespace kernels
{

//////////////////////////// Part 1: Kernel generation routines ////////////////////////////////////


//generate code for C = op1(A) * op2(B), where A, B, C can have different storage layouts and opX(D) = D or trans(D)
template <typename StringType>
void generate_matrix_unary_element_ops(StringType & source, std::string const & numeric_string,
                                       std::string const & funcname, std::string const & op, std::string const & op_name, bool is_row_major)
{
  source.append("__kernel void "); source.append(funcname); source.append("_"); source.append(op_name); source.append("(\n");
  source.append("          __global "); source.append(numeric_string); source.append(" * A, \n");
  source.append("          unsigned int A_start1, unsigned int A_start2, \n");
  source.append("          unsigned int A_inc1,   unsigned int A_inc2, \n");
  source.append("          unsigned int A_size1,  unsigned int A_size2, \n");
  source.append("          unsigned int A_internal_size1,  unsigned int A_internal_size2, \n");

  source.append("          __global const "); source.append(numeric_string); source.append(" * B, \n");
  source.append("          unsigned int B_start1, unsigned int B_start2, \n");
  source.append("          unsigned int B_inc1,   unsigned int B_inc2, \n");
  source.append("          unsigned int B_internal_size1,  unsigned int B_internal_size2) { \n");

  if (is_row_major)
  {
    source.append("  unsigned int row_gid = get_global_id(0) / get_local_size(0); \n");
    source.append("  unsigned int col_gid = get_global_id(0) % get_local_size(0); \n");

    source.append("  for (unsigned int row = row_gid; row < A_size1; row += get_num_groups(0)) \n");
    source.append("    for (unsigned int col = col_gid; col < A_size2; col += get_local_size(0)) \n");
    source.append("      A[(row * A_inc1 + A_start1) * A_internal_size2 + col * A_inc2 + A_start2] \n");
    source.append("        "); source.append(op); source.append(" "); source.append(funcname); source.append("(B[(row * B_inc1 + B_start1) * B_internal_size2 + col * B_inc2 + B_start2]); \n");
  }
  else
  {
    source.append("  unsigned int row_gid = get_global_id(0) % get_local_size(0); \n");
    source.append("  unsigned int col_gid = get_global_id(0) / get_local_size(0); \n");

    source.append("  for (unsigned int col = col_gid; col < A_size2; col += get_num_groups(0)) \n");
    source.append("    for (unsigned int row = row_gid; row < A_size1; row += get_local_size(0)) \n");
    source.append("      A[(row * A_inc1 + A_start1) + (col * A_inc2 + A_start2) * A_internal_size1] \n");
    source.append("        "); source.append(op); source.append(" "); source.append(funcname); source.append("(B[(row * B_inc1 + B_start1) + (col * B_inc2 + B_start2) * B_internal_size1]); \n");
  }
  source.append("} \n");
}

template <typename StringType>
void generate_matrix_unary_element_ops(StringType & source, std::string const & numeric_string, std::string const & funcname, bool is_row_major)
{
  generate_matrix_unary_element_ops(source, numeric_string, funcname, "=", "assign", is_row_major);
  //generate_matrix_unary_element_ops(source, numeric_string, funcname, "+=", "plus", is_row_major);
  //generate_matrix_unary_element_ops(source, numeric_string, funcname, "-=", "minus", is_row_major);
}

//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for elementwise-operations such as element_sin() on/with dense matrix objects of type viennacl::matrix<>. */
template <typename NumericT, typename F>
struct matrix_element
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_matrix_element_" + detail::type_to_string(F());
  }

  static void init(viennacl::ocl::context & ctx)
  {
    viennacl::ocl::DOUBLE_PRECISION_CHECKER<NumericT>::apply(ctx);
    std::string numeric_string = viennacl::ocl::type_to_string<NumericT>::apply();

    static std::map<cl_context, bool> init_done;
    if (!init_done[ctx.handle().get()])
    {
      std::string source;
      source.reserve(8192);
      bool is_row_major = viennacl::is_row_major<F>::value;

      viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

      // unary operations
      if (numeric_string == "float" || numeric_string == "double")
      {
        generate_matrix_unary_element_ops(source, numeric_string, "acos",  is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "asin",  is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "atan",  is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "ceil",  is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "cos",   is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "cosh",  is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "exp",   is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "fabs",  is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "floor", is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "log",   is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "log10", is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "sin",   is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "sinh",  is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "sqrt",  is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "tan",   is_row_major);
        generate_matrix_unary_element_ops(source, numeric_string, "tanh",  is_row_major);
      }
      else
      {
        generate_matrix_unary_element_ops(source, numeric_string, "abs", is_row_major);
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

