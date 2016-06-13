#ifndef VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_ELEMENT_HPP
#define VIENNACL_LINALG_OPENCL_KERNELS_VECTOR_ELEMENT_HPP

#include "viennacl/tools/tools.hpp"
#include "viennacl/ocl/kernel.hpp"
#include "viennacl/ocl/platform.hpp"
#include "viennacl/ocl/utils.hpp"

/** @file viennacl/linalg/opencl/kernels/vector_element.hpp
 *  @brief OpenCL kernel file for element-wise vector operations */
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
template <typename StringT>
void generate_vector_unary_element_ops(StringT & source, std::string const & numeric_string,
                                       std::string const & funcname, std::string const & op, std::string const & op_name)
{
  source.append("__kernel void "); source.append(funcname); source.append("_"); source.append(op_name); source.append("(\n");
  source.append("    __global "); source.append(numeric_string); source.append(" * vec1, \n");
  source.append("    uint4 size1, \n");
  source.append("    __global "); source.append(numeric_string); source.append(" * vec2, \n");
  source.append("    uint4 size2) { \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size1.z; i += get_global_size(0)) \n");
  if (numeric_string[0] == 'u' && funcname == "abs") // abs() on unsigned does not work on MacOS X 10.6.8, so we use the identity:
  {
    source.append("    vec1[i*size1.y+size1.x] "); source.append(op); source.append(" vec2[i*size2.y+size2.x]; \n");
  }
  else
  {
    source.append("    vec1[i*size1.y+size1.x] "); source.append(op); source.append(" "); source.append(funcname); source.append("(vec2[i*size2.y+size2.x]); \n");
  }
  source.append("} \n");
}

template <typename StringT>
void generate_vector_unary_element_ops(StringT & source, std::string const & numeric_string, std::string const & funcname)
{
  generate_vector_unary_element_ops(source, numeric_string, funcname, "=", "assign");
  //generate_vector_unary_element_ops(source, numeric_string, funcname, "+=", "plus");
  //generate_vector_unary_element_ops(source, numeric_string, funcname, "-=", "minus");
}

template <typename StringT>
void generate_vector_binary_element_ops(StringT & source, std::string const & numeric_string, int op_type) //op_type: {0: product, 1: division, 2: power}
{
  std::string kernel_name_suffix;
  if (op_type == 0)
    kernel_name_suffix = "prod";
  else if (op_type == 1)
    kernel_name_suffix = "div";
  else
    kernel_name_suffix = "pow";

  // generic kernel for the vector operation v1 = alpha * v2 + beta * v3, where v1, v2, v3 are not necessarily distinct vectors
  source.append("__kernel void element_" + kernel_name_suffix + "(\n");
  source.append("    __global "); source.append(numeric_string); source.append(" * vec1, \n");
  source.append("    unsigned int start1, \n");
  source.append("    unsigned int inc1, \n");
  source.append("    unsigned int size1, \n");

  source.append("    __global const "); source.append(numeric_string); source.append(" * vec2, \n");
  source.append("    unsigned int start2, \n");
  source.append("    unsigned int inc2, \n");

  source.append("    __global const "); source.append(numeric_string); source.append(" * vec3, \n");
  source.append("   unsigned int start3, \n");
  source.append("   unsigned int inc3, \n");

  source.append("   unsigned int op_type) \n");
  source.append("{ \n");
  source.append("  for (unsigned int i = get_global_id(0); i < size1; i += get_global_size(0)) \n");
  if (op_type == 0)
    source.append("    vec1[i*inc1+start1] = vec2[i*inc2+start2] * vec3[i*inc3+start3]; \n");
  else if (op_type == 1)
    source.append("    vec1[i*inc1+start1] = vec2[i*inc2+start2] / vec3[i*inc3+start3]; \n");
  else if (op_type == 2)
    source.append("    vec1[i*inc1+start1] = pow(vec2[i*inc2+start2], vec3[i*inc3+start3]); \n");

  source.append("} \n");
}

//////////////////////////// Part 2: Main kernel class ////////////////////////////////////

// main kernel class
/** @brief Main kernel class for generating OpenCL kernels for elementwise operations other than addition and subtraction on/with viennacl::vector<>. */
template<typename NumericT>
struct vector_element
{
  static std::string program_name()
  {
    return viennacl::ocl::type_to_string<NumericT>::apply() + "_vector_element";
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

      viennacl::ocl::append_double_precision_pragma<NumericT>(ctx, source);

      // unary operations
      if (numeric_string == "float" || numeric_string == "double")
      {
        generate_vector_unary_element_ops(source, numeric_string, "acos");
        generate_vector_unary_element_ops(source, numeric_string, "asin");
        generate_vector_unary_element_ops(source, numeric_string, "atan");
        generate_vector_unary_element_ops(source, numeric_string, "ceil");
        generate_vector_unary_element_ops(source, numeric_string, "cos");
        generate_vector_unary_element_ops(source, numeric_string, "cosh");
        generate_vector_unary_element_ops(source, numeric_string, "exp");
        generate_vector_unary_element_ops(source, numeric_string, "fabs");
        generate_vector_unary_element_ops(source, numeric_string, "floor");
        generate_vector_unary_element_ops(source, numeric_string, "log");
        generate_vector_unary_element_ops(source, numeric_string, "log10");
        generate_vector_unary_element_ops(source, numeric_string, "sin");
        generate_vector_unary_element_ops(source, numeric_string, "sinh");
        generate_vector_unary_element_ops(source, numeric_string, "sqrt");
        generate_vector_unary_element_ops(source, numeric_string, "tan");
        generate_vector_unary_element_ops(source, numeric_string, "tanh");
      }
      else
      {
        generate_vector_unary_element_ops(source, numeric_string, "abs");
      }

      // binary operations
      generate_vector_binary_element_ops(source, numeric_string, 0);
      generate_vector_binary_element_ops(source, numeric_string, 1);
      if (numeric_string == "float" || numeric_string == "double")
        generate_vector_binary_element_ops(source, numeric_string, 2);

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

