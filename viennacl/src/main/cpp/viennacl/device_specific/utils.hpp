#ifndef VIENNACL_DEVICE_SPECIFIC_UTILS_HPP
#define VIENNACL_DEVICE_SPECIFIC_UTILS_HPP

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


/** @file viennacl/device_specific/utils.hpp
    @brief Internal utils
*/

#include <sstream>

#include "viennacl/detail/matrix_def.hpp"
#include "viennacl/detail/vector_def.hpp"

#include "viennacl/device_specific/forwards.h"
#include "viennacl/ocl/forwards.h"

#include "viennacl/scheduler/forwards.h"

#include "viennacl/traits/size.hpp"
#include "viennacl/traits/handle.hpp"
#include "viennacl/traits/row_major.hpp"

#include "viennacl/tools/tools.hpp"

namespace viennacl
{
namespace device_specific
{
namespace utils
{

//CUDA Conversion
inline std::string opencl_source_to_cuda_source(std::string const & opencl_src)
{
  std::string res = opencl_src;

  viennacl::tools::find_and_replace(res,"__attribute__","//__attribute__");

  //Pointer
  viennacl::tools::find_and_replace(res, "__global float*", "float*");
  viennacl::tools::find_and_replace(res, "__local float*", "float*");

  viennacl::tools::find_and_replace(res, "__global double*", "double*");
  viennacl::tools::find_and_replace(res, "__local double*", "double*");

  //Qualifiers
  viennacl::tools::find_and_replace(res,"__global","__device__");
  viennacl::tools::find_and_replace(res,"__kernel","__global__");
  viennacl::tools::find_and_replace(res,"__constant","__constant__");
  viennacl::tools::find_and_replace(res,"__local","__shared__");

  //Indexing
  viennacl::tools::find_and_replace(res,"get_num_groups(0)","gridDim.x");
  viennacl::tools::find_and_replace(res,"get_num_groups(1)","gridDim.y");

  viennacl::tools::find_and_replace(res,"get_local_size(0)","blockDim.x");
  viennacl::tools::find_and_replace(res,"get_local_size(1)","blockDim.y");

  viennacl::tools::find_and_replace(res,"get_group_id(0)","blockIdx.x");
  viennacl::tools::find_and_replace(res,"get_group_id(1)","blockIdx.y");

  viennacl::tools::find_and_replace(res,"get_local_id(0)","threadIdx.x");
  viennacl::tools::find_and_replace(res,"get_local_id(1)","threadIdx.y");

  viennacl::tools::find_and_replace(res,"get_global_id(0)","(blockIdx.x*blockDim.x + threadIdx.x)");
  viennacl::tools::find_and_replace(res,"get_global_id(1)","(blockIdx.y*blockDim.y + threadIdx.y)");

  //Synchronization
  viennacl::tools::find_and_replace(res,"barrier(CLK_LOCAL_MEM_FENCE)","__syncthreads()");
  viennacl::tools::find_and_replace(res,"barrier(CLK_GLOBAL_MEM_FENCE)","__syncthreads()");


  return res;
}

static std::string numeric_type_to_string(scheduler::statement_node_numeric_type const & type){
  switch (type)
  {
  //case scheduler::CHAR_TYPE: return "char";
  //case scheduler::UCHAR_TYPE: return "unsigned char";
  //case scheduler::SHORT_TYPE: return "short";
  //case scheduler::USHORT_TYPE: return "unsigned short";
  case scheduler::INT_TYPE:  return "int";
  case scheduler::UINT_TYPE: return "unsigned int";
  case scheduler::LONG_TYPE:  return "long";
  case scheduler::ULONG_TYPE: return "unsigned long";
  case scheduler::FLOAT_TYPE : return "float";
  case scheduler::DOUBLE_TYPE : return "double";
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}


template<class Fun>
static typename Fun::result_type call_on_host_scalar(scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.type_family == scheduler::SCALAR_TYPE_FAMILY && bool("Must be called on a host scalar"));
  switch (element.numeric_type)
  {
  //case scheduler::CHAR_TYPE: return fun(element.host_char);
  //case scheduler::UCHAR_TYPE: return fun(element.host_uchar);
  //case scheduler::SHORT_TYPE: return fun(element.host_short);
  //case scheduler::USHORT_TYPE: return fun(element.host_ushort);
  case scheduler::INT_TYPE:  return fun(element.host_int);
  case scheduler::UINT_TYPE: return fun(element.host_uint);
  case scheduler::LONG_TYPE:  return fun(element.host_long);
  case scheduler::ULONG_TYPE: return fun(element.host_ulong);
  case scheduler::FLOAT_TYPE : return fun(element.host_float);
  case scheduler::DOUBLE_TYPE : return fun(element.host_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}

template<class Fun>
static typename Fun::result_type call_on_scalar(scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.type_family == scheduler::SCALAR_TYPE_FAMILY && bool("Must be called on a scalar"));
  switch (element.numeric_type)
  {
  //case scheduler::CHAR_TYPE: return fun(*element.scalar_char);
  //case scheduler::UCHAR_TYPE: return fun(*element.scalar_uchar);
  //case scheduler::SHORT_TYPE: return fun(*element.scalar_short);
  //case scheduler::USHORT_TYPE: return fun(*element.scalar_ushort);
  case scheduler::INT_TYPE:  return fun(*element.scalar_int);
  case scheduler::UINT_TYPE: return fun(*element.scalar_uint);
  case scheduler::LONG_TYPE:  return fun(*element.scalar_long);
  case scheduler::ULONG_TYPE: return fun(*element.scalar_ulong);
  case scheduler::FLOAT_TYPE : return fun(*element.scalar_float);
  case scheduler::DOUBLE_TYPE : return fun(*element.scalar_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}

template<class Fun>
static typename Fun::result_type call_on_vector(scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.type_family == scheduler::VECTOR_TYPE_FAMILY && bool("Must be called on a vector"));
  switch (element.numeric_type)
  {
  //case scheduler::CHAR_TYPE: return fun(*element.vector_char);
  //case scheduler::UCHAR_TYPE: return fun(*element.vector_uchar);
  //case scheduler::SHORT_TYPE: return fun(*element.vector_short);
  //case scheduler::USHORT_TYPE: return fun(*element.vector_ushort);
  case scheduler::INT_TYPE:  return fun(*element.vector_int);
  case scheduler::UINT_TYPE: return fun(*element.vector_uint);
  case scheduler::LONG_TYPE:  return fun(*element.vector_long);
  case scheduler::ULONG_TYPE: return fun(*element.vector_ulong);
  case scheduler::FLOAT_TYPE : return fun(*element.vector_float);
  case scheduler::DOUBLE_TYPE : return fun(*element.vector_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}

template<class Fun>
static typename Fun::result_type call_on_implicit_vector(scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.type_family == scheduler::VECTOR_TYPE_FAMILY   && bool("Must be called on a implicit_vector"));
  assert(element.subtype     == scheduler::IMPLICIT_VECTOR_TYPE && bool("Must be called on a implicit_vector"));
  switch (element.numeric_type)
  {
  //case scheduler::CHAR_TYPE: return fun(*element.implicit_vector_char);
  //case scheduler::UCHAR_TYPE: return fun(*element.implicit_vector_uchar);
  //case scheduler::SHORT_TYPE: return fun(*element.implicit_vector_short);
  //case scheduler::USHORT_TYPE: return fun(*element.implicit_vector_ushort);
  case scheduler::INT_TYPE:  return fun(*element.implicit_vector_int);
  case scheduler::UINT_TYPE: return fun(*element.implicit_vector_uint);
  case scheduler::LONG_TYPE:  return fun(*element.implicit_vector_long);
  case scheduler::ULONG_TYPE: return fun(*element.implicit_vector_ulong);
  case scheduler::FLOAT_TYPE : return fun(*element.implicit_vector_float);
  case scheduler::DOUBLE_TYPE : return fun(*element.implicit_vector_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}

template<class Fun>
static typename Fun::result_type call_on_matrix(scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.type_family == scheduler::MATRIX_TYPE_FAMILY && bool("Must be called on a matrix"));
  switch (element.numeric_type)
  {
  //case scheduler::CHAR_TYPE: return fun(*element.matrix_char);
  //case scheduler::UCHAR_TYPE: return fun(*element.matrix_uchar);
  //case scheduler::SHORT_TYPE: return fun(*element.matrix_short);
  //case scheduler::USHORT_TYPE: return fun(*element.matrix_ushort);
  case scheduler::INT_TYPE:  return fun(*element.matrix_int);
  case scheduler::UINT_TYPE: return fun(*element.matrix_uint);
  case scheduler::LONG_TYPE:  return fun(*element.matrix_long);
  case scheduler::ULONG_TYPE: return fun(*element.matrix_ulong);
  case scheduler::FLOAT_TYPE : return fun(*element.matrix_float);
  case scheduler::DOUBLE_TYPE : return fun(*element.matrix_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}


template<class Fun>
static typename Fun::result_type call_on_implicit_matrix(scheduler::lhs_rhs_element element, Fun const & fun){
  assert(element.subtype     == scheduler::IMPLICIT_MATRIX_TYPE && bool("Must be called on a implicit matrix"));
  switch (element.numeric_type)
  {
  //case scheduler::CHAR_TYPE: return fun(*element.implicit_matrix_char);
  //case scheduler::UCHAR_TYPE: return fun(*element.implicit_matrix_uchar);
  //case scheduler::SHORT_TYPE: return fun(*element.implicit_matrix_short);
  //case scheduler::USHORT_TYPE: return fun(*element.implicit_matrix_ushort);
  case scheduler::INT_TYPE:  return fun(*element.implicit_matrix_int);
  case scheduler::UINT_TYPE: return fun(*element.implicit_matrix_uint);
  case scheduler::LONG_TYPE:  return fun(*element.implicit_matrix_long);
  case scheduler::ULONG_TYPE: return fun(*element.implicit_matrix_ulong);
  case scheduler::FLOAT_TYPE : return fun(*element.implicit_matrix_float);
  case scheduler::DOUBLE_TYPE : return fun(*element.implicit_matrix_double);
  default : throw generator_not_supported_exception("Unsupported Scalartype");
  }
}

template<class Fun>
static typename Fun::result_type call_on_element(scheduler::lhs_rhs_element const & element, Fun const & fun){
  switch (element.type_family)
  {
  case scheduler::SCALAR_TYPE_FAMILY:
    if (element.subtype == scheduler::HOST_SCALAR_TYPE)
      return call_on_host_scalar(element, fun);
    else
      return call_on_scalar(element, fun);
  case scheduler::VECTOR_TYPE_FAMILY :
    if (element.subtype == scheduler::IMPLICIT_VECTOR_TYPE)
      return call_on_implicit_vector(element, fun);
    else
      return call_on_vector(element, fun);
  case scheduler::MATRIX_TYPE_FAMILY:
    if (element.subtype == scheduler::IMPLICIT_MATRIX_TYPE)
      return call_on_implicit_matrix(element, fun);
    else
      return call_on_matrix(element,fun);
  default:
    throw generator_not_supported_exception("Unsupported datastructure type : Not among {Scalar, Vector, Matrix}");
  }
}

struct scalartype_size_fun
{
  typedef vcl_size_t result_type;
  result_type operator()(float const &) const { return sizeof(float); }
  result_type operator()(double const &) const { return sizeof(double); }
  template<class T> result_type operator()(T const &) const { return sizeof(typename viennacl::result_of::cpu_value_type<T>::type); }
};

struct internal_size_fun
{
  typedef vcl_size_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::internal_size(t); }
};

struct size_fun
{
  typedef vcl_size_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::size(t); }
};

struct stride_fun
{
  typedef vcl_size_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::stride(t); }
};

struct start1_fun
{
  typedef vcl_size_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::start1(t); }
};

struct start2_fun
{
  typedef vcl_size_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::start2(t); }
};

struct leading_stride
{
  typedef vcl_size_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::row_major(t)?viennacl::traits::stride2(t):viennacl::traits::stride1(t); }
};

struct leading_start
{
  typedef vcl_size_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::row_major(t)?viennacl::traits::start2(t):viennacl::traits::start1(t); }
};

struct stride1_fun
{
  typedef vcl_size_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::stride1(t); }
};

struct stride2_fun
{
  typedef vcl_size_t result_type;
  template<class T> result_type operator()(T const &t) const { return viennacl::traits::stride2(t); }
};

struct handle_fun
{
  typedef cl_mem result_type;
  template<class T>
  result_type operator()(T const &t) const { return viennacl::traits::opencl_handle(t); }
};

struct internal_size1_fun
{
  typedef vcl_size_t result_type;
  template<class T>
  result_type operator()(T const &t) const { return viennacl::traits::internal_size1(t); }
};

struct row_major_fun
{
  typedef bool result_type;
  template<class T>
  result_type operator()(T const &t) const { return viennacl::traits::row_major(t); }
};

struct internal_size2_fun
{
  typedef vcl_size_t result_type;
  template<class T>
  result_type operator()(T const &t) const { return viennacl::traits::internal_size2(t); }
};

struct size1_fun
{
  typedef vcl_size_t result_type;
  template<class T>
  result_type operator()(T const &t) const { return viennacl::traits::size1(t); }
};

struct size2_fun
{
  typedef vcl_size_t result_type;
  template<class T>
  result_type operator()(T const &t) const { return viennacl::traits::size2(t); }
};

template<class T, class U>
struct is_same_type { enum { value = 0 }; };

template<class T>
struct is_same_type<T,T> { enum { value = 1 }; };

inline bool is_reduction(scheduler::statement_node const & node)
{
  return node.op.type_family==scheduler::OPERATION_VECTOR_REDUCTION_TYPE_FAMILY
      || node.op.type_family==scheduler::OPERATION_COLUMNS_REDUCTION_TYPE_FAMILY
      || node.op.type_family==scheduler::OPERATION_ROWS_REDUCTION_TYPE_FAMILY
      || node.op.type==scheduler::OPERATION_BINARY_INNER_PROD_TYPE
      || node.op.type==scheduler::OPERATION_BINARY_MAT_VEC_PROD_TYPE;
}

inline bool is_index_reduction(scheduler::op_element const & op)
{
  return op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE
      || op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGMAX_TYPE
      || op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE
      || op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGMIN_TYPE;
}
template<class T>
struct type_to_string;
template<> struct type_to_string<unsigned char> { static const char * value() { return "uchar"; } };
template<> struct type_to_string<char> { static const char * value() { return "char"; } };
template<> struct type_to_string<unsigned short> { static const char * value() { return "ushort"; } };
template<> struct type_to_string<short> { static const char * value() { return "short"; } };
template<> struct type_to_string<unsigned int> { static const char * value() { return "uint"; } };
template<> struct type_to_string<int> { static const char * value() { return "int"; } };
template<> struct type_to_string<unsigned long> { static const char * value() { return "ulong"; } };
template<> struct type_to_string<long> { static const char * value() { return "long"; } };
template<> struct type_to_string<float> { static const char * value() { return "float"; } };
template<> struct type_to_string<double> { static const char * value() { return "double"; } };


template<class T>
struct first_letter_of_type;
template<> struct first_letter_of_type<char> { static char value() { return 'c'; } };
template<> struct first_letter_of_type<unsigned char> { static char value() { return 'd'; } };
template<> struct first_letter_of_type<short> { static char value() { return 's'; } };
template<> struct first_letter_of_type<unsigned short> { static char value() { return 't'; } };
template<> struct first_letter_of_type<int> { static char value() { return 'i'; } };
template<> struct first_letter_of_type<unsigned int> { static char value() { return 'j'; } };
template<> struct first_letter_of_type<long> { static char value() { return 'l'; } };
template<> struct first_letter_of_type<unsigned long> { static char value() { return 'm'; } };
template<> struct first_letter_of_type<float> { static char value() { return 'f'; } };
template<> struct first_letter_of_type<double> { static char value() { return 'd'; } };

class kernel_generation_stream : public std::ostream
{
  class kgenstream : public std::stringbuf
  {
  public:
    kgenstream(std::ostringstream& osstream,unsigned int const & tab_count) : oss_(osstream), tab_count_(tab_count){ }
    int sync() {
      for (unsigned int i=0; i<tab_count_;++i)
        oss_ << "    ";
      oss_ << str();
      str("");
      return !oss_;
    }
#if defined(_MSC_VER)
    ~kgenstream() throw() {  pubsync(); }
#else
    ~kgenstream() {  pubsync(); }
#endif
  private:
    std::ostream& oss_;
    unsigned int const & tab_count_;
  };

public:
  kernel_generation_stream() : std::ostream(new kgenstream(oss,tab_count_)), tab_count_(0){ }
#if defined(_MSC_VER)
  ~kernel_generation_stream() throw() { delete rdbuf(); }
#else
  ~kernel_generation_stream(){ delete rdbuf(); }
#endif

  std::string str(){ return oss.str(); }
  void inc_tab(){ ++tab_count_; }
  void dec_tab(){ --tab_count_; }
private:
  unsigned int tab_count_;
  std::ostringstream oss;
};

inline bool node_leaf(scheduler::op_element const & op)
{
  using namespace scheduler;
  return op.type==OPERATION_UNARY_NORM_1_TYPE
      || op.type==OPERATION_UNARY_NORM_2_TYPE
      || op.type==OPERATION_UNARY_NORM_INF_TYPE
      || op.type==OPERATION_UNARY_TRANS_TYPE
      || op.type==OPERATION_BINARY_MAT_VEC_PROD_TYPE
      || op.type==OPERATION_BINARY_MAT_MAT_PROD_TYPE
      || op.type==OPERATION_BINARY_INNER_PROD_TYPE
      || op.type==OPERATION_BINARY_MATRIX_DIAG_TYPE
      || op.type==OPERATION_BINARY_VECTOR_DIAG_TYPE
      || op.type==OPERATION_BINARY_MATRIX_ROW_TYPE
      || op.type==OPERATION_BINARY_MATRIX_COLUMN_TYPE
      || op.type_family==OPERATION_VECTOR_REDUCTION_TYPE_FAMILY
      || op.type_family==OPERATION_ROWS_REDUCTION_TYPE_FAMILY
      || op.type_family==OPERATION_COLUMNS_REDUCTION_TYPE_FAMILY;
}

inline bool elementwise_operator(scheduler::op_element const & op)
{
  using namespace scheduler;
  return op.type== OPERATION_BINARY_ASSIGN_TYPE
      || op.type== OPERATION_BINARY_INPLACE_ADD_TYPE
      || op.type== OPERATION_BINARY_INPLACE_SUB_TYPE
      || op.type== OPERATION_BINARY_ADD_TYPE
      || op.type== OPERATION_BINARY_SUB_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_PROD_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_DIV_TYPE
      || op.type== OPERATION_BINARY_MULT_TYPE
      || op.type== OPERATION_BINARY_DIV_TYPE;
}

inline bool elementwise_function(scheduler::op_element const & op)
{
  using namespace scheduler;
  return

      op.type == OPERATION_UNARY_CAST_CHAR_TYPE
      || op.type == OPERATION_UNARY_CAST_UCHAR_TYPE
      || op.type == OPERATION_UNARY_CAST_SHORT_TYPE
      || op.type == OPERATION_UNARY_CAST_USHORT_TYPE
      || op.type == OPERATION_UNARY_CAST_INT_TYPE
      || op.type == OPERATION_UNARY_CAST_UINT_TYPE
      || op.type == OPERATION_UNARY_CAST_LONG_TYPE
      || op.type == OPERATION_UNARY_CAST_ULONG_TYPE
      || op.type == OPERATION_UNARY_CAST_HALF_TYPE
      || op.type == OPERATION_UNARY_CAST_FLOAT_TYPE
      || op.type == OPERATION_UNARY_CAST_DOUBLE_TYPE

      || op.type== OPERATION_UNARY_ABS_TYPE
      || op.type== OPERATION_UNARY_ACOS_TYPE
      || op.type== OPERATION_UNARY_ASIN_TYPE
      || op.type== OPERATION_UNARY_ATAN_TYPE
      || op.type== OPERATION_UNARY_CEIL_TYPE
      || op.type== OPERATION_UNARY_COS_TYPE
      || op.type== OPERATION_UNARY_COSH_TYPE
      || op.type== OPERATION_UNARY_EXP_TYPE
      || op.type== OPERATION_UNARY_FABS_TYPE
      || op.type== OPERATION_UNARY_FLOOR_TYPE
      || op.type== OPERATION_UNARY_LOG_TYPE
      || op.type== OPERATION_UNARY_LOG10_TYPE
      || op.type== OPERATION_UNARY_SIN_TYPE
      || op.type== OPERATION_UNARY_SINH_TYPE
      || op.type== OPERATION_UNARY_SQRT_TYPE
      || op.type== OPERATION_UNARY_TAN_TYPE
      || op.type== OPERATION_UNARY_TANH_TYPE

      || op.type== OPERATION_BINARY_ELEMENT_POW_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_EQ_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_NEQ_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_GREATER_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_LESS_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_GEQ_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_LEQ_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_FMAX_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_FMIN_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_MAX_TYPE
      || op.type== OPERATION_BINARY_ELEMENT_MIN_TYPE;

}

inline scheduler::lhs_rhs_element & lhs_rhs_element(scheduler::statement const & st, vcl_size_t idx, leaf_t leaf)
{
  using namespace tree_parsing;
  assert(leaf==LHS_NODE_TYPE || leaf==RHS_NODE_TYPE);
  if (leaf==LHS_NODE_TYPE)
    return const_cast<scheduler::lhs_rhs_element &>(st.array()[idx].lhs);
  return const_cast<scheduler::lhs_rhs_element &>(st.array()[idx].rhs);
}

inline unsigned int size_of(scheduler::statement_node_numeric_type type)
{
  using namespace scheduler;
  switch (type)
  {
  case UCHAR_TYPE:
  case CHAR_TYPE: return 1;

  case USHORT_TYPE:
  case SHORT_TYPE:
  case HALF_TYPE: return 2;

  case UINT_TYPE:
  case INT_TYPE:
  case FLOAT_TYPE: return 4;

  case ULONG_TYPE:
  case LONG_TYPE:
  case DOUBLE_TYPE: return 8;

  default: throw generator_not_supported_exception("Unsupported scalartype");
  }
}

inline std::string append_width(std::string const & str, unsigned int width)
{
  if (width==1)
    return str;
  return str + tools::to_string(width);
}

}
}
}
#endif
