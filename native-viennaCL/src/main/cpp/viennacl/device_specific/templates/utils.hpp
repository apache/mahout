#ifndef VIENNACL_DEVICE_SPECIFIC_TEMPLATES_REDUCTION_UTILS_HPP
#define VIENNACL_DEVICE_SPECIFIC_TEMPLATES_REDUCTION_UTILS_HPP

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


/** @file viennacl/device_specific/templates/utils.hpp
 *
 * A collection of utilities for the device specific execution templates.
*/

#include <vector>

#include "viennacl/scheduler/forwards.h"

#include "viennacl/device_specific/tree_parsing.hpp"
#include "viennacl/device_specific/utils.hpp"


namespace viennacl
{
namespace device_specific
{

inline void compute_reduction(utils::kernel_generation_stream & os, std::string acc, std::string cur, scheduler::op_element const & op)
{
  if (utils::elementwise_function(op))
    os << acc << "=" << tree_parsing::evaluate(op.type) << "(" << acc << "," << cur << ");" << std::endl;
  else
    os << acc << "= (" << acc << ")" << tree_parsing::evaluate(op.type)  << "(" << cur << ");" << std::endl;
}

inline void compute_index_reduction(utils::kernel_generation_stream & os, std::string acc, std::string cur, std::string const & acc_value, std::string const & cur_value, scheduler::op_element const & op)
{
  //        os << acc << " = " << cur_value << ">" << acc_value  << "?" << cur << ":" << acc << ";" << std::endl;
  os << acc << "= select(" << acc << "," << cur << "," << cur_value << ">" << acc_value << ");" << std::endl;
  os << acc_value << "=";
  if (op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE) os << "fmax";
  if (op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGMAX_TYPE) os << "max";
  if (op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE) os << "fmin";
  if (op.type==scheduler::OPERATION_BINARY_ELEMENT_ARGMIN_TYPE) os << "min";
  os << "(" << acc_value << "," << cur_value << ");"<< std::endl;
}

inline void process_all(std::string const & type_key, std::string const & str,
                        utils::kernel_generation_stream & stream, std::vector<mapping_type> const & mappings)
{
  for (std::vector<mapping_type>::const_iterator mit = mappings.begin(); mit != mappings.end(); ++mit)
    for (mapping_type::const_iterator mmit = mit->begin(); mmit != mit->end(); ++mmit)
      if (mmit->second->type_key()==type_key)
        stream << mmit->second->process(str) << std::endl;
}


inline void process_all_at(std::string const & type_key, std::string const & str,
                           utils::kernel_generation_stream & stream, std::vector<mapping_type> const & mappings,
                           vcl_size_t root_idx, leaf_t leaf)
{
  for (std::vector<mapping_type>::const_iterator mit = mappings.begin(); mit != mappings.end(); ++mit)
  {
    mapped_object * obj = at(*mit, mapping_key(root_idx, leaf)).get();
    if (obj->type_key()==type_key)
      stream << obj->process(str) << std::endl;
  }
}

inline std::string neutral_element(scheduler::op_element const & op)
{
  switch (op.type)
  {
  case scheduler::OPERATION_BINARY_ADD_TYPE : return "0";
  case scheduler::OPERATION_BINARY_MULT_TYPE : return "1";
  case scheduler::OPERATION_BINARY_DIV_TYPE : return "1";
  case scheduler::OPERATION_BINARY_ELEMENT_FMAX_TYPE : return "-INFINITY";
  case scheduler::OPERATION_BINARY_ELEMENT_ARGFMAX_TYPE : return "-INFINITY";
  case scheduler::OPERATION_BINARY_ELEMENT_MAX_TYPE : return "-INFINITY";
  case scheduler::OPERATION_BINARY_ELEMENT_ARGMAX_TYPE : return "-INFINITY";
  case scheduler::OPERATION_BINARY_ELEMENT_FMIN_TYPE : return "INFINITY";
  case scheduler::OPERATION_BINARY_ELEMENT_ARGFMIN_TYPE : return "INFINITY";
  case scheduler::OPERATION_BINARY_ELEMENT_MIN_TYPE : return "INFINITY";
  case scheduler::OPERATION_BINARY_ELEMENT_ARGMIN_TYPE : return "INFINITY";

  default: throw generator_not_supported_exception("Unsupported reduction operator : no neutral element known");
  }
}

}
}

#endif
