#ifndef VIENNACL_DEVICE_SPECIFIC_EXECUTE_HPP
#define VIENNACL_DEVICE_SPECIFIC_EXECUTE_HPP

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


/** @file viennacl/device_specific/execute.hpp
    @brief the user interface for the code generator
*/

#include <cstring>
#include <vector>
#include <typeinfo>

#include "viennacl/scheduler/forwards.h"
#include "viennacl/device_specific/forwards.h"
#include "viennacl/device_specific/templates/template_base.hpp"
#include "viennacl/device_specific/tree_parsing.hpp"
#include "viennacl/device_specific/execution_handler.hpp"

#include "viennacl/tools/tools.hpp"
#include "viennacl/tools/timer.hpp"

namespace viennacl
{
namespace device_specific
{

inline void execute(template_base const & T, statements_container const & statements, viennacl::ocl::context & ctx = viennacl::ocl::current_context(), bool force_compilation = false)
{
  //Generate program name
  std::string program_name = tree_parsing::statements_representation(statements, BIND_TO_HANDLE);
  execution_handler handler(program_name, ctx, ctx.current_device(), force_compilation);
  handler.add(program_name, T, statements);
  handler.execute(program_name, statements);
}

}
}
#endif
