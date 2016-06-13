#ifndef VIENNACL_TRAITS_CONTEXT_HPP_
#define VIENNACL_TRAITS_CONTEXT_HPP_

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

/** @file viennacl/traits/context.hpp
    @brief Extracts the underlying context from objects
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"
#include "viennacl/context.hpp"
#include "viennacl/traits/handle.hpp"

namespace viennacl
{
namespace traits
{

// Context
/** @brief Returns an ID for the currently active memory domain of an object */
template<typename T>
viennacl::context context(T const & t)
{
#ifdef VIENNACL_WITH_OPENCL
  if (traits::active_handle_id(t) == OPENCL_MEMORY)
    return viennacl::context(traits::opencl_handle(t).context());
#endif

  return viennacl::context(traits::active_handle_id(t));
}

/** @brief Returns an ID for the currently active memory domain of an object */
inline viennacl::context context(viennacl::backend::mem_handle const & h)
{
#ifdef VIENNACL_WITH_OPENCL
  if (h.get_active_handle_id() == OPENCL_MEMORY)
    return viennacl::context(h.opencl_handle().context());
#endif

  return viennacl::context(h.get_active_handle_id());
}

} //namespace traits
} //namespace viennacl


#endif
