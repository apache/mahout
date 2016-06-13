#ifndef VIENNACL_OCL_UTILS_HPP_
#define VIENNACL_OCL_UTILS_HPP_

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

/** @file viennacl/ocl/utils.hpp
    @brief Provides OpenCL-related utilities.
*/

#include <vector>
#include <string>
#include "viennacl/ocl/backend.hpp"
#include "viennacl/ocl/device.hpp"

namespace viennacl
{
namespace ocl
{

/** @brief Ensures that double precision types are only allocated if it is supported by the device. If double precision is requested for a device not capable of providing that, a double_precision_not_provided_error is thrown.
 */
template<typename ScalarType>
struct DOUBLE_PRECISION_CHECKER
{
  static void apply(viennacl::ocl::context const &) {}
};

/** \cond */
template<>
struct DOUBLE_PRECISION_CHECKER<double>
{
  static void apply(viennacl::ocl::context const & ctx)
  {
    if (!ctx.current_device().double_support())
      throw viennacl::ocl::double_precision_not_provided_error();
  }
};
/** \endcond */

/** \brief Helper class for converting a type to its string representation. */
template<typename T>
struct type_to_string;

/** \cond */
template<> struct type_to_string<char>   { static std::string apply() { return "char";  } };
template<> struct type_to_string<short>  { static std::string apply() { return "short"; } };
template<> struct type_to_string<int>    { static std::string apply() { return "int";   } };
template<> struct type_to_string<long>   { static std::string apply() { return "long";  } };

template<> struct type_to_string<unsigned char>   { static std::string apply() { return "uchar";  } };
template<> struct type_to_string<unsigned short>  { static std::string apply() { return "ushort"; } };
template<> struct type_to_string<unsigned int>    { static std::string apply() { return "uint";   } };
template<> struct type_to_string<unsigned long>   { static std::string apply() { return "ulong";  } };

template<> struct type_to_string<float>  { static std::string apply() { return "float";  } };
template<> struct type_to_string<double> { static std::string apply() { return "double"; } };
/** \endcond */

template<typename T>
void append_double_precision_pragma(viennacl::ocl::context const & /*ctx*/, std::string & /*source*/) {}

template<>
inline void append_double_precision_pragma<double>(viennacl::ocl::context const & ctx, std::string & source)
{
  source.append("#pragma OPENCL EXTENSION " + ctx.current_device().double_support_extension() + " : enable\n\n");
}

} //ocl
} //viennacl
#endif
