#ifndef VIENNACL_LINALG_DETAIL_OP_APPLIER_HPP
#define VIENNACL_LINALG_DETAIL_OP_APPLIER_HPP

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

/** @file viennacl/linalg/detail/op_applier.hpp
 *
 * @brief Defines the action of certain unary and binary operators and its arguments (for host execution).
*/

#include "viennacl/forwards.h"
#include <cmath>

namespace viennacl
{
namespace linalg
{
namespace detail
{

/** @brief Worker class for decomposing expression templates.
  *
  * @tparam A    Type to which is assigned to
  * @tparam OP   One out of {op_assign, op_inplace_add, op_inplace_sub}
  @ @tparam T    Right hand side of the assignment
*/
template<typename OpT>
struct op_applier
{
  typedef typename OpT::ERROR_UNKNOWN_OP_TAG_PROVIDED    error_type;
};

/** \cond */
template<>
struct op_applier<op_element_binary<op_prod> >
{
  template<typename T>
  static void apply(T & result, T const & x, T const & y) { result = x * y; }
};

template<>
struct op_applier<op_element_binary<op_div> >
{
  template<typename T>
  static void apply(T & result, T const & x, T const & y) { result = x / y; }
};

template<>
struct op_applier<op_element_binary<op_pow> >
{
  template<typename T>
  static void apply(T & result, T const & x, T const & y) { result = std::pow(x, y); }
};

#define VIENNACL_MAKE_UNARY_OP_APPLIER(funcname)  \
template<> \
struct op_applier<op_element_unary<op_##funcname> > \
{ \
  template<typename T> \
  static void apply(T & result, T const & x) { using namespace std; result = funcname(x); } \
}

VIENNACL_MAKE_UNARY_OP_APPLIER(abs);
VIENNACL_MAKE_UNARY_OP_APPLIER(acos);
VIENNACL_MAKE_UNARY_OP_APPLIER(asin);
VIENNACL_MAKE_UNARY_OP_APPLIER(atan);
VIENNACL_MAKE_UNARY_OP_APPLIER(ceil);
VIENNACL_MAKE_UNARY_OP_APPLIER(cos);
VIENNACL_MAKE_UNARY_OP_APPLIER(cosh);
VIENNACL_MAKE_UNARY_OP_APPLIER(exp);
VIENNACL_MAKE_UNARY_OP_APPLIER(fabs);
VIENNACL_MAKE_UNARY_OP_APPLIER(floor);
VIENNACL_MAKE_UNARY_OP_APPLIER(log);
VIENNACL_MAKE_UNARY_OP_APPLIER(log10);
VIENNACL_MAKE_UNARY_OP_APPLIER(sin);
VIENNACL_MAKE_UNARY_OP_APPLIER(sinh);
VIENNACL_MAKE_UNARY_OP_APPLIER(sqrt);
VIENNACL_MAKE_UNARY_OP_APPLIER(tan);
VIENNACL_MAKE_UNARY_OP_APPLIER(tanh);

#undef VIENNACL_MAKE_UNARY_OP_APPLIER
/** \endcond */

}
}
}

#endif // VIENNACL_LINALG_DETAIL_OP_EXECUTOR_HPP
