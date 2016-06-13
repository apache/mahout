#ifndef VIENNACL_TOOLS_TOOLS_HPP_
#define VIENNACL_TOOLS_TOOLS_HPP_

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

/** @file viennacl/tools/tools.hpp
    @brief Various little tools used here and there in ViennaCL.
*/

#include <string>
#include <fstream>
#include <sstream>
#include "viennacl/forwards.h"
#include "viennacl/tools/adapter.hpp"

#include <vector>
#include <map>

namespace viennacl
{
namespace tools
{

/** \cond */
/** @brief Supply suitable increment functions for the iterators: */
template<class NumericT, typename F, unsigned int AlignmentV>
struct MATRIX_ITERATOR_INCREMENTER<viennacl::row_iteration, viennacl::matrix<NumericT, F, AlignmentV> >
{
  static void apply(const viennacl::matrix<NumericT, F, AlignmentV> & /*mat*/, unsigned int & row, unsigned int & /*col*/)
  {
    ++row;
  }
};

template<class NumericT, typename F, unsigned int AlignmentV>
struct MATRIX_ITERATOR_INCREMENTER<viennacl::col_iteration, viennacl::matrix<NumericT, F, AlignmentV> >
{
  static void apply(const viennacl::matrix<NumericT, F, AlignmentV> & /*mat*/, unsigned int & /*row*/, unsigned int & col)
  {
    ++col;
  }
};
/** \endcond */


/** @brief A guard that checks whether the floating point type of GPU types is either float or double */
template<typename T>
struct CHECK_SCALAR_TEMPLATE_ARGUMENT
{
  typedef typename T::ERROR_SCALAR_MUST_HAVE_TEMPLATE_ARGUMENT_FLOAT_OR_DOUBLE  ResultType;
};

/** \cond */
template<>
struct CHECK_SCALAR_TEMPLATE_ARGUMENT<float>
{
  typedef float  ResultType;
};

template<>
struct CHECK_SCALAR_TEMPLATE_ARGUMENT<double>
{
  typedef double  ResultType;
};
/** \endcond */



/** @brief Reads a text from a file into a std::string
*
* @param filename   The filename
* @return The text read from the file
*/
inline std::string read_text_from_file(const std::string & filename)
{
  std::ifstream f(filename.c_str());
  if (!f) return std::string();

  std::stringstream result;
  std::string tmp;
  while (std::getline(f, tmp))
    result << tmp << std::endl;

  return result.str();
}

/** @brief Replaces all occurances of a substring by another stringstream
*
* @param text   The string to search in
* @param to_search  The substring to search for
* @param to_replace The replacement for found substrings
* @return The resulting string
*/
inline std::string str_replace(const std::string & text, std::string to_search, std::string to_replace)
{
  std::string::size_type pos = 0;
  std::string result;
  std::string::size_type found;
  while ( (found = text.find(to_search, pos)) != std::string::npos )
  {
    result.append(text.substr(pos,found-pos));
    result.append(to_replace);
    pos = found + to_search.length();
  }
  if (pos < text.length())
    result.append(text.substr(pos));
  return result;
}

/** @brief Rounds an integer to the next multiple of another integer
*
* @tparam INT_TYPE  The integer type
* @param to_reach   The integer to be rounded up (ceil operation)
* @param base       The base
* @return The smallest multiple of 'base' such that to_reach <= base
*/
template<class INT_TYPE>
INT_TYPE align_to_multiple(INT_TYPE to_reach, INT_TYPE base)
{
  if (to_reach % base == 0) return to_reach;
  return ((to_reach / base) + 1) * base;
}


/** @brief Rounds an integer to the previous multiple of another integer
*
* @tparam INT_TYPE  The integer type
* @param to_reach   The integer to be rounded down (floor operation)
* @param base       The base
* @return The biggest multiple of 'base' such that to_reach >= base
*/
template<class INT_TYPE>
INT_TYPE round_down_to_prevous_multiple(INT_TYPE to_reach, INT_TYPE base)
{
  if (to_reach % base == 0) return to_reach;
  return (to_reach / base) * base;
}

/** @brief Replace in a source string a pattern by another
 *
 * @param source The source string
 * @param find String to find
 * @param replace String to replace
 */
int inline find_and_replace(std::string & source, std::string const & find, std::string const & replace)
{
  int num=0;
  vcl_size_t fLen = find.size();
  vcl_size_t rLen = replace.size();
  for (vcl_size_t pos=0; (pos=source.find(find, pos))!=std::string::npos; pos+=rLen)
  {
    num++;
    source.replace(pos, fLen, replace);
  }
  return num;
}

/** @brief  Returns true if pred returns true for any of the elements in the range [first,last), and false otherwise.*/
template<class InputIterator, class UnaryPredicate>
bool any_of (InputIterator first, InputIterator last, UnaryPredicate pred)
{
  while (first!=last)
  {
    if (pred(*first)) return true;
    ++first;
  }
  return false;
}

/** @brief Create a double precision kernel out of a single precision kernel
*
* @param source          The source string
* @param fp_extension    An info string that specifies the OpenCL double precision extension
* @return   The double precision kernel
*/
inline std::string make_double_kernel(std::string const & source, std::string const & fp_extension)
{
  std::stringstream ss;
  ss << "#pragma OPENCL EXTENSION " << fp_extension << " : enable\n\n";

  std::string result = ss.str();
  result.append(str_replace(source, "float", "double"));
  return result;
}


/** @brief Removes the const qualifier from a type */
template<typename T>
struct CONST_REMOVER
{
  typedef T   ResultType;
};

/** \cond */
template<typename T>
struct CONST_REMOVER<const T>
{
  typedef T   ResultType;
};
/** \endcond */


/////// CPU scalar type deducer ///////////

/** @brief Obtain the cpu scalar type from a type, including a GPU type like viennacl::scalar<T>
*
* @tparam T   Either a CPU scalar type or a GPU scalar type
*/
template<typename T>
struct CPU_SCALAR_TYPE_DEDUCER
{
  //force compiler error if type cannot be deduced
  //typedef T       ResultType;
};

/** \cond */
template<>
struct CPU_SCALAR_TYPE_DEDUCER< float >
{
  typedef float       ResultType;
};

template<>
struct CPU_SCALAR_TYPE_DEDUCER< double >
{
  typedef double       ResultType;
};

template<typename T>
struct CPU_SCALAR_TYPE_DEDUCER< viennacl::scalar<T> >
{
  typedef T       ResultType;
};

template<typename T, unsigned int A>
struct CPU_SCALAR_TYPE_DEDUCER< viennacl::vector<T, A> >
{
  typedef T       ResultType;
};

template<typename T, typename F, unsigned int A>
struct CPU_SCALAR_TYPE_DEDUCER< viennacl::matrix<T, F, A> >
{
  typedef T       ResultType;
};


template<typename T, typename F, unsigned int A>
struct CPU_SCALAR_TYPE_DEDUCER< viennacl::matrix_expression<const matrix<T, F, A>, const matrix<T, F, A>, op_trans> >
{
  typedef T       ResultType;
};
/** \endcond */

//
// Converts a scalar type when necessary unless it is a viennacl::scalar<> (typical use-case: convert user-provided floats to double (and vice versa) for OpenCL kernels)
//

template<typename HostScalarType>
viennacl::scalar<HostScalarType> const & promote_if_host_scalar(viennacl::scalar<HostScalarType> const & s) { return s; }

template<typename HostScalarType>
viennacl::scalar_expression<const viennacl::scalar<HostScalarType>,
const viennacl::scalar<HostScalarType>,
viennacl::op_flip_sign> const &
promote_if_host_scalar(viennacl::scalar_expression<const viennacl::scalar<HostScalarType>,
                       const viennacl::scalar<HostScalarType>,
                       viennacl::op_flip_sign> const & s) { return s; }

template<typename HostScalarType>
HostScalarType promote_if_host_scalar(float s) { return s; }

template<typename HostScalarType>
HostScalarType promote_if_host_scalar(double s) { return s; }

template<typename HostScalarType>
HostScalarType promote_if_host_scalar(long s) { return s; }

template<typename HostScalarType>
HostScalarType promote_if_host_scalar(unsigned long s) { return s; }

template<typename HostScalarType>
HostScalarType promote_if_host_scalar(int s) { return s; }

template<typename HostScalarType>
HostScalarType promote_if_host_scalar(unsigned int s) { return s; }

template<class T>
inline std::string to_string ( T const t )
{
  std::stringstream ss;
  ss << t;
  return ss.str();
}

} //namespace tools
} //namespace viennacl


#endif
