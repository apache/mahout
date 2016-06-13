#ifndef VIENNACL_TOOLS_RANDOM_HPP
#define VIENNACL_TOOLS_RANDOM_HPP

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

#include <ctime>
#include <cstdlib>
#include <cmath>


/** @file   viennacl/tools/random.hpp
 *  @brief  A small collection of sequential random number generators.
 *
 *  Should not be considered a source of high-quality random numbers.
 *  It is, however, enough to produce meaningful initial guesses.
 */

namespace viennacl
{
namespace tools
{

/** @brief Random number generator for returning uniformly distributed values in the closed interval [0, 1]
 *
 *  Currently based on rand(), which may have fairly poor quality.
 *  To be replaced in the future, but serves the purpose for the time being.
 */
template<typename NumericT>
class uniform_random_numbers
{
public:
  NumericT operator()() const { return static_cast<NumericT>(double(rand()) / double(RAND_MAX)); }
};


/** @brief Random number generator for returning normally distributed values
  *
  * Implementation based on Box-Muller transformation:
  * https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
  */
template<typename NumericT>
class normal_random_numbers
{
public:
  normal_random_numbers(NumericT mean = NumericT(0), NumericT sigma = NumericT(1)) : mean_(mean), sigma_(sigma) {}

  NumericT operator()() const
  {
    NumericT u = 0;
    while (std::fabs(u) <= 0 || u >= NumericT(1))
     u = static_cast<NumericT>(double(rand()) / double(RAND_MAX));

    NumericT v = 0;
    while (std::fabs(v) <= 0 || v >= NumericT(1))
     v = static_cast<NumericT>(double(rand()) / double(RAND_MAX));

    return mean_ + sigma_ * std::sqrt(NumericT(-2) * std::log(u)) * std::cos(NumericT(6.28318530717958647692) * v);
  }

private:
  NumericT mean_;
  NumericT sigma_;
};

}
}

#endif

