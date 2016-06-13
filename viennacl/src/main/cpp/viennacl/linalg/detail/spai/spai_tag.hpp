#ifndef VIENNACL_LINALG_DETAIL_SPAI_SPAI_TAG_HPP
#define VIENNACL_LINALG_DETAIL_SPAI_SPAI_TAG_HPP

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


/** @file viennacl/linalg/detail/spai/spai_tag.hpp
    @brief Implementation of the spai tag holding SPAI configuration parameters. Experimental.

    SPAI code contributed by Nikolay Lukash
*/


#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <math.h>
#include <cmath>
#include <sstream>
#include "viennacl/ocl/backend.hpp"
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/storage.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/matrix_expression.hpp"
#include "boost/numeric/ublas/detail/matrix_assign.hpp"

#include "viennacl/linalg/detail/spai/block_matrix.hpp"
#include "viennacl/linalg/detail/spai/block_vector.hpp"

namespace viennacl
{
namespace linalg
{
namespace detail
{
namespace spai
{

/** @brief A tag for SPAI
 *
 * Contains values for the algorithm.
 * Must be passed to spai_precond constructor
 */
class spai_tag
{
  /** @brief Constructor
   *
   * @param residual_norm_threshold   Calculate until the norm of the residual falls below this threshold
   * @param iteration_limit           maximum number of iterations
   * @param residual_threshold        determines starting threshold in residual vector for including new indices into set J
   * @param is_static                 determines if static version of SPAI should be used
   * @param is_right                  determines if left or right preconditioner should be used
   */
public:
  spai_tag(double residual_norm_threshold = 1e-3,
           unsigned int iteration_limit = 5,
           double residual_threshold = 1e-2,
           bool is_static = false,
           bool is_right = false)
    : residual_norm_threshold_(residual_norm_threshold),
      iteration_limit_(iteration_limit),
      residual_threshold_(residual_threshold),
      is_static_(is_static),
      is_right_(is_right) {}

  double getResidualNormThreshold() const { return residual_norm_threshold_; }

  double getResidualThreshold() const { return residual_threshold_; }

  unsigned int getIterationLimit () const { return iteration_limit_; }

  bool getIsStatic() const { return is_static_; }

  bool getIsRight() const { return is_right_; }

  long getBegInd() const { return beg_ind_; }

  long getEndInd() const { return end_ind_; }



  void setResidualNormThreshold(double residual_norm_threshold)
  {
    if (residual_norm_threshold > 0)
      residual_norm_threshold_ = residual_norm_threshold;
  }

  void setResidualThreshold(double residual_threshold)
  {
    if (residual_threshold > 0)
      residual_threshold_ = residual_threshold;
  }

  void setIterationLimit(unsigned int iteration_limit)
  {
    if (iteration_limit > 0)
      iteration_limit_ = iteration_limit;
  }

  void setIsRight(bool is_right) { is_right_ = is_right; }

  void setIsStatic(bool is_static) { is_static_ = is_static; }

  void setBegInd(long beg_ind) { beg_ind_ = beg_ind; }

  void setEndInd(long end_ind){ end_ind_ = end_ind; }


private:
  double        residual_norm_threshold_;
  unsigned int  iteration_limit_;
  long          beg_ind_;
  long          end_ind_;
  double        residual_threshold_;
  bool          is_static_;
  bool          is_right_;
};

}
}
}
}
#endif
