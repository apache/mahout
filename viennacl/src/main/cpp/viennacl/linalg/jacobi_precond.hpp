#ifndef VIENNACL_LINALG_JACOBI_PRECOND_HPP_
#define VIENNACL_LINALG_JACOBI_PRECOND_HPP_

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

/** @file viennacl/linalg/jacobi_precond.hpp
    @brief Implementation of a simple Jacobi preconditioner
*/

#include <vector>
#include <cmath>
#include "viennacl/forwards.h"
#include "viennacl/vector.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/tools/tools.hpp"
#include "viennacl/linalg/sparse_matrix_operations.hpp"
#include "viennacl/linalg/row_scaling.hpp"

#include <map>

namespace viennacl
{
namespace linalg
{

/** @brief A tag for a jacobi preconditioner
*/
class jacobi_tag {};


/** @brief Jacobi preconditioner class, can be supplied to solve()-routines. Generic version for non-ViennaCL matrices.
*/
template<typename MatrixT,
          bool is_viennacl = detail::row_scaling_for_viennacl<MatrixT>::value >
class jacobi_precond
{
  typedef typename MatrixT::value_type      NumericType;

  public:
    jacobi_precond(MatrixT const & mat, jacobi_tag const &) : diag_A_(viennacl::traits::size1(mat))
    {
      init(mat);
    }

    void init(MatrixT const & mat)
    {
      diag_A_.resize(viennacl::traits::size1(mat));  //resize without preserving values

      for (typename MatrixT::const_iterator1 row_it = mat.begin1();
            row_it != mat.end1();
            ++row_it)
      {
        bool diag_found = false;
        for (typename MatrixT::const_iterator2 col_it = row_it.begin();
              col_it != row_it.end();
              ++col_it)
        {
          if (col_it.index1() == col_it.index2())
          {
            diag_A_[col_it.index1()] = *col_it;
            diag_found = true;
          }
        }
        if (!diag_found)
          throw zero_on_diagonal_exception("ViennaCL: Zero in diagonal encountered while setting up Jacobi preconditioner!");
      }
    }


    /** @brief Apply to res = b - Ax, i.e. jacobi applied vec (right hand side),  */
    template<typename VectorT>
    void apply(VectorT & vec) const
    {
      assert(viennacl::traits::size(diag_A_) == viennacl::traits::size(vec) && bool("Size mismatch"));
      for (vcl_size_t i=0; i<diag_A_.size(); ++i)
        vec[i] /= diag_A_[i];
    }

  private:
    std::vector<NumericType> diag_A_;
};


/** @brief Jacobi preconditioner class, can be supplied to solve()-routines.
*
*  Specialization for compressed_matrix
*/
template<typename MatrixT>
class jacobi_precond<MatrixT, true>
{
    typedef typename viennacl::result_of::cpu_value_type<typename MatrixT::value_type>::type  NumericType;

  public:
    jacobi_precond(MatrixT const & mat, jacobi_tag const &) : diag_A_(mat.size1(), viennacl::traits::context(mat))
    {
      init(mat);
    }


    void init(MatrixT const & mat)
    {
      detail::row_info(mat, diag_A_, detail::SPARSE_ROW_DIAGONAL);
    }


    template<unsigned int AlignmentV>
    void apply(viennacl::vector<NumericType, AlignmentV> & vec) const
    {
      assert(viennacl::traits::size(diag_A_) == viennacl::traits::size(vec) && bool("Size mismatch"));
      vec = element_div(vec, diag_A_);
    }

  private:
    viennacl::vector<NumericType> diag_A_;
};

}
}




#endif



