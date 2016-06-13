#ifndef VIENNACL_SRC_BLAS3_HPP
#define VIENNACL_SRC_BLAS3_HPP

/* =========================================================================
   Copyright (c) 2010-2014, Institute for Microelectronics,
                            Institute for Analysis and Scientific Computing,
                            TU Wien.
   Portions of this software are copyright by UChicago Argonne, LLC.

                            -----------------
                  ViennaCL - The Vienna Computing Library
                            -----------------

   Project Head:    Karl Rupp                   rupp@iue.tuwien.ac.at

   (A list of authors and contributors can be found in the PDF manual)

   License:         MIT (X11), see file LICENSE in the base directory
============================================================================= */

// include necessary system headers
#include <iostream>

#include "viennacl.hpp"
#include "viennacl_private.hpp"

//include basic scalar and vector types of ViennaCL
#include "viennacl/scalar.hpp"
#include "viennacl/vector.hpp"

#include "viennacl/vector.hpp"
#include "viennacl/matrix.hpp"
#include "viennacl/linalg/direct_solve.hpp"
#include "viennacl/linalg/prod.hpp"

namespace detail
{
  template <typename ScalarType, typename MatrixTypeA, typename MatrixTypeB, typename MatrixTypeC>
  void gemm_dispatch(ScalarType alpha,
                     MatrixTypeA const & A, ViennaCLTranspose transA,
                     MatrixTypeB const & B, ViennaCLTranspose transB,
                     ScalarType beta,
                     MatrixTypeC & C)
  {

    if (transA == ViennaCLTrans && transB == ViennaCLTrans)
      viennacl::linalg::prod_impl(viennacl::trans(A), viennacl::trans(B), C, alpha, beta);
    else if (transA == ViennaCLTrans && transB == ViennaCLNoTrans)
      viennacl::linalg::prod_impl(viennacl::trans(A), B, C, alpha, beta);
    else if (transA == ViennaCLNoTrans && transB == ViennaCLTrans)
      viennacl::linalg::prod_impl(A, viennacl::trans(B), C, alpha, beta);
    else if (transA == ViennaCLNoTrans && transB == ViennaCLNoTrans)
      viennacl::linalg::prod_impl(A, B, C, alpha, beta);
    //else
    //  return ViennaCLGenericFailure;
  }
}


#endif
