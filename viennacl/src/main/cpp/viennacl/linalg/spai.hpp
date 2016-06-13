#ifndef VIENNACL_LINALG_SPAI_HPP
#define VIENNACL_LINALG_SPAI_HPP

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

/** @file viennacl/linalg/spai.hpp
    @brief Main include file for the sparse approximate inverse preconditioner family (SPAI and FSPAI).  Experimental.

    Most implementation contributed by Nikolay Lukash.
*/


#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <algorithm>
#include <vector>
#include <math.h>
#include <map>

// ViennaCL includes
#include "viennacl/linalg/detail/spai/spai_tag.hpp"
#include "viennacl/linalg/qr.hpp"
#include "viennacl/linalg/prod.hpp"
#include "viennacl/linalg/detail/spai/spai-dynamic.hpp"
#include "viennacl/linalg/detail/spai/spai-static.hpp"
#include "viennacl/linalg/detail/spai/sparse_vector.hpp"
#include "viennacl/linalg/detail/spai/block_matrix.hpp"
#include "viennacl/linalg/detail/spai/block_vector.hpp"
#include "viennacl/linalg/detail/spai/fspai.hpp"
#include "viennacl/linalg/detail/spai/spai.hpp"

//boost includes
#include "boost/numeric/ublas/vector.hpp"
#include "boost/numeric/ublas/matrix.hpp"
#include "boost/numeric/ublas/matrix_proxy.hpp"
#include "boost/numeric/ublas/vector_proxy.hpp"
#include "boost/numeric/ublas/storage.hpp"
#include "boost/numeric/ublas/io.hpp"
#include "boost/numeric/ublas/lu.hpp"
#include "boost/numeric/ublas/triangular.hpp"
#include "boost/numeric/ublas/matrix_expression.hpp"


namespace viennacl
{
    namespace linalg
    {

        typedef viennacl::linalg::detail::spai::spai_tag         spai_tag;
        typedef viennacl::linalg::detail::spai::fspai_tag        fspai_tag;

        /** @brief Implementation of the SParse Approximate Inverse Algorithm for a generic, uBLAS-compatible matrix type.
         * @param Matrix matrix that is used for computations
         * @param Vector vector that is used for computations
         */
        //UBLAS version
        template<typename MatrixType>
        class spai_precond
        {
        public:
            typedef typename MatrixType::value_type ScalarType;
            typedef typename boost::numeric::ublas::vector<ScalarType> VectorType;
            /** @brief Constructor
             * @param A matrix whose approximate inverse is calculated. Must be quadratic.
             * @param tag spai tag
             */
            spai_precond(const MatrixType& A,
                         const spai_tag& tag): tag_(tag){

                //VCLMatrixType vcl_Ap((unsigned int)A.size2(), (unsigned int)A.size1()), vcl_A((unsigned int)A.size1(), (unsigned int)A.size2()),
                //vcl_At((unsigned int)A.size1(), (unsigned int)A.size2());
                //UBLASDenseMatrixType dA = A;
                MatrixType pA(A.size1(), A.size2());
                MatrixType At;
                //std::cout<<A<<std::endl;
                if (!tag_.getIsRight()){
                    viennacl::linalg::detail::spai::sparse_transpose(A, At);
                }else{
                    At = A;
                }
                pA = At;
                viennacl::linalg::detail::spai::initPreconditioner(pA, spai_m_);
                viennacl::linalg::detail::spai::computeSPAI(At, spai_m_, tag_);
                //(At, pA, tag_.getIsRight(), tag_.getIsStatic(), (ScalarType)_tag.getResidualNormThreshold(), (unsigned int)_tag.getIterationLimit(),
                 //_spai_m);

            }
            /** @brief Application of current preconditioner, multiplication on the right-hand side vector
             * @param vec rhs vector
             */
            void apply(VectorType& vec) const {
                vec = viennacl::linalg::prod(spai_m_, vec);
            }
        private:
            // variables
            spai_tag tag_;
            // result of SPAI
            MatrixType spai_m_;
        };

        //VIENNACL version
        /** @brief Implementation of the SParse Approximate Inverse Algorithm for a ViennaCL compressed_matrix.
         * @param Matrix matrix that is used for computations
         * @param Vector vector that is used for computations
         */
        template<typename ScalarType, unsigned int MAT_ALIGNMENT>
        class spai_precond< viennacl::compressed_matrix<ScalarType, MAT_ALIGNMENT> >
        {
            typedef viennacl::compressed_matrix<ScalarType, MAT_ALIGNMENT> MatrixType;
            typedef boost::numeric::ublas::compressed_matrix<ScalarType> UBLASSparseMatrixType;
            typedef viennacl::vector<ScalarType> VectorType;
            typedef viennacl::matrix<ScalarType> VCLDenseMatrixType;

            typedef boost::numeric::ublas::vector<ScalarType> UBLASVectorType;
        public:

            /** @brief Constructor
             * @param A matrix whose approximate inverse is calculated. Must be quadratic.
             * @param tag spai tag
             */
            spai_precond(const MatrixType& A,
                         const spai_tag& tag): tag_(tag), spai_m_(viennacl::traits::context(A))
            {
                viennacl::ocl::context & ctx = const_cast<viennacl::ocl::context &>(viennacl::traits::opencl_handle(A).context());
                viennacl::linalg::opencl::kernels::spai<ScalarType>::init(ctx);

                MatrixType At(A.size1(), A.size2(), viennacl::context(ctx));
                UBLASSparseMatrixType ubls_A(A.size1(), A.size2()), ubls_spai_m;
                UBLASSparseMatrixType ubls_At;
                viennacl::copy(A, ubls_A);
                if (!tag_.getIsRight()){
                    viennacl::linalg::detail::spai::sparse_transpose(ubls_A, ubls_At);
                }
                else{
                    ubls_At = ubls_A;
                }
                //current pattern is A
                //pA = ubls_At;
                //execute SPAI with ublas matrix types
                viennacl::linalg::detail::spai::initPreconditioner(ubls_At, ubls_spai_m);
                viennacl::copy(ubls_At, At);
                viennacl::linalg::detail::spai::computeSPAI(At, ubls_At, ubls_spai_m, spai_m_, tag_);
                //viennacl::copy(ubls_spai_m, spai_m_);
                tmp_.resize(A.size1(), viennacl::traits::context(A), false);
            }
            /** @brief Application of current preconditioner, multiplication on the right-hand side vector
             * @param vec rhs vector
             */
            void apply(VectorType& vec) const {
                tmp_ = viennacl::linalg::prod(spai_m_, vec);
                vec = tmp_;
            }
        private:
            // variables
            spai_tag tag_;
            // result of SPAI
            MatrixType spai_m_;
            mutable VectorType tmp_;
        };


        //
        // FSPAI
        //

        /** @brief Implementation of the Factored SParse Approximate Inverse Algorithm for a generic, uBLAS-compatible matrix type.
        * @param Matrix matrix that is used for computations
        * @param Vector vector that is used for computations
        */
        //UBLAS version
        template<typename MatrixType>
        class fspai_precond
        {
            typedef typename MatrixType::value_type ScalarType;
            typedef typename boost::numeric::ublas::vector<ScalarType> VectorType;
            typedef typename boost::numeric::ublas::matrix<ScalarType> UBLASDenseMatrixType;
            typedef typename viennacl::matrix<ScalarType> VCLMatrixType;
        public:

            /** @brief Constructor
            * @param A matrix whose approximate inverse is calculated. Must be quadratic.
            * @param tag SPAI configuration tag
            */
            fspai_precond(const MatrixType& A,
                        const fspai_tag& tag): tag_(tag)
            {
                MatrixType pA = A;
                viennacl::linalg::detail::spai::computeFSPAI(A, pA, L, L_trans, tag_);
            }

            /** @brief Application of current preconditioner, multiplication on the right-hand side vector
            * @param vec rhs vector
            */
            void apply(VectorType& vec) const
            {
              VectorType temp = viennacl::linalg::prod(L_trans, vec);
              vec = viennacl::linalg::prod(L, temp);
            }

        private:
            // variables
            const fspai_tag & tag_;
            // result of SPAI
            MatrixType L;
            MatrixType L_trans;
        };





        //
        // ViennaCL version
        //
        /** @brief Implementation of the Factored SParse Approximate Inverse Algorithm for a ViennaCL compressed_matrix.
        * @param Matrix matrix that is used for computations
        * @param Vector vector that is used for computations
        */
        template<typename ScalarType, unsigned int MAT_ALIGNMENT>
        class fspai_precond< viennacl::compressed_matrix<ScalarType, MAT_ALIGNMENT> >
        {
            typedef viennacl::compressed_matrix<ScalarType, MAT_ALIGNMENT>   MatrixType;
            typedef viennacl::vector<ScalarType> VectorType;
            typedef viennacl::matrix<ScalarType> VCLDenseMatrixType;
            typedef boost::numeric::ublas::compressed_matrix<ScalarType> UBLASSparseMatrixType;
            typedef boost::numeric::ublas::vector<ScalarType> UBLASVectorType;
        public:

            /** @brief Constructor
            * @param A matrix whose approximate inverse is calculated. Must be quadratic.
            * @param tag SPAI configuration tag
            */
            fspai_precond(const MatrixType & A,
                          const fspai_tag & tag) : tag_(tag), L(viennacl::traits::context(A)), L_trans(viennacl::traits::context(A)), temp_apply_vec_(A.size1(), viennacl::traits::context(A))
            {
                //UBLASSparseMatrixType ubls_A;
                UBLASSparseMatrixType ublas_A(A.size1(), A.size2());
                UBLASSparseMatrixType pA(A.size1(), A.size2());
                UBLASSparseMatrixType ublas_L(A.size1(), A.size2());
                UBLASSparseMatrixType ublas_L_trans(A.size1(), A.size2());
                viennacl::copy(A, ublas_A);
                //viennacl::copy(ubls_A, vcl_A);
                //vcl_At = viennacl::linalg::prod(vcl_A, vcl_A);
                //vcl_pA = viennacl::linalg::prod(vcl_A, vcl_At);
                //viennacl::copy(vcl_pA, pA);
                pA = ublas_A;
                //execute SPAI with ublas matrix types
                viennacl::linalg::detail::spai::computeFSPAI(ublas_A, pA, ublas_L, ublas_L_trans, tag_);
                //copy back to GPU
                viennacl::copy(ublas_L, L);
                viennacl::copy(ublas_L_trans, L_trans);
            }


            /** @brief Application of current preconditioner, multiplication on the right-hand side vector
            * @param vec rhs vector
            */
            void apply(VectorType& vec) const
            {
              temp_apply_vec_ = viennacl::linalg::prod(L_trans, vec);
              vec = viennacl::linalg::prod(L, temp_apply_vec_);
            }

        private:
            // variables
            const fspai_tag & tag_;
            MatrixType L;
            MatrixType L_trans;
            mutable VectorType temp_apply_vec_;
        };


    }
}
#endif
