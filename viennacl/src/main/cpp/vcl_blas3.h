/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <stdlib.h>

#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"

namespace mmul {

    // currently row-major only

    // Dense BLAS-3
    // dense %*% dense
//    void dense_dense_mmul(double* lhs, long lhs_rows, long lhs_cols, double* rhs, long rhs_rows, long rhs_cols, double* result)
   // viennacl::matrix<float> dense_dense_mmul(viennacl::matrix<float> lhs, viennacl::matrix<float> rhs);

    // CSR sparse matrices BLAS-3
    // dense %*% sparse (CSR)
//    viennacl::matrix<double> dense_sparse_mmul(viennacl::matrix<double> lhs, viennacl::compressed_matrix<double> rhs);
//    viennacl::matrix<float> dense_sparse_mmul(viennacl::matrix<float> lhs, viennacl::compressed_matrix<float> rhs);

    // sparse (CSR) %*% dense
//    viennacl::matrix<double> sparse_dense_mmul(viennacl::compressed_matrix<double> lhs, viennacl::matrix<double> rhs);
//    viennacl::matrix<float> sparse_dense_mmul(viennacl::compressed_matrix<float> lhs, viennacl::matrix<float> rhs);

    // sparse(CSR) %*% sparse (CSR)
//    viennacl::compressed_matrix<double> sparse_sparse_mmul(viennacl::compressed_matrix<double> lhs, viennacl::compressed_matrix<double> rhs);
//    viennacl::compressed_matrix<float> sparse_sparse_mmul(viennacl::compressed_matrix<float> lhs, viennacl::compressed_matrix<float> rhs);

    // sparse (CSR) %*% sparse (COO)
//    viennacl::compressed_matrix<double> sparse_sparse_mmul(viennacl::compressed_matrix<double> lhs, viennacl::coordinate_matrix<double> rhs);
//    viennacl::compressed_matrix<float> sparse_sparse_mmul(viennacl::compressed_matrix<float> lhs, viennacl::coordinate_matrix<float> rhs);



    // COO sparse matrices BLAS-3
    // dense %*% sparse (COO)
//    viennacl::matrix<double> dense_sparse_mmul(viennacl::matrix<double> lhs, viennacl::coordinate_matrix<double> rhs);
//    viennacl::matrix<float> dense_sparse_mmul(viennacl::matrix<float> lhs, viennacl::coordinate_matrix<float> rhs);

    // sparse (COO) %*% dense
//    viennacl::matrix<double> sparse_dense_mmul(viennacl::coordinate_matrix<double> lhs, viennacl::matrix<double> rhs);
//    viennacl::matrix<float> sparse_dense_mmul(viennacl::coordinate_matrix<float> lhs, viennacl::matrix<float> rhs);

    // sparse (COO) %*% sparse (COO)
//    viennacl::coordinate_matrix<double> sparse_sparse_mmul(viennacl::coordinate_matrix<double> lhs, viennacl::coordinate_matrix<double> rhs);
//    viennacl::coordinate_matrix<float> sparse_sparse_mmul(viennacl::coordinate_matrix<float> lhs, viennacl::coordinate_matrix<float> rhs);

    // sparse (COO) %*% sparse (CSR)
//    viennacl::coordinate_matrix<double> sparse_sparse_mmul(viennacl::coordinate_matrix<double> lhs, viennacl::compressed_matrix<double> rhs);
//    viennacl::coordinate_matrix<float> sparse_sparse_mmul(viennacl::coordinate_matrix<float> lhs, viennacl::compressed_matrix<float> rhs);


    // bridge to JNI functions

    // CSR sparse matrices BLAS-3
    // dense %*% dense
     void dense_dense_mmul(double* lhs, long lhs_rows, long lhs_cols, double* rhs, long rhs_rows, long rhs_cols, double* result)
//    float* dense_dense_mmul(double* lhs, long lhs_rows, long lhs_cols, double* rhs, long rhs_rows, long rhs_cols);

    // dense %*% sparse (CSR)/(COO) with matrix from memory/std matrix preperation
//    void dense_sparse_mmul(double* lhs, long lhs_rows, long lhs_cols, std::vector<std::vector<double> > rhs, bool isRhsCSR);
//    float* dense_sparse_mmul(float * lhs, long lhs_rows, long lhs_cols, std::vector<std::vector<float> > rhs, bool isRhsCSR);

    // sparse (CSR)/(COO) %*% dense
//    void  sparse_dense_mmul(std::vector<std::vector<double> > lhs, double* rhs, long rhs_rows, long rhs_cols);
//    float*  sparse_dense_mmul(std::vector<std::vector<float> > lhs, float* rhs, long rhs_rows, long rhs_cols);

    // sparse (CSR)/(COO) %*% sparse (CSR)/(COO)
//    std::vector<std::vector<double> > sparse_sparse_mmul(std::vector<std::vector<double> > lhs, bool isLhsCsr, std::vector<std::vector<double> >, bool isRhsCsr);
//    std::vector<std::vector<double> > sparse_sparse_mmul(std::vector<std::vector<float> > lhs, bool isLhsCsr, std::vector<std::vector<float> > rhs,  bool isRhsCsr);
}