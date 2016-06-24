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
#include "vcl_blas3.h"

#include "viennacl/matrix.hpp"
#include "viennacl/compressed_matrix.hpp"
#include "viennacl/coordinate_matrix.hpp"
#include "viennacl/linalg/prod.hpp"
#include "forwards.h"
#include "mem_handle.hpp"


using namespace mmul;
//
//    // currently row-major only
//
//    // Dense BLAS-3
//    // dense %*% dense
//    viennacl::matrix<double> dense_dense_mmul(viennacl::matrix<double> lhs, viennacl::matrix<double> rhs) {
//     // viennacl::matrix<double> = new viennacl(lhs)
//      return viennacl::linalg::prod_impl(lhs, rhs);
//    }
//    viennacl::matrix<float> dense_dense_mmul(viennacl::matrix<float> lhs, viennacl::matrix<float> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//
//    // CSR sparse matrices BLAS-3
//    // dense %*% sparse (CSR)
//    viennacl::matrix<double> dense_sparse_mmul(viennacl::matrix<double> lhs, viennacl::compressed_matrix<double> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//    viennacl::matrix<float> dense_sparse_mmul(viennacl::matrix<float> lhs, viennacl::compressed_matrix<float> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//
//    // sparse (CSR) %*% dense
//    viennacl::matrix<double> sparse_dense_mmul(viennacl::compressed_matrix<double> lhs, viennacl::matrix<double> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//    viennacl::matrix<float> sparse_dense_mmul(viennacl::compressed_matrix<float> lhs, viennacl::matrix<float> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//
//    // sparse(CSR) %*% sparse (CSR)
//    viennacl::compressed_matrix<double> sparse_sparse_mmul(viennacl::compressed_matrix<double> lhs, viennacl::compressed_matrix<double> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//    viennacl::compressed_matrix<float> sparse_sparse_mmul(viennacl::compressed_matrix<float> lhs, viennacl::compressed_matrix<float> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//
//    // sparse (CSR) %*% sparse (COO)
//    viennacl::compressed_matrix<double> sparse_sparse_mmul(viennacl::compressed_matrix<double> lhs, viennacl::coordinate_matrix<double> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//    viennacl::compressed_matrix<float> sparse_sparse_mmul(viennacl::compressed_matrix<float> lhs, viennacl::coordinate_matrix<float> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//
//
//    // COO sparse matrices BLAS-3
//    // dense %*% sparse (COO)
//    viennacl::matrix<double> dense_sparse_mmul(viennacl::matrix<double> lhs, viennacl::coordinate_matrix<double> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//    viennacl::matrix<float> dense_sparse_mmul(viennacl::matrix<float> lhs, viennacl::coordinate_matrix<float> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//
//    // sparse (COO) %*% dense
//    viennacl::matrix<double> sparse_dense_mmul(viennacl::coordinate_matrix<double> lhs, viennacl::matrix<double> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//    viennacl::matrix<float> sparse_dense_mmul(viennacl::coordinate_matrix<float> lhs, viennacl::matrix<float> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//
//    // sparse (COO) %*% sparse (COO)
//    viennacl::coordinate_matrix<double> sparse_sparse_mmul(viennacl::coordinate_matrix<double> lhs, viennacl::coordinate_matrix<double> rhs) {
//     return viennacl::linalg::prod(lhs, rhs);
//    }
//    viennacl::coordinate_matrix<float> sparse_sparse_mmul(viennacl::coordinate_matrix<float> lhs, viennacl::coordinate_matrix<float> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//
//    // sparse (COO) %*% sparse (CSR)
//    viennacl::coordinate_matrix<double> sparse_sparse_mmul(viennacl::coordinate_matrix<double> lhs, viennacl::compressed_matrix<double> rhs) {
//      return viennacl::linalg::prod(lhs, rhs);
//    }
//    viennacl::coordinate_matrix<float> sparse_sparse_mmul(viennacl::coordinate_matrix<float> lhs, viennacl::compressed_matrix<float> rhs) {
//       return viennacl::linalg::prod(lhs, rhs);
//    }

    // bridge to JNI functions

    // dense matrices BLAS-3
    // dense %*% dense
    void dense_dense_mmul(double* lhs, long lhs_rows, long lhs_cols, double* rhs, long rhs_rows, long rhs_cols, double* result) {
       viennacl::matrix<double> mx_a(lhs, viennacl::memory_types::main_memory, lhs_rows, lhs_cols);
       viennacl::matrix<double> mx_b(rhs, viennacl::memory_types::main_memory, rhs_rows, rhs_cols);

       // resulting matrix
       viennacl::matrix<double> result(lhs_rows, rhs_cols);

       result = viennacl::linalg::prod(mx_a, mx_b);

    }

//    float* dense_dense_mmul(float* lhs, float* rhs) {
//      viennacl::matrix mx_a = new viennacl::matrix<double>(lhs, viennacl::memory_types::main_memory, lhs_rows, lhs_cols);
//      viennacl::matrix mx_b = new viennacl::matrix<double>(rhs, viennacl::memory_types::main_memory, rhs_rows, rhs_cols);
//
//      viennacl::matrix res = dense_dense_mmul(mx_a, mx_b);
//
//      return res.
//
//    }


