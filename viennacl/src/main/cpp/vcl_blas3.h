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

     // bridge to JNI functions

    // CSR sparse matrices BLAS-3
    // dense %*% dense
     void dense_dense_mmul(double* lhs, long lhs_rows, long lhs_cols, double* rhs, long rhs_rows, long rhs_cols, double* result)

    // dense %*% sparse (CSR)/ with matrix from memory/std matrix preperation
//    void dense_sparse_mmul(double* lhs, long lhs_rows, long lhs_cols, std::vector<std::vector<double> > rhs);

    // sparse (CSR) %*% dense
//     void  sparse_dense_mmul(std::vector<std::vector<double> > lhs, double* rhs, long rhs_rows, long rhs_cols, double* result);

     // sparse (CSR) %*% sparse (CSR)
//     void  sparse_sparse_mmul(std::vector<std::vector<double> >* lhs, std::vector<std::vector<double> >* rhs, std::vector<std::vector<double> >* res);
}