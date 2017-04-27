/**
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


package org.apache.mahout.cuda

import jcuda._
import jcuda.jcublas._

final class DenseMatrix {

  var vals = new jcuda.Pointer()

  var trans = CUSPARSE_OPERATION_NON_TRANSPOSE  // use dense
  var descr = new cusparseMatDescr()

  var nrows = 0
  var ncols = 0


  def this(ctx: Context, nrow: Int, ncol: Int) {
    this()

    nrows = nrow
    ncols = ncol

    nonz = nonzeros
    if (nonzeros > 0) {
      cudaMalloc(vals, nonzeros*jcuda.Sizeof.DOUBLE)
    }

    // create and setup matrix descriptor
    cusparseCreateMatDescr(descr)
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO)
  }

  //def set (...)

  def close() {
    cudaFree(row_ptr)
    if (nonz > 0) {
      cudaFree(col_ind)
      cudaFree(vals)
    }
  }
}

