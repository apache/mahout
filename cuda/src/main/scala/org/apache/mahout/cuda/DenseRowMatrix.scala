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

import jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO
import jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL
import jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE

final class DenseRowMatrix {

  var vals = new jcuda.Pointer()

  var trans = CUBLAS_OP_N
  var descr = new CUDA_ARRAY_DESCRIPTOR()

  var nrows = 0
  var ncols = 0

  /**
    * Initalize empty Dense Matrix
    * @param ctx
    * @param nrow
    * @param ncol
    */
  def this(ctx: Context, nrow: Int, ncol: Int) {
    this()

    nrows = nrow
    ncols = ncol

    cublasAlloc(nrows * ncols * jcuda.Sizeof.DOUBLE, vals)

    // create and setup matrix descriptor
    // Todo: do we want these? for dense %*% sparse?
    //cusblasCreateMatDescr(descr)
    //cusblasSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL)
    //(descr, CUSPARSE_INDEX_BASE_ZERO)
    allocate()


  }


  cudaMemcpy(row_ptr, jcuda.Pointer.to(rowJumper), (nrow+1)*jcuda.Sizeof.INT, cudaMemcpyHostToDevice)


  def set ()

  def close() {
    cublasFree(vals)
  }
}

