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

// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.

package org.apache.mahout.cuda

import java.nio._

import jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO
import jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL
import jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE
import jcuda.jcusparse.JCusparse._
import jcuda.jcusparse._
import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._

final class CompressedMatrix {

  var row_ptr = new jcuda.Pointer()
  var col_ind = new jcuda.Pointer()
  var vals = new jcuda.Pointer()

  var trans = CUSPARSE_OPERATION_NON_TRANSPOSE
  var descr = new cusparseMatDescr()

  var nrows = 0
  var ncols = 0
  var nonz = 0

  def this(ctx: Context, nrow: Int, ncol: Int, nonzeros: Int = 0) {
    this()

    nrows = nrow
    ncols = ncol
    cudaMalloc(row_ptr, (nrow+1)*jcuda.Sizeof.INT)

    nonz = nonzeros
    if (nonzeros > 0) {
      cudaMalloc(col_ind, nonzeros*jcuda.Sizeof.INT)
      cudaMalloc(vals, nonzeros*jcuda.Sizeof.DOUBLE)
    }

    // create and setup matrix descriptor
    cusparseCreateMatDescr(descr)
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO)
  }

  def set(rowJumper: Array[Int],
          colIndices: Array[Int],
          elements: Array[Double],
          nrow: Int,
          ncol: Int,
          nonzeros: Int) {
    cudaMemcpy(row_ptr, jcuda.Pointer.to(rowJumper), (nrow+1)*jcuda.Sizeof.INT, cudaMemcpyHostToDevice)
    cudaMemcpy(col_ind, jcuda.Pointer.to(colIndices), (nonzeros)*jcuda.Sizeof.INT, cudaMemcpyHostToDevice)
    cudaMemcpy(vals, jcuda.Pointer.to(elements), (nonzeros)*jcuda.Sizeof.DOUBLE, cudaMemcpyHostToDevice)
  }

  def close() {
    cudaFree(row_ptr)
    if (nonz > 0) {
      cudaFree(col_ind)
      cudaFree(vals)
    }
  }
}

