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
import jcuda.jcublas.JCublas._
import jcuda.jcusparse.JCusparse._
import jcuda.jcusparse._
import jcuda.jcusparse.cusparseIndexBase.CUSPARSE_INDEX_BASE_ZERO
import jcuda.jcusparse.cusparseMatrixType.CUSPARSE_MATRIX_TYPE_GENERAL
import jcuda.jcusparse.cusparseOperation.CUSPARSE_OPERATION_NON_TRANSPOSE

import jcuda.runtime._
import jcuda.runtime.JCuda
import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._


final class DenseRowMatrix {

  var vals = new jcuda.Pointer()

  // default = not transposed.
  var trans = 'n'
  var descr = new jcuda.driver.CUDA_ARRAY_DESCRIPTOR()

  var nrows = 0
  var ncols = 0

  var context = new Context

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
    context = ctx

    // allocate empty space on the GPU
    cublasAlloc(nrows * ncols * jcuda.Sizeof.DOUBLE, jcuda.Sizeof.DOUBLE , vals)

    // create and setup matrix descriptor
    // Todo: do we want these? for dense %*% sparse?
    // JCuda.cublasCreateMatDescr(descr)
    // cublasSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL)
    // cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO)

  }

  /**
    * Initalize a new Dense matrix with data supplied
    * @param ctx
    * @param nrow
    * @param ncol
    * @param data double[][] of Dense array elements
    */
  def this(ctx: Context, nrow: Int, ncol: Int, data: Array[Array[Double]]) {
    this()

    nrows = nrow
    ncols = ncol
    context = ctx

    // allocate empty space on the GPU
    cublasAlloc(nrows * ncols * jcuda.Sizeof.DOUBLE, jcuda.Sizeof.DOUBLE, vals)

    // create and setup matrix descriptor
    // Todo: do we want these? for dense %*% sparse?
    // cusblasCreateMatDescr(descr)
    // cusblasSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL)
    // cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO)

    cudaMemcpy(vals, jcuda.Pointer.to(data.toList.flatten.toArray),
      (nrow) * (ncol) * jcuda.Sizeof.DOUBLE,
      cudaMemcpyHostToDevice)
  }

  /** Constructor with values on the device already.
    *
    * @param ctx
    * @param nrow
    * @param ncol
    * @param data
    */
  def this(ctx: Context, nrow: Int, ncol: Int, data: Pointer) {
    this()

    nrows = nrow
    ncols = ncol
    context = ctx

    vals = data

    // create and setup matrix descriptor
    // Todo: do we need these? for dense %*% sparse?
    //cusblasCreateMatDescr(descr)
    //cusblasSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL)
    //cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO)

  }

  /**Set values with an 2d Array
    *
    * @param data
    */
  def set (data: Array[Array[Double]]): Unit = {
    // Allocate row-major
    cublasAlloc(data.length * data(0).length * jcuda.Sizeof.DOUBLE,
      jcuda.Sizeof.DOUBLE, vals)
    cudaMemcpy(vals, jcuda.Pointer.to(data.toList.flatten.toArray),
      data.length * data(0).length * jcuda.Sizeof.DOUBLE,
      cudaMemcpyHostToDevice)
  }

  /** Set values with a pointer that is alredy created
    *
    * @param data
    */
  def set (data: Pointer): Unit = {
    vals = data
  }

  private[cuda] def flatten2dArray(arr2d: Array[Array[Double]]): Array[Double] = {
    arr2d.toList.flatten.toArray
  }

  def close() {
    cublasFree(vals)
  }
}

