/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.

package org.apache.mahout

import java.nio._

import org.apache.mahout.logging._
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.mahout.math.backend.incore._
import scala.collection.JavaConversions._

import jcuda.runtime.JCuda._
import jcuda.runtime.cudaMemcpyKind._

import jcuda._
import jcuda.jcublas._




package object cuda {

  private implicit val log = getLog(GPUMMul.getClass)

  /** Copy cuda data back into a Mahout DenseMatrix
    *
    * @param src a (flattened) 2D cuda array
    * @return A Mahout DenseMatrix
    */
  def fromCUDADenseRM(src: DenseRowMatrix): Matrix = {

    val nrowIntern = src.nrows
    val ncolIntern = src.ncols

    var dbuff = new Pointer()

    // again will be doullbe copying.. consider copying directly from cuda memory
    // into each row..
    val jvmData = Array.ofDim[Double](nrowIntern,ncolIntern) //Double](nrowIntern * ncolIntern)
    val cudaData = new Array[Double](nrowIntern * ncolIntern)
    cudaMemcpy(jcuda.Pointer.to(cudaData), src.vals, (nrowIntern * ncolIntern)*jcuda.Sizeof.DOUBLE, cudaMemcpyDeviceToHost)

    // We could speed this up by doing a transpose here
    // assuming that the matrix is in columnMajor format
    // TODO: consider this getting late so make it work now.
    var srcOffset = 0
    val ncol = src.ncols
    val rows = for (irow ← 0 until src.nrows) yield {

      val rowvec = new Array[Double](ncol)
      System.arraycopy(cudaData, srcOffset , rowvec , 0 , ncol)
      srcOffset += ncolIntern
      rowvec
    }

    // Always! use shallow = true to avoid yet another copying.
    // even another from viennacl :)
    new DenseMatrix(rows.toArray, true)
  }

  /**
    * Convert from Mahout DenseMatrix to matrix
    * @param src
    * @return
    */
  def toCudaDenseRM(src: Matrix, ctx: Context): cuda.DenseRowMatrix = {

        val valuesF = classOf[DenseMatrix].getDeclaredField("values")
        valuesF.setAccessible(true)
        val values = valuesF.get(src).asInstanceOf[Array[Array[Double]]]
        val cudaMx = new cuda.DenseRowMatrix(ctx, src.nrow, src.ncol, values)

        cudaMx
  }


  // TODO replace this with repackColumnMajor or use a different dgemm algorithm?
  // Most Mahout in-core matrices are row-major and we're using CSR so we may need to see
  // if JCuda is using an optimal csr/RowMajor DGEMM algortithm.
  // TODO: check with NS on this
//  private[cuda] def repackRowMajor(mx: Matrix, nrowIntern: Int, ncolIntern: Int): Array[Double] = {
//
//    assert(mx.nrow <= nrowIntern && mx.ncol <= ncolIntern)
//
//    val dbuff = Array.ofDim[Double](nrowIntern * ncolIntern)
//
//    mx match {
//      case dm: DenseMatrix ⇒
//        val valuesF = classOf[DenseMatrix].getDeclaredField("values")
//        valuesF.setAccessible(true)
//        val values = valuesF.get(dm).asInstanceOf[Array[Array[Double]]]
//        var dstOffset = 0
//        for (irow ← 0 until mx.nrow) {
//          val rowarr = values(irow)
//          //dbuff.position(dstOffset).put(rowarr, 0, rowarr.size min ncolIntern)
//          System.arraycopy(rowarr, 0, dbuff, dstOffset, rowarr.size min ncolIntern)
//          dstOffset += ncolIntern
//        }
//      case _ ⇒
//        // Naive copying. Could be sped up for a DenseMatrix. TODO.
//        for (row ← mx) {
//          val dstOffset = row.index * ncolIntern
//          for (el ← row.nonZeroes) dbuff.update(dstOffset + el.index) = el
//        }
//    }
//  }

  /**
    *
    * @param mxSrc
    * @param ctx
    * @return
    */
  def toCudaCmpMatrix(mxSrc: Matrix, ctx: Context): CompressedMatrix = {
    val (jumpers, colIdcs, els) = repackCSR(mxSrc)
    val compMx = new CompressedMatrix(ctx, mxSrc.nrow, mxSrc.ncol, els.length)
    compMx.set(jumpers, colIdcs, els, mxSrc.nrow, mxSrc.ncol, els.length)
    compMx
  }

  private def repackCSR(mx: Matrix): (Array[Int], Array[Int], Array[Double]) = {
    val nzCnt = mx.map(_.getNumNonZeroElements).sum
    val jumpers = new Array[Int](mx.nrow + 1)
    val colIdcs = new Array[Int](nzCnt + 0)
    val els = new Array[Double](nzCnt)
    var posIdx = 0

    var sortCols = false

    // Row-wise loop. Rows may not necessarily come in order. But we have to have them in-order.
    for (irow ← 0 until mx.nrow) {

      val row = mx(irow, ::)
      jumpers(irow) = posIdx

      // Remember row start index in case we need to restart conversion of this row if out-of-order
      // column index is detected
      val posIdxStart = posIdx

      // Retry loop: normally we are done in one pass thru it unless we need to re-run it because
      // out-of-order column was detected.
      var done = false
      while (!done) {

        // Is the sorting mode on?
        if (sortCols) {

          // Sorting of column indices is on. So do it.
          row.nonZeroes()
            // Need to convert to a strict collection out of iterator
            .map(el ⇒ el.index → el.get)
            // Sorting requires Sequence api
            .toSeq
            // Sort by column index
            .sortBy(_._1)
            // Flush to the CSR buffers.
            .foreach { case (index, v) ⇒
              colIdcs(posIdx) = index
              els(posIdx) = v
              posIdx += 1
            }

          // Never need to retry if we are already in the sorting mode.
          done = true

        } else {

          // Try to run unsorted conversion here, switch lazily to sorted if out-of-order column is
          // detected.
          var lastCol = 0
          val nzIter = row.nonZeroes().iterator()
          var abortNonSorted = false

          while (nzIter.hasNext && !abortNonSorted) {

            val el = nzIter.next()
            val index = el.index

            if (index < lastCol) {

              // Out of order detected: abort inner loop, reset posIdx and retry with sorting on.
              abortNonSorted = true
              sortCols = true
              posIdx = posIdxStart

            } else {

              // Still in-order: save element and column, continue.
              els(posIdx) = el
              colIdcs(posIdx) = index
              posIdx += 1

              // Remember last column seen.
              lastCol = index
            }
          } // inner non-sorted

          // Do we need to re-run this row with sorting?
          done = !abortNonSorted

        } // if (sortCols)

      } // while (!done) retry loop

    } // row-wise loop

    // Make sure Mahout matrix did not cheat on non-zero estimate.
    assert(posIdx == nzCnt)

    jumpers(mx.nrow) = nzCnt

    (jumpers, colIdcs, els)
  }

  /**
    * Dense %*% Dense
    * @param a
    * @param b
    * @param ctx
    * @return
    */
  def prod(a: DenseRowMatrix, b: DenseRowMatrix, ctx: Context): DenseRowMatrix = {
    val m = a.nrows
    val n = b.ncols
    val k = b.nrows

   // val d_A = valuesF.get(a).asInstanceOf[Array[Array[Double]]]


    val c: DenseRowMatrix = new DenseRowMatrix(ctx, m, n)
    val d_C: Pointer = new Pointer()
    cudaMalloc(c.vals,  m * n * jcuda.Sizeof.DOUBLE)

    //C = alpha * op(A) * op(B) + beta * C,
    //where op(X) = X or op(X) = transpose(X),
    // using transpose here because Mahout Matrices in general
    // are row-major,  hardcoding this for now..
    JCublas.cublasDgemm('t', 't', m, n, k,
      1.0d,    // alpha
      a.vals, m, // A, lda
      b.vals, k, // B , ldb
      0.0d,    // beta
      d_C,     // pointer to results
      n)    // todo: check on this are we correct here?

     // set the data of c to the results
     // may need to allocate data here or the other side.
     c.set(d_C)
    c
  }

  /**
    * Sparse %*% Sparse
    * @param a
    * @param b
    * @param ctx
    * @return
    */
  def prod(a: CompressedMatrix, b: CompressedMatrix, ctx: Context): CompressedMatrix = {
    var m = a.nrows
    var n = b.ncols
    var k = b.nrows

    var c: CompressedMatrix = new CompressedMatrix(ctx, m, n)

    // step 1: compute nnz count
    var nnzC = new Array[Int](1)
    nnzC(0) = 0
    cusparseXcsrgemmNnz(ctx.sparseHandle, a.trans, b.trans, m, n, k,
      a.descr, a.nonz, a.row_ptr, a.col_ind,
      b.descr, b.nonz, b.row_ptr, b.col_ind,
      c.descr, c.row_ptr, jcuda.Pointer.to(nnzC))
    c.nonz = nnzC(0)
    if (c.nonz == 0) {
      var baseC = new Array[Int](1)
      cudaMemcpy(jcuda.Pointer.to(nnzC), c.row_ptr.withByteOffset(m * jcuda.Sizeof.INT), jcuda.Sizeof.INT, cudaMemcpyDeviceToHost)
      cudaMemcpy(jcuda.Pointer.to(baseC), c.row_ptr, jcuda.Sizeof.INT, cudaMemcpyDeviceToHost)
      c.nonz = nnzC(0) - baseC(0)
    }

    // step 2: allocate and compute matrix product
    cudaMalloc(c.col_ind, jcuda.Sizeof.INT * c.nonz);
    cudaMalloc(c.vals, jcuda.Sizeof.DOUBLE * c.nonz);
    cusparseDcsrgemm(ctx.sparseHandle, a.trans, b.trans, m, n, k,
      a.descr, a.nonz,
      a.vals, a.row_ptr, a.col_ind,
      b.descr, b.nonz,
      b.vals, b.row_ptr, b.col_ind,
      c.descr,
      c.vals, c.row_ptr, c.col_ind);
    c
  }


  def fromCudaCmpMatrix(src: CompressedMatrix): Matrix = {
    val m = src.nrows
    val n = src.ncols
    val NNz = src.nonz

    log.debug("m=" + m.toString() + ", n=" + n.toString() + ", nnz=" + NNz.toString())

    val row_ptr = new Array[Int](m + 1)
    val col_idx = new Array[Int](NNz)
    val values = new Array[Double](NNz)

    cudaMemcpy(jcuda.Pointer.to(row_ptr), src.row_ptr, (m+1)*jcuda.Sizeof.INT, cudaMemcpyDeviceToHost)
    cudaMemcpy(jcuda.Pointer.to(col_idx), src.col_ind, (NNz)*jcuda.Sizeof.INT, cudaMemcpyDeviceToHost)
    cudaMemcpy(jcuda.Pointer.to(values), src.vals, (NNz)*jcuda.Sizeof.DOUBLE, cudaMemcpyDeviceToHost)

    val srMx = new SparseRowMatrix(m, n)

    // read the values back into the matrix
    var j = 0
    // row wise, copy any non-zero elements from row(i-1,::)
    for (i <- 1 to m) {
      // for each nonzero element, set column col(idx(j) value to vals(j)
      while (j < row_ptr(i)) {
        srMx(i - 1, col_idx(j)) = values(j)
        j += 1
      }
    }
    srMx
  }

}
