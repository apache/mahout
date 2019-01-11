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
package org.apache.mahout.viennacl

import java.nio._

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.mahout.math.backend.incore._
import scala.collection.JavaConversions._
import org.apache.mahout.viennacl.opencl.javacpp.{CompressedMatrix, Context, DenseRowMatrix, Functions, VCLVector}
import org.apache.mahout.viennacl.opencl.javacpp.Context
import org.bytedeco.javacpp.{DoublePointer, IntPointer}



package object opencl {

  type IntConvertor = Int => Int

  def toVclDenseRM(src: Matrix, vclCtx: Context = new Context(Context.MAIN_MEMORY)): DenseRowMatrix = {
    vclCtx.memoryType match {
      case Context.MAIN_MEMORY ⇒
        val vclMx = new DenseRowMatrix(
          data = repackRowMajor(src, src.nrow, src.ncol),
          nrow = src.nrow,
          ncol = src.ncol,
          ctx = vclCtx
        )
        vclMx
      case _ ⇒
        val vclMx = new DenseRowMatrix(src.nrow, src.ncol, vclCtx)
        fastCopy(src, vclMx)
        vclMx
    }
  }


  /**
    * Convert a dense row VCL matrix to mahout matrix.
    *
    * @param src
    * @return
    */
  def fromVclDenseRM(src: DenseRowMatrix): Matrix = {
    val nrowIntern = src.internalnrow
    val ncolIntern = src.internalncol

    // A technical debt here:

    // We do double copying here, this is obviously suboptimal, but hopefully we'll compensate
    // this with gains from running superlinear algorithms in VCL.
    val dbuff = new DoublePointer(nrowIntern * ncolIntern)
    Functions.fastCopy(src, dbuff)
    var srcOffset = 0
    val ncol = src.ncol
    val rows = for (irow ← 0 until src.nrow) yield {

      val rowvec = new Array[Double](ncol)
      dbuff.position(srcOffset).get(rowvec)

      srcOffset += ncolIntern
      rowvec
    }

    // Always! use shallow = true to avoid yet another copying.
    new DenseMatrix(rows.toArray, true)
  }

  def fastCopy(mxSrc: Matrix, dst: DenseRowMatrix) = {
    val nrowIntern = dst.internalnrow
    val ncolIntern = dst.internalncol

    assert(nrowIntern >= mxSrc.nrow && ncolIntern >= mxSrc.ncol)

    val rmajorData = repackRowMajor(mxSrc, nrowIntern, ncolIntern)
    Functions.fastCopy(rmajorData, new DoublePointer(rmajorData).position(rmajorData.limit()), dst)

    rmajorData.close()
  }

  private def repackRowMajor(mx: Matrix, nrowIntern: Int, ncolIntern: Int): DoublePointer = {

    assert(mx.nrow <= nrowIntern && mx.ncol <= ncolIntern)

    val dbuff = new DoublePointer(nrowIntern * ncolIntern)

    mx match {
      case dm: DenseMatrix ⇒
        val valuesF = classOf[DenseMatrix].getDeclaredField("values")
        valuesF.setAccessible(true)
        val values = valuesF.get(dm).asInstanceOf[Array[Array[Double]]]
        var dstOffset = 0
        for (irow ← 0 until mx.nrow) {
          val rowarr = values(irow)
          dbuff.position(dstOffset).put(rowarr, 0, rowarr.size min ncolIntern)
          dstOffset += ncolIntern
        }
        dbuff.position(0)
      case _ ⇒
        // Naive copying. Could be sped up for a DenseMatrix. TODO.
        for (row ← mx) {
          val dstOffset = row.index * ncolIntern
          for (el ← row.nonZeroes) dbuff.put(dstOffset + el.index, el)
        }
    }

    dbuff
  }

  /**
    *
    * @param mxSrc
    * @param ctx
    * @return
    */
  def toVclCmpMatrixAlt(mxSrc: Matrix, ctx: Context): CompressedMatrix = {

    // use repackCSR(matrix, ctx) to convert all ints to unsigned ints if Context is Ocl
    // val (jumpers, colIdcs, els) = repackCSRAlt(mxSrc)
    val (jumpers, colIdcs, els) = repackCSR(mxSrc, ctx)

    val compMx = new CompressedMatrix(mxSrc.nrow, mxSrc.ncol, els.capacity().toInt, ctx)
    compMx.set(jumpers, colIdcs, els, mxSrc.nrow, mxSrc.ncol, els.capacity().toInt)
    compMx
  }

  private def repackCSRAlt(mx: Matrix): (IntPointer, IntPointer, DoublePointer) = {
    val nzCnt = mx.map(_.getNumNonZeroElements).sum
    val jumpers = new IntPointer(mx.nrow + 1L)
    val colIdcs = new IntPointer(nzCnt + 0L)
    val els = new DoublePointer(nzCnt)
    var posIdx = 0

    var sortCols = false

    // Row-wise loop. Rows may not necessarily come in order. But we have to have them in-order.
    for (irow ← 0 until mx.nrow) {

      val row = mx(irow, ::)
      jumpers.put(irow.toLong, posIdx)

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
              colIdcs.put(posIdx.toLong, index)
              els.put(posIdx.toLong, v)
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
              els.put(posIdx, el)
              colIdcs.put(posIdx.toLong, index)
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

    jumpers.put(mx.nrow.toLong, nzCnt)

    (jumpers, colIdcs, els)
  }

  // same as repackCSRAlt except converts to jumpers, colIdcs to unsigned ints before setting
  private def repackCSR(mx: Matrix, context: Context): (IntPointer, IntPointer, DoublePointer) = {
    val nzCnt = mx.map(_.getNumNonZeroElements).sum
    val jumpers = new IntPointer(mx.nrow + 1L)
    val colIdcs = new IntPointer(nzCnt + 0L)
    val els = new DoublePointer(nzCnt)
    var posIdx = 0

    var sortCols = false

    def convertInt: IntConvertor = if(context.memoryType == Context.OPENCL_MEMORY) {
      int2cl_uint
    } else {
      i: Int => i: Int
    }

    // Row-wise loop. Rows may not necessarily come in order. But we have to have them in-order.
    for (irow ← 0 until mx.nrow) {

      val row = mx(irow, ::)
      jumpers.put(irow.toLong, posIdx)

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
            .toIndexedSeq
            // Sort by column index
            .sortBy(_._1)
            // Flush to the CSR buffers.
            .foreach { case (index, v) ⇒
            // convert to cl_uint if context is OCL
            colIdcs.put(posIdx.toLong, convertInt(index))
            els.put(posIdx.toLong, v)
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
              els.put(posIdx, el)
              // convert to cl_uint if context is OCL
              colIdcs.put(posIdx.toLong, convertInt(index))
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

    // convert to cl_uint if context is OCL
    jumpers.put(mx.nrow.toLong, convertInt(nzCnt))

    (jumpers, colIdcs, els)
  }



  def fromVclCompressedMatrix(src: CompressedMatrix): Matrix = {
    val m = src.size1
    val n = src.size2
    val NNz = src.nnz

    val row_ptr_handle = src.handle1
    val col_idx_handle = src.handle2
    val element_handle = src.handle

    val row_ptr = new IntPointer((m + 1).toLong)
    val col_idx = new IntPointer(NNz.toLong)
    val values = new DoublePointer(NNz.toLong)

    Functions.memoryReadInt(row_ptr_handle, 0, (m + 1) * 4, row_ptr, false)
    Functions.memoryReadInt(col_idx_handle, 0, NNz * 4, col_idx, false)
    Functions.memoryReadDouble(element_handle, 0, NNz * 8, values, false)

    val rowPtr = row_ptr.asBuffer()
    val colIdx = col_idx.asBuffer()
    val vals = values.asBuffer()

    rowPtr.rewind()
    colIdx.rewind()
    vals.rewind()


    val srMx = new SparseRowMatrix(m, n)

    // read the values back into the matrix
    var j = 0
    // row wise, copy any non-zero elements from row(i-1,::)
    for (i <- 1 to m) {
      // for each nonzero element, set column col(idx(j) value to vals(j)
      while (j < rowPtr.get(i)) {
        srMx(i - 1, colIdx.get(j)) = vals.get(j)
        j += 1
      }
    }
    srMx
  }

  def toVclVec(vec: Vector, ctx: Context): VCLVector = {

    vec match {
      case vec: DenseVector => {
        val valuesF = classOf[DenseVector].getDeclaredField("values")
        valuesF.setAccessible(true)
        val values = valuesF.get(vec).asInstanceOf[Array[Double]]
        val el_ptr = new DoublePointer(values.length.toLong)
        el_ptr.put(values, 0, values.length)

        new VCLVector(el_ptr, ctx.memoryType, values.length)
      }

      case vec: SequentialAccessSparseVector => {
        val it = vec.iterateNonZero
        val size = vec.size()
        val el_ptr = new DoublePointer(size.toLong)
        while (it.hasNext) {
          val el: Vector.Element = it.next
          el_ptr.put(el.index, el.get())
        }
        new VCLVector(el_ptr, ctx.memoryType, size)
      }

      case vec: RandomAccessSparseVector => {
        val it = vec.iterateNonZero
        val size = vec.size()
        val el_ptr = new DoublePointer(size.toLong)
        while (it.hasNext) {
          val el: Vector.Element = it.next
          el_ptr.put(el.index, el.get())
        }
        new VCLVector(el_ptr, ctx.memoryType, size)
      }
      case _ => throw new IllegalArgumentException("Vector sub-type not supported.")
    }

  }

  def fromVClVec(vclVec: VCLVector): Vector = {
    val size = vclVec.size
    val element_handle = vclVec.handle
    val ele_ptr = new DoublePointer(size)
    Functions.memoryReadDouble(element_handle, 0, size * 8, ele_ptr, false)

    // for now just assume its dense since we only have one flavor of
    // VCLVector
    val mVec = new DenseVector(size)
    for (i <- 0 until size) {
      mVec.setQuick(i, ele_ptr.get(i + 0L))
    }

    mVec
  }


  // TODO: Fix this?  cl_uint must be an unsigned int per each machine's representation of such.
  // this is currently not working anyways.
  // cl_uint is needed for OpenCl sparse Buffers
  // per https://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/scalarDataTypes.html
  // it is simply an unsigned int, so strip the sign.
  def int2cl_uint(i: Int): Int = {
    ((i >>> 1) << 1) + (i & 1)
  }


}
