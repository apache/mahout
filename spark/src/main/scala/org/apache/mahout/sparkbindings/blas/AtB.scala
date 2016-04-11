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

package org.apache.mahout.sparkbindings.blas

import reflect.ClassTag
import collection._

import org.apache.mahout.logging._
import org.apache.mahout.math._
import drm._
import org.apache.mahout.sparkbindings.drm._
import org.apache.spark.rdd.RDD
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm.logical.OpAtB

import scala.collection.mutable.ArrayBuffer

object AtB {

  private final implicit val log = getLog(AtB.getClass)

  def atb[A: ClassTag](operator: OpAtB[A], srcA: DrmRddInput[A], srcB: DrmRddInput[A]): DrmRddInput[Int] = {
    atb_nograph_mmul(operator, srcA, srcB, operator.A.partitioningTag == operator.B.partitioningTag)
  }
  /**
   * The logic for computing A'B is pretty much map-side generation of partial outer product blocks
   * over co-grouped rows of A and B. If A and B are identically partitioned, we can just directly
   * zip all the rows. Otherwise, we need to inner-join them first.
   *
   */
  @deprecated("slow, will remove", since = "0.10.2")
  def atb_nograph[A: ClassTag](operator: OpAtB[A], srcA: DrmRddInput[A], srcB: DrmRddInput[A],
                               zippable: Boolean = false): DrmRddInput[Int] = {

    val rddA = srcA.asRowWise()
    val rddB = srcB.asRowWise()


    val prodNCol = operator.ncol
    val prodNRow = operator.nrow
    val aNRow = operator.A.nrow

    // Approximate number of final partitions. We take bigger partitions as our guide to number of
    // elements per partition. TODO: do it better.
    // Elements per partition, bigger of two operands.
    val epp = aNRow.toDouble * prodNRow / rddA.partitions.length max aNRow.toDouble * prodNCol /
      rddB.partitions.length

    // Number of partitions we want to converge to in the product. For now we simply extrapolate that
    // assuming product density and operand densities being about the same; and using the same element
    // per partition number in the product as the bigger of two operands.
    val numProductPartitions = (prodNCol.toDouble * prodNRow / epp).ceil.toInt

    if (log.isDebugEnabled) log.debug(s"AtB: #parts $numProductPartitions for $prodNRow x $prodNCol geometry.")

    val zipped = if (zippable) {

      log.debug("A and B for A'B are identically distributed, performing row-wise zip.")

      rddA.zip(other = rddB)

    } else {

      log.debug("A and B for A'B are not identically partitioned, performing inner join.")

      rddA.join(other = rddB, numPartitions = numProductPartitions).map({ case (key, (v1,
      v2)) => (key -> v1) -> (key -> v2)
      })
    }

    computeAtBZipped2(zipped, nrow = operator.nrow, ancol = operator.A.ncol, bncol = operator.B.ncol,
      numPartitions = numProductPartitions)
  }

  private[sparkbindings] def atb_nograph_mmul[A:ClassTag](operator:OpAtB[A], srcA: DrmRddInput[A], srcB:DrmRddInput[A], zippable:Boolean = false):DrmRddInput[Int] = {

    debug("operator mmul-A'B(Spark)")

    val prodNCol = operator.ncol
    val prodNRow = safeToNonNegInt(operator.nrow)
    val aNRow = safeToNonNegInt(operator.A.nrow)

    val rddA = srcA.asRowWise()
    val rddB = srcB.asRowWise()

    // Approximate number of final partitions. We take bigger partitions as our guide to number of
    // elements per partition. TODO: do it better.
    // Elements per partition, bigger of two operands.
    val epp = aNRow.toDouble * prodNRow / rddA.partitions.length max aNRow.toDouble * prodNCol /
      rddB.partitions.length

    // Number of partitions we want to converge to in the product. For now we simply extrapolate that
    // assuming product density and operand densities being about the same; and using the same element
    // per partition number in the product as the bigger of two operands.
    val numProductPartitions = (prodNCol.toDouble * prodNRow / epp).ceil.toInt min prodNRow

    if (log.isDebugEnabled) log.debug(s"AtB mmul: #parts $numProductPartitions for $prodNRow x $prodNCol geometry.")

    val zipped = if (zippable) {

      debug("mmul-A'B - zip: are identically distributed, performing row-wise zip.")

      val blockdRddA = srcA.asBlockified(operator.A.ncol)
      val blockdRddB = srcB.asBlockified(operator.B.ncol)

      blockdRddA

        // Zip
        .zip(other = blockdRddB)

        // Throw away the keys
        .map { case ((_, blockA), (_, blockB)) => blockA -> blockB}

    } else {

      debug("mmul-A'B: cogroup for non-identically distributed stuff.")

      // To take same route, we'll join stuff row-wise, blockify it here and then proceed with the
      // same computation path. Although it is possible we could shave off one shuffle here. TBD.

      rddA

        // Do full join. We can't get away with partial join because it is going to lose some rows
        // in case we have missing rows on either side.
        .cogroup(other = rddB, numPartitions = rddA.partitions.length max rddB.partitions.length )


        // Merge groups.
        .mapPartitions{iter =>

        val aRows = new ArrayBuffer[Vector](1000)
        val bRows = new ArrayBuffer[Vector](1000)

        // Populate hanging row buffs
        iter.foreach{case (_, (arowbag,browbag)) =>

          // Some up all vectors, if any, for a row. If we have > 1 that means original matrix had
          // non-uniquely keyed rows which is generally a matrix format inconsistency (should not
          // happen).
          aRows += (if (arowbag.isEmpty)
            new SequentialAccessSparseVector(prodNRow)
          else arowbag.reduce(_ += _))

          bRows += (if (browbag.isEmpty)
            new SequentialAccessSparseVector(prodNCol)
          else browbag.reduce(_ += _))
        }

        // Transform collection of vectors into blocks.
        val blockNRow = aRows.size
        assert(blockNRow == bRows.size)

        val aBlock:Matrix = new SparseRowMatrix(blockNRow, prodNRow, aRows.toArray)
        val bBlock:Matrix = new SparseRowMatrix(blockNRow, prodNCol, bRows.toArray)

        // Form pairwise result
        Iterator(aBlock -> bBlock)
      }
    }

    computeAtBZipped3(pairwiseRdd = zipped, nrow = prodNRow, ancol = prodNRow, bncol = aNRow,
      numPartitions = numProductPartitions)

  }
  /**
   * Compute, combine and accumulate outer products for every key. The incoming tuple structure
   * is (partNo, (vecA, vecB)), so for every `partNo` we compute an outer product of the form {{{
   *   vecA cross vecB
   * }}}
   * @param pairwiseRdd
   * @return
   */
  @deprecated("slow, will remove", since = "0.10.2")
  private[sparkbindings] def combineOuterProducts(pairwiseRdd: RDD[(Int, (Vector, Vector))], numPartitions: Int) = {

    pairwiseRdd

      // Reduce individual partitions
      .combineByKey(createCombiner = (t: (Vector, Vector)) => {

      val vecA = t._1
      val vecB = t._2

      // Create partition accumulator. Generally, summation of outer products probably calls for
      // dense accumulators. However, let's assume extremely sparse cases are still possible, and
      // by default assume any sparse case is an extremely sparse case. May need to tweak further.
      val mxC: Matrix = if (!vecA.isDense && !vecB.isDense)
        new SparseRowMatrix(vecA.length, vecB.length)
      else
        new DenseMatrix(vecA.length, vecB.length)

      // Add outer product of arow and bRowFrag to mxC
      addOuterProduct(mxC, vecA, vecB)

    }, mergeValue = (mxC: Matrix, t: (Vector, Vector)) => {
      // Merge of a combiner with another outer product fragment.
      val vecA = t._1
      val vecB = t._2

      addOuterProduct(mxC, vecA, vecB)

    }, mergeCombiners = (mxC1: Matrix, mxC2: Matrix) => {

      // Merge of 2 combiners.
      mxC1 += mxC2

    }, numPartitions = numPartitions)
  }

  private[sparkbindings] def computeAtBZipped3[A: ClassTag](pairwiseRdd: RDD[(Matrix, Matrix)], nrow: Int,
                                                            ancol: Int, bncol: Int, numPartitions: Int) = {

    val ranges = computeEvenSplits(nrow, numPartitions)

    val rdd = pairwiseRdd.flatMap{ case (blockA, blockB) ⇒

      // Handling microscopic Pat's cases. Any slicing doesn't work well on 0-row matrix. This
      // probably should be fixed in the in-core matrix implementations.
      if (blockA.nrow == 0 )
        Iterator.empty
      else
      // Output each partial outer product with its correspondent partition #.
        Iterator.tabulate(numPartitions) {part ⇒

          val mBlock = blockA(::, ranges(part)).t %*% blockB
          part → mBlock
        }
    }

      // Reduce.
      .reduceByKey(_ += _, numPartitions = numPartitions)

      // Produce keys
      .map { case (blockKey, block) ⇒ ranges(blockKey).toArray → block }

    debug(s"A'B mmul #parts: ${rdd.partitions.length}.")

    rdd
  }

  private[sparkbindings] def computeAtBZipped2[A: ClassTag](zipped: RDD[(DrmTuple[A], DrmTuple[A])], nrow: Long,
                                                            ancol: Int, bncol: Int, numPartitions: Int) = {

    // The plan of this approach is to send a_i and parts of b_i to partitoin reducers which actually
    // do outer product sum update locally (instead of sending outer blocks). Thus it should minimize
    // expense for IO and also in-place partition block accum update should be much more efficient
    // than forming outer block matrices and perform matrix-on-patrix +.
    // Figure out appriximately block height per partition of the result.
    val blockHeight = safeToNonNegInt((nrow - 1) / numPartitions) + 1

    val partitionedRdd = zipped

      // Split B-rows into partitions using blockHeight
      .mapPartitions { iter =>

      val offsets = (0 until numPartitions).map(_ * blockHeight)
      val ranges = offsets.map(offs => offs until (offs + blockHeight min ancol))

      // Transform into series of (part -> (arow, part-brow)) tuples (keyed by part #).
      iter.flatMap { case ((_, arow), (_, brow)) =>

        ranges.view.zipWithIndex.map { case (arange, partNum) =>
          partNum -> (arow(arange).cloned -> brow)
        }
      }
    }

    val blockRdd = combineOuterProducts(partitionedRdd, numPartitions)

      // Add ordinal row keys.
      .map { case (blockNum, block) =>

      // Starting key
      var offset = blockNum * blockHeight

      var keys = Array.tabulate(block.nrow)(offset + _)
      keys -> block

    }

    blockRdd
  }

  /** Given already zipped, joined rdd of rows of A' and B, compute their product A'B */
  @deprecated("slow, will remove", since = "0.10.2")
  private[sparkbindings] def computeAtBZipped[A: ClassTag](zipped: RDD[(DrmTuple[A], DrmTuple[A])], nrow: Long,
                                                           ancol: Int, bncol: Int, numPartitions: Int) = {

    // Since Q and A are partitioned same way,we can just zip their rows and proceed from there by
    // forming outer products. Our optimizer lacks this primitive, so we will implement it using RDDs
    // directly. We try to compile B' = A'Q now by collecting outer products of rows of A and Q. At
    // this point we need to split n-range  of B' into sutiable number of partitions.

    if (log.isDebugEnabled) {
      log.debug(s"AtBZipped:zipped #parts ${zipped.partitions.length}")
      log.debug(s"AtBZipped:Targeted #parts $numPartitions")
    }

    // Figure out appriximately block height per partition of the result.
    val blockHeight = safeToNonNegInt((nrow - 1) / numPartitions) + 1

    val rddBt = zipped

      // Produce outer product blocks
      .flatMap { case ((aKey, aRow), (qKey, qRow)) =>
      for (blockKey <- Stream.range(0, numPartitions)) yield {
        val blockStart = blockKey * blockHeight
        val blockEnd = ancol min (blockStart + blockHeight)

        // Create block by cross product of proper slice of aRow and qRow
        blockKey -> (aRow(blockStart until blockEnd) cross qRow)

        // TODO: computing tons of cross product matrices seems to be pretty inefficient here. More
        // likely single streaming algorithm of updates will perform much better here. So rewrite
        // this using mapPartitions with numPartitions block accumulators.

      }
    }
      //      .combineByKey(
      //        createCombiner = (mx:Matrix) => mx,
      //        mergeValue = (c:Matrix,mx:Matrix) => c += mx,
      //        mergeCombiners = (c1:Matrix,c2:Matrix) => c1 += c2,
      //        numPartitions = numPartitions
      //      )
      // Doesn't look like use of combineByKey produces any better results than reduceByKey. So keeping
      // reduceByKey for simplicity. Combiners probably doesn't mean reduceByKey doesn't combine map-side.
      // Combine blocks by just summing them up
      .reduceByKey((block1, block2) => block1 += block2, numPartitions)

      // Throw away block key, generate row keys instead.
      .map { case (blockKey, block) =>
      val blockStart = blockKey * blockHeight
      val rowKeys = Array.tabulate(block.nrow)(blockStart + _)
      rowKeys -> block
    }

    if (log.isDebugEnabled) log.debug(s"AtBZipped #parts ${rddBt.partitions.length}")

    rddBt
  }

}
