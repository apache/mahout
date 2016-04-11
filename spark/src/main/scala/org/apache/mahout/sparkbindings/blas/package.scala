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

package org.apache.mahout.sparkbindings

import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.spark.rdd.RDD

import scala.collection.JavaConversions._
import scala.collection._

/**
 * This validation contains distributed algorithms that distributed matrix expression optimizer picks
 * from.
 */
package object blas {

  implicit def drmRdd2ops[K](rdd: DrmRdd[K]): DrmRddOps[K] = new DrmRddOps[K](rdd)

  /**
   * Rekey matrix dataset keys to consecutive int keys.
   * @param rdd incoming matrix row-wise dataset
   *
   * @param computeMap if true, also compute mapping between old and new keys
   * @tparam K existing key parameter
   * @return
   */
  private[sparkbindings] def rekeySeqInts[K](rdd: DrmRdd[K], computeMap: Boolean = true): (DrmRdd[Int],
    Option[RDD[(K, Int)]]) = {

    // Spark context please.
    val sctx = rdd.context
    import sctx._

    // First, compute partition sizes.
    val partSizes = rdd.mapPartitionsWithIndex((part, iter) => Iterator(part -> iter.size))

      // Collect in-core
      .collect()

    // Starting indices
    var startInd = new Array[Int](rdd.partitions.length)

    // Save counts
    for (pc <- partSizes) startInd(pc._1) = pc._2

    // compute cumulative sum
    val siBcast = broadcast(startInd.scanLeft(0)(_ + _).init)

    // Compute key -> int index map:
    val keyMap = if (computeMap) {
      Some(rdd

        // Process individual partition with index, output `key -> index` tuple
        .mapPartitionsWithIndex((part, iter) => {

        // Start index for this partition
        val si = siBcast.value(part)
        iter.zipWithIndex.map { case ((key, _), index) => key -> (index + si)}
      })) // Some

    } else {

      // Were not asked to compute key mapping
      None
    }

    // Finally, do the transform
    val intRdd = rdd

      // Re-number each partition
      .mapPartitionsWithIndex((part, iter) => {

      // Start index for this partition
      val si = siBcast.value(part)

      // Iterate over data by producing sequential row index and retaining vector value.
      iter.zipWithIndex.map { case ((_, vec), ind) => si + ind -> vec}
    })

    // Finally, return drm -> keymap result

    intRdd -> keyMap

  }


  /**
   * Fills in missing rows in an Int-indexed matrix by putting in empty row vectors for the missing
   * keys.
   */
  private[mahout] def fixIntConsistency(op: DrmLike[Int], src: DrmRdd[Int]): DrmRdd[Int] = {

    if (op.canHaveMissingRows) {

      val rdd = src
      val sc = rdd.sparkContext
      val dueRows = safeToNonNegInt(op.nrow)
      val dueCols = op.ncol

      // Compute the fix.
      sc

        // Bootstrap full key set
        .parallelize(0 until dueRows, numSlices = rdd.partitions.length max 1)

        // Enable PairedFunctions
        .map(_ -> Unit)

        // Cogroup with all rows
        .cogroup(other = rdd)

        // Filter out out-of-bounds
        .filter { case (key, _) => key >= 0 && key < dueRows}

        // Coalesce and output RHS
        .map { case (key, (seqUnit, seqVec)) =>
        val acc = seqVec.headOption.getOrElse(new SequentialAccessSparseVector(dueCols))
        val vec = if (seqVec.nonEmpty) (acc /: seqVec.tail)(_ + _) else acc
        key -> vec
      }

    } else src

  }

  /** Method to do `mxC += a cross b` in-plcae a bit more efficiently than this expression does. */
  def addOuterProduct(mxC: Matrix, a: Vector, b: Vector): Matrix = {

    // Try to pay attention to density a bit here when computing and adding the outer product of
    // arow and brow fragment.
    if (b.isDense)
      for (ela <- a.nonZeroes) mxC(ela.index, ::) := { (i, x) => x + ela * b(i)}
    else
      for (ela <- a.nonZeroes; elb <- b.nonZeroes()) mxC(ela.index, elb.index) += ela * elb

    mxC
  }

  /**
   * Compute ranges of more or less even splits of total `nrow` number
   *
   * @param nrow
   * @param numSplits
   * @return
   */
  @inline
  private[blas] def computeEvenSplits(nrow: Long, numSplits: Int): IndexedSeq[Range] = {
    require(numSplits <= nrow, "Requested amount of splits greater than number of data points.")
    require(nrow >= 1)
    require(numSplits >= 1)

    // Base split -- what is our base split size?
    val baseSplit = safeToNonNegInt(nrow / numSplits)

    // Slack -- how many splits will have to be incremented by 1 though?
    val slack = safeToNonNegInt(nrow % numSplits)

    // Compute ranges. We need to set ranges so that numSplits - slack splits have size of baseSplit;
    // and `slack` splits have size baseSplit + 1. Here is how we do it: First, we compute the range
    // offsets:
    val offsets = (0 to numSplits).map(i => i * (baseSplit + 1) - (0 max i - slack))
    // And then we connect the ranges using gaps between offsets:
    offsets.sliding(2).map(offs => offs(0) until offs(1)).toIndexedSeq
  }

  /**
   * Estimate number of partitions for the product of A %*% B.
   *
   * We take average per-partition element count of product as higher of the same of A and B. (prefer
   * larger partitions of operands).
   *
   * @param anrow A.nrow
   * @param ancol A.ncol
   * @param bncol B.ncol
   * @param aparts partitions in A
   * @param bparts partitions in B
   * @return recommended partitions
   */
  private[blas] def estimateProductPartitions(anrow:Long, ancol:Long, bncol:Long, aparts:Int, bparts:Int):Int = {

    // Compute per-partition element density in A
    val eppA = anrow.toDouble * ancol/ aparts

    // Compute per-partition element density in B
    val eppB = ancol.toDouble * bncol / bparts

    // Take the maximum element density into account. Is it a good enough?
    val epp = eppA max eppB

    // product partitions
    val prodParts = anrow * bncol / epp

    val nparts = math.round(prodParts).toInt max 1

    // Constrain nparts to maximum of anrow to prevent guaranteed empty partitions.
    if (nparts > anrow) anrow.toInt else nparts
  }

}
