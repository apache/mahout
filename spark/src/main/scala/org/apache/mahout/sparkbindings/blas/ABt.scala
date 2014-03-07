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

import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.sparkbindings.drm.plan.OpABt
import scala.reflect.ClassTag
import org.apache.mahout.sparkbindings.drm._
import org.apache.mahout.math.{Matrix, SparseRowMatrix}
import org.apache.spark.SparkContext._

/** Contains RDD plans for ABt operator */
object ABt {

  /**
   * General entry point for AB' operator.
   *
   * @param operator the AB' operator
   * @param srcA A source RDD
   * @param srcB B source RDD 
   * @tparam K
   */
  def abt[K: ClassTag](
      operator: OpABt[K],
      srcA: DrmRddInput[K],
      srcB: DrmRddInput[Int]): DrmRddInput[K] =
    abt_nograph(operator, srcA, srcB)

  /**
   * Computes AB' without GraphX.
   *
   * General idea here is that we split both A and B vertically into blocks (one block per split),
   * then compute cartesian join of the blocks of both data sets. This creates tuples of the form of
   * (A-block, B-block). We enumerate A-blocks and transform this into (A-block-id, A-block, B-block)
   * and then compute A-block %*% B-block', thus producing tuples (A-block-id, AB'-block).
   *
   * The next step is to group the above tuples by A-block-id and stitch al AB'-blocks in the group
   * horizontally, forming single vertical block of the final product AB'.
   *
   * This logic is complicated a little by the fact that we have to keep block row and column keys
   * so that the stitching of AB'-blocks happens according to integer row indices of the B input.
   */
  private[blas] def abt_nograph[K: ClassTag](
      operator: OpABt[K],
      srcA: DrmRddInput[K],
      srcB: DrmRddInput[Int]): DrmRddInput[K] = {

    // Blockify everything.
    val blocksA = srcA.toBlockifiedDrmRdd()

        // mark row-blocks with group id
        .mapPartitionsWithIndex((part, iter) => {
      val rowBlockId = part
      val (blockKeys, block) = iter.next()

      // Each partition must have exactly one block due to implementation guarantees of blockify()
      iter.ensuring(!_.hasNext)

      // the output is (row block id, array of row keys, and the matrix representing the block).
      Iterator((rowBlockId, blockKeys, block))
    })

    val blocksB = srcB.toBlockifiedDrmRdd()

    // Final product's geometry. We want to extract that into local variables since we want to use
    // them as closure attributes.
    val prodNCol = operator.ncol
    val prodNRow = operator.nrow
    
    // Approximate number of final partitions.
    val numProductPartitions =
      if (blocksA.partitions.size > blocksB.partitions.size) {
        ((prodNCol.toDouble / operator.A.ncol) * blocksA.partitions.size).ceil.toInt
      } else {
        ((prodNRow.toDouble / operator.B.ncol) * blocksB.partitions.size).ceil.toInt
      }

    //srcA.partitions.size.max(that = srcB.partitions.size)


    // The plan.
    val blockifiedRdd :BlockifiedDrmRdd[K] = blocksA

        // Build Cartesian. It may require a bit more memory there at tasks.
        .cartesian(blocksB)

        // Multiply blocks
        .map({

      // Our structure here after the Cartesian (full-join):
      case ((blockId, rowKeys, blA), (colKeys, blB)) =>

        // Compute block-product -- even though columns will require re-keying later, the direct
        // multiplication still works.
        val blockProd = blA %*% blB.t

        // Output block in the form (blockIds, row keys, colKeys, block matrix).
        blockId ->(rowKeys, colKeys, blockProd)

    })

        // Combine -- this is probably the most efficient
        .combineByKey[(Array[K],Matrix)](

          createCombiner = (t:(Array[K],Array[Int],Matrix)) => t match {
            case (rowKeys, colKeys, blockProd) =>

              // Accumulator is a row-wise block of sparse vectors.
              val acc:Matrix = new SparseRowMatrix(rowKeys.size, prodNCol)

              // Update accumulator using colKeys as column index indirection
              colKeys.view.zipWithIndex.foreach({
                case (col, srcCol) =>
                  acc(::, col) := blockProd(::, srcCol)
              })
              rowKeys -> acc
          },

          mergeValue = (a: (Array[K], Matrix), v: (Array[K], Array[Int], Matrix)) => {
            val (_, acc) = a
            val (_, colKeys, blockProd) = v

            // Note the assignment rather than +=. We really assume that B' operand matrix is keyed
            // uniquely!
            colKeys.view.zipWithIndex.foreach({
              case (col, srcCol) => acc(::, col) := blockProd(::, srcCol)
            })
            a
          },

          mergeCombiners = (c1: (Array[K], Matrix), c2: (Array[K], Matrix)) => {
            c1._2 += c2._2
            c1
          })

        // Combine leaves residual block key -- we don't need that.
        .map(_._2)

    new DrmRddInput(blockifiedSrc = Some(blockifiedRdd))
  }
}
