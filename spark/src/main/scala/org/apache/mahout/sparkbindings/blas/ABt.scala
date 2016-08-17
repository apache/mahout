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
import org.apache.spark.rdd.RDD

import scala.reflect.ClassTag
import org.apache.mahout.sparkbindings._
import org.apache.mahout.math.drm.BlockifiedDrmTuple
import org.apache.mahout.sparkbindings.drm._
import org.apache.mahout.math.{DenseMatrix, Matrix, SparseMatrix, SparseRowMatrix}
import org.apache.mahout.math.drm.logical.OpABt
import org.apache.mahout.logging._
import org.apache.mahout.math.flavor.MatrixFlavor

/** Contains RDD plans for ABt operator */
object ABt {

  private final implicit val log = getLog(ABt.getClass)

  /**
   * General entry point for AB' operator.
   *
   * @param operator the AB' operator
   * @param srcA A source RDD
   * @param srcB B source RDD 
   * @tparam K
   */
  def abt[K](
      operator: OpABt[K],
      srcA: DrmRddInput[K],
      srcB: DrmRddInput[Int]): DrmRddInput[K] = {

    debug("operator AB'(Spark)")
    abt_nograph(operator, srcA, srcB)(operator.keyClassTag)
  }

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
    val blocksA = srcA.asBlockified(operator.A.ncol)

    val blocksB = srcB.asBlockified(operator.B.ncol)

    val prodNCol = operator.ncol
    val prodNRow = operator.nrow
    // We are actually computing AB' here. 
    val numProductPartitions = estimateProductPartitions(anrow = prodNRow, ancol = operator.A.ncol,
      bncol = prodNCol, aparts = blocksA.partitions.length, bparts = blocksB.partitions.length)

    debug(
      s"AB': #parts = $numProductPartitions; A #parts=${blocksA.partitions.length}, B #parts=${blocksB.partitions.length}."+
      s"A=${operator.A.nrow}x${operator.A.ncol}, B=${operator.B.nrow}x${operator.B.ncol},AB'=${prodNRow}x$prodNCol."
    )

    // blockwise multiplication function
    def mmulFunc(tupleA: BlockifiedDrmTuple[K], tupleB: BlockifiedDrmTuple[Int]): (Array[K], Array[Int], Matrix) = {
      val (keysA, blockA) = tupleA
      val (keysB, blockB) = tupleB

      var ms = traceDo(System.currentTimeMillis())

      // We need to send keysB to the aggregator in order to know which columns are being updated.
      val result = (keysA, keysB, blockA %*% blockB.t)

      ms = traceDo(System.currentTimeMillis() - ms.get)
      trace(
        s"block multiplication of(${blockA.nrow}x${blockA.ncol} x ${blockB.ncol}x${blockB.nrow} is completed in $ms " +
          "ms.")
      trace(s"block multiplication types: blockA: ${blockA.getClass.getName}(${blockA.t.getClass.getName}); " +
        s"blockB: ${blockB.getClass.getName}.")

      result
    }

    val blockwiseMmulRdd =

    // Combine blocks pairwise.
      pairwiseApply(blocksA, blocksB, mmulFunc)

        // Now reduce proper product blocks.
        .combineByKey(

          // Empty combiner += value
          createCombiner = (t: (Array[K], Array[Int], Matrix)) =>  {
            val (rowKeys, colKeys, block) = t

            val comb = if (block.getFlavor == MatrixFlavor.SPARSELIKE) {
              new SparseMatrix(prodNCol, block.nrow).t
            } else {
              new DenseMatrix(prodNCol, block.nrow).t
            }

            for ((col, i) <- colKeys.zipWithIndex) comb(::, col) := block(::, i)
            rowKeys -> comb
          },

          // Combiner += value
          mergeValue = (comb: (Array[K], Matrix), value: (Array[K], Array[Int], Matrix)) => {
            val (rowKeys, c) = comb
            val (_, colKeys, block) = value
            for ((col, i) <- colKeys.zipWithIndex) c(::, col) := block(::, i)
            comb
          },

          // Combiner + Combiner
          mergeCombiners = (comb1: (Array[K], Matrix), comb2: (Array[K], Matrix)) => {
            comb1._2 += comb2._2
            comb1
          },

          numPartitions = blocksA.partitions.length max blocksB.partitions.length
        )


    // Created BlockifiedRDD-compatible structure.
    val blockifiedRdd = blockwiseMmulRdd

      // throw away A-partition #
      .map{case (_,tuple) => tuple}

    val numPartsResult = blockifiedRdd.partitions.length

    // See if we need to rebalance away from A granularity.
    if (numPartsResult * 2 < numProductPartitions || numPartsResult / 2 > numProductPartitions) {

      debug(s"Will re-coalesce from $numPartsResult to $numProductPartitions")

      val rowRdd = deblockify(blockifiedRdd).coalesce(numPartitions = numProductPartitions)

      rowRdd

    } else {

      // We don't have a terribly different partition
      blockifiedRdd
    }

  }

  /**
   * This function tries to use join instead of cartesian to group blocks together without bloating
   * the number of partitions. Hope is that we can apply pairwise reduction of block pair right away
   * so if the data to one of the join parts is streaming, the result is still fitting to memory,
   * since result size is much smaller than the operands.
   *
   * @param blocksA blockified RDD for A
   * @param blocksB blockified RDD for B
   * @param blockFunc a function over (blockA, blockB). Implies `blockA %*% blockB.t` but perhaps may be
   *                  switched to another scheme based on which of the sides, A or B, is bigger.
   */
  private def pairwiseApply[K1, K2, T](blocksA: BlockifiedDrmRdd[K1], blocksB: BlockifiedDrmRdd[K2], blockFunc:
  (BlockifiedDrmTuple[K1], BlockifiedDrmTuple[K2]) => T): RDD[(Int, T)] = {

    // We will be joining blocks in B to blocks in A using A-partition as a key.

    // Prepare A side.
    val blocksAKeyed = blocksA.mapPartitionsWithIndex { (part, blockIter) =>

      val r = if (blockIter.hasNext) Some(part -> blockIter.next) else Option.empty[(Int, BlockifiedDrmTuple[K1])]

      require(!blockIter.hasNext, s"more than 1 (${blockIter.size + 1}) blocks per partition and A of AB'")

      r.toIterator
    }

    // Prepare B-side.
    val aParts = blocksA.partitions.length
    val blocksBKeyed = blocksB.flatMap(bTuple => for (blockKey <- (0 until aParts).view) yield blockKey -> bTuple )

    // Perform the inner join. Let's try to do a simple thing now.
    blocksAKeyed.join(blocksBKeyed, numPartitions = aParts)

    // Apply product function which should produce smaller products. Hopefully, this streams blockB's in
    .map{case (partKey,(blockA, blockB)) => partKey -> blockFunc(blockA, blockB)}

  }

  private[blas] def abt_nograph_cart[K: ClassTag](
      operator: OpABt[K],
      srcA: DrmRddInput[K],
      srcB: DrmRddInput[Int]): DrmRddInput[K] = {

    // Blockify everything.
    val blocksA = srcA.asBlockified(operator.A.ncol)

        // Mark row-blocks with group id
        .mapPartitionsWithIndex((part, iter) => {

      if (iter.isEmpty) {
        Iterator.empty
      } else {

        val rowBlockId = part
        val (blockKeys, block) = iter.next()

        // Each partition must have exactly one block due to implementation guarantees of blockify()
        assert(!iter.hasNext, "Partition #%d is expected to have at most 1 block at AB'.".format(part))

        // the output is (row block id, array of row keys, and the matrix representing the block).
        Iterator((rowBlockId, blockKeys, block))
      }
    })

    val blocksB = srcB.asBlockified(operator.B.ncol)

    // Final product's geometry. We want to extract that into local variables since we want to use
    // them as closure attributes.
    val prodNCol = operator.ncol
    val prodNRow = operator.nrow
    val aNCol = operator.A.ncol

    // Approximate number of final partitions. We take bigger partitions as our guide to number of
    // elements per partition. TODO: do it better.

    // Elements per partition, bigger of two operands.
    val epp = aNCol.toDouble * prodNRow / blocksA.partitions.length max aNCol.toDouble * prodNCol /
      blocksB.partitions.length

    // Number of partitions we want to converge to in the product. For now we simply extrapolate that
    // assuming product density and operand densities being about the same; and using the same element
    // per partition number in the product as the bigger of two operands.
    val numProductPartitions = (prodNCol.toDouble * prodNRow / epp).ceil.toInt

    debug(
      s"AB': #parts = $numProductPartitions; A #parts=${blocksA.partitions.length}, B #parts=${blocksB.partitions.length}.")

    // The plan.
    var blockifiedRdd: BlockifiedDrmRdd[K] = blocksA

        // Build Cartesian. It generates a LOT of tasks. TODO: figure how to fix performance of AB'
        // operator. The thing is that product after map is really small one (partition fraction x
        // partition fraction) so they can be combined into much bigger chunks.
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

          createCombiner = (t: (Array[K], Array[Int], Matrix)) => t match {

            // Create combiner structure out of two products. Our combiner is sparse row matrix
            // initialized to final product partition block dimensions.
            case (rowKeys, colKeys, blockProd) =>

              // Accumulator is a row-wise block of sparse vectors. Since we assign to columns,
              // the most efficient is perhaps to create column-oriented block here.
              val acc:Matrix = new SparseRowMatrix(prodNCol, rowKeys.length).t

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
          },

          // Cartesian will tend to produce much more partitions that we actually need for co-grouping,
          // and as a result, we may see empty partitions than we actually need.
          numPartitions = numProductPartitions
        )

        // Combine leaves residual block key -- we don't need that.
        .map(_._2)

    // This may produce more than one block per partition. Most implementation rely on convention of
    // having at most one block per partition.
    blockifiedRdd = rbind(blockifiedRdd)

    blockifiedRdd
  }


}
