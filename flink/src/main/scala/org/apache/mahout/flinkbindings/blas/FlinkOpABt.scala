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

package org.apache.mahout.flinkbindings.blas

import org.apache.flink.api.common.functions._
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala.DataSet
import org.apache.flink.util.Collector
import org.apache.mahout.logging._
import org.apache.mahout.math.drm.{BlockifiedDrmTuple, DrmTuple}
import org.apache.mahout.math.drm.logical.{OpAB, OpABt}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.{Matrix, SparseMatrix, SparseRowMatrix}
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm._
import org.apache.flink.configuration.Configuration
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm.{BlockifiedFlinkDrm, FlinkDrm}
import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical.OpABt
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

/** Contains DataSet plans for ABt operator */
object FlinkOpABt {

  private final implicit val log = getLog(FlinkOpABt.getClass)

  /**
   * General entry point for AB' operator.
   *
   * @param operator the AB' operator
   * @param srcA A source DataSet
   * @param srcB B source DataSet 
   * @tparam K
   */
  def abt[K: ClassTag: TypeInformation](
      operator: OpABt[K],
      srcA: FlinkDrm[K],
      srcB: FlinkDrm[Int]): FlinkDrm[K] = {

    debug("operator AB'(Flink)")
    abt_nograph[K](operator, srcA, srcB)
  }

  /**
   * Computes AB'
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
  private[flinkbindings] def abt_nograph[K](
      operator: OpABt[K],
      srcA: FlinkDrm[K],
      srcB: FlinkDrm[Int]): FlinkDrm[K] = {

    // Blockify everything.
    val blocksA = srcA.asBlockified
    val blocksB = srcB.asBlockified

    val prodNCol = operator.ncol
    val prodNRow = operator.nrow

    implicit val ktag = srcA.classTag

    // We are actually computing AB' here. 
    //    val numProductPartitions = estimateProductPartitions(anrow = prodNRow, ancol = operator.A.ncol,
    //      bncol = prodNCol, aparts = blocksA.executionEnvironment.getParallelism,
    //      bparts = blocksB.executionEnvironment.getParallelism)
    //
    //    debug(
    //      s"AB': #parts = $numProductPartitions; A #parts=${blocksA.executionEnvironment.getParallelism}, B #parts=${blocksB.executionEnvironment.getParallelism}."+
    //      s"A=${operator.A.nrow}x${operator.A.ncol}, B=${operator.B.nrow}x${operator.B.ncol},AB'=${prodNRow}x$prodNCol."
    //    )
    //
    // blockwise multiplication function
    def mmulFunc(tupleA: BlockifiedDrmTuple[K], tupleB: BlockifiedDrmTuple[Int]): (Array[K], Array[Int], Matrix) = {
      val (keysA, blockA) = tupleA
      val (keysB, blockB) = tupleB

      //      var ms = traceDo(System.currentTimeMillis())

      // We need to send keysB to the aggregator in order to know which columns are being updated.
      val result = (keysA, keysB, blockA %*% blockB.t)

      //      ms = traceDo(System.currentTimeMillis() - ms.get)
      //      trace(
      //        s"block multiplication of(${blockA.nrow}x${blockA.ncol} x ${blockB.ncol}x${blockB.nrow} is completed in $ms " +
      //          "ms.")
      //      trace(s"block multiplication types: blockA: ${blockA.getClass.getName}(${blockA.t.getClass.getName}); " +
      //        s"blockB: ${blockB.getClass.getName}.")

      result.asInstanceOf[(Array[K], Array[Int], Matrix)]
    }


    implicit val typeInformation = FlinkEngine.generateTypeInformation[(Array[K], Matrix)]
    implicit val typeInformation2 = FlinkEngine.generateTypeInformation[(Int, (Array[K], Array[Int], Matrix))]
    implicit val typeInformation3 = FlinkEngine.generateTypeInformation[(Array[K], Array[Int], Matrix)]

        val blockwiseMmulDataSet =

        // Combine blocks pairwise.
          pairwiseApply(blocksA.asBlockified.ds, blocksB.asBlockified.ds, mmulFunc)

            // Now reduce proper product blocks.

            // group by the partition key
            .groupBy(0)

            .combineGroup(new RichGroupCombineFunction[(Int, (Array[K], Array[Int], Matrix)), (Array[K], Array[Int], Matrix)] {

               def combine(values: java.lang.Iterable[(Int, (Array[K], Array[Int], Matrix))],
                           out: Collector[(Array[K], Array[Int], Matrix)]): Unit = {

                val tuple = values.iterator().next
                val rowKeys = tuple._2._1
                val colKeys = tuple._2._2
                val block = tuple._2._3

                val comb = new SparseMatrix(prodNCol, block.nrow).t
                for ((col, i) <- colKeys.zipWithIndex) comb(::, col) := block(::, i)
                val res = (rowKeys,colKeys, comb)

                out.collect(res)
              }
            })

               .combineGroup( new GroupCombineFunction[(Array[K], Array[Int], Matrix), (Array[K], Matrix)]{

                 def combine(values: java.lang.Iterable[(Array[K], Array[Int], Matrix)],
                             out: Collector[(Array[K], Matrix)]): Unit = {

                   val vals = values.iterator().next()

                   val (rowKeys, c) = (vals._1, vals._3)
                   val (_, colKeys, block) = (vals._1, vals._2, vals._3)
                   for ((col, i) <- colKeys.zipWithIndex) c(::, col) := block(::, i)
                   out.collect(rowKeys, c)
                 }
               })

               .reduce(new ReduceFunction[(Array[K], Matrix)] {

                  def reduce(mx1: (Array[K], Matrix), mx2: (Array[K], Matrix)): (Array[K], Matrix) = {
                    mx1._2 += mx2._2
                    mx1
                  }
               })

                // Created BlockifiedDataSet-compatible structure.
                val blockifiedDataSet = blockwiseMmulDataSet
                  // throw away A-partition #  this is done in flink.
//                  .map{tuple => tuple._2}
    //
    //
    //
                val numPartsResult = blockifiedDataSet.getParallelism
    //
    //         See if we need to rebalance away from A granularity.
    //            if (numPartsResult * 2 < numProductPartitions || numPartsResult / 2 > numProductPartitions) {
    //
    //              debug(s"Will re-coalesce from $numPartsResult to $numProductPartitions")
    //
    //              val rowDataSet = blockifiedDataSet.asRowWise  //.coalesce(numPartitions = numProductPartitions)
    //
    //              rowDataSet
    //
    //            } else {
    //
    //         We don't have a terribly different partition
    //        blockifiedDataSet
    //      }
    //    }
    //
    //  }

    implicit val typeInformationDrm = FlinkEngine.generateTypeInformation[K]
    new BlockifiedFlinkDrm(ds = blockifiedDataSet, ncol = prodNCol)
//    null.asInstanceOf[FlinkDrm[K]]

  }
    /**
      * This function tries to use join instead of cartesian to group blocks together without bloating
      * the number of partitions. Hope is that we can apply pairwise reduction of block pair right away
      * so if the data to one of the join parts is streaming, the result is still fitting to memory,
      * since result size is much smaller than the operands.
      *
      * @param blocksA   blockified DataSet for A
      * @param blocksB   blockified DataSet for B
      * @param blockFunc a function over (blockA, blockB). Implies `blockA %*% blockB.t` but perhaps may be
      *                  switched to another scheme based on which of the sides, A or B, is bigger.
      */
      private def pairwiseApply[K1, K2, T]( blocksA: BlockifiedDrmDataSet[K1], blocksB: BlockifiedDrmDataSet[K2], blockFunc:
      (BlockifiedDrmTuple[K1], BlockifiedDrmTuple[K2]) =>
        (Array[K1], Array[Int], Matrix) ): DataSet[(Int, (Array[K1], Array[Int], Matrix))] = {


      implicit val typeInformationA = FlinkEngine.generateTypeInformation[(Int, Array[K1], Matrix)]
      implicit val typeInformationProd = FlinkEngine.generateTypeInformation[(Int, (Array[K1], Array[Int], Matrix))]

      // We will be joining blocks in B to blocks in A using A-partition as a key.

      // Prepare A side.
      val blocksAKeyed = blocksA.mapPartition( new RichMapPartitionFunction[BlockifiedDrmTuple[K1],
                                                            (Int, Array[K1], Matrix)] {

//         override def open(params: Configuration): Unit = {
//           val runtime = this.getIterationRuntimeContext
//           //part = runtime.getIndexOfThisSubtask
//         }

         def mapPartition(values: java.lang.Iterable[BlockifiedDrmTuple[K1]], out: Collector[(Int, Array[K1], Matrix)]): Unit  = {

           val blockIter = values.iterator()

           val part = getIterationRuntimeContext.getIndexOfThisSubtask

//           val r = if (blockIter.hasNext()) part -> blockIter.next() else Option.empty[(Int, BlockifiedDrmTuple[K1])]
           val r =  part -> blockIter.next
           require(!blockIter.hasNext, s"more than 1 (${blockIter.asScala.size + 1}) blocks per partition and A of AB'")

           out.collect((r._1, r._2._1, r._2._2))
         }
       })

       implicit val typeInformationB = FlinkEngine.generateTypeInformation[(Int, (Array[K2], Matrix))]

      // Prepare B-side.
        val aParts = blocksAKeyed.getParallelism
        val blocksBKeyed = blocksB.flatMap(bTuple => for (blockKey <- (0 until aParts).view) yield blockKey -> bTuple )

        implicit val typeInformationJ = FlinkEngine.generateTypeInformation[(Int, ((Array[K1], Matrix),(Int, (Array[K2], Matrix))))]
        implicit val typeInformationJprod = FlinkEngine.generateTypeInformation[(Int, T)]

        // Perform the inner join.
        //blocksAKeyed.join(blocksBKeyed, numPartitions = aParts)
        blocksAKeyed.join(blocksBKeyed).where(0).equalTo(0){ (l, r) =>
          (l._1 , ((l._2, l._3), (r._1, r._2)))
             }


        // Apply product function which should produce smaller products. Hopefully, this streams blockB's in
    //    .map{case (partKey,(blockA, blockB)) => partKey -> blockFunc(blockA, blockB)}
          .map{tuple => tuple._1 -> blockFunc((tuple._2._1), (tuple._2._2._2))}

      }





  }
