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

import org.apache.mahout.sparkbindings.drm._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.logging._
import RLikeOps._
import org.apache.spark.SparkContext._
import org.apache.mahout.math.{DenseVector, Vector, SequentialAccessSparseVector}
import org.apache.mahout.math.drm.logical.OpAt


/** A' algorithms */
object At {

  private final implicit val log = getLog(At.getClass)

  def at(
      operator: OpAt,
      srcA: DrmRddInput[Int]): DrmRddInput[Int] = at_nograph(operator = operator, srcA = srcA)

  /**
   * Non-GraphX spark implementation of transposition.
   *
   * The idea here is simple: compile vertical column vectors of every partition block as sparse
   * vectors of the <code>A.nrow</code> length; then group them by their column index and sum the
   * groups into final rows of the transposed matrix.
   */
  private[blas] def at_nograph(operator: OpAt, srcA: DrmRddInput[Int]): DrmRddInput[Int] = {

    debug("operator A'.")

    val drmRdd = srcA.asBlockified(operator.A.ncol)
    val numPartitions = drmRdd.partitions.size
    val ncol = operator.ncol

    debug(s"A' #parts = $numPartitions.")

    // Validity of this conversion must be checked at logical operator level.
    val nrow = operator.nrow.toInt
    val atRdd = drmRdd
        // Split
        .flatMap({
      case (keys, blockA) =>
        (0 until blockA.ncol).view.map(blockCol => {
          // Compute sparse vector. This should be quick if we assign values siquentially.
          val colV: Vector = new SequentialAccessSparseVector(ncol)
          keys.view.zipWithIndex.foreach({
            case (row, blockRow) => colV(row) = blockA(blockRow, blockCol)
          })

          blockCol -> colV
        })
    })

        // Regroup
        .groupByKey(numPartitions = numPartitions)

        // Reduce
        .map({
      case (key, vSeq) =>
        var v: Vector = vSeq.reduce(_ + _)
        key -> v
    }).densify()

    atRdd
  }

}
