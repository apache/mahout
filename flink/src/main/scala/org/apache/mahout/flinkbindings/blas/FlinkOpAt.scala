/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.mahout.flinkbindings.blas

import org.apache.flink.api.scala._
import org.apache.mahout.flinkbindings.drm.{FlinkDrm, RowsFlinkDrm}
import org.apache.mahout.math.{SequentialAccessSparseVector, Vector}
import org.apache.mahout.math.drm.logical.OpAt
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.Array.canBuildFrom

/**
 * Implementation of Flink At
 */
object FlinkOpAt {

  /**
   * The idea here is simple: compile vertical column vectors of every partition block as sparse
   * vectors of the <code>A.nrow</code> length; then group them by their column index and sum the
   * groups into final rows of the transposed matrix.
   */
  def sparseTrick(op: OpAt, A: FlinkDrm[Int]): FlinkDrm[Int] = {
    val ncol = op.ncol // # of rows of A, i.e. # of columns of A^T

    val sparseParts = A.asBlockified.ds.flatMap {
      blockifiedTuple =>
        val keys = blockifiedTuple._1
        val block = blockifiedTuple._2

        (0 until block.ncol).map {
          columnIndex =>
            val columnVector: Vector = new SequentialAccessSparseVector(ncol)

            keys.zipWithIndex.foreach {
              case (key, idx) => columnVector(key) = block(idx, columnIndex)
            }

            (columnIndex, columnVector)
        }
    }

    val regrouped = sparseParts.groupBy(0)

    val sparseTotal = regrouped.reduce{
      (left, right) =>
        (left._1, left._2 + right._2)
    }

    // TODO: densify or not?
    new RowsFlinkDrm(sparseTotal, ncol)
  }



}