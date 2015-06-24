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

import java.lang.Iterable

import scala.Array.canBuildFrom
import scala.collection.JavaConverters.asScalaBufferConverter

import org.apache.flink.api.common.functions.FlatMapFunction
import org.apache.flink.api.common.functions.GroupReduceFunction
import org.apache.flink.shaded.com.google.common.collect.Lists
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.SequentialAccessSparseVector
import org.apache.mahout.math.Vector
import org.apache.mahout.math.drm.DrmTuple
import org.apache.mahout.math.drm.logical.OpAt
import org.apache.mahout.math.scalabindings.RLikeOps._

/**
 * Implementation is taken from Spark's At
 * https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/At.scala
 */
object FlinkOpAt {

  /**
   * The idea here is simple: compile vertical column vectors of every partition block as sparse
   * vectors of the <code>A.nrow</code> length; then group them by their column index and sum the
   * groups into final rows of the transposed matrix.
   */
  def sparseTrick(op: OpAt, A: FlinkDrm[Int]): FlinkDrm[Int] = {
    val ncol = op.ncol // # of rows of A, i.e. # of columns of A^T

    val sparseParts = A.blockify.ds.flatMap(new FlatMapFunction[(Array[Int], Matrix), DrmTuple[Int]] {
      def flatMap(typle: (Array[Int], Matrix), out: Collector[DrmTuple[Int]]): Unit = typle match {
        case (keys, block) => {
          (0 until block.ncol).map(columnIdx => {
            val columnVector: Vector = new SequentialAccessSparseVector(ncol)

            keys.zipWithIndex.foreach { case (key, idx) =>
              columnVector(key) = block(idx, columnIdx)
            }

            out.collect(new Tuple2(columnIdx, columnVector))
          })
        }
      }
    })

    val regrouped = sparseParts.groupBy(tuple_1[Vector])

    val sparseTotal = regrouped.reduceGroup(new GroupReduceFunction[(Int, Vector), DrmTuple[Int]] {
      def reduce(values: Iterable[(Int, Vector)], out: Collector[DrmTuple[Int]]): Unit = {
        val it = Lists.newArrayList(values).asScala
        val (idx, _) = it.head
        val vector = it map { case (idx, vec) => vec } reduce (_ + _)
        out.collect(idx -> vector)
      }
    })

    // TODO: densify or not?
    new RowsFlinkDrm(sparseTotal, ncol)
  }

}