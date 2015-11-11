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

import java.util.List

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.mahout.flinkbindings.drm.{BlockifiedFlinkDrm, FlinkDrm}
import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.math.drm.logical.OpAx
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.reflect.ClassTag

import org.apache.flink.api.scala._


/**
 * Implementation is taken from Spark's Ax
 * https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/Ax.scala
 */
object FlinkOpAx {

  def blockifiedBroadcastAx[K: ClassTag](op: OpAx[K], A: FlinkDrm[K]): FlinkDrm[K] = {
    implicit val ctx = A.context

    val singletonDataSetX = ctx.env.fromElements(op.x)

    val out = A.asBlockified.ds.map(new RichMapFunction[(Array[K], Matrix), (Array[K], Matrix)] {
      var x: Vector = null

      override def open(params: Configuration): Unit = {
        val runtime = this.getRuntimeContext
        val dsX: List[Vector] = runtime.getBroadcastVariable("vector")
        x = dsX.get(0)
      }

      override def map(tuple: (Array[K], Matrix)): (Array[K], Matrix) = tuple match {
        case (keys, mat) => (keys, (mat %*% x).toColMatrix)
      }
    }).withBroadcastSet(singletonDataSetX, "vector")

    new BlockifiedFlinkDrm(out, op.nrow.toInt)
  }
}