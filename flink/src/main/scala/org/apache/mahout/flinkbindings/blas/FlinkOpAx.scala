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

import java.util

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.mahout.flinkbindings.FlinkEngine
import org.apache.mahout.flinkbindings.drm.{BlockifiedFlinkDrm, FlinkDrm, RowsFlinkDrm}
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical.{OpAtx, OpAx}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.{Matrix, Vector}

/**
 * Implementation of Flink Ax
 */
object FlinkOpAx {

  def blockifiedBroadcastAx[K: TypeInformation](op: OpAx[K], A: FlinkDrm[K]): FlinkDrm[K] = {
    implicit val ctx = A.context
    implicit val kTag = op.keyClassTag

    val singletonDataSetX = ctx.env.fromElements(op.x)

    val out = A.asBlockified.ds.map(new RichMapFunction[(Array[K], Matrix), (Array[K], Matrix)] {
      var x: Vector = null

      override def open(params: Configuration): Unit = {
        val runtime = this.getRuntimeContext
        val dsX: util.List[Vector] = runtime.getBroadcastVariable("vector")
        x = dsX.get(0)
      }

      override def map(tuple: (Array[K], Matrix)): (Array[K], Matrix) = tuple match {
        case (keys, mat) => (keys, (mat %*% x).toColMatrix)
      }
    }).withBroadcastSet(singletonDataSetX, "vector")

    new BlockifiedFlinkDrm(out, op.nrow.toInt)
  }


  def atx_with_broadcast(op: OpAtx, srcA: FlinkDrm[Int]): FlinkDrm[Int] = {
    implicit val ctx = srcA.context

    val dataSetA = srcA.asBlockified.ds

    // broadcast the vector x to the back end
    val bcastX = drmBroadcast(op.x)

    implicit val typeInformation = createTypeInformation[(Array[Int],Matrix)]
    val inCoreM = dataSetA.map {
      tuple =>
        tuple._1.zipWithIndex.map {
          case (key, idx) => tuple._2(idx, ::) * bcastX.value(key)
        }
          .reduce(_ += _)
    }
      // All-reduce
      .reduce(_ += _)

      // collect result
      .collect().head

      // Convert back to mtx
      .toColMatrix

    // This doesn't do anything now
    val res = FlinkEngine.parallelize(inCoreM, parallelismDegree = 1)

    new RowsFlinkDrm[Int](res, 1)

  }

}