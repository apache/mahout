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

import scala.reflect.ClassTag
import org.apache.mahout.math.drm.logical.OpTimesRightMatrix
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.DiagonalMatrix
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._

/**
 * Implementation is taken from Spark's OpTimesRightMatrix:
 * https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/AinCoreB.scala
 */
object FlinkOpTimesRightMatrix {

  def drmTimesInCore[K: ClassTag](op: OpTimesRightMatrix[K], A: FlinkDrm[K], inCoreB: Matrix): FlinkDrm[K] = {
    implicit val ctx = A.context

    val singletonDataSetB = ctx.env.fromElements(inCoreB)

    val res = A.blockify.ds.map(new RichMapFunction[(Array[K], Matrix), (Array[K], Matrix)] {
      var inCoreB: Matrix = null

      override def open(params: Configuration): Unit = {
        val runtime = this.getRuntimeContext()
        val dsB: java.util.List[Matrix] = runtime.getBroadcastVariable("matrix")
        inCoreB = dsB.get(0)
      }

      override def map(tuple: (Array[K], Matrix)): (Array[K], Matrix) = tuple match {
        case (keys, block_A) => (keys, block_A %*% inCoreB)
      }

    }).withBroadcastSet(singletonDataSetB, "matrix")

    new BlockifiedFlinkDrm(res, op.ncol)
  }

}