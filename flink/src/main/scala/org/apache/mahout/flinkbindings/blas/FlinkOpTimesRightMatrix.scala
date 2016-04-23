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

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.java.typeutils.TypeExtractor
import org.apache.flink.api.scala._
import org.apache.flink.configuration.Configuration
import org.apache.mahout.flinkbindings.drm.{BlockifiedFlinkDrm, FlinkDrm}
import org.apache.mahout.math.drm.logical.OpTimesRightMatrix
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.{DenseMatrix, Matrix}

/**
 * Implementation of OpTimesRightMatrix:
 */
object FlinkOpTimesRightMatrix {

  def drmTimesInCore[K: TypeInformation](op: OpTimesRightMatrix[K], A: FlinkDrm[K], inCoreB: Matrix): FlinkDrm[K] = {
    implicit val ctx = A.context
    implicit val kTag = op.keyClassTag

    /* HACK: broadcasting the matrix using Flink's .withBroadcastSet(singletonDataSetB) on a matrix causes a backend Kryo
     * Issue resulting in a stackOverflow error.
     * 
     * Quick fix is to instead break the matrix down into a list of rows and then rebuild it on the back end
     * 
     * TODO: this is obviously very inefficient... need to use the correct broadcast on the matrix itself.
     */
    val rows = (0 until inCoreB.nrow).map(i => (i, inCoreB(i, ::)))
    val dataSetType = TypeExtractor.getForObject(rows.head)
    val singletonDataSetB = ctx.env.fromCollection(rows)

    val res = A.asBlockified.ds.map(new RichMapFunction[(Array[K], Matrix), (Array[K], Matrix)] {
      var inCoreB: Matrix = null

      override def open(params: Configuration): Unit = {
        val runtime = this.getRuntimeContext
        val dsB: java.util.List[(Int, org.apache.mahout.math.Vector)] = runtime.getBroadcastVariable("matrix")
        val m = dsB.size()
        val n = dsB.get(0)._2.size
        val isDense = dsB.get(0)._2.isDense

        inCoreB = isDense match {
          case true => new DenseMatrix(m, n)
          case false => new DenseMatrix(m, n)
        }
        for (i <- 0 until m) {
          inCoreB(i, ::) := dsB.get(i)._2
        }

      }
     
      override def map(tuple: (Array[K], Matrix)): (Array[K], Matrix) = tuple match {
        case (keys, block_A) => (keys, block_A %*% inCoreB)
      }

    }).withBroadcastSet(singletonDataSetB, "matrix")

    new BlockifiedFlinkDrm(res, op.ncol)
  }

}