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

import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.java.typeutils.TypeExtractor
import org.apache.mahout.flinkbindings.{FlinkByteBCast, FlinkEngine}
import org.apache.mahout.math.scalabindings._

import scala.reflect.ClassTag

import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.{DenseMatrix, MatrixWritable, Matrix}
import org.apache.mahout.math.drm.logical.OpTimesRightMatrix
import org.apache.mahout.math.scalabindings.RLikeOps._

import org.apache.flink.api.scala._

/**
 * Implementation is taken from Spark's OpTimesRightMatrix:
 * https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/AinCoreB.scala
 */
object FlinkOpTimesRightMatrix {

  def drmTimesInCore[K: TypeInformation: ClassTag](op: OpTimesRightMatrix[K], A: FlinkDrm[K], inCoreB: Matrix): FlinkDrm[K] = {
    implicit val ctx = A.context

  //  val singletonDataSetB = ctx.env.fromElements(inCoreB)

    println("AT FLINK OPTIMESRIGHTMATRIX!!!!!!!!!!!!!!!")
//    val inCoreBcastB = FlinkEngine.drmBroadcast(inCoreB)
//    val singletonDataSetB = ctx.env.fromElements(inCoreB)

    val rows = (0 until inCoreB.nrow).map(i => (i, inCoreB(i, ::)))
    val dataSetType = TypeExtractor.getForObject(rows.head)
    //TODO: Make Sure that this is the correct partitioning scheme
    val singletonDataSetB = ctx.env.fromCollection(rows)

    val res = A.asBlockified.ds.map(new RichMapFunction[(Array[K], Matrix), (Array[K], Matrix)] {
      var inCoreBc: Matrix = null

      override def open(params: Configuration): Unit = {
        val runtime = this.getRuntimeContext
        val dsB: java.util.List[Matrix] = runtime.getBroadcastVariable("matrix")
//        inCoreBc = dsB.get(0).asInstanceOf[FlinkByteBCast[Matrix]].value
        val inCoreBc = dsB.get(0)//.asInstanceOf[MatrixWritable].get()

      }
     // inCoreBc = inCoreBcastB.value

      override def map(tuple: (Array[K], Matrix)): (Array[K], Matrix) = tuple match {
        case (keys, block_A) => (keys, block_A %*% inCoreBc)
      }

    }).withBroadcastSet(singletonDataSetB, "matrix")
    println("Finished FLINK OPTIMESRIGHTMATRIX!!!!!!!!!!!!!!!")

    new BlockifiedFlinkDrm(res, op.ncol)
  }

}