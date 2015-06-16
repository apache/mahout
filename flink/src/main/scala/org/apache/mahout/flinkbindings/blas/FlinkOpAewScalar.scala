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
import org.apache.mahout.math.drm.logical.OpAewScalar
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.flink.api.common.functions.MapFunction
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm

/**
 * Implementation is inspired by Spark-binding's OpAewScalar
 * (see https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/AewB.scala) 
 */
object FlinkOpAewScalar {

  def opScalarNoSideEffect[K: ClassTag](op: OpAewScalar[K], A: FlinkDrm[K], scalar: Double): FlinkDrm[K] = {
    val function = EWOpsCloning.strToFunction(op.op)

    val res = A.blockify.ds.map(new MapFunction[(Array[K], Matrix), (Array[K], Matrix)] {
      def map(tuple: (Array[K], Matrix)): (Array[K], Matrix) = tuple match {
        case (keys, mat) => (keys, function(mat, scalar))
      }
    })

    new BlockifiedFlinkDrm(res, op.ncol)
  }

}

object EWOpsCloning {

  type MatrixScalarFunc = (Matrix, Double) => Matrix

  def strToFunction(op: String): MatrixScalarFunc = op match {
    case "+" => plusScalar
    case "-" => minusScalar
    case "*" => timesScalar
    case "/" => divScalar
    case "-:" => scalarMinus
    case "/:" => scalarDiv
    case _ => throw new IllegalArgumentException(s"Unsupported elementwise operator: $op")
  }

  val plusScalar: MatrixScalarFunc = (A, s) => A + s
  val minusScalar: MatrixScalarFunc = (A, s) => A - s
  val scalarMinus: MatrixScalarFunc = (A, s) => s -: A
  val timesScalar: MatrixScalarFunc = (A, s) => A * s
  val divScalar: MatrixScalarFunc = (A, s) => A / s
  val scalarDiv: MatrixScalarFunc = (A, s) => s /: A
}

