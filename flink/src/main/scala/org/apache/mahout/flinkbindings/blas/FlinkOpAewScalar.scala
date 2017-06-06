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
import org.apache.mahout.flinkbindings.drm.{BlockifiedFlinkDrm, FlinkDrm}
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.drm.logical.{AbstractUnaryOp, OpAewScalar, TEwFunc}
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.collection.JavaConversions._
import scala.reflect.ClassTag

import org.apache.flink.api.scala._

/**
 * Implementation if Flink OpAewScalar
 */
object FlinkOpAewScalar {

  final val PROPERTY_AEWB_INPLACE = "mahout.math.AewB.inplace"
  private def isInplace = System.getProperty(PROPERTY_AEWB_INPLACE, "false").toBoolean

  @Deprecated
  def opScalarNoSideEffect[K: TypeInformation](op: OpAewScalar[K], A: FlinkDrm[K], scalar: Double): FlinkDrm[K] = {
    val function = EWOpsCloning.strToFunction(op.op)
    implicit val kTag = op.keyClassTag


    val res = A.asBlockified.ds.map{
      tuple => (tuple._1, function(tuple._2, scalar))
    }

    new BlockifiedFlinkDrm(res, op.ncol)
  }

  def opUnaryFunction[K: TypeInformation](op: AbstractUnaryOp[K, K] with TEwFunc, A: FlinkDrm[K]): FlinkDrm[K] = {
    val f = op.f
    val inplace = isInplace


    implicit val kTag = op.keyClassTag

    val res = if (op.evalZeros) {
      A.asBlockified.ds.map{
        tuple =>
          val (keys, block) = tuple
          val newBlock = if (inplace) block else block.cloned
          newBlock := ((_, _, x) => f(x))
          (keys, newBlock)
      }
    } else {
      A.asBlockified.ds.map{
        tuple =>
          val (keys, block) = tuple
          val newBlock = if (inplace) block else block.cloned
          for (row <- newBlock; el <- row.nonZeroes) el := f(el.get)
          (keys, newBlock)
      }
    }

    new BlockifiedFlinkDrm(res, op.ncol)

  }

}

@Deprecated
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

