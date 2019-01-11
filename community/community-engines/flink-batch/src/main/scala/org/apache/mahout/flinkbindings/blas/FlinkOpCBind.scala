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
import org.apache.flink.api.scala._
import org.apache.mahout.flinkbindings.drm._
import org.apache.mahout.math._
import org.apache.mahout.math.drm.logical.{OpCbind, OpCbindScalar}
import org.apache.mahout.math.scalabindings.RLikeOps._

import scala.collection.JavaConversions._

/**
 * Implementation of Flink's cbind
 */
object FlinkOpCBind {

  def cbind[K: TypeInformation](op: OpCbind[K], A: FlinkDrm[K], B: FlinkDrm[K]): FlinkDrm[K] = {
    val n = op.ncol
    val n1 = op.A.ncol
    val n2 = op.B.ncol

    implicit val classTag = op.A.keyClassTag

    val rowsA = A.asRowWise.ds
    val rowsB = B.asRowWise.ds

    val res: DataSet[(K, Vector)] =
      rowsA.coGroup(rowsB).where(0).equalTo(0) {
        (left, right) =>
          (left.toIterable.headOption, right.toIterable.headOption) match {
            case (Some((idx, a)), Some((_, b))) =>
              val result = if (a.isDense && b.isDense) {
                new DenseVector(n)
              } else {
                new SequentialAccessSparseVector(n)
              }

              result(0 until n1) := a
              result(n1 until n) := b

              (idx, result)
            case (Some((idx, a)), None) =>
              val result: Vector = if (a.isDense) {
                new DenseVector(n)
              } else {
                new SequentialAccessSparseVector(n)
              }
              result(n1 until n) := a

              (idx, result)
            case (None, Some((idx, b))) =>
              val result: Vector = if (b.isDense) {
                new DenseVector(n)
              } else {
                new SequentialAccessSparseVector(n)
              }
              result(n1 until n) := b

              (idx, result)
            case (None, None) =>
              throw new RuntimeException("CoGroup should have at least one non-empty input.")
          }
      }

    new RowsFlinkDrm(res.asInstanceOf[DataSet[(K, Vector)]], ncol=op.ncol)
  }

  def cbindScalar[K: TypeInformation](op: OpCbindScalar[K], A: FlinkDrm[K], x: Double): FlinkDrm[K] = {
    val left = op.leftBind
    val ds = A.asBlockified.ds

    implicit val kTag= op.keyClassTag

    def cbind(mat: Matrix, x: Double, left: Boolean): Matrix = {
      val ncol = mat.ncol
      val newMat = mat.like(mat.nrow, ncol + 1)

      if (left) {
        newMat.zip(mat).foreach { case (newVec, origVec) =>
          newVec(0) = x
          newVec(1 to ncol) := origVec
        }
      } else {
        newMat.zip(mat).foreach { case (newVec, origVec) =>
          newVec(ncol) = x
          newVec(0 until ncol) := origVec
        }
      }

      newMat
    }

    val out = A.asBlockified.ds.map {
      tuple => (tuple._1, cbind(tuple._2, x, left))
    }

    new BlockifiedFlinkDrm(out, op.ncol)
  }
}