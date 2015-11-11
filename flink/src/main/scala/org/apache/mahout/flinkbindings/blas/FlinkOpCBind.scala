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
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import scala.reflect.ClassTag
import org.apache.flink.api.common.functions.CoGroupFunction
import org.apache.flink.api.common.functions.MapFunction
import org.apache.flink.api.scala.DataSet
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm._
import org.apache.mahout.math._
import org.apache.mahout.math.drm.logical.OpCbind
import org.apache.mahout.math.drm.logical.OpCbindScalar
import org.apache.mahout.math.scalabindings.RLikeOps._
import com.google.common.collect.Lists
import org.apache.mahout.flinkbindings.DrmDataSet

import org.apache.mahout.math.scalabindings._

/**
 * Implementation is taken from Spark's cbind
 * https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/CbindAB.scala
 */
object FlinkOpCBind {

  def cbind[K: ClassTag](op: OpCbind[K], A: FlinkDrm[K], B: FlinkDrm[K]): FlinkDrm[K] = {
    val n = op.ncol
    val n1 = op.A.ncol
    val n2 = op.B.ncol

    val classTag = extractRealClassTag(op.A)
    val joiner = selector[Vector, Any](classTag.asInstanceOf[ClassTag[Any]]) 

    val rowsA = A.asRowWise.ds.asInstanceOf[DrmDataSet[Any]]
    val rowsB = B.asRowWise.ds.asInstanceOf[DrmDataSet[Any]]

    val res: DataSet[(Any, Vector)] = 
      rowsA.coGroup(rowsB).where(joiner).equalTo(joiner)
        .`with`(new CoGroupFunction[(_, Vector), (_, Vector), (_, Vector)] {
      def coGroup(it1java: Iterable[(_, Vector)], it2java: Iterable[(_, Vector)], 
                  out: Collector[(_, Vector)]): Unit = {
        val it1 = Lists.newArrayList(it1java).asScala
        val it2 = Lists.newArrayList(it2java).asScala

        if (it1.nonEmpty && it2.nonEmpty) {
          val (idx, a) = it1.head
          val (_, b) = it2.head

          val result: Vector = if (a.isDense && b.isDense) {
            new DenseVector(n) 
          } else {
            new SequentialAccessSparseVector(n)
          }

          result(0 until n1) := a
          result(n1 until n) := b

          out.collect((idx, result))
        } else if (it1.isEmpty && it2.nonEmpty) {
          val (idx, b) = it2.head
          val result: Vector = if (b.isDense) { 
            new DenseVector(n)
          } else {
            new SequentialAccessSparseVector(n)
          }
          result(n1 until n) := b
          out.collect((idx, result))
        } else if (it1.nonEmpty && it2.isEmpty) {
          val (idx, a) = it1.head
          val result: Vector = if (a.isDense) {
            new DenseVector(n)
          } else {
            new SequentialAccessSparseVector(n)
          }
          result(n1 until n) := a
          out.collect((idx, result))
        }
      }
    })

    new RowsFlinkDrm(res.asInstanceOf[DataSet[(K, Vector)]], ncol=op.ncol)
  }

  def cbindScalar[K: ClassTag](op: OpCbindScalar[K], A: FlinkDrm[K], x: Double): FlinkDrm[K] = {
    val left = op.leftBind
    val ds = A.asBlockified.ds

    val out = A.asBlockified.ds.map(new MapFunction[(Array[K], Matrix), (Array[K], Matrix)] {
      def map(tuple: (Array[K], Matrix)): (Array[K], Matrix) = tuple match {
        case (keys, mat) => (keys, cbind(mat, x, left))
      }

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
            newVec(0 to (ncol - 1)) := origVec
          }
        }

        newMat
      }
    })

    new BlockifiedFlinkDrm(out, op.ncol)
  }

}