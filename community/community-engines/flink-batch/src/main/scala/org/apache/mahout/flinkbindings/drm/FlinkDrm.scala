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
package org.apache.mahout.flinkbindings.drm

import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._
import org.apache.mahout.flinkbindings.{BlockifiedDrmDataSet, DrmDataSet, FlinkDistributedContext, wrapContext}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.{DenseMatrix, Matrix, SparseRowMatrix}

import scala.reflect.ClassTag

trait FlinkDrm[K] {
  def executionEnvironment: ExecutionEnvironment
  def context: FlinkDistributedContext
  def isBlockified: Boolean

  def asBlockified: BlockifiedFlinkDrm[K]
  def asRowWise: RowsFlinkDrm[K]

  def classTag: ClassTag[K]
}

class RowsFlinkDrm[K: TypeInformation: ClassTag](val ds: DrmDataSet[K], val ncol: Int) extends FlinkDrm[K] {

  def executionEnvironment = ds.getExecutionEnvironment
  def context: FlinkDistributedContext = ds.getExecutionEnvironment

  def isBlockified = false

  def asBlockified : BlockifiedFlinkDrm[K] = {
    val ncolLocal = ncol
    val classTag = implicitly[ClassTag[K]]

    val parts = ds.mapPartition {
      values =>
        val (keys, vectors) = values.toIterable.unzip

        if (vectors.nonEmpty) {
          val vector = vectors.head
          val matrix: Matrix = if (vector.isDense) {
            val matrix = new DenseMatrix(vectors.size, ncolLocal)
            vectors.zipWithIndex.foreach { case (vec, idx) => matrix(idx, ::) := vec }
            matrix
          } else {
            new SparseRowMatrix(vectors.size, ncolLocal, vectors.toArray)
          }

          Seq((keys.toArray(classTag), matrix))
        } else {
          Seq()
        }
    }

    new BlockifiedFlinkDrm[K](parts, ncol)
  }

  def asRowWise = this

  def classTag = implicitly[ClassTag[K]]

}

class BlockifiedFlinkDrm[K: TypeInformation: ClassTag](val ds: BlockifiedDrmDataSet[K], val ncol: Int) extends FlinkDrm[K] {


  def executionEnvironment = ds.getExecutionEnvironment
  def context: FlinkDistributedContext = ds.getExecutionEnvironment


  def isBlockified = true

  def asBlockified = this

  def asRowWise = {
    val out = ds.flatMap {
      tuple =>
        val keys = tuple._1
        val block = tuple._2

        keys.view.zipWithIndex.map {
          case (key, idx) => (key, block(idx, ::))
        }
    }

    new RowsFlinkDrm[K](out, ncol)
  }

  def classTag = implicitly[ClassTag[K]]

}