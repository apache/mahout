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

import scala.collection.JavaConverters.asScalaBufferConverter
import scala.reflect.ClassTag

import org.apache.flink.api.common.functions.FlatMapFunction
import org.apache.flink.api.common.functions.GroupReduceFunction
import org.apache.flink.api.scala.DataSet
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.Vector
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical.OpAtB
import org.apache.mahout.math.scalabindings.RLikeOps._

import com.google.common.collect.Lists

import org.apache.flink.api.scala._

/**
 * Implementation is taken from Spark's AtB
 * https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/AtB.scala
 */
object FlinkOpAtB {

  def notZippable[K: ClassTag](op: OpAtB[K], At: FlinkDrm[K], B: FlinkDrm[K]): FlinkDrm[Int] = {

    val rowsAt = At.asRowWise.ds.asInstanceOf[DrmDataSet[Any]]
    val rowsB = B.asRowWise.ds.asInstanceOf[DrmDataSet[Any]]
    val joined = rowsAt.join(rowsB).where(0).equalTo(0)

    val ncol = op.ncol
    val nrow = op.nrow.toInt
    val blockHeight = 10
    val blockCount = safeToNonNegInt((nrow - 1) / blockHeight + 1)

    val preProduct: DataSet[(Int, Matrix)] = 
             joined.flatMap(new FlatMapFunction[Tuple2[(_, Vector), (_, Vector)], (Int, Matrix)] {
      def flatMap(in: Tuple2[(_, Vector), (_, Vector)],
                  out: Collector[(Int, Matrix)]): Unit = {
        val avec = in._1._2
        val bvec = in._1._2

        0.until(blockCount) map { blockKey =>
          val blockStart = blockKey * blockHeight
          val blockEnd = Math.min(nrow, blockStart + blockHeight)

          val outer = avec(blockStart until blockEnd) cross bvec
          out.collect(blockKey -> outer)
        }
      }
    })

    val res: BlockifiedDrmDataSet[Int] = 
      preProduct.groupBy(0)
                .reduceGroup(new GroupReduceFunction[(Int, Matrix), BlockifiedDrmTuple[Int]] {
      def reduce(values: Iterable[(Int, Matrix)], out: Collector[BlockifiedDrmTuple[Int]]): Unit = {
        val it = Lists.newArrayList(values).asScala
        val (idx, _) = it.head

        val block = it.map { t => t._2 }.reduce { (m1, m2) => m1 + m2 }

        val keys = idx.until(block.nrow).toArray[Int]
        out.collect(keys -> block)
      }
    })

    new BlockifiedFlinkDrm(res, ncol)
  }

}