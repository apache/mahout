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
package org.apache.mahout.flinkbindings

import java.lang.Iterable

import org.apache.flink.api.common.functions.RichMapPartitionFunction
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._
import org.apache.flink.api.scala.utils._
import org.apache.flink.configuration.Configuration
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.{RandomAccessSparseVector, Vector}

import scala.reflect.ClassTag

package object blas {

  /**
    * Rekey matrix dataset keys to consecutive int keys.
    * @param drmDataSet incoming matrix row-wise dataset
    * @param computeMap if true, also compute mapping between old and new keys
    * @tparam K existing key parameter
    * @return
    */
  private[flinkbindings] def rekeySeqInts[K: ClassTag: TypeInformation](drmDataSet: FlinkDrm[K],
                                                                 computeMap: Boolean = true): (DrmLike[Int],
    Option[DataSet[(K, Int)]]) = {

    implicit val dc = drmDataSet.context

    val datasetA = drmDataSet.asRowWise.ds

    val ncols = drmDataSet.asRowWise.ncol

    // Flink environment
    val env = datasetA.getExecutionEnvironment

    // First, compute partition sizes.
    val partSizes = DataSetUtils(datasetA).countElementsPerPartition.collect().toList

    // Starting indices
    var startInd = new Array[Int](datasetA.getParallelism)

    // Save counts
    for (pc <- partSizes) startInd(pc._1) = pc._2.toInt

    // compute cumulative sum
    val cumulativeSum = startInd.scanLeft(0)(_ + _).init

    val vector: Vector = new RandomAccessSparseVector(cumulativeSum.length)

    cumulativeSum.indices.foreach { i => vector(i) = cumulativeSum(i).toDouble }

    val bCast = FlinkEngine.drmBroadcast(vector)

    implicit val typeInformation = createTypeInformation[(K, Int)]

    // Compute key -> int index map:
    val keyMap = if (computeMap) {
      Some(
        datasetA.mapPartition(new RichMapPartitionFunction[(K, Vector), (K, Int)] {

          // partition number
          var part: Int = 0

          // get the index of the partition
          override def open(params: Configuration): Unit = {
            part = getRuntimeContext.getIndexOfThisSubtask
          }

          override def mapPartition(iterable: Iterable[(K, Vector)], collector: Collector[(K, Int)]): Unit = {
            val k = iterable.iterator().next._1
            val si = bCast.value.get(part)
            collector.collect(k -> (part + si).toInt)
          }
        }))
    } else {
      None
    }

    // Finally, do the transform
    val intDataSet = datasetA

      // Re-number each partition
      .mapPartition(new RichMapPartitionFunction[(K, Vector), (Int, Vector)] {

        // partition number
        var part: Int = 0

        // get the index of the partition
        override def open(params: Configuration): Unit = {
          part = getRuntimeContext.getIndexOfThisSubtask
        }

        override def mapPartition(iterable: Iterable[(K, Vector)], collector: Collector[(Int, Vector)]): Unit = {
          val k = iterable.iterator().next._2
          val si = bCast.value.get(part)
          collector.collect((part + si).toInt -> k)
        }
      })

    // Finally, return drm -> keymap result
    datasetWrap(intDataSet) -> keyMap
  }
}