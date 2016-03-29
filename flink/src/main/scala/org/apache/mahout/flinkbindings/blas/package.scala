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
import org.apache.flink.api.scala._
import org.apache.flink.util.Collector

import scala.collection._

package object blas {

  /**
    * To compute tuples (PartitionIndex, PartitionElementCount)
    *
    * @param drmDataSet
    * @tparam K
    * @return (PartitionIndex, PartitionElementCount)
    */
  //TODO: Remove this when FLINK-3657 is merged into Flink codebase and
  // replace by call to DataSetUtils.countElementsPerPartition(DataSet[K])
  private[mahout] def countsPerPartition[K](drmDataSet: DataSet[K]): DataSet[(Int, Int)] = {
    drmDataSet.mapPartition {
      new RichMapPartitionFunction[K, (Int, Int)] {
        override def mapPartition(iterable: Iterable[K], collector: Collector[(Int, Int)]) = {
          val count: Int = Iterator(iterable).size
          val index: Int = getRuntimeContext.getIndexOfThisSubtask
          collector.collect((index, count))
        }
      }
    }
  }
}