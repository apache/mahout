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

//@Deprecated
//class DataSetOps[K: ClassTag](val ds: DataSet[K]) {

  /**
   * Implementation taken from http://stackoverflow.com/questions/30596556/zipwithindex-on-apache-flink
   * 
   * TODO: remove when FLINK-2152 is committed and released 
   */
//  def zipWithIndex(): DataSet[(Int, K)] = {
//
//     first for each partition count the number of elements - to calculate the offsets
//    val counts = ds.mapPartition(new RichMapPartitionFunction[K, (Int, Int)] {
//      override def mapPartition(values: Iterable[K], out: Collector[(Int, Int)]): Unit = {
//        val cnt: Int = values.asScala.count(_ => true)
//        val subtaskIdx = getRuntimeContext.getIndexOfThisSubtask
//        out.collect((subtaskIdx, cnt))
//      }
//    })

    // then use the offsets to index items of each partition
//    val zipped = ds.mapPartition(new RichMapPartitionFunction[K, (Int, K)] {
//        var offset: Int = 0
//
//        override def open(parameters: Configuration): Unit = {
//          val offsetsJava: java.util.List[(Int, Int)] =
//                  getRuntimeContext.getBroadcastVariable("counts")
//          val offsets = offsetsJava.asScala
//
//          val sortedOffsets =
//            offsets sortBy { case (id, _) => id } map { case (_, cnt) => cnt }
//
//          val subtaskId = getRuntimeContext.getIndexOfThisSubtask
//          offset = sortedOffsets.take(subtaskId).sum
//        }
//
//        override def mapPartition(values: Iterable[K], out: Collector[(Int, K)]): Unit = {
//          val it = values.asScala
//          it.zipWithIndex.foreach { case (value, idx) =>
//            out.collect((idx + offset, value))
//          }
//        }
//    }).withBroadcastSet(counts, "counts")
//
//    zipped
//  }
//
//}