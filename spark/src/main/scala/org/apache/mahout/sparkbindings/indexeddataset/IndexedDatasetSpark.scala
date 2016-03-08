/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.mahout.sparkbindings.indexeddataset

import org.apache.mahout.drivers.TextDelimitedIndexedDatasetWriter
import org.apache.mahout.math.drm.{DistributedContext, CheckpointedDrm}
import org.apache.mahout.math.{RandomAccessSparseVector, indexeddataset}
import org.apache.mahout.math.indexeddataset._
import org.apache.mahout.sparkbindings._
import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._


/**
 * Spark implementation of [[org.apache.mahout.math.indexeddataset.IndexedDataset]] providing the Spark specific
 * dfsWrite method
 * @param matrix a [[org.apache.mahout.sparkbindings.drm.CheckpointedDrmSpark]] to wrap
 * @param rowIDs a bidirectional map for Mahout Int IDs to/from application specific string IDs
 * @param columnIDs a bidirectional map for Mahout Int IDs to/from application specific string IDs
 */
class IndexedDatasetSpark(val matrix: CheckpointedDrm[Int], val rowIDs: BiDictionary,
    val columnIDs: BiDictionary)
  extends IndexedDataset {

  /** Secondary constructor enabling immutability */
  def this(id2: IndexedDatasetSpark){
    this(id2.matrix, id2.rowIDs, id2.columnIDs)
  }
  
  /**
   * Factory method used to create this extending class when the interface of
   * [[org.apache.mahout.math.indexeddataset.IndexedDataset]] is all that is known.
   */
  override def create(matrix: CheckpointedDrm[Int], rowIDs: BiDictionary,
      columnIDs: BiDictionary):
    IndexedDatasetSpark = {
    new IndexedDatasetSpark(matrix, rowIDs, columnIDs)
  }

  /**
   * Implements the core method to write [[org.apache.mahout.math.indexeddataset.IndexedDataset]]. Override and
   * replace the writer to change how it is written.
   */
  override def dfsWrite(dest: String, schema: Schema = DefaultIndexedDatasetWriteSchema)
      (implicit sc: DistributedContext):
    Unit = {
    val writer = new TextDelimitedIndexedDatasetWriter(schema)(sc)
    writer.writeTo(this, dest)
  }
}

object IndexedDatasetSpark {
  
  def apply(elements: RDD[(String, String)], existingRowIDs: Option[BiDictionary] = None)(implicit sc: SparkContext) = {

    // create separate collections of rowID and columnID tokens
    val rowIDs = elements.map { case (rowID, _) => rowID }.distinct().collect()
    val columnIDs = elements.map { case (_, columnID) => columnID }.distinct().collect()

    // create BiDictionary(s) for bi-directional lookup of ID by either Mahout ID or external ID
    // broadcast them for access in distributed processes, so they are not recalculated in every task.
    //val rowIDDictionary = BiDictionary.append(existingRowIDs, rowIDs)
    val rowIDDictionary = existingRowIDs match {
      case Some(d) => d.merge(rowIDs)
      case None =>  new BiDictionary(rowIDs)
    }
    val rowIDDictionary_bcast = sc.broadcast(rowIDDictionary)

    val columnIDDictionary = new BiDictionary(keys = columnIDs)
    val columnIDDictionary_bcast = sc.broadcast(columnIDDictionary)

    val ncol = columnIDDictionary.size
    val nrow = rowIDDictionary.size

    val indexedInteractions =
      elements.map { case (rowID, columnID) =>
        val rowIndex = rowIDDictionary_bcast.value.getOrElse(rowID, -1)
        val columnIndex = columnIDDictionary_bcast.value.getOrElse(columnID, -1)

        rowIndex -> columnIndex
      }
        // group by IDs to form row vectors
        .groupByKey().map { case (rowIndex, columnIndexes) =>
        val row = new RandomAccessSparseVector(ncol)
        for (columnIndex <- columnIndexes) {
          row.setQuick(columnIndex, 1.0)
        }
        rowIndex -> row
      }.asInstanceOf[DrmRdd[Int]]

    // wrap the DrmRdd and a CheckpointedDrm, which can be used anywhere a DrmLike[Int] is needed
    val drmInteractions = drmWrap[Int](indexedInteractions)

    new IndexedDatasetSpark(drmInteractions, rowIDDictionary, columnIDDictionary)
  }

}

