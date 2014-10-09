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

import com.google.common.collect.BiMap
import org.apache.mahout.drivers.TextDelimitedIndexedDatasetWriter
import org.apache.mahout.math.drm.{DistributedContext, CheckpointedDrm}
import org.apache.mahout.math.indexeddataset
import org.apache.mahout.math.indexeddataset.{DefaultIndexedDatasetWriteSchema, Reader, Schema, IndexedDataset}

/**
 * Spark implementation of [[indexeddataset.IndexedDataset]] providing the Spark specific dfsWrite method
 */

class IndexedDatasetSpark(val matrix: CheckpointedDrm[Int], val rowIDs: BiMap[String,Int], val columnIDs: BiMap[String,Int])
  extends IndexedDataset {

  /** secondary constructor enabling immutability */
  def this(id2: IndexedDatasetSpark){
    this(id2.matrix, id2.rowIDs, id2.columnIDs)
  }

  override def create(matrix: CheckpointedDrm[Int], rowIDs: BiMap[String,Int], columnIDs: BiMap[String,Int]):
    IndexedDatasetSpark = {
    new IndexedDatasetSpark(matrix, rowIDs, columnIDs)
  }

  /**
   * Adds the equivalent of blank rows to the sparse CheckpointedDrm, which only changes the row cardinality value.
   * No changes are made to the underlying drm. Implements the core method [[indexeddataset.IndexedDataset#dfsWrite]]
   * @param n number to use for row carnindality, should be larger than current
   * @note should be done before any optimizer actions are performed on the matrix or you'll get unpredictable
   *       results.
   */
  override def newRowCardinality(n: Int): IndexedDatasetSpark = {
    new IndexedDatasetSpark(matrix.newRowCardinality(n), rowIDs, columnIDs)
  }

  /** implements the core method [[indexeddataset.IndexedDataset#dfsWrite]]*/
  override def dfsWrite(dest: String, schema: Schema = DefaultIndexedDatasetWriteSchema)
      (implicit sc: DistributedContext):
    Unit = {
    val writer = new TextDelimitedIndexedDatasetWriter(schema)(sc)
    writer.writeTo(this, dest)
  }
}

