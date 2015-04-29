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
import org.apache.mahout.math.indexeddataset
import org.apache.mahout.math.indexeddataset._

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

