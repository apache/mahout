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

package org.apache.mahout.drivers

import com.google.common.collect.BiMap
import org.apache.mahout.math.drm.DrmLike

/**
  * Wraps a [[org.apache.mahout.sparkbindings.drm.DrmLike]] object with two [[com.google.common.collect.BiMap]]s to store ID/label translation dictionaries.
  * The purpose of this class is to wrap a DrmLike[C] with bidirectional ID mappings so
  * a user specified label or ID can be stored and mapped to and from the [[scala.Int]] ordinal ID
  * used internal to Mahout Core code.
  *
  * Example: For a transpose job the [[org.apache.mahout.drivers.IndexedDataset#matrix]]: [[org.apache.mahout.sparkbindings.drm.DrmLike]] is passed into the DSL code
  * that transposes the values, then a resulting [[org.apache.mahout.drivers.IndexedDataset]] is created from the transposed DrmLike object with swapped dictionaries (since the rows and columns are transposed). The new
  * [[org.apache.mahout.drivers.IndexedDataset]] is returned.
  *
  * @param matrix  DrmLike[Int], representing the distributed matrix storing the actual data.
  * @param rowIDs BiMap[String, Int] storing a bidirectional mapping of external String ID to
  *                  and from the ordinal Mahout Int ID. This one holds row labels
  * @param columnIDs BiMap[String, Int] storing a bidirectional mapping of external String
  *                  ID to and from the ordinal Mahout Int ID. This one holds column labels
  */

case class IndexedDataset(matrix: DrmLike[Int], rowIDs: BiMap[String,Int], columnIDs: BiMap[String,Int]) {
}

/**
  * Companion object for the case class [[org.apache.mahout.drivers.IndexedDataset]] primarily used to get a secondary constructor for
  * making one [[org.apache.mahout.drivers.IndexedDataset]] from another. Used when you have a factory like [[org.apache.mahout.drivers.IndexedDatasetStore]]
  * {{{
  * val indexedDataset = IndexedDataset(indexedDatasetStore.read)
  * }}}
  */

object IndexedDataset {
  /** Secondary constructor for [[org.apache.mahout.drivers.IndexedDataset]] */
  def apply(id2: IndexedDataset) = new IndexedDataset(id2.matrix,  id2.rowIDs, id2.columnIDs)
}
