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
import org.apache.mahout.math.drm.CheckpointedDrm
import org.apache.mahout.sparkbindings.drm.CheckpointedDrmSpark
import org.apache.mahout.sparkbindings._

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
  * @todo Often no need for both or perhaps either dictionary, so save resources by allowing
  *       to be not created when not needed.
  */

case class IndexedDataset(var matrix: CheckpointedDrm[Int], var rowIDs: BiMap[String,Int], var columnIDs: BiMap[String,Int]) {

  // we must allow the row dimension to be adjusted in the case where the data read in is incomplete and we
  // learn this afterwards

  /**
   * Adds the equivalent of blank rows to the sparse CheckpointedDrm, which only changes the row cardinality value.
   * No physical changes are made to the underlying drm.
   * @param n number to use for row carnindality, should be larger than current
   * @note should be done before any BLAS optimizer actions are performed on the matrix or you'll get unpredictable
   *       results.
   */
  def newRowCardinality(n: Int): IndexedDataset = {
    assert(n > -1)
    assert( n >= matrix.nrow)
    val drmRdd = matrix.asInstanceOf[CheckpointedDrmSpark[Int]].rdd
    val ncol = matrix.ncol
    val newMatrix = drmWrap[Int](drmRdd, n, ncol)
    new IndexedDataset(newMatrix, rowIDs, columnIDs)
  }
}

/**
  * Companion object for the case class [[org.apache.mahout.drivers.IndexedDataset]] primarily used to get a secondary constructor for
  * making one [[org.apache.mahout.drivers.IndexedDataset]] from another. Used when you have a factory like [[org.apache.mahout.drivers.IndexedDatasetStore]]
  * {{{
  *   val indexedDataset = IndexedDataset(indexedDatasetReader.readTuplesFrom(source))
  * }}}
  */

object IndexedDataset {
  /** Secondary constructor for [[org.apache.mahout.drivers.IndexedDataset]] */
  def apply(id2: IndexedDataset) = new IndexedDataset(id2.matrix,  id2.rowIDs, id2.columnIDs)
}
