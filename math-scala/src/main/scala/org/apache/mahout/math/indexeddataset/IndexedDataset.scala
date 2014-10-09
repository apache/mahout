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

package org.apache.mahout.math.indexeddataset

import com.google.common.collect.BiMap
import org.apache.mahout.math.drm.{DistributedContext, CheckpointedDrm}
import org.apache.mahout.math.indexeddataset

/**
  * Wraps a [[org.apache.mahout.spark.CheckpointedDrm]] object with two [[com.google.common.collect.BiMap]]s to store
  * ID/label translation dictionaries.
  * The purpose of this class is to wrap a DrmLike[C] with bidirectional ID mappings so
  * a user specified label or ID can be stored and mapped to and from the [[scala.Int]] ordinal ID
  * used internal to Mahout Core code.
  *
  * @todo Often no need for both or perhaps either dictionary, so save resources by allowing
  *       to be not created when not needed.
  */

trait IndexedDataset {
  val matrix: CheckpointedDrm[Int]
  val rowIDs: BiMap[String,Int]
  val columnIDs: BiMap[String,Int]

  /**
   * Adds the equivalent of blank rows to the sparse CheckpointedDrm, which only changes the row cardinality value.
   * No changes are made to the underlying drm. We must allow the row dimension to be adjusted in the case where
   * the data read in is incomplete and we learn this afterwards.
   * @param n number to use for row cardinality, should be larger than current
   * @note should be done before any optimizer actions are performed on the matrix or you'll get unpredictable
   *       results.
   */
  def newRowCardinality(n: Int): IndexedDataset

  /**
   * Write a text delimited file(s) with the row and column IDs from dictionaries.
   * @param dest
   * @param schema
   */
  def dfsWrite(dest: String, schema: Schema)(implicit sc: DistributedContext): Unit

  /**
   * Factory method, creates an extended class
   */

  def create(matrix: CheckpointedDrm[Int], rowIDs: BiMap[String,Int], columnIDs: BiMap[String,Int]):
    IndexedDataset
}

