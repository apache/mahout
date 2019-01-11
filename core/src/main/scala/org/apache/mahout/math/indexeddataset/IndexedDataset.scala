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

import org.apache.mahout.math.drm.{DistributedContext, CheckpointedDrm}

/**
 * Wrap an  [[org.apache.mahout.math.drm.DrmLike]] with bidirectional ID mappings [[org.apache.mahout.math.indexeddataset.BiDictionary]]
 * so a user specified labels/IDs can be stored and mapped to and from the Mahout Int ID used internal to Mahout
 * core code.
 * @todo Often no need for both or perhaps either dictionary, so save resources by allowing to be not created
 *       when not needed.
 */

trait IndexedDataset {
  val matrix: CheckpointedDrm[Int]
  val rowIDs: BiDictionary
  val columnIDs: BiDictionary

  /**
   * Write a text delimited file(s) with the row and column IDs from dictionaries.
   * @param dest write location, usually a directory
   * @param schema params to control writing
   * @param sc the [[org.apache.mahout.math.drm.DistributedContext]] used to do a distributed write
   */
  def dfsWrite(dest: String, schema: Schema)(implicit sc: DistributedContext): Unit

  /** Factory method, creates the extending class  and returns a new instance */
  def create(matrix: CheckpointedDrm[Int], rowIDs: BiDictionary, columnIDs: BiDictionary):
    IndexedDataset

  /**
   * Adds the equivalent of blank rows to the sparse CheckpointedDrm, which only changes the row cardinality value.
   * No changes are made to the underlying drm.
   * @param n number to use for new row cardinality, should be larger than current
   * @return a new IndexedDataset or extending class with new cardinality
   * @note should be done before any optimizer actions are performed on the matrix or you'll get unpredictable
   *       results.
   */
  def newRowCardinality(n: Int): IndexedDataset = {
    // n is validated in matrix
    this.create(matrix.newRowCardinality(n), rowIDs, columnIDs)
  }

}

