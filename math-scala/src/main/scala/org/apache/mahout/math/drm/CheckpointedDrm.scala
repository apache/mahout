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

package org.apache.mahout.math.drm

import org.apache.mahout.math.Matrix
import scala.reflect.ClassTag

/**
 * Checkpointed DRM API. This is a matrix that has optimized RDD lineage behind it and can be
 * therefore collected or saved.
 * @tparam K matrix key type (e.g. the keys of sequence files once persisted)
 */
trait CheckpointedDrm[K] extends DrmLike[K] {

  def collect: Matrix

  def dfsWrite(path: String)

  /** If this checkpoint is already declared cached, uncache. */
  def uncache(): this.type

  /**
   * Explicit extraction of key class Tag since traits don't support context bound access; but actual
   * implementation knows it
   */
  def keyClassTag: ClassTag[K]


  /** changes the number of rows without touching the underlying data */
  def newRowCardinality(n: Int): CheckpointedDrm[K]

}
