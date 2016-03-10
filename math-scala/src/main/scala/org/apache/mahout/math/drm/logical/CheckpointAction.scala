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

package org.apache.mahout.math.drm.logical

import scala.util.Random
import org.apache.mahout.math.drm._

/** Implementation of distributed expression checkpoint and optimizer. */
abstract class CheckpointAction[K] extends DrmLike[K] {

  protected[mahout] lazy val partitioningTag: Long = Random.nextLong()

  private[mahout] var cp:Option[CheckpointedDrm[K]] = None

  def isIdenticallyPartitioned(other:DrmLike[_]) =
    partitioningTag!= 0L && partitioningTag == other.partitioningTag

  /**
   * Action operator -- does not necessary means Spark action; but does mean running BLAS optimizer
   * and writing down Spark graph lineage since last checkpointed DRM.
   */
  def checkpoint(cacheHint: CacheHint.CacheHint): CheckpointedDrm[K] = cp match {
    case None =>
      implicit val cpTag = this.keyClassTag
      val plan = context.optimizerRewrite(this)
      val physPlan = context.toPhysical(plan, cacheHint)
      cp = Some(physPlan)
      physPlan
    case Some(cp) => cp
  }

}

