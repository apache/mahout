/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.h2obindings

import scala.reflect._
import org.apache.mahout.math._
import org.apache.mahout.math.drm._

object H2OEngine extends DistributedEngine {
  def colMeans[K:ClassTag](drm: CheckpointedDrm[K]): Vector = ???
  def colSums[K:ClassTag](drm: CheckpointedDrm[K]): Vector = ???
  def norm[K: ClassTag](drm: CheckpointedDrm[K]): Double = ???
  def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix] = ???
  def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector] = ???
  def drmFromHDFS(path: String, parMin: Int = 0)(implicit dc: DistributedContext): CheckpointedDrm[_] = ???
  def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Int] = ???
  def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Long] = ???
  def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Int] = ???
  def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[String] = ???
  def toPhysical[K:ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] = ???
}
