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
import org.apache.mahout.math.drm.logical._

import org.apache.mahout.h2obindings.drm._

import water._
import water.fvec._

object H2OEngine extends DistributedEngine {
  def colMeans[K:ClassTag](drm: CheckpointedDrm[K]): Vector =
    H2OHelper.colMeans (drm.frame)

  def colSums[K:ClassTag](drm: CheckpointedDrm[K]): Vector =
    H2OHelper.colSums (drm.frame)

  def norm[K: ClassTag](drm: CheckpointedDrm[K]): Double =
    H2OHelper.sumSqr (drm.frame)

  def numNonZeroElementsPerColumn[K: ClassTag](drm: CheckpointedDrm[K]): Vector =
    H2OHelper.nonZeroCnt (drm.frame)

  def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix] =
    new H2OBCast(m)

  def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector] =
    new H2OBCast(v)

  /* XXX - H2O parser does not support seqfile */
  def drmFromHDFS(path: String, parMin: Int = 0)(implicit dc: DistributedContext): CheckpointedDrm[_] =
    new CheckpointedDrmH2O (H2OHelper.frame_from_file (path), dc)

  def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Int] =
    new CheckpointedDrmH2O (H2OHelper.empty_frame (nrow, ncol, numPartitions), dc)

  def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Long] =
    new CheckpointedDrmH2O (H2OHelper.empty_frame (nrow, ncol, numPartitions), dc)

  def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Int] =
    new CheckpointedDrmH2O (H2OHelper.frame_from_matrix (m, numPartitions), dc)

  def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[String] =
    new CheckpointedDrmH2O (H2OHelper.frame_from_matrix (m, numPartitions), dc)

  def toPhysical[K:ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] =
    new CheckpointedDrmH2O (tr2phys (plan), plan.context)

  // H2O specific

  private def tr2phys[K: ClassTag](oper: DrmLike[K]): Frame = ???

  implicit def cp2cph2o[K:ClassTag](drm: CheckpointedDrm[K]): CheckpointedDrmH2O[K] = drm.asInstanceOf[CheckpointedDrmH2O[K]]
}
