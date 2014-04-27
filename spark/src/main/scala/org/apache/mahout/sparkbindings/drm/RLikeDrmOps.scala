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

package org.apache.mahout.sparkbindings.drm

import scala.reflect.ClassTag
import org.apache.mahout.sparkbindings.drm.plan._
import org.apache.mahout.math.{Matrices, SparseColumnMatrix, Vector, Matrix}
import org.apache.mahout.sparkbindings.drm.plan.OpTimesLeftMatrix
import org.apache.mahout.sparkbindings.drm.plan.OpAt
import org.apache.mahout.sparkbindings.drm.plan.OpAB
import org.apache.mahout.sparkbindings.drm.plan.OpTimesRightMatrix
import org.apache.hadoop.io.Writable
import org.apache.spark.SparkContext._

class RLikeDrmOps[K: ClassTag](drm: DrmLike[K]) extends DrmLikeOps[K](drm) {

  import RLikeDrmOps._

  def +(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = '+')

  def -(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = '-')

  def *(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = '*')

  def /(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = '/')

  def +(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "+")

  def -(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "-")

  def -:(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "-:")

  def *(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "*")

  def /(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "/")

  def /:(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "/:")

  def :%*%(that: DrmLike[Int]): DrmLike[K] = OpAB[K](A = this.drm, B = that)

  def %*%[B:ClassTag](that:DrmLike[B]):DrmLike[K] = OpABAnyKey[B,K](A=this.drm, B=that)

  def %*%(that: DrmLike[Int]): DrmLike[K] = this :%*% that

  def :%*%(that: Matrix): DrmLike[K] = OpTimesRightMatrix[K](A = this.drm, right = that)

  def %*%(that: Matrix): DrmLike[K] = this :%*% that

  def :%*%(that: Vector): DrmLike[K] = OpAx(A = this.drm, x = that)

  def %*%(that: Vector): DrmLike[K] = :%*%(that)

  def t: DrmLike[Int] = OpAtAnyKey(A = drm)
}

class RLikeDrmIntOps(drm: DrmLike[Int]) extends RLikeDrmOps[Int](drm) {

  override def t: DrmLike[Int] = OpAt(A = drm)

  def %*%:[K: ClassTag](that: DrmLike[K]): DrmLike[K] = OpAB[K](A = that, B = this.drm)

  def %*%:(that: Matrix): DrmLike[Int] = OpTimesLeftMatrix(left = that, A = this.drm)


}

object RLikeDrmOps {
  implicit def drmInt2RLikeOps(drm: DrmLike[Int]): RLikeDrmIntOps = new RLikeDrmIntOps(drm)

  implicit def drm2RLikeOps[K: ClassTag](drm: DrmLike[K]): RLikeDrmOps[K] = new RLikeDrmOps[K](drm)

  implicit def rlikeOps2Drm[K: ClassTag](ops: RLikeDrmOps[K]): DrmLike[K] = ops.drm

  implicit def ops2Drm[K: ClassTag](ops: DrmLikeOps[K]): DrmLike[K] = ops.drm

  implicit def cp2cpops[K:ClassTag](cp:CheckpointedDrm[K]):CheckpointedOps[K] = new CheckpointedOps(cp)

  /**
   * This is probably dangerous since it triggers implicit checkpointing with default storage level
   * setting.
   */
  implicit def drm2cpops[K:ClassTag](drm:DrmLike[K]):CheckpointedOps[K] = new CheckpointedOps(drm.checkpoint())
}
