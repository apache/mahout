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

import scala.reflect.ClassTag
import org.apache.mahout.math.{Vector, Matrix}
import org.apache.mahout.math.drm.logical._

class RLikeDrmOps[K: ClassTag](drm: DrmLike[K]) extends DrmLikeOps[K](drm) {

  import RLikeDrmOps._

  def +(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = "+")

  def -(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = "-")

  def *(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = "*")

  def /(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = "/")

  def +(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "+")

  def +:(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "+")

  def -(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "-")

  def -:(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "-:")

  def *(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "*")

  def *:(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "*")

  def /(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "/")

  def /:(that: Double): DrmLike[K] = OpAewScalar[K](A = this, scalar = that, op = "/:")

  def :%*%(that: DrmLike[Int]): DrmLike[K] = OpAB[K](A = this.drm, B = that)

  def %*%[B: ClassTag](that: DrmLike[B]): DrmLike[K] = OpABAnyKey[B, K](A = this.drm, B = that)

  def %*%(that: DrmLike[Int]): DrmLike[K] = this :%*% that

  def :%*%(that: Matrix): DrmLike[K] = OpTimesRightMatrix[K](A = this.drm, right = that)

  def %*%(that: Matrix): DrmLike[K] = this :%*% that

  def :%*%(that: Vector): DrmLike[K] = OpAx(A = this.drm, x = that)

  def %*%(that: Vector): DrmLike[K] = :%*%(that)

  def t: DrmLike[Int] = OpAtAnyKey(A = drm)

  def cbind(that: DrmLike[K]) = OpCbind(A = this.drm, B = that)

  def rbind(that: DrmLike[K]) = OpRbind(A = this.drm, B = that)
}

class RLikeDrmIntOps(drm: DrmLike[Int]) extends RLikeDrmOps[Int](drm) {

  import org.apache.mahout.math._
  import scalabindings._
  import RLikeOps._
  import RLikeDrmOps._
  import scala.collection.JavaConversions._

  override def t: DrmLike[Int] = OpAt(A = drm)

  def %*%:[K: ClassTag](that: DrmLike[K]): DrmLike[K] = OpAB[K](A = that, B = this.drm)

  def %*%:(that: Matrix): DrmLike[Int] = OpTimesLeftMatrix(left = that, A = this.drm)

  /** Row sums. This is of course applicable to Int-keyed distributed matrices only. */
  def rowSums(): Vector = {
    drm.mapBlock(ncol = 1) { case (keys, block) =>
      // Collect block-wise rowsums and output them as one-column matrix.
      keys -> dense(block.rowSums).t
    }
      .collect(::, 0)
  }

  /** Counts the non-zeros elements in each row returning a vector of the counts */
  def numNonZeroElementsPerRow(): Vector = {
    drm.mapBlock(ncol = 1) { case (keys, block) =>
      // Collect block-wise row non-zero counts and output them as a one-column matrix.
      keys -> dense(block.numNonZeroElementsPerRow).t
    }
      .collect(::, 0)
  }

  /** Row means */
  def rowMeans(): Vector = {
    drm.mapBlock(ncol = 1) { case (keys, block) =>
      // Collect block-wise row means and output them as one-column matrix.
      keys -> dense(block.rowMeans).t
    }
        .collect(::, 0)
  }

  /** Return diagonal vector */
  def diagv: Vector = {
    require(drm.ncol == drm.nrow, "Must be square to extract diagonal")
    drm.mapBlock(ncol = 1) { case (keys, block) =>
      keys -> dense(for (r <- block.view) yield r(keys(r.index))).t
    }
        .collect(::, 0)
  }

}

object RLikeDrmOps {

  implicit def double2ScalarOps(x:Double) = new DrmDoubleScalarOps(x)

  implicit def drmInt2RLikeOps(drm: DrmLike[Int]): RLikeDrmIntOps = new RLikeDrmIntOps(drm)

  implicit def drm2RLikeOps[K: ClassTag](drm: DrmLike[K]): RLikeDrmOps[K] = new RLikeDrmOps[K](drm)

  implicit def rlikeOps2Drm[K: ClassTag](ops: RLikeDrmOps[K]): DrmLike[K] = ops.drm

  implicit def ops2Drm[K: ClassTag](ops: DrmLikeOps[K]): DrmLike[K] = ops.drm

  // Removed in move to 1.2.1 PR #74 https://github.com/apache/mahout/pull/74/files
  // Not sure why.
  // implicit def cp2cpops[K: ClassTag](cp: CheckpointedDrm[K]): CheckpointedOps[K] = new CheckpointedOps(cp)

  /**
   * This is probably dangerous since it triggers implicit checkpointing with default storage level
   * setting.
   */
  implicit def drm2cpops[K: ClassTag](drm: DrmLike[K]): CheckpointedOps[K] = new CheckpointedOps(drm.checkpoint())
}
