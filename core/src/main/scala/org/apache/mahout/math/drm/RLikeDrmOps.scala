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
import collection._
import JavaConversions._
import org.apache.mahout.math.{Vector, Matrix}
import org.apache.mahout.math.drm.logical._
import org.apache.mahout.math.scalabindings._
import RLikeOps._

class RLikeDrmOps[K](drm: DrmLike[K]) extends DrmLikeOps[K](drm) {

  import RLikeDrmOps._
  import org.apache.mahout.math.scalabindings._

  def +(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = "+")

  def -(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = "-")

  def *(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = "*")

  def /(that: DrmLike[K]): DrmLike[K] = OpAewB[K](A = this, B = that, op = "/")

  def +(that: Double): DrmLike[K] = OpAewUnaryFunc[K](A = this, f = _ + that, evalZeros = true)

  def +:(that: Double): DrmLike[K] = OpAewUnaryFunc[K](A = this, f = that + _, evalZeros = true)

  def -(that: Double): DrmLike[K] = OpAewUnaryFunc[K](A = this, f = _ - that, evalZeros = true)

  def -:(that: Double): DrmLike[K] = OpAewUnaryFunc[K](A = this, f = that - _, evalZeros = true)

  def *(that: Double): DrmLike[K] = OpAewUnaryFunc[K](A = this, f = _ * that)

  def *:(that: Double): DrmLike[K] = OpAewUnaryFunc[K](A = this, f = that * _)

  def ^(that: Double): DrmLike[K] = that match {
    // Special handling of x ^2 and x ^ 0.5: we want consistent handling of x ^ 2 and x * x since
    // pow(x,2) function return results different from x * x; but much of the code uses this
    // interchangeably. Not having this done will create things like NaN entries on main diagonal
    // of a distance matrix.
    case 2.0 ⇒ OpAewUnaryFunc[K](A = this, f = x ⇒ x * x)
    case 0.5 ⇒ OpAewUnaryFunc[K](A = this, f = math.sqrt _)
    case _ ⇒ OpAewUnaryFunc[K](A = this, f = math.pow(_, that))
  }

  def /(that: Double): DrmLike[K] = OpAewUnaryFunc[K](A = this, f = _ / that, evalZeros = that == 0.0)

  def /:(that: Double): DrmLike[K] = OpAewUnaryFunc[K](A = this, f = that / _, evalZeros = true)

  def :%*%[B](that: DrmLike[B]): DrmLike[K] = OpABAnyKey[B,K](A = this.drm, B=that)

  def %*%[B](that: DrmLike[B]): DrmLike[K] = this :%*% that

  def :%*%(that: Matrix): DrmLike[K] = OpTimesRightMatrix[K](A = this.drm, right = that)

  def %*%(that: Matrix): DrmLike[K] = this :%*% that

  def :%*%(that: Vector): DrmLike[K] = OpAx(A = this.drm, x = that)

  def %*%(that: Vector): DrmLike[K] = :%*%(that)

  def t: DrmLike[Int] = OpAtAnyKey(A = drm)

  def cbind(that: DrmLike[K]): DrmLike[K] = OpCbind(A = this.drm, B = that)

  def cbind(that: Double): DrmLike[K] = OpCbindScalar(A = this.drm, x = that, leftBind = false)

  def rbind(that: DrmLike[K]): DrmLike[K] = OpRbind(A = this.drm, B = that)

  /**
   * `rowSums` method for non-int keyed matrices.
   *
   * Slight problem here is the limitation of in-memory representation of Colt's Matrix, which can
   * only have String row labels. Therefore, internally we do ".toString()" call on each key object,
   * and then put it into [[Matrix]] row label bindings, at which point they are coerced to be Strings.
   *
   * This is obviously a suboptimal behavior, so as TODO we have here future enhancements of `collect'.
   *
   * @return map of row keys into row sums, front-end collected.
   */
  def rowSumsMap(): Map[String, Double] = {

    implicit val ktag = drm.keyClassTag

    val m = drm.mapBlock(ncol = 1) { case (keys, block) =>
      keys -> dense(block.rowSums).t
    }.collect
    m.getRowLabelBindings.map { case (key, idx) => key -> m(idx, 0)}
  }
}

class RLikeDrmIntOps(drm: DrmLike[Int]) extends RLikeDrmOps[Int](drm) {

  import org.apache.mahout.math._
  import scalabindings._
  import RLikeDrmOps._

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

  implicit def double2ScalarOps(x: Double) = new DrmDoubleScalarOps(x)

  implicit def drmInt2RLikeOps(drm: DrmLike[Int]): RLikeDrmIntOps = new RLikeDrmIntOps(drm)

  implicit def drm2RLikeOps[K](drm: DrmLike[K]): RLikeDrmOps[K] = new RLikeDrmOps[K](drm)

  implicit def rlikeOps2Drm[K](ops: RLikeDrmOps[K]): DrmLike[K] = ops.drm

  implicit def ops2Drm[K](ops: DrmLikeOps[K]): DrmLike[K] = ops.drm

  implicit def drm2cpops[K](drm: DrmLike[K]): CheckpointedOps[K] = new CheckpointedOps(drm)
}
