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

package org.apache.mahout.math.decompositions

import math._
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import RLikeDrmOps._
import org.apache.mahout.sparkbindings._
import scala.collection.JavaConversions._
import org.apache.spark.SparkContext._
import org.apache.mahout.common.RandomUtils
import org.apache.mahout.math.decompositions.ALS.InCoreResult

object ALSImplicit {

  /**
   * In-core version of ALS implicit.
   * <P/>
   * This is not really performance-optimized method at this point; more like a validation prototype
   * for the distributed one (dalsImplicit).
   * <P/>
   */
  def alsImplicit(
      inCoreC: Matrix,
      c0: Double = 1.0,
      k: Int = 50,
      lambda: Double = 0.0001,
      maxIterations: Int = 10,
      convergenceThreshold: Double = 0.05
      ): ALS.InCoreResult = {

    val rnd = RandomUtils.getRandom()
    val m = inCoreC.nrow
    val n = inCoreC.ncol
    var inCoreV = new DenseMatrix(n, k) := Matrices.symmetricUniformView(n, k, rnd.nextInt()) * 0.01
    var inCoreU = new DenseMatrix(m, k)

    var inCoreD = (inCoreC cloned)
    inCoreD := ((r, c, v) => abs(v))

    var inCoreP = (inCoreC cloned)
    inCoreP := ((r, c, v) => if (v > 0) 1.0 else 0.0)

    // Num non-base confidence entries
    val numPoints = if (convergenceThreshold > 0) inCoreC.foldLeft(0)(_ + _.getNumNonZeroElements) else 0
    var rmseList = List.empty[Double]

    var i = 0
    var stop =  false
    while (i < maxIterations && !stop) {
      updateU(inCoreU, inCoreV, inCoreD, inCoreP, k, lambda, c0)
      updateU(inCoreV, inCoreU, inCoreD.t, inCoreP.t, k, lambda, c0)

      if ( convergenceThreshold > 0 ) {

        // MSE , weighed by confidence of measurement
        val mse = ((inCoreP - inCoreU %*% inCoreV.t) * inCoreC).norm / numPoints
        val rmse = sqrt(mse)

        // Measure relative improvement over previous iteration and bail out if it doesn't exceed
        // minimum convergence threshold.
        if (! rmseList.isEmpty && (rmseList.last - rmse) / rmseList.last <= convergenceThreshold ) {
          stop = true
        }

        // Augment mse list.
        rmseList :+= rmse
      }

      i += 1
    }
    new InCoreResult(inCoreU, inCoreV, rmseList)
  }

  private def updateU(inCoreU: Matrix, inCoreV: Matrix, inCoreD: Matrix, inCoreP: Matrix, k: Int, lambda: Double,
      c0: Double) = {

    val m = inCoreD.nrow

    val c0vtv = (inCoreV.t %*% inCoreV) * c0

    var i = 0
    while (i < m) {
      val d_i = inCoreD(i, ::)
      val p_i = inCoreP(i, ::)
      val n_u = d_i.getNumNonZeroElements
      inCoreU(i, ::) = solve(
        a = c0vtv + (inCoreV.t %*%: diagv(inCoreD(i, ::))) %*% inCoreV + diag(n_u * lambda, k),
        b = (inCoreV.t %*%: diagv(d_i + c0)) %*% p_i
      )

      i += 1
    }
    inCoreU
  }

  /**
   * See MAHOUT-1365 for details.
   *
   * Implicit feedback has two inputs: the preference matrix and confidence matrix. Preference on
   * user/item interaciton is always 1 (prefer) or 0 (do not prefer). Confidence encodes informal
   * weighing on how confident we are about preference. Things we don't have any observations for
   * are usually encoded with baseline confidence (c0 parameter).Things we do get observations for
   * have higher confidence. Thus, it is assumed that all confidence cells are greater or equal c0.
   * <P/>
   *
   * Next, for the purposes of compactness of input, we sparsify and combine both inputs in the
   * following way.
   * <P/>
   *
   * First, we assume that we don't have observations for every combination of user/item pair, so
   * (C-c0) matrix is sparse.
   * <P/>
   *
   * Second, we can use sign to encode preferences, i.e.
   * <pre>
   * C*(i,j) = (C(i,j)-c0) if P(i,j)==1;
   * and
   * C*(i,j) = -(C(i,j)-c0) if P(i,j)=0.
   * </pre>
   *
   * Note in that we assume all entries with baseline confidence having P = 0 (no preference).
   * <P/>
   *
   * In reality this input vectorizes without need to ever form dense inputs since we only encode
   * entries that do have any observations.
   * <P/>
   *
   * @param drmC confidence/preference encoded input C* as explained above
   * @param c0 baseline confidence value.
   * @param k factorization rank (~50...100 is probably enough)
   * @param lambda regularization for this iteration
   * @param maxIterations maximum iterations to run
   * @param convergenceTreshold reserved, not used at this point
   */
  def dalsImplicit(
      drmC: DrmLike[Int],
      c0: Double = 1.0,
      k: Int = 50,
      lambda: Double = 0.0001,
      maxIterations: Int = 10,
      convergenceTreshold: Double = 0.10
      ): ALS.Result[Int] = {
    val drmA = drmC
    val drmAt = drmC.t

    // cbind(U,A):
    var drmUA = drmA.mapBlock(ncol = k + drmA.ncol) {
      case (keys, block) =>
        val uaBlock = block.like(block.nrow, block.ncol + k)
        //        uaBlock(::, 0 until k) :=
        //            Matrices.symmetricUniformView(uaBlock.nrow, k, RandomUtils.getRandom().nextInt()) * 0.01
        uaBlock(::, k until uaBlock.ncol) := block
        keys -> uaBlock
    }

    // cbind(V,A'):
    var drmVAt = drmAt.mapBlock(ncol = k + drmAt.ncol) {
      case (keys, block) =>
        val vatBlock = block.like(block.nrow, block.ncol + k)
        val rnd = RandomUtils.getRandom()
        vatBlock(::, 0 until k) :=
            Matrices.symmetricUniformView(vatBlock.nrow, k, rnd.nextInt()) * 0.01
        vatBlock(::, k until vatBlock.ncol) := block
        keys -> vatBlock
    }
        .checkpoint()

    var i = 0
    var stop = false

    var drmVAtOld: DrmLike[Int] = null
    var drmUAOld: DrmLike[Int] = null

    while (i < maxIterations && !stop) {

      // Update U-A, relinquish stuff explicitly from block manager to alleviate GC concerns and swaps
      if (drmUAOld != null) drmUAOld.uncache()
      drmUAOld = drmUA
      drmUA = updateUA(drmUA, drmVAt, k, lambda, c0)

      // Update V-A'
      if (drmVAtOld != null) drmVAtOld.uncache()
      drmVAtOld = drmVAt
      drmVAt = updateUA(drmVAt, drmUA, k, lambda, c0)

      i += 1
    }

    new ALS.Result[Int](drmU = drmUA(::, 0 until k), drmV = drmVAt(::, 0 until k), iterationsRMSE = Iterable())
  }

  private def updateUA(drmUA: DrmLike[Int], drmVAt: DrmLike[Int], k: Int, lambda: Double, c0: Double): DrmLike[Int] = {

    implicit val ctx = drmUA.context

    val n = drmUA.ncol
    val drmV = drmVAt(::, 0 until k)

    val c0vtvBcast = drmBroadcast((drmV.t %*% drmV).collect * c0)

    val uaRdd = drmUA.rdd.cogroup(other = generateMessages(drmVAt, k)).map {
      case (uKey, (uavecs, msgs)) =>

        val uavec = uavecs.head
        val urow = uavec(0 until k)
        val arow = uavec(k until n)

        val vsum = new DenseVector(k)

        // We will modify m here; so we must clone the broadcast value to avoid side effects.
        // The modifications to the broadcast value seem to be visible in other closures, at least
        // in the "local" master tests.
        val m: Matrix = c0vtvBcast.value cloned

        var n_u = 0
        msgs.foreach {
          case (vKey, vrow) =>
            val c_star = arow(vKey)
            assert(c_star != 0.0)

            // (1) if arow[vKey] > 0 means p == 1, in which case we update accumulator vsum.
            if (c_star > 0) vsum += vrow * (c0 + c_star)

            // (2) Update m: confidence value is absolute of c*.
            // We probably could do that more efficiently here?
            m += vrow * abs(c_star) cross vrow
            n_u += 1
        }

        if (n_u > 0) {
          m += diag(n_u * lambda, k)
        }

        // Update u-vec
        urow := solve(a = m, b = vsum)
        uKey -> uavec
    }

    drmWrap(uaRdd, ncol = n, cacheHint = CacheHint.MEMORY_AND_DISK)
  }

  // This generates RDD of messages in a form RDD[destVIndex ->(srcUIindex,u-row-vec)]
  private def generateMessages(drmUA: DrmLike[Int], k: Int) = {

    val n = drmUA.ncol
    // Now we delve into Spark-specific processing.
    drmUA.rdd.flatMap {
      case (rowKey, row) =>
        val uvec = new DenseVector(k) := row(0 until k)
        val payload = rowKey -> uvec
        val cvec = row(k until n)
        cvec.nonZeroes().map(_.index() -> payload)
    }
  }

}
