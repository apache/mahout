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
      inCoreA: Matrix,
      c0: Double = 1.0,
      k: Int = 50,
      lambda: Double = 0.0001,
      maxIterations: Int = 10,
      convergenceThreshold: Double = 0.05,
      wr: Boolean = true
      ): ALS.InCoreResult = {

    val rnd = RandomUtils.getRandom()
    val m = inCoreA.nrow
    val n = inCoreA.ncol
    var inCoreV = new DenseMatrix(n, k) := Matrices.symmetricUniformView(n, k, rnd.nextInt()) * 0.01
    var inCoreU = new DenseMatrix(m, k)

    var inCoreD = (inCoreA cloned)
    inCoreD := ((r, c, v) => abs(v))

    var inCoreP = (inCoreA cloned)
    inCoreP := ((r, c, v) => if (v > 0) 1.0 else 0.0)

    // Num non-base confidence entries
    val numPoints = if (convergenceThreshold > 0) inCoreA.foldLeft(0)(_ + _.getNumNonZeroElements) else 0
    var rmseList = List.empty[Double]

    var i = 0
    var stop = false

    // A sparse matrix that contains true confidences (C_0 + C*) except where no observations. We
    // need this for computing RMSE the same way we do for the distributed version.
    val inCoreCC0 = if (convergenceThreshold > 0) {
      (inCoreD cloned) := ((r, c, v) => if (v > 0.0) v + c0 else 0.0)
    } else {
      null: Matrix
    }

    while (i < maxIterations && !stop) {
      updateU(inCoreU, inCoreV, inCoreD, inCoreP, k, lambda, c0, wr)
      updateU(inCoreV, inCoreU, inCoreD.t, inCoreP.t, k, lambda, c0, wr)

      if (convergenceThreshold > 0) {

        // MSE , weighed by confidence of measurement and ifnoring no-observation c0 items
        val mse = ((inCoreP - inCoreU %*% inCoreV.t) * inCoreCC0).norm / numPoints
        val rmse = sqrt(mse)

        // Measure relative improvement over previous iteration and bail out if it doesn't exceed
        // minimum convergence threshold.
        if (!rmseList.isEmpty && (rmseList.last - rmse) / rmseList.last <= convergenceThreshold) {
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
      c0: Double, wr: Boolean) = {

    val m = inCoreD.nrow

    val c0vtv = (inCoreV.t %*% inCoreV) * c0

    var i = 0
    while (i < m) {
      val d_i = inCoreD(i, ::)
      val p_i = inCoreP(i, ::)
      val n_u = if (wr) d_i.getNumNonZeroElements else 1.0
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
   * @param drmA confidence/preference encoded input C* as explained above
   * @param c0 baseline confidence value.
   * @param k factorization rank (~50...100 is probably enough)
   * @param lambda regularization for this iteration
   * @param maxIterations maximum iterations to run
   * @param convergenceThreshold reserved, not used at this point
   * @param wr use/do not use weighed regularization
   */
  def dalsImplicit(
      drmA: DrmLike[Int],
      c0: Double = 1.0,
      k: Int = 50,
      lambda: Double = 0.0001,
      maxIterations: Int = 10,
      convergenceThreshold: Double = 0.05,
      wr: Boolean = true
      ): ALS.Result[Int] = {

    val drmAt = drmA.t

    // cbind(U,A):
    var drmUA = drmA.mapBlock(ncol = k + drmA.ncol) {
      case (keys, block) =>
        val uaBlock = block.like(block.nrow, block.ncol + k)
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
    var rmseList = List.empty[Double]

    while (i < maxIterations && !stop) {

      // Update U-A, relinquish stuff explicitly from block manager to alleviate GC concerns and swaps
      if (drmUAOld != null) drmUAOld.uncache()
      drmUAOld = drmUA
      drmUA = updateUA(drmUA, drmVAt, k, lambda, c0, wr)

      // Update V-A'
      if (drmVAtOld != null) drmVAtOld.uncache()
      drmVAtOld = drmVAt
      drmVAt = updateUA(drmVAt, drmUA, k, lambda, c0, wr)

      if (convergenceThreshold > 0) {

        val rmse = drmse(
          drmUC = drmUA,
          drmVCt = drmVAt,
          c0 = c0,
          k = k
        )

        // Measure relative improvement over previous iteration and bail out if it doesn't exceed
        // minimum convergence threshold.
        if (!rmseList.isEmpty && (rmseList.last - rmse) / rmseList.last <= convergenceThreshold) {
          stop = true
        }

        // Augment mse list.
        rmseList :+= rmse
      }


      i += 1
    }

    new ALS.Result[Int](drmU = drmUA(::, 0 until k), drmV = drmVAt(::, 0 until k), iterationsRMSE = rmseList)
  }

  // Returns (P,C) tuple out of encoded hybrid C/P value
  private def decodePC(chybrid: Double) = if (chybrid > 0) 1.0 -> chybrid else 0.0 -> -chybrid


  /**
   * Compute distributed rmse for non-zero elemenst of C only, which are interpreted the same way as
   * in dalsImplicit()
   *
   * @param drmC Confidence matrix in hybrid encoding (same incoding as the input to implicit feedback)
   * @param drmU U matrix, m x k
   * @param drmV V matrix, n x k
   * @param c0 baseline confidence
   * @return rmse
   */
  def drmse(drmC: DrmLike[Int], drmU: DrmLike[Int], drmV: DrmLike[Int], c0: Double): Double =
    drmse(
      drmUC = drmU cbind drmC,
      drmVCt = drmV cbind drmC.t,
      k = drmU.ncol,
      c0 = c0
    )

  /**
   * Compute distributed RMSE for non-zero elements of C only, which are interpreted the same way as
   * in dalsImplicit().
   * <P/>
   *
   * Slightly different api. Sometimes U cbind C and V cbind C' are already available.
   * <P/>
   *
   * @param drmUC drmU cbind drmC
   * @param drmVCt drmV cbind drmC.t
   */
  def drmse(drmUC: DrmLike[Int], drmVCt: DrmLike[Int], k: Int, c0: Double): Double = {

    // Not so subtle problem here is that U %*% V.t would create a matrix much larger than we need.
    // We don't want to compute data points for the entire U %*% V.t to be able to derive RMSE.
    val m = safeToNonNegInt(drmUC.nrow)
    val n = safeToNonNegInt(drmVCt.nrow)

    // Sum squared residuals and count:
    val (sse, cnt) = drmUC.rdd

        // Prepare messages in the form of v-index -> (u-index -> uvec)
        .flatMap {
      case (key, vec) =>

        // Clone V-vector as dense
        val uvec = dvec(fromV = vec(0 until k))
        // C-row
        val crow = vec(k until n)

        // Output flatmap iterator over messages. This should work well for sparse A input (which
        // it is).
        for (crowEl <- crow.nonZeroes().view) yield crowEl.index() -> (key -> uvec)
    }

        // Join messages with RHS that is cbind(V,C')
        .cogroup(other = drmVCt.rdd)

        // Compute (sumSquaredResiduals -> residualCount):
        .map {
      case (vkey, (msgs, vctSeq)) =>

        if (vctSeq.isEmpty) {
          0.0 -> 0
        } else {
          val vvec = vctSeq.head(0 until k)
          val ccol = vctSeq.head(k until k + m)

          // Squared residuals
          ((0.0, 0) /: msgs) {
            case ((acc, cnt), (ukey, uvec)) =>

              // Decode relevant p (preference) and c* (confidence over baseline) values:
              val (p, cstar) = decodePC(ccol(ukey))

              // We compute error as prediction minus p in the source, multiplied (weighed) by originally
              // rated confidence, c0 + c*:
              val err = ((uvec dot vvec) - p) * (cstar + c0)

              // Accumulate squared error, increase residual count
              (acc + err * err, cnt + 1)
          }
        }
    }

        // Sum (sse, cnt) up between partitions, tasks
        .reduce {
      case ((acc1, cnt1), (acc2, cnt2)) => (acc1 + acc2, cnt1 + cnt2)
    }

    assert(cnt > 0)

    // This would be MSE
    sqrt(sse / cnt)
  }

  private def updateUA(drmUA: DrmLike[Int], drmVAt: DrmLike[Int], k: Int, lambda: Double, c0: Double,
      wr: Boolean): DrmLike[Int] = {

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
            val (p, c_star) = decodePC(arow(vKey))

            assert(c_star != 0.0)

            // (1) if arow[vKey] > 0 means p == 1, in which case we update accumulator vsum.
            if (p > 0) vsum += vrow * (c0 + c_star)

            // (2) Update m: confidence value is absolute of c*.
            // We probably could do that more efficiently here?
            m += (vrow * c_star) cross vrow
            n_u += 1
        }

        if (n_u > 0) {
          // If we are asked not to use weighed regularization, don't.
          if (!wr) n_u = 1
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
        val uvec = dvec(fromV = row(0 until k))
        val payload = rowKey -> uvec
        val cvec = row(k until n)
        cvec.nonZeroes().map(_.index() -> payload)
    }
  }

}
