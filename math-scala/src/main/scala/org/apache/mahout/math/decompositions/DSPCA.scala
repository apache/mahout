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

import org.apache.mahout.math.{Matrices, Vector}
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm._
import RLikeDrmOps._
import org.apache.mahout.common.RandomUtils

object DSPCA {

  /**
   * Distributed Stochastic PCA decomposition algorithm. A logical reflow of the "SSVD-PCA options.pdf"
   * document of the MAHOUT-817.
   *
   * @param drmA input matrix A
   * @param k request SSVD rank
   * @param p oversampling parameter
   * @param q number of power iterations (hint: use either 0 or 1)
   * @return (U,V,s). Note that U, V are non-checkpointed matrices (i.e. one needs to actually use them
   *         e.g. save them to hdfs in order to trigger their computation.
   */
  def dspca[K](drmA: DrmLike[K], k: Int, p: Int = 15, q: Int = 0):
  (DrmLike[K], DrmLike[Int], Vector) = {

    // Some mapBlock() calls need it
    implicit val ktag =  drmA.keyClassTag

    val drmAcp = drmA.checkpoint()
    implicit val ctx = drmAcp.context

    val m = drmAcp.nrow
    val n = drmAcp.ncol
    assert(k <= (m min n), "k cannot be greater than smaller of m, n.")
    val pfxed = safeToNonNegInt((m min n) - k min p)

    // Actual decomposition rank
    val r = k + pfxed

    // Dataset mean
    val mu = drmAcp.colMeans

    val mtm = mu dot mu

    // We represent Omega by its seed.
    val omegaSeed = RandomUtils.getRandom().nextInt()
    val omega = Matrices.symmetricUniformView(n, r, omegaSeed)

    // This done in front in a single-threaded fashion for now. Even though it doesn't require any
    // memory beyond that is required to keep xi around, it still might be parallelized to backs
    // for significantly big n and r. TODO
    val s_o = omega.t %*% mu

    val bcastS_o = drmBroadcast(s_o)
    val bcastMu = drmBroadcast(mu)

    var drmY = drmAcp.mapBlock(ncol = r) {
      case (keys, blockA) ⇒
        val s_o:Vector = bcastS_o
        val blockY = blockA %*% Matrices.symmetricUniformView(n, r, omegaSeed)
        for (row ← 0 until blockY.nrow) blockY(row, ::) -= s_o
        keys → blockY
    }
        // Checkpoint Y
        .checkpoint()

    var drmQ = dqrThin(drmY, checkRankDeficiency = false)._1.checkpoint()

    var s_q = drmQ.colSums()
    var bcastVarS_q = drmBroadcast(s_q)

    // This actually should be optimized as identically partitioned map-side A'B since A and Q should
    // still be identically partitioned.
    var drmBt = (drmAcp.t %*% drmQ).checkpoint()

    var s_b = (drmBt.t %*% mu).collect(::, 0)
    var bcastVarS_b = drmBroadcast(s_b)

    for (i ← 0 until q) {

      // These closures don't seem to live well with outside-scope vars. This doesn't record closure
      // attributes correctly. So we create additional set of vals for broadcast vars to properly 
      // create readonly closure attributes in this very scope.
      val bcastS_q = bcastVarS_q
      val bcastMuInner = bcastMu

      // Fix Bt as B' -= xi cross s_q
      drmBt = drmBt.mapBlock() {
        case (keys, block) ⇒
          val s_q: Vector = bcastS_q
          val mu: Vector = bcastMuInner
          keys.zipWithIndex.foreach {
            case (key, idx) ⇒ block(idx, ::) -= s_q * mu(key)
          }
          keys → block
      }

      drmY.uncache()
      drmQ.uncache()

      val bCastSt_b = drmBroadcast(s_b -=: mtm * s_q)

      drmY = (drmAcp %*% drmBt)
          // Fix Y by subtracting st_b from each row of the AB'
          .mapBlock() {
        case (keys, block) ⇒
          val st_b: Vector = bCastSt_b
          block := { (_, c, v) ⇒ v - st_b(c) }
          keys → block
      }
          // Checkpoint Y
          .checkpoint()

      drmQ = dqrThin(drmY, checkRankDeficiency = false)._1.checkpoint()

      s_q = drmQ.colSums()
      bcastVarS_q = drmBroadcast(s_q)

      // This on the other hand should be inner-join-and-map A'B optimization since A and Q_i are not
      // identically partitioned anymore.
      drmBt = (drmAcp.t %*% drmQ).checkpoint()

      s_b = (drmBt.t %*% mu).collect(::, 0)
      bcastVarS_b = drmBroadcast(s_b)
    }

    val c = s_q cross s_b
    val inCoreBBt = (drmBt.t %*% drmBt).checkpoint(CacheHint.NONE).collect -=:
        c -=: c.t +=: mtm *=: (s_q cross s_q)
    val (inCoreUHat, d) = eigen(inCoreBBt)
    val s = d.sqrt

    // Since neither drmU nor drmV are actually computed until actually used, we don't need the flags
    // instructing compute (or not compute) either of the U,V outputs anymore. Neat, isn't it?
    val drmU = drmQ %*% inCoreUHat
    val drmV = drmBt %*% (inCoreUHat %*% diagv(1 / s))

    (drmU(::, 0 until k), drmV(::, 0 until k), s(0 until k))
  }

}
