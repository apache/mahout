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

import scala.math._
import org.apache.mahout.math.{Matrices, Matrix}
import org.apache.mahout.common.RandomUtils
import org.apache.log4j.Logger
import org.apache.mahout.math.scalabindings._
import RLikeOps._

private[math] object SSVD {

  private val log = Logger.getLogger(SSVD.getClass)

  /**
   * In-core SSVD algorithm.
   *
   * @param a input matrix A
   * @param k request SSVD rank
   * @param p oversampling parameter
   * @param q number of power iterations
   * @return (U,V,s)
   */
  def ssvd(a: Matrix, k: Int, p: Int = 15, q: Int = 0) = {
    val m = a.nrow
    val n = a.ncol
    if (k > min(m, n))
      throw new IllegalArgumentException(
        "k cannot be greater than smaller of m,n")
    val pfxed = min(p, min(m, n) - k)

    // Actual decomposition rank
    val r = k + pfxed

    val rnd = RandomUtils.getRandom
    val omega = Matrices.symmetricUniformView(n, r, rnd.nextInt)

    var y = a %*% omega
    var yty = y.t %*% y
    val at = a.t
    var ch = chol(yty)
    assert(ch.isPositiveDefinite, "Rank-deficiency detected during s-SVD")
    var bt = ch.solveRight(at %*% y)

    // Power iterations
    for (i ← 0 until q) {
      y = a %*% bt
      yty = y.t %*% y
      ch = chol(yty)
      bt = ch.solveRight(at %*% y)
    }

    val bbt = bt.t %*% bt
    val (uhat, d) = eigen(bbt)

    val s = d.sqrt
    val u = ch.solveRight(y) %*% uhat
    val v = bt %*% (uhat %*% diagv(1 /: s))

    (u(::, 0 until k), v(::, 0 until k), s(0 until k))
  }

  /**
   * PCA based on SSVD that runs without forming an always-dense A-(colMeans(A)) input for SVD. This
   * follows the solution outlined in MAHOUT-817. For in-core version it, for most part, is supposed
   * to save some memory for sparse inputs by removing direct mean subtraction.<P>
   *
   * Hint: Usually one wants to use AV which is approsimately USigma, i.e.<code>u %*%: diagv(s)</code>.
   * If retaining distances and orignal scaled variances not that important, the normalized PCA space
   * is just U.
   *
   * Important: data points are considered to be rows.
   *
   * @param a input matrix A
   * @param k request SSVD rank
   * @param p oversampling parameter
   * @param q number of power iterations
   * @return (U,V,s)
   */
  def spca(a:Matrix, k: Int, p: Int = 15, q: Int = 0) = {
    val m = a.nrow
    val n = a.ncol
    if (k > min(m, n))
      throw new IllegalArgumentException(
        "k cannot be greater than smaller of m,n")
    val pfxed = min(p, min(m, n) - k)

    // Actual decomposition rank
    val r = k + pfxed

    val rnd = RandomUtils.getRandom
    val omega = Matrices.symmetricUniformView(n, r, rnd.nextInt)

    // Dataset mean
    val mu = a.colMeans()
    val mtm = mu dot mu

    if (log.isDebugEnabled) log.debug("xi=%s".format(mu))

    var y = a %*% omega

    // Fixing y
    val s_o = omega.t %*% mu
    y := ((r,c,v) ⇒ v - s_o(c))

    var yty = y.t %*% y
    var ch = chol(yty)
//    assert(ch.isPositiveDefinite, "Rank-deficiency detected during s-SVD")

    // This is implicit Q of QR(Y)
    var qm = ch.solveRight(y)
    var bt = a.t %*% qm
    var s_q = qm.colSums()
    var s_b = bt.t %*% mu

    // Power iterations
    for (i ← 0 until q) {

      // Fix bt
      bt -= mu cross s_q

      y = a %*% bt

      // Fix Y again.
      val st_b = s_b -=: mtm * s_q
      y := ((r,c,v) ⇒ v - st_b(c))

      yty = y.t %*% y
      ch = chol(yty)
      qm = ch.solveRight(y)
      bt = a.t %*% qm
      s_q = qm.colSums()
      s_b = bt.t %*% mu
    }

    val c = s_q cross s_b

    // BB' computation becomes
    val bbt = bt.t %*% bt -= c -= c.t += (mtm * s_q cross s_q)

    val (uhat, d) = eigen(bbt)

    val s = d.sqrt
    val u = qm %*% uhat
    val v = bt %*% (uhat %*%: diagv(1 /: s))

    (u(::, 0 until k), v(::, 0 until k), s(0 until k))

  }

}
