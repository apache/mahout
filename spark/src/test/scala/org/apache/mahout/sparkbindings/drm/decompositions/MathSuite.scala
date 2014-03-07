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

package org.apache.mahout.sparkbindings.drm.decompositions

import org.scalatest.{Matchers, FunSuite}
import org.apache.mahout.sparkbindings.test.MahoutLocalContext
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.sparkbindings.drm._

/**
 *
 * @author dmitriy
 */
class MathSuite extends FunSuite with Matchers with MahoutLocalContext {

  test("thin distributed qr") {

    val inCoreA = dense(
      (1, 2, 3, 4),
      (2, 3, 4, 5),
      (3, -4, 5, 6),
      (4, 5, 6, 7),
      (8, 6, 7, 8)
    )

    val A = drmParallelize(inCoreA, numPartitions = 2)
    val (drmQ, inCoreR) = dqrThin(A, checkRankDeficiency = false)

    // Assert optimizer still knows Q and A are identically partitioned
    drmQ.partitioningTag should equal (A.partitioningTag)

    drmQ.rdd.partitions.size should be(A.rdd.partitions.size)

    // Should also be zippable
    drmQ.rdd.zip(other = A.rdd)

    val inCoreQ = drmQ.collect

    printf("A=\n%s\n", inCoreA)
    printf("Q=\n%s\n", inCoreQ)
    printf("R=\n%s\n", inCoreR)

    val (qControl, rControl) = qr(inCoreA)
    printf("qControl=\n%s\n", qControl)
    printf("rControl=\n%s\n", rControl)

    // Validate with Cholesky
    val ch = chol(inCoreA.t %*% inCoreA)
    printf("A'A=\n%s\n", inCoreA.t %*% inCoreA)
    printf("L:\n%s\n", ch.getL)

    val rControl2 = (ch.getL cloned).t
    val qControl2 = ch.solveRight(inCoreA)
    printf("qControl2=\n%s\n", qControl2)
    printf("rControl2=\n%s\n", rControl2)

    // Housholder approach seems to be a little bit more stable
    (rControl - inCoreR).norm should be < 1E-5
    (qControl - inCoreQ).norm should be < 1E-5

    // Assert identicity with in-core Cholesky-based -- this should be tighter.
    (rControl2 - inCoreR).norm should be < 1E-10
    (qControl2 - inCoreQ).norm should be < 1E-10

  }

  test("dssvd - the naive-est - q=0") {
    ddsvdNaive(q = 0)
  }

  test("ddsvd - naive - q=1") {
    ddsvdNaive(q = 1)
  }

  test("ddsvd - naive - q=2") {
    ddsvdNaive(q = 2)
  }


  def ddsvdNaive(q: Int) {
    val inCoreA = dense(
      (1, 2, 3, 4),
      (2, 3, 4, 5),
      (3, -4, 5, 6),
      (4, 5, 6, 7),
      (8, 6, 7, 8)
    )
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    val (drmU, drmV, s) = dssvd(drmA, k = 4, q = q)
    val (inCoreU, inCoreV) = (drmU.collect, drmV.collect)

    printf("U:\n%s\n", inCoreU)
    printf("V:\n%s\n", inCoreV)
    printf("Sigma:\n%s\n", s)

    (inCoreA - (inCoreU %*%: diagv(s)) %*% inCoreV.t).norm should be < 1E-5
  }


}
