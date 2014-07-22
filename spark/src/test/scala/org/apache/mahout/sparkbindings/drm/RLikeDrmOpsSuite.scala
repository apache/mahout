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

import org.scalatest.FunSuite
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import drm._
import org.apache.mahout.sparkbindings._
import RLikeDrmOps._
import test.DistributedSparkSuite

/** ==R-like DRM DSL operation tests -- Spark== */
class RLikeDrmOpsSuite extends FunSuite with DistributedSparkSuite with RLikeDrmOpsSuiteBase {

  test("C = A + B missing rows") {
    val sc = mahoutCtx.asInstanceOf[SparkDistributedContext].sc

    // Concoct an rdd with missing rows
    val aRdd: DrmRdd[Int] = sc.parallelize(
      0 -> dvec(1, 2, 3) ::
          3 -> dvec(4, 5, 6) :: Nil
    ).map { case (key, vec) => key -> (vec: Vector)}

    val bRdd: DrmRdd[Int] = sc.parallelize(
      1 -> dvec(2, 3, 4) ::
          2 -> dvec(3, 4, 5) :: Nil
    ).map { case (key, vec) => key -> (vec: Vector)}

    val drmA = drmWrap(rdd=aRdd)
    val drmB = drmWrap(rdd = bRdd, nrow = 4, canHaveMissingRows = true)
    val drmC = drmA + drmB
    val controlC = drmA.collect + drmB.collect

    (drmC -: controlC).norm should be < 1e-10

  }

  test("C = cbind(A, B) with missing rows") {
    val sc = mahoutCtx.asInstanceOf[SparkDistributedContext].sc

    // Concoct an rdd with missing rows
    val aRdd: DrmRdd[Int] = sc.parallelize(
      1 -> dvec(2, 2, 3) ::
          3 -> dvec(4, 5, 6) :: Nil
    ).map { case (key, vec) => key -> (vec: Vector)}

    val bRdd: DrmRdd[Int] = sc.parallelize(
      1 -> dvec(2, 3, 4) ::
          2 -> dvec(3, 4, 5) :: Nil
    ).map { case (key, vec) => key -> (vec: Vector)}

    val drmA = drmWrap(rdd=aRdd)
    val drmB = drmWrap(rdd = bRdd, nrow = 4, canHaveMissingRows = true)
    val drmC = drmA.cbind(drmB)
    val controlC = new DenseMatrix(safeToNonNegInt(drmA.nrow), drmA.ncol + drmB.ncol)
    controlC(::, 0 until drmA.ncol) := drmA
    controlC(::, drmA.ncol until drmA.ncol + drmB.ncol) := drmB

    (drmC -: controlC).norm should be < 1e-10

  }

  test("B = A + 1.0 missing rows") {

    val sc = mahoutCtx.asInstanceOf[SparkDistributedContext].sc

    // Concoct an rdd with missing rows
    val aRdd: DrmRdd[Int] = sc.parallelize(
      0 -> dvec(1, 2, 3) ::
          3 -> dvec(3, 4, 5) :: Nil
    ).map { case (key, vec) => key -> (vec: Vector)}

    val drmA = drmWrap(rdd = aRdd)

    drmA.canHaveMissingRows should equal(true)

    val inCoreA = drmA.collect

    printf("collected A = \n%s\n", inCoreA)

    val controlB = inCoreA + 1.0

    val drmB = drmA + 1.0

    printf ("collected B = \n%s\n", drmB.collect)

    (drmB -: controlB).norm should be < 1e-10

    // Test that unary operators don't obscure the fact that source had missing rows
    val drmC = drmA.mapBlock() { case (keys, block) =>
      keys -> block
    } + 1.0

    (drmC -: controlB).norm should be < 1e-10

  }

}
