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
import org.apache.mahout.math.drm.logical.{OpAtB, OpAewUnaryFuncFusion}
import org.apache.mahout.logging._

import scala.util.Random

/** ==R-like DRM DSL operation tests -- Spark== */
class RLikeDrmOpsSuite extends FunSuite with DistributedSparkSuite with RLikeDrmOpsSuiteBase {

  private final implicit val log = getLog(classOf[RLikeDrmOpsSuite])

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

  test("A'B, bigger") {

    val rnd = new Random()
    val a = new SparseRowMatrix(200, 1544) := { _ => rnd.nextGaussian() }
    val b = new SparseRowMatrix(200, 300) := { _ => rnd.nextGaussian() }

    var ms = System.currentTimeMillis()
    val atb = a.t %*% b
    ms = System.currentTimeMillis() - ms

    println(s"in-core mul ms: $ms")

    val drmA = drmParallelize(a, numPartitions = 2)
    val drmB = drmParallelize(b, numPartitions = 2)

    ms = System.currentTimeMillis()
    val drmAtB = drmA.t %*% drmB
    val mxAtB = drmAtB.collect
    ms = System.currentTimeMillis() - ms

    println(s"a'b plan:${drmAtB.context.engine.optimizerRewrite(drmAtB)}")
    println(s"a'b plan contains ${drmAtB.rdd.partitions.size} partitions.")
    println(s"distributed mul ms: $ms.")

    (atb - mxAtB).norm should be < 1e-5

  }

  test("C = At %*% B , zippable") {

    val mxA = dense((1, 2), (3, 4), (-3, -5))

    val A = drmParallelize(mxA, numPartitions = 2)
      .mapBlock()({
      case (keys, block) => keys.map(_.toString) -> block
    })

    val B = (A + 1.0)

      .mapBlock() { case (keys, block) ⇒
      val nblock = new SparseRowMatrix(block.nrow, block.ncol) := block
      keys → nblock
    }

    B.collect

    val C = A.t %*% B

    mahoutCtx.optimizerRewrite(C) should equal(OpAtB[String](A, B))

    val inCoreC = C.collect
    val inCoreControlC = mxA.t %*% (mxA + 1.0)

    (inCoreC - inCoreControlC).norm should be < 1E-10

  }

}
