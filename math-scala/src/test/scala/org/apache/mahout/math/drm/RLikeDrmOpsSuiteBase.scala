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

import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import RLikeDrmOps._
import decompositions._
import org.apache.mahout.math.drm.logical._
import org.apache.mahout.math.drm.logical.OpAtx
import org.apache.mahout.math.drm.logical.OpAtB
import org.apache.mahout.math.drm.logical.OpAtA
import org.apache.mahout.math.drm.logical.OpAewUnaryFuncFusion

import scala.util.Random

/** Common engine tests for distributed R-like DRM operations */
trait RLikeDrmOpsSuiteBase extends DistributedMahoutSuite with Matchers {
  this: FunSuite =>

  val epsilon = 1E-5

  test("A.t") {

    val inCoreA = dense((1, 2, 3), (3, 4, 5))

    val A = drmParallelize(inCoreA)

    val inCoreAt = A.t.collect

    // Assert first norm of difference is less than error margin.
    (inCoreAt - inCoreA.t).norm should be < epsilon

  }

  test("C = A %*% B") {

    val inCoreA = dense((1, 2), (3, 4))
    val inCoreB = dense((3, 5), (4, 6))

    val A = drmParallelize(inCoreA, numPartitions = 2)
    val B = drmParallelize(inCoreB, numPartitions = 2)

    // Actual
    val inCoreCControl = inCoreA %*% inCoreB

    // Distributed operation
    val C = A %*% B
    val inCoreC = C.collect
    println(inCoreC)

    (inCoreC - inCoreCControl).norm should be < 1E-10

    // We also should be able to collect via implicit checkpoint
    val inCoreC2 = C.collect
    println(inCoreC2)

    (inCoreC2 - inCoreCControl).norm should be < 1E-10

  }

  test("C = A %*% B mapBlock {}") {

    val inCoreA = dense((1, 2), (3, 4))
    val inCoreB = dense((3, 5), (4, 6))

    val A = drmParallelize(inCoreA, numPartitions = 2).checkpoint()
    val B = drmParallelize(inCoreB, numPartitions = 2).checkpoint()

    // Actual
    val inCoreCControl = inCoreA %*% inCoreB

    A.colSums()
    B.colSums()


    val x = drmBroadcast(dvec(0, 0))
    val x2 = drmBroadcast(dvec(0, 0))
    // Distributed operation
    val C = (B.t %*% A.t).t.mapBlock() {
      case (keys, block) =>
        for (row <- 0 until block.nrow) block(row, ::) += x.value + x2
        keys -> block
    }

    val inCoreC = C checkpoint CacheHint.NONE collect;
    println(inCoreC)

    (inCoreC - inCoreCControl).norm should be < 1E-10

    // We also should be able to collect via implicit checkpoint
    val inCoreC2 = C.collect
    println(inCoreC2)

    (inCoreC2 - inCoreCControl).norm should be < 1E-10

    val inCoreQ = dqrThin(C)._1.collect

    printf("Q=\n%s\n", inCoreQ)

    // Assert unit-orthogonality
    ((inCoreQ(::, 0) dot inCoreQ(::, 0)) - 1.0).abs should be < 1e-10
    (inCoreQ(::, 0) dot inCoreQ(::, 1)).abs should be < 1e-10

  }

  test("C = A %*% B incompatible B keys") {

    val inCoreA = dense((1, 2), (3, 4))
    val inCoreB = dense((3, 5), (4, 6))

    val A = drmParallelize(inCoreA, numPartitions = 2)
    val B = drmParallelize(inCoreB, numPartitions = 2)
        // Re-key B into DrmLike[String] instead of [Int]
        .mapBlock()({
      case (keys, block) => keys.map(_.toString) -> block
    })

    val C = A %*% B

    intercept[IllegalArgumentException] {
      // This plan must not compile
      C.checkpoint()
    }
  }

  test("Spark-specific C = At %*% B , join") {

    val inCoreA = dense((1, 2), (3, 4), (-3, -5))
    val inCoreB = dense((3, 5), (4, 6), (0, 1))

    val A = drmParallelize(inCoreA, numPartitions = 2)
    val B = drmParallelize(inCoreB, numPartitions = 2)

    val C = A.t %*% B

    mahoutCtx.optimizerRewrite(C) should equal(OpAtB[Int](A, B))

    val inCoreC = C.collect
    val inCoreControlC = inCoreA.t %*% inCoreB

    (inCoreC - inCoreControlC).norm should be < 1E-10

  }


  test("C = At %*% B , join, String-keyed") {

    val inCoreA = dense((1, 2), (3, 4), (-3, -5))
    val inCoreB = dense((3, 5), (4, 6), (0, 1))

    val A = drmParallelize(inCoreA, numPartitions = 2)
        .mapBlock()({
      case (keys, block) => keys.map(_.toString) -> block
    })

    val B = drmParallelize(inCoreB, numPartitions = 2)
        .mapBlock()({
      case (keys, block) => keys.map(_.toString) -> block
    })

    val C = A.t %*% B

    mahoutCtx.optimizerRewrite(C) should equal(OpAtB[String](A, B))

    val inCoreC = C.collect
    val inCoreControlC = inCoreA.t %*% inCoreB

    (inCoreC - inCoreControlC).norm should be < 1E-10

  }

  test("C = At %*% B , zippable, String-keyed") {

    val inCoreA = dense((1, 2), (3, 4), (-3, -5))

    val A = drmParallelize(inCoreA, numPartitions = 2)
        .mapBlock()({
      case (keys, block) ⇒ keys.map(_.toString) → block
    })

    // Dense-A' x sparse-B used to produce error. We sparsify B here to test this as well.
    val B = (A + 1.0).mapBlock() { case (keys, block) ⇒
      keys → (new SparseRowMatrix(block.nrow, block.ncol) := block)
    }

    val C = A.t %*% B

    mahoutCtx.optimizerRewrite(C) should equal(OpAtB[String](A, B))

    val inCoreC = C.collect
    val inCoreControlC = inCoreA.t %*% (inCoreA + 1.0)

    (inCoreC - inCoreControlC).norm should be < 1E-10

  }

  test ("C = A %*% B.t") {

    val inCoreA = dense((1, 2), (3, 4), (-3, -5))

    val A = drmParallelize(inCoreA, numPartitions = 2)

    val B = A + 1.0

    val C = A %*% B.t

    mahoutCtx.optimizerRewrite(C) should equal(OpABt[Int](A, B))

    val inCoreC = C.collect
    val inCoreControlC = inCoreA %*% (inCoreA + 1.0).t

    (inCoreC - inCoreControlC).norm should be < 1E-10

  }

  test("C = A %*% inCoreB") {

    val inCoreA = dense((1, 2, 3), (3, 4, 5), (4, 5, 6), (5, 6, 7))
    val inCoreB = dense((3, 5, 7, 10), (4, 6, 9, 10), (5, 6, 7, 7))

    val A = drmParallelize(inCoreA, numPartitions = 2)
    val C = A %*% inCoreB

    val inCoreC = C.collect
    val inCoreCControl = inCoreA %*% inCoreB

    println(inCoreC)
    (inCoreC - inCoreCControl).norm should be < 1E-10

  }

  test("C = inCoreA %*%: B") {

    val inCoreA = dense((1, 2, 3), (3, 4, 5), (4, 5, 6), (5, 6, 7))
    val inCoreB = dense((3, 5, 7, 10), (4, 6, 9, 10), (5, 6, 7, 7))

    val B = drmParallelize(inCoreB, numPartitions = 2)
    val C = inCoreA %*%: B

    val inCoreC = C.collect
    val inCoreCControl = inCoreA %*% inCoreB

    println(inCoreC)
    (inCoreC - inCoreCControl).norm should be < 1E-10

  }

  test("C = A.t %*% A") {
    val inCoreA = dense((1, 2, 3), (3, 4, 5), (4, 5, 6), (5, 6, 7))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val AtA = A.t %*% A

    // Assert optimizer detects square
    mahoutCtx.optimizerRewrite(action = AtA) should equal(OpAtA(A))

    val inCoreAtA = AtA.collect
    val inCoreAtAControl = inCoreA.t %*% inCoreA

    (inCoreAtA - inCoreAtAControl).norm should be < 1E-10
  }

  test("C = A.t %*% A fat non-graph") {
    // Hack the max in-mem size for this test
    System.setProperty("mahout.math.AtA.maxInMemNCol", "540")

    val inCoreA = Matrices.uniformView(400, 550, 1234)
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val AtA = A.t %*% A

    // Assert optimizer detects square
    mahoutCtx.optimizerRewrite(action = AtA) should equal(OpAtA(A))

    val inCoreAtA = AtA.collect
    val inCoreAtAControl = inCoreA.t %*% inCoreA

    (inCoreAtA - inCoreAtAControl).norm should be < 1E-10
  }

  test("C = A.t %*% A non-int key") {
    val inCoreA = dense((1, 2, 3), (3, 4, 5), (4, 5, 6), (5, 6, 7))
    val AintKeyd = drmParallelize(m = inCoreA, numPartitions = 2)
    val A = AintKeyd.mapBlock() {
      case (keys, block) => keys.map(_.toString) -> block
    }

    val AtA = A.t %*% A

    // Assert optimizer detects square
    mahoutCtx.optimizerRewrite(action = AtA) should equal(OpAtA(A))

    val inCoreAtA = AtA.collect
    val inCoreAtAControl = inCoreA.t %*% inCoreA

    (inCoreAtA - inCoreAtAControl).norm should be < 1E-10
  }

  test("C = A + B") {

    val inCoreA = dense((1, 2), (3, 4))
    val inCoreB = dense((3, 5), (4, 6))

    val A = drmParallelize(inCoreA, numPartitions = 2)
    val B = drmParallelize(inCoreB, numPartitions = 2)

    val C = A + B
    val inCoreC = C.collect

    // Actual
    val inCoreCControl = inCoreA + inCoreB

    (inCoreC - inCoreCControl).norm should be < 1E-10
  }

  test("C = A + B, identically partitioned") {

    val inCoreA = dense((1, 2, 3), (3, 4, 5), (5, 6, 7))

    val A = drmParallelize(inCoreA, numPartitions = 2)

//    printf("A.nrow=%d.\n", A.rdd.count())

    // Create B which would be identically partitioned to A. mapBlock() by default will do the trick.
    val B = A.mapBlock() {
      case (keys, block) =>
        val bBlock = block.like() := { (r, c, v) => util.Random.nextDouble()}
        keys -> bBlock
    }
        // Prevent repeated computation non-determinism
        // removing this checkpoint() will cause the same error in spark Tests
        // as we're seeing in Flink with this test.  ie  util.Random.nextDouble()
        // is being called more than once (note that it is not seeded in the closure)
        .checkpoint()

    val inCoreB = B.collect

    printf("A=\n%s\n", inCoreA)
    printf("B=\n%s\n", inCoreB)

    val C = A + B

    val inCoreC = C.collect

    printf("C=\n%s\n", inCoreC)

    // Actual
    val inCoreCControl = inCoreA + inCoreB

    (inCoreC - inCoreCControl).norm should be < 1E-10
  }


  test("C = A + B side test 1") {

    val inCoreA = dense((1, 2), (3, 4))
    val inCoreB = dense((3, 5), (4, 6))

    val A = drmParallelize(inCoreA, numPartitions = 2)
    val B = drmParallelize(inCoreB, numPartitions = 2)

    val C = A + B
    val inCoreC = C.collect

    val inCoreD = (A + B).collect

    // Actual
    val inCoreCControl = inCoreA + inCoreB

    (inCoreC - inCoreCControl).norm should be < 1E-10
    (inCoreD - inCoreCControl).norm should be < 1E-10
  }

  test("C = A + B side test 2") {

    val inCoreA = dense((1, 2), (3, 4))
    val inCoreB = dense((3, 5), (4, 6))

    val A = drmParallelize(inCoreA, numPartitions = 2).checkpoint()
    val B = drmParallelize(inCoreB, numPartitions = 2)

    val C = A + B
    val inCoreC = C.collect

    val inCoreD = (A + B).collect

    // Actual
    val inCoreCControl = inCoreA + inCoreB

    (inCoreC - inCoreCControl).norm should be < 1E-10
    (inCoreD - inCoreCControl).norm should be < 1E-10
  }

  test("C = A + B side test 3") {

    val inCoreA = dense((1, 2), (3, 4))
    val inCoreB = dense((3, 5), (4, 6))

    val B = drmParallelize(inCoreB, numPartitions = 2)
    //    val A = (drmParallelize(inCoreA, numPartitions = 2) + B).checkpoint(CacheHint.MEMORY_ONLY_SER)
    val A = (drmParallelize(inCoreA, numPartitions = 2) + B).checkpoint(CacheHint.MEMORY_ONLY)

    val C = A + B
    val inCoreC = C.collect

    val inCoreD = (A + B).collect

    // Actual
    val inCoreCControl = inCoreA + inCoreB * 2.0

    (inCoreC - inCoreCControl).norm should be < 1E-10
    (inCoreD - inCoreCControl).norm should be < 1E-10
  }

  test("Ax") {
    val inCoreA = dense(
      (1, 2),
      (3, 4),
      (20, 30)
    )
    val x = dvec(10, 3)

    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    val ax = (drmA %*% x).collect(::, 0)

    ax should equal(inCoreA %*% x)
  }

  test("A'x") {
    val inCoreA = dense(
      (1, 2),
      (3, 4),
      (20, 30)
    )
    val x = dvec(10, 3, 4)

    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    mahoutCtx.optimizerRewrite(drmA.t %*% x) should equal(OpAtx(drmA, x))

    val atx = (drmA.t %*% x).collect(::, 0)

    atx should equal(inCoreA.t %*% x)
  }

  test("colSums, colMeans") {
    val inCoreA = dense(
      (1, 2),
      (3, 4),
      (20, 30)
    )
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    drmA.colSums() should equal(inCoreA.colSums())
    drmA.colMeans() should equal(inCoreA.colMeans())
  }

  test("rowSums, rowMeans") {
    val inCoreA = dense(
      (1, 2),
      (3, 4),
      (20, 30)
    )
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    drmA.rowSums() should equal(inCoreA.rowSums())
    drmA.rowMeans() should equal(inCoreA.rowMeans())
  }

  test("A.diagv") {
    val inCoreA = dense(
      (1, 2, 3),
      (3, 4, 5),
      (20, 30, 7)
    )
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    drmA.diagv should equal(inCoreA.diagv)
  }

  test("numNonZeroElementsPerColumn") {
    val inCoreA = dense(
      (0, 2),
      (3, 0),
      (0, -30)

    )
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    drmA.numNonZeroElementsPerColumn() should equal(inCoreA.numNonZeroElementsPerColumn())
  }

  test("C = A cbind B, cogroup") {

    val inCoreA = dense((1, 2), (3, 4))
    val inCoreB = dense((3, 5), (4, 6))
    val controlC = dense((1, 2, 3, 5), (3, 4, 4, 6))

    val A = drmParallelize(inCoreA, numPartitions = 2).checkpoint()
    val B = drmParallelize(inCoreB, numPartitions = 2).checkpoint()

    (A.cbind(B) -: controlC).norm should be < 1e-10

  }

  test("C = A cbind B, zip") {

    val inCoreA = dense((1, 2), (3, 4))
    val controlC = dense((1, 2, 2, 3), (3, 4, 4, 5))

    val A = drmParallelize(inCoreA, numPartitions = 2).checkpoint()

    (A.cbind(A + 1.0) -: controlC).norm should be < 1e-10

  }

  test("B = 1 cbind A") {
    val inCoreA = dense((1, 2), (3, 4))
    val control = dense((1, 1, 2), (1, 3, 4))

    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    (control - (1 cbind drmA) ).norm should be < 1e-10
  }

  test("B = A cbind 1") {
    val inCoreA = dense((1, 2), (3, 4))
    val control = dense((1, 2, 1), (3, 4, 1))

    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    (control - (drmA cbind 1) ).norm should be < 1e-10
  }

  test("B = A + 1.0") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val controlB = inCoreA + 1.0

    val drmB = drmParallelize(m = inCoreA, numPartitions = 2) + 1.0

    (drmB -: controlB).norm should be < 1e-10
  }
  
  test("C = A rbind B") {

    val inCoreA = dense((1, 2), (3, 5))
    val inCoreB = dense((7, 11), (13, 17))
    val controlC = dense((1, 2), (3, 5), (7, 11), (13, 17))

    val A = drmParallelize(inCoreA, numPartitions = 2).checkpoint()
    val B = drmParallelize(inCoreB, numPartitions = 2).checkpoint()
    
    (A.rbind(B) -: controlC).norm should be < 1e-10
  }

  test("C = A rbind B, with empty") {

    val inCoreA = dense((1, 2), (3, 5))
    val emptyB = drmParallelizeEmpty(nrow = 2, ncol = 2, numPartitions = 2)
    val controlC = dense((1, 2), (3, 5), (0, 0), (0, 0))

    val A = drmParallelize(inCoreA, numPartitions = 2).checkpoint()

    (A.rbind(emptyB) -: controlC).norm should be < 1e-10
  }

  /** Test dsl overloads over scala operations over matrices */
  test("scalarOps") {
    val drmA = drmParallelize(m = dense(
      (1, 2, 3),
      (3, 4, 5),
      (7, 8, 9)
    ),
      numPartitions = 2)

    (10 * drmA - (10 *: drmA)).norm shouldBe 0

  }

  test("A * A -> sqr(A) rewrite ") {
    val mxA = dense(
      (1, 2, 3),
      (3, 4, 5),
      (7, 8, 9)
    )

    val mxAAControl = mxA * mxA

    val drmA = drmParallelize(mxA, 2)
    val drmAA = drmA * drmA

    val optimized = drmAA.context.engine.optimizerRewrite(drmAA)
    println(s"optimized:$optimized")
    optimized.isInstanceOf[OpAewUnaryFunc[Int]] shouldBe true

    (mxAAControl -= drmAA).norm should be < 1e-10
  }

  test("B = 1 + 2 * (A * A) ew unary function fusion") {
    val mxA = dense(
      (1, 2, 3),
      (3, 0, 5)
    )
    val controlB = mxA.cloned := { (x) => 1 + 2 * x * x}

    val drmA = drmParallelize(mxA, 2)

    // We need to use parenthesis, otherwise optimizer will see it as (2A) * (A) and that would not
    // be rewritten as 2 * sqr(A). It is not that clever (yet) to try commutativity optimizations.
    val drmB = 1 + 2 * (drmA * drmA)

    val optimized = mahoutCtx.engine.optimizerRewrite(drmB)
    println(s"optimizer rewritten:$optimized")
    optimized.isInstanceOf[OpAewUnaryFuncFusion[Int]] shouldBe true

    (controlB - drmB).norm should be < 1e-10

  }

  test("functional apply()") {
    val mxA = sparse (
      (1 -> 3) :: (7 -> 7) :: Nil,
      (4 -> 5) :: (5 -> 8) :: Nil
    )

    val mxAControl = mxA cloned
    val drmA = drmParallelize(mxA)

    (drmA(x => x + 1).collect - (mxAControl + 1)).norm should be < 1e-7
    (drmA(x => x * 2).collect - (2 * mxAControl)).norm should be < 1e-7

  }


}
