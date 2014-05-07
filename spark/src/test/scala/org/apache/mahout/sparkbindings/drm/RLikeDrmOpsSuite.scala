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

import org.scalatest.{Matchers, FunSuite}
import org.apache.mahout.sparkbindings.test.MahoutLocalContext
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.sparkbindings.drm._
import RLikeDrmOps._
import org.apache.mahout.sparkbindings.drm.plan.{OpAtx, OpAtB, OpAtA, CheckpointAction}
import org.apache.spark.SparkContext
import scala.collection.mutable.ArrayBuffer
import org.apache.mahout.math.Matrices
import org.apache.mahout.sparkbindings.blas
import org.apache.spark.storage.StorageLevel

/** R-like DRM DSL operation tests */
class RLikeDrmOpsSuite extends FunSuite with Matchers with MahoutLocalContext {

  import RLikeOps._

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


    val x = drmBroadcast(dvec(0,0))
    val x2 = drmBroadcast(dvec(0,0))
    // Distributed operation
    val C = (B.t %*% A.t).t.mapBlock() {
      case (keys, block) =>
        for (row <- 0 until block.nrow) block(row,::) += x.value + x2
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
      case (keys,block) => keys.map(_.toString) -> block
    })

    val C = A %*% B

    intercept[IllegalArgumentException] {
      // This plan must not compile
      C.checkpoint()
    }
  }

  test("C = At %*% B , join") {

    val inCoreA = dense((1, 2), (3, 4),(-3, -5))
    val inCoreB = dense((3, 5), (4, 6), (0, 1))

    val A = drmParallelize(inCoreA, numPartitions = 2)
    val B = drmParallelize(inCoreB, numPartitions = 2)

    val C = A.t %*% B

    CheckpointAction.optimize(C) should equal (OpAtB[Int](A,B))

    val inCoreC = C.collect
    val inCoreControlC = inCoreA.t %*% inCoreB

    (inCoreC - inCoreControlC).norm should be < 1E-10

  }

  test("C = At %*% B , join, String-keyed") {

    val inCoreA = dense((1, 2), (3, 4),(-3, -5))
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

    CheckpointAction.optimize(C) should equal (OpAtB[String](A,B))

    val inCoreC = C.collect
    val inCoreControlC = inCoreA.t %*% inCoreB

    (inCoreC - inCoreControlC).norm should be < 1E-10

  }

  test("C = At %*% B , zippable, String-keyed") {

    val inCoreA = dense((1, 2), (3, 4),(-3, -5))

    val A = drmParallelize(inCoreA, numPartitions = 2)
        .mapBlock()({
      case (keys, block) => keys.map(_.toString) -> block
    })

    val B =  A + 1.0

    val C = A.t %*% B

    CheckpointAction.optimize(C) should equal (OpAtB[String](A,B))

    val inCoreC = C.collect
    val inCoreControlC = inCoreA.t %*% (inCoreA + 1.0)

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
    CheckpointAction.optimize(action = AtA) should equal(OpAtA(A))

    val inCoreAtA = AtA.collect
    val inCoreAtAControl = inCoreA.t %*% inCoreA

    (inCoreAtA - inCoreAtAControl).norm should be < 1E-10
  }

  test("C = A.t %*% A fat non-graph") {
    // Hack the max in-mem size for this test
    System.setProperty(blas.AtA.PROPERTY_ATA_MAXINMEMNCOL, "540")

    val inCoreA = Matrices.uniformView(400, 550, 1234)
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val AtA = A.t %*% A

    // Assert optimizer detects square
    CheckpointAction.optimize(action = AtA) should equal(OpAtA(A))

    val inCoreAtA = AtA.collect
    val inCoreAtAControl = inCoreA.t %*% inCoreA

    (inCoreAtA - inCoreAtAControl).norm should be < 1E-10
    log.debug("test done.")
  }


  test("C = A.t %*% A non-int key") {
    val inCoreA = dense((1, 2, 3), (3, 4, 5), (4, 5, 6), (5, 6, 7))
    val AintKeyd = drmParallelize(m = inCoreA, numPartitions = 2)
    val A = AintKeyd.mapBlock() {
      case (keys, block) => keys.map(_.toString) -> block
    }

    val AtA = A.t %*% A

    // Assert optimizer detects square
    CheckpointAction.optimize(action = AtA) should equal(OpAtA(A))

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
    val inCoreCControl = inCoreA +  inCoreB * 2.0

    (inCoreC - inCoreCControl).norm should be < 1E-10
    (inCoreD - inCoreCControl).norm should be < 1E-10
  }

  test ("general side")  {
    val sc = implicitly[SparkContext]
    val k1 = sc.parallelize(Seq(ArrayBuffer(0,1,2,3)))
//      .persist(StorageLevel.MEMORY_ONLY)   // -- this will demonstrate immutability side effect!
      .persist(StorageLevel.MEMORY_ONLY_SER)

    println(k1.map(_ += 4).collect.head)
    println(k1.map(_ += 4).collect.head)
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

    CheckpointAction.optimize(drmA.t %*% x) should equal (OpAtx(drmA, x))

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

    drmA.colSums() should equal (inCoreA.colSums())
    drmA.colMeans() should equal (inCoreA.colMeans())
  }

}
