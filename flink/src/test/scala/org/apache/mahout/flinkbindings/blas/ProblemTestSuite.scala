/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements. See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied. See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */
package org.apache.mahout.flinkbindings.blas

import org.apache.flink.api.scala._
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm.CheckpointedFlinkDrm
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical.{OpAx, _}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.junit.runner.RunWith
import org.scalatest.{FunSuite,Matchers}
import org.scalatest.junit.JUnitRunner

import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import RLikeDrmOps._
import decompositions._

import scala.reflect.ClassTag


class ProblemTestSuite extends FunSuite with DistributedFlinkSuite with Matchers {

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
      // flink problem is here... checkpoint is not doing what it should
      // ie. greate a physical plan w/o side effects
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
// Passing now.
//  test("C = inCoreA %*%: B") {
//
//    val inCoreA = dense((1, 2, 3), (3, 4, 5), (4, 5, 6), (5, 6, 7))
//    val inCoreB = dense((3, 5, 7, 10), (4, 6, 9, 10), (5, 6, 7, 7))
//
//    val B = drmParallelize(inCoreB, numPartitions = 2)
//    val C = inCoreA %*%: B
//
//    val inCoreC = C.collect
//    val inCoreCControl = inCoreA %*% inCoreB
//
//    println(inCoreC)
//    (inCoreC - inCoreCControl).norm should be < 1E-10
//
//  }

  test("dsqDist(X,Y)") {
    val m = 100
    val n = 300
    val d = 7
    val mxX = Matrices.symmetricUniformView(m, d, 12345).cloned -= 5
    val mxY = Matrices.symmetricUniformView(n, d, 1234).cloned += 10
    val (drmX, drmY) = (drmParallelize(mxX, 3), drmParallelize(mxY, 4))

    val mxDsq = dsqDist(drmX, drmY).collect
    val mxDsqControl = new DenseMatrix(m, n) := { (r, c, _) ⇒ (mxX(r, ::) - mxY(c, ::)) ^= 2 sum }
    (mxDsq - mxDsqControl).norm should be < 1e-7
  }

  test("dsqDist(X)") {
    val m = 100
    val d = 7
    val mxX = Matrices.symmetricUniformView(m, d, 12345).cloned -= 5
    val drmX = drmParallelize(mxX, 3)

    val mxDsq = dsqDist(drmX).collect
    val mxDsqControl = sqDist(drmX)
    (mxDsq - mxDsqControl).norm should be < 1e-7
  }

  test("DRM DFS i/o (local)") {

    val uploadPath = TmpDir + "UploadedDRM"

    val inCoreA = dense((1, 2, 3), (3, 4, 5))
    val drmA = drmParallelize(inCoreA)

    drmA.dfsWrite(path = uploadPath)

    println(inCoreA)

    // Load back from hdfs
    val drmB = drmDfsRead(path = uploadPath)

    // Make sure keys are correctly identified as ints
    drmB.checkpoint(CacheHint.NONE).keyClassTag shouldBe ClassTag.Int

    // Collect back into in-core
    val inCoreB = drmB.collect

    // Print out to see what it is we collected:
    println(inCoreB)

    (inCoreA - inCoreB).norm should be < 1e-7
  }

  test("DRM parallelizeEmpty") {

    val drmEmpty = drmParallelizeEmpty(100, 50)

    // collect back into in-core
    val inCoreEmpty = drmEmpty.collect

    inCoreEmpty.sum.abs should be < 1e-7
    drmEmpty.nrow shouldBe 100
    drmEmpty.ncol shouldBe 50
    inCoreEmpty.nrow shouldBe 100
    inCoreEmpty.ncol shouldBe 50

  }




}