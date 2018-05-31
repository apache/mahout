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

import scala.reflect.{ClassTag,classTag}

/** Common tests for DrmLike operators to be executed by all distributed engines. */
trait DrmLikeOpsSuiteBase extends DistributedMahoutSuite with Matchers {
  this: FunSuite ⇒

  test("mapBlock") {

    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = A.mapBlock(/* Inherit width */) {
      case (keys, block) ⇒ keys → (block += 1.0)
    }

    val inCoreB = B.collect
    val inCoreBControl = inCoreA + 1.0

    println(inCoreB)

    // Assert they are the same
    (inCoreB - inCoreBControl).norm should be < 1E-10
    B.keyClassTag shouldBe ClassTag.Int

  }

  test ("mapBlock implicit keying") {

    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = A.mapBlock(/* Inherit width */) {
      case (keys, block) ⇒ keys.map { k ⇒ k.toString } → block
    }

    B.keyClassTag shouldBe classTag[String]

  }


  test("allReduceBlock") {

    val mxA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6))
    val drmA = drmParallelize(mxA, numPartitions = 2)

    try {
      val mxB = drmA.allreduceBlock { case (keys, block) ⇒
        block(::, 0 until 2).t %*% block(::, 2 until 3)
      }

      val mxControl = mxA(::, 0 until 2).t %*% mxA(::, 2 until 3)

      (mxB - mxControl).norm should be < 1e-10

    } catch {
      case e: UnsupportedOperationException ⇒ // Some engines may not support this, so ignore.
    }

  }

  test("col range") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = A(::, 1 to 2)
    val inCoreB = B.collect
    val inCoreBControl = inCoreA(::, 1 to 2)

    println(inCoreB)

    // Assert they are the same
    (inCoreB - inCoreBControl).norm should be < 1E-10

  }

  test("row range") {

    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = A(1 to 2, ::)
    val inCoreB = B.collect
    val inCoreBControl = inCoreA(1 to 2, ::)

    println(inCoreB)

    // Assert they are the same
    (inCoreB - inCoreBControl).norm should be < 1E-10

  }

  test("col, row range") {

    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = A(1 to 2, 1 to 2)
    val inCoreB = B.collect
    val inCoreBControl = inCoreA(1 to 2, 1 to 2)

    println(inCoreB)

    // Assert they are the same
    (inCoreB - inCoreBControl).norm should be < 1E-10

  }

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

}
