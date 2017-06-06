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
package org.apache.mahout.flinkbindings

import org.apache.mahout.flinkbindings._
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.junit.runner.RunWith
import org.scalatest.FunSuite
import org.scalatest.junit.JUnitRunner
import org.slf4j.Logger
import org.slf4j.LoggerFactory


class RLikeOpsSuite extends FunSuite with DistributedFlinkSuite {

  val LOGGER = LoggerFactory.getLogger(getClass())

  test("A %*% x") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val x: Vector = (0, 1, 2)

    val res = A %*% x

    val b = res.collect(::, 0)
    assert(b == dvec(8, 11, 14))
  }

  test("A.t") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val res = A.t.collect

    val expected = inCoreA.t
    assert((res - expected).norm < 1e-6)
  }

  test("A.t %*% x") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val x = dvec(3, 11)
    val res = (A.t %*% x).collect(::, 0)

    val expected = inCoreA.t %*% x 
    assert((res - expected).norm(2) < 1e-6)
  }

  test("A.t %*% B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A.t %*% B

    val expected = inCoreA.t %*% inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A %*% B.t") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A %*% B.t

    val expected = inCoreA %*% inCoreB.t
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A.t %*% A") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A.t %*% A

    val expected = inCoreA.t %*% inCoreA
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A %*% B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4)).t
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A %*% B

    val expected = inCoreA %*% inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A %*% B.t test 2") {
    val mxA = Matrices.symmetricUniformView(10, 7, 80085)
    val mxB = Matrices.symmetricUniformView(30, 7, 31337)
    val A = drmParallelize(mxA, 3)
    val B = drmParallelize(mxB, 4)

    val ABt = (A %*% B.t).collect
    (ABt - mxA %*% mxB.t).norm should be < 1e-7
  }

  test("ABt test") {
    val mxX = dense((1, 2), (2, 3), (3, 4), (5, 6), (7, 8))
    val mxY = dense((1, 2), (2, 3), (3, 4), (5, 6), (7, 8),
                    (1, 2), (2, 3), (3, 4), (5, 6), (7, 8))

    val drmX = drmParallelize(mxX, 3)
    val drmY = drmParallelize(mxY, 4)

    val XYt = (drmX %*% drmY.t).collect
    val control = mxX %*% mxY.t
    (XYt - control).norm should be < 1e-7
  }


  test("A * scalar") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A * 5
    assert((res.collect - inCoreA * 5).norm < 1e-6)
  }

  test("A / scalar") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4)).t
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A / 5
    assert((res.collect - (inCoreA / 5)).norm < 1e-6)
  }

  test("A + scalar") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A + 5
    assert((res.collect - (inCoreA + 5)).norm < 1e-6)
  }

  test("A - scalar") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A - 5
    assert((res.collect - (inCoreA - 5)).norm < 1e-6)
  }

  test("A * B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A * B
    val expected = inCoreA * inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A / B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A / B
    val expected = inCoreA / inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A + B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A + B
    val expected = inCoreA + inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A - B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A - B
    val expected = inCoreA - inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A cbind B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A cbind B
    val expected = dense((1, 2, 1, 2), (2, 3, 3, 4), (3, 4, 11, 4))
    assert((res.collect - expected).norm < 1e-6)
  }

  test("1 cbind A") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = 1 cbind A
    val expected = dense((1, 1, 2), (1, 2, 3), (1, 3, 4))
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A cbind 1") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A cbind 1
    val expected = dense((1, 2, 1), (2, 3, 1), (3, 4, 1))
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A rbind B") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4))
    val inCoreB = dense((1, 2), (3, 4), (11, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = drmParallelize(m = inCoreB, numPartitions = 2)

    val res = A rbind B
    val expected = dense((1, 2), (2, 3), (3, 4), (1, 2), (3, 4), (11, 4))
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A row slice") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4), (4, 4), (5, 5), (6, 7))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A(2 until 5, ::)
    val expected = inCoreA(2 until 5, ::)
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A column slice") {
    val inCoreA = dense((1, 2, 1, 2), (2, 3, 3, 4), (3, 4, 11, 4))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A(::, 0 until 2)
    val expected = inCoreA(::, 0 until 2)
    assert((res.collect - expected).norm < 1e-6)
  }

  test("A %*% inCoreB") {
    val inCoreA = dense((1, 2), (2, 3), (3, 4)).t
    val inCoreB = dense((1, 2), (3, 4), (11, 4))

    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val res = A %*% inCoreB

    val expected = inCoreA %*% inCoreB
    assert((res.collect - expected).norm < 1e-6)
  }

  test("drmBroadcast") {
    val inCoreA = dense((1, 2), (3, 4), (11, 4))
    val x = dvec(1, 2)
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val b = drmBroadcast(x)

    val res = A.mapBlock(1) { case (idx, block) =>
      (idx, (block %*% b).toColMatrix)
    }

    val expected = inCoreA %*% x
    assert((res.collect(::, 0) - expected).norm(2) < 1e-6)
  }

  test("A.t %*% B with Long keys") {
    val inCoreA = dense((1, 2), (3, 4), (3, 5))
    val inCoreB = dense((3, 5), (4, 6), (0, 1))

    val A = drmParallelize(inCoreA, numPartitions = 2).mapBlock()({
      case (keys, block) => (keys.map(_.toLong), block)
    })

    val B = drmParallelize(inCoreB, numPartitions = 2).mapBlock()({
      case (keys, block) => (keys.map(_.toLong), block)
    })

    val C = A.t %*% B
    val inCoreC = C.collect
    val expected = inCoreA.t %*% inCoreB

    (inCoreC - expected).norm should be < 1E-10
  }


}