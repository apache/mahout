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
import org.apache.mahout.sparkbindings.drm.plan.{OpAtB, OpAtA, CheckpointAction}

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


  test("C = A.t %*% A non-int key") {
    val inCoreA = dense((1, 2, 3), (3, 4, 5), (4, 5, 6), (5, 6, 7))
    val AintKeyd = drmParallelize(m = inCoreA, numPartitions = 2)
    val A = AintKeyd.mapBlock()({
      case (keys, block) => keys.map(_.toString) -> block
    })

    val AtA = A.t %*% A

    // Assert optimizer detects square
    CheckpointAction.optimize(action = AtA) should equal(OpAtA(A))

    val inCoreAtA = AtA.collect
    val inCoreAtAControl = inCoreA.t %*% inCoreA

    (inCoreAtA - inCoreAtAControl).norm should be < 1E-10
  }

}
