package org.apache.mahout.sparkbindings.drm

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


import org.apache.mahout.math.scalabindings._
import RLikeOps._
import RLikeDrmOps._
import org.scalatest.{Matchers, FunSuite}
import org.apache.mahout.sparkbindings.test.MahoutLocalContext

/** Tests for DrmLikeOps */
class DrmLikeOpsSuite extends FunSuite with MahoutLocalContext {

  test("mapBlock") {

    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    val B = A.mapBlock(/* Inherit width */) {
      case (keys, block) => keys -> (block += 1.0)
    }

    val inCoreB = B.collect
    val inCoreBControl = inCoreA + 1.0

    println(inCoreB)

    // Assert they are the same
    (inCoreB - inCoreBControl).norm should be < 1E-10

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

}
