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

import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.sparkbindings._
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.scalatest.FunSuite

import scala.reflect.ClassTag

/** Tests for DrmLikeOps */
class DrmLikeOpsSuite extends FunSuite with DistributedSparkSuite with DrmLikeOpsSuiteBase {

  test("exact, min and auto ||") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (4, 5, 6))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    A.rdd.partitions.length should equal(2)

    (A + 1.0).par(exact = 4).rdd.partitions.length should equal(4)
    A.par(exact = 2).rdd.partitions.length should equal(2)
    A.par(exact = 1).rdd.partitions.length should equal(1)

    A.par(min = 4).rdd.partitions.length should equal(4)
    A.par(min = 2).rdd.partitions.length should equal(2)
    A.par(min = 1).rdd.partitions.length should equal(2)
    A.par(auto = true).rdd.partitions.length should equal(10)
    A.par(exact = 10).par(auto = true).rdd.partitions.length should equal(10)
    A.par(exact = 11).par(auto = true).rdd.partitions.length should equal(19)
    A.par(exact = 20).par(auto = true).rdd.partitions.length should equal(19)

    A.keyClassTag shouldBe ClassTag.Int
    A.par(auto = true).keyClassTag shouldBe ClassTag.Int

    an[IllegalArgumentException] shouldBe thrownBy {A.par(exact = 0)}
    an[IllegalArgumentException] shouldBe thrownBy {A.par()}
  }

}
