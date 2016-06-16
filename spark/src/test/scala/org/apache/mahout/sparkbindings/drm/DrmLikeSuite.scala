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
import drm._
import RLikeOps._
import RLikeDrmOps._
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite


/** DRMLike tests -- just run common DRM tests in Spark. */
class DrmLikeSuite extends FunSuite with DistributedSparkSuite with DrmLikeSuiteBase {

  test("drmParallellize produces drm with no missing rows") {
    val inCoreA = dense((1, 2, 3), (3, 4, 5))
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    drmA.canHaveMissingRows shouldBe false
  }

  test("DRM blockify dense") {

    val inCoreA = dense((1, 2, 3), (3, 4, 5))
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    (inCoreA - drmA.mapBlock() {
      case (keys, block) =>
        if (!block.isInstanceOf[DenseMatrix])
          throw new AssertionError("Block must be dense.")
        keys -> block
    }).norm should be < 1e-4
  }

  test("DRM blockify sparse -> DRM") {

    val inCoreA = sparse(
      (1, 2, 3),
      0 -> 3 :: 2 -> 5 :: Nil
    )
    val drmA = drmParallelize(inCoreA, numPartitions = 2)

    (inCoreA - drmA.mapBlock() {
      case (keys, block) =>
        if (block.isInstanceOf[SparseRowMatrix])
          throw new AssertionError("Block must be dense.")
        keys -> block
    }).norm should be < 1e-4
  }

}
