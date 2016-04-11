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

import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.scalatest.FunSuite


class DrmLikeOpsSuite extends FunSuite with DistributedFlinkSuite {

  test("norm") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    (inCoreA.norm - A.norm) should be < 1e-6
  }

  test("colSums") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    
    (inCoreA.colSums - A.colSums).norm(2) should be < 1e-6
  }

  test("rowSums") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    
    (inCoreA.rowSums - A.rowSums).norm(2) should be < 1e-6
  }

  test("rowMeans") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)
    
    (inCoreA.rowMeans - A.rowMeans).norm(2) should be < 1e-6
  }

  test("numNonZeroElementsPerColumn") {
    val A = dense((0, 2), (3, 0), (0, -30))
    val drmA = drmParallelize(A, numPartitions = 2)

    drmA.numNonZeroElementsPerColumn() should equal(A.numNonZeroElementsPerColumn())
  }


  test("drmParallelizeEmpty") {
    val emptyDrm = drmParallelizeEmpty(nrow = 2, ncol = 2, numPartitions = 2)
    val expected = dense((0, 0), (0, 0))

    (emptyDrm.collect - expected).norm should be < 1e-6
  }

}