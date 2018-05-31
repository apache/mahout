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

package org.apache.mahout.h2obindings.ops

import org.scalatest.FunSuite
import org.apache.mahout.h2obindings.test.DistributedH2OSuite
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.math.drm._
import RLikeOps._
import RLikeDrmOps._
import org.apache.mahout.math.drm._

/** Tests for A' algorithms */
class AtSuite extends FunSuite with DistributedH2OSuite {
  test("At") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))
    val A = drmParallelize(m = inCoreA, numPartitions = 2)

    val AtDrm = A.t
    val inCoreAt = AtDrm.collect
    val inCoreControlAt = inCoreA.t

    println(inCoreAt)
    assert((inCoreAt - inCoreControlAt).norm < 1E-5)
  }
}
