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

package org.apache.mahout.sparkbindings.blas

import org.scalatest.FunSuite
import org.apache.mahout.sparkbindings.test.DistributedSparkSuite
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm._
import RLikeDrmOps._
import org.apache.spark.SparkContext._
import org.apache.mahout.math.drm.logical.OpAewB
import org.apache.mahout.sparkbindings.drm.CheckpointedDrmSpark

/** Elementwise matrix operation tests */
class AewBSuite extends FunSuite with DistributedSparkSuite {

  test("A * B Hadamard") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (7, 8, 9))
    val inCoreB = dense((3, 4, 5), (5, 6, 7), (0, 0, 0), (9, 8, 7))
    val drmA = drmParallelize(m = inCoreA, numPartitions = 2)
    val drmB = drmParallelize(m = inCoreB)

    val op = new OpAewB(drmA, drmB, "*")

    val drmM = new CheckpointedDrmSpark(AewB.a_ew_b(op, srcA = drmA, srcB = drmB), op.nrow, op.ncol)

    val inCoreM = drmM.collect
    val inCoreMControl = inCoreA * inCoreB

    assert((inCoreM - inCoreMControl).norm < 1E-10)

  }

  test("A + B Elementwise") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (7, 8, 9))
    val inCoreB = dense((3, 4, 5), (5, 6, 7), (0, 0, 0), (9, 8, 7))
    val drmA = drmParallelize(m = inCoreA, numPartitions = 2)
    val drmB = drmParallelize(m = inCoreB)

    val op = new OpAewB(drmA, drmB, "+")

    val drmM = new CheckpointedDrmSpark(AewB.a_ew_b(op, srcA = drmA, srcB = drmB), op.nrow, op.ncol)

    val inCoreM = drmM.collect
    val inCoreMControl = inCoreA + inCoreB

    assert((inCoreM - inCoreMControl).norm < 1E-10)

  }

  test("A - B Elementwise") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5), (7, 8, 9))
    val inCoreB = dense((3, 4, 5), (5, 6, 7), (0, 0, 0), (9, 8, 7))
    val drmA = drmParallelize(m = inCoreA, numPartitions = 2)
    val drmB = drmParallelize(m = inCoreB)

    val op = new OpAewB(drmA, drmB, "-")

    val drmM = new CheckpointedDrmSpark(AewB.a_ew_b(op, srcA = drmA, srcB = drmB), op.nrow, op.ncol)

    val inCoreM = drmM.collect
    val inCoreMControl = inCoreA - inCoreB

    assert((inCoreM - inCoreMControl).norm < 1E-10)

  }

  test("A / B Elementwise") {
    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 0), (7, 8, 9))
    val inCoreB = dense((3, 4, 5), (5, 6, 7), (10, 20, 30), (9, 8, 7))
    val drmA = drmParallelize(m = inCoreA, numPartitions = 2)
    val drmB = drmParallelize(m = inCoreB)

    val op = new OpAewB(drmA, drmB, "/")

    val drmM = new CheckpointedDrmSpark(AewB.a_ew_b(op, srcA = drmA, srcB = drmB), op.nrow, op.ncol)

    val inCoreM = drmM.collect
    val inCoreMControl = inCoreA / inCoreB

    assert((inCoreM - inCoreMControl).norm < 1E-10)

  }

}
