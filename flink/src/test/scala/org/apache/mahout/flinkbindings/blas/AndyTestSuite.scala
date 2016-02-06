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

@RunWith(classOf[JUnitRunner])
class AndyTestSuite extends FunSuite with DistributedFlinkSuite with Matchers {

//  test("Ax blockified") {
//    val inCoreA = dense((1, 2, 3), (2, 3, 4), (3, 4, 5))
//    val A = drmParallelize(m = inCoreA, numPartitions = 2)
//    val x: Vector = (0, 1, 2)
//
//    val opAx = new OpAx(A, x)
//    val res = FlinkOpAx.blockifiedBroadcastAx(opAx, A)
//    val drm = new CheckpointedFlinkDrm(res.asRowWise.ds)
//    val output = drm.collect
//
//    val b = output(::, 0)
//    assert(b == dvec(8, 11, 14))
//  }



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



}