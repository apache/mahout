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
import scala.reflect.ClassTag

/** Common DRM tests to be run by all distributed engines. */
trait DrmLikeSuiteBase extends DistributedMahoutSuite with Matchers {
  this: FunSuite =>

  test("DRM DFS i/o (local)") {

    val uploadPath = TmpDir + "UploadedDRM"

    val inCoreA = dense((1, 2, 3), (3, 4, 5))
    val drmA = drmParallelize(inCoreA)

    drmA.dfsWrite(path = uploadPath)

    println(inCoreA)

    // Load back from hdfs
    val drmB = drmDfsRead(path = uploadPath)

    // Make sure keys are correctly identified as ints
    drmB.checkpoint(CacheHint.NONE).keyClassTag shouldBe ClassTag.Int

    // Collect back into in-core
    val inCoreB = drmB.collect

    // Print out to see what it is we collected:
    println(inCoreB)

    (inCoreA - inCoreB).norm should be < 1e-7
  }

  test("DRM parallelizeEmpty") {

    val drmEmpty = drmParallelizeEmpty(100, 50)

    // collect back into in-core
    val inCoreEmpty = drmEmpty.collect

    inCoreEmpty.sum.abs should be < 1e-7
    drmEmpty.nrow shouldBe 100
    drmEmpty.ncol shouldBe 50
    inCoreEmpty.nrow shouldBe 100
    inCoreEmpty.ncol shouldBe 50

  }



}
