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

package org.apache.mahout.math.algorithms

import org.apache.mahout.math.algorithms.preprocessing._
import org.apache.mahout.math.drm.drmParallelize
import org.apache.mahout.math.scalabindings.{dense, sparse, svec}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}

import org.apache.mahout.test.DistributedMahoutSuite

trait ClusteringSuiteBase extends DistributedMahoutSuite with Matchers {

  this: FunSuite =>

  test("canopy test") {
    val drmA = drmParallelize(dense((1.0, 1.2, 1.3, 1.4), (1.1, 1.5, 2.5, 1.0), (6.0, 5.2, -5.2, 5.3), (7.0,6.0, 5.0, 5.0), (10.0, 1.0, 20.0, -10.0)))

    import org.apache.mahout.math.algorithms.clustering.CanopyClustering

    val model = new CanopyClustering().fit(drmA, 't1 -> 6.5, 't2 -> 5.5, 'distanceMeasure -> 'Chebyshev)
    val myAnswer = model.cluster(drmA).collect

    val correctAnswer = dense((0.0), (0.0), (1.0), (0.0), (2.0))

    val epsilon = 1E-6
    (myAnswer.norm - correctAnswer.norm) should be <= epsilon
  }
}