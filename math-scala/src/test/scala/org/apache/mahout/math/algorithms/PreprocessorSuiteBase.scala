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

import org.apache.mahout.math.algorithms.preprocessing.{AsFactor, AsFactorModel}
import org.apache.mahout.math.drm.drmParallelize
import org.apache.mahout.math.scalabindings.{dense, sparse, svec}
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.test.DistributedMahoutSuite
import org.scalatest.{FunSuite, Matchers}

trait PreprocessorSuiteBase extends DistributedMahoutSuite with Matchers {

  this: FunSuite =>

  test("asfactor test") {
    val A = drmParallelize(dense(
      (3, 2, 1, 2),
      (0, 0, 0, 0),
      (1, 1, 1, 1)), numPartitions = 2)

    // 0 -> 2, 3 -> 5, 6 -> 9
    val factorizer: AsFactorModel = new AsFactor().fit(A)

    val factoredA = factorizer.transform(A)

    println(factoredA)
    println(factorizer.factorMap)
    val correctAnswer = sparse(
      svec((3 → 1.0) :: (6 → 1.0) :: (8 → 1.0) :: (11 → 1.0) :: Nil, cardinality = 12),
      svec((0 → 1.0) :: (4 → 1.0) :: (7 → 1.0) :: ( 9 → 1.0) :: Nil, cardinality = 12),
      svec((1 → 1.0) :: (5 → 1.0) :: (8 → 1.0) :: (10 → 1.0) :: Nil, cardinality = 12)
    )

    val myAnswer = factoredA.collect

    val epsilon = 1E-6
    (myAnswer.norm - correctAnswer.norm) should be <= epsilon
    (myAnswer.norm - correctAnswer.norm) should be <= epsilon

  }
}
