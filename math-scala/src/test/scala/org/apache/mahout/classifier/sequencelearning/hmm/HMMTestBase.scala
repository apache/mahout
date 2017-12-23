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

package org.apache.mahout.classifier.sequencelearning.hmm

import org.apache.mahout.math.drm
import org.apache.mahout.math._
import org.apache.mahout.math.drm.DrmLike
import org.apache.mahout.math.drm.{CheckpointedDrm, drmParallelizeEmpty}
import org.apache.mahout.math.drm.RLikeDrmOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.test.DistributedMahoutSuite
import org.apache.mahout.test.MahoutSuite
import org.scalatest.{FunSuite, Matchers}
import collection._
import JavaConversions._
import collection.JavaConversions

trait HMMTestBase extends DistributedMahoutSuite with Matchers { this:FunSuite =>

  val epsilon = 0.0001

  test("Simple Standard NHMM Model") {
    var transitionMatrix = dense(
      (0.5, 0.1, 0.1, 0.3),
      (0.4, 0.4, 0.1, 0.1),
      (0.1, 0.0, 0.8, 0.1),
      (0.1, 0.1, 0.1, 0.7))

    var emissionMatrix = dense(
      (0.8, 0.1, 0.1),
      (0.6, 0.1, 0.3),
      (0.1, 0.8, 0.1),
      (0.0, 0.1, 0.9))

    var initialProbabilities = dvec(0.2, 0.1, 0.4, 0.3)
    val observations = dense((1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0))

    val observationsDrm:DrmLike[Long] = drm.drmParallelize(m = observations, numPartitions = 1)
    // Re-key into DrmLike[Long] instead of [Int]
        .mapBlock()({
      case (keys, block) => keys.map(_.toLong) -> block
        })
    val initModel = new HMMModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val trainedModel = HiddenMarkovModel.train(initModel, observationsDrm, 0.1, 10, false)
    trainedModel.validate()

    var transitionMatrixExpected = dense(
      (0.2319, 0.0993, 0.0005, 0.6683),
      (0.0001, 0.3345, 0.6654, 0),
      (0.5975, 0, 0.4025, 0),
      (0.0024, 0.6657, 0, 0.3319))

    var emissionMatrixExpected = dense(
      (0.9995, 0.0004, 0.0001),
      (0.9943, 0.0036, 0.0021),
      (0.0059, 0.9941, 0),
      (0, 0, 1))

    var initialProbabilitiesExpected = dvec(0, 0, 1.0, 0)

    for (indexI <- 0 to trainedModel.getNumberOfHiddenStates - 1) {
      trainedModel.getInitialProbabilities.getQuick(indexI) - initialProbabilitiesExpected.getQuick(indexI) should be < epsilon

      for (indexJ <- 0 to trainedModel.getNumberOfHiddenStates - 1) {
        trainedModel.getTransitionMatrix.getQuick(indexI, indexJ) - transitionMatrixExpected.getQuick(indexI, indexJ) should be < epsilon
      }

      for (indexJ <- 0 to trainedModel.getNumberOfObservableSymbols - 1) {
        trainedModel.getEmissionMatrix.getQuick(indexI, indexJ) - emissionMatrixExpected.getQuick(indexI, indexJ) should be < epsilon
      }
    }
  }

}
