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
  val transitionMatrix = dense(
    (0.5, 0.1, 0.1, 0.3),
    (0.4, 0.4, 0.1, 0.1),
    (0.1, 0.0, 0.8, 0.1),
    (0.1, 0.1, 0.1, 0.7))

  val emissionMatrix = dense(
    (0.8, 0.1, 0.1),
    (0.6, 0.1, 0.3),
    (0.1, 0.8, 0.1),
    (0.0, 0.1, 0.9))

  val initialProbabilities = dvec(0.2, 0.1, 0.4, 0.3)
  val observations = dense((1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0))
  val multipleObservations = sparse(
    (0, 1) :: (1, 0) :: (2, 2) :: (3, 2) :: (4, 0) :: (5, 0) :: (6, 1) :: (7, 1) :: (8, 1) ::  (9, 0) :: (10, 2) :: (11, 0) :: (12, 1) :: (13, 0) :: (14, 0) :: Nil,
    (0, 1) :: (1, 2) :: (2, 0) :: (3, 1) :: (4, 0) :: (5, 0) :: (6, 1) :: (7, 2) :: (8, 1) ::  (9, 2) :: (10, 0) :: (11, 2) :: (12, 1) ::  Nil)

  val transitionMatrixExpected = dense(
    (0.2319, 0.0993, 0.0005, 0.6683),
    (0.0001, 0.3345, 0.6654, 0),
    (0.5975, 0, 0.4025, 0),
    (0.0024, 0.6657, 0, 0.3319))

  val emissionMatrixExpected = dense(
    (0.9995, 0.0004, 0.0001),
    (0.9943, 0.0036, 0.0021),
    (0.0059, 0.9941, 0),
    (0, 0, 1))

  val initialProbabilitiesExpected = dvec(0, 0, 1.0, 0)

  val mTransitionMatrixExpected = dense(
    (0.3271599282990272, 0.1086335029682492, 0.18057285889124233, 0.38363370984148126),
    (0.26045820264596237, 0.5859457737112057, 0.13554086908114193, 0.01805515456169015),
    (0.5619947559092133, 0.0, 0.04416126697971683, 0.3938439771110697),
    (0.4458621414345233, 0.3784235350634106, 0.06095036408301116, 0.11476395941905497))

  val mEmissionMatrixExpected = dense(
    (0.8074678111696147, 0.11708554052631344, 0.07544664830407177),
    (0.5718025775415391, 0.3336663565194518, 0.09453106593900908),
    (0.013579876754409118, 0.9688892915228535, 0.01753083172273737),
    (0.0, 0.1770898706525618, 0.8229101293474382))

  val mInitialProbabilitiesExpected = dvec(1.1527590099040409E-7, 1.314705472922704E-6, 0.9999909531404041, 7.616878221414395E-6)

  def compareModels(trainedModel:HMMModel, iExpected:DenseVector, tExpected:DenseMatrix, eExpected:DenseMatrix):Unit = {
    for (indexI <- 0 until trainedModel.getNumberOfHiddenStates) {
      trainedModel.getInitialProbabilities.getQuick(indexI) - iExpected.getQuick(indexI) should be < epsilon

      for (indexJ <- 0 until trainedModel.getNumberOfHiddenStates) {
        trainedModel.getTransitionMatrix.getQuick(indexI, indexJ) - tExpected.getQuick(indexI, indexJ) should be < epsilon
      }

      for (indexJ <- 0 until trainedModel.getNumberOfObservableSymbols) {
        trainedModel.getEmissionMatrix.getQuick(indexI, indexJ) - eExpected.getQuick(indexI, indexJ) should be < epsilon
      }
    }
  }

  test("Simple Standard HMM Model") {
    val observationsDrm:DrmLike[Long] = drm.drmParallelize(m = observations, numPartitions = 1)
    // Re-key into DrmLike[Long] instead of [Int]
      .mapBlock()({
        case (keys, block) => keys.map(_.toLong) -> block
      })

    val initModel = new HMMModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val trainedModel = HiddenMarkovModel.train(initModel, observationsDrm, 0.1, 10, false)
    trainedModel.validate()
    compareModels(trainedModel, initialProbabilitiesExpected, transitionMatrixExpected, emissionMatrixExpected)
  }

  test("Simple Standard HMM Model with scaling") {
    val observationsDrm:DrmLike[Long] = drm.drmParallelize(m = observations, numPartitions = 1)
    // Re-key into DrmLike[Long] instead of [Int]
      .mapBlock()({
        case (keys, block) => keys.map(_.toLong) -> block
      })

    val initModel = new HMMModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val trainedModel = HiddenMarkovModel.train(initModel, observationsDrm, 0.1, 10, true)
    trainedModel.validate()
    compareModels(trainedModel, initialProbabilitiesExpected, transitionMatrixExpected, emissionMatrixExpected)
  }

  test("Simple Standard HMM Model with multiple observations") {
    val observationsDrm:DrmLike[Long] = drm.drmParallelize(m = multipleObservations, numPartitions = 1)
    // Re-key into DrmLike[Long] instead of [Int]
      .mapBlock()({
        case (keys, block) => keys.map(_.toLong) -> block
      })

    val initModel = new HMMModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val trainedModel = HiddenMarkovModel.train(initModel, observationsDrm, 0.1, 10, false)
    trainedModel.validate()
    compareModels(trainedModel, mInitialProbabilitiesExpected, mTransitionMatrixExpected, mEmissionMatrixExpected)
  }

  test("Simple Standard HMM Model with multiple observations with multiple partitions") {
    val observationsDrm:DrmLike[Long] = drm.drmParallelize(m = multipleObservations, numPartitions = 2)
    // Re-key into DrmLike[Long] instead of [Int]
      .mapBlock()({
        case (keys, block) => keys.map(_.toLong) -> block
      })

    val initModel = new HMMModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val trainedModel = HiddenMarkovModel.train(initModel, observationsDrm, 0.1, 10, false)
    trainedModel.validate()
    compareModels(trainedModel, mInitialProbabilitiesExpected, mTransitionMatrixExpected, mEmissionMatrixExpected)
  }

  test("Decode an observation sequence with viterbi algorithm") {
    val initModel = new HMMModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val observationSequence = dvec(1, 0, 2, 2, 0, 0, 1)
    val expectedHiddenSeq = dvec(2, 0, 3, 3, 0, 0, 2)
    val hiddenSeq = HiddenMarkovModel.decode(initModel, observationSequence, false)
    for (i <- 0 until observationSequence.size) {
      expectedHiddenSeq.getQuick(i).toInt shouldBe hiddenSeq.getQuick(i).toInt
    }
  }

  test("Decode an observation sequence with viterbi algorithm (with log scaling)") {
    val initModel = new HMMModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val observationSequence = dvec(1, 0, 2, 2, 0, 0, 1)
    val expectedHiddenSeq = dvec(2, 0, 3, 3, 0, 0, 2)
    val hiddenSeq = HiddenMarkovModel.decode(initModel, observationSequence, true)
    for (i <- 0 until observationSequence.size) {
      expectedHiddenSeq.getQuick(i).toInt shouldBe hiddenSeq.getQuick(i).toInt
    }
  }

  test("likelihood of an observation seqeunce given a model") {
    val initModel = new HMMModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val observationSequence = dvec(1, 0, 2, 2, 0, 0, 1)
    val expectedLikelihood = 1.8425e-4
    val likelihood = HiddenMarkovModel.likelihood(initModel, observationSequence, false)
    likelihood - expectedLikelihood should be < epsilon
  }

  test("likelihood of an observation seqeunce given a model (with scaling)") {
    val initModel = new HMMModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val observationSequence = dvec(1, 0, 2, 2, 0, 0, 1)
    val expectedLikelihood = 1.8425e-4
    val likelihood = HiddenMarkovModel.likelihood(initModel, observationSequence, true)
    likelihood - expectedLikelihood should be < epsilon
  }
}
