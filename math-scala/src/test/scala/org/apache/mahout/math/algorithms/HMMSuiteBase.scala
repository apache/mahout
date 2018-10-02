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

package org.apache.mahout.math.algorithms

import org.apache.mahout.math.algorithms.hmm._
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

trait HMMSuiteBase extends DistributedMahoutSuite with Matchers {
  this:FunSuite =>

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
  val multipleObservations = dense((1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0),
  (1, 2, 0, 1, 0, 0, 1, 2, 1, 2, 0, 2, 1))
   
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
    (0.0, 0.0009, 0.0, 0.9991),
    (0.0, 0.8180, 0.1820, 0.0),
    (0.4001, 0.0, 0.0, 0.5999),
    (0.1431, 0.4912, 0.2228, 0.1429))

  val mEmissionMatrixExpected = dense(
    (1.0, 0.0, 0.0),
    (0.6430, 0.3570, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0))

  val mInitialProbabilitiesExpected = dvec(0.0, 0.0, 1.0, 0.0)

  val cTransitionMatrixExpected = dense(
    (0.2336, 0.0997, 0.0, 0.6667),
    (0.0, 0.3333, 0.6667, 0.0),
    (0.6, 0.0, 0.400, 0.0),
    (0.0, 0.6667, 0.0, 0.3333))

  val cEmissionMatrixExpected = dense(
    (1.0, 0.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0001, 0.9999, 0.0),
    (0.0, 0.0, 1.0))

  val cInitialProbabilitiesExpected = dvec(0.0, 0.0, 1.0, 0.0)

  def compareModels(trainedModel:HiddenMarkovModel, iExpected:DenseVector, tExpected:DenseMatrix, eExpected:DenseMatrix):Unit = {
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
    val observationsDrm = drm.drmParallelize(m = observations, numPartitions = 1)

    val initModel = new HiddenMarkovModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val trainedModel = new HiddenMarkovModelFitter().fit(observationsDrm, 'initModel -> initModel, 'epsilon -> 0.1, 'iterations -> 10, 'scale -> false)
    trainedModel.validate()
    compareModels(trainedModel, initialProbabilitiesExpected, transitionMatrixExpected, emissionMatrixExpected)
  }

  test("Simple Standard HMM Model (smaller convergence)") {
    val observationsDrm = drm.drmParallelize(m = observations, numPartitions = 1)

    val initModel = new HiddenMarkovModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val trainedModel = new HiddenMarkovModelFitter().fit(observationsDrm, 'initModel -> initModel, 'epsilon -> 0.0001, 'iterations -> 100, 'scale -> false)
    trainedModel.validate()
    compareModels(trainedModel, cInitialProbabilitiesExpected, cTransitionMatrixExpected, cEmissionMatrixExpected)
  }

  test("Simple Standard HMM Model with scaling") {
    val observationsDrm = drm.drmParallelize(m = observations, numPartitions = 1)

    val initModel = new HiddenMarkovModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val trainedModel = new HiddenMarkovModelFitter().fit(observationsDrm, 'initModel -> initModel, 'epsilon -> 0.1, 'iterations -> 10, 'scale -> true)
    trainedModel.validate()
    compareModels(trainedModel, initialProbabilitiesExpected, transitionMatrixExpected, emissionMatrixExpected)
  }

  test("Simple Standard HMM Model with multiple observations") {
    val observationsDrm = drm.drmParallelize(m = multipleObservations, numPartitions = 1)

    val initModel = new HiddenMarkovModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val trainedModel = new HiddenMarkovModelFitter().fit(observationsDrm, 'initModel -> initModel, 'epsilon -> 0.0001, 'iterations -> 1000, 'scale -> false)
    trainedModel.validate()
    compareModels(trainedModel, mInitialProbabilitiesExpected, mTransitionMatrixExpected, mEmissionMatrixExpected)
  }

  test("Simple Standard HMM Model with multiple observations with multiple partitions") {
    val observationsDrm = drm.drmParallelize(m = multipleObservations, numPartitions = 2)

    val initModel = new HiddenMarkovModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val trainedModel = new HiddenMarkovModelFitter().fit(observationsDrm, 'initModel -> initModel, 'epsilon -> 0.0001, 'iterations -> 1000, 'scale -> false)
    trainedModel.validate()
    compareModels(trainedModel, mInitialProbabilitiesExpected, mTransitionMatrixExpected, mEmissionMatrixExpected)
  }

  test("Decode an observation sequence with viterbi algorithm") {
    val initModel = new HiddenMarkovModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val observationSequence = dvec(1, 0, 2, 2, 0, 0, 1)
    val expectedHiddenSeq = dvec(2, 0, 3, 3, 0, 0, 2)
    val hiddenSeq = initModel.decode(observationSequence, false)
    for (i <- 0 until observationSequence.size) {
      expectedHiddenSeq.getQuick(i).toInt shouldBe hiddenSeq.getQuick(i).toInt
    }
  }

  test("Decode an observation sequence with viterbi algorithm (with log scaling)") {
    val initModel = new HiddenMarkovModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val observationSequence = dvec(1, 0, 2, 2, 0, 0, 1)
    val expectedHiddenSeq = dvec(2, 0, 3, 3, 0, 0, 2)
    val hiddenSeq = initModel.decode(observationSequence, true)
    for (i <- 0 until observationSequence.size) {
      expectedHiddenSeq.getQuick(i).toInt shouldBe hiddenSeq.getQuick(i).toInt
    }
  }

  test("likelihood of an observation seqeunce given a model") {
    val initModel = new HiddenMarkovModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val observationSequence = dvec(1, 0, 2, 2, 0, 0, 1)
    val expectedLikelihood = 1.8425e-4
    val likelihood = initModel.likelihood(observationSequence, false)
    likelihood - expectedLikelihood should be < epsilon
  }

  test("likelihood of an observation seqeunce given a model (with scaling)") {
    val initModel = new HiddenMarkovModel(4, 3, transitionMatrix, emissionMatrix, initialProbabilities)
    val observationSequence = dvec(1, 0, 2, 2, 0, 0, 1)
    val expectedLikelihood = 1.8425e-4
    val likelihood = initModel.likelihood(observationSequence, true)
    likelihood - expectedLikelihood should be < epsilon
  }
}
