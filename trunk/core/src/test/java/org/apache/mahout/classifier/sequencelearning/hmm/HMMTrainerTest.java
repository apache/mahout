/**
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

package org.apache.mahout.classifier.sequencelearning.hmm;

import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class HMMTrainerTest extends HMMTestBase {

  @Test
  public void testViterbiTraining() {
    // initialize the expected model parameters (from R)
    // expected transition matrix
    double[][] transitionE = {{0.3125, 0.0625, 0.3125, 0.3125},
        {0.25, 0.25, 0.25, 0.25}, {0.5, 0.071429, 0.357143, 0.071429},
        {0.5, 0.1, 0.1, 0.3}};
    // initialize the emission matrix
    double[][] emissionE = {{0.882353, 0.058824, 0.058824},
        {0.333333, 0.333333, 0.3333333}, {0.076923, 0.846154, 0.076923},
        {0.111111, 0.111111, 0.777778}};

    // train the given network to the following output sequence
    int[] observed = {1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0};

    HmmModel trained = HmmTrainer.trainViterbi(getModel(), observed, 0.5, 0.1, 10, false);

    // now check whether the model matches our expectations
    Matrix emissionMatrix = trained.getEmissionMatrix();
    Matrix transitionMatrix = trained.getTransitionMatrix();

    for (int i = 0; i < trained.getNrOfHiddenStates(); ++i) {
      for (int j = 0; j < trained.getNrOfHiddenStates(); ++j) {
        assertEquals(transitionMatrix.getQuick(i, j), transitionE[i][j], EPSILON);
      }

      for (int j = 0; j < trained.getNrOfOutputStates(); ++j) {
        assertEquals(emissionMatrix.getQuick(i, j), emissionE[i][j], EPSILON);
      }
    }

  }

  @Test
  public void testScaledViterbiTraining() {
    // initialize the expected model parameters (from R)
    // expected transition matrix
    double[][] transitionE = {{0.3125, 0.0625, 0.3125, 0.3125},
        {0.25, 0.25, 0.25, 0.25}, {0.5, 0.071429, 0.357143, 0.071429},
        {0.5, 0.1, 0.1, 0.3}};
    // initialize the emission matrix
    double[][] emissionE = {{0.882353, 0.058824, 0.058824},
        {0.333333, 0.333333, 0.3333333}, {0.076923, 0.846154, 0.076923},
        {0.111111, 0.111111, 0.777778}};

    // train the given network to the following output sequence
    int[] observed = {1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0};

    HmmModel trained = HmmTrainer.trainViterbi(getModel(), observed, 0.5, 0.1, 10,
        true);

    // now check whether the model matches our expectations
    Matrix emissionMatrix = trained.getEmissionMatrix();
    Matrix transitionMatrix = trained.getTransitionMatrix();

    for (int i = 0; i < trained.getNrOfHiddenStates(); ++i) {
      for (int j = 0; j < trained.getNrOfHiddenStates(); ++j) {
        assertEquals(transitionMatrix.getQuick(i, j), transitionE[i][j],
            EPSILON);
      }

      for (int j = 0; j < trained.getNrOfOutputStates(); ++j) {
        assertEquals(emissionMatrix.getQuick(i, j), emissionE[i][j],
            EPSILON);
      }
    }

  }

  @Test
  public void testBaumWelchTraining() {
    // train the given network to the following output sequence
    int[] observed = {1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0};

    // expected values from Matlab HMM package / R HMM package
    double[] initialExpected = {0, 0, 1.0, 0};
    double[][] transitionExpected = {{0.2319, 0.0993, 0.0005, 0.6683},
        {0.0001, 0.3345, 0.6654, 0}, {0.5975, 0, 0.4025, 0},
        {0.0024, 0.6657, 0, 0.3319}};
    double[][] emissionExpected = {{0.9995, 0.0004, 0.0001},
        {0.9943, 0.0036, 0.0021}, {0.0059, 0.9941, 0}, {0, 0, 1}};

    HmmModel trained = HmmTrainer.trainBaumWelch(getModel(), observed, 0.1, 10,
        false);

    Vector initialProbabilities = trained.getInitialProbabilities();
    Matrix emissionMatrix = trained.getEmissionMatrix();
    Matrix transitionMatrix = trained.getTransitionMatrix();

    for (int i = 0; i < trained.getNrOfHiddenStates(); ++i) {
      assertEquals(initialProbabilities.get(i), initialExpected[i],
          0.0001);
      for (int j = 0; j < trained.getNrOfHiddenStates(); ++j) {
        assertEquals(transitionMatrix.getQuick(i, j),
            transitionExpected[i][j], 0.0001);
      }
      for (int j = 0; j < trained.getNrOfOutputStates(); ++j) {
        assertEquals(emissionMatrix.getQuick(i, j),
            emissionExpected[i][j], 0.0001);
      }
    }
  }

  @Test
  public void testScaledBaumWelchTraining() {
    // train the given network to the following output sequence
    int[] observed = {1, 0, 2, 2, 0, 0, 1, 1, 1, 0, 2, 0, 1, 0, 0};

    // expected values from Matlab HMM package / R HMM package
    double[] initialExpected = {0, 0, 1.0, 0};
    double[][] transitionExpected = {{0.2319, 0.0993, 0.0005, 0.6683},
        {0.0001, 0.3345, 0.6654, 0}, {0.5975, 0, 0.4025, 0},
        {0.0024, 0.6657, 0, 0.3319}};
    double[][] emissionExpected = {{0.9995, 0.0004, 0.0001},
        {0.9943, 0.0036, 0.0021}, {0.0059, 0.9941, 0}, {0, 0, 1}};

    HmmModel trained = HmmTrainer
        .trainBaumWelch(getModel(), observed, 0.1, 10, true);

    Vector initialProbabilities = trained.getInitialProbabilities();
    Matrix emissionMatrix = trained.getEmissionMatrix();
    Matrix transitionMatrix = trained.getTransitionMatrix();

    for (int i = 0; i < trained.getNrOfHiddenStates(); ++i) {
      assertEquals(initialProbabilities.get(i), initialExpected[i],
          0.0001);
      for (int j = 0; j < trained.getNrOfHiddenStates(); ++j) {
        assertEquals(transitionMatrix.getQuick(i, j),
            transitionExpected[i][j], 0.0001);
      }
      for (int j = 0; j < trained.getNrOfOutputStates(); ++j) {
        assertEquals(emissionMatrix.getQuick(i, j),
            emissionExpected[i][j], 0.0001);
      }
    }
  }

}
