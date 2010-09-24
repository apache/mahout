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

import junit.framework.Assert;

import org.apache.mahout.math.Matrix;
import org.junit.Test;

public class HMMAlgorithmsTest extends HMMTestBase {

  /**
   * Test the forward algorithm by comparing the alpha values with the values
   * obtained from HMM R model. We test the test observation sequence "O1" "O0"
   * "O2" "O2" "O0" "O0" "O1" by comparing the generated alpha values to the
   * R-generated "reference".
   */
  @Test
  public void testForwardAlgorithm() {
    // intialize the expected alpha values
    double alphaExpectedA[][] = {
        {0.02, 0.0392, 0.002438, 0.00035456, 0.0011554672, 7.158497e-04,
            4.614927e-05},
        {0.01, 0.0054, 0.001824, 0.00069486, 0.0007586904, 2.514137e-04,
            1.721505e-05},
        {0.32, 0.0262, 0.002542, 0.00038026, 0.0001360234, 3.002345e-05,
            9.659608e-05},
        {0.03, 0.0000, 0.013428, 0.00951084, 0.0000000000, 0.000000e+00,
            2.428986e-05},};
    // fetch the alpha matrix using the forward algorithm
    Matrix alpha = HmmAlgorithms.forwardAlgorithm(model, sequence, false);
    // first do some basic checking
    Assert.assertNotNull(alpha);
    Assert.assertEquals(alpha.numCols(), 4);
    Assert.assertEquals(alpha.numRows(), 7);
    // now compare the resulting matrices
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 7; ++j)
        Assert.assertEquals(alphaExpectedA[i][j], alpha.get(j, i), 0.00001);
  }

  @Test
  public void testLogScaledForwardAlgorithm() {
    // intialize the expected alpha values
    double alphaExpectedA[][] = {
        {0.02, 0.0392, 0.002438, 0.00035456, 0.0011554672, 7.158497e-04,
            4.614927e-05},
        {0.01, 0.0054, 0.001824, 0.00069486, 0.0007586904, 2.514137e-04,
            1.721505e-05},
        {0.32, 0.0262, 0.002542, 0.00038026, 0.0001360234, 3.002345e-05,
            9.659608e-05},
        {0.03, 0.0000, 0.013428, 0.00951084, 0.0000000000, 0.000000e+00,
            2.428986e-05},};
    // fetch the alpha matrix using the forward algorithm
    Matrix alpha = HmmAlgorithms.forwardAlgorithm(model, sequence, true);
    // first do some basic checking
    Assert.assertNotNull(alpha);
    Assert.assertEquals(alpha.numCols(), 4);
    Assert.assertEquals(alpha.numRows(), 7);
    // now compare the resulting matrices
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 7; ++j)
        Assert.assertEquals(Math.log(alphaExpectedA[i][j]), alpha.get(j, i),
            0.00001);
  }

  /**
   * Test the backward algorithm by comparing the beta values with the values
   * obtained from HMM R model. We test the following observation sequence "O1"
   * "O0" "O2" "O2" "O0" "O0" "O1" by comparing the generated beta values to the
   * R-generated "reference".
   */
  @Test
  public void testBackwardAlgorithm() {
    // intialize the expected beta values
    double betaExpectedA[][] = {
        {0.0015730559, 0.003543656, 0.00738264, 0.040692, 0.0848, 0.17, 1},
        {0.0017191865, 0.002386795, 0.00923652, 0.052232, 0.1018, 0.17, 1},
        {0.0003825772, 0.001238558, 0.00259464, 0.012096, 0.0664, 0.66, 1},
        {0.0004390858, 0.007076994, 0.01063512, 0.013556, 0.0304, 0.17, 1}};
    // fetch the beta matrix using the backward algorithm
    Matrix beta = HmmAlgorithms.backwardAlgorithm(model, sequence, false);
    // first do some basic checking
    Assert.assertNotNull(beta);
    Assert.assertEquals(beta.numCols(), 4);
    Assert.assertEquals(beta.numRows(), 7);
    // now compare the resulting matrices
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 7; ++j)
        Assert.assertEquals(betaExpectedA[i][j], beta.get(j, i), 0.00001);
  }

  @Test
  public void testLogScaledBackwardAlgorithm() {
    // intialize the expected beta values
    double betaExpectedA[][] = {
        {0.0015730559, 0.003543656, 0.00738264, 0.040692, 0.0848, 0.17, 1},
        {0.0017191865, 0.002386795, 0.00923652, 0.052232, 0.1018, 0.17, 1},
        {0.0003825772, 0.001238558, 0.00259464, 0.012096, 0.0664, 0.66, 1},
        {0.0004390858, 0.007076994, 0.01063512, 0.013556, 0.0304, 0.17, 1}};
    // fetch the beta matrix using the backward algorithm
    Matrix beta = HmmAlgorithms.backwardAlgorithm(model, sequence, true);
    // first do some basic checking
    Assert.assertNotNull(beta);
    Assert.assertEquals(beta.numCols(), 4);
    Assert.assertEquals(beta.numRows(), 7);
    // now compare the resulting matrices
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 7; ++j)
        Assert.assertEquals(Math.log(betaExpectedA[i][j]), beta.get(j, i),
            0.00001);
  }

  @Test
  public void testViterbiAlgorithm() {
    // initialize the expected hidden sequence
    int[] expected = {2, 0, 3, 3, 0, 0, 2};
    // fetch the viterbi generated sequence
    int[] computed = HmmAlgorithms.viterbiAlgorithm(model, sequence, false);
    // first make sure we return the correct size
    Assert.assertNotNull(computed);
    Assert.assertEquals(computed.length, sequence.length);
    // now check the contents
    for (int i = 0; i < sequence.length; ++i)
      Assert.assertEquals(expected[i], computed[i]);
  }

  @Test
  public void testLogScaledViterbiAlgorithm() {
    // initialize the expected hidden sequence
    int[] expected = {2, 0, 3, 3, 0, 0, 2};
    // fetch the viterbi generated sequence
    int[] computed = HmmAlgorithms.viterbiAlgorithm(model, sequence, true);
    // first make sure we return the correct size
    Assert.assertNotNull(computed);
    Assert.assertEquals(computed.length, sequence.length);
    // now check the contents
    for (int i = 0; i < sequence.length; ++i)
      Assert.assertEquals(expected[i], computed[i]);

  }

}
