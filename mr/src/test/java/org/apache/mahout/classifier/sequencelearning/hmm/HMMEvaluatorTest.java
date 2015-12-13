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
import org.junit.Test;

public class HMMEvaluatorTest extends HMMTestBase {

  /**
   * Test to make sure the computed model likelihood ist valid. Included tests
   * are: a) forwad == backward likelihood b) model likelihood for test seqeunce
   * is the expected one from R reference
   */
  @Test
  public void testModelLikelihood() {
    // compute alpha and beta values
      Matrix alpha = HmmAlgorithms.forwardAlgorithm(getModel(), getSequence(), HmmAlgorithms.ScalingMethod.NOSCALING, null);
      Matrix beta = HmmAlgorithms.backwardAlgorithm(getModel(), getSequence(), HmmAlgorithms.ScalingMethod.NOSCALING, null);
    // now test whether forward == backward likelihood
      double forwardLikelihood = HmmEvaluator.modelLikelihood(alpha, HmmAlgorithms.ScalingMethod.NOSCALING, null);
    double backwardLikelihood = HmmEvaluator.modelLikelihood(getModel(), getSequence(),
							     beta, HmmAlgorithms.ScalingMethod.NOSCALING, null);
    assertEquals(forwardLikelihood, backwardLikelihood, EPSILON);
    // also make sure that the likelihood matches the expected one
    assertEquals(1.8425e-4, forwardLikelihood, EPSILON);
  }

  /**
   * Test to make sure the computed model likelihood ist valid. Included tests
   * are: a) forwad == backward likelihood b) model likelihood for test seqeunce
   * is the expected one from R reference
   */
  @Test
  public void testScaledModelLikelihood() {
    // compute alpha and beta values
      Matrix alpha = HmmAlgorithms.forwardAlgorithm(getModel(), getSequence(), HmmAlgorithms.ScalingMethod.LOGSCALING, null);
      Matrix beta = HmmAlgorithms.backwardAlgorithm(getModel(), getSequence(), HmmAlgorithms.ScalingMethod.LOGSCALING, null);
    // now test whether forward == backward likelihood
      double forwardLikelihood = HmmEvaluator.modelLikelihood(alpha, HmmAlgorithms.ScalingMethod.LOGSCALING, null);
    double backwardLikelihood = HmmEvaluator.modelLikelihood(getModel(), getSequence(),
							     beta, HmmAlgorithms.ScalingMethod.LOGSCALING, null);
    assertEquals(forwardLikelihood, backwardLikelihood, EPSILON);
    // also make sure that the likelihood matches the expected one
    assertEquals(1.8425e-4, forwardLikelihood, EPSILON);
  }

    /**
   * Test to make sure the computed model likelihood ist valid. Included tests
   * are: a) forwad == backward likelihood b) model likelihood for test seqeunce
   * is the expected one from R reference
   */
  @Test
  public void testReScaledModelLikelihood() {
      double[] scalingFactors = new double[getSequence().length];
    // compute alpha and beta values
      Matrix alpha = HmmAlgorithms.forwardAlgorithm(getModel(), getSequence(), HmmAlgorithms.ScalingMethod.RESCALING, scalingFactors);
      Matrix beta = HmmAlgorithms.backwardAlgorithm(getModel(), getSequence(), HmmAlgorithms.ScalingMethod.RESCALING, scalingFactors);
    // now test whether forward == backward likelihood
      double forwardLikelihood = HmmEvaluator.modelLikelihood(alpha, HmmAlgorithms.ScalingMethod.RESCALING, scalingFactors);
    double backwardLikelihood = HmmEvaluator.modelLikelihood(getModel(), getSequence(),
							     beta, HmmAlgorithms.ScalingMethod.RESCALING, scalingFactors);
    assertEquals(forwardLikelihood, backwardLikelihood, EPSILON);
    // also make sure that the likelihood matches the expected one
    assertEquals(1.8425e-4, forwardLikelihood, EPSILON);
  }
    
}
