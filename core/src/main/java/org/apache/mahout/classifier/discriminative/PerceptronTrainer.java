/* Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.classifier.discriminative;

import org.apache.mahout.math.Vector;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Implements training according to the perceptron update rule.
 */
public class PerceptronTrainer extends LinearTrainer {
  
  private static final Logger log = LoggerFactory.getLogger(PerceptronTrainer.class);

  /** Rate the model is to be updated with at each step. */
  private final double learningRate;
  
  public PerceptronTrainer(int dimension, double threshold,
                           double learningRate, double init, double initBias) {
    super(dimension, threshold, init, initBias);
    this.learningRate = learningRate;
  }
  
  /**
   * {@inheritDoc} Perceptron update works such that in case the predicted label
   * does not match the real label, the weight vector is updated as follows: In
   * case the prediction was positive but should have been negative, the weight vector
   * is set to the sum of weight vector and example (multiplied by the learning rate).
   * 
   * In case the prediction was negative but should have been positive, the example
   * vector (multiplied by the learning rate) is subtracted from the weight vector.
   */
  @Override
  protected void update(double label, Vector dataPoint, LinearModel model) {
    double factor = 1.0;
    if (label == 0.0) {
      factor = -1.0;
    }
    
    Vector updateVector = dataPoint.times(factor).times(this.learningRate);
    log.debug("Updatevec: {}", updateVector);
    
    model.addDelta(updateVector);
    model.shiftBias(factor * this.learningRate);
    log.debug("{}", model);
  }
}
