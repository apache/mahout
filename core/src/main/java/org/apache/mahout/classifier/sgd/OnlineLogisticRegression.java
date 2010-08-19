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

package org.apache.mahout.classifier.sgd;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.BinaryFunction;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.UnaryFunction;

/**
 * Extends the basic on-line logistic regression learner with a specific set of learning
 * rate annealing schedules.
 */
public class OnlineLogisticRegression extends AbstractOnlineLogisticRegression {
  // these next two control decayFactor^steps exponential type of annealing
  // learning rate and decay factor
  private double mu0 = 1;
  private double decayFactor = 1 - 1e-3;

  // these next two control 1/steps^forget type annealing
  private int stepOffset = 10;
  // -1 equals even weighting of all examples, 0 means only use exponential annealing
  private double forgettingExponent = -0.5;

  // controls how per term annealing works
  private int perTermAnnealingOffset = 20;

  private OnlineLogisticRegression() {
    // private constructor available for Gson, but not normal use
  }

  public OnlineLogisticRegression(int numCategories, int numFeatures, PriorFunction prior) {
    this.numCategories = numCategories;
    this.prior = prior;

    updateSteps = new DenseVector(numFeatures);
    updateCounts = new DenseVector(numFeatures).assign(perTermAnnealingOffset);
    beta = new DenseMatrix(numCategories - 1, numFeatures);
  }

  /**
   * Chainable configuration option.
   *
   * @param alpha New value of decayFactor, the exponential decay rate for the learning rate.
   * @return This, so other configurations can be chained.
   */
  public OnlineLogisticRegression alpha(double alpha) {
    this.decayFactor = alpha;
    return this;
  }

  public OnlineLogisticRegression lambda(double lambda) {
    // we only over-ride this to provide a more restrictive return type
    super.lambda(lambda);
    return this;
  }

  /**
   * Chainable configuration option.
   *
   * @param learningRate New value of initial learning rate.
   * @return This, so other configurations can be chained.
   */
  public OnlineLogisticRegression learningRate(double learningRate) {
    this.mu0 = learningRate;
    return this;
  }

  public OnlineLogisticRegression stepOffset(int stepOffset) {
    this.stepOffset = stepOffset;
    return this;
  }

  public OnlineLogisticRegression decayExponent(double decayExponent) {
    if (decayExponent > 0) {
      decayExponent = -decayExponent;
    }
    this.forgettingExponent = decayExponent;
    return this;
  }


  @Override
  public double perTermLearningRate(int j) {
    return Math.sqrt(perTermAnnealingOffset / updateCounts.get(j));
  }

  @Override
  public double currentLearningRate() {
    return mu0 * Math.pow(decayFactor, getStep()) * Math.pow(getStep() + stepOffset, forgettingExponent);
  }

  public void copyFrom(OnlineLogisticRegression other) {
    super.copyFrom(other);
    mu0 = other.mu0;
    decayFactor = other.decayFactor;

    stepOffset = other.stepOffset;
    forgettingExponent = other.forgettingExponent;

    perTermAnnealingOffset = other.perTermAnnealingOffset;
  }

  public OnlineLogisticRegression copy() {
    close();
    OnlineLogisticRegression r = new OnlineLogisticRegression(numCategories(), numFeatures(), prior);
    r.copyFrom(this);
    return r;
  }

}
