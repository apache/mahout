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

import org.apache.hadoop.io.Writable;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Extends the basic on-line logistic regression learner with a specific set of learning
 * rate annealing schedules.
 */
public class OnlineLogisticRegression extends AbstractOnlineLogisticRegression implements Writable {
  public static final int WRITABLE_VERSION = 1;

  // these next two control decayFactor^steps exponential type of annealing
  // learning rate and decay factor
  private double mu0 = 1;
  private double decayFactor = 1 - 1.0e-3;

  // these next two control 1/steps^forget type annealing
  private int stepOffset = 10;
  // -1 equals even weighting of all examples, 0 means only use exponential annealing
  private double forgettingExponent = -0.5;

  // controls how per term annealing works
  private int perTermAnnealingOffset = 20;

  public OnlineLogisticRegression() {
    // private constructor available for serialization, but not normal use
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

  @Override
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

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(WRITABLE_VERSION);
    out.writeDouble(mu0);
    out.writeDouble(getLambda()); 
    out.writeDouble(decayFactor);
    out.writeInt(stepOffset);
    out.writeInt(step);
    out.writeDouble(forgettingExponent);
    out.writeInt(perTermAnnealingOffset);
    out.writeInt(numCategories);
    MatrixWritable.writeMatrix(out, beta);
    PolymorphicWritable.write(out, prior);
    VectorWritable.writeVector(out, updateCounts);
    VectorWritable.writeVector(out, updateSteps);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int version = in.readInt();
    if (version == WRITABLE_VERSION) {
      mu0 = in.readDouble();
      lambda(in.readDouble()); 
      decayFactor = in.readDouble();
      stepOffset = in.readInt();
      step = in.readInt();
      forgettingExponent = in.readDouble();
      perTermAnnealingOffset = in.readInt();
      numCategories = in.readInt();
      beta = MatrixWritable.readMatrix(in);
      prior = PolymorphicWritable.read(in, PriorFunction.class);

      updateCounts = VectorWritable.readVector(in);
      updateSteps = VectorWritable.readVector(in);
    } else {
      throw new IOException("Incorrect object version, wanted " + WRITABLE_VERSION + " got " + version);
    }
  }
}
