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
import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * Online passive aggressive learner that tries to minimize the label ranking hinge loss.
 * Implements a multi-class linear classifier minimizing rank loss.
 *  based on "Online passive aggressive algorithms" by Cramer et al, 2006.
 *  Note: Its better to use classifyNoLink because the loss function is based
 *  on ensuring that the score of the good label is larger than the next
 *  highest label by some margin. The conversion to probability is just done
 *  by exponentiating and dividing by the sum and is empirical at best.
 *  Your features should be pre-normalized in some sensible range, for example,
 *  by subtracting the mean and standard deviation, if they are very
 *  different in magnitude from each other.
 */
public class PassiveAggressive extends AbstractVectorClassifier implements OnlineLearner, Writable {

  private static final Logger log = LoggerFactory.getLogger(PassiveAggressive.class);

  public static final int WRITABLE_VERSION = 1;

  // the learning rate of the algorithm
  private double learningRate = 0.1;

  // loss statistics.
  private int lossCount = 0;
  private double lossSum = 0;

  // coefficients for the classification.  This is a dense matrix
  // that is (numCategories ) x numFeatures
  private Matrix weights;

  // number of categories we are classifying.
  private int numCategories;

  public PassiveAggressive(int numCategories, int numFeatures) {
    this.numCategories = numCategories;
    weights = new DenseMatrix(numCategories, numFeatures);
    weights.assign(0.0);
  }

  /**
   * Chainable configuration option.
   *
   * @param learningRate New value of initial learning rate.
   * @return This, so other configurations can be chained.
   */
  public PassiveAggressive learningRate(double learningRate) {
    this.learningRate = learningRate;
    return this;
  }

  public void copyFrom(PassiveAggressive other) {
    learningRate = other.learningRate;
    numCategories = other.numCategories;
    weights = other.weights;
  }

  @Override
  public int numCategories() {
    return numCategories;
  }

  @Override
  public Vector classify(Vector instance) {
    Vector result = classifyNoLink(instance);
    // Convert to probabilities by exponentiation.
    double max = result.maxValue();
    result.assign(Functions.minus(max)).assign(Functions.EXP);
    result = result.divide(result.norm(1));

    return result.viewPart(1, result.size() - 1);
  }

  @Override
  public Vector classifyNoLink(Vector instance) {
    Vector result = new DenseVector(weights.numRows());
    result.assign(0);
    for (int i = 0; i < weights.numRows(); i++) {
      result.setQuick(i, weights.viewRow(i).dot(instance));
    }
    return result;
  }

  @Override
  public double classifyScalar(Vector instance) {
    double v1 = weights.viewRow(0).dot(instance);
    double v2 = weights.viewRow(1).dot(instance);
    v1 = Math.exp(v1);
    v2 = Math.exp(v2);
    return v2 / (v1 + v2);
  }

  public int numFeatures() {
    return weights.numCols();
  }

  public PassiveAggressive copy() {
    close();
    PassiveAggressive r = new PassiveAggressive(numCategories(), numFeatures());
    r.copyFrom(this);
    return r;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(WRITABLE_VERSION);
    out.writeDouble(learningRate);
    out.writeInt(numCategories);
    MatrixWritable.writeMatrix(out, weights);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    int version = in.readInt();
    if (version == WRITABLE_VERSION) {
      learningRate = in.readDouble();
      numCategories = in.readInt();
      weights = MatrixWritable.readMatrix(in);
    } else {
      throw new IOException("Incorrect object version, wanted " + WRITABLE_VERSION + " got " + version);
    }
  }

  @Override
  public void close() {
      // This is an online classifier, nothing to do.
  }

  @Override
  public void train(long trackingKey, String groupKey, int actual, Vector instance) {
    if (lossCount > 1000) {
      log.info("Avg. Loss = {}", lossSum / lossCount);
      lossCount = 0;
      lossSum = 0;
    }
    Vector result = classifyNoLink(instance);
    double myScore = result.get(actual);
    // Find the highest score that is not actual.
    int otherIndex = result.maxValueIndex();
    double otherValue = result.get(otherIndex);
    if (otherIndex == actual) {
      result.setQuick(otherIndex, Double.NEGATIVE_INFINITY);
      otherIndex = result.maxValueIndex();
      otherValue = result.get(otherIndex);
    }
    double loss = 1.0 - myScore + otherValue;
    lossCount += 1;
    if (loss >= 0) {
      lossSum += loss;
      double tau = loss / (instance.dot(instance) + 0.5 / learningRate);
      Vector delta = instance.clone();
      delta.assign(Functions.mult(tau));
      weights.viewRow(actual).assign(delta, Functions.PLUS);
//      delta.addTo(weights.viewRow(actual));
      delta.assign(Functions.mult(-1));
      weights.viewRow(otherIndex).assign(delta, Functions.PLUS);
//      delta.addTo(weights.viewRow(otherIndex));
    }
  }

  @Override
  public void train(long trackingKey, int actual, Vector instance) {
    train(trackingKey, null, actual, instance);
  }

  @Override
  public void train(int actual, Vector instance) {
    train(0, null, actual, instance);
  }

}
