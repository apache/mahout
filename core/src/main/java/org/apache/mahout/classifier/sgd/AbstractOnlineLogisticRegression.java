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

import org.apache.mahout.classifier.AbstractVectorClassifier;
import org.apache.mahout.classifier.OnlineLearner;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.UnaryFunction;

import java.util.Iterator;

/**
 * Generic definition of a 1 of n logistic regression classifier that returns probabilities in
 * response to a feature vector.  This classifier uses 1 of n-1 coding where the 0-th category
 * is not stored explicitly.
 * <p/>
 * TODO: implement symbolic input with string, overall cooccurrence and n-gram hash encoding
 * TODO: implement reporter system to monitor progress
 *
 * Provides the based SGD based algorithm for learning a logistic regression, but omits all
 * annealing of learning rates.  Any extension of this abstract class must define the overall
 * and per-term annealing for themselves.
 */
public abstract class AbstractOnlineLogisticRegression extends AbstractVectorClassifier implements OnlineLearner {
  // coefficients for the classification.  This is a dense matrix
  // that is (numCategories-1) x numFeatures
  protected Matrix beta;

  // number of categories we are classifying.  This should the number of rows of beta plus one.
  protected int numCategories;

  private int step = 0;

  // information about how long since coefficient rows were updated.  This allows lazy regularization.
  protected Vector updateSteps;

  // information about how many updates we have had on a location.  This allows per-term
  // annealing a la confidence weighted learning.
  protected Vector updateCounts;

  // weight of the prior on beta
  private double lambda = 1e-5;
  protected PriorFunction prior;

  // can we ignore any further regularization when doing classification?
  private boolean sealed = false;

  /**
   * Chainable configuration option.
   *
   * @param lambda New value of lambda, the weighting factor for the prior distribution.
   * @return This, so other configurations can be chained.
   */
  public AbstractOnlineLogisticRegression lambda(double lambda) {
    this.lambda = lambda;
    return this;
  }

  private Vector logisticLink(Vector v) {
    double max = v.maxValue();
    if (max < 40) {
      v.assign(Functions.EXP);
      double sum = 1 + v.norm(1);
      return v.divide(sum);
    } else {
      v.assign(Functions.minus(max)).assign(Functions.EXP);
      return v;
    }
  }

  /**
   * Returns n-1 probabilities, one for each category but the 0-th.  The probability of the 0-th
   * category is 1 - sum(this result).
   *
   * @param instance A vector of features to be classified.
   * @return A vector of probabilities, one for each of the first n-1 categories.
   */
  public Vector classify(Vector instance) {
    // apply pending regularization to whichever coefficients matter
    regularize(instance);

    Vector v = beta.times(instance);
    return logisticLink(v);
  }

  /**
   * Returns a single scalar probability in the case where we have two categories.  Using this
   * method avoids an extra vector allocation as opposed to calling classify() or an extra two
   * vector allocations relative to classifyFull().
   *
   * @param instance The vector of features to be classified.
   * @return The probability of the first of two categories.
   * @throws IllegalArgumentException If the classifier doesn't have two categories.
   */
  public double classifyScalar(Vector instance) {
    if (numCategories() != 2) {
      throw new IllegalArgumentException("Can only call classifyScalar with two categories");
    }

    // apply pending regularization to whichever coefficients matter
    regularize(instance);

    // result is a vector with one element so we can just use dot product
    double r = Math.exp(beta.getRow(0).dot(instance));
    return r / (1 + r);
  }

  @Override
  public void train(long trackingKey, int actual, Vector instance) {
    train(actual, instance);
  }

  public void train(int actual, Vector instance) {
    unseal();

    double learningRate = currentLearningRate();

    // push coefficients back to zero based on the prior
    regularize(instance);

    // what does the current model say?
    Vector v = classify(instance);

    // update each row of coefficients according to result
    for (int i = 0; i < numCategories - 1; i++) {
      double gradientBase = -v.getQuick(i);
      // the use of i+1 instead of i here is what makes the 0-th category be the one without coefficients
      if ((i + 1) == actual) {
        gradientBase += 1;
      }

      // then we apply the gradientBase to the resulting element.
      Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
      while (nonZeros.hasNext()) {
        Vector.Element updateLocation = nonZeros.next();
        int j = updateLocation.index();

        double newValue = beta.get(i, j) + learningRate * gradientBase * instance.get(j) * perTermLearningRate(j);
        beta.set(i, j, newValue);
      }
    }

    // remember that these elements got updated
    Iterator<Vector.Element> i = instance.iterateNonZero();
    while (i.hasNext()) {
      Vector.Element element = i.next();
      int j = element.index();
      updateSteps.setQuick(j, getStep());
      updateCounts.setQuick(j, updateCounts.getQuick(j) + 1);
    }
    nextStep();

  }

  public void regularize(Vector instance) {
    if (updateSteps == null || isSealed()) {
      return;
    }

    // anneal learning rate
    double learningRate = currentLearningRate();

    // here we lazily apply the prior to make up for our neglect
    for (int i = 0; i < numCategories - 1; i++) {
      Iterator<Vector.Element> nonZeros = instance.iterateNonZero();
      while (nonZeros.hasNext()) {
        Vector.Element updateLocation = nonZeros.next();
        int j = updateLocation.index();
        double missingUpdates = getStep() - updateSteps.get(j);
        if (missingUpdates > 0) {
          double rate = getLambda() * learningRate * perTermLearningRate(j);
          double newValue = prior.age(beta.get(i, j), missingUpdates, rate);
          beta.set(i, j, newValue);
          updateSteps.set(j, getStep());
        }
      }
    }
  }

  // these two abstract methods are how extensions can modify the basic learning behavior of this object.

  public abstract double perTermLearningRate(int j);

  public abstract double currentLearningRate();

  public void setPrior(PriorFunction prior) {
    this.prior = prior;
  }

  public PriorFunction getPrior() {
    return prior;
  }

  public Matrix getBeta() {
    close();
    return beta;
  }

  public void setBeta(int i, int j, double betaIJ) {
    beta.set(i, j, betaIJ);
  }

  public int numCategories() {
    return numCategories;
  }

  public int numFeatures() {
    return beta.numCols();
  }

  public double getLambda() {
    return lambda;
  }

  public int getStep() {
    return step;
  }

  protected void nextStep() {
    step++;
  }

  public boolean isSealed() {
    return sealed;
  }

  protected void unseal() {
    sealed = false;
  }

  private void regularizeAll() {
    Vector all = new DenseVector(beta.numCols());
    all.assign(1);
    regularize(all);
  }

  public void close() {
    if (!sealed) {
      step++;
      regularizeAll();
      sealed = true;
    }
  }

  public void copyFrom(AbstractOnlineLogisticRegression other) {
    // number of categories we are classifying.  This should the number of rows of beta plus one.
    if (numCategories != other.numCategories) {
      throw new IllegalArgumentException("Can't copy unless number of target categories is the same");
    }

    beta.assign(other.beta);

    step = other.step;

    updateSteps.assign(other.updateSteps);
    updateCounts.assign(other.updateCounts);
  }

  public boolean validModel() {
    double k = beta.aggregate(Functions.PLUS, new UnaryFunction() {
      @Override
      public double apply(double v) {
        return Double.isNaN(v) || Double.isInfinite(v) ? 1 : 0;
      }
    });
    return k < 1;
  }
}
