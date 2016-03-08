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
import org.apache.mahout.math.Vector.Element;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

import com.google.common.base.Preconditions;

/**
 * Generic definition of a 1 of n logistic regression classifier that returns probabilities in
 * response to a feature vector.  This classifier uses 1 of n-1 coding where the 0-th category
 * is not stored explicitly.
 * <p/>
 * Provides the SGD based algorithm for learning a logistic regression, but omits all
 * annealing of learning rates.  Any extension of this abstract class must define the overall
 * and per-term annealing for themselves.
 */
public abstract class AbstractOnlineLogisticRegression extends AbstractVectorClassifier implements OnlineLearner {
  // coefficients for the classification.  This is a dense matrix
  // that is (numCategories-1) x numFeatures
  protected Matrix beta;

  // number of categories we are classifying.  This should the number of rows of beta plus one.
  protected int numCategories;

  protected int step;

  // information about how long since coefficient rows were updated.  This allows lazy regularization.
  protected Vector updateSteps;

  // information about how many updates we have had on a location.  This allows per-term
  // annealing a la confidence weighted learning.
  protected Vector updateCounts;

  // weight of the prior on beta
  private double lambda = 1.0e-5;
  protected PriorFunction prior;

  // can we ignore any further regularization when doing classification?
  private boolean sealed;

  // by default we don't do any fancy training
  private Gradient gradient = new DefaultGradient();

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

  /**
   * Computes the inverse link function, by default the logistic link function.
   *
   * @param v The output of the linear combination in a GLM.  Note that the value
   *          of v is disturbed.
   * @return A version of v with the link function applied.
   */
  public static Vector link(Vector v) {
    double max = v.maxValue();
    if (max >= 40) {
      // if max > 40, we subtract the large offset first
      // the size of the max means that 1+sum(exp(v)) = sum(exp(v)) to within round-off
      v.assign(Functions.minus(max)).assign(Functions.EXP);
      return v.divide(v.norm(1));
    } else {
      v.assign(Functions.EXP);
      return v.divide(1 + v.norm(1));
    }
  }

  /**
   * Computes the binomial logistic inverse link function.
   *
   * @param r The value to transform.
   * @return The logit of r.
   */
  public static double link(double r) {
    if (r < 0.0) {
      double s = Math.exp(r);
      return s / (1.0 + s);
    } else {
      double s = Math.exp(-r);
      return 1.0 / (1.0 + s);
    }
  }

  @Override
  public Vector classifyNoLink(Vector instance) {
    // apply pending regularization to whichever coefficients matter
    regularize(instance);
    return beta.times(instance);
  }

  public double classifyScalarNoLink(Vector instance) {
    return beta.viewRow(0).dot(instance);
  }

  /**
   * Returns n-1 probabilities, one for each category but the 0-th.  The probability of the 0-th
   * category is 1 - sum(this result).
   *
   * @param instance A vector of features to be classified.
   * @return A vector of probabilities, one for each of the first n-1 categories.
   */
  @Override
  public Vector classify(Vector instance) {
    return link(classifyNoLink(instance));
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
  @Override
  public double classifyScalar(Vector instance) {
    Preconditions.checkArgument(numCategories() == 2, "Can only call classifyScalar with two categories");

    // apply pending regularization to whichever coefficients matter
    regularize(instance);

    // result is a vector with one element so we can just use dot product
    return link(classifyScalarNoLink(instance));
  }

  @Override
  public void train(long trackingKey, String groupKey, int actual, Vector instance) {
    unseal();

    double learningRate = currentLearningRate();

    // push coefficients back to zero based on the prior
    regularize(instance);

    // update each row of coefficients according to result
    Vector gradient = this.gradient.apply(groupKey, actual, instance, this);
    for (int i = 0; i < numCategories - 1; i++) {
      double gradientBase = gradient.get(i);

      // then we apply the gradientBase to the resulting element.
      for (Element updateLocation : instance.nonZeroes()) {
        int j = updateLocation.index();

        double newValue = beta.getQuick(i, j) + gradientBase * learningRate * perTermLearningRate(j) * instance.get(j);
        beta.setQuick(i, j, newValue);
      }
    }

    // remember that these elements got updated
    for (Element element : instance.nonZeroes()) {
      int j = element.index();
      updateSteps.setQuick(j, getStep());
      updateCounts.incrementQuick(j, 1);
    }
    nextStep();

  }

  @Override
  public void train(long trackingKey, int actual, Vector instance) {
    train(trackingKey, null, actual, instance);
  }

  @Override
  public void train(int actual, Vector instance) {
    train(0, null, actual, instance);
  }

  public void regularize(Vector instance) {
    if (updateSteps == null || isSealed()) {
      return;
    }

    // anneal learning rate
    double learningRate = currentLearningRate();

    // here we lazily apply the prior to make up for our neglect
    for (int i = 0; i < numCategories - 1; i++) {
      for (Element updateLocation : instance.nonZeroes()) {
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

  public void setGradient(Gradient gradient) {
    this.gradient = gradient;
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

  @Override
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

  @Override
  public void close() {
    if (!sealed) {
      step++;
      regularizeAll();
      sealed = true;
    }
  }

  public void copyFrom(AbstractOnlineLogisticRegression other) {
    // number of categories we are classifying.  This should the number of rows of beta plus one.
    Preconditions.checkArgument(numCategories == other.numCategories,
            "Can't copy unless number of target categories is the same");

    beta.assign(other.beta);

    step = other.step;

    updateSteps.assign(other.updateSteps);
    updateCounts.assign(other.updateCounts);
  }

  public boolean validModel() {
    double k = beta.aggregate(Functions.PLUS, new DoubleFunction() {
      @Override
      public double apply(double v) {
        return Double.isNaN(v) || Double.isInfinite(v) ? 1 : 0;
      }
    });
    return k < 1;
  }

}
