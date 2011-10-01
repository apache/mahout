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

package org.apache.mahout.classifier;

import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.Vector;

import com.google.common.base.Preconditions;

/**
 * Defines the interface for classifiers that take input as a vector.  This is implemented
 * as an abstract class so that it can implement a number of handy convenience methods
 * related to classification of vectors.
 */
public abstract class AbstractVectorClassifier {
  // ------ These are all that are necessary to define a vector classifier.

  /**
   * Returns the number of categories for the target variable.  A vector classifier
   * will encode it's output using a zero-based 1 of numCategories encoding.
   * @return The number of categories.
   */
  public abstract int numCategories();

  /**
   * Classify a vector returning a vector of numCategories-1 scores.  It is assumed that
   * the score for the missing category is one minus the sum of the scores that are returned.
   *
   * Note that the missing score is the 0-th score.
   * @param instance  A feature vector to be classified.
   * @return  A vector of probabilities in 1 of n-1 encoding.
   */
  public abstract Vector classify(Vector instance);

  /**
   * Classify a vector, but don't apply the inverse link function.  For logistic regression
   * and other generalized linear models, this is just the linear part of the classification.
   * @param features  A feature vector to be classified.
   * @return  A vector of scores.  If transformed by the link function, these will become probabilities.
   */
  public Vector classifyNoLink(Vector features) {
    throw new UnsupportedOperationException(
        this.getClass().getName() + " doesn't support classification without a link");
  }

  /**
   * Classifies a vector in the special case of a binary classifier where
   * {@link #classify(Vector)} would return a vector with only one element.  As such,
   * using this method can void the allocation of a vector.
   * @param instance   The feature vector to be classified.
   * @return The score for category 1.
   *
   * @see #classify(Vector)
   */
  public abstract double classifyScalar(Vector instance);

  // ------- From here on, we have convenience methods that provide an easier API to use.

  /**
   * Returns n probabilities, one for each category.  If you can use an n-1 coding, and are touchy
   * about allocation performance, then the classify method is probably better to use.  The 0-th
   * element of the score vector returned by this method is the missing score as computed by the
   * classify method.
   *
   * @see #classify(Vector)
   * @see #classifyFull(Vector r, Vector instance)
   *
   * @param instance A vector of features to be classified.
   * @return A vector of probabilities, one for each category.
   */
  public Vector classifyFull(Vector instance) {
    return classifyFull(new DenseVector(numCategories()), instance);
  }

  /**
   * Returns n probabilities, one for each category into a pre-allocated vector.  One
   * vector allocation is still done in the process of multiplying by the coefficient
   * matrix, but that is hard to avoid.  The cost of such an ephemeral allocation is
   * very small in any case compared to the multiplication itself.
   *
   * @param r        Where to put the results.
   * @param instance A vector of features to be classified.
   * @return A vector of probabilities, one for each category.
   */
  public Vector classifyFull(Vector r, Vector instance) {
    r.viewPart(1, numCategories() - 1).assign(classify(instance));
    r.setQuick(0, 1.0 - r.zSum());
    return r;
  }


  /**
   * Returns n-1 probabilities, one for each category but the last, for each row of a matrix. The
   * probability of the missing 0-th category is 1 - rowSum(this result).
   *
   * @param data The matrix whose rows are vectors to classify
   * @return A matrix of scores, one row per row of the input matrix, one column for each but the
   *         last category.
   */
  public Matrix classify(Matrix data) {
    Matrix r = new DenseMatrix(data.numRows(), numCategories() - 1);
    for (int row = 0; row < data.numRows(); row++) {
      r.assignRow(row, classify(data.viewRow(row)));
    }
    return r;
  }

  /**
   * Returns n probabilities, one for each category, for each row of a matrix.
   *
   * @param data The matrix whose rows are vectors to classify
   * @return A matrix of scores, one row per row of the input matrix, one column for each but the
   *         last category.
   */
  public Matrix classifyFull(Matrix data) {
    Matrix r = new DenseMatrix(data.numRows(), numCategories());
    for (int row = 0; row < data.numRows(); row++) {
      classifyFull(r.viewRow(row), data.viewRow(row));
    }
    return r;
  }

  /**
   * Returns a vector of probabilities of the first category, one for each row of a matrix. This
   * only makes sense if there are exactly two categories, but calling this method in that case can
   * save a number of vector allocations.
   *
   * @param data The matrix whose rows are vectors to classify
   * @return A vector of scores, with one value per row of the input matrix.
   */
  public Vector classifyScalar(Matrix data) {
    Preconditions.checkArgument(numCategories() == 2, "Can only call classifyScalar with two categories");

    Vector r = new DenseVector(data.numRows());
    for (int row = 0; row < data.numRows(); row++) {
      r.set(row, classifyScalar(data.viewRow(row)));
    }
    return r;
  }

  /**
   * Returns a measure of how good the classification for a particular example actually is.
   *
   * @param actual The correct category for the example.
   * @param data   The vector to be classified.
   * @return The log likelihood of the correct answer as estimated by the current model.  This will
   *         always be <= 0 and larger (closer to 0) indicates better accuracy.  In order to simplify
   *         code that maintains running averages, we bound this value at -100.
   */
  public double logLikelihood(int actual, Vector data) {
    if (numCategories() == 2) {
      double p = classifyScalar(data);
      if (actual > 0) {
        return Math.max(-100.0, Math.log(p));
      } else {
        return Math.max(-100.0, Math.log1p(-p));
      }
    } else {
      Vector p = classify(data);
      if (actual > 0) {
        return Math.max(-100.0, Math.log(p.get(actual - 1)));
      } else {
        return Math.max(-100.0, Math.log1p(-p.zSum()));
      }
    }
  }
}
