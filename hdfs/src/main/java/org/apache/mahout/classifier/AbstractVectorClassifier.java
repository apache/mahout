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
 * Defines the interface for classifiers that take a vector as input. This is
 * implemented as an abstract class so that it can implement a number of handy
 * convenience methods related to classification of vectors.
 *
 * <p>
 * A classifier takes an input vector and calculates the scores (usually
 * probabilities) that the input vector belongs to one of {@code n}
 * categories. In {@code AbstractVectorClassifier} each category is denoted
 * by an integer {@code c} between {@code 0} and {@code n-1}
 * (inclusive).
 *
 * <p>
 * New users should start by looking at {@link #classifyFull} (not {@link #classify}).
 *
 */
public abstract class AbstractVectorClassifier {

  /** Minimum allowable log likelihood value. */
  public static final double MIN_LOG_LIKELIHOOD = -100.0;

   /**
    * Returns the number of categories that a target variable can be assigned to.
    * A vector classifier will encode it's output as an integer from
    * {@code 0} to {@code numCategories()-1} (inclusive).
    *
    * @return The number of categories.
    */
  public abstract int numCategories();

  /**
   * Compute and return a vector containing {@code n-1} scores, where
   * {@code n} is equal to {@code numCategories()}, given an input
   * vector {@code instance}. Higher scores indicate that the input vector
   * is more likely to belong to that category. The categories are denoted by
   * the integers {@code 0} through {@code n-1} (inclusive), and the
   * scores in the returned vector correspond to categories 1 through
   * {@code n-1} (leaving out category 0). It is assumed that the score for
   * category 0 is one minus the sum of the scores in the returned vector.
   *
   * @param instance  A feature vector to be classified.
   * @return A vector of probabilities in 1 of {@code n-1} encoding.
   */
  public abstract Vector classify(Vector instance);
  
  /**
   * Compute and return a vector of scores before applying the inverse link
   * function. For logistic regression and other generalized linear models, this
   * is just the linear part of the classification.
   * 
   * <p>
   * The implementation of this method provided by {@code AbstractVectorClassifier} throws an
   * {@link UnsupportedOperationException}. Your subclass must explicitly override this method to support
   * this operation.
   * 
   * @param features  A feature vector to be classified.
   * @return A vector of scores. If transformed by the link function, these will become probabilities.
   */
  public Vector classifyNoLink(Vector features) {
    throw new UnsupportedOperationException(this.getClass().getName()
        + " doesn't support classification without a link");
  }

  /**
   * Classifies a vector in the special case of a binary classifier where
   * {@link #classify(Vector)} would return a vector with only one element. As
   * such, using this method can avoid the allocation of a vector.
   * 
   * @param instance The feature vector to be classified.
   * @return The score for category 1.
   * 
   * @see #classify(Vector)
   */
  public abstract double classifyScalar(Vector instance);

  /**
   * Computes and returns a vector containing {@code n} scores, where
   * {@code n} is {@code numCategories()}, given an input vector
   * {@code instance}. Higher scores indicate that the input vector is more
   * likely to belong to the corresponding category. The categories are denoted
   * by the integers {@code 0} through {@code n-1} (inclusive).
   *
   * <p>
   * Using this method it is possible to classify an input vector, for example,
   * by selecting the category with the largest score. If
   * {@code classifier} is an instance of
   * {@code AbstractVectorClassifier} and {@code input} is a
   * {@code Vector} of features describing an element to be classified,
   * then the following code could be used to classify {@code input}.<br>
   * {@code
   * Vector scores = classifier.classifyFull(input);<br>
   * int assignedCategory = scores.maxValueIndex();<br>
   * } Here {@code assignedCategory} is the index of the category
   * with the maximum score.
   *
   * <p>
   * If an {@code n-1} encoding is acceptable, and allocation performance
   * is an issue, then the {@link #classify(Vector)} method is probably better
   * to use.
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
   * Computes and returns a vector containing {@code n} scores, where
   * {@code n} is {@code numCategories()}, given an input vector
   * {@code instance}. Higher scores indicate that the input vector is more
   * likely to belong to the corresponding category. The categories are denoted
   * by the integers {@code 0} through {@code n-1} (inclusive). The
   * main difference between this method and {@link #classifyFull(Vector)} is
   * that this method allows a user to provide a previously allocated
   * {@code Vector r} to store the returned scores.
   *
   * <p>
   * Using this method it is possible to classify an input vector, for example,
   * by selecting the category with the largest score. If
   * {@code classifier} is an instance of
   * {@code AbstractVectorClassifier}, {@code result} is a non-null
   * {@code Vector}, and {@code input} is a {@code Vector} of
   * features describing an element to be classified, then the following code
   * could be used to classify {@code input}.<br>
   * {@code
   * Vector scores = classifier.classifyFull(result, input); // Notice that scores == result<br>
   * int assignedCategory = scores.maxValueIndex();<br>
   * } Here {@code assignedCategory} is the index of the category
   * with the maximum score.
   *
   * @param r Where to put the results.
   * @param instance  A vector of features to be classified.
   * @return A vector of scores/probabilities, one for each category.
   */
  public Vector classifyFull(Vector r, Vector instance) {
    r.viewPart(1, numCategories() - 1).assign(classify(instance));
    r.setQuick(0, 1.0 - r.zSum());
    return r;
  }


  /**
   * Returns n-1 probabilities, one for each categories 1 through
   * {@code n-1}, for each row of a matrix, where {@code n} is equal
   * to {@code numCategories()}. The probability of the missing 0-th
   * category is 1 - rowSum(this result).
   *
   * @param data  The matrix whose rows are the input vectors to classify
   * @return A matrix of scores, one row per row of the input matrix, one column for each but the last category.
   */
  public Matrix classify(Matrix data) {
    Matrix r = new DenseMatrix(data.numRows(), numCategories() - 1);
    for (int row = 0; row < data.numRows(); row++) {
      r.assignRow(row, classify(data.viewRow(row)));
    }
    return r;
  }

  /**
   * Returns a matrix where the rows of the matrix each contain {@code n} probabilities, one for each category.
   *
   * @param data  The matrix whose rows are the input vectors to classify
   * @return A matrix of scores, one row per row of the input matrix, one column for each but the last category.
   */
  public Matrix classifyFull(Matrix data) {
    Matrix r = new DenseMatrix(data.numRows(), numCategories());
    for (int row = 0; row < data.numRows(); row++) {
      classifyFull(r.viewRow(row), data.viewRow(row));
    }
    return r;
  }

  /**
   * Returns a vector of probabilities of category 1, one for each row
   * of a matrix. This only makes sense if there are exactly two categories, but
   * calling this method in that case can save a number of vector allocations.
   * 
   * @param data  The matrix whose rows are vectors to classify
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
   * Returns a measure of how good the classification for a particular example
   * actually is.
   * 
   * @param actual  The correct category for the example.
   * @param data  The vector to be classified.
   * @return The log likelihood of the correct answer as estimated by the current model. This will always be <= 0
   *  and larger (closer to 0) indicates better accuracy. In order to simplify code that maintains eunning averages,
   *  we bound this value at -100.
   */
  public double logLikelihood(int actual, Vector data) {
    if (numCategories() == 2) {
      double p = classifyScalar(data);
      if (actual > 0) {
        return Math.max(MIN_LOG_LIKELIHOOD, Math.log(p));
      } else {
        return Math.max(MIN_LOG_LIKELIHOOD, Math.log1p(-p));
      }
    } else {
      Vector p = classify(data);
      if (actual > 0) {
        return Math.max(MIN_LOG_LIKELIHOOD, Math.log(p.get(actual - 1)));
      } else {
        return Math.max(MIN_LOG_LIKELIHOOD, Math.log1p(-p.zSum()));
      }
    }
  }
}
