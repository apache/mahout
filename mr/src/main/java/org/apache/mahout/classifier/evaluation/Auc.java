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

package org.apache.mahout.classifier.evaluation;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.list.DoubleArrayList;

import com.google.common.base.Preconditions;

import java.util.Random;

/**
 * Computes AUC and a few other accuracy statistics without storing huge amounts of data.  This is
 * done by keeping uniform samples of the positive and negative scores.  Then, when AUC is to be
 * computed, the remaining scores are sorted and a rank-sum statistic is used to compute the AUC.
 * Since AUC is invariant with respect to down-sampling of either positives or negatives, this is
 * close to correct and is exactly correct if maxBufferSize or fewer positive and negative scores
 * are examined.
 */
public class Auc {

  private int maxBufferSize = 10000;
  private final DoubleArrayList[] scores = {new DoubleArrayList(), new DoubleArrayList()};
  private final Random rand;
  private int samples;
  private final double threshold;
  private final Matrix confusion;
  private final DenseMatrix entropy;

  private boolean probabilityScore = true;

  private boolean hasScore;

  /**
   * Allocates a new data-structure for accumulating information about AUC and a few other accuracy
   * measures.
   * @param threshold The threshold to use in computing the confusion matrix.
   */
  public Auc(double threshold) {
    confusion = new DenseMatrix(2, 2);
    entropy = new DenseMatrix(2, 2);
    this.rand = RandomUtils.getRandom();
    this.threshold = threshold;
  }

  public Auc() {
    this(0.5);
  }

  /**
   * Adds a score to the AUC buffers.
   *
   * @param trueValue Whether this score is for a true-positive or a true-negative example.
   * @param score     The score for this example.
   */
  public void add(int trueValue, double score) {
    Preconditions.checkArgument(trueValue == 0 || trueValue == 1, "True value must be 0 or 1");
    hasScore = true;

    int predictedClass = score > threshold ? 1 : 0;
    confusion.set(trueValue, predictedClass, confusion.get(trueValue, predictedClass) + 1);

    samples++;
    if (isProbabilityScore()) {
      double limited = Math.max(1.0e-20, Math.min(score, 1 - 1.0e-20));
      double v0 = entropy.get(trueValue, 0);
      entropy.set(trueValue, 0, (Math.log1p(-limited) - v0) / samples + v0);

      double v1 = entropy.get(trueValue, 1);
      entropy.set(trueValue, 1, (Math.log(limited) - v1) / samples + v1);
    }

    // add to buffers
    DoubleArrayList buf = scores[trueValue];
    if (buf.size() >= maxBufferSize) {
      // but if too many points are seen, we insert into a random
      // place and discard the predecessor.  The random place could
      // be anywhere, possibly not even in the buffer.
      // this is a special case of Knuth's permutation algorithm
      // but since we don't ever shuffle the first maxBufferSize
      // samples, the result isn't just a fair sample of the prefixes
      // of all permutations.  The CONTENTs of the result, however,
      // will be a fair and uniform sample of maxBufferSize elements
      // chosen from all elements without replacement
      int index = rand.nextInt(samples);
      if (index < buf.size()) {
        buf.set(index, score);
      }
    } else {
      // for small buffers, we collect all points without permuting
      // since we sort the data later, permuting now would just be
      // pedantic
      buf.add(score);
    }
  }

  public void add(int trueValue, int predictedClass) {
    hasScore = false;
    Preconditions.checkArgument(trueValue == 0 || trueValue == 1, "True value must be 0 or 1");
    confusion.set(trueValue, predictedClass, confusion.get(trueValue, predictedClass) + 1);
  }

  /**
   * Computes the AUC of points seen so far.  This can be moderately expensive since it requires
   * that all points that have been retained be sorted.
   *
   * @return The value of the Area Under the receiver operating Curve.
   */
  public double auc() {
    Preconditions.checkArgument(hasScore, "Can't compute AUC for classifier without a score");
    scores[0].sort();
    scores[1].sort();

    double n0 = scores[0].size();
    double n1 = scores[1].size();

    if (n0 == 0 || n1 == 0) {
      return 0.5;
    }

    // scan the data
    int i0 = 0;
    int i1 = 0;
    int rank = 1;
    double rankSum = 0;
    while (i0 < n0 && i1 < n1) {

      double v0 = scores[0].get(i0);
      double v1 = scores[1].get(i1);

      if (v0 < v1) {
        i0++;
        rank++;
      } else if (v1 < v0) {
        i1++;
        rankSum += rank;
        rank++;
      } else {
        // ties have to be handled delicately
        double tieScore = v0;

        // how many negatives are tied?
        int k0 = 0;
        while (i0 < n0 && scores[0].get(i0) == tieScore) {
          k0++;
          i0++;
        }

        // and how many positives
        int k1 = 0;
        while (i1 < n1 && scores[1].get(i1) == tieScore) {
          k1++;
          i1++;
        }

        // we found k0 + k1 tied values which have
        // ranks in the half open interval [rank, rank + k0 + k1)
        // the average rank is assigned to all
        rankSum += (rank + (k0 + k1 - 1) / 2.0) * k1;
        rank += k0 + k1;
      }
    }

    if (i1 < n1) {
      rankSum += (rank + (n1 - i1 - 1) / 2.0) * (n1 - i1);
      rank += (int) (n1 - i1);
    }

    return (rankSum / n1 - (n1 + 1) / 2) / n0;
  }

  /**
   * Returns the confusion matrix for the classifier supposing that we were to use a particular
   * threshold.
   * @return The confusion matrix.
   */
  public Matrix confusion() {
    return confusion;
  }

  /**
   * Returns a matrix related to the confusion matrix and to the log-likelihood.  For a
   * pretty accurate classifier, N + entropy is nearly the same as the confusion matrix
   * because log(1-eps) \approx -eps if eps is small.
   *
   * For lower accuracy classifiers, this measure will give us a better picture of how
   * things work our.
   *
   * Also, by definition, log-likelihood = sum(diag(entropy))
   * @return Returns a cell by cell break-down of the log-likelihood
   */
  public Matrix entropy() {
    if (!hasScore) {
      // find a constant score that would optimize log-likelihood, but use a dash of Bayesian
      // conservatism to avoid dividing by zero or taking log(0)
      double p = (0.5 + confusion.get(1, 1)) / (1 + confusion.get(0, 0) + confusion.get(1, 1));
      entropy.set(0, 0, confusion.get(0, 0) * Math.log1p(-p));
      entropy.set(0, 1, confusion.get(0, 1) * Math.log(p));
      entropy.set(1, 0, confusion.get(1, 0) * Math.log1p(-p));
      entropy.set(1, 1, confusion.get(1, 1) * Math.log(p));
    }
    return entropy;
  }

  public void setMaxBufferSize(int maxBufferSize) {
    this.maxBufferSize = maxBufferSize;
  }

  public boolean isProbabilityScore() {
    return probabilityScore;
  }

  public void setProbabilityScore(boolean probabilityScore) {
    this.probabilityScore = probabilityScore;
  }
}
