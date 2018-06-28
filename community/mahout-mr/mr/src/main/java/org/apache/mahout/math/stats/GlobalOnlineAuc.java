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

package org.apache.mahout.math.stats;

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseMatrix;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Matrix;
import org.apache.mahout.math.MatrixWritable;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.VectorWritable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.util.Random;

/**
 * Computes a running estimate of AUC (see http://en.wikipedia.org/wiki/Receiver_operating_characteristic).
 * <p/>
 * Since AUC is normally a global property of labeled scores, it is almost always computed in a
 * batch fashion.  The probabilistic definition (the probability that a random element of one set
 * has a higher score than a random element of another set) gives us a way to estimate this
 * on-line.
 *
 * @see GroupedOnlineAuc
 */
public class GlobalOnlineAuc implements OnlineAuc {
  enum ReplacementPolicy {
    FIFO, FAIR, RANDOM
  }

  // increasing this to 100 causes very small improvements in accuracy.  Decreasing it to 2
  // causes substantial degradation for the FAIR and RANDOM policies, but almost no change
  // for the FIFO policy
  public static final int HISTORY = 10;

  // defines the exponential averaging window for results
  private int windowSize = Integer.MAX_VALUE;

  // FIFO has distinctly the best properties as a policy.  See OnlineAucTest for details
  private ReplacementPolicy policy = ReplacementPolicy.FIFO;
  private final Random random = RandomUtils.getRandom();
  private Matrix scores;
  private Vector averages;
  private Vector samples;

  public GlobalOnlineAuc() {
    int numCategories = 2;
    scores = new DenseMatrix(numCategories, HISTORY);
    scores.assign(Double.NaN);
    averages = new DenseVector(numCategories);
    averages.assign(0.5);
    samples = new DenseVector(numCategories);
  }

  @Override
  public double addSample(int category, String groupKey, double score) {
    return addSample(category, score);
  }

  @Override
  public double addSample(int category, double score) {
    int n = (int) samples.get(category);
    if (n < HISTORY) {
      scores.set(category, n, score);
    } else {
      switch (policy) {
        case FIFO:
          scores.set(category, n % HISTORY, score);
          break;
        case FAIR:
          int j1 = random.nextInt(n + 1);
          if (j1 < HISTORY) {
            scores.set(category, j1, score);
          }
          break;
        case RANDOM:
          int j2 = random.nextInt(HISTORY);
          scores.set(category, j2, score);
          break;
        default:
          throw new IllegalStateException("Unknown policy: " + policy);
      }
    }

    samples.set(category, n + 1);

    if (samples.minValue() >= 1) {
      // compare to previous scores for other category
      Vector row = scores.viewRow(1 - category);
      double m = 0.0;
      double count = 0.0;
      for (Vector.Element element : row.all()) {
        double v = element.get();
        if (Double.isNaN(v)) {
          continue;
        }
        count++;
        if (score > v) {
          m++;
          // } else if (score < v) {
          // m += 0
        } else if (score == v) {
          m += 0.5;
        }
      }
      averages.set(category, averages.get(category)
        + (m / count - averages.get(category)) / Math.min(windowSize, samples.get(category)));
    }
    return auc();
  }

  @Override
  public double auc() {
    // return an unweighted average of all averages.
    return (1 - averages.get(0) + averages.get(1)) / 2;
  }

  public double value() {
    return auc();
  }

  @Override
  public void setPolicy(ReplacementPolicy policy) {
    this.policy = policy;
  }

  @Override
  public void setWindowSize(int windowSize) {
    this.windowSize = windowSize;
  }

  @Override
  public void write(DataOutput out) throws IOException {
    out.writeInt(windowSize);
    out.writeInt(policy.ordinal());
    MatrixWritable.writeMatrix(out, scores);
    VectorWritable.writeVector(out, averages);
    VectorWritable.writeVector(out, samples);
  }

  @Override
  public void readFields(DataInput in) throws IOException {
    windowSize = in.readInt();
    policy = ReplacementPolicy.values()[in.readInt()];

    scores = MatrixWritable.readMatrix(in);
    averages = VectorWritable.readVector(in);
    samples = VectorWritable.readVector(in);
  }

}
