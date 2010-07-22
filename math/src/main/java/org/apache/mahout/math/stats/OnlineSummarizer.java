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

package org.apache.mahout.math.stats;

import org.apache.mahout.math.list.DoubleArrayList;

/**
 * Computes on-line estimates of mean, variance and all five quartiles (notably including the
 * median).  Since this is done in a completely incremental fashion (that is what is meant by
 * on-line) estimates are available at any time and the amount of memory used is constant.  Somewhat
 * surprisingly, the quantile estimates are about as good as you would get if you actually kept all
 * of the samples.
 * <p/>
 * The method used for mean and variance is Welford's method.  See
 * <p/>
 * http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#On-line_algorithm
 * <p/>
 * The method used for computing the quartiles is a simplified form of the stochastic approximation
 * method described in the article "Incremental Quantile Estimation for Massive Tracking" by Chen,
 * Lambert and Pinheiro
 * <p/>
 * See
 * <p/>
 * http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.105.1580
 */
public class OnlineSummarizer {
  boolean sorted = true;

  // the first several samples are kept so we can boot-strap our estimates cleanly
  private DoubleArrayList starter = new DoubleArrayList(100);

  // quartile estimates
  private double q[] = new double[5];

  // mean and variance estimates
  private double mean = 0;
  private double variance = 0;

  // number of samples seen so far
  private int n = 0;

  public void add(double sample) {
    sorted = false;

    n++;
    double oldMean = mean;
    mean += (sample - mean) / n;
    double diff = (sample - mean) * (sample - oldMean);
    variance += (diff - variance) / n;

    if (n < 100) {
      starter.add(sample);
    } else if (n == 100) {
      starter.add(sample);
      q[0] = min();
      q[1] = quartile(1);
      q[2] = quartile(2);
      q[3] = quartile(3);
      q[4] = max();
      starter = null;
    } else {
      q[0] = Math.min(sample, q[0]);
      q[4] = Math.max(sample, q[4]);

      double rate = 2 * (q[3] - q[1]) / n;
      q[1] += (Math.signum(sample - q[1]) - 0.5) * rate;
      q[2] += (Math.signum(sample - q[2])) * rate;
      q[3] += (Math.signum(sample - q[3]) + 0.5) * rate;

      if (q[1] < q[0]) {
        q[1] = q[0];
      }

      if (q[3] > q[4]) {
        q[3] = q[4];
      }
    }
  }

  public int count() {
    return n;
  }

  public double mean() {
    return mean;
  }

  public double sd() {
    return Math.sqrt(variance);
  }

  public double min() {
    sort();
    if (n == 0) {
      throw new IllegalArgumentException("Must have at least one sample to estimate minimum value");
    }
    if (n <= 100) {
      return starter.get(0);
    } else {
      return q[0];
    }
  }

  private void sort() {
    if (!sorted && starter != null) {
      starter.sortFromTo(0, 99);
      sorted = true;
    }
  }

  public double max() {
    sort();
    if (n == 0) {
      throw new IllegalArgumentException("Must have at least one sample to estimate maximum value");
    }
    if (n <= 100) {
      return starter.get(99);
    } else {
      return q[4];
    }
  }

  public double quartile(int i) {
    sort();
    switch (i) {
      case 0:
        return min();
      case 1:
      case 2:
      case 3:
        if (n > 100) {
          return q[i];
        } else if (n < 2) {
          throw new IllegalArgumentException("Must have at least two samples to estimate quartiles");
        } else {
          double x = i * (n - 1) / 4.0;
          int k = (int) Math.floor(x);
          double u = x - k;
          return starter.get(k) * (1 - u) + starter.get(k + 1) * u;
        }
      case 4:
        return max();
      default:
        throw new IllegalArgumentException("Quartile number must be in the range [0..4] not " + i);
    }
  }

  public double median() {
    return quartile(2);
  }
}
