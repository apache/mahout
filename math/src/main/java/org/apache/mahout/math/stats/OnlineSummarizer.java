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

  private TDigest quantiles = new TDigest(100);

  // mean and variance estimates
  private double mean;
  private double variance;

  // number of samples seen so far
  private int n;

  public void add(double sample) {
    n++;
    double oldMean = mean;
    mean += (sample - mean) / n;
    double diff = (sample - mean) * (sample - oldMean);
    variance += (diff - variance) / n;

    quantiles.add(sample);
  }

  public int getCount() {
    return n;
  }

  public double getMean() {
    return mean;
  }

  public double getSD() {
    return Math.sqrt(variance);
  }

  public double getMin() {
    return getQuartile(0);
  }

  public double getMax() {
    return getQuartile(4);
  }

  public double getQuartile(int i) {
    return quantiles.quantile(0.25 * i);
  }

  public double quantile(double q) {
    return quantiles.quantile(q);
  }

  public double getMedian() {
    return getQuartile(2);
  }
}
