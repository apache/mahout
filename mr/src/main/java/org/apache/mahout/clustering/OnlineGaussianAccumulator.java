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
package org.apache.mahout.clustering;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.SquareRootFunction;

/**
 * An online Gaussian statistics accumulator based upon Knuth (who cites Welford) which is declared to be
 * numerically-stable. See http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 */
public class OnlineGaussianAccumulator implements GaussianAccumulator {

  private double sumWeight;
  private Vector mean;
  private Vector s;
  private Vector variance;

  @Override
  public double getN() {
    return sumWeight;
  }

  @Override
  public Vector getMean() {
    return mean;
  }

  @Override
  public Vector getStd() {
    return variance.clone().assign(new SquareRootFunction());
  }

  /* from Wikipedia: http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
   * 
   * Weighted incremental algorithm
   * 
   * def weighted_incremental_variance(dataWeightPairs):
   * mean = 0
   * S = 0
   * sumweight = 0
   * for x, weight in dataWeightPairs: # Alternately "for x in zip(data, weight):"
   *     temp = weight + sumweight
   *     Q = x - mean
   *      R = Q * weight / temp
   *      S = S + sumweight * Q * R
   *      mean = mean + R
   *      sumweight = temp
   *  Variance = S / (sumweight-1)  # if sample is the population, omit -1
   *  return Variance
   */
  @Override
  public void observe(Vector x, double weight) {
    double temp = weight + sumWeight;
    Vector q;
    if (mean == null) {
      mean = x.like();
      q = x.clone();
    } else {
      q = x.minus(mean);
    }
    Vector r = q.times(weight).divide(temp);
    if (s == null) {
      s = q.times(sumWeight).times(r);
    } else {
      s = s.plus(q.times(sumWeight).times(r));
    }
    mean = mean.plus(r);
    sumWeight = temp;
    variance = s.divide(sumWeight - 1); //  # if sample is the population, omit -1
  }

  @Override
  public void compute() {
    // nothing to do here!
  }

  @Override
  public double getAverageStd() {
    if (sumWeight == 0.0) {
      return 0.0;
    } else {
      Vector std = getStd();
      return std.zSum() / std.size();
    }
  }

  @Override
  public Vector getVariance() {
    return variance;
  }

}
