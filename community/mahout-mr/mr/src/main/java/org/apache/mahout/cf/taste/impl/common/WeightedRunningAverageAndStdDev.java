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

package org.apache.mahout.cf.taste.impl.common;

/**
 * This subclass also provides for a weighted estimate of the sample standard deviation.
 * See <a href="http://en.wikipedia.org/wiki/Mean_square_weighted_deviation">estimate formulae here</a>.
 */
public final class WeightedRunningAverageAndStdDev extends WeightedRunningAverage implements RunningAverageAndStdDev {

  private double totalSquaredWeight;
  private double totalWeightedData;
  private double totalWeightedSquaredData;

  public WeightedRunningAverageAndStdDev() {
    totalSquaredWeight = 0.0;
    totalWeightedData = 0.0;
    totalWeightedSquaredData = 0.0;
  }
  
  @Override
  public synchronized void addDatum(double datum, double weight) {
    super.addDatum(datum, weight);
    totalSquaredWeight += weight * weight;
    double weightedData = datum * weight;
    totalWeightedData += weightedData;
    totalWeightedSquaredData += weightedData * datum;
  }
  
  @Override
  public synchronized void removeDatum(double datum, double weight) {
    super.removeDatum(datum, weight);
    totalSquaredWeight -= weight * weight;
    if (totalSquaredWeight <= 0.0) {
      totalSquaredWeight = 0.0;
    }
    double weightedData = datum * weight;
    totalWeightedData -= weightedData;
    if (totalWeightedData <= 0.0) {
      totalWeightedData = 0.0;
    }
    totalWeightedSquaredData -= weightedData * datum;
    if (totalWeightedSquaredData <= 0.0) {
      totalWeightedSquaredData = 0.0;
    }
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public synchronized void changeDatum(double delta, double weight) {
    throw new UnsupportedOperationException();
  }
  

  @Override
  public synchronized double getStandardDeviation() {
    double totalWeight = getTotalWeight();
    return Math.sqrt((totalWeightedSquaredData * totalWeight - totalWeightedData * totalWeightedData)
                         / (totalWeight * totalWeight - totalSquaredWeight));
  }

  @Override
  public RunningAverageAndStdDev inverse() {
    return new InvertedRunningAverageAndStdDev(this);
  }
  
  @Override
  public synchronized String toString() {
    return String.valueOf(String.valueOf(getAverage()) + ',' + getStandardDeviation());
  }

}
