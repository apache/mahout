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

package org.apache.mahout.math.hadoop.stats;

/**
 * Holds the total values needed to compute mean and standard deviation
 * Provides methods for their computation
 */
public final class VarianceTotals {

  private double sumOfSquares;
  private double sum;
  private double totalCount;

  public double getSumOfSquares() {
    return sumOfSquares;
  }

  public void setSumOfSquares(double sumOfSquares) {
    this.sumOfSquares = sumOfSquares;
  }

  public double getSum() {
    return sum;
  }

  public void setSum(double sum) {
    this.sum = sum;
  }

  public double getTotalCount() {
    return totalCount;
  }

  public void setTotalCount(double totalCount) {
    this.totalCount = totalCount;
  }

  public double computeMean() {
    return sum / totalCount;
  }

  public double computeVariance() {
    return ((totalCount * sumOfSquares) - (sum * sum))
          / (totalCount * (totalCount - 1.0));
  }

  public double computeVarianceForGivenMean(double mean) {
    return (sumOfSquares - totalCount * mean * mean)
          / (totalCount - 1.0);
  }

}
