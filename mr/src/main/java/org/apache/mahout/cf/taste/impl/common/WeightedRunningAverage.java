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

import java.io.Serializable;

import com.google.common.base.Preconditions;

public class WeightedRunningAverage implements RunningAverage, Serializable {

  private double totalWeight;
  private double average;

  public WeightedRunningAverage() {
    totalWeight = 0.0;
    average = Double.NaN;
  }

  @Override
  public synchronized void addDatum(double datum) {
    addDatum(datum, 1.0);
  }

  public synchronized void addDatum(double datum, double weight) {
    double oldTotalWeight = totalWeight;
    totalWeight += weight;
    if (oldTotalWeight <= 0.0) {
      average = datum;
    } else {
      average = average * oldTotalWeight / totalWeight + datum * weight / totalWeight;
    }
  }

  @Override
  public synchronized void removeDatum(double datum) {
    removeDatum(datum, 1.0);
  }

  public synchronized void removeDatum(double datum, double weight) {
    double oldTotalWeight = totalWeight;
    totalWeight -= weight;
    if (totalWeight <= 0.0) {
      average = Double.NaN;
      totalWeight = 0.0;
    } else {
      average = average * oldTotalWeight / totalWeight - datum * weight / totalWeight;
    }
  }

  @Override
  public synchronized void changeDatum(double delta) {
    changeDatum(delta, 1.0);
  }

  public synchronized void changeDatum(double delta, double weight) {
    Preconditions.checkArgument(weight <= totalWeight, "weight must be <= totalWeight");
    average += delta * weight / totalWeight;
  }

  public synchronized double getTotalWeight() {
    return totalWeight;
  }

  /** @return {@link #getTotalWeight()} */
  @Override
  public synchronized int getCount() {
    return (int) totalWeight;
  }

  @Override
  public synchronized double getAverage() {
    return average;
  }

  @Override
  public RunningAverage inverse() {
    return new InvertedRunningAverage(this);
  }

  @Override
  public synchronized String toString() {
    return String.valueOf(average);
  }

}
