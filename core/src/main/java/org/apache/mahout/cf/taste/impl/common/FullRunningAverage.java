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

/**
 * <p>
 * A simple class that can keep track of a running average of a series of numbers. One can add to or remove
 * from the series, as well as update a datum in the series. The class does not actually keep track of the
 * series of values, just its running average, so it doesn't even matter if you remove/change a value that
 * wasn't added.
 * </p>
 */
public class FullRunningAverage implements RunningAverage, Serializable {
  
  private int count;
  private double average;
  
  public FullRunningAverage() {
    this(0, Double.NaN);
  }

  public FullRunningAverage(int count, double average) {
    this.count = count;
    this.average = average;
  }

  /**
   * @param datum
   *          new item to add to the running average
   */
  @Override
  public synchronized void addDatum(double datum) {
    if (++count == 1) {
      average = datum;
    } else {
      average = average * (count - 1) / count + datum / count;
    }
  }
  
  /**
   * @param datum
   *          item to remove to the running average
   * @throws IllegalStateException
   *           if count is 0
   */
  @Override
  public synchronized void removeDatum(double datum) {
    if (count == 0) {
      throw new IllegalStateException();
    }
    if (--count == 0) {
      average = Double.NaN;
    } else {
      average = average * (count + 1) / count - datum / count;
    }
  }
  
  /**
   * @param delta
   *          amount by which to change a datum in the running average
   * @throws IllegalStateException
   *           if count is 0
   */
  @Override
  public synchronized void changeDatum(double delta) {
    if (count == 0) {
      throw new IllegalStateException();
    }
    average += delta / count;
  }
  
  @Override
  public synchronized int getCount() {
    return count;
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
