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
 * A simple class that represents a fixed value of an average and count. This is useful
 * when an API needs to return {@link RunningAverage} but is not in a position to accept
 * updates to it.
 * </p>
 */
public class FixedRunningAverage implements RunningAverage, Serializable {

  private final double average;
  private final int count;

  public FixedRunningAverage(double average, int count) {
    this.average = average;
    this.count = count;
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public synchronized void addDatum(double datum) {
    throw new UnsupportedOperationException();
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public synchronized void removeDatum(double datum) {
    throw new UnsupportedOperationException();
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public synchronized void changeDatum(double delta) {
    throw new UnsupportedOperationException();
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
