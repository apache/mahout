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

/** <p>Extends {@link FullRunningAverage} to add a running standard deviation computation.</p> */
public final class FullRunningAverageAndStdDev extends FullRunningAverage implements RunningAverageAndStdDev {

  private double stdDev;
  private double sumX2;

  public FullRunningAverageAndStdDev() {
    stdDev = Double.NaN;
  }

  @Override
  public double getStandardDeviation() {
    return stdDev;
  }

  @Override
  public void addDatum(double datum) {
    super.addDatum(datum);
    sumX2 += datum * datum;
    recomputeStdDev();
  }

  @Override
  public void removeDatum(double datum) {
    super.removeDatum(datum);
    sumX2 -= datum * datum;
    recomputeStdDev();
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void changeDatum(double delta) {
    throw new UnsupportedOperationException();
  }

  private void recomputeStdDev() {
    int count = getCount();
    if (count > 1) {
      double average = getAverage();
      stdDev = Math.sqrt((sumX2 - average * average * (double) count) / (double) (count - 1));
    } else {
      stdDev = Double.NaN;
    }
  }

  @Override
  public String toString() {
    return String.valueOf(String.valueOf(getAverage()) + ',' + stdDev);
  }

}
