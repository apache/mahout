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
 * <p>
 * Extends {@link CompactRunningAverage} to add a running standard deviation computation.
 * Uses Welford's method, as described at http://www.johndcook.com/standard_deviation.html
 * </p>
 */
public final class CompactRunningAverageAndStdDev extends CompactRunningAverage implements RunningAverageAndStdDev {

  private float stdDev = Float.NaN;
  private float mk;
  private float sk;

  @Override
  public synchronized double getStandardDeviation() {
    return stdDev;
  }

  @Override
  public synchronized void addDatum(double datum) {
    super.addDatum(datum);
    int count = getCount();
    if (count == 1) {
      mk = (float) datum;
      sk = 0.0f;
    } else {
      float oldmk = mk;
      float diff = (float) datum - oldmk;
      mk += diff / count;
      sk += diff * (datum - mk);
    }
    recomputeStdDev();
  }

  @Override
  public synchronized void removeDatum(double datum) {
    int oldCount = getCount();
    super.removeDatum(datum);
    float oldmk = mk;
    mk = (oldCount * oldmk - (float) datum) / (oldCount - 1);
    sk -= (datum - mk) * (datum - oldmk);
    recomputeStdDev();
  }

  /**
   * @throws UnsupportedOperationException
   */
  @Override
  public void changeDatum(double delta) {
    throw new UnsupportedOperationException();
  }

  private synchronized void recomputeStdDev() {
    int count = getCount();
    if (count > 1) {
      stdDev = (float) Math.sqrt(sk / (count - 1));
    } else {
      stdDev = Float.NaN;
    }
  }

  @Override
  public synchronized String toString() {
    return String.valueOf(String.valueOf(getAverage()) + ',' + stdDev);
  }

}
