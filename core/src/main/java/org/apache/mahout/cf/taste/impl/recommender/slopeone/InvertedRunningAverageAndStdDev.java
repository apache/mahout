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

package org.apache.mahout.cf.taste.impl.recommender.slopeone;

import org.apache.mahout.cf.taste.impl.common.RunningAverageAndStdDev;

final class InvertedRunningAverageAndStdDev implements RunningAverageAndStdDev {

  private final RunningAverageAndStdDev delegate;

  InvertedRunningAverageAndStdDev(RunningAverageAndStdDev delegate) {
    this.delegate = delegate;
  }

  @Override
  public void addDatum(double datum) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void removeDatum(double datum) {
    throw new UnsupportedOperationException();
  }

  @Override
  public void changeDatum(double delta) {
    throw new UnsupportedOperationException();
  }

  @Override
  public int getCount() {
    return delegate.getCount();
  }

  @Override
  public double getAverage() {
    return -delegate.getAverage();
  }

  @Override
  public double getStandardDeviation() {
    return delegate.getStandardDeviation();
  }

}