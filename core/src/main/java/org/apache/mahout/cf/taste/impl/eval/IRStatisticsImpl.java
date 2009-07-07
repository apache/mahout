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

package org.apache.mahout.cf.taste.impl.eval;

import org.apache.mahout.cf.taste.eval.IRStatistics;

import java.io.Serializable;

public final class IRStatisticsImpl implements IRStatistics, Serializable {

  private final double precision;
  private final double recall;
  private final double fallOut;

  IRStatisticsImpl(double precision, double recall, double fallOut) {
    if (precision < 0.0 || precision > 1.0) {
      throw new IllegalArgumentException("Illegal precision: " + precision);
    }
    if (recall < 0.0 || recall > 1.0) {
      throw new IllegalArgumentException("Illegal recall: " + recall);
    }
    if (fallOut < 0.0 || fallOut > 1.0) {
      throw new IllegalArgumentException("Illegal fallOut: " + fallOut);
    }
    this.precision = precision;
    this.recall = recall;
    this.fallOut = fallOut;
  }

  @Override
  public double getPrecision() {
    return precision;
  }

  @Override
  public double getRecall() {
    return recall;
  }

  @Override
  public double getFallOut() {
    return fallOut;
  }

  @Override
  public double getF1Measure() {
    return getFNMeasure(1.0);
  }

  @Override
  public double getFNMeasure(double n) {
    double sum = n * precision + recall;
    return sum == 0.0 ? Double.NaN : (1.0 + n) * precision * recall / sum;
  }

  @Override
  public String toString() {
    return "IRStatisticsImpl[precision:" + precision + ",recall:" + recall + ",fallOut:" + fallOut + ']';
  }

}