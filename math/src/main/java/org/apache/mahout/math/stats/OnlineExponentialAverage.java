/*
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

package org.apache.mahout.math.stats;

/**
 * Computes an online average that is exponentially weighted toward recent time-embedded samples.
 */
public class OnlineExponentialAverage {

  private final double alpha;
  private double lastT;
  private double s;
  private double w;
  private double t;

  /**
   * Creates an averager that has a specified time constant for discounting old data. The time
   * constant, alpha, is the time at which an older sample is discounted to 1/e relative to current
   * data.  Roughly speaking, data that is more than 3*alpha old doesn't matter any more and data
   * that is more recent than alpha/3 is about as important as current data.
   *
   * See http://tdunning.blogspot.com/2011/03/exponential-weighted-averages-with.html for a
   * derivation.  See http://tdunning.blogspot.com/2011/03/exponentially-weighted-averaging-for.html
   * for the rate method.
   *
   * @param alpha The time constant for discounting old data and state.
   */
  public OnlineExponentialAverage(double alpha) {
    this.alpha = alpha;
  }

  public void add(double t, double x) {
    double pi = Math.exp(-(t - lastT) / alpha);
    s = x + pi * s;
    w = 1.0 + pi * w;
    this.t = t - lastT + pi * this.t;
    lastT = t;
  }

  public double mean() {
    return s / w;
  }

  public double meanRate() {
    return s / t;
  }
}
