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

package org.apache.mahout.math.random;

import com.google.common.base.Preconditions;
import org.apache.mahout.common.RandomUtils;

import java.util.Random;

/**
 * Samples from an empirical cumulative distribution.
 */
public final class Empirical extends AbstractSamplerFunction {
  private final Random gen;
  private final boolean exceedMinimum;
  private final boolean exceedMaximum;

  private final double[] x;
  private final double[] y;
  private final int n;

  /**
   * Sets up a sampler for a specified empirical cumulative distribution function.  The distribution
   * can have optional exponential tails on either or both ends, but otherwise does a linear
   * interpolation between known points.
   *
   * @param exceedMinimum  Should we generate samples less than the smallest quantile (i.e. generate a left tail)?
   * @param exceedMaximum  Should we generate samples greater than the largest observed quantile (i.e. generate a right
   *                       tail)?
   * @param samples        The number of samples observed to get the quantiles.
   * @param ecdf           Alternating values that represent which percentile (in the [0..1] range)
   *                       and values.  For instance, if you have the min, median and max of 1, 3, 10
   *                       you should pass 0.0, 1, 0.5, 3, 1.0, 10.  Note that the list must include
   *                       the 0-th (1.0-th) quantile if the left (right) tail is not allowed.
   */
  public Empirical(boolean exceedMinimum, boolean exceedMaximum, int samples, double... ecdf) {
    Preconditions.checkArgument(ecdf.length % 2 == 0, "ecdf must have an even count of values");
    Preconditions.checkArgument(samples >= 3, "Sample size must be >= 3");

    // if we can't exceed the observed bounds, then we have to be given the bounds.
    Preconditions.checkArgument(exceedMinimum || ecdf[0] == 0);
    Preconditions.checkArgument(exceedMaximum || ecdf[ecdf.length - 2] == 1);

    gen = RandomUtils.getRandom();

    n = ecdf.length / 2;
    x = new double[n];
    y = new double[n];

    double lastX = ecdf[1];
    double lastY = ecdf[0];
    for (int i = 0; i < ecdf.length; i += 2) {
      // values have to be monotonic increasing
      Preconditions.checkArgument(i == 0 || ecdf[i + 1] > lastY);
      y[i / 2] = ecdf[i + 1];
      lastY = y[i / 2];

      // quantiles have to be in [0,1] and be monotonic increasing
      Preconditions.checkArgument(ecdf[i] >= 0 && ecdf[i] <= 1);
      Preconditions.checkArgument(i == 0 || ecdf[i] > lastX);

      x[i / 2] = ecdf[i];
      lastX = x[i / 2];
    }

    // squeeze a bit to allow for unobserved tails
    double x0 = exceedMinimum ? 0.5 / samples : 0;
    double x1 = 1 - (exceedMaximum ? 0.5 / samples : 0);
    for (int i = 0; i < n; i++) {
      x[i] = x[i] * (x1 - x0) + x0;
    }

    this.exceedMinimum = exceedMinimum;
    this.exceedMaximum = exceedMaximum;
  }

  @Override
  public Double sample() {
    return sample(gen.nextDouble());
  }

  public double sample(double u) {
    if (exceedMinimum && u < x[0]) {
      // generate from left tail
      if (u == 0) {
        u = 1.0e-16;
      }
      return y[0] + Math.log(u / x[0]) * x[0] * (y[1] - y[0]) / (x[1] - x[0]);
    } else if (exceedMaximum && u > x[n - 1]) {
      if (u == 1) {
        u = 1 - 1.0e-16;
      }
      // generate from right tail
      double dy = y[n - 1] - y[n - 2];
      double dx = x[n - 1] - x[n - 2];
      return y[n - 1] - Math.log((1 - u) / (1 - x[n - 1])) * (1 - x[n - 1]) * dy / dx;
    } else {
      // linear interpolation
      for (int i = 1; i < n; i++) {
        if (x[i] > u) {
          double dy = y[i] - y[i - 1];
          double dx = x[i] - x[i - 1];
          return y[i - 1] + (u - x[i - 1]) * dy / dx;
        }
      }
      throw new RuntimeException(String.format("Can't happen (%.3f is not in [%.3f,%.3f]", u, x[0], x[n - 1]));
    }
  }
}
