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

package org.apache.mahout.math.jet.random;

import org.apache.commons.math3.analysis.UnivariateFunction;
import org.apache.commons.math3.analysis.integration.RombergIntegrator;
import org.apache.commons.math3.analysis.integration.UnivariateIntegrator;
import org.junit.Assert;

import java.util.Arrays;

/**
 * Provides a consistency check for continuous distributions that relates the pdf, cdf and
 * samples.  The pdf is checked against the cdf by quadrature.  The sampling is checked
 * against the cdf using a G^2 (similar to chi^2) test.
 */
public final class DistributionChecks {

  private DistributionChecks() {
  }

  public static void checkDistribution(final AbstractContinousDistribution dist,
                                       double[] x,
                                       double offset,
                                       double scale,
                                       int n) {
    double[] xs = Arrays.copyOf(x, x.length);
    for (int i = 0; i < xs.length; i++) {
      xs[i] = xs[i]*scale+ offset;
    }
    Arrays.sort(xs);

    // collect samples
    double[] y = new double[n];
    for (int i = 0; i < n; i++) {
      y[i] = dist.nextDouble();
    }
    Arrays.sort(y);

    // compute probabilities for bins
    double[] p = new double[xs.length + 1];
    double lastP = 0;
    for (int i = 0; i < xs.length; i++) {
      double thisP = dist.cdf(xs[i]);
      p[i] = thisP - lastP;
      lastP = thisP;
    }
    p[p.length - 1] = 1 - lastP;

    // count samples in each bin
    int[] k = new int[xs.length + 1];
    int lastJ = 0;
    for (int i = 0; i < k.length - 1; i++) {
      int j = 0;
      while (j < n && y[j] < xs[i]) {
        j++;
      }
      k[i] = j - lastJ;
      lastJ = j;
    }
    k[k.length - 1] = n - lastJ;

    // now verify probabilities by comparing to integral of pdf
    UnivariateIntegrator integrator = new RombergIntegrator();
    for (int i = 0; i < xs.length - 1; i++) {
      double delta = integrator.integrate(1000000, new UnivariateFunction() {
        @Override
        public double value(double v) {
          return dist.pdf(v);
        }
      }, xs[i], xs[i + 1]);
      Assert.assertEquals(delta, p[i + 1], 1.0e-6);
    }

    // finally compute G^2 of observed versus predicted.  See http://en.wikipedia.org/wiki/G-test
    double sum = 0;
    for (int i = 0; i < k.length; i++) {
      if (k[i] != 0) {
        sum += k[i] * Math.log(k[i] / p[i] / n);
      }
    }
    sum *= 2;

    // sum is chi^2 distributed with degrees of freedom equal to number of partitions - 1
    int dof = k.length - 1;
    // fisher's approximation is that sqrt(2*x) is approximately unit normal with mean sqrt(2*dof-1)
    double z = Math.sqrt(2 * sum) - Math.sqrt(2 * dof - 1);
    Assert.assertTrue(String.format("offset=%.3f scale=%.3f Z = %.1f", offset, scale, z), Math.abs(z) < 3);
  }

  static void checkCdf(double offset,
                       double scale,
                       AbstractContinousDistribution dist,
                       double[] breaks,
                       double[] quantiles) {
    int i = 0;
    for (double x : breaks) {
      Assert.assertEquals(String.format("m=%.3f sd=%.3f x=%.3f", offset, scale, x),
          quantiles[i], dist.cdf(x * scale + offset), 1.0e-6);
      i++;
    }
  }
}
