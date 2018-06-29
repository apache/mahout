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

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.MahoutTestCase;
import org.junit.Test;

import java.util.Arrays;
import java.util.Locale;
import java.util.Random;

public final class GammaTest extends MahoutTestCase {

  @Test
  public void testNextDouble() {
    double[] z = new double[100000];
    Random gen = RandomUtils.getRandom();
    for (double alpha : new double[]{1, 2, 10, 0.1, 0.01, 100}) {
      Gamma g = new Gamma(alpha, 1, gen);
      for (int i = 0; i < z.length; i++) {
        z[i] = g.nextDouble();
      }
      Arrays.sort(z);

      // verify that empirical CDF matches theoretical one pretty closely
      for (double q : seq(0.01, 1, 0.01)) {
        double p = z[(int) (q * z.length)];
        assertEquals(q, g.cdf(p), 0.01);
      }
    }
  }

  @Test
  public void testCdf() {
    Random gen = RandomUtils.getRandom();

    // verify scaling for special case of alpha = 1
    for (double beta : new double[]{1, 0.1, 2, 100}) {
      Gamma g1 = new Gamma(1, beta, gen);
      Gamma g2 = new Gamma(1, 1, gen);
      for (double x : seq(0, 0.99, 0.1)) {
        assertEquals(String.format(Locale.ENGLISH, "Rate invariance: x = %.4f, alpha = 1, beta = %.1f", x, beta),
          1 - Math.exp(-x * beta), g1.cdf(x), 1.0e-9);
        assertEquals(String.format(Locale.ENGLISH, "Rate invariance: x = %.4f, alpha = 1, beta = %.1f", x, beta),
          g2.cdf(beta * x), g1.cdf(x), 1.0e-9);
      }
    }

    // now test scaling for a selection of values of alpha
    for (double alpha : new double[]{0.01, 0.1, 1, 2, 10, 100, 1000}) {
      Gamma g = new Gamma(alpha, 1, gen);
      for (double beta : new double[]{0.1, 1, 2, 100}) {
        Gamma g1 = new Gamma(alpha, beta, gen);
        for (double x : seq(0, 0.9999, 0.001)) {
          assertEquals(
            String.format(Locale.ENGLISH, "Rate invariance: x = %.4f, alpha = %.2f, beta = %.1f", x, alpha, beta),
            g.cdf(x * beta), g1.cdf(x), 0);
        }
      }
    }

    // now check against known values computed using R for various values of alpha
    checkGammaCdf(0.01, 1, 0.0000000, 0.9450896, 0.9516444, 0.9554919, 0.9582258, 0.9603474, 0.9620810, 0.9635462, 0.9648148, 0.9659329, 0.9669321);
    checkGammaCdf(0.1, 1, 0.0000000, 0.7095387, 0.7591012, 0.7891072, 0.8107067, 0.8275518, 0.8413180, 0.8529198, 0.8629131, 0.8716623, 0.8794196);
    checkGammaCdf(1, 1, 0.0000000, 0.1812692, 0.3296800, 0.4511884, 0.5506710, 0.6321206, 0.6988058, 0.7534030, 0.7981035, 0.8347011, 0.8646647);
    checkGammaCdf(10, 1, 0.000000e+00, 4.649808e-05, 8.132243e-03, 8.392402e-02, 2.833757e-01, 5.420703e-01, 7.576078e-01, 8.906006e-01, 9.567017e-01, 9.846189e-01, 9.950046e-01);
    checkGammaCdf(100, 1, 0.000000e+00, 3.488879e-37, 1.206254e-15, 1.481528e-06, 1.710831e-02, 5.132988e-01, 9.721363e-01, 9.998389e-01, 9.999999e-01, 1.000000e+00, 1.000000e+00);

//    > pgamma(seq(0,0.02,by=0.002),0.01,1)
//     [1] 0.0000000 0.9450896 0.9516444 0.9554919 0.9582258 0.9603474 0.9620810 0.9635462 0.9648148 0.9659329 0.9669321
//    > pgamma(seq(0,0.2,by=0.02),0.1,1)
//     [1] 0.0000000 0.7095387 0.7591012 0.7891072 0.8107067 0.8275518 0.8413180 0.8529198 0.8629131 0.8716623 0.8794196
//    > pgamma(seq(0,2,by=0.2),1,1)
//     [1] 0.0000000 0.1812692 0.3296800 0.4511884 0.5506710 0.6321206 0.6988058 0.7534030 0.7981035 0.8347011 0.8646647
//    > pgamma(seq(0,20,by=2),10,1)
//     [1] 0.000000e+00 4.649808e-05 8.132243e-03 8.392402e-02 2.833757e-01 5.420703e-01 7.576078e-01 8.906006e-01 9.567017e-01 9.846189e-01 9.950046e-01
//    > pgamma(seq(0,200,by=20),100,1)
//     [1] 0.000000e+00 3.488879e-37 1.206254e-15 1.481528e-06 1.710831e-02 5.132988e-01 9.721363e-01 9.998389e-01 9.999999e-01 1.000000e+00 1.000000e+00
  }

  private static void checkGammaCdf(double alpha, double beta, double... values) {
    Gamma g = new Gamma(alpha, beta, RandomUtils.getRandom());
    int i = 0;
    for (double x : seq(0, 2 * alpha, 2 * alpha / 10)) {
      assertEquals(String.format(Locale.ENGLISH, "alpha=%.2f, i=%d, x=%.2f", alpha, i, x),
                                        values[i], g.cdf(x), 1.0e-7);
      i++;
    }
  }

  private static double[] seq(double from, double to, double by) {
    double[] r = new double[(int) Math.ceil(0.999999 * (to - from) / by)];
    int i = 0;
    for (double x = from; x < to - (to - from) * 1.0e-6; x += by) {
      r[i++] = x;
    }
    return r;
  }

  @Test
  public void testPdf() {
    Random gen = RandomUtils.getRandom();
    for (double alpha : new double[]{0.01, 0.1, 1, 2, 10, 100}) {
      for (double beta : new double[]{0.1, 1, 2, 100}) {
        Gamma g1 = new Gamma(alpha, beta, gen);
        for (double x : seq(0, 0.99, 0.1)) {
          double p = Math.pow(beta, alpha) * Math.pow(x, alpha - 1) *
            Math.exp(-beta * x - org.apache.mahout.math.jet.stat.Gamma.logGamma(alpha));
          assertEquals(String.format(Locale.ENGLISH, "alpha=%.2f, beta=%.2f, x=%.2f\n", alpha, beta, x),
            p, g1.pdf(x), 1.0e-9);
        }
      }
    }
  }
}
