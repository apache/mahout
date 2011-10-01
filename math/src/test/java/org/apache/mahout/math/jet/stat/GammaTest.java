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

package org.apache.mahout.math.jet.stat;

import com.google.common.base.Charsets;
import com.google.common.base.Splitter;
import com.google.common.collect.Iterables;
import com.google.common.io.CharStreams;
import com.google.common.io.InputSupplier;
import com.google.common.io.Resources;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.MahoutTestCase;
import org.junit.Test;

import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Random;

public final class GammaTest extends MahoutTestCase {

  @Test
  public void testGamma() {
    double[] x = {1, 2, 5, 10, 20, 50, 100};
    double[] expected = {
        1.000000e+00, 1.000000e+00, 2.400000e+01, 3.628800e+05, 1.216451e+17, 6.082819e+62, 9.332622e+155
    };

    for (int i = 0; i < x.length; i++) {
      assertEquals(expected[i], Gamma.gamma(x[i]), expected[i] * 1.0e-5);
      assertEquals(gammaInteger(x[i]), Gamma.gamma(x[i]), expected[i] * 1.0e-5);
      assertEquals(gammaInteger(x[i]), Math.exp(Gamma.logGamma(x[i])), expected[i] * 1.0e-5);
    }
  }

  @Test
  public void testNegativeArgForGamma() {
    double[] x = {-30.3, -20.7, -10.5, -1.1, 0.5, 0.99, -0.999};
    double[] expected = {
        -5.243216e-33, -1.904051e-19, -2.640122e-07, 9.714806e+00, 1.772454e+00, 1.005872e+00, -1.000424e+03
    };

    for (int i = 0; i < x.length; i++) {
      assertEquals(expected[i], Gamma.gamma(x[i]), Math.abs(expected[i] * 1.0e-5));
      assertEquals(Math.abs(expected[i]), Math.abs(Math.exp(Gamma.logGamma(x[i]))), Math.abs(expected[i] * 1.0e-5));
    }
  }

  private static double gammaInteger(double x) {
    double r = 1;
    for (int i = 2; i < x; i++) {
      r *= i;
    }
    return r;
  }

  @Test
  public void testBigX() {
    assertEquals(factorial(4), 4 * 3 * 2, 0);
    assertEquals(factorial(4), Gamma.gamma(5), 0);
    assertEquals(factorial(14), Gamma.gamma(15), 0);
    assertEquals(factorial(34), Gamma.gamma(35), 1.0e-15 * factorial(34));
    assertEquals(factorial(44), Gamma.gamma(45), 1.0e-15 * factorial(44));

    assertEquals(-6.884137e-40 + 3.508309e-47, Gamma.gamma(-35.1), 1.0e-52);
    assertEquals(-3.915646e-41 - 3.526813e-48 - 1.172516e-55, Gamma.gamma(-35.9), 1.0e-52);
    assertEquals(-2000000000.577215, Gamma.gamma(-0.5e-9), 1.0e-15 * 2000000000.577215);
    assertEquals(1999999999.422784, Gamma.gamma(0.5e-9), 1.0e-15 * 1999999999.422784);
    assertEquals(1.324296658017984e+252, Gamma.gamma(146.1), 1.0e-10 * 1.324296658017984e+252);

    for (double x : new double[]{5, 15, 35, 45, -35.1, -35.9, -0.5e-9, 0.5e-9, 146.1}) {
      double ref = Math.log(Math.abs(Gamma.gamma(x)));
      double actual = Gamma.logGamma(x);
      double diff = Math.abs(ref - actual) / ref;
      assertEquals("gamma versus logGamma at " + x + " (diff = " + diff + ')', 0, (ref - actual) / ref, 1.0e-8);
    }
  }

  private static double factorial(int n) {
    double r = 1;
    for (int i = 2; i <= n; i++) {
      r *= i;
    }
    return r;
  }

  @Test
  public void beta() {
    Random r = RandomUtils.getRandom();
    for (int i = 0; i < 200; i++) {
      double alpha = -50 * Math.log1p(-r.nextDouble());
      double beta = -50 * Math.log1p(-r.nextDouble());
      double ref = Math.exp(Gamma.logGamma(alpha) + Gamma.logGamma(beta) - Gamma.logGamma(alpha + beta));
      double actual = Gamma.beta(alpha, beta);
      double err = (ref - actual) / ref;
      assertEquals("beta at (" + alpha + ", " + beta + ") relative error = " + err, 0, err, 1.0e-10);
    }
  }

  @Test
  public void incompleteBeta() throws IOException {
    Splitter onComma = Splitter.on(",").trimResults();

    InputSupplier<InputStreamReader> input =
        Resources.newReaderSupplier(Resources.getResource("beta-test-data.csv"), Charsets.UTF_8);
    boolean header = true;
    for (String line : CharStreams.readLines(input)) {
      if (header) {
        // skip
        header = false;
      } else {
        Iterable<String> values = onComma.split(line);
        double alpha = Double.parseDouble(Iterables.get(values, 0));
        double beta = Double.parseDouble(Iterables.get(values, 1));
        double x = Double.parseDouble(Iterables.get(values, 2));
        double ref = Double.parseDouble(Iterables.get(values, 3));
        double actual = Gamma.incompleteBeta(alpha, beta, x);
        assertEquals(alpha + "," + beta + ',' + x, ref, actual, ref * 1.0e-5);
      }
    }
  }


}
