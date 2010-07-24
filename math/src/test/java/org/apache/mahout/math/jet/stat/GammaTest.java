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

import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class GammaTest {
  @Test
  public void testGamma() {
    double[] x = new double[]{1, 2, 5, 10, 20, 50, 100};
    double[] expected = new double[]{1.000000e+00, 1.000000e+00, 2.400000e+01, 3.628800e+05, 1.216451e+17, 6.082819e+62, 9.332622e+155};

    for (int i = 0; i < x.length; i++) {
      assertEquals(expected[i], Gamma.gamma(x[i]), expected[i] * 1e-5);
      assertEquals(gammaInteger(x[i]), Gamma.gamma(x[i]), expected[i] * 1e-5);
      assertEquals(gammaInteger(x[i]), Math.exp(Gamma.logGamma(x[i])), expected[i] * 1e-5);
    }
  }

  @Test
  public void testNegativeArgForGamma() {
    double[] x = new double[]{-30.3,-20.7,-10.5,-1.1,0.5,0.99,-0.999};
    double[] expected = new double[]{-5.243216e-33, -1.904051e-19, -2.640122e-07, 9.714806e+00,  1.772454e+00,  1.005872e+00, -1.000424e+03};

    for (int i = 0; i < x.length; i++) {
      assertEquals(expected[i], Gamma.gamma(x[i]), Math.abs(expected[i] * 1e-5));
      assertEquals(Math.abs(expected[i]), Math.abs(Math.exp(Gamma.logGamma(x[i]))), Math.abs(expected[i] * 1e-5));
    }
  }

  private double gammaInteger(double x) {
    double r = 1;
    for (int i = 2; i < x ; i++) {
      r *= i;
    }
    return r;
  }
}
