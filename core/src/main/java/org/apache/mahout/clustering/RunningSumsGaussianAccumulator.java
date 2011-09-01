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
package org.apache.mahout.clustering;

import org.apache.mahout.math.Vector;
import org.apache.mahout.math.function.Functions;
import org.apache.mahout.math.function.SquareRootFunction;

/**
 * An online Gaussian accumulator that uses a running power sums approach as reported 
 * on http://en.wikipedia.org/wiki/Standard_deviation
 * Suffers from overflow, underflow and roundoff error but has minimal observe-time overhead
 */
public class RunningSumsGaussianAccumulator implements GaussianAccumulator {

  private double s0;
  private Vector s1;
  private Vector s2;
  private Vector mean;
  private Vector std;

  @Override
  public double getN() {
    return s0;
  }

  @Override
  public Vector getMean() {
    return mean;
  }

  @Override
  public Vector getStd() {
    return std;
  }

  @Override
  public double getAverageStd() {
    if (s0 == 0.0) {
      return 0.0;
    } else {
      return std.zSum() / std.size();
    }
  }

  @Override
  public Vector getVariance() {
    return std.times(std);
  }

  @Override
  public void observe(Vector x, double weight) {
    s0 += weight;
    Vector weightedX = x.times(weight);
    if (s1 == null) {
      s1 = weightedX;
    } else {
      s1.assign(weightedX, Functions.PLUS);
    }
    Vector x2 = x.times(x).times(weight);
    if (s2 == null) {
      s2 = x2;
    } else {
      s2.assign(x2, Functions.PLUS);
    }
  }

  @Override
  public void compute() {
    if (s0 != 0.0) {
      mean = s1.divide(s0);
      std = s2.times(s0).minus(s1.times(s1)).assign(new SquareRootFunction()).divide(s0);
    }
  }

}
