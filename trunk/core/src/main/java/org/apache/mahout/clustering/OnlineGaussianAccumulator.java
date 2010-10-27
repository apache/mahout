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
import org.apache.mahout.math.function.SquareRootFunction;

/**
 * An online Gaussian statistics accumulator based upon Knuth (who cites Welford) which is declared to be
 * numerically-stable. See http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
 */
public class OnlineGaussianAccumulator implements GaussianAccumulator {
  private double n = 0;

  private Vector mean;

  private Vector m2;

  private Vector variance;

  @Override
  public double getN() {
    return n;
  }

  @Override
  public Vector getMean() {
    return mean;
  }

  @Override
  public Vector getStd() {
    return variance.clone().assign(new SquareRootFunction());
  }

  @Override
  public void observe(Vector x) {
    n++;
    Vector delta;
    if (mean != null) {
      delta = x.minus(mean);
    } else {
      mean = x.like();
      delta = x.clone();
    }
    mean = mean.plus(delta.divide(n));
    if (m2 != null) {
      m2 = m2.plus(delta.times(x.minus(mean)));
    } else {
      m2 = delta.times(x.minus(mean));
    }
    variance = m2.divide(n - 1);
  }

  @Override
  public void compute() {
    // nothing to do here!
  }

  @Override
  public double getAverageStd() {
    if (n == 0) {
      return 0;
    } else {
      Vector std = getStd();
      return std.zSum() / std.size();
    }
  }

  @Override
  public Vector getVariance() {
    return variance;
  }

}
