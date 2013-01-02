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

import com.google.common.collect.Lists;
import org.apache.commons.math3.distribution.PoissonDistribution;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.common.RandomWrapper;

import java.util.List;

/**
 * Samples from a Poisson distribution.  Should probably not be used for lambda > 1000 or so.
 */
public final class PoissonSampler extends AbstractSamplerFunction {

  private double limit;
  private Multinomial<Integer> partial;
  private final RandomWrapper gen;
  private final PoissonDistribution pd;

  public PoissonSampler(double lambda) {
    limit = 1;
    gen = RandomUtils.getRandom();
    pd = new PoissonDistribution(gen.getRandomGenerator(),
                                 lambda,
                                 PoissonDistribution.DEFAULT_EPSILON,
                                 PoissonDistribution.DEFAULT_MAX_ITERATIONS);
  }

  @Override
  public Double sample() {
    return sample(gen.nextDouble());
  }

  double sample(double u) {
    if (u < limit) {
      List<WeightedThing<Integer>> steps = Lists.newArrayList();
      limit = 1;
      int i = 0;
      while (u / 20 < limit) {
        double pdf = pd.probability(i);
        limit -= pdf;
        steps.add(new WeightedThing<Integer>(i, pdf));
        i++;
      }
      steps.add(new WeightedThing<Integer>(steps.size(), limit));
      partial = new Multinomial<Integer>(steps);
    }
    return partial.sample(u);
  }
}
