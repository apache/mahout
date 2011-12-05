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
package org.apache.mahout.math.stats;

import org.apache.mahout.common.MahoutTestCase;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.Vector;
import org.junit.Test;

public class SamplerTest extends MahoutTestCase {

  @Test
  public void testDiscreteSampler() {
    Vector distribution = new DenseVector(new double[] {1, 0, 2, 3, 5, 0});
    Sampler sampler = new Sampler(RandomUtils.getRandom(), distribution);
    Vector sampledDistribution = distribution.like();
    int i = 0;
    while (i < 100000) {
      int index = sampler.sample();
      sampledDistribution.set(index, sampledDistribution.get(index) + 1);
      i++;
    }
    assertTrue("sampled distribution is far from the original",
        l1Dist(distribution, sampledDistribution) < 1.0e-2);
  }

  private static double l1Dist(Vector v, Vector w) {
    return v.normalize(1.0).minus(w.normalize(1)).norm(1.0);
  }
}
