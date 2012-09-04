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

import org.apache.commons.math.distribution.PoissonDistribution;
import org.apache.commons.math.distribution.PoissonDistributionImpl;
import org.apache.mahout.common.RandomUtils;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

public class PoissonSamplerTest {
    @Before
    public void setUp() {
        RandomUtils.useTestSeed();
    }

    @Test
    public void testBasics() {
        for (double alpha : new double[]{0.1, 1, 10, 100}) {
            checkDistribution(new PoissonSampler(alpha), alpha);
        }
    }

    private void checkDistribution(PoissonSampler pd, double alpha) {
        int[] count = new int[(int) Math.max(10, 5 * alpha)];
        for (int i = 0; i < 10000; i++) {
            count[pd.sample().intValue()]++;
        }

        PoissonDistribution ref = new PoissonDistributionImpl(alpha);
        for (int i = 0; i < count.length; i++) {
            assertEquals(ref.probability(i), count[i] / 10000.0, 2e-2);
        }
    }
}
