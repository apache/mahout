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

import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.DenseVector;
import org.apache.mahout.math.MahoutTestCase;
import org.apache.mahout.math.Vector;
import org.apache.mahout.math.stats.OnlineSummarizer;
import org.junit.Before;
import org.junit.Test;

public class MultiNormalTest extends MahoutTestCase {
    @Override
    @Before
    public void setUp() {
        RandomUtils.useTestSeed();
    }

    @Test
    public void testDiagonal() {
        DenseVector offset = new DenseVector(new double[]{6, 3, 0});
        MultiNormal n = new MultiNormal(
                new DenseVector(new double[]{1, 2, 5}), offset);

        OnlineSummarizer[] s = {
                new OnlineSummarizer(),
                new OnlineSummarizer(),
                new OnlineSummarizer()
        };

        OnlineSummarizer[] cross = {
                new OnlineSummarizer(),
                new OnlineSummarizer(),
                new OnlineSummarizer()
        };

        for (int i = 0; i < 10000; i++) {
            Vector v = n.sample();
            for (int j = 0; j < 3; j++) {
                s[j].add(v.get(j) - offset.get(j));
                int k1 = j % 2;
                int k2 = (j + 1) / 2 + 1;
                cross[j].add((v.get(k1) - offset.get(k1)) * (v.get(k2) - offset.get(k2)));
            }
        }

        for (int j = 0; j < 3; j++) {
            assertEquals(0, s[j].getMean() / s[j].getSD(), 0.04);
            assertEquals(0, cross[j].getMean() / cross[j].getSD(), 0.04);
        }
    }


    @Test
    public void testRadius() {
        MultiNormal gen = new MultiNormal(0.1, new DenseVector(10));
        OnlineSummarizer s = new OnlineSummarizer();
        for (int i = 0; i < 10000; i++) {
            double x = gen.sample().norm(2) / Math.sqrt(10);
            s.add(x);
        }
        assertEquals(0.1, s.getMean(), 0.01);

    }
}
