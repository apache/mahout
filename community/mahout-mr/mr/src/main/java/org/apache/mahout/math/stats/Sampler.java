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

package org.apache.mahout.math.stats;

import com.google.common.base.Preconditions;
import org.apache.mahout.math.Vector;

import java.util.Arrays;
import java.util.Random;

/**
 * Discrete distribution sampler:
 *
 * Samples from a given discrete distribution: you provide a source of randomness and a Vector
 * (cardinality N) which describes a distribution over [0,N), and calls to sample() sample
 * from 0 to N using this distribution
 */
public class Sampler {

  private final Random random;
  private final double[] sampler;

  public Sampler(Random random) {
    this.random = random;
    sampler = null;
  }

  public Sampler(Random random, double[] sampler) {
    this.random = random;
    this.sampler = sampler;
  }

  public Sampler(Random random, Vector distribution) {
    this.random = random;
    this.sampler = samplerFor(distribution);
  }

  public int sample(Vector distribution) {
    return sample(samplerFor(distribution));
  }

  public int sample() {
    Preconditions.checkNotNull(sampler,
      "Sampler must have been constructed with a distribution, or else sample(Vector) should be used to sample");
    return sample(sampler);
  }

  private static double[] samplerFor(Vector vectorDistribution) {
    int size = vectorDistribution.size();
    double[] partition = new double[size];
    double norm = vectorDistribution.norm(1);
    double sum = 0;
    for (int i = 0; i < size; i++) {
      sum += vectorDistribution.get(i) / norm;
      partition[i] = sum;
    }
    return partition;
  }

  private int sample(double[] sampler) {
    int index = Arrays.binarySearch(sampler, random.nextDouble());
    return index < 0 ? -(index + 1) : index;
  }
}
