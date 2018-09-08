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

import com.google.common.base.Preconditions;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.list.DoubleArrayList;

import java.util.Random;

/**
 *
 * Generates samples from a generalized Chinese restaurant process (or Pittman-Yor process).
 *
 * The number of values drawn exactly once will asymptotically be equal to the discount parameter
 * as the total number of draws T increases without bound.  The number of unique values sampled will
 * increase as O(alpha * log T) if discount = 0 or O(alpha * T^discount) for discount > 0.
 */
public final class ChineseRestaurant implements Sampler<Integer> {

  private final double alpha;
  private double weight = 0;
  private double discount = 0;
  private final DoubleArrayList weights = new DoubleArrayList();
  private final Random rand = RandomUtils.getRandom();

  /**
   * Constructs a Dirichlet process sampler.  This is done by setting discount = 0.
   * @param alpha  The strength parameter for the Dirichlet process.
   */
  public ChineseRestaurant(double alpha) {
    this(alpha, 0);
  }

  /**
   * Constructs a Pitman-Yor sampler.
   *
   * @param alpha     The strength parameter that drives the number of unique values as a function of draws.
   * @param discount  The discount parameter that drives the percentage of values that occur once in a large sample.
   */
  public ChineseRestaurant(double alpha, double discount) {
    Preconditions.checkArgument(alpha > 0, "Strength Parameter, alpha must be greater then 0!");
    Preconditions.checkArgument(discount >= 0 && discount <= 1, "Must be: 0 <= discount <= 1");
    this.alpha = alpha;
    this.discount = discount;
  }

  @Override
  public Integer sample() {
    double u = rand.nextDouble() * (alpha + weight);
    for (int j = 0; j < weights.size(); j++) {
      // select existing options with probability (w_j - d) / (alpha + w)
      if (u < weights.get(j) - discount) {
        weights.set(j, weights.get(j) + 1);
        weight++;
        return j;
      } else {
        u -= weights.get(j) - discount;
      }
    }

    // if no existing item selected, pick new item with probability (alpha - d*t) / (alpha + w)
    // where t is number of pre-existing cases
    weights.add(1);
    weight++;
    return weights.size() - 1;
  }

  /**
   * @return the number of unique values that have been returned.
   */
  public int size() {
    return weights.size();
  }

  /**
   * @return the number draws so far.
   */
  public int count() {
    return (int) weight;
  }

  /**
   * @param j Which value to test.
   * @return  The number of times that j has been returned so far.
   */
  public int count(int j) {
    Preconditions.checkArgument(j >= 0);

    if (j < weights.size()) {
      return (int) weights.get(j);
    } else {
      return 0;
    }
  }
}
