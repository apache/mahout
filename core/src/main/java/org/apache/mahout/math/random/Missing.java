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

import java.util.Random;

import org.apache.mahout.common.RandomUtils;

/**
 * Models data with missing values.  Note that all variables with the same fraction of missing
 * values will have the same sequence of missing values.  Similarly, if two variables have
 * missing probabilities of p1 > p2, then all of the p2 missing values will also be missing for
 * p1.
 */
public final class Missing<T> implements Sampler<T> {
  private final Random gen;
  private final double p;
  private final Sampler<T> delegate;
  private final T missingMarker;

  public Missing(int seed, double p, Sampler<T> delegate, T missingMarker) {
    this.p = p;
    this.delegate = delegate;
    this.missingMarker = missingMarker;
    gen = RandomUtils.getRandom(seed);
  }

  public Missing(double p, Sampler<T> delegate, T missingMarker) {
    this(1, p, delegate, missingMarker);
  }

  public Missing(double p, Sampler<T> delegate) {
    this(1, p, delegate, null);
  }

  @Override
  public T sample() {
    if (gen.nextDouble() >= p) {
      return delegate.sample();
    } else {
      return missingMarker;
    }
  }
}
