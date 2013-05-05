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

/**
 * Handy for creating multinomial distributions of things.
 */
public final class WeightedThing<T> implements Comparable<WeightedThing<T>> {
  private double weight;
  private final T value;

  public WeightedThing(T thing, double weight) {
    this.value = Preconditions.checkNotNull(thing);
    this.weight = weight;
  }

  public WeightedThing(double weight) {
    this.value = null;
    this.weight = weight;
  }

  public T getValue() {
    return value;
  }

  public double getWeight() {
    return weight;
  }

  public void setWeight(double weight) {
    this.weight = weight;
  }

  @Override
  public int compareTo(WeightedThing<T> other) {
    return Double.compare(this.weight, other.weight);
  }

  @Override
  public boolean equals(Object o) {
    if (o instanceof WeightedThing) {
      @SuppressWarnings("unchecked")
      WeightedThing<T> other = (WeightedThing<T>) o;
      return weight == other.weight && value.equals(other.value);
    }
    return false;
  }

  @Override
  public int hashCode() {
    return 31 * RandomUtils.hashDouble(weight) + value.hashCode();
  }
}
