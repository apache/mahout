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

package org.apache.mahout.math;

import org.apache.mahout.math.function.Functions;

/**
 * A centroid is a weighted vector.  We have it delegate to the vector itself for lots of operations
 * to make it easy to use vector search classes and such.
 */
public class Centroid extends WeightedVector {
  public Centroid(WeightedVector original) {
    super(original.getVector().like().assign(original), original.getWeight(), original.getIndex());
  }

  public Centroid(int key, Vector initialValue) {
    super(initialValue, 1, key);
  }

  public Centroid(int key, Vector initialValue, double weight) {
    super(initialValue, weight, key);
  }

  public static Centroid create(int key, Vector initialValue) {
    if (initialValue instanceof WeightedVector) {
      return new Centroid(key, new DenseVector(initialValue), ((WeightedVector) initialValue).getWeight());
    } else {
      return new Centroid(key, new DenseVector(initialValue), 1);
    }
  }

  public void update(Vector v) {
    if (v instanceof Centroid) {
      Centroid c = (Centroid) v;
      update(c.delegate, c.getWeight());
    } else {
      update(v, 1);
    }
  }

  public void update(Vector other, final double wy) {
    final double wx = getWeight();
    delegate.assign(other, Functions.reweigh(wx, wy));
    setWeight(wx + wy);
  }

  @Override
  public Centroid like() {
    return new Centroid(getIndex(), getVector().like(), getWeight());
  }

  /**
   * Gets the index of this centroid.  Use getIndex instead to maintain standard names.
   */
  @Deprecated
  public int getKey() {
    return getIndex();
  }

  public void addWeight(double newWeight) {
    setWeight(getWeight() + newWeight);
  }

  @Override
  public String toString() {
    return String.format("key = %d, weight = %.2f, vector = %s", getIndex(), getWeight(), delegate);
  }

  @SuppressWarnings("CloneDoesntCallSuperClone")
  @Override
  public Centroid clone() {
    return new Centroid(this);
  }
}
