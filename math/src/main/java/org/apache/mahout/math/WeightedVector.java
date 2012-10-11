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

/**
 * Decorates a vector with a floating point weight and an index.
 */
public class WeightedVector extends DelegatingVector implements Comparable<WeightedVector> {
  private static final int INVALID_INDEX = -1;
  private double weight;
  private int index;

  protected WeightedVector(double weight, int index) {
    super();
    this.weight = weight;
    this.index = index;
  }

  public WeightedVector(Vector v, double weight, int index) {
    super(v);
    this.weight = weight;
    this.index = index;
  }

  public WeightedVector(Vector v, Vector projection, int index) {
    super(v);
    this.index = index;
    this.weight = v.dot(projection);
  }

  public static WeightedVector project(Vector v, Vector projection) {
    return project(v, projection, INVALID_INDEX);
  }

  public static WeightedVector project(Vector v, Vector projection, int index) {
    return new WeightedVector(v, projection, index);
  }

  public double getWeight() {
    return weight;
  }


  @Override
  public int compareTo(WeightedVector other) {
    if (this == other) {
      return 0;
    }
    int r = Double.compare(weight, other.getWeight());
    if (r == 0 || Math.abs(weight - other.getWeight()) < 1e-8) {
      double diff = this.minus(other).norm(1);
      if (diff < 1e-12) {
        return 0;
      } else {
        for (Vector.Element element : this) {
          r = Double.compare(element.get(), other.get(element.index()));
          if (r != 0) {
            return r;
          }
        }
        return 0;
      }
    } else {
      return r;
    }
  }

  public int getIndex() {
    return index;
  }

  public void setWeight(double newWeight) {
    this.weight = newWeight;
  }

  public void setIndex(int index) {
    this.index = index;
  }

  @Override
  public Vector like() {
    return new WeightedVector(getVector().like(), weight, index);
  }

  @Override
  public String toString() {
    return String.format("index=%d, weight=%.2f, v=%s", index, weight, getVector());
  }
}
