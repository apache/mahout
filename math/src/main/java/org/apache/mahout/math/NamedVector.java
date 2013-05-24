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

import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;

public class NamedVector implements Vector {

  private Vector delegate;
  private String name;

  public NamedVector() {
  }

  public NamedVector(NamedVector other) {
    this.delegate = other.getDelegate();
    this.name = other.getName();
  }

  public NamedVector(Vector delegate, String name) {
    if (delegate == null || name == null) {
      throw new IllegalArgumentException();
    }
    this.delegate = delegate;
    this.name = name;
  }

  public String getName() {
    return name;
  }

  public Vector getDelegate() {
    return delegate;
  }

  @Override
  public int hashCode() {
    return delegate.hashCode();
  }

  /**
   * To not break transitivity with other {@link Vector}s, this does not compare name.
   */
  @SuppressWarnings("EqualsWhichDoesntCheckParameterClass")
  @Override
  public boolean equals(Object other) {
    return delegate.equals(other);
  }

  @SuppressWarnings("CloneDoesntCallSuperClone")
  @Override
  public NamedVector clone() {
    return new NamedVector(delegate.clone(), name);
  }

  @Override
  public Iterable<Element> all() {
    return delegate.all();
  }

  @Override
  public Iterable<Element> nonZeroes() {
    return delegate.nonZeroes();
  }

  @Override
  public String asFormatString() {
    return toString();
  }

  @Override
  public String toString() {
    StringBuilder bldr = new StringBuilder();
    bldr.append(name).append(':').append(delegate.toString());
    return bldr.toString();
  }

  @Override
  public Vector assign(double value) {
    return delegate.assign(value);
  }

  @Override
  public Vector assign(double[] values) {
    return delegate.assign(values);
  }

  @Override
  public Vector assign(Vector other) {
    return delegate.assign(other);
  }

  @Override
  public Vector assign(DoubleFunction function) {
    return delegate.assign(function);
  }

  @Override
  public Vector assign(Vector other, DoubleDoubleFunction function) {
    return delegate.assign(other, function);
  }

  @Override
  public Vector assign(DoubleDoubleFunction f, double y) {
    return delegate.assign(f, y);
  }

  @Override
  public int size() {
    return delegate.size();
  }

  @Override
  public boolean isDense() {
    return delegate.isDense();
  }

  @Override
  public boolean isSequentialAccess() {
    return delegate.isSequentialAccess();
  }

  @Override
  public Element getElement(int index) {
    return delegate.getElement(index);
  }

  /**
   * Merge a set of (index, value) pairs into the vector.
   *
   * @param updates an ordered mapping of indices to values to be merged in.
   */
  @Override
  public void mergeUpdates(OrderedIntDoubleMapping updates) {
    delegate.mergeUpdates(updates);
  }

  @Override
  public Vector divide(double x) {
    return delegate.divide(x);
  }

  @Override
  public double dot(Vector x) {
    return delegate.dot(x);
  }

  @Override
  public double get(int index) {
    return delegate.get(index);
  }

  @Override
  public double getQuick(int index) {
    return delegate.getQuick(index);
  }

  @Override
  public NamedVector like() {
    return new NamedVector(delegate.like(), name);
  }

  @Override
  public Vector minus(Vector x) {
    return delegate.minus(x);
  }

  @Override
  public Vector normalize() {
    return delegate.normalize();
  }

  @Override
  public Vector normalize(double power) {
    return delegate.normalize(power);
  }

  @Override
  public Vector logNormalize() {
    return delegate.logNormalize();
  }

  @Override
  public Vector logNormalize(double power) {
    return delegate.logNormalize(power);
  }

  @Override
  public double norm(double power) {
    return delegate.norm(power);
  }

  @Override
  public double maxValue() {
    return delegate.maxValue();
  }

  @Override
  public int maxValueIndex() {
    return delegate.maxValueIndex();
  }

  @Override
  public double minValue() {
    return delegate.minValue();
  }

  @Override
  public int minValueIndex() {
    return delegate.minValueIndex();
  }

  @Override
  public Vector plus(double x) {
    return delegate.plus(x);
  }

  @Override
  public Vector plus(Vector x) {
    return delegate.plus(x);
  }

  @Override
  public void set(int index, double value) {
    delegate.set(index, value);
  }

  @Override
  public void setQuick(int index, double value) {
    delegate.setQuick(index, value);
  }

  @Override
  public void incrementQuick(int index, double increment) {
    delegate.incrementQuick(index, increment);
  }

  @Override
  public int getNumNonZeroElements() {
    return delegate.getNumNonZeroElements();
  }

  @Override
  public int getNumNondefaultElements() {
    return delegate.getNumNondefaultElements();
  }

  @Override
  public Vector times(double x) {
    return delegate.times(x);
  }

  @Override
  public Vector times(Vector x) {
    return delegate.times(x);
  }

  @Override
  public Vector viewPart(int offset, int length) {
    return delegate.viewPart(offset, length);
  }

  @Override
  public double zSum() {
    return delegate.zSum();
  }

  @Override
  public Matrix cross(Vector other) {
    return delegate.cross(other);
  }

  @Override
  public double aggregate(DoubleDoubleFunction aggregator, DoubleFunction map) {
    return delegate.aggregate(aggregator, map);
  }

  @Override
  public double aggregate(Vector other, DoubleDoubleFunction aggregator, DoubleDoubleFunction combiner) {
    return delegate.aggregate(other, aggregator, combiner);
  }

  @Override
  public double getLengthSquared() {
    return delegate.getLengthSquared();
  }

  @Override
  public double getDistanceSquared(Vector v) {
    return delegate.getDistanceSquared(v);
  }

  @Override
  public double getLookupCost() {
    return delegate.getLookupCost();
  }

  @Override
  public double getIteratorAdvanceCost() {
    return delegate.getIteratorAdvanceCost();
  }

  @Override
  public boolean isAddConstantTime() {
    return delegate.isAddConstantTime();
  }
}
