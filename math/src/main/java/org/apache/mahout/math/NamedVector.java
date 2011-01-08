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

import java.util.Iterator;

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
    if (delegate == null) {
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

  public int hashCode() {
    return delegate.hashCode();
  }

  /**
   * To not break transitivity with other {@link Vector}s, this does not compare name.
   */
  public boolean equals(Object other) {
    return delegate.equals(other);
  }

  @Override
  public NamedVector clone() {
    return new NamedVector(delegate.clone(), name);
  }

  public String asFormatString() {
    return delegate.asFormatString();
  }

  public Vector assign(double value) {
    return delegate.assign(value);
  }

  public Vector assign(double[] values) {
    return delegate.assign(values);
  }

  public Vector assign(Vector other) {
    return delegate.assign(other);
  }

  public Vector assign(DoubleFunction function) {
    return delegate.assign(function);
  }

  public Vector assign(Vector other, DoubleDoubleFunction function) {
    return delegate.assign(other, function);
  }

  public Vector assign(DoubleDoubleFunction f, double y) {
    return delegate.assign(f, y);
  }

  public int size() {
    return delegate.size();
  }

  public boolean isDense() {
    return delegate.isDense();
  }

  public boolean isSequentialAccess() {
    return delegate.isSequentialAccess();
  }

  public Iterator<Element> iterator() {
    return delegate.iterator();
  }

  public Iterator<Element> iterateNonZero() {
    return delegate.iterateNonZero();
  }

  public Element getElement(int index) {
    return delegate.getElement(index);
  }

  public Vector divide(double x) {
    return delegate.divide(x);
  }

  public double dot(Vector x) {
    return delegate.dot(x);
  }

  public double get(int index) {
    return delegate.get(index);
  }

  public double getQuick(int index) {
    return delegate.getQuick(index);
  }

  public NamedVector like() {
    return new NamedVector(delegate.like(), name);
  }

  public Vector minus(Vector x) {
    return delegate.minus(x);
  }

  public Vector normalize() {
    return delegate.normalize();
  }

  public Vector normalize(double power) {
    return delegate.normalize(power);
  }

  public Vector logNormalize() {
    return delegate.logNormalize();
  }

  public Vector logNormalize(double power) {
    return delegate.logNormalize(power);
  }

  public double norm(double power) {
    return delegate.norm(power);
  }

  public double maxValue() {
    return delegate.maxValue();
  }

  public int maxValueIndex() {
    return delegate.maxValueIndex();
  }

  public double minValue() {
    return delegate.minValue();
  }

  public int minValueIndex() {
    return delegate.minValueIndex();
  }

  public Vector plus(double x) {
    return delegate.plus(x);
  }

  public Vector plus(Vector x) {
    return delegate.plus(x);
  }

  public void set(int index, double value) {
    delegate.set(index, value);
  }

  public void setQuick(int index, double value) {
    delegate.setQuick(index, value);
  }

  public int getNumNondefaultElements() {
    return delegate.getNumNondefaultElements();
  }

  public Vector times(double x) {
    return delegate.times(x);
  }

  public Vector times(Vector x) {
    return delegate.times(x);
  }

  public Vector viewPart(int offset, int length) {
    return delegate.viewPart(offset, length);
  }

  public double zSum() {
    return delegate.zSum();
  }

  public Matrix cross(Vector other) {
    return delegate.cross(other);
  }

  public double aggregate(DoubleDoubleFunction aggregator, DoubleFunction map) {
    return delegate.aggregate(aggregator, map);
  }

  public double aggregate(Vector other, DoubleDoubleFunction aggregator, DoubleDoubleFunction combiner) {
    return delegate.aggregate(other, aggregator, combiner);
  }

  public double getLengthSquared() {
    return delegate.getLengthSquared();
  }

  public double getDistanceSquared(Vector v) {
    return delegate.getDistanceSquared(v);
  }

  public void addTo(Vector v) {
    delegate.addTo(v);
  }

}
