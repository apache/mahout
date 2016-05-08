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

import com.google.common.base.Preconditions;
import org.apache.mahout.common.RandomUtils;
import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.Functions;

/** Implementations of generic capabilities like sum of elements and dot products */
public abstract class AbstractVector implements Vector, LengthCachingVector {

  private int size;
  protected double lengthSquared = -1.0;

  protected AbstractVector(int size) {
    this.size = size;
  }

  @Override
  public Iterable<Element> all() {
    return new Iterable<Element>() {
      @Override
      public Iterator<Element> iterator() {
        return AbstractVector.this.iterator();
      }
    };
  }

  @Override
  public Iterable<Element> nonZeroes() {
    return new Iterable<Element>() {
      @Override
      public Iterator<Element> iterator() {
        return iterateNonZero();
      }
    };
  }

  /**
   * Iterates over all elements <p/> * NOTE: Implementations may choose to reuse the Element returned for performance
   * reasons, so if you need a copy of it, you should call {@link #getElement(int)} for the given index
   *
   * @return An {@link Iterator} over all elements
   */
  protected abstract Iterator<Element> iterator();

  /**
   * Iterates over all non-zero elements. <p/> NOTE: Implementations may choose to reuse the Element returned for
   * performance reasons, so if you need a copy of it, you should call {@link #getElement(int)} for the given index
   *
   * @return An {@link Iterator} over all non-zero elements
   */
  protected abstract Iterator<Element> iterateNonZero();
  /**
   * Aggregates a vector by applying a mapping function fm(x) to every component and aggregating
   * the results with an aggregating function fa(x, y).
   *
   * @param aggregator used to combine the current value of the aggregation with the result of map.apply(nextValue)
   * @param map a function to apply to each element of the vector in turn before passing to the aggregator
   * @return the result of the aggregation
   */
  @Override
  public double aggregate(DoubleDoubleFunction aggregator, DoubleFunction map) {
    if (size == 0) {
      return 0;
    }

    // If the aggregator is associative and commutative and it's likeLeftMult (fa(0, y) = 0), and there is
    // at least one zero in the vector (size > getNumNondefaultElements) and applying fm(0) = 0, the result
    // gets cascaded through the aggregation and the final result will be 0.
    if (aggregator.isAssociativeAndCommutative() && aggregator.isLikeLeftMult()
        && size > getNumNondefaultElements() && !map.isDensifying()) {
      return 0;
    }

    double result;
    if (isSequentialAccess() || aggregator.isAssociativeAndCommutative()) {
      Iterator<Element> iterator;
      // If fm(0) = 0 and fa(x, 0) = x, we can skip all zero values.
      if (!map.isDensifying() && aggregator.isLikeRightPlus()) {
        iterator = iterateNonZero();
        if (!iterator.hasNext()) {
          return 0;
        }
      } else {
        iterator = iterator();
      }
      Element element = iterator.next();
      result = map.apply(element.get());
      while (iterator.hasNext()) {
        element = iterator.next();
        result = aggregator.apply(result, map.apply(element.get()));
      }
    } else {
      result = map.apply(getQuick(0));
      for (int i = 1; i < size; i++) {
        result = aggregator.apply(result, map.apply(getQuick(i)));
      }
    }

    return result;
  }

  @Override
  public double aggregate(Vector other, DoubleDoubleFunction aggregator, DoubleDoubleFunction combiner) {
    Preconditions.checkArgument(size == other.size(), "Vector sizes differ");
    if (size == 0) {
      return 0;
    }
    return VectorBinaryAggregate.aggregateBest(this, other, aggregator, combiner);
  }

  /**
   * Subclasses must override to return an appropriately sparse or dense result
   *
   * @param rows    the row cardinality
   * @param columns the column cardinality
   * @return a Matrix
   */
  protected abstract Matrix matrixLike(int rows, int columns);

  @Override
  public Vector viewPart(int offset, int length) {
    if (offset < 0) {
      throw new IndexException(offset, size);
    }
    if (offset + length > size) {
      throw new IndexException(offset + length, size);
    }
    return new VectorView(this, offset, length);
  }

  @SuppressWarnings("CloneDoesntDeclareCloneNotSupportedException")
  @Override
  public Vector clone() {
    try {
      AbstractVector r = (AbstractVector) super.clone();
      r.size = size;
      r.lengthSquared = lengthSquared;
      return r;
    } catch (CloneNotSupportedException e) {
      throw new IllegalStateException("Can't happen");
    }
  }

  @Override
  public Vector divide(double x) {
    if (x == 1.0) {
      return clone();
    }
    Vector result = createOptimizedCopy();
    for (Element element : result.nonZeroes()) {
      element.set(element.get() / x);
    }
    return result;
  }

  @Override
  public double dot(Vector x) {
    if (size != x.size()) {
      throw new CardinalityException(size, x.size());
    }
    if (this == x) {
      return getLengthSquared();
    }
    return aggregate(x, Functions.PLUS, Functions.MULT);
  }

  protected double dotSelf() {
    return aggregate(Functions.PLUS, Functions.pow(2));
  }

  @Override
  public double get(int index) {
    if (index < 0 || index >= size) {
      throw new IndexException(index, size);
    }
    return getQuick(index);
  }

  @Override
  public Element getElement(int index) {
    return new LocalElement(index);
  }

  @Override
  public Vector normalize() {
    return divide(Math.sqrt(getLengthSquared()));
  }

  @Override
  public Vector normalize(double power) {
    return divide(norm(power));
  }

  @Override
  public Vector logNormalize() {
    return logNormalize(2.0, Math.sqrt(getLengthSquared()));
  }

  @Override
  public Vector logNormalize(double power) {
    return logNormalize(power, norm(power));
  }

  public Vector logNormalize(double power, double normLength) {
    // we can special case certain powers
    if (Double.isInfinite(power) || power <= 1.0) {
      throw new IllegalArgumentException("Power must be > 1 and < infinity");
    } else {
      double denominator = normLength * Math.log(power);
      Vector result = createOptimizedCopy();
      for (Element element : result.nonZeroes()) {
        element.set(Math.log1p(element.get()) / denominator);
      }
      return result;
    }
  }

  @Override
  public double norm(double power) {
    if (power < 0.0) {
      throw new IllegalArgumentException("Power must be >= 0");
    }
    // We can special case certain powers.
    if (Double.isInfinite(power)) {
      return aggregate(Functions.MAX, Functions.ABS);
    } else if (power == 2.0) {
      return Math.sqrt(getLengthSquared());
    } else if (power == 1.0) {
      double result = 0.0;
      Iterator<Element> iterator = this.iterateNonZero();
      while (iterator.hasNext()) {
        result += Math.abs(iterator.next().get());
      }
      return result;
      // TODO: this should ideally be used, but it's slower.
      // return aggregate(Functions.PLUS, Functions.ABS);
    } else if (power == 0.0) {
      return getNumNonZeroElements();
    } else {
      return Math.pow(aggregate(Functions.PLUS, Functions.pow(power)), 1.0 / power);
    }
  }

  @Override
  public double getLengthSquared() {
    if (lengthSquared >= 0.0) {
      return lengthSquared;
    }
    return lengthSquared = dotSelf();
  }

  @Override
  public void invalidateCachedLength() {
    lengthSquared = -1;
  }

  @Override
  public double getDistanceSquared(Vector that) {
    if (size != that.size()) {
      throw new CardinalityException(size, that.size());
    }
    double thisLength = getLengthSquared();
    double thatLength = that.getLengthSquared();
    double dot = dot(that);
    double distanceEstimate = thisLength + thatLength - 2 * dot;
    if (distanceEstimate > 1.0e-3 * (thisLength + thatLength)) {
      // The vectors are far enough from each other that the formula is accurate.
      return Math.max(distanceEstimate, 0);
    } else {
      return aggregate(that, Functions.PLUS, Functions.MINUS_SQUARED);
    }
  }

  @Override
  public double maxValue() {
    if (size == 0) {
      return Double.NEGATIVE_INFINITY;
    }
    return aggregate(Functions.MAX, Functions.IDENTITY);
  }

  @Override
  public int maxValueIndex() {
    int result = -1;
    double max = Double.NEGATIVE_INFINITY;
    int nonZeroElements = 0;
    Iterator<Element> iter = this.iterateNonZero();
    while (iter.hasNext()) {
      nonZeroElements++;
      Element element = iter.next();
      double tmp = element.get();
      if (tmp > max) {
        max = tmp;
        result = element.index();
      }
    }
    // if the maxElement is negative and the vector is sparse then any
    // unfilled element(0.0) could be the maxValue hence we need to
    // find one of those elements
    if (nonZeroElements < size && max < 0.0) {
      for (Element element : all()) {
        if (element.get() == 0.0) {
          return element.index();
        }
      }
    }
    return result;
  }

  @Override
  public double minValue() {
    if (size == 0) {
      return Double.POSITIVE_INFINITY;
    }
    return aggregate(Functions.MIN, Functions.IDENTITY);
  }

  @Override
  public int minValueIndex() {
    int result = -1;
    double min = Double.POSITIVE_INFINITY;
    int nonZeroElements = 0;
    Iterator<Element> iter = this.iterateNonZero();
    while (iter.hasNext()) {
      nonZeroElements++;
      Element element = iter.next();
      double tmp = element.get();
      if (tmp < min) {
        min = tmp;
        result = element.index();
      }
    }
    // if the maxElement is positive and the vector is sparse then any
    // unfilled element(0.0) could be the maxValue hence we need to
    // find one of those elements
    if (nonZeroElements < size && min > 0.0) {
      for (Element element : all()) {
        if (element.get() == 0.0) {
          return element.index();
        }
      }
    }
    return result;
  }

  @Override
  public Vector plus(double x) {
    Vector result = createOptimizedCopy();
    if (x == 0.0) {
      return result;
    }
    return result.assign(Functions.plus(x));
  }

  @Override
  public Vector plus(Vector that) {
    if (size != that.size()) {
      throw new CardinalityException(size, that.size());
    }
    return createOptimizedCopy().assign(that, Functions.PLUS);
  }

  @Override
  public Vector minus(Vector that) {
    if (size != that.size()) {
      throw new CardinalityException(size, that.size());
    }
    return createOptimizedCopy().assign(that, Functions.MINUS);
  }

  @Override
  public void set(int index, double value) {
    if (index < 0 || index >= size) {
      throw new IndexException(index, size);
    }
    setQuick(index, value);
  }

  @Override
  public void incrementQuick(int index, double increment) {
    setQuick(index, getQuick(index) + increment);
  }

  @Override
  public Vector times(double x) {
    if (x == 0.0) {
      return like();
    }
    return createOptimizedCopy().assign(Functions.mult(x));
  }

  /**
   * Copy the current vector in the most optimum fashion. Used by immutable methods like plus(), minus().
   * Use this instead of vector.like().assign(vector). Sub-class can choose to override this method.
   *
   * @return a copy of the current vector.
   */
  protected Vector createOptimizedCopy() {
    return createOptimizedCopy(this);
  }

  private static Vector createOptimizedCopy(Vector vector) {
    Vector result;
    if (vector.isDense()) {
      result = vector.like().assign(vector, Functions.SECOND_LEFT_ZERO);
    } else {
      result = vector.clone();
    }
    return result;
  }

  @Override
  public Vector times(Vector that) {
    if (size != that.size()) {
      throw new CardinalityException(size, that.size());
    }

    if (this.getNumNondefaultElements() <= that.getNumNondefaultElements()) {
      return createOptimizedCopy(this).assign(that, Functions.MULT);
    } else {
      return createOptimizedCopy(that).assign(this, Functions.MULT);
    }
  }

  @Override
  public double zSum() {
    return aggregate(Functions.PLUS, Functions.IDENTITY);
  }

  @Override
  public int getNumNonZeroElements() {
    int count = 0;
    Iterator<Element> it = iterateNonZero();
    while (it.hasNext()) {
      if (it.next().get() != 0.0) {
        count++;
      }
    }
    return count;
  }

  @Override
  public Vector assign(double value) {
    Iterator<Element> it;
    if (value == 0.0) {
      // Make all the non-zero values 0.
      it = iterateNonZero();
      while (it.hasNext()) {
        it.next().set(value);
      }
    } else {
      if (isSequentialAccess() && !isAddConstantTime()) {
        // Update all the non-zero values and queue the updates for the zero vaues.
        // The vector will become dense.
        it = iterator();
        OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
        while (it.hasNext()) {
          Element element = it.next();
          if (element.get() == 0.0) {
            updates.set(element.index(), value);
          } else {
            element.set(value);
          }
        }
        mergeUpdates(updates);
      } else {
        for (int i = 0; i < size; ++i) {
          setQuick(i, value);
        }
      }
    }
    invalidateCachedLength();
    return this;
  }

  @Override
  public Vector assign(double[] values) {
    if (size != values.length) {
      throw new CardinalityException(size, values.length);
    }
    if (isSequentialAccess() && !isAddConstantTime()) {
      OrderedIntDoubleMapping updates = new OrderedIntDoubleMapping();
      Iterator<Element> it = iterator();
      while (it.hasNext()) {
        Element element = it.next();
        int index = element.index();
        if (element.get() == 0.0) {
          updates.set(index, values[index]);
        } else {
          element.set(values[index]);
        }
      }
      mergeUpdates(updates);
    } else {
      for (int i = 0; i < size; ++i) {
        setQuick(i, values[i]);
      }
    }
    invalidateCachedLength();
    return this;
  }

  @Override
  public Vector assign(Vector other) {
    return assign(other, Functions.SECOND);
  }

  @Override
  public Vector assign(DoubleDoubleFunction f, double y) {
    Iterator<Element> iterator = f.apply(0, y) == 0 ? iterateNonZero() : iterator();
    while (iterator.hasNext()) {
      Element element = iterator.next();
      element.set(f.apply(element.get(), y));
    }
    invalidateCachedLength();
    return this;
  }

  @Override
  public Vector assign(DoubleFunction f) {
    Iterator<Element> iterator = !f.isDensifying() ? iterateNonZero() : iterator();
    while (iterator.hasNext()) {
      Element element = iterator.next();
      element.set(f.apply(element.get()));
    }
    invalidateCachedLength();
    return this;
  }

  @Override
  public Vector assign(Vector other, DoubleDoubleFunction function) {
    if (size != other.size()) {
      throw new CardinalityException(size, other.size());
    }
    VectorBinaryAssign.assignBest(this, other, function);
    invalidateCachedLength();
    return this;
  }

  @Override
  public Matrix cross(Vector other) {
    Matrix result = matrixLike(size, other.size());
    Iterator<Vector.Element> it = iterateNonZero();
    while (it.hasNext()) {
      Vector.Element e = it.next();
      int row = e.index();
      result.assignRow(row, other.times(getQuick(row)));
    }
    return result;
  }

  @Override
  public final int size() {
    return size;
  }

  @Override
  public String asFormatString() {
    return toString();
  }

  @Override
  public int hashCode() {
    int result = size;
    Iterator<Element> iter = iterateNonZero();
    while (iter.hasNext()) {
      Element ele = iter.next();
      result += ele.index() * RandomUtils.hashDouble(ele.get());
    }
    return result;
  }

  /**
   * Determines whether this {@link Vector} represents the same logical vector as another
   * object. Two {@link Vector}s are equal (regardless of implementation) if the value at
   * each index is the same, and the cardinalities are the same.
   */
  @Override
  public boolean equals(Object o) {
    if (this == o) {
      return true;
    }
    if (!(o instanceof Vector)) {
      return false;
    }
    Vector that = (Vector) o;
    return size == that.size() && aggregate(that, Functions.PLUS, Functions.MINUS_ABS) == 0.0;
  }

  @Override
  public String toString() {
    return toString(null);
  }

  public String toString(String[] dictionary) {
    StringBuilder result = new StringBuilder();
    result.append('{');
    for (int index = 0; index < size; index++) {
      double value = getQuick(index);
      if (value != 0.0) {
        result.append(dictionary != null && dictionary.length > index ? dictionary[index] : index);
        result.append(':');
        result.append(value);
        result.append(',');
      }
    }
    if (result.length() > 1) {
      result.setCharAt(result.length() - 1, '}');
    } else {
      result.append('}');
    }
    return result.toString();
  }

  /**
   * toString() implementation for sparse vectors via {@link #nonZeroes()} method
   * @return String representation of the vector
   */
  public String sparseVectorToString() {
    Iterator<Element> it = iterateNonZero();
    if (!it.hasNext()) {
      return "{}";
    }
    else {
      StringBuilder result = new StringBuilder();
      result.append('{');
      while (it.hasNext()) {
        Vector.Element e = it.next();
        result.append(e.index());
        result.append(':');
        result.append(e.get());
        result.append(',');
      }
      result.setCharAt(result.length() - 1, '}');
      return result.toString();
    }
  }

  protected final class LocalElement implements Element {
    int index;

    LocalElement(int index) {
      this.index = index;
    }

    @Override
    public double get() {
      return getQuick(index);
    }

    @Override
    public int index() {
      return index;
    }

    @Override
    public void set(double value) {
      setQuick(index, value);
    }
  }
}
