/**
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

import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

import org.apache.mahout.math.function.BinaryFunction;
import org.apache.mahout.math.function.UnaryFunction;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;

/** Implementations of generic capabilities like sum of elements and dot products */
public abstract class AbstractVector implements Vector {

  /**
   * User-settable mapping between String labels and Integer indices. Marked transient so that it will not be serialized
   * with each vector instance.
   */
  private transient Map<String, Integer> bindings;

  private String name;

  protected int size;

  protected double lengthSquared = -1.0;

  protected AbstractVector() {
    this(null, 0);
  }

  protected AbstractVector(String name) {
    this(name, 0);
  }

  protected AbstractVector(String name, int size) {
    this.name = name;
    this.size = size;
  }

  public double aggregate(BinaryFunction aggregator, UnaryFunction map) {
    double result = 0.0;
    int size = size();
    for(int i=0; i<size; i++) {
      result = aggregator.apply(result, map.apply(getQuick(i)) );
    }
    return result;
  }

  public double aggregate(Vector other, BinaryFunction aggregator, BinaryFunction combiner) {
    double result = 0.0;
    int size = size();
    for(int i=0; i<size; i++) {
      result = aggregator.apply(result, combiner.apply(getQuick(i), other.getQuick(i)));
    }
    return result;
  }

  /**
   * Subclasses must override to return an appropriately sparse or dense result
   *
   * @param rows    the row cardinality
   * @param columns the column cardinality
   * @return a Matrix
   */
  protected abstract Matrix matrixLike(int rows, int columns);

  public Vector viewPart(int offset, int length) {
    if (length > size) {
      throw new CardinalityException();
    }
    if (offset < 0 || offset + length > size) {
      throw new IndexException();
    }
    return new VectorView(this, offset, length);
  }

  @Override
  public Vector clone() {
    AbstractVector clone;
    try {
      clone = (AbstractVector) super.clone();
    } catch (CloneNotSupportedException cnse) {
      throw new IllegalStateException(cnse); // Can't happen
    }
    if (bindings != null) {
      clone.bindings = (Map<String, Integer>) ((HashMap<String, Integer>) bindings).clone();
    }
    // name is OK
    return clone;
  }

  public Vector divide(double x) {
    Vector result = clone();
    Iterator<Element> iter = result.iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      element.set(element.get() / x);
    }
    return result;
  }

  public double dot(Vector x) {
    if (size() != x.size()) {
      throw new CardinalityException(size(), x.size());
    }
    if (this == x) {
      return dotSelf();
    }
    double result = 0.0;
    Iterator<Element> iter = iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      result += element.get() * x.getQuick(element.index());
    }
    return result;
  }
  
  public double dotSelf() {
    double result = 0.0;
    if (this instanceof DenseVector) {
      for (int i = 0; i < size(); i++) {
        double value = this.getQuick(i);
        result += value * value;
      }
      return result;
    } else {
      Iterator<Element> iter = iterateNonZero();
      while (iter.hasNext()) {
        double value = iter.next().get();
        result += value * value;
      }
      return result;
    }
  }

  public double get(int index) {
    if (index >= 0 && index < size()) {
      return getQuick(index);
    } else {
      throw new IndexException();
    }
  }

  public Vector minus(Vector x) {
    if (size() != x.size()) {
      throw new CardinalityException();
    }
    if (x instanceof RandomAccessSparseVector || x instanceof DenseVector) {
      // TODO: if both are RandomAccess check the numNonDefault to determine which to iterate
      Vector result = x.clone();
      Iterator<Element> iter = iterateNonZero();
      while (iter.hasNext()) {
        Element e = iter.next();
        result.setQuick(e.index(), e.get() - result.getQuick(e.index()));
      }
      return result;
    } else { // TODO: check the numNonDefault elements to further optimize 
      Vector result = clone();
      Iterator<Element> iter = x.iterateNonZero();
      while (iter.hasNext()) {
        Element e = iter.next();
        result.setQuick(e.index(), getQuick(e.index()) - e.get());
      }
      return result;
    }
  }

  public Vector normalize() {
    return divide(Math.sqrt(dotSelf()));
  }

  public Vector normalize(double power) {
    return divide(norm(power));
  }

  public double norm(double power) {
    if (power < 0.0) {
      throw new IllegalArgumentException("Power must be >= 0");
    }
    // we can special case certain powers
    if (Double.isInfinite(power)) {
      double val = 0.0;
      Iterator<Element> iter = this.iterateNonZero();
      while (iter.hasNext()) {
        val = Math.max(val, Math.abs(iter.next().get()));
      }
      return val;
    } else if (power == 2.0) {
      return Math.sqrt(dotSelf());
    } else if (power == 1.0) {
      double val = 0.0;
      Iterator<Element> iter = this.iterateNonZero();
      while (iter.hasNext()) {
        val += Math.abs(iter.next().get());
      }
      return val;
    } else if (power == 0.0) {
      // this is the number of non-zero elements
      double val = 0.0;
      Iterator<Element> iter = this.iterateNonZero();
      while (iter.hasNext()) {
        val += iter.next().get() == 0 ? 0 : 1;
      }
      return val;
    } else {
      double val = 0.0;
      Iterator<Element> iter = this.iterateNonZero();
      while (iter.hasNext()) {
        Element element = iter.next();
        val += Math.pow(element.get(), power);
      }
      return Math.pow(val, 1.0 / power);
    }
  }

  public double getLengthSquared() {
    if (lengthSquared >= 0.0) {
      return lengthSquared;
    }
    return lengthSquared = dotSelf();
  }

  public double getDistanceSquared(Vector v) {
    if(v.size() != size()) {
      throw new CardinalityException();
    }
    // if this and v has a cached lengthSquared, dot product is quickest way to compute this.
    if(lengthSquared >= 0 && v instanceof AbstractVector && ((AbstractVector)v).lengthSquared >= 0) {
      return lengthSquared + v.getLengthSquared() - 2 * this.dot(v);
    }
    Vector randomlyAccessed;
    Iterator<Element> it;
    double d = 0.0;
    if (lengthSquared >= 0.0) {
      it = v.iterateNonZero();
      randomlyAccessed = this;
      d += lengthSquared;
    } else { // TODO: could be further optimized, figure out which one is smaller, etc
      it = iterateNonZero();
      randomlyAccessed = v;
      d += v.getLengthSquared();
    }
    while(it.hasNext()) {
      Element e = it.next();
      double value = e.get();
      d += value * (value - 2.0 * randomlyAccessed.getQuick(e.index()));
    }
    assert(d > -1.0e-9); // round-off errors should never be too far off!
    return Math.abs(d);
  }

  public double maxValue() {
    double result = Double.NEGATIVE_INFINITY;
    int nonZeroElements = 0;
    Iterator<Element> iter = this.iterateNonZero();
    while (iter.hasNext()) {
      nonZeroElements++;
      Element element = iter.next();
      result = Math.max(result, element.get());
    }
    if (nonZeroElements < size()) {
      return Math.max(result, 0.0);
    }
    return result;
  }
  
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
    // unfilled element(0.0) could be the maxValue hence return -1;
    if (nonZeroElements < size() && max < 0.0) {
      iter = this.iterateAll();
      while (iter.hasNext()) {
        Element element = iter.next();
        double tmp = element.get();
        if (tmp == 0.0) {
          return element.index();
        }
      }
    }
    return result;
  }

  public Vector plus(double x) {
    Vector result = clone();
    int size = result.size();
    for (int i = 0; i < size; i++) {
      result.setQuick(i, getQuick(i) + x);
    }
    return result;
  }

  public Vector plus(Vector x) {
    if (size() != x.size()) {
      throw new CardinalityException();
    }
    //TODO: get smarter about this, if we are adding a dense to a sparse, then we should return a dense
    Vector result = clone();
    Iterator<Element> iter = x.iterateNonZero();
    while (iter.hasNext()) {
      Element e = iter.next();
      int index = e.index();
      result.setQuick(index, getQuick(index) + e.get());
    }
    return result;
  }

  public void addTo(Vector v) {
    Iterator<Element> it = iterateNonZero();
    while(it.hasNext() ) {
      Element e = it.next();
      int index = e.index();
      v.setQuick(index, v.getQuick(index) + e.get());
    }
  }

  public void set(int index, double value) {
    if (index >= 0 && index < size()) {
      setQuick(index, value);
    } else {
      throw new IndexException();
    }
  }

  public Vector times(double x) {
    Vector result = clone();
    Iterator<Element> iter = result.iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      element.set(element.get() * x);
    }

    return result;
  }

  public Vector times(Vector x) {
    if (size() != x.size()) {
      throw new CardinalityException();
    }
    Vector result = clone();
    Iterator<Element> iter = result.iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      element.set(element.get() * x.getQuick(element.index()));
    }

    return result;
  }

  public double zSum() {
    double result = 0.0;
    Iterator<Element> iter = iterateNonZero();
    while (iter.hasNext()) {
      result += iter.next().get();
    }

    return result;
  }

  public Vector assign(double value) {
    int size = size();
    for (int i = 0; i < size; i++) {
      setQuick(i, value);
    }
    return this;
  }

  public Vector assign(double[] values) {
    if (values.length != size()) {
      throw new CardinalityException();
    }
    int size = size();
    for (int i = 0; i < size; i++) {
      setQuick(i, values[i]);
    }
    return this;
  }

  public Vector assign(Vector other) {
    if (other.size() != size()) {
      throw new CardinalityException();
    }
    int size = size();
    for (int i = 0; i < size; i++) {
      setQuick(i, other.getQuick(i));
    }
    return this;
  }

  public Vector assign(BinaryFunction f, double y) {
    Iterator<Element> it;
    if(f.apply(0, y) == 0) {
      it = iterateNonZero();
    } else {
      it = iterateAll();
    }
    while(it.hasNext()) {
      Element e = it.next();
      e.set(f.apply(e.get(), y));
    }
    return this;
  }

  public Vector assign(UnaryFunction function) {
    Iterator<Element> it;
    if(function.apply(0) == 0) {
      it = iterateNonZero();
    } else {
      it = iterateAll();
    }
    while(it.hasNext()) {
      Element e = it.next();
      e.set(function.apply(e.get()));
    }
    return this;
  }

  public Vector assign(Vector other, BinaryFunction function) {
    if (other.size() != size()) {
      throw new CardinalityException();
    }
    for (int i = 0; i < size(); i++) {
      setQuick(i, function.apply(getQuick(i), other.getQuick(i)));
    }
    return this;
  }

  public Matrix cross(Vector other) {
    Matrix result = matrixLike(size(), other.size());
    for (int row = 0; row < size(); row++) {
      result.assignRow(row, other.times(getQuick(row)));
    }
    return result;
  }

  /**
   * Decodes a point from its string representation.
   *
   * @param formattedString a formatted String produced by asFormatString. Note the payload remainder: it is optional,
   *                        but can be present.
   * @return the n-dimensional point
   */
  public static Vector decodeVector(String formattedString) {
    Type vectorType = new TypeToken<Vector>() {
    }.getType();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(vectorType, new JsonVectorAdapter());
    Gson gson = builder.create();
    return gson.fromJson(formattedString, vectorType);
  }

  public String getName() {
    return name;
  }

  public void setName(String name) {
    this.name = name;
  }

  public int size() {
    return size;  
  }

  public String asFormatString() {
    Type vectorType = new TypeToken<Vector>() {
    }.getType();
    GsonBuilder builder = new GsonBuilder();
    builder.registerTypeAdapter(vectorType, new JsonVectorAdapter());
    Gson gson = builder.create();
    return gson.toJson(this, vectorType);
  }

  /**
   * Compare whether two Vector implementations have the same elements, regardless of the implementation and name. Two
   * Vectors are equivalent if they have the same cardinality and all of their values are the same. <p/> Does not
   * compare {@link Vector#getName()}.
   *
   * @param left  The left hand Vector to compare
   * @param right The right hand Vector
   * @return true if the two Vectors have the same cardinality and the same values
   * @see #strictEquivalence(Vector, Vector)
   * @see Vector#equals(Object)
   */
  public static boolean equivalent(Vector left, Vector right) {
    if (left == right) {
      return true;
    }
    int leftCardinality = left.size();
    if (leftCardinality == right.size()) {
      for (int i = 0; i < leftCardinality; i++) {
        if (left.getQuick(i) != right.getQuick(i)) {
          return false;
        }

      }
    } else {
      return false;
    }
    return true;
  }

  /**
   * Compare whether two Vector implementations are the same, including the underlying implementation. Two Vectors are
   * the same if they have the same cardinality, same name and all of their values are the same.
   *
   * @param left  The left hand Vector to compare
   * @param right The right hand Vector
   * @return true if the two Vectors have the same cardinality and the same values
   */
  public static boolean strictEquivalence(Vector left, Vector right) {
    if (left == right) {
      return true;
    }
    if (!(left.getClass().equals(right.getClass()))) {
      return false;
    }
    String leftName = left.getName();
    String rightName = right.getName();
    if (leftName != null && rightName != null && !leftName.equals(rightName)) {
      return false;
    } else if ((leftName != null && rightName == null)
        || (rightName != null && leftName == null)) {
      return false;
    }

    int leftCardinality = left.size();
    if (leftCardinality == right.size()) {
      for (int i = 0; i < leftCardinality; i++) {
        if (left.getQuick(i) != right.getQuick(i)) {
          return false;
        }

      }
    } else {
      return false;
    }
    return true;
  }

  @Override
  public int hashCode() {
    int prime = 31;
    int result = prime + ((name == null) ? 0 : name.hashCode());
    result = prime * result + size();
    Iterator<Element> iter = iterateNonZero();
    while (iter.hasNext()) {
      Element ele = iter.next();
      long v = Double.doubleToLongBits(ele.get());
      result += (ele.index() * (int)(v^(v>>32)));
    }
    return result;
   }


  public double get(String label) throws IndexException, UnboundLabelException {
    if (bindings == null) {
      throw new UnboundLabelException();
    }
    Integer index = bindings.get(label);
    if (index == null) {
      throw new UnboundLabelException();
    }
    return get(index);
  }

  public Map<String, Integer> getLabelBindings() {
    return bindings;
  }

  public void set(String label, double value) throws IndexException,
      UnboundLabelException {
    if (bindings == null) {
      throw new UnboundLabelException();
    }
    Integer index = bindings.get(label);
    if (index == null) {
      throw new UnboundLabelException();
    }
    set(index, value);
  }

  public void setLabelBindings(Map<String, Integer> bindings) {
    this.bindings = bindings;
  }

  public void set(String label, int index, double value) throws IndexException {
    if (bindings == null) {
      bindings = new HashMap<String, Integer>();
    }
    bindings.put(label, index);
    set(index, value);
  }
}
