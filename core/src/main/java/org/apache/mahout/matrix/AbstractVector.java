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

package org.apache.mahout.matrix;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.reflect.TypeToken;
import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.lang.reflect.Type;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

/** Implementations of generic capabilities like sum of elements and dot products */
public abstract class AbstractVector implements Vector {

  /**
   * User-settable mapping between String labels and Integer indices. Marked transient so that it will not be serialized
   * with each vector instance.
   */
  private transient Map<String, Integer> bindings;

  protected String name;

  protected AbstractVector() {
  }

  protected AbstractVector(String name) {
    this.name = name;
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
  public abstract Vector.Element getElement(int index);


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

  @Override
  public Vector divide(double x) {
    Vector result = clone();
    Iterator<Element> iter = result.iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      int index = element.index();
      result.setQuick(index, element.get() / x);
    }

    return result;
  }

  @Override
  public double dot(Vector x) {
    if (size() != x.size()) {
      throw new CardinalityException();
    }
    double result = 0;
    Iterator<Element> iter = iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      result += element.get() * x.getQuick(element.index());
    }

    return result;
  }

  @Override
  public double get(int index) {
    if (index >= 0 && index < size()) {
      return getQuick(index);
    } else {
      throw new IndexException();
    }
  }

  @Override
  public Vector minus(Vector x) {
    if (size() != x.size()) {
      throw new CardinalityException();
    }
    Vector result = clone();
    Iterator<Vector.Element> iter = x.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element e = iter.next();
      result.setQuick(e.index(), getQuick(e.index()) - e.get());
    }
    return result;
  }

  @Override
  public Vector normalize() {
    return divide(Math.sqrt(dot(this)));
  }

  @Override
  public Vector normalize(double power) {
    return divide(norm(power));
  }

  @Override
  public double norm(double power) {
    if (power < 0.0) {
      throw new IllegalArgumentException("Power must be >= 0");
    }
    // we can special case certain powers
    if (Double.isInfinite(power)) {
      return maxValue();
    } else if (power == 2.0) {
      return Math.sqrt(dot(this));
    } else if (power == 1.0) {
      return zSum();
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

  @Override
  public double maxValue() {
    double result = Double.MIN_VALUE;
    for (int i = 0; i < size(); i++) {
      result = Math.max(result, getQuick(i));
    }
    return result;
  }

  @Override
  public int maxValueIndex() {
    int result = -1;
    double max = Double.MIN_VALUE;
    for (int i = 0; i < size(); i++) {
      double tmp = getQuick(i);
      if (tmp > max) {
        max = tmp;
        result = i;
      }
    }
    return result;
  }

  @Override
  public Vector plus(double x) {
    Vector result = clone();
    for (int i = 0; i < result.size(); i++) {
      result.setQuick(i, getQuick(i) + x);
    }
    return result;
  }

  @Override
  public Vector plus(Vector x) {
    if (size() != x.size()) {
      throw new CardinalityException();
    }
    //TODO: get smarter about this, if we are adding a dense to a sparse, then we should return a dense
    Vector result = clone();
    Iterator<Vector.Element> iter = x.iterateNonZero();
    while (iter.hasNext()) {
      Vector.Element e = iter.next();
      result.setQuick(e.index(), getQuick(e.index()) + e.get());
    }

    /*for (int i = 0; i < result.size(); i++)
      result.setQuick(i, getQuick(i) + x.getQuick(i));*/
    return result;
  }

  @Override
  public void set(int index, double value) {
    if (index >= 0 && index < size()) {
      setQuick(index, value);
    } else {
      throw new IndexException();
    }
  }

  @Override
  public Vector times(double x) {
    Vector result = clone();
    Iterator<Element> iter = iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      int index = element.index();
      result.setQuick(index, element.get() * x);
    }

    return result;
  }

  @Override
  public Vector times(Vector x) {
    if (size() != x.size()) {
      throw new CardinalityException();
    }
    Vector result = clone();
    Iterator<Element> iter = result.iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      int index = element.index();
      result.setQuick(index, element.get() * x.getQuick(index));
    }

    return result;
  }

  @Override
  public double zSum() {
    double result = 0;
    Iterator<Element> iter = iterateNonZero();
    while (iter.hasNext()) {
      Element element = iter.next();
      result += element.get();
    }

    return result;
  }

  @Override
  public Vector assign(double value) {
    for (int i = 0; i < size(); i++) {
      setQuick(i, value);
    }
    return this;
  }

  @Override
  public Vector assign(double[] values) {
    if (values.length != size()) {
      throw new CardinalityException();
    }
    for (int i = 0; i < size(); i++) {
      setQuick(i, values[i]);
    }
    return this;
  }

  @Override
  public Vector assign(Vector other) {
    if (other.size() != size()) {
      throw new CardinalityException();
    }
    for (int i = 0; i < size(); i++) {
      setQuick(i, other.getQuick(i));
    }
    return this;
  }

  @Override
  public Vector assign(BinaryFunction f, double y) {
    for (int i = 0; i < size(); i++) {
      setQuick(i, f.apply(getQuick(i), y));
    }
    return this;
  }

  @Override
  public Vector assign(UnaryFunction function) {
    for (int i = 0; i < size(); i++) {
      setQuick(i, function.apply(getQuick(i)));
    }
    return this;
  }

  @Override
  public Vector assign(Vector other, BinaryFunction function) {
    if (other.size() != size()) {
      throw new CardinalityException();
    }
    for (int i = 0; i < size(); i++) {
      setQuick(i, function.apply(getQuick(i), other.getQuick(i)));
    }
    return this;
  }

  @Override
  public Matrix cross(Vector other) {
    Matrix result = matrixLike(size(), other.size());
    for (int row = 0; row < size(); row++) {
      result.assignRow(row, other.times(getQuick(row)));
    }
    return result;
  }

  /**
   * Decodes a point from its WritableComparable<?> representation.
   *
   * @param writableComparable a WritableComparable<?> produced by asWritableComparable. Note the payload remainder: it
   *                           is optional, but can be present.
   * @return the n-dimensional point
   */
  public static Vector decodeVector(WritableComparable<?> writableComparable) {
    return decodeVector(writableComparable.toString());
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

  @Override
  public String getName() {
    return name;

  }

  @Override
  public void setName(String name) {
    this.name = name;
  }

  @Override
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

  @Override
  public Map<String, Integer> getLabelBindings() {
    return bindings;
  }

  @Override
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

  @Override
  public void setLabelBindings(Map<String, Integer> bindings) {
    this.bindings = bindings;
  }

  @Override
  public void set(String label, int index, double value) throws IndexException {
    if (bindings == null) {
      bindings = new HashMap<String, Integer>();
    }
    bindings.put(label, index);
    set(index, value);
  }

  // cache most recent vector instance class name
  private static String instanceClassName;

  // cache most recent vector instance class
  private static Class<? extends Vector> instanceClass;

  /** Read and return a vector from the input */
  public static Vector readVector(DataInput in) throws IOException {
    String vectorClassName = in.readUTF();
    Vector vector;
    try {
      if (!vectorClassName.equals(instanceClassName)) {
        instanceClassName = vectorClassName;
        instanceClass = Class.forName(vectorClassName).asSubclass(Vector.class);
      }
      vector = instanceClass.newInstance();
    } catch (ClassNotFoundException e) {
      throw new RuntimeException(e);
    } catch (IllegalAccessException e) {
      throw new RuntimeException(e);
    } catch (InstantiationException e) {
      throw new RuntimeException(e);
    }
    vector.readFields(in);
    return vector;
  }

  /** Write the vector to the output */
  public static void writeVector(DataOutput out, Vector vector)
      throws IOException {
    String vectorClassName = vector.getClass().getName();
    out.writeUTF(vectorClassName);
    vector.write(out);

  }

}
