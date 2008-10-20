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

import org.apache.hadoop.io.Writable;
import org.apache.hadoop.io.WritableComparable;

/**
 * Implementations of generic capabilities like sum of elements and dot products
 */
public abstract class AbstractVector implements Vector, Writable {

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#asFormatString()
   */
  public abstract WritableComparable asWritableComparable();

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#cardinality()
   */
  public abstract int cardinality();

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#copy()
   */
  public abstract Vector copy();

  public abstract boolean haveSharedCells(Vector other);

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#getQuick(int)
   */
  public abstract double getQuick(int index);

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#like()
   */
  public abstract Vector like();

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#like(int)
   */
  public abstract Vector like(int cardinality);

  /**
   * Subclasses must override to return an appropriately sparse or dense result
   * 
   * @param rows
   *            the row cardinality
   * @param columns
   *            the column cardinality
   * @return a Matrix
   */
  protected abstract Matrix matrixLike(int rows, int columns);

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#setQuick(int, double)
   */
  public abstract void setQuick(int index, double value);

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#size()
   */
  public abstract int size();

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#toArray()
   */
  public abstract double[] toArray();

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#viewPart(int, int)
   */
  public abstract Vector viewPart(int offset, int length)
      throws CardinalityException, IndexException;

  /**
   * Returns an iterator for traversing the Vector, but not in any particular
   * order. The actual implementations may make some guarantees about the order
   * in which the vector is traversed. Otherwise, the traversal order is
   * undefined.
   * 
   * @see java.lang.Iterable#iterator()
   */
  public abstract java.util.Iterator<Vector.Element> iterator();

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#getElement
   */
  // @Override JDK 1.6
  public Vector.Element getElement(int index) {
    return new Element(index);
  }

  public class Element implements Vector.Element {
    private final int ind;

    public Element(int ind) {
      this.ind = ind;
    }

    public double get() {
      return getQuick(ind);
    }

    public int index() {
      return ind;
    }

    public void set(double value) {
      setQuick(ind, value);
    }
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#normalize(double)
   */
  public Vector divide(double x) {
    Vector result = copy();
    for (int i = 0; i < result.cardinality(); i++)
      result.setQuick(i, getQuick(i) / x);
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#dot(org.apache.mahout.matrix.Vector)
   */
  public double dot(Vector x) throws CardinalityException {
    if (cardinality() != x.cardinality())
      throw new CardinalityException();
    double result = 0;
    for (int i = 0; i < cardinality(); i++)
      result += getQuick(i) * x.getQuick(i);
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#get(int)
   */
  public double get(int index) throws IndexException {
    if (index >= 0 && index < cardinality())
      return getQuick(index);
    else
      throw new IndexException();
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#minus(org.apache.mahout.matrix.Vector)
   */
  public Vector minus(Vector x) throws CardinalityException {
    if (cardinality() != x.cardinality())
      throw new CardinalityException();
    Vector result = copy();
    for (int i = 0; i < result.cardinality(); i++)
      result.setQuick(i, getQuick(i) - x.getQuick(i));
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#normalize()
   */
  public Vector normalize() {
    double divSq = 0;
    try {
      divSq = Math.sqrt(dot(this));
    } catch (CardinalityException e) {
      // cannot occur with dot(this)
    }
    return divide(divSq);
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#plus(double)
   */
  public Vector plus(double x) {
    Vector result = copy();
    for (int i = 0; i < result.cardinality(); i++)
      result.setQuick(i, getQuick(i) + x);
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#plus(org.apache.mahout.matrix.Vector)
   */
  public Vector plus(Vector x) throws CardinalityException {
    if (cardinality() != x.cardinality())
      throw new CardinalityException();
    Vector result = copy();
    for (int i = 0; i < result.cardinality(); i++)
      result.setQuick(i, getQuick(i) + x.getQuick(i));
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#set(int, double)
   */
  public void set(int index, double value) throws IndexException {
    if (index >= 0 && index < cardinality())
      setQuick(index, value);
    else
      throw new IndexException();
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#times(double)
   */
  public Vector times(double x) {
    Vector result = copy();
    for (int i = 0; i < result.cardinality(); i++)
      result.setQuick(i, getQuick(i) * x);
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#times(org.apache.mahout.matrix.Vector)
   */
  public Vector times(Vector x) throws CardinalityException {
    if (cardinality() != x.cardinality())
      throw new CardinalityException();
    Vector result = copy();
    for (int i = 0; i < result.cardinality(); i++)
      result.setQuick(i, getQuick(i) * x.getQuick(i));
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#zSum()
   */
  public double zSum() {
    double result = 0;
    for (int i = 0; i < cardinality(); i++)
      result += getQuick(i);
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#assign(double)
   */
  public Vector assign(double value) {
    for (int i = 0; i < cardinality(); i++)
      setQuick(i, value);
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#assign(double[])
   */
  public Vector assign(double[] values) throws CardinalityException {
    if (values.length != cardinality())
      throw new CardinalityException();
    for (int i = 0; i < cardinality(); i++)
      setQuick(i, values[i]);
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#assign(org.apache.mahout.matrix.Vector)
   */
  public Vector assign(Vector other) throws CardinalityException {
    if (other.cardinality() != cardinality())
      throw new CardinalityException();
    for (int i = 0; i < cardinality(); i++)
      setQuick(i, other.getQuick(i));
    return this;
  }

  /* (non-Javadoc)
   * @see org.apache.mahout.matrix.Vector#assign(org.apache.mahout.matrix.BinaryFunction, double)
   */
  public Vector assign(BinaryFunction f, double y) {
    for (int i = 0; i < cardinality(); i++) {
      setQuick(i, f.apply(getQuick(i), y));
    }
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#assign(org.apache.mahout.utils.matrix.DoubleFunction)
   */
  public Vector assign(UnaryFunction function) {
    for (int i = 0; i < cardinality(); i++)
      setQuick(i, function.apply(getQuick(i)));
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#assign(org.apache.mahout.matrix.Vector,
   *      org.apache.mahout.utils.matrix.DoubleDoubleFunction)
   */
  public Vector assign(Vector other, BinaryFunction function)
      throws CardinalityException {
    if (other.cardinality() != cardinality())
      throw new CardinalityException();
    for (int i = 0; i < cardinality(); i++)
      setQuick(i, function.apply(getQuick(i), other.getQuick(i)));
    return this;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#cross(org.apache.mahout.matrix.Vector)
   */
  public Matrix cross(Vector other) {
    Matrix result = matrixLike(cardinality(), other.cardinality());
    for (int row = 0; row < cardinality(); row++)
      try {
        result.assignRow(row, other.times(getQuick(row)));
      } catch (CardinalityException e) {
        // cannot happen since result is other's cardinality
      }
    return result;
  }

  /*
   * (non-Javadoc)
   * 
   * @see org.apache.mahout.matrix.Vector#asFormatString()
   */
  public abstract String asFormatString();

  /**
   * Decodes a point from its WritableComparable representation.
   * 
   * @param writableComparable
   *            a WritableComparable produced by asWritableComparable. Note the
   *            payload remainder: it is optional, but can be present.
   * @return the n-dimensional point
   */
  public static Vector decodeVector(WritableComparable writableComparable) {
    return decodeVector(writableComparable.toString());
  }

  /**
   * Decodes a point from its string representation.
   * 
   * @param formattedString
   *            a formatted String produced by asFormatString. Note the payload
   *            remainder: it is optional, but can be present.
   * @return the n-dimensional point
   */
  public static Vector decodeVector(String formattedString) {
    Vector result;
    if (formattedString.trim().startsWith("[s"))
      result = SparseVector.decodeFormat(formattedString);
    else
      result = DenseVector.decodeFormat(formattedString);
    return result;
  }

}
