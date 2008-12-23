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
 * The basic interface including numerous convenience functions
 */
public interface Vector extends Iterable<Vector.Element>, Writable {

  /**
   * Return a formatted WritableComparable<?> suitable for output
   *
   * @return formatted WritableComparable
   */
  WritableComparable<?> asWritableComparable();

  /**
   * Return a formatted String suitable for output
   *
   * @return
   */
  String asFormatString();

  /**
   * Assign the value to all elements of the receiver
   *
   * @param value
   *            a double value
   * @return the modified receiver
   */
  Vector assign(double value);

  /**
   * Assign the values to the receiver
   *
   * @param values
   *            a double[] of values
   * @return the modified receiver
   * @throws CardinalityException
   *             if the cardinalities differ
   */
  Vector assign(double[] values);

  /**
   * Assign the other matrix values to the receiver
   *
   * @param other
   *            a Vector
   * @return the modified receiver
   * @throws CardinalityException
   *             if the cardinalities differ
   */
  Vector assign(Vector other);

  /**
   * Apply the function to each element of the receiver
   *
   * @param function
   *            a DoubleFunction to apply
   * @return the modified receiver
   */
  Vector assign(UnaryFunction function);

  /**
   * Apply the function to each element of the receiver and the corresponding
   * element of the other argument
   *
   * @param other
   *            a Vector containing the second arguments to the function
   * @param function
   *            a DoubleDoubleFunction to apply
   * @return the modified receiver
   * @throws CardinalityException
   *             if the cardinalities differ
   */
  Vector assign(Vector other, BinaryFunction function);

  /**
   * Apply the function to each element of the receiver, using the y value as
   * the second argument of the BinaryFunction
   * 
   * @param f a BinaryFunction to be applied
   * @param y a double value to be argument to the function
   * @return the modified receiver
   */
  Vector assign(BinaryFunction f, double y);
         
  /**
   * Return the cardinality of the recipient (the maximum number of values)
   *
   * @return an int
   */
  int cardinality();

  /**
   * Return a copy of the recipient
   *
   * @return a new Vector
   */
  Vector copy();

  /**
   * Return an object of Vector.Element representing an element of this Vector.
   * Useful when designing new iterator types.
   *
   * @param index
   *            Index of the Vector.Element required
   * @return The Vector.Element Object
   */
  Element getElement(int index);

  interface Element {
    /**
     * @return the value of this vector element.
     */
    double get();

    /**
     * @return the index of this vector element.
     */
    int index();

    /**
     * @param value
     *            Set the current element to value.
     */
    void set(double value);
  }

  /**
   * Return a new matrix containing the values of the recipient divided by the
   * argument
   *
   * @param x
   *            a double value
   * @return a new Vector
   */
  Vector divide(double x);

  /**
   * Return the dot product of the recipient and the argument
   *
   * @param x
   *            a Vector
   * @return a new Vector
   * @throws CardinalityException
   *             if the cardinalities differ
   */
  double dot(Vector x);

  /**
   * Return the value at the given index
   *
   * @param index
   *            an int index
   * @return the double at the index
   * @throws IndexException
   *             if the index is out of bounds
   */
  double get(int index);

  /**
   * Return the value at the given index, without checking bounds
   *
   * @param index
   *            an int index
   * @return the double at the index
   */
  double getQuick(int index);

  /**
   * Return if the other matrix and the receiver share any underlying data cells
   *
   * @param other
   *            a Vector
   * @return true if the other matrix has common data cells
   */
  boolean haveSharedCells(Vector other);

  /**
   * Return an empty matrix of the same underlying class as the receiver
   *
   * @return a Vector
   */
  Vector like();

  /**
   * Return an empty matrix of the same underlying class as the receiver and of
   * the given cardinality
   *
   * @param cardinality
   *            an int specifying the desired cardinality
   * @return a Vector
   */
  Vector like(int cardinality);

  /**
   * Return a new matrix containing the element by element difference of the
   * recipient and the argument
   *
   * @param x
   *            a Vector
   * @return a new Vector
   * @throws CardinalityException
   *             if the cardinalities differ
   */
  Vector minus(Vector x);

  /**
   * Return a new matrix containing the normalized values of the recipient
   *
   * @return a new Vector
   */
  Vector normalize();

  /**
   * Return a new matrix containing the sum of each value of the recipient and
   * the argument
   *
   * @param x
   *            a double
   * @return a new Vector
   */
  Vector plus(double x);

  /**
   * Return a new matrix containing the element by element sum of the recipient
   * and the argument
   *
   * @param x
   *            a Vector
   * @return a new Vector
   * @throws CardinalityException
   *             if the cardinalities differ
   */
  Vector plus(Vector x);

  /**
   * Set the value at the given index
   *
   * @param index
   *            an int index into the receiver
   * @param value
   *            a double value to set
   * @throws IndexException
   *             if the index is out of bounds
   */
  void set(int index, double value);

  /**
   * Set the value at the given index, without checking bounds
   *
   * @param index
   *            an int index into the receiver
   * @param value
   *            a double value to set
   */
  void setQuick(int index, double value);

  /**
   * Return the number of values in the recipient
   *
   * @return an int
   */
  int size();

  /**
   * Return a new matrix containing the product of each value of the recipient
   * and the argument
   *
   * @param x
   *            a double argument
   * @return a new Vector
   */
  Vector times(double x);

  /**
   * Return a new matrix containing the element-wise product of the recipient
   * and the argument
   *
   * @param x
   *            a Vector argument
   * @return a new Vector
   * @throws CardinalityException
   *             if the cardinalities differ
   */
  Vector times(Vector x);

  /**
   * Return the element of the recipient as a double[]
   *
   * @return a double[]
   */
  double[] toArray();

  /**
   * Return a new matrix containing the subset of the recipient
   *
   * @param offset
   *            an int offset into the receiver
   * @param length
   *            the cardinality of the desired result
   * @return a new Vector
   * @throws CardinalityException
   *             if the length is greater than the cardinality of the receiver
   * @throws IndexException
   *             if the offset is negative or the offset+length is outside of
   *             the receiver
   */
  Vector viewPart(int offset, int length);

  /**
   * Return the sum of all the elements of the receiver
   *
   * @return a double
   */
  double zSum();

  /**
   * Return the cross product of the receiver and the other vector
   *
   * @param other
   *            another Vector
   * @return a Matrix
   */
  Matrix cross(Vector other);

  /*
   * Need stories for these but keeping them here for now.
   *
   */
  // void getNonZeros(IntArrayList jx, DoubleArrayList values);
  // void foreachNonZero(IntDoubleFunction f);
  // double aggregate(DoubleDoubleFunction aggregator, DoubleFunction map);
  // double aggregate(Vector other, DoubleDoubleFunction aggregator,
  // DoubleDoubleFunction map);
  // NewVector assign(Vector y, DoubleDoubleFunction function, IntArrayList
  // nonZeroIndexes);
}
