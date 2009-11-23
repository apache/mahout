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

import java.util.Iterator;
import java.util.Map;

/**
 * The basic interface including numerous convenience functions <p/> NOTE: All implementing classes must have a
 * constructor that takes an int for cardinality and a no-arg constructor that can be used for marshalling the Writable
 * instance <p/> NOTE: Implementations may choose to reuse the Vector.Element in the Iterable methods
 */
public interface Vector extends Cloneable, Writable {

  /**
   * Vectors may have a name associated with them, which makes them easy to identify
   *
   * @return The name, or null if one has not been set
   */
  String getName();

  /**
   * Set a name for this vector.  Need not be unique in a set of Vectors, but probably is more useful if it is. In other
   * words, Mahout does not check for uniqueness.
   *
   * @param name The name
   */
  void setName(String name);

  /** @return a formatted String suitable for output */
  String asFormatString();

  /**
   * Assign the value to all elements of the receiver
   *
   * @param value a double value
   * @return the modified receiver
   */
  Vector assign(double value);

  /**
   * Assign the values to the receiver
   *
   * @param values a double[] of values
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  Vector assign(double[] values);

  /**
   * Assign the other matrix values to the receiver
   *
   * @param other a Vector
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  Vector assign(Vector other);

  /**
   * Apply the function to each element of the receiver
   *
   * @param function a DoubleFunction to apply
   * @return the modified receiver
   */
  Vector assign(UnaryFunction function);

  /**
   * Apply the function to each element of the receiver and the corresponding element of the other argument
   *
   * @param other    a Vector containing the second arguments to the function
   * @param function a DoubleDoubleFunction to apply
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  Vector assign(Vector other, BinaryFunction function);

  /**
   * Apply the function to each element of the receiver, using the y value as the second argument of the BinaryFunction
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
  int size();

  /**
   * Return a copy of the recipient
   *
   * @return a new Vector
   */
  Vector clone();

  /**
   * Iterates over all elements <p/> * NOTE: Implementations may choose to reuse the Element returned for performance
   * reasons, so if you need a copy of it, you should call {@link #getElement} for the given index
   *
   * @return An {@link java.util.Iterator} over all elements
   */
  Iterator<Element> iterateAll();

  /**
   * Iterates over all non-zero elements. <p/> NOTE: Implementations may choose to reuse the Element returned for
   * performance reasons, so if you need a copy of it, you should call {@link #getElement} for the given index
   *
   * @return An {@link java.util.Iterator} over all non-zero elements
   */
  Iterator<Element> iterateNonZero();

  /**
   * Return the value at the index defined by the label
   *
   * @param label a String label that maps to an index
   * @return the double at the index
   * @throws IndexException        if the index is out of bounds
   * @throws UnboundLabelException if the label is unbound
   */
  double get(String label) throws IndexException, UnboundLabelException;

  /**
   * Return a map of the current label bindings of the receiver
   *
   * @return a Map<String, Integer>
   */
  Map<String, Integer> getLabelBindings();

  /**
   * Return an object of Vector.Element representing an element of this Vector. Useful when designing new iterator
   * types.
   *
   * @param index Index of the Vector.Element required
   * @return The Vector.Element Object
   */
  Element getElement(int index);

  /**
   * A holder for information about a specific item in the Vector. <p/> When using with an Iterator, the implementation
   * may choose to reuse this element, so you may need to make a copy if you want to keep it
   */
  interface Element {
    /** @return the value of this vector element. */
    double get();

    /** @return the index of this vector element. */
    int index();

    /** @param value Set the current element to value. */
    void set(double value);
  }

  /**
   * Return a new matrix containing the values of the recipient divided by the argument
   *
   * @param x a double value
   * @return a new Vector
   */
  Vector divide(double x);

  /**
   * Return the dot product of the recipient and the argument
   *
   * @param x a Vector
   * @return a new Vector
   * @throws CardinalityException if the cardinalities differ
   */
  double dot(Vector x);

  /**
   * Return the value at the given index
   *
   * @param index an int index
   * @return the double at the index
   * @throws IndexException if the index is out of bounds
   */
  double get(int index);

  /**
   * Return the value at the given index, without checking bounds
   *
   * @param index an int index
   * @return the double at the index
   */
  double getQuick(int index);

  /**
   * Return if the other matrix and the receiver share any underlying data cells
   *
   * @param other a Vector
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
   * Return an empty matrix of the same underlying class as the receiver and of the given cardinality
   *
   * @param cardinality an int specifying the desired cardinality
   * @return a Vector
   */
  Vector like(int cardinality);

  /**
   * Return a new matrix containing the element by element difference of the recipient and the argument
   *
   * @param x a Vector
   * @return a new Vector
   * @throws CardinalityException if the cardinalities differ
   */
  Vector minus(Vector x);

  /**
   * Return a new matrix containing the normalized (L_2 norm) values of the recipient
   *
   * @return a new Vector
   */
  Vector normalize();

  /**
   * Return a new Vector containing the normalized (L_power norm) values of the recipient. <p/> See
   * http://en.wikipedia.org/wiki/Lp_space <p/> Technically, when 0 < power < 1, we don't have a norm, just a metric,
   * but we'll overload this here. <p/> Also supports power == 0 (number of non-zero elements) and power = {@link
   * Double#POSITIVE_INFINITY} (max element). Again, see the Wikipedia page for more info
   *
   * @param power The power to use. Must be >= 0. May also be {@link Double#POSITIVE_INFINITY}. See the Wikipedia link
   *              for more on this.
   * @return a new Vector
   */
  Vector normalize(double power);

  /**
   * Return the k-norm of the vector. <p/> See 
   * http://en.wikipedia.org/wiki/Lp_space <p/> Technically, when 
   * 0 &gt; power &lt; 1, we don't have a norm, just a metric, but 
   * we'll overload this here. Also supports power == 0 (number of non-zero elements) and
   * power = {@link Double#POSITIVE_INFINITY} (max element). Again, see the Wikipedia page
   * for more info.
   *
   * @param power The power to use.
   *
   * @see #normalize(double) 
   *
   */
  double norm(double power);

  /** @return The maximum value in the Vector */
  double maxValue();

  /** @return The index of the maximum value */
  int maxValueIndex();

  /**
   * Return a new matrix containing the sum of each value of the recipient and the argument
   *
   * @param x a double
   * @return a new Vector
   */
  Vector plus(double x);

  /**
   * Return a new matrix containing the element by element sum of the recipient and the argument
   *
   * @param x a Vector
   * @return a new Vector
   * @throws CardinalityException if the cardinalities differ
   */
  Vector plus(Vector x);

  /**
   * Set the value at the index that is mapped to the label
   *
   * @param label a String label that maps to an index
   * @param value the double value at the index
   */
  void set(String label, double value) throws IndexException,
      UnboundLabelException;

  /**
   * Set the value at the index and add the label to the bindings
   *
   * @param label a String label that maps to an index
   * @param index an int index
   * @param value a double value
   */
  void set(String label, int index, double value) throws IndexException;

  /**
   * Sets a map of label bindings in the receiver
   *
   * @param bindings a {@link Map<String, Integer>} of label bindings
   */
  void setLabelBindings(Map<String, Integer> bindings);

  /**
   * Set the value at the given index
   *
   * @param index an int index into the receiver
   * @param value a double value to set
   * @throws IndexException if the index is out of bounds
   */
  void set(int index, double value);

  /**
   * Set the value at the given index, without checking bounds
   *
   * @param index an int index into the receiver
   * @param value a double value to set
   */
  void setQuick(int index, double value);

  /**
   * Return the number of values in the recipient
   *
   * @return an int
   */
  int getNumNondefaultElements();

  /**
   * Return a new matrix containing the product of each value of the recipient and the argument
   *
   * @param x a double argument
   * @return a new Vector
   */
  Vector times(double x);

  /**
   * Return a new matrix containing the element-wise product of the recipient and the argument
   *
   * @param x a Vector argument
   * @return a new Vector
   * @throws CardinalityException if the cardinalities differ
   */
  Vector times(Vector x);

  /**
   * Return a new matrix containing the subset of the recipient
   *
   * @param offset an int offset into the receiver
   * @param length the cardinality of the desired result
   * @return a new Vector
   * @throws CardinalityException if the length is greater than the cardinality of the receiver
   * @throws IndexException       if the offset is negative or the offset+length is outside of the receiver
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
   * @param other another Vector
   * @return a Matrix
   */
  Matrix cross(Vector other);

  /*
   * Need stories for these but keeping them here for now.
   */
  // void getNonZeros(IntArrayList jx, DoubleArrayList values);
  // void foreachNonZero(IntDoubleFunction f);
  // double aggregate(DoubleDoubleFunction aggregator, DoubleFunction map);
  // double aggregate(Vector other, DoubleDoubleFunction aggregator,
  // DoubleDoubleFunction map);
  // NewVector assign(Vector y, DoubleDoubleFunction function, IntArrayList
  // nonZeroIndexes);


  /** Return the sum of squares of all elements in the vector. Square root of this value is the length of the vector. */
  double getLengthSquared();

  /** Get the square of the distance between this vector and the other vector. */
  double getDistanceSquared(Vector v);

  /** Add the elements to the other vector and results are stored in that vector. */
  void addTo(Vector v);

}
