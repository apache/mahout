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

import org.apache.mahout.math.function.DoubleDoubleFunction;
import org.apache.mahout.math.function.DoubleFunction;
import org.apache.mahout.math.function.VectorFunction;

import java.util.Map;

/** The basic interface including numerous convenience functions */
public interface Matrix extends Cloneable, VectorIterable {

  /** @return a formatted String suitable for output */
  String asFormatString();

  /**
   * Assign the value to all elements of the receiver
   *
   * @param value a double value
   * @return the modified receiver
   */
  Matrix assign(double value);

  /**
   * Assign the values to the receiver
   *
   * @param values a double[] of values
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  Matrix assign(double[][] values);

  /**
   * Assign the other vector values to the receiver
   *
   * @param other a Matrix
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  Matrix assign(Matrix other);

  /**
   * Apply the function to each element of the receiver
   *
   * @param function a DoubleFunction to apply
   * @return the modified receiver
   */
  Matrix assign(DoubleFunction function);

  /**
   * Apply the function to each element of the receiver and the corresponding element of the other argument
   *
   * @param other    a Matrix containing the second arguments to the function
   * @param function a DoubleDoubleFunction to apply
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  Matrix assign(Matrix other, DoubleDoubleFunction function);

  /**
   * Assign the other vector values to the column of the receiver
   *
   * @param column the int row to assign
   * @param other  a Vector
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  Matrix assignColumn(int column, Vector other);

  /**
   * Assign the other vector values to the row of the receiver
   *
   * @param row   the int row to assign
   * @param other a Vector
   * @return the modified receiver
   * @throws CardinalityException if the cardinalities differ
   */
  Matrix assignRow(int row, Vector other);

  /**
   * Collects the results of a function applied to each row of a matrix.
   * @param f  The function to be applied to each row.
   * @return  The vector of results.
   */
  Vector aggregateRows(VectorFunction f);

  /**
   * Collects the results of a function applied to each column of a matrix.
   * @param f  The function to be applied to each column.
   * @return  The vector of results.
   */
  Vector aggregateColumns(VectorFunction f);

  /**
   * Collects the results of a function applied to each element of a matrix and then
   * aggregated.
   * @param combiner  A function that combines the results of the mapper.
   * @param mapper  A function to apply to each element.
   * @return  The result.
   */
  double aggregate(DoubleDoubleFunction combiner, DoubleFunction mapper);

  /**
   * @return The number of rows in the matrix.
   */
  int columnSize();

  /**
   * @return Returns the number of rows in the matrix.
   */
  int rowSize();

  /**
   * Return a copy of the recipient
   *
   * @return a new Matrix
   */
  Matrix clone();

  /**
   * Returns matrix determinator using Laplace theorem
   *
   * @return a matrix determinator
   */
  double determinant();

  /**
   * Return a new matrix containing the values of the recipient divided by the argument
   *
   * @param x a double value
   * @return a new Matrix
   */
  Matrix divide(double x);

  /**
   * Return the value at the given indexes
   *
   * @param row    an int row index
   * @param column an int column index
   * @return the double at the index
   * @throws IndexException if the index is out of bounds
   */
  double get(int row, int column);

  /**
   * Return the value at the given indexes, without checking bounds
   *
   * @param row    an int row index
   * @param column an int column index
   * @return the double at the index
   */
  double getQuick(int row, int column);

  /**
   * Return an empty matrix of the same underlying class as the receiver
   *
   * @return a Matrix
   */
  Matrix like();

  /**
   * Returns an empty matrix of the same underlying class as the receiver and of the specified size.
   *
   * @param rows    the int number of rows
   * @param columns the int number of columns
   */
  Matrix like(int rows, int columns);

  /**
   * Return a new matrix containing the element by element difference of the recipient and the argument
   *
   * @param x a Matrix
   * @return a new Matrix
   * @throws CardinalityException if the cardinalities differ
   */
  Matrix minus(Matrix x);

  /**
   * Return a new matrix containing the sum of each value of the recipient and the argument
   *
   * @param x a double
   * @return a new Matrix
   */
  Matrix plus(double x);

  /**
   * Return a new matrix containing the element by element sum of the recipient and the argument
   *
   * @param x a Matrix
   * @return a new Matrix
   * @throws CardinalityException if the cardinalities differ
   */
  Matrix plus(Matrix x);

  /**
   * Set the value at the given index
   *
   * @param row    an int row index into the receiver
   * @param column an int column index into the receiver
   * @param value  a double value to set
   * @throws IndexException if the index is out of bounds
   */
  void set(int row, int column, double value);

  void set(int row, double[] data);

  /**
   * Set the value at the given index, without checking bounds
   *
   * @param row    an int row index into the receiver
   * @param column an int column index into the receiver
   * @param value  a double value to set
   */
  void setQuick(int row, int column, double value);

  /**
   * Return the number of values in the recipient
   *
   * @return an int[2] containing [row, column] count
   */
  int[] getNumNondefaultElements();

  /**
   * Return a new matrix containing the product of each value of the recipient and the argument
   *
   * @param x a double argument
   * @return a new Matrix
   */
  Matrix times(double x);

  /**
   * Return a new matrix containing the product of the recipient and the argument
   *
   * @param x a Matrix argument
   * @return a new Matrix
   * @throws CardinalityException if the cardinalities are incompatible
   */
  Matrix times(Matrix x);

  /**
   * Return a new matrix that is the transpose of the receiver
   *
   * @return the transpose
   */
  Matrix transpose();

  /**
   * Return the sum of all the elements of the receiver
   *
   * @return a double
   */
  double zSum();

  /**
   * Return a map of the current column label bindings of the receiver
   *
   * @return a Map<String, Integer>
   */
  Map<String, Integer> getColumnLabelBindings();

  /**
   * Return a map of the current row label bindings of the receiver
   *
   * @return a Map<String, Integer>
   */
  Map<String, Integer> getRowLabelBindings();

  /**
   * Sets a map of column label bindings in the receiver
   *
   * @param bindings a Map<String, Integer> of label bindings
   */
  void setColumnLabelBindings(Map<String, Integer> bindings);

  /**
   * Sets a map of row label bindings in the receiver
   *
   * @param bindings a Map<String, Integer> of label bindings
   */
  void setRowLabelBindings(Map<String, Integer> bindings);

  /**
   * Return the value at the given labels
   *
   * @param rowLabel    a String row label
   * @param columnLabel a String column label
   * @return the double at the index
   * 
   * @throws IndexException if the index is out of bounds
   */
  double get(String rowLabel, String columnLabel);

  /**
   * Set the value at the given index
   *
   * @param rowLabel    a String row label
   * @param columnLabel a String column label
   * @param value       a double value to set
   * @throws IndexException if the index is out of bounds
   */
  void set(String rowLabel, String columnLabel, double value);

  /**
   * Set the value at the given index, updating the row and column label bindings
   *
   * @param rowLabel    a String row label
   * @param columnLabel a String column label
   * @param row         an int row index
   * @param column      an int column index
   * @param value       a double value
   */
  void set(String rowLabel, String columnLabel, int row, int column, double value);

  /**
   * Sets the row values at the given row label
   *
   * @param rowLabel a String row label
   * @param rowData  a double[] array of row data
   */
  void set(String rowLabel, double[] rowData);

  /**
   * Sets the row values at the given row index and updates the row labels
   *
   * @param rowLabel the String row label
   * @param row      an int the row index
   * @param rowData  a double[] array of row data
   */
  void set(String rowLabel, int row, double[] rowData);

  /*
   * Need stories for these but keeping them here for now.
   * 
   */
  // void getNonZeros(IntArrayList jx, DoubleArrayList values);
  // void foreachNonZero(IntDoubleFunction f);
  // double aggregate(DoubleDoubleFunction aggregator, DoubleFunction map);
  // double aggregate(Matrix other, DoubleDoubleFunction aggregator,
  // DoubleDoubleFunction map);
  // NewMatrix assign(Matrix y, DoubleDoubleFunction function, IntArrayList
  // nonZeroIndexes);

  /**
   * Return a view into part of a matrix.  Changes to the view will change the
   * original matrix.
   *
   * @param offset an int[2] offset into the receiver
   * @param size   the int[2] size of the desired result
   * @return a matrix that shares storage with part of the original matrix.
   * @throws CardinalityException if the length is greater than the cardinality of the receiver
   * @throws IndexException       if the offset is negative or the offset+length is outside of the receiver
   */
  Matrix viewPart(int[] offset, int[] size);

  /**
   * Return a view into part of a matrix.  Changes to the view will change the
   * original matrix.
   *
   * @param rowOffset           The first row of the view
   * @param rowsRequested       The number of rows in the view
   * @param columnOffset        The first column in the view
   * @param columnsRequested    The number of columns in the view
   * @return a matrix that shares storage with part of the original matrix.
   * @throws CardinalityException if the length is greater than the cardinality of the receiver
   * @throws IndexException       if the offset is negative or the offset+length is outside of the
   *                              receiver
   */
  Matrix viewPart(int rowOffset, int rowsRequested, int columnOffset, int columnsRequested);

  /**
   * Return a reference to a row.  Changes to the view will change the original matrix.
   * @param row  The index of the row to return.
   * @return A vector that shares storage with the original.
   */
  Vector viewRow(int row);

  /**
   * Return a reference to a column.  Changes to the view will change the original matrix.
   * @param column  The index of the column to return.
   * @return A vector that shares storage with the original.
   */
  Vector viewColumn(int column);

  /**
   * Returns a reference to the diagonal of a matrix. Changes to the view will change
   * the original matrix.
   * @return A vector that shares storage with the original matrix.
   */
  Vector viewDiagonal();
}
