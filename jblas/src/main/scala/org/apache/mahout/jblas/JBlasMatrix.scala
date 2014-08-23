/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.jblas

import org.apache.mahout.math.Matrix
import org.apache.mahout.math.Vector
import org.apache.mahout.math.AbstractMatrix
import org.apache.mahout.math.DenseMatrix
import org.apache.mahout.math.SparseMatrix

import org.apache.mahout.math.IndexException
import org.apache.mahout.math.CardinalityException

import org.jblas.DoubleMatrix

class JBlasMatrix(nrows: Int, ncols: Int, val jm: DoubleMatrix) extends AbstractMatrix(nrows, ncols) {
  def this(nrows: Int, ncols: Int) = this(nrows, ncols, new DoubleMatrix(nrows, ncols))

  def setQuick(row: Int, col: Int, v: Double): Unit = jm.put(row, col, v)

  def like(nrow: Int, ncol: Int): Matrix = new JBlasMatrix(nrow, ncol)

  def like: Matrix = new JBlasMatrix(rowSize, columnSize)

  def getQuick(row: Int, col: Int): Double = jm.get(row, col)

  def assignRow(row: Int, v: Vector): Matrix = {
    if (columnSize != v.size)
      throw new CardinalityException(columnSize, v.size)
    if (row < 0 || row > rowSize)
      throw new IndexException(row, rowSize)
    for (i <- 0 until columnSize)
      setQuick(row, i, v.getQuick(i))
    this
  }

  def assignColumn(col: Int, v: Vector): Matrix = {
    if (rowSize != v.size)
      throw new CardinalityException(rowSize, v.size)
    if (col < 0 || col > columnSize)
      throw new IndexException(col, columnSize)
    for (i <- 0 until rowSize)
      setQuick(i, col, v.getQuick(i))
    this
  }

  override def times(other: Matrix): Matrix = {
    other match {
      case j: JBlasMatrix => new JBlasMatrix(nrows, ncols, jm.mmul(j.jm))
      case _ => super.times(other)
    }
  }
}
