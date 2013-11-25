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

package org.apache.mahout.math.scalabindings

import org.apache.mahout.math.{Matrices, QRDecomposition, Vector, Matrix}
import scala.collection.JavaConversions._
import org.apache.mahout.math.function.{DoubleFunction, Functions}
import scala.math._

class MatrixOps(val m: Matrix) {

  import MatrixOps._

  def nrow = m.rowSize()

  def ncol = m.columnSize()


  def unary_- = m.assign(Functions.NEGATE)

  def +=(that: Matrix) = m.assign(that, Functions.PLUS)

  def -=(that: Matrix) = m.assign(that, Functions.MINUS)

  def +=(that: Double) = m.assign(new DoubleFunction {
    def apply(x: Double): Double = x + that
  })

  def -=(that: Double) = +=(-that)

  def +(that: Matrix) = cloned += that

  def -(that: Matrix) = cloned -= that

  // m.plus(that)?

  def +(that: Double) = cloned += that

  def -(that: Double) = cloned -= that


  def norm = sqrt(m.aggregate(Functions.PLUS, Functions.SQUARE))

  def pnorm(p: Int) = pow(m.aggregate(Functions.PLUS, Functions.chain(Functions.ABS, Functions.pow(p))), 1.0 / p)

  def apply(row: Int, col: Int) = m.get(row, col)

  def update(row: Int, col: Int, v: Double): Matrix = {
    m.setQuick(row, col, v);
    m
  }

  def update(rowRange: Range, colRange: Range, that: Matrix) = apply(rowRange, colRange) := that

  def update(row: Int, colRange: Range, that: Vector) = apply(row, colRange) := that

  def update(rowRange: Range, col: Int, that: Vector) = apply(rowRange, col) := that

  def apply(rowRange: Range, colRange: Range): Matrix = {

    if (rowRange == :: &&
        colRange == ::) return m

    val rr = if (rowRange == ::) (0 until m.nrow)
    else rowRange
    val cr = if (colRange == ::) (0 until m.ncol)
    else colRange

    return m.viewPart(rr.start, rr.length, cr.start, cr.length)

  }

  def apply(row: Int, colRange: Range): Vector = {
    var r = m.viewRow(row)
    if (colRange != ::) r = r.viewPart(colRange.start, colRange.length)
    r
  }

  def apply(rowRange: Range, col: Int): Vector = {
    var c = m.viewColumn(col)
    if (rowRange != ::) c = c.viewPart(rowRange.start, rowRange.length)
    c
  }

  /**
   * Warning: This provides read-only view only.
   * In most cases that's what one wants. To get a copy,
   * use <code>m.t cloned</code>
   * @return transposed view
   */
  def t = Matrices.transposedView(m)

  def det = m.determinant()

  def sum = m.zSum()

  def :=(that: Matrix) = m.assign(that)

  /**
   * Assigning from a row-wise collection of vectors
   * @param that
   */
  def :=(that: TraversableOnce[Vector]) = {
    var row = 0
    that.foreach(v => {
      m.assignRow(row, v)
      row += 1
    })
  }

  def :=(f: (Int, Int, Double) => Double): Matrix = {
    for (r <- 0 until nrow; c <- 0 until ncol) m(r, c) = f(r, c, m(r, c))
    m
  }

  def cloned: Matrix = m.like := m

  /**
   * Ideally, we would probably want to override equals(). But that is not
   * possible without modifying AbstractMatrix implementation in Mahout
   * which would require discussion at Mahout team.
   * @param that
   * @return
   */
  def equiv(that: Matrix) =
    that != null &&
        nrow == that.nrow &&
        m.view.zip(that).forall(t => {
          t._1.equiv(t._2)
        })

  def nequiv(that: Matrix) = !equiv(that)

  def ===(that: Matrix) = equiv(that)

  def !==(that: Matrix) = nequiv(that)

  /**
   * test if rank == min(nrow,ncol).
   * @return
   */
  def isFullRank: Boolean =
    new QRDecomposition(if (nrow < ncol) m t else m cloned).hasFullRank
}

object MatrixOps {
  implicit def m2ops(m: Matrix): MatrixOps = new MatrixOps(m)

  implicit def v2ops(v: Vector): VectorOps = new VectorOps(v)
}