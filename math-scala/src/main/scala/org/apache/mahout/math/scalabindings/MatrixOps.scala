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

import org.apache.mahout.math.flavor.TraversingStructureEnum
import org.apache.mahout.math.function.{DoubleFunction, Functions, VectorFunction}
import org.apache.mahout.math.{Matrices, Matrix, QRDecomposition, Vector}

import scala.collection.JavaConversions._
import scala.collection._
import scala.math._

class MatrixOps(val m: Matrix) {

  import MatrixOps._

  // We need this for some functions below (but it would screw some functions above)
  import RLikeOps.v2vOps

  def nrow = m.rowSize()

  def ncol = m.columnSize()

  /**
   * Warning: this creates a clone (as in mx * -1), in many applications inplace inversion `mx *= -1`
   * might be an infinitely better choice.
   */
  def unary_- = cloned.assign(Functions.NEGATE)

  def +=(that: Matrix) = m.assign(that, Functions.PLUS)

  def +=:(that:Matrix) = m += that

  def +=:(that:Double) = m += that

  def -=(that: Matrix) = m.assign(that, Functions.MINUS)

  def +=(that: Double) = m.assign(new DoubleFunction {
    def apply(x: Double): Double = x + that
  })

  def -=(that: Double) = +=(-that)

  def -=:(that: Double) = m.assign(Functions.minus(that))

  /** A := B - A which is -(A - B) */
  def -=:(that: Matrix) = m.assign(that, Functions.chain(Functions.NEGATE, Functions.MINUS))

  def +(that: Matrix) = cloned += that

  def -(that: Matrix) = cloned -= that

  def -:(that: Matrix) = that - m

  // m.plus(that)?

  def +(that: Double) = cloned += that

  def +:(that:Double) = cloned += that

  def -(that: Double) = cloned -= that

  def -:(that: Double) = that -=: cloned

  def norm = math.sqrt(m.aggregate(Functions.PLUS, Functions.SQUARE))

  def pnorm(p: Int) = pow(m.aggregate(Functions.PLUS, Functions.chain(Functions.ABS, Functions.pow(p))), 1.0 / p)

  def apply(row: Int, col: Int) = m.get(row, col)

  def update(row: Int, col: Int, that: Double): Matrix = {
    m.setQuick(row, col, that)
    m
  }

  def update(rowRange: Range, colRange: Range, that: Double) = apply(rowRange, colRange) := that

  def update(row: Int, colRange: Range, that: Double) = apply(row, colRange) := that

  def update(rowRange: Range, col: Int, that: Double) = apply(rowRange, col) := that

  def update(rowRange: Range, colRange: Range, that: Matrix) = apply(rowRange, colRange) := that

  def update(row: Int, colRange: Range, that: Vector) = apply(row, colRange) := that

  def update(rowRange: Range, col: Int, that: Vector) = apply(rowRange, col) := that
  
  
  def apply(rowRange: Range, colRange: Range): Matrix = {

    if (rowRange == :: &&
        colRange == ::) return m

    val rr = if (rowRange == ::) 0 until m.nrow
    else rowRange
    val cr = if (colRange == ::) 0 until m.ncol
    else colRange

    m.viewPart(rr.start, rr.length, cr.start, cr.length)

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
    * Apply a function element-wise without side-effects to the argument (creates a new matrix).
    *
    * @param f         element-wise function "value" ⇒ "new value"
    * @param evalZeros Do we have to process zero elements? true, false, auto: if auto, we will test
    *                  the supplied function for `f(0) != 0`, and depending on the result, will
    *                  decide if we want evaluation for zero elements. WARNING: the AUTO setting
    *                  may not always work correctly for functions that are meant to run in a specific
    *                  backend context, or non-deterministic functions, such as {-1,0,1} random
    *                  generators.
    * @return new DRM with the element-wise function applied.
    */
  def apply(f: Double ⇒ Double, evalZeros: AutoBooleanEnum.T): Matrix = {
    val ezeros = evalZeros match {
      case AutoBooleanEnum.TRUE ⇒ true
      case AutoBooleanEnum.FALSE ⇒ false
      case AutoBooleanEnum.AUTO ⇒ f(0) != 0
    }
    if (ezeros) m.cloned := f else m.cloned ::= f
  }

  /**
    * Apply a function element-wise without side-effects to the argument (creates a new matrix).
    *
    * @param f         element-wise function (row, column, value) ⇒ "new value"
    * @param evalZeros Do we have to process zero elements? true, false, auto: if auto, we will test
    *                  the supplied function for `f(0) != 0`, and depending on the result, will
    *                  decide if we want evaluation for zero elements. WARNING: the AUTO setting
    *                  may not always work correctly for functions that are meant to run in a specific
    *                  backend context, or non-deterministic functions, such as {-1,0,1} random
    *                  generators.
    * @return new DRM with the element-wise function applied.
    */
  def apply(f: (Int, Int, Double) ⇒ Double, evalZeros: AutoBooleanEnum.T): Matrix = {
    val ezeros = evalZeros match {
      case AutoBooleanEnum.TRUE ⇒ true
      case AutoBooleanEnum.FALSE ⇒ false
      case AutoBooleanEnum.AUTO ⇒ f(0,0,0) != 0
    }
    if (ezeros) m.cloned := f else m.cloned ::= f
  }

  /** A version of function apply with default AUTO treatment of `evalZeros`. */
  def apply(f: Double ⇒ Double): Matrix = apply(f, AutoBooleanEnum.AUTO)

  /** A version of function apply with default AUTO treatment of `evalZeros`. */
  def apply(f: (Int, Int, Double) ⇒ Double): Matrix = apply(f, AutoBooleanEnum.AUTO)


  /**
    * Warning: This provides read-only view only.
   * In most cases that's what one wants. To get a copy,
   * use <code>m.t cloned</code>
    *
    * @return transposed view
   */
  def t = Matrices.transposedView(m)

  def det = m.determinant()

  def sum = m.zSum()

  def :=(that: Matrix) = m.assign(that)

  /**
   * Assigning from a row-wise collection of vectors
    *
    * @param that -
   */
  def :=(that: TraversableOnce[Vector]) = {
    var row = 0
    that.foreach(v => {
      m.assignRow(row, v)
      row += 1
    })
  }

  def :=(that: Double) = m.assign(that)

  def :=(f: (Int, Int, Double) => Double): Matrix = {
    import RLikeOps._
    m.getFlavor.getStructure match {
      case TraversingStructureEnum.COLWISE | TraversingStructureEnum.SPARSECOLWISE =>
        for (col <- t; el <- col.all) el := f(el.index, col.index, el)
      case default =>
        for (row <- m; el <- row.all) el := f(row.index, el.index, el)
    }
    m
  }

  /** Functional assign with (Double) => Double */
  def :=(f: (Double) => Double): Matrix = {
    import RLikeOps._
    m.getFlavor.getStructure match {
      case TraversingStructureEnum.COLWISE | TraversingStructureEnum.SPARSECOLWISE =>
        for (col <- t; el <- col.all) el := f(el)
      case default =>
        for (row <- m; el <- row.all) el := f(el)
    }
    m
  }

  /** Sparse assign: iterate and assign over non-zeros only */
  def ::=(f: (Int, Int, Double) => Double): Matrix = {

    import RLikeOps._

    m.getFlavor.getStructure match {
      case TraversingStructureEnum.COLWISE | TraversingStructureEnum.SPARSECOLWISE =>
        for (col <- t; el <- col.nonZeroes) el := f(el.index, col.index, el)
      case default =>
        for (row <- m; el <- row.nonZeroes) el := f(row.index, el.index, el)
    }
    m
  }

  /** Sparse function assign: iterate and assign over non-zeros only */
  def ::=(f: (Double) => Double): Matrix = {

    import RLikeOps._

    m.getFlavor.getStructure match {
      case TraversingStructureEnum.COLWISE | TraversingStructureEnum.SPARSECOLWISE =>
        for (col <- t; el <- col.nonZeroes) el := f(el)
      case default =>
        for (row <- m; el <- row.nonZeroes) el := f(el)
    }
    m
  }

    def cloned: Matrix = m.like := m

  /**
   * Ideally, we would probably want to override equals(). But that is not
   * possible without modifying AbstractMatrix implementation in Mahout
   * which would require discussion at Mahout team.
    *
    * @param that
   * @return
   */
  def equiv(that: Matrix) =

  // Warning: TODO: This would actually create empty objects in SparseMatrix. Should really implement
  // merge-type comparison strategy using iterateNonEmpty.
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
    *
    * @return
   */
  def isFullRank: Boolean =
    new QRDecomposition(if (nrow < ncol) m t else m cloned).hasFullRank

  def colSums() = m.aggregateColumns(vectorSumFunc)

  def rowSums() = m.aggregateRows(vectorSumFunc)

  def colMeans() = if (m.nrow == 0) colSums() else colSums() /= m.nrow

  def rowMeans() = if (m.ncol == 0) rowSums() else rowSums() /= m.ncol

  /* Diagonal */
  def diagv: Vector = m.viewDiagonal()

  /* Diagonal assignment */
  def diagv_=(that: Vector) = diagv := that

  /* Diagonal assignment */
  def diagv_=(that: Double) = diagv := that

  /* Row and Column non-zero element counts */
  def numNonZeroElementsPerColumn() = m.aggregateColumns(vectorCountNonZeroElementsFunc)

  def numNonZeroElementsPerRow() = m.aggregateRows(vectorCountNonZeroElementsFunc)
}

object MatrixOps {

  import RLikeOps.v2vOps

  implicit def m2ops(m: Matrix): MatrixOps = new MatrixOps(m)

  private def vectorSumFunc = new VectorFunction {
    def apply(f: Vector): Double = f.sum
  }

  private def vectorCountNonZeroElementsFunc = new VectorFunction {
    //def apply(f: Vector): Double = f.aggregate(Functions.PLUS, Functions.notEqual(0))
    def apply(f: Vector): Double = f.getNumNonZeroElements().toDouble
  }

}