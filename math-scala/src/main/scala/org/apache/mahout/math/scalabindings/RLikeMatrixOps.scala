/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.mahout.math.scalabindings

import org.apache.mahout.math.function.Functions
import org.apache.mahout.math.{Vector, Matrix}
import scala.collection.JavaConversions._
import RLikeOps._

class RLikeMatrixOps(m: Matrix) extends MatrixOps(m) {

  /** Structure-optimized mmul */
  def %*%(that: Matrix) = MMul(m, that, None)

  def :%*%(that:Matrix) = %*%(that)

  def %*%:(that: Matrix) = that :%*% m

  /**
   * The "legacy" matrix-matrix multiplication.
   *
   * @param that right hand operand
   * @return matrix multiplication result
   * @deprecated use %*%
   */
  def %***%(that: Matrix) = m.times(that)

  /**
   * matrix-vector multiplication
   * @param that
   * @return
   */
  def %*%(that: Vector) = m.times(that)

  /**
   * Hadamard product
   *
   * @param that
   * @return
   */

  def *(that: Matrix) = cloned *= that

  def *(that: Double) = cloned *= that

  def *:(that:Double) = cloned *= that

  def /(that: Matrix) = cloned /= that

  def /:(that: Matrix) = that / m

  def /(that: Double) = cloned /= that

  /** 1.0 /: A is eqivalent to R's 1.0/A */
  def /:(that: Double) = that /=: cloned

  /**
   * in-place Hadamard product. We probably don't want to use assign
   * to optimize for sparse operations, in case of Hadamard product
   * it really can be done
   * @param that
   */
  def *=(that: Matrix) = {
    m.assign(that, Functions.MULT)
    m
  }

  /** A *=: B is equivalent to B *= A. Included for completeness. */
  def *=:(that: Matrix) = m *= that

  /** Elementwise deletion */
  def /=(that: Matrix) = {
    m.zip(that).foreach(t ⇒ t._1.vector() /= t._2.vector)
    m
  }

  def *=(that: Double) = {
    m.foreach(_.vector() *= that)
    m
  }

  /** 5.0 *=: A is equivalent to A *= 5.0. Included for completeness. */
  def *=:(that: Double) = m *= that

  def /=(that: Double) = {
    m ::= { x ⇒ x / that }
    m
  }

  /** 1.0 /=: A is equivalent to A = 1.0/A in R */
  def /=:(that: Double) = {
    if (that != 0.0) m := { x ⇒ that / x }
    m
  }

  def ^=(that: Double) = {
    that match {
      // Special handling of x ^2 and x ^ 0.5: we want consistent handling of x ^ 2 and x * x since
      // pow(x,2) function return results different from x * x; but much of the code uses this
      // interchangeably. Not having this done will create things like NaN entries on main diagonal
      // of a distance matrix.
      case 2.0 ⇒ m ::= { x ⇒ x * x }
      case 0.5 ⇒ m ::= math.sqrt _
      case _ ⇒ m ::= { x ⇒ math.pow(x, that) }
    }
  }

  def ^(that: Double) = m.cloned ^= that

  def cbind(that: Matrix): Matrix = {
    require(m.nrow == that.nrow)
    if (m.ncol > 0) {
      if (that.ncol > 0) {
        val mx = m.like(m.nrow, m.ncol + that.ncol)
        mx(::, 0 until m.ncol) := m
        mx(::, m.ncol until mx.ncol) := that
        mx
      } else m
    } else that
  }

  def cbind(that: Double): Matrix = {
    val mx = m.like(m.nrow, m.ncol + 1)
    mx(::, 0 until m.ncol) := m
    if (that != 0.0) mx(::, m.ncol) := that
    mx
  }

  def rbind(that: Matrix): Matrix = {
    require(m.ncol == that.ncol)
    if (m.nrow > 0) {
      if (that.nrow > 0) {
        val mx = m.like(m.nrow + that.nrow, m.ncol)
        mx(0 until m.nrow, ::) := m
        mx(m.nrow until mx.nrow, ::) := that
        mx
      } else m
    } else that
  }

  def rbind(that: Double): Matrix = {
    val mx = m.like(m.nrow + 1, m.ncol)
    mx(0 until m.nrow, ::) := m
    if (that != 0.0) mx(m.nrow, ::) := that
    mx
  }
}

