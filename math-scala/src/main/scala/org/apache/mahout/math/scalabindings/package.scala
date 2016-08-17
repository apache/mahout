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

package org.apache.mahout.math

import org.apache.mahout.math.solver.EigenDecomposition

import collection._
import scala.util.Random

/**
 * Mahout matrices and vectors' scala syntactic sugar
 */
package object scalabindings {


  // Reserved "ALL" range
  final val `::`: Range = null

  // values for stochastic sparsityAnalysis
  final val z95 = 1.959964
  final val z80 = 1.281552
  final val maxSamples = 500
  final val minSamples = 15

  // Some enums
  object AutoBooleanEnum extends Enumeration {
    type T = Value
    val TRUE, FALSE, AUTO = Value
  }

  implicit def seq2Vector(s: TraversableOnce[AnyVal]) =
    new DenseVector(s.map(_.asInstanceOf[Number].doubleValue()).toArray)

  implicit def tuple2TravOnce2svec[V <: AnyVal](sdata: TraversableOnce[(Int, V)]) = svec(sdata)

  implicit def t1vec(s: Tuple1[AnyVal]): Vector = prod2Vec(s)

  implicit def t2vec(s: Tuple2[AnyVal, AnyVal]): Vector = prod2Vec(s)

  implicit def t3vec(s: Tuple3[AnyVal, AnyVal, AnyVal]): Vector = prod2Vec(s)

  implicit def t4vec(s: Tuple4[AnyVal, AnyVal, AnyVal, AnyVal]): Vector = prod2Vec(s)

  implicit def t5vec(s: Tuple5[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal]): Vector = prod2Vec(s)

  implicit def t6vec(s: Tuple6[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal]): Vector = prod2Vec(s)

  implicit def t7vec(s: Tuple7[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal]): Vector = prod2Vec(s)

  implicit def t8vec(s: Tuple8[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal]): Vector = prod2Vec(s)

  implicit def t9vec(s: Tuple9[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal]): Vector =
    prod2Vec(s)

  implicit def t10vec(s: Tuple10[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal])
  : Vector = prod2Vec(s)

  implicit def t11vec(s: Tuple11[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal])
  : Vector = prod2Vec(s)

  implicit def t12vec(s: Tuple12[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal, AnyVal])
  : Vector = prod2Vec(s)

  implicit def t13vec(s: Tuple13[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal, AnyVal, AnyVal])
  : Vector = prod2Vec(s)

  implicit def t14vec(s: Tuple14[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal, AnyVal, AnyVal, AnyVal])
  : Vector = prod2Vec(s)

  implicit def t15vec(s: Tuple15[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal, AnyVal, AnyVal, AnyVal, AnyVal])
  : Vector = prod2Vec(s)

  implicit def t16vec(s: Tuple16[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal])
  : Vector = prod2Vec(s)

  implicit def t17vec(s: Tuple17[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal])
  : Vector = prod2Vec(s)

  implicit def t18vec(s: Tuple18[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal])
  : Vector = prod2Vec(s)

  implicit def t19vec(s: Tuple19[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal])
  : Vector = prod2Vec(s)

  implicit def t20vec(s: Tuple20[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal])
  : Vector = prod2Vec(s)

  implicit def t21vec(s: Tuple21[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal])
  : Vector = prod2Vec(s)

  implicit def t22vec(s: Tuple22[AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal
      , AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal, AnyVal])
  : Vector = prod2Vec(s)


  def prod2Vec(s: Product) = new DenseVector(s.productIterator.
      map(_.asInstanceOf[Number].doubleValue()).toArray)

  def diagv(v: Vector): DiagonalMatrix = new DiagonalMatrix(v)

  def diag(v: Double, size: Int): DiagonalMatrix =
    new DiagonalMatrix(new DenseVector(Array.fill(size)(v)))

  def eye(size: Int) = new DiagonalMatrix(1.0, size)

  /**
   * Create dense matrix out of inline arguments -- rows -- which can be tuples,
   * iterables of Double, or just single Number (for columnar vectors)
   * @param rows
   * @tparam R
   * @return
   */
  def dense[R](rows: R*): DenseMatrix = {
    import RLikeOps._
    val data = for (r ← rows) yield {
      r match {
        case n: Number ⇒ Array(n.doubleValue())
        case t: Vector ⇒ Array.tabulate(t.length)(t(_))
        case t: Array[Double] ⇒ t
        case t: Iterable[_] ⇒
          t.head match {
            case ss: Double ⇒ t.asInstanceOf[Iterable[Double]].toArray
            case vv: Vector ⇒
              val m = new DenseMatrix(t.size, t.head.asInstanceOf[Vector].length)
              t.asInstanceOf[Iterable[Vector]].view.zipWithIndex.foreach {
                case (v, idx) ⇒ m(idx, ::) := v
              }
              return m
          }
        case t: Product ⇒ t.productIterator.map(_.asInstanceOf[Number].doubleValue()).toArray
        case t: Array[Array[Double]] ⇒ if (rows.size == 1)
          return new DenseMatrix(t)
        else
          throw new IllegalArgumentException(
            "double[][] data parameter can be the only argument for dense()")
        case t: Array[Vector] ⇒
          val m = new DenseMatrix(t.size, t.head.length)
          t.view.zipWithIndex.foreach {
            case (v, idx) ⇒ m(idx, ::) := v
          }
          return m
        case _ ⇒ throw new IllegalArgumentException("unsupported type in the inline Matrix initializer")
      }
    }
    new DenseMatrix(data.toArray)
  }

  /**
   * Default initializes are always row-wise.
   * create a sparse,
   * e.g. {{{
   *
   * m = sparse(
   *   (0,5)::(9,3)::Nil,
   *   (2,3.5)::(7,8)::Nil
   * )
   *
   * }}}
   *
   * @param rows
   * @return
   */

  def sparse(rows: Vector*): SparseRowMatrix = {
    import RLikeOps._
    val nrow = rows.size
    val ncol = rows.map(_.size()).max
    val m = new SparseRowMatrix(nrow, ncol)
    m := rows.map { row ⇒
      if (row.length < ncol) {
        val newRow = row.like(ncol)
        newRow(0 until row.length) := row
        newRow
      }
      else row
    }
    m

  }

  /**
   * create a sparse vector out of list of tuple2's
   * @param sdata cardinality
   * @return
   */
  def svec(sdata: TraversableOnce[(Int, AnyVal)], cardinality: Int = -1) = {
    val required = if (sdata.nonEmpty) sdata.map(_._1).max + 1 else 0
    var tmp = -1
    if (cardinality < 0) {
      tmp = required
    } else if (cardinality < required) {
      throw new IllegalArgumentException(s"Required cardinality %required but got %cardinality")
    } else {
      tmp = cardinality
    }
    val initialCapacity = sdata.size
    val sv = new RandomAccessSparseVector(tmp, initialCapacity)
    sdata.foreach(t ⇒ sv.setQuick(t._1, t._2.asInstanceOf[Number].doubleValue()))
    sv
  }

  def dvec(fromV: Vector) = new DenseVector(fromV)

  def dvec(ddata: TraversableOnce[Double]) = new DenseVector(ddata.toArray)

  def dvec(numbers: Number*) = new DenseVector(numbers.map(_.doubleValue()).toArray)

  def chol(m: Matrix, pivoting: Boolean = false) = new CholeskyDecomposition(m, pivoting)

  /**
   * computes SVD
   * @param m svd input
   * @return (U,V, singular-values-vector)
   */
  def svd(m: Matrix) = {
    val svdObj = new SingularValueDecomposition(m)
    (svdObj.getU, svdObj.getV, new DenseVector(svdObj.getSingularValues))
  }

  /**
   * Computes Eigendecomposition of a symmetric matrix
   * @param m symmetric input matrix
   * @return (V, eigen-values-vector)
   */
  def eigen(m: Matrix) = {
    val ed = new EigenDecomposition(m, true)
    (ed.getV, ed.getRealEigenvalues)
  }


  /**
   * More general version of eigen decomposition
   * @param m
   * @param symmetric
   * @return (V, eigenvalues-real-vector, eigenvalues-imaginary-vector)
   */
  def eigenFull(m: Matrix, symmetric: Boolean = true) {
    val ed = new EigenDecomposition(m, symmetric)
    (ed.getV, ed.getRealEigenvalues, ed.getImagEigenvalues)
  }

  /**
   * QR.
   *
   * Right now Mahout's QR seems to be using argument for in-place transformations,
   * so the matrix context gets messed after this. Hence we force cloning of the
   * argument before passing it to Mahout's QR so to keep expected semantics.
   * @param m
   * @return (Q,R)
   */
  def qr(m: Matrix) = {
    import MatrixOps._
    val qrdec = new QRDecomposition(m cloned)
    (qrdec.getQ, qrdec.getR)
  }

  /**
   * Solution <tt>X</tt> of <tt>A*X = B</tt> using QR-Decomposition, where <tt>A</tt> is a square, non-singular matrix.
   *
   * @param a
   * @param b
   * @return (X)
   */
  def solve(a: Matrix, b: Matrix): Matrix = {
    import MatrixOps._
    if (a.nrow != a.ncol) {
      throw new IllegalArgumentException("supplied matrix A is not square")
    }
    val qr = new QRDecomposition(a cloned)
    if (!qr.hasFullRank) {
      throw new IllegalArgumentException("supplied matrix A is singular")
    }
    qr.solve(b)
  }

  /**
   * Solution <tt>A^{-1}</tt> of <tt>A*A^{-1} = I</tt> using QR-Decomposition, where <tt>A</tt> is a square,
   * non-singular matrix. Here only for compatibility with R semantics.
   *
   * @param a
   * @return (A^{-1})
   */
  def solve(a: Matrix): Matrix = {
    import MatrixOps._
    solve(a, eye(a.nrow))
  }

  /**
   * Solution <tt>x</tt> of <tt>A*x = b</tt> using QR-Decomposition, where <tt>A</tt> is a square, non-singular matrix.
   *
   * @param a
   * @param b
   * @return (x)
   */
  def solve(a: Matrix, b: Vector): Vector = {
    import RLikeOps._
    val x = solve(a, b.toColMatrix)
    x(::, 0)
  }

  ///////////////////////////////////////////////////////////
  // Elementwise unary functions. Actually this requires creating clones to avoid side effects. For
  // efficiency reasons one may want to actually do in-place exression assignments instead, e.g.
  //
  // m := exp _

  import RLikeOps._
  import scala.math._

  def mexp(m: Matrix): Matrix = m.cloned := exp _

  def vexp(v: Vector): Vector = v.cloned := exp _

  def mlog(m: Matrix): Matrix = m.cloned := log _

  def vlog(v: Vector): Vector = v.cloned := log _

  def mabs(m: Matrix): Matrix = m.cloned ::= (abs(_: Double))

  def vabs(v: Vector): Vector = v.cloned ::= (abs(_: Double))

  def msqrt(m: Matrix): Matrix = m.cloned ::= sqrt _

  def vsqrt(v: Vector): Vector = v.cloned ::= sqrt _

  def msignum(m: Matrix): Matrix = m.cloned ::= (signum(_: Double))

  def vsignum(v: Vector): Vector = v.cloned ::= (signum(_: Double))

  //////////////////////////////////////////////////////////
  // operation funcs


  /** Matrix-matrix unary func */
  type MMUnaryFunc = (Matrix, Option[Matrix]) ⇒ Matrix
  /** Binary matrix-matrix operations which may save result in-place, optionally */
  type MMBinaryFunc = (Matrix, Matrix, Option[Matrix]) ⇒ Matrix
  type MVBinaryFunc = (Matrix, Vector, Option[Matrix]) ⇒ Matrix
  type VMBinaryFunc = (Vector, Matrix, Option[Matrix]) ⇒ Matrix
  type MDBinaryFunc = (Matrix, Double, Option[Matrix]) ⇒ Matrix


  /////////////////////////////////////
  // Miscellaneous in-core utilities

  /**
   * Compute column-wise means and variances.
   *
   * @return colMeans → colVariances
   */
  def colMeanVars(mxA:Matrix): (Vector, Vector) = {
    val mu = mxA.colMeans()
    val variance = (mxA * mxA colMeans) -= mu ^ 2
    mu → variance
  }

  /**
   * Compute column-wise means and stdevs.
   * @param mxA input
   * @return colMeans → colStdevs
   */
  def colMeanStdevs(mxA:Matrix) = {
    val (mu, variance) = colMeanVars(mxA)
    mu → (variance ::= math.sqrt _)
  }

  /** Compute square distance matrix. We assume data points are row-wise, similar to R's dist(). */
  def sqDist(mxX: Matrix): Matrix = {

    val s = mxX ^ 2 rowSums

    (mxX %*% mxX.t) := { (r, c, x) ⇒ s(r) + s(c) - 2 * x}
  }

  /**
   * Pairwise squared distance computation.
   * @param mxX X, m x d
   * @param mxY Y, n x d
   * @return pairwise squaired distances of row-wise data points in X and Y (m x n)
   */
  def sqDist(mxX: Matrix, mxY: Matrix): Matrix = {

    val s = mxX ^ 2 rowSums

    val t = mxY ^ 2 rowSums

    // D = s*1' + 1*t' - 2XY'
    (mxX %*% mxY.t) := { (r, c, d) ⇒ s(r) + t(c) - 2.0 * d}
  }

  def dist(mxX: Matrix): Matrix = sqDist(mxX) := sqrt _

  def dist(mxX: Matrix, mxY: Matrix): Matrix = sqDist(mxX, mxY) := sqrt _

  /**
    * Check the density of an in-core matrix based on supplied criteria.
    * Returns true if we think mx is densier than threshold with at least 80% confidence.
    *
    * @param mx  The matrix to check density of.
    * @param threshold the threshold of non-zero elements above which we consider a Matrix Dense
    */
  def densityAnalysis(mx: Matrix, threshold: Double = 0.25): Boolean = {

    require(threshold >= 0.0 && threshold <= 1.0)
    var n = minSamples
    var mean = 0.0
    val rnd = new Random()
    val dimm = mx.nrow
    val dimn = mx.ncol
    val pq = threshold * (1 - threshold)

    for (s ← 0 until minSamples) {
      if (mx(rnd.nextInt(dimm), rnd.nextInt(dimn)) != 0.0) mean += 1
    }
    mean /= minSamples
    val iv = z80 * math.sqrt(pq / n)

    if (mean < threshold - iv) return false // sparse
    else if (mean > threshold + iv) return true // dense

    while (n < maxSamples) {
      // Determine upper bound we may need for n to likely relinquish the uncertainty. Here, we use
      // confidence interval formula but solved for n.
      val ivNeeded = math.abs(threshold - mean) max 1e-11

      val stderr = ivNeeded / z80
      val nNeeded = (math.ceil(pq / (stderr * stderr)).toInt max n min maxSamples) - n

      var meanNext = 0.0
      for (s ← 0 until nNeeded) {
        if (mx(rnd.nextInt(dimm), rnd.nextInt(dimn)) != 0.0) meanNext += 1
      }
      mean = (n * mean + meanNext) / (n + nNeeded)
      n += nNeeded

      // Are we good now?
      val iv = z80 * math.sqrt(pq / n)
      if (mean < threshold - iv) return false // sparse
      else if (mean > threshold + iv) return true // dense
    }

    return mean > threshold // if (mean > threshold) dense

  }



}
