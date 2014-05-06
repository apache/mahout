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

import org.apache.mahout.math._
import org.apache.mahout.math.solver.EigenDecomposition

/**
 * Mahout matrices and vectors' scala syntactic sugar
 */
package object scalabindings {

  // Reserved "ALL" range
  final val `::`: Range = null

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
    import MatrixOps._
    val data = for (r <- rows) yield {
      r match {
        case n: Number => Array(n.doubleValue())
        case t: Product => t.productIterator.map(_.asInstanceOf[Number].doubleValue()).toArray
        case t: Vector => Array.tabulate(t.length)(t(_))
        case t: Array[Double] => t
        case t: Iterable[Double] => t.toArray
        case t: Array[Array[Double]] => if (rows.size == 1)
          return new DenseMatrix(t)
        else
          throw new IllegalArgumentException(
            "double[][] data parameter can be the only argument for dense()")
        case t:Array[Vector] =>
          val m = new DenseMatrix(t.size,t.head.length)
          t.view.zipWithIndex.foreach({case(v,idx) => m(idx,::) := v})
          return m
        case _ => throw new IllegalArgumentException("unsupported type in the inline Matrix initializer")
      }
    }
    new DenseMatrix(data.toArray)
  }

  /**
   * Default initializes are always row-wise.
   * create a sparse,
   * e.g.
   * m = sparse(
   * (0,5)::(9,3)::Nil,
   * (2,3.5)::(7,8)::Nil
   * )
   *
   * @param rows
   * @return
   */

  def sparse(rows: Vector*): SparseRowMatrix = {
    import MatrixOps._
    val nrow = rows.size
    val ncol = rows.map(_.size()).max
    val m = new SparseRowMatrix(nrow, ncol)
    m := rows
    m

  }

  /**
   * create a sparse vector out of list of tuple2's
   * @param sdata
   * @return
   */
  def svec(sdata: TraversableOnce[(Int, AnyVal)]) = {
    val cardinality = if (sdata.size > 0) sdata.map(_._1).max + 1 else 0
    val initialCapacity = sdata.size
    val sv = new RandomAccessSparseVector(cardinality, initialCapacity)
    sdata.foreach(t => sv.setQuick(t._1, t._2.asInstanceOf[Number].doubleValue()))
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
    import MatrixOps._
    val x = solve(a, b.toColMatrix)
    x(::, 0)
  }

  def ssvd(a: Matrix, k: Int, p: Int = 15, q: Int = 0) = SSVD.ssvd(a, k, p, q)

  /**
   * PCA based on SSVD that runs without forming an always-dense A-(colMeans(A)) input for SVD. This
   * follows the solution outlined in MAHOUT-817. For in-core version it, for most part, is supposed
   * to save some memory for sparse inputs by removing direct mean subtraction.<P>
   *
   * Hint: Usually one wants to use AV which is approsimately USigma, i.e.<code>u %*%: diagv(s)</code>.
   * If retaining distances and orignal scaled variances not that important, the normalized PCA space
   * is just U.
   *
   * Important: data points are considered to be rows.
   *
   * @param a input matrix A
   * @param k request SSVD rank
   * @param p oversampling parameter
   * @param q number of power iterations
   * @return (U,V,s)
   */
  def spca(a: Matrix, k: Int, p: Int = 15, q: Int = 0) =
    SSVD.spca(a = a, k = k, p = p, q = q)

}
