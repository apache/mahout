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

import org.apache.mahout.math._
import org.apache.mahout.math.flavor.{BackEnum, TraversingStructureEnum}
import org.apache.mahout.math.function.Functions
import RLikeOps._
import org.apache.mahout.logging._

import scala.collection.JavaConversions._

object MMul extends MMBinaryFunc {

  private final implicit val log = getLog(MMul.getClass)

  override def apply(a: Matrix, b: Matrix, r: Option[Matrix]): Matrix = {

    require(a.ncol == b.nrow, "Incompatible matrix sizes in matrix multiplication.")

    val (af, bf) = (a.getFlavor, b.getFlavor)
    val backs = (af.getBacking, bf.getBacking)
    val sd = (af.getStructure, densityAnalysis(a), bf.getStructure, densityAnalysis(b))

    val alg: MMulAlg = backs match {

      // Both operands are jvm memory backs.
      case (BackEnum.JVMMEM, BackEnum.JVMMEM) ⇒

        sd match {

          // Multiplication cases by a diagonal matrix.
          case (TraversingStructureEnum.VECTORBACKED, _, TraversingStructureEnum.COLWISE, _)
            if a.isInstanceOf[DiagonalMatrix] ⇒ jvmDiagCW
          case (TraversingStructureEnum.VECTORBACKED, _, TraversingStructureEnum.SPARSECOLWISE, _)
            if a.isInstanceOf[DiagonalMatrix] ⇒ jvmDiagCW
          case (TraversingStructureEnum.VECTORBACKED, _, TraversingStructureEnum.ROWWISE, _)
            if a.isInstanceOf[DiagonalMatrix] ⇒ jvmDiagRW
          case (TraversingStructureEnum.VECTORBACKED, _, TraversingStructureEnum.SPARSEROWWISE, _)
            if a.isInstanceOf[DiagonalMatrix] ⇒ jvmDiagRW

          case (TraversingStructureEnum.COLWISE, _, TraversingStructureEnum.VECTORBACKED, _)
            if b.isInstanceOf[DiagonalMatrix] ⇒ jvmCWDiag
          case (TraversingStructureEnum.SPARSECOLWISE, _, TraversingStructureEnum.VECTORBACKED, _)
            if b.isInstanceOf[DiagonalMatrix] ⇒ jvmCWDiag
          case (TraversingStructureEnum.ROWWISE, _, TraversingStructureEnum.VECTORBACKED, _)
            if b.isInstanceOf[DiagonalMatrix] ⇒ jvmRWDiag
          case (TraversingStructureEnum.SPARSEROWWISE, _, TraversingStructureEnum.VECTORBACKED, _)
            if b.isInstanceOf[DiagonalMatrix] ⇒ jvmRWDiag

          // Dense-dense cases
          case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) if a eq b.t ⇒ jvmDRWAAt
          case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) if a.t eq b ⇒ jvmDRWAAt
          case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) ⇒ jvmRWCW
          case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.ROWWISE, true) ⇒ jvmRWRW
          case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.COLWISE, true) ⇒ jvmCWCW
          case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) if a eq b.t ⇒ jvmDCWAAt
          case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) if a.t eq b ⇒ jvmDCWAAt
          case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) ⇒ jvmCWRW

          // Sparse row matrix x sparse row matrix (array of vectors)
          case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.ROWWISE, false) ⇒ jvmSparseRWRW
          case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.COLWISE, false) ⇒ jvmSparseRWCW
          case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.ROWWISE, false) ⇒ jvmSparseCWRW
          case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.COLWISE, false) ⇒ jvmSparseCWCW

          // Sparse matrix x sparse matrix (hashtable of vectors)
          case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.SPARSEROWWISE, false) ⇒
            jvmSparseRowRWRW
          case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.SPARSECOLWISE, false) ⇒
            jvmSparseRowRWCW
          case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.SPARSEROWWISE, false) ⇒
            jvmSparseRowCWRW
          case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.SPARSECOLWISE, false) ⇒
            jvmSparseRowCWCW

          // Sparse matrix x non-like
          case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ jvmSparseRowRWRW
          case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseRowRWCW
          case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ jvmSparseRowCWRW
          case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseCWCW
          case (TraversingStructureEnum.ROWWISE, _, TraversingStructureEnum.SPARSEROWWISE, false) ⇒ jvmSparseRWRW
          case (TraversingStructureEnum.ROWWISE, _, TraversingStructureEnum.SPARSECOLWISE, false) ⇒ jvmSparseRWCW
          case (TraversingStructureEnum.COLWISE, _, TraversingStructureEnum.SPARSEROWWISE, false) ⇒ jvmSparseCWRW
          case (TraversingStructureEnum.COLWISE, _, TraversingStructureEnum.SPARSECOLWISE, false) ⇒ jvmSparseRowCWCW

          // Everything else including at least one sparse LHS or RHS argument
          case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ jvmSparseRWRW
          case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseRWCW
          case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ jvmSparseCWRW
          case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseCWCW2flips

          // Sparse methods are only effective if the first argument is sparse, so we need to do a swap.
          case (_, _, _, false) ⇒ (a, b, r) ⇒ apply(b.t, a.t, r.map {_.t}).t

          // Default jvm-jvm case.
          case _ ⇒ jvmRWCW
        }
    }

    alg(a, b, r)
  }

  type MMulAlg = MMBinaryFunc

  @inline
  private def jvmRWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {

    require(r.forall(mxR ⇒ mxR.nrow == a.nrow && mxR.ncol == b.ncol))
    val (m, n) = (a.nrow, b.ncol)

    val mxR = r.getOrElse(if (densityAnalysis(a)) a.like(m, n) else b.like(m, n))

    for (row ← 0 until mxR.nrow; col ← 0 until mxR.ncol) {
      // this vector-vector should be sort of optimized, right?
      mxR(row, col) = a(row, ::) dot b(::, col)
    }
    mxR
  }


  @inline
  private def jvmRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {

    // A bit hackish: currently, this relies a bit on the fact that like produces RW(?)
    val bclone = b.like(b.ncol, b.nrow).t
    for (brow ← b) bclone(brow.index(), ::) := brow

    require(bclone.getFlavor.getStructure == TraversingStructureEnum.COLWISE || bclone.getFlavor.getStructure ==
      TraversingStructureEnum.SPARSECOLWISE, "COL wise conversion assumption of RHS is wrong, do over this code.")

    jvmRWCW(a, bclone, r)
  }

  private def jvmCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    jvmRWRW(b.t, a.t, r.map(_.t)).t
  }

  private def jvmCWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    // This is a primary contender with Outer Prod sum algo.
    // Here, we force-reorient both matrices and run RWCW.
    // A bit hackish: currently, this relies a bit on the fact that clone always produces RW(?)
    val aclone = a.cloned

    require(aclone.getFlavor.getStructure == TraversingStructureEnum.ROWWISE || aclone.getFlavor.getStructure ==
      TraversingStructureEnum.SPARSEROWWISE, "Row wise conversion assumption of RHS is wrong, do over this code.")

    jvmRWRW(aclone, b, r)
  }

  private def jvmSparseRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    val mxR = r.getOrElse(b.like(a.nrow, b.ncol))

    // This is basically almost the algorithm from SparseMatrix.times
    for (arow ← a; ael ← arow.nonZeroes)
      mxR(arow.index(), ::).assign(b(ael.index, ::), Functions.plusMult(ael))

    mxR
  }

  private def jvmSparseRowRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    val mxR = r.getOrElse(b.like(a.nrow, b.ncol))
    for (arow ← a.iterateNonEmpty(); ael ← arow.vector.nonZeroes)
      mxR(arow.index(), ::).assign(b(ael.index, ::), Functions.plusMult(ael))

    mxR
  }

  private def jvmSparseRowCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    jvmSparseRowRWRW(b.t, a.t, r.map(_.t)).t

  private def jvmSparseRowCWCW2flips(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    jvmSparseRowRWRW(a cloned, b cloned, r)

  private def jvmSparseRowRWCW(a: Matrix, b: Matrix, r: Option[Matrix]) =
    jvmSparseRowRWRW(a, b cloned, r)


  private def jvmSparseRowCWRW(a: Matrix, b: Matrix, r: Option[Matrix]) =
    jvmSparseRowRWRW(a cloned, b, r)

  private def jvmSparseRWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    jvmSparseRWRW(a, b.cloned, r)

  private def jvmSparseCWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    jvmSparseRWRW(a cloned, b, r)

  private def jvmSparseCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    jvmSparseRWRW(b.t, a.t, r.map(_.t)).t

  private def jvmSparseCWCW2flips(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    jvmSparseRWRW(a cloned, b cloned, r)

  private def jvmDiagRW(diagm:Matrix, b:Matrix, r:Option[Matrix] = None):Matrix = {
    val mxR = r.getOrElse(b.like(diagm.nrow, b.ncol))

    for (del ← diagm.diagv.nonZeroes())
      mxR(del.index, ::).assign(b(del.index, ::), Functions.plusMult(del))

    mxR
  }

  private def jvmDiagCW(diagm: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    val mxR = r.getOrElse(b.like(diagm.nrow, b.ncol))
    for (bcol ← b.t) mxR(::, bcol.index()) := bcol * diagm.diagv
    mxR
  }

  private def jvmCWDiag(a: Matrix, diagm: Matrix, r: Option[Matrix] = None) =
    jvmDiagRW(diagm, a.t, r.map {_.t}).t

  private def jvmRWDiag(a: Matrix, diagm: Matrix, r: Option[Matrix] = None) =
    jvmDiagCW(diagm, a.t, r.map {_.t}).t


  /** Dense column-wise AA' */
  private def jvmDCWAAt(a:Matrix, b:Matrix, r:Option[Matrix] = None) = {
    // a.t must be equiv. to b. Cloning must rewrite to row-wise.
    jvmDRWAAt(a.cloned,null,r)
  }

  /** Dense Row-wise AA' */
  private def jvmDRWAAt(a:Matrix, b:Matrix, r:Option[Matrix] = None) = {
    // a.t must be equiv to b.

    debug("AAt computation detected.")

    // Check dimensions if result is supplied.
    require(r.forall(mxR ⇒ mxR.nrow == a.nrow && mxR.ncol == a.nrow))

    val mxR = r.getOrElse(a.like(a.nrow, a.nrow))

    // This is symmetric computation. Compile upper triangular first.
    for (row ← 0 until mxR.nrow) {
      // diagonal value
      mxR(row, row) = a(row, ::).aggregate(Functions.PLUS, Functions.SQUARE)

      for ( col ← row + 1 until mxR.ncol) {
        // this vector-vector should be sort of optimized, right?
        val v = a(row, ::) dot a(col, ::)

        mxR(row, col) = v
        mxR(col,row) = v
      }
    }

    mxR
  }

  private def jvmOuterProdSum(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {

    // This may be already laid out for outer product computation, which may be faster than reorienting
    // both matrices? need to check.
    val (m, n) = (a.nrow, b.ncol)

    // Prefer col-wise result iff a is dense and b is sparse. In all other cases default to row-wise.
    val preferColWiseR = densityAnalysis(a) && !densityAnalysis(b)

    val mxR = r.getOrElse {
      (densityAnalysis(a), preferColWiseR) match {
        case (false, false) ⇒ b.like(m, n)
        case (false, true) ⇒ b.like(n, m).t
        case (true, false) ⇒ a.like(m, n)
        case (true, true) ⇒ a.like(n, m).t
      }
    }

    // Loop outer products
    if (preferColWiseR) {
      // this means B is sparse and A is not, so we need to iterate over b values and update R columns with +=
      // one at a time.
      for ((acol, brow) ← a.t.zip(b); bel ← brow.nonZeroes) mxR(::, bel.index()) += bel * acol
    } else {
      for ((acol, brow) ← a.t.zip(b); ael ← acol.nonZeroes()) mxR(ael.index(), ::) += ael * brow
    }

    mxR
  }
}
