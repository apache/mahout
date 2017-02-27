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

package org.apache.mahout.viennacl.openmp

import org.apache.mahout.logging._
import org.apache.mahout.math
import org.apache.mahout.math._
import org.apache.mahout.math.backend.incore.MMulSolver
import org.apache.mahout.math.flavor.{BackEnum, TraversingStructureEnum}
import org.apache.mahout.math.function.Functions
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._
import org.apache.mahout.viennacl.openmp.javacpp.Functions._
import org.apache.mahout.viennacl.openmp.javacpp.LinalgFunctions._
import org.apache.mahout.viennacl.openmp.javacpp.{CompressedMatrix, Context, DenseRowMatrix}

import scala.collection.JavaConversions._

object OMPMMul extends MMBinaryFunc {

  private implicit val log = getLog(OMPMMul.getClass)

  override def apply(a: Matrix, b: Matrix, r: Option[Matrix]): Matrix = {

    require(a.ncol == b.nrow, "Incompatible matrix sizes in matrix multiplication.")

    val (af, bf) = (a.getFlavor, b.getFlavor)
    val backs = (af.getBacking, bf.getBacking)
    val sd = (af.getStructure, math.scalabindings.densityAnalysis(a), bf.getStructure, densityAnalysis(b))


    try {

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
            case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) if a eq b.t ⇒ ompDRWAAt
            case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) if a.t eq b ⇒ ompDRWAAt
            case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) ⇒ ompRWCW
            case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.ROWWISE, true) ⇒ jvmRWRW
            case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.COLWISE, true) ⇒ jvmCWCW
            case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) if a eq b.t ⇒ jvmDCWAAt
            case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) if a.t eq b ⇒ jvmDCWAAt
            case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) ⇒ jvmCWRW

            // Sparse row matrix x sparse row matrix (array of vectors)
            case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.ROWWISE, false) ⇒ ompSparseRWRW
            case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.COLWISE, false) ⇒ jvmSparseRWCW
            case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.ROWWISE, false) ⇒ jvmSparseCWRW
            case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.COLWISE, false) ⇒ jvmSparseCWCW

            // Sparse matrix x sparse matrix (hashtable of vectors)
            case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.SPARSEROWWISE, false) ⇒
              ompSparseRowRWRW
            case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.SPARSECOLWISE, false) ⇒
              jvmSparseRowRWCW
            case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.SPARSEROWWISE, false) ⇒
              jvmSparseRowCWRW
            case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.SPARSECOLWISE, false) ⇒
              jvmSparseRowCWCW

            // Sparse matrix x non-like
            case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ ompSparseRowRWRW
            case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseRowRWCW
            case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ jvmSparseRowCWRW
            case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseCWCW
            case (TraversingStructureEnum.ROWWISE, _, TraversingStructureEnum.SPARSEROWWISE, false) ⇒ ompSparseRWRW
            case (TraversingStructureEnum.ROWWISE, _, TraversingStructureEnum.SPARSECOLWISE, false) ⇒ jvmSparseRWCW
            case (TraversingStructureEnum.COLWISE, _, TraversingStructureEnum.SPARSEROWWISE, false) ⇒ jvmSparseCWRW
            case (TraversingStructureEnum.COLWISE, _, TraversingStructureEnum.SPARSECOLWISE, false) ⇒ jvmSparseRowCWCW

            // Everything else including at least one sparse LHS or RHS argument
            case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ ompSparseRWRW
            case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseRWCW
            case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ jvmSparseCWRW
            case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseCWCW2flips

            // Sparse methods are only effective if the first argument is sparse, so we need to do a swap.
            case (_, _, _, false) ⇒ (a, b, r) ⇒ apply(b.t, a.t, r.map {
              _.t
            }).t

            // Default jvm-jvm case.
            // for some reason a SrarseRowMatrix DRM %*% SrarseRowMatrix DRM was dumping off to here
            case _ ⇒ ompRWCW
          }
      }

      alg(a, b, r)
    } catch {
      // TODO FASTHACK:  just revert to JVM if there is an exception..
      //  eg. java.lang.nullPointerException if more openCL contexts
      //  have been created than number of GPU cards.
      //  better option wuold be to fall back to OpenCl First.
      case ex: Exception =>
        println(ex.getMessage + "falling back to JVM MMUL")
        return MMul(a, b, r)
    }
  }

  type MMulAlg = MMBinaryFunc

  @inline
  private def ompRWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    println("ompRWCW")
    //
    //    require(r.forall(mxR ⇒ mxR.nrow == a.nrow && mxR.ncol == b.ncol))
    //    val (m, n) = (a.nrow, b.ncol)
    //
    //    val mxR = r.getOrElse(if (densityAnalysis(a)) a.like(m, n) else b.like(m, n))
    //
    //    for (row ← 0 until mxR.nrow; col ← 0 until mxR.ncol) {
    //      // this vector-vector should be sort of optimized, right?
    //      mxR(row, col) = a(row, ::) dot b(::, col)
    //    }
    //    mxR

    val hasElementsA = a.zSum() >  0.0
    val hasElementsB = b.zSum() >  0.0

    // A has a sparse matrix structure of unknown size.  We do not want to
    // simply convert it to a Dense Matrix which may result in an OOM error.

    // If it is empty use JVM MMul, since we can not convert it to a VCL CSR Matrix.
    if (!hasElementsA)  {
      println("Matrix a has zero elements can not convert to CSR")
      return MMul(a, b, r)
    }

    // CSR matrices are efficient up to 50% non-zero
    if (b.getFlavor.isDense) {
      var ms = System.currentTimeMillis()
      val oclCtx = new Context(Context.MAIN_MEMORY)
      val oclA = toVclCmpMatrixAlt(a, oclCtx)
      val oclB = toVclDenseRM(b, oclCtx)
      val oclC = new DenseRowMatrix(prod(oclA, oclB))
      val mxC = fromVclDenseRM(oclC)
      ms = System.currentTimeMillis() - ms
      debug(s"ViennaCL/OpenMP multiplication time: $ms ms.")

      oclA.close()
      oclB.close()
      oclC.close()

      mxC
    } else {
      // Fall back to JVM based MMul if either matrix is sparse and empty
      if (!hasElementsA || !hasElementsB)  {
        println("Matrix a or b has zero elements can not convert to CSR")
        return MMul(a, b, r)
      }

      var ms = System.currentTimeMillis()
      val hostClCtx = new Context(Context.MAIN_MEMORY)
      val oclA = toVclCmpMatrixAlt(a, hostClCtx)
      val oclB = toVclCmpMatrixAlt(b, hostClCtx)
      val oclC = new CompressedMatrix(prod(oclA, oclB))
      val mxC = fromVclCompressedMatrix(oclC)
      ms = System.currentTimeMillis() - ms
      debug(s"ViennaCL/OpenMP multiplication time: $ms ms.")

      oclA.close()
      oclB.close()
      oclC.close()

      mxC
    }
  }


  @inline
  private def jvmRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    log.info("Using jvmRWRW method")
    // A bit hackish: currently, this relies a bit on the fact that like produces RW(?)
    val bclone = b.like(b.ncol, b.nrow).t
    for (brow ← b) bclone(brow.index(), ::) := brow

    require(bclone.getFlavor.getStructure == TraversingStructureEnum.COLWISE || bclone.getFlavor.getStructure ==
      TraversingStructureEnum.SPARSECOLWISE, "COL wise conversion assumption of RHS is wrong, do over this code.")

    ompRWCW(a, bclone, r)
  }

  private def jvmCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    log.info("Using jvmCWCW method")
    jvmRWRW(b.t, a.t, r.map(_.t)).t
  }

  private def jvmCWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    log.info("Using jvmCWRW method")
    // This is a primary contender with Outer Prod sum algo.
    // Here, we force-reorient both matrices and run RWCW.
    // A bit hackish: currently, this relies a bit on the fact that clone always produces RW(?)
    val aclone = a.cloned

    require(aclone.getFlavor.getStructure == TraversingStructureEnum.ROWWISE || aclone.getFlavor.getStructure ==
      TraversingStructureEnum.SPARSEROWWISE, "Row wise conversion assumption of RHS is wrong, do over this code.")

    jvmRWRW(aclone, b, r)
  }

  // left is Sparse right is any
  private def ompSparseRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    log.info("Using ompSparseRWRW method")
    val mxR = r.getOrElse(b.like(a.nrow, b.ncol))

    /* Make sure that the matrix is not empty.  VCL {{compressed_matrix}}s must
       have nnz > 0
       N.B. This method is horribly inefficent. However there is a difference between
       getNumNonDefaultElements() and getNumNonZeroElements() which we do not always
       have access to. We created MAHOUT-1882 for this.
    */

    val hasElementsA = a.zSum() >  0.0
    val hasElementsB = b.zSum() >  0.0

    // A has a sparse matrix structure of unknown size.  We do not want to
    // simply convert it to a Dense Matrix which may result in an OOM error.
    // If it is empty use JVM MMul, since we can not convert it to a VCL CSR Matrix.
    if (!hasElementsA)  {
      log.warn("Matrix a has zero elements can not convert to CSR")
      return MMul(a, b, r)
    }

    // CSR matrices are efficient up to 50% non-zero
    if(b.getFlavor.isDense) {
      var ms = System.currentTimeMillis()
      val hostClCtx = new Context(Context.MAIN_MEMORY)
      val oclA = toVclCmpMatrixAlt(a, hostClCtx)
      val oclB = toVclDenseRM(b, hostClCtx)
      val oclC = new DenseRowMatrix(prod(oclA, oclB))
      val mxC = fromVclDenseRM(oclC)
      ms = System.currentTimeMillis() - ms
      log.debug(s"ViennaCL/OpenMP multiplication time: $ms ms.")

      oclA.close()
      oclB.close()
      oclC.close()

      mxC
    } else {
      // Fall back to JVM based MMul if either matrix is sparse and empty
      if (!hasElementsA || !hasElementsB)  {
        log.warn("Matrix a or b has zero elements can not convert to CSR")
        return MMul(a, b, r)
      }

      var ms = System.currentTimeMillis()
      val hostClCtx = new Context(Context.MAIN_MEMORY)
      val oclA = toVclCmpMatrixAlt(a, hostClCtx)
      val oclB = toVclCmpMatrixAlt(b, hostClCtx)
      val oclC = new CompressedMatrix(prod(oclA, oclB))
      val mxC = fromVclCompressedMatrix(oclC)
      ms = System.currentTimeMillis() - ms
      log.debug(s"ViennaCL/OpenMP multiplication time: $ms ms.")

      oclA.close()
      oclB.close()
      oclC.close()

      mxC
    }

  }

  //sparse %*% dense
  private def ompSparseRowRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    log.info("Using ompSparseRowRWRW method")
    val hasElementsA = a.zSum() >  0

    // A has a sparse matrix structure of unknown size.  We do not want to
    // simply convert it to a Dense Matrix which may result in an OOM error.
    // If it is empty fall back to  JVM MMul, since we can not convert it
    // to a VCL CSR Matrix.
    if (!hasElementsA)  {
      log.warn("Matrix a has zero elements can not convert to CSR")
      return MMul(a, b, r)
    }

    var ms = System.currentTimeMillis()
    val hostClCtx = new Context(Context.MAIN_MEMORY)
    val oclA = toVclCmpMatrixAlt(a, hostClCtx)
    val oclB = toVclDenseRM(b, hostClCtx)
    val oclC = new DenseRowMatrix(prod(oclA, oclB))
    val mxC = fromVclDenseRM(oclC)
    ms = System.currentTimeMillis() - ms
    log.debug(s"ViennaCL/OpenMP multiplication time: $ms ms.")

    oclA.close()
    oclB.close()
    oclC.close()

    mxC
  }

  private def jvmSparseRowCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    ompSparseRowRWRW(b.t, a.t, r.map(_.t)).t

  private def jvmSparseRowCWCW2flips(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    ompSparseRowRWRW(a cloned, b cloned, r)

  private def jvmSparseRowRWCW(a: Matrix, b: Matrix, r: Option[Matrix]) =
    ompSparseRowRWRW(a, b cloned, r)

  private def jvmSparseRowCWRW(a: Matrix, b: Matrix, r: Option[Matrix]) =
    ompSparseRowRWRW(a cloned, b, r)

  private def jvmSparseRWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    ompSparseRWRW(a, b.cloned, r)

  private def jvmSparseCWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    ompSparseRWRW(a cloned, b, r)

  private def jvmSparseCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    ompSparseRWRW(b.t, a.t, r.map(_.t)).t

  private def jvmSparseCWCW2flips(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    ompSparseRWRW(a cloned, b cloned, r)

  private def jvmDiagRW(diagm:Matrix, b:Matrix, r:Option[Matrix] = None):Matrix = {
    log.info("Using jvmDiagRW method")
    val mxR = r.getOrElse(b.like(diagm.nrow, b.ncol))

    for (del ← diagm.diagv.nonZeroes())
      mxR(del.index, ::).assign(b(del.index, ::), Functions.plusMult(del))

    mxR
  }

  private def jvmDiagCW(diagm: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    log.info("Using jvmDiagCW method")
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
    ompDRWAAt(a.cloned,null,r)
  }

  /** Dense Row-wise AA' */
  // We probably will not want to use this for the actual release unless A is cached already
  // but adding for testing purposes.
  private def ompDRWAAt(a:Matrix, b:Matrix, r:Option[Matrix] = None) = {
    // a.t must be equiv to b.
    log.info("Executing on OMP")
    log.debug("AAt computation detected; passing off to OMP")

    // Check dimensions if result is supplied.
    require(r.forall(mxR ⇒ mxR.nrow == a.nrow && mxR.ncol == a.nrow))

    val mxR = r.getOrElse(a.like(a.nrow, a.nrow))

    var ms = System.currentTimeMillis()
    val hostClCtx = new Context(Context.MAIN_MEMORY)
    val oclA = toVclDenseRM(src = a, hostClCtx)
    val oclAt = new DenseRowMatrix(trans(oclA))
    val oclC = new DenseRowMatrix(prod(oclA, oclAt))

    val mxC = fromVclDenseRM(oclC)
    ms = System.currentTimeMillis() - ms
    log.debug(s"ViennaCL/OpenMP multiplication time: $ms ms.")

    oclA.close()
    //oclApr.close()
    oclAt.close()
    oclC.close()

    mxC

  }

  private def jvmOuterProdSum(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    log.info("Using jvmOuterProdSum method")
    // Need to check whether this is already laid out for outer product computation, which may be faster than
    // reorienting both matrices.
    val (m, n) = (a.nrow, b.ncol)

    // Prefer col-wise result iff a is dense and b is sparse. In all other cases default to row-wise.
    val preferColWiseR = a.getFlavor.isDense && !b.getFlavor.isDense

    val mxR = r.getOrElse {
      (a.getFlavor.isDense, preferColWiseR) match {
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
