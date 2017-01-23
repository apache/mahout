object GPUMMul extends MMulSolver {

  private final implicit val log = getLog(GPUMMul.getClass)

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
            case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) if a eq b.t ⇒ gpuDRWAAt
            case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) if a.t eq b ⇒ gpuDRWAAt
            case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.COLWISE, true) ⇒ gpuRWCW
            case (TraversingStructureEnum.ROWWISE, true, TraversingStructureEnum.ROWWISE, true) ⇒ jvmRWRW
            case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.COLWISE, true) ⇒ jvmCWCW
            case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) if a eq b.t ⇒ jvmDCWAAt
            case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) if a.t eq b ⇒ jvmDCWAAt
            case (TraversingStructureEnum.COLWISE, true, TraversingStructureEnum.ROWWISE, true) ⇒ jvmCWRW

            // Sparse row matrix x sparse row matrix (array of vectors)
            case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.ROWWISE, false) ⇒ gpuSparseRWRW
            case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.COLWISE, false) ⇒ jvmSparseRWCW
            case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.ROWWISE, false) ⇒ jvmSparseCWRW
            case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.COLWISE, false) ⇒ jvmSparseCWCW

            // Sparse matrix x sparse matrix (hashtable of vectors)
            case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.SPARSEROWWISE, false) ⇒
              gpuSparseRowRWRW
            case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.SPARSECOLWISE, false) ⇒
              jvmSparseRowRWCW
            case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.SPARSEROWWISE, false) ⇒
              jvmSparseRowCWRW
            case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.SPARSECOLWISE, false) ⇒
              jvmSparseRowCWCW

            // Sparse matrix x non-like
            case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ gpuSparseRowRWRW
            case (TraversingStructureEnum.SPARSEROWWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseRowRWCW
            case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ jvmSparseRowCWRW
            case (TraversingStructureEnum.SPARSECOLWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseCWCW
            case (TraversingStructureEnum.ROWWISE, _, TraversingStructureEnum.SPARSEROWWISE, false) ⇒ gpuSparseRWRW
            case (TraversingStructureEnum.ROWWISE, _, TraversingStructureEnum.SPARSECOLWISE, false) ⇒ jvmSparseRWCW
            case (TraversingStructureEnum.COLWISE, _, TraversingStructureEnum.SPARSEROWWISE, false) ⇒ jvmSparseCWRW
            case (TraversingStructureEnum.COLWISE, _, TraversingStructureEnum.SPARSECOLWISE, false) ⇒ jvmSparseRowCWCW

            // Everything else including at least one sparse LHS or RHS argument
            case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ gpuSparseRWRW
            case (TraversingStructureEnum.ROWWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseRWCW
            case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.ROWWISE, _) ⇒ jvmSparseCWRW
            case (TraversingStructureEnum.COLWISE, false, TraversingStructureEnum.COLWISE, _) ⇒ jvmSparseCWCW2flips

            // Sparse methods are only effective if the first argument is sparse, so we need to do a swap.
            case (_, _, _, false) ⇒ (a, b, r) ⇒ apply(b.t, a.t, r.map {
              _.t
            }).t

            // Default jvm-jvm case.
            // for some reason a SrarseRowMatrix DRM %*% SrarseRowMatrix DRM was dumping off to here
            case _ ⇒ gpuRWCW
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
  private def gpuRWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    println("gpuRWCW")
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
    if(b.getFlavor.isDense) {
      var ms = System.currentTimeMillis()
      val oclCtx = new Context(Context.OPENCL_MEMORY)
      val oclA = toVclCmpMatrixAlt(a, oclCtx)
      val oclB = toVclDenseRM(b, oclCtx)
      val oclC = new DenseRowMatrix(prod(oclA, oclB))
      val mxC = fromVclDenseRM(oclC)
      ms = System.currentTimeMillis() - ms
      debug(s"ViennaCL/OpenCL multiplication time: $ms ms.")

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
      val oclCtx = new Context(Context.OPENCL_MEMORY)
      val oclA = toVclCmpMatrixAlt(a, oclCtx)
      val oclB = toVclCmpMatrixAlt(b, oclCtx)
      val oclC = new CompressedMatrix(prod(oclA, oclB))
      val mxC = fromVclCompressedMatrix(oclC)
      ms = System.currentTimeMillis() - ms
      debug(s"ViennaCL/OpenCL multiplication time: $ms ms.")

      oclA.close()
      oclB.close()
      oclC.close()

      mxC
    }
  }


  @inline
  private def jvmRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    println("jvmRWRW")
    // A bit hackish: currently, this relies a bit on the fact that like produces RW(?)
    val bclone = b.like(b.ncol, b.nrow).t
    for (brow ← b) bclone(brow.index(), ::) := brow

    require(bclone.getFlavor.getStructure == TraversingStructureEnum.COLWISE || bclone.getFlavor.getStructure ==
      TraversingStructureEnum.SPARSECOLWISE, "COL wise conversion assumption of RHS is wrong, do over this code.")

    gpuRWCW(a, bclone, r)
  }

  private def jvmCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    println("jvmCWCW")
    jvmRWRW(b.t, a.t, r.map(_.t)).t
  }

  private def jvmCWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    println("jvmCWRW")
    // This is a primary contender with Outer Prod sum algo.
    // Here, we force-reorient both matrices and run RWCW.
    // A bit hackish: currently, this relies a bit on the fact that clone always produces RW(?)
    val aclone = a.cloned

    require(aclone.getFlavor.getStructure == TraversingStructureEnum.ROWWISE || aclone.getFlavor.getStructure ==
      TraversingStructureEnum.SPARSEROWWISE, "Row wise conversion assumption of RHS is wrong, do over this code.")

    jvmRWRW(aclone, b, r)
  }

  // left is Sparse right is any
  private def gpuSparseRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    println("gpuSparseRWRW")
    val mxR = r.getOrElse(b.like(a.nrow, b.ncol))


//    // This is basically almost the algorithm from SparseMatrix.times
//    for (arow ← a; ael ← arow.nonZeroes)
//      mxR(arow.index(), ::).assign(b(ael.index, ::), Functions.plusMult(ael))
//
//    mxR

    // make sure that the matrix is not empty.  VCL {{compressed_matrix}}s must
    // hav nnz > 0
    // this method is horribly inefficent.  however there is a difference between
    // getNumNonDefaultElements() and getNumNonZeroElements() which we do not always
    // have access to  created MAHOUT-1882 for this
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
    if(b.getFlavor.isDense) {
      var ms = System.currentTimeMillis()
      val oclCtx = new Context(Context.OPENCL_MEMORY)
      val oclA = toVclCmpMatrixAlt(a, oclCtx)
      val oclB = toVclDenseRM(b, oclCtx)
      val oclC = new DenseRowMatrix(prod(oclA, oclB))
      val mxC = fromVclDenseRM(oclC)
      ms = System.currentTimeMillis() - ms
      debug(s"ViennaCL/OpenCL multiplication time: $ms ms.")

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
      val oclCtx = new Context(Context.OPENCL_MEMORY)
      val oclA = toVclCmpMatrixAlt(a, oclCtx)
      val oclB = toVclCmpMatrixAlt(b, oclCtx)
      val oclC = new CompressedMatrix(prod(oclA, oclB))
      val mxC = fromVclCompressedMatrix(oclC)
      ms = System.currentTimeMillis() - ms
      debug(s"ViennaCL/OpenCL multiplication time: $ms ms.")

      oclA.close()
      oclB.close()
      oclC.close()

      mxC
    }

  }

  //sparse %*% dense
  private def gpuSparseRowRWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    println("gpuSparseRowRWRW")
    val hasElementsA = a.zSum() >  0

    // A has a sparse matrix structure of unknown size.  We do not want to
    // simply convert it to a Dense Matrix which may result in an OOM error.
    // If it is empty fall back to  JVM MMul, since we can not convert it
    // to a VCL CSR Matrix.
    if (!hasElementsA)  {
      println("Matrix a has zero elements can not convert to CSR")
      return MMul(a, b, r)
    }

    var ms = System.currentTimeMillis()
    val oclCtx = new Context(Context.OPENCL_MEMORY)
    val oclA = toVclCmpMatrixAlt(a, oclCtx)
    val oclB = toVclDenseRM(b, oclCtx)
    val oclC = new DenseRowMatrix(prod(oclA, oclB))
    val mxC = fromVclDenseRM(oclC)
    ms = System.currentTimeMillis() - ms
    debug(s"ViennaCL/OpenCL multiplication time: $ms ms.")

    oclA.close()
    oclB.close()
    oclC.close()

    mxC
  }

  private def jvmSparseRowCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRowRWRW(b.t, a.t, r.map(_.t)).t

  private def jvmSparseRowCWCW2flips(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRowRWRW(a cloned, b cloned, r)

  private def jvmSparseRowRWCW(a: Matrix, b: Matrix, r: Option[Matrix]) =
    gpuSparseRowRWRW(a, b cloned, r)


  private def jvmSparseRowCWRW(a: Matrix, b: Matrix, r: Option[Matrix]) =
    gpuSparseRowRWRW(a cloned, b, r)

  private def jvmSparseRWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRWRW(a, b.cloned, r)

  private def jvmSparseCWRW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRWRW(a cloned, b, r)

  private def jvmSparseCWCW(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRWRW(b.t, a.t, r.map(_.t)).t

  private def jvmSparseCWCW2flips(a: Matrix, b: Matrix, r: Option[Matrix] = None) =
    gpuSparseRWRW(a cloned, b cloned, r)

  private def jvmDiagRW(diagm:Matrix, b:Matrix, r:Option[Matrix] = None):Matrix = {
    println("jvmDiagRW")
    val mxR = r.getOrElse(b.like(diagm.nrow, b.ncol))

    for (del ← diagm.diagv.nonZeroes())
      mxR(del.index, ::).assign(b(del.index, ::), Functions.plusMult(del))

    mxR
  }

  private def jvmDiagCW(diagm: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    println("jvmDiagCW")
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
    gpuDRWAAt(a.cloned,null,r)
  }

  /** Dense Row-wise AA' */
  // we probably will not want to use this for the actual release unless A is cached already
  // but adding for testing purposes.
  private def gpuDRWAAt(a:Matrix, b:Matrix, r:Option[Matrix] = None) = {
    // a.t must be equiv to b.
    println("executing on gpu")
    debug("AAt computation detected; passing off to GPU")

    // Check dimensions if result is supplied.
    require(r.forall(mxR ⇒ mxR.nrow == a.nrow && mxR.ncol == a.nrow))

    val mxR = r.getOrElse(a.like(a.nrow, a.nrow))

    var ms = System.currentTimeMillis()
    val oclCtx = new Context(Context.OPENCL_MEMORY)
    val oclA = toVclDenseRM(src = a, oclCtx)
    val oclAt = new DenseRowMatrix(trans(oclA))
    val oclC = new DenseRowMatrix(prod(oclA, oclAt))

    val mxC = fromVclDenseRM(oclC)
    ms = System.currentTimeMillis() - ms
    debug(s"ViennaCL/OpenCL multiplication time: $ms ms.")

    oclA.close()
    //oclApr.close()
    oclAt.close()
    oclC.close()

    mxC

  }

  private def jvmOuterProdSum(a: Matrix, b: Matrix, r: Option[Matrix] = None): Matrix = {
    println("jvmOuterProdSum")
    // This may be already laid out for outer product computation, which may be faster than reorienting
    // both matrices? need to check.
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
