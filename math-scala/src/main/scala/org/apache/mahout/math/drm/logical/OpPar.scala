package org.apache.mahout.math.drm.logical

import org.apache.mahout.math.drm.DrmLike

/** Parallelism operator */
case class OpPar[K](
    override var A: DrmLike[K],
    val minSplits: Int = -1,
    val exactSplits: Int = -1)
    extends AbstractUnaryOp[K, K] {

  /**
    * Explicit extraction of key class Tag since traits don't support context bound access; but actual
    * implementation knows it
    */
  override def keyClassTag = A.keyClassTag

  /** R-like syntax for number of rows. */
  def nrow: Long = A.nrow

  /** R-like syntax for number of columns */
  def ncol: Int = A.ncol
}
