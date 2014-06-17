package org.apache.mahout.math.drm.logical

import scala.reflect.ClassTag
import org.apache.mahout.math.drm.DrmLike
import scala.util.Random

/** cbind() logical operator */
case class OpCbind[K: ClassTag](
    override var A: DrmLike[K],
    override var B: DrmLike[K]
    ) extends AbstractBinaryOp[K, K, K] {

  assert(A.nrow == B.nrow, "arguments must have same number of rows")

  override protected[mahout] lazy val partitioningTag: Long =
    if (A.partitioningTag == B.partitioningTag) A.partitioningTag
    else Random.nextLong()

  /** R-like syntax for number of rows. */
  def nrow: Long = A.nrow

  /** R-like syntax for number of columns */
  def ncol: Int = A.ncol + B.ncol

}
