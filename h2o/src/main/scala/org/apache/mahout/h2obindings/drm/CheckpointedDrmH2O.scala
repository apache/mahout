package org.apache.mahout.h2obindings.drm

import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm._
import org.apache.mahout.h2obindings._

import scala.reflect._

/** H2O-specific optimizer-checkpointed DRM. */
class CheckpointedDrmH2O[K: ClassTag](
  val h2odrm: H2ODrm,
  val context: DistributedContext
) extends CheckpointedDrm[K] {

  def collect: Matrix = H2OHelper.matrix_from_drm(h2odrm)
  /* XXX: call frame.remove */
  def uncache(): Unit = return

  def writeDRM(path: String): Unit = H2OHdfs.drm_to_file(path, h2odrm)

  def checkpoint(cacheHint: CacheHint.CacheHint): CheckpointedDrm[K] = this

  def ncol: Int = h2odrm.frame.numCols

  def nrow: Long = h2odrm.frame.numRows

  protected[mahout] def partitioningTag: Long = h2odrm.frame.anyVec.group.hashCode
}
