package org.apache.mahout.h2obindings.drm

import org.apache.mahout.math.{Matrix, Vector}
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm._
import org.apache.mahout.h2obindings._

import water._
import water.fvec._

import scala.reflect._

/** H2O-specific optimizer-checkpointed DRM. */
class CheckpointedDrmH2O[K: ClassTag](
  val frame: Frame,
  val labels: Vec,
  protected[mahout] val context: DistributedContext
) extends CheckpointedDrm[K] {

  def this(frame: Frame, context: DistributedContext) =
    this(frame, null, context)

  def collect: Matrix = H2OHelper.matrix_from_frame(frame, labels)
  /* XXX: call frame.remove */
  def uncache(): Unit = return
  /* XXX: H2O does not support seqfile format yet */
  def writeDRM(path: String): Unit = ???


  def checkpoint(cacheHint: CacheHint.CacheHint): CheckpointedDrm[K] = this

  def ncol: Int = frame.numCols

  def nrow: Long = frame.numRows

  protected[mahout] def partitioningTag: Long = frame.anyVec.group.hashCode
}