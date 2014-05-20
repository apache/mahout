package org.apache.mahout.h2obindings.drm

import org.apache.mahout.math.{SparseMatrix, DenseMatrix, Matrix, Vector}
import math._
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm._
import org.apache.mahout.h2obindings._

import water._
import water.fvec._

import scala.reflect._

/** H2O-specific optimizer-checkpointed DRM. */
class CheckpointedDrmH2O[K: ClassTag](
    val frame: Frame
) extends CheckpointedDrm[K] {

  def collect: Matrix = ???
  def uncache(): Unit = ???
  def writeDRM(path: String): Unit = ???


  def checkpoint(cacheHint: CacheHint.CacheHint): CheckpointedDrm[K] = ???
  protected[mahout] val context: DistributedContext = ???
  def ncol: Int = ???
  def nrow: Long = ???
  protected[mahout] def partitioningTag: Long = ???
}
