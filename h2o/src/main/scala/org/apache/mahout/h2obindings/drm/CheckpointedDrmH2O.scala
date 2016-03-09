package org.apache.mahout.h2obindings.drm

import org.apache.mahout.h2obindings._
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.drm.CacheHint.CacheHint
import org.apache.mahout.math.drm._

import scala.reflect._

/**
  * H2O-specific optimizer-checkpointed DRM.
  *
  * @param h2odrm Underlying Frame and optional label Vec to wrap around
  * @param context Distributed context to the H2O Cloud
  * @tparam K Matrix key type
  */
class CheckpointedDrmH2O[K: ClassTag](
  val h2odrm: H2ODrm,
  val context: DistributedContext,
  override val cacheHint: CacheHint
) extends CheckpointedDrm[K] {

  override val keyClassTag: ClassTag[K] = classTag[K]

  /**
    * Collecting DRM to in-core Matrix
    *
    * If key in DRM is Int, then matrix is collected using key as row index.
    * Otherwise, order of rows in result is undefined but key.toString is applied
    * as rowLabelBindings of the in-core matrix.
    */
  def collect: Matrix = H2OHelper.matrixFromDrm(h2odrm)

  /* XXX: call frame.remove */
  def uncache(): this.type = this

  /**
    * Persist DRM to on-disk over HDFS in Mahout DRM format.
    */
  def dfsWrite(path: String): Unit = H2OHdfs.drmToFile(path, h2odrm)

  /**
    * Action operator - Eagerly evaluate the lazily built operator graph to create
    *                   a CheckpointedDrm
    */
  def checkpoint(cacheHint: CacheHint.CacheHint): CheckpointedDrm[K] = this

  def ncol: Int = h2odrm.frame.numCols

  def nrow: Long = h2odrm.frame.numRows

  def canHaveMissingRows: Boolean = false

  protected[mahout] def partitioningTag: Long = h2odrm.frame.anyVec.group.hashCode

  /** stub need to make IndexedDataset core but since drmWrap is not in H2O left for someone else */
  override def newRowCardinality(n: Int): CheckpointedDrm[K] = {
    throw new UnsupportedOperationException("CheckpointedDrmH2O#newRowCardinality is not implemented.")
    /* this is the Spark impl
    assert(n > -1)
    assert( n >= nrow)
    val newCheckpointedDrm = drmWrap[K](rdd, n, ncol)
    newCheckpointedDrm
    */
    this
  }

}
