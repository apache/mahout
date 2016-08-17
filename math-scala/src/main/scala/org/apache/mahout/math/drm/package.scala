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

import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings.RLikeOps._
import org.apache.mahout.math.scalabindings._

import scala.reflect.ClassTag
import org.apache.mahout.math.drm.logical.OpAewUnaryFunc

import collection._

package object drm {

  /** Drm row-wise tuple */
  type DrmTuple[K] = (K, Vector)

  /** Drm block-wise tuple: Array of row keys and the matrix block. */
  type BlockifiedDrmTuple[K] = (Array[K], _ <: Matrix)


  /** Block-map func */
  type BlockMapFunc[S, R] = BlockifiedDrmTuple[S] ⇒ BlockifiedDrmTuple[R]

  type BlockMapFunc2[S] = BlockifiedDrmTuple[S] ⇒ Matrix

  type BlockReduceFunc = (Matrix, Matrix) ⇒ Matrix

  /** CacheHint type */
  //  type CacheHint = CacheHint.CacheHint

  def safeToNonNegInt(x: Long): Int = {
    assert(x == x << -31 >>> -31, "transformation from long to Int is losing significant bits, or is a negative number")
    x.toInt
  }

  /** Broadcast support API */
  def drmBroadcast(m:Matrix)(implicit ctx:DistributedContext):BCast[Matrix] = ctx.drmBroadcast(m)

  /** Broadcast support API */
  def drmBroadcast(v:Vector)(implicit ctx:DistributedContext):BCast[Vector] = ctx.drmBroadcast(v)

  /** Load DRM from hdfs (as in Mahout DRM format) */
  def drmDfsRead (path: String)(implicit ctx: DistributedContext): CheckpointedDrm[_] = ctx.drmDfsRead(path)

  /** Shortcut to parallelizing matrices with indices, ignore row labels. */
  def drmParallelize(m: Matrix, numPartitions: Int = 1)
      (implicit sc: DistributedContext): CheckpointedDrm[Int] = drmParallelizeWithRowIndices(m, numPartitions)(sc)

  /** Parallelize in-core matrix as a distributed matrix, using row ordinal indices as data set keys. */
  def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int = 1)
      (implicit ctx: DistributedContext): CheckpointedDrm[Int] = ctx.drmParallelizeWithRowIndices(m, numPartitions)

  /** Parallelize in-core matrix as a distributed matrix, using row labels as a data set keys. */
  def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int = 1)
      (implicit ctx: DistributedContext): CheckpointedDrm[String] = ctx.drmParallelizeWithRowLabels(m, numPartitions)

  /** This creates an empty DRM with specified number of partitions and cardinality. */
  def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int = 10)
      (implicit ctx: DistributedContext): CheckpointedDrm[Int] = ctx.drmParallelizeEmpty(nrow, ncol, numPartitions)

  /** Creates empty DRM with non-trivial height */
  def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int = 10)
      (implicit ctx: DistributedContext): CheckpointedDrm[Long] = ctx.drmParallelizeEmptyLong(nrow, ncol, numPartitions)

  /** Implicit broadcast -> value conversion. */
  implicit def bcast2val[T](bcast: BCast[T]): T = bcast.value

  /** Just throw all engine operations into context as well. */
  implicit def ctx2engine(ctx: DistributedContext): DistributedEngine = ctx.engine

  implicit def drm2drmCpOps[K](drm: CheckpointedDrm[K]): CheckpointedOps[K] =
    new CheckpointedOps[K](drm)

  /**
   * We assume that whenever computational action is invoked without explicit checkpoint, the user
   * doesn't imply caching
   */
  implicit def drm2Checkpointed[K](drm: DrmLike[K]): CheckpointedDrm[K] = drm.checkpoint(CacheHint.NONE)

  /** Implicit conversion to in-core with NONE caching of the result. */
  implicit def drm2InCore[K](drm: DrmLike[K]): Matrix = drm.collect

  /** Do vertical concatenation of collection of blockified tuples */
  private[mahout] def rbind[K:ClassTag](blocks: Iterable[BlockifiedDrmTuple[K]]): BlockifiedDrmTuple[K] = {
    assert(blocks.nonEmpty, "rbind: 0 blocks passed in")
    if (blocks.size == 1) {
      // No coalescing required.
      blocks.head
    } else {
      // compute total number of rows in a new block
      val m = blocks.view.map(_._2.nrow).sum
      val n = blocks.head._2.ncol
      val coalescedBlock = blocks.head._2.like(m, n)
      val coalescedKeys = new Array[K](m)
      var row = 0
      for (elem <- blocks.view) {
        val block = elem._2
        val rowEnd = row + block.nrow
        coalescedBlock(row until rowEnd, ::) := block
        elem._1.copyToArray(coalescedKeys, row)
        row = rowEnd
      }
      coalescedKeys -> coalescedBlock
    }
  }

  /**
   * Convert arbitrarily-keyed matrix to int-keyed matrix. Some algebra will accept only int-numbered
   * row matrices. So this method is to help.
   *
   * @param drmX input to be transcoded
   * @param computeMap collect `old key -> int key` map to front-end?
   * @tparam K key type
   * @return Sequentially keyed matrix + (optionally) map from non-int key to [[Int]] key. If the
   *         key type is actually Int, then we just return the argument with None for the map,
   *         regardless of computeMap parameter.
   */
  def drm2IntKeyed[K](drmX: DrmLike[K], computeMap: Boolean = false): (DrmLike[Int], Option[DrmLike[K]]) =
    drmX.context.engine.drm2IntKeyed(drmX, computeMap)

  /**
   * (Optional) Sampling operation. Consistent with Spark semantics of the same.
   * @param drmX
   * @param fraction
   * @param replacement
   * @tparam K
   * @return samples
   */
  def drmSampleRows[K](drmX: DrmLike[K], fraction: Double, replacement: Boolean = false): DrmLike[K] =
    drmX.context.engine.drmSampleRows(drmX, fraction, replacement)

  def drmSampleKRows[K](drmX: DrmLike[K], numSamples: Int, replacement: Boolean = false): Matrix =
    drmX.context.engine.drmSampleKRows(drmX, numSamples, replacement)

  /**
    * Convert a DRM sample into a Tab Separated Vector (TSV) to be loaded into an R-DataFrame
    * for plotting and sketching
    * @param drmX - DRM
    * @param samplePercent - Percentage of Sample elements from the DRM to be fished out for plotting
    * @tparam K
    * @return TSV String
    */
  def drmSampleToTSV[K](drmX: DrmLike[K], samplePercent: Double = 1): String = {

    val drmSize = drmX.checkpoint().numRows()
    val sampleRatio: Double = 1.0 * samplePercent / 100
    val numSamples: Int = (drmSize * sampleRatio).toInt

    val plotMatrix = drmSampleKRows(drmX, numSamples, replacement = false)

    // Plot Matrix rows
    val matrixRows = plotMatrix.numRows()
    val matrixCols = plotMatrix.numCols()

    // Convert the Plot Matrix Rows to TSV
    var str = ""

    for (i <- 0 until matrixRows) {
      for (j <- 0 until matrixCols) {
        str += plotMatrix(i, j)
        if (j <= matrixCols - 2) {
          str += '\t'
        }
      }
      str += '\n'
    }

    str
  }

  ///////////////////////////////////////////////////////////
  // Elementwise unary functions on distributed operands.
  def dexp[K](drmA: DrmLike[K]): DrmLike[K] = new OpAewUnaryFunc[K](drmA, math.exp, true)

  def dlog[K](drmA: DrmLike[K]): DrmLike[K] = new OpAewUnaryFunc[K](drmA, math.log, true)

  def dabs[K](drmA: DrmLike[K]): DrmLike[K] = new OpAewUnaryFunc[K](drmA, math.abs)

  def dsqrt[K](drmA: DrmLike[K]): DrmLike[K] = new OpAewUnaryFunc[K](drmA, math.sqrt)

  def dsignum[K](drmA: DrmLike[K]): DrmLike[K] = new OpAewUnaryFunc[K](drmA, math.signum)
  
  ///////////////////////////////////////////////////////////
  // Misc. math utilities.

  /**
   * Compute column wise means and variances -- distributed version.
   *
   * @param drmA Note: will pin input to cache if not yet pinned.
   * @tparam K
   * @return colMeans → colVariances
   */
  def dcolMeanVars[K](drmA: DrmLike[K]): (Vector, Vector) = {

    import RLikeDrmOps._

    val drmAcp = drmA.checkpoint()

    val mu = drmAcp colMeans

    // Compute variance using mean(x^2) - mean(x)^2
    val variances = (drmAcp ^ 2 colMeans) -=: mu * mu

    mu → variances
  }

  /**
   * Compute column wise means and standard deviations -- distributed version.
   * @param drmA note: input will be pinned to cache if not yet pinned
   * @return colMeans → colStdevs
   */
  def dcolMeanStdevs[K](drmA: DrmLike[K]): (Vector, Vector) = {
    val (mu, vars) = dcolMeanVars(drmA)
    mu → (vars ::= math.sqrt _)
  }

  /**
   * Thin column-wise mean and covariance matrix computation. Same as [[dcolMeanCov()]] but suited for
   * thin and tall inputs where covariance matrix can be reduced and finalized in driver memory.
   * 
   * @param drmA note: will pin input to cache if not yet pinned.
   * @return mean → covariance matrix (in core)
   */
  def dcolMeanCovThin[K: ClassTag](drmA: DrmLike[K]):(Vector, Matrix) = {

    import RLikeDrmOps._

    val drmAcp = drmA.checkpoint()
    val mu = drmAcp colMeans
    val mxCov = (drmAcp.t %*% drmAcp).collect /= drmAcp.nrow -= (mu cross mu)
    mu → mxCov
  }

  /**
   * Compute COV(X) matrix and mean of row-wise data set. X is presented as row-wise input matrix A.
   *
   * This is a "wide" procedure, covariance matrix is returned as a DRM.
   *
   * @param drmA note: will pin input into cache if not yet pinned.
   * @return mean → covariance DRM
   */
  def dcolMeanCov[K: ClassTag](drmA: DrmLike[K]): (Vector, DrmLike[Int]) = {

    import RLikeDrmOps._

    implicit val ctx = drmA.context
    val drmAcp = drmA.checkpoint()

    val bcastMu = drmBroadcast(drmAcp colMeans)

    // We use multivaraite analogue COV(X)=E(XX')-mu*mu'. In our case E(XX') = (A'A)/A.nrow.
    // Compute E(XX')
    val drmSigma = (drmAcp.t %*% drmAcp / drmAcp.nrow)

      // Subtract mu*mu'. In this case we assume mu*mu' may still be big enough to be treated by
      // driver alone, so we redistribute this operation as well. Hence it may look a bit cryptic.
      .mapBlock() { case (keys, block) ⇒

      // Pin mu as vector reference to memory.
      val mu:Vector = bcastMu

      keys → (block := { (r, c, v) ⇒ v - mu(keys(r)) * mu(c) })
    }

    // return (mu, cov(X) ("bigSigma")).
    (bcastMu: Vector) → drmSigma
  }

  /** Distributed Squared distance matrix computation. */
  def dsqDist(drmX: DrmLike[Int]): DrmLike[Int] = {

    // This is a specific case of pairwise distances of X and Y.

    import RLikeDrmOps._

    // Context needed
    implicit val ctx = drmX.context

    // Pin to cache if hasn't been pinned yet
    val drmXcp = drmX.checkpoint()

    // Compute column sum of squares
    val s = drmXcp ^ 2 rowSums

    val sBcast = drmBroadcast(s)

    (drmXcp %*% drmXcp.t)

      // Apply second part of the formula as per in-core algorithm
      .mapBlock() { case (keys, block) ⇒

      // Slurp broadcast to memory
      val s = sBcast: Vector

      // Update in-place
      block := { (r, c, x) ⇒ s(keys(r)) + s(c) - 2 * x}

      keys → block
    }
  }


  /**
   * Compute fold-in distances (distributed version). Here, we use pretty much the same math as with
   * squared distances.
   *
   * D_sq = s*1' + 1*t' - 2*X*Y'
   *
   * where s is row sums of hadamard product(X, X), and, similarly,
   * s is row sums of Hadamard product(Y, Y).
   *
   * @param drmX m x d row-wise dataset. Pinned to cache if not yet pinned.
   * @param drmY n x d row-wise dataset. Pinned to cache if not yet pinned.
   * @return m x d pairwise squared distance matrix (between rows of X and Y)
   */
  def dsqDist(drmX: DrmLike[Int], drmY: DrmLike[Int]): DrmLike[Int] = {

    import RLikeDrmOps._

    implicit val ctx = drmX.context

    val drmXcp = drmX.checkpoint()
    val drmYcp = drmY.checkpoint()

    val sBcast = drmBroadcast(drmXcp ^ 2 rowSums)
    val tBcast = drmBroadcast(drmYcp ^ 2 rowSums)

    (drmX %*% drmY.t)

      // Apply the rest of the formula
      .mapBlock() { case (keys, block) =>

      // Cache broadcast representations in local task variable
      val s = sBcast: Vector
      val t = tBcast: Vector

      block := { (r, c, x) => s(keys(r)) + t(c) - 2 * x}
      keys → block
    }
  }
}

package object indexeddataset {
  /** Load IndexedDataset from text delimited files */
  def indexedDatasetDFSRead(src: String,
      schema: Schema = DefaultIndexedDatasetReadSchema,
      existingRowIDs: Option[BiDictionary] = None)
    (implicit ctx: DistributedContext):
    IndexedDataset = ctx.indexedDatasetDFSRead(src, schema, existingRowIDs)

  def indexedDatasetDFSReadElements(src: String,
      schema: Schema = DefaultIndexedDatasetReadSchema,
      existingRowIDs: Option[BiDictionary] = None)
    (implicit ctx: DistributedContext):
    IndexedDataset = ctx.indexedDatasetDFSReadElements(src, schema, existingRowIDs)

}

