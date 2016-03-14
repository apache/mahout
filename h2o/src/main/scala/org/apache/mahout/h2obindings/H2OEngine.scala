/*
 *  Licensed to the Apache Software Foundation (ASF) under one or more
 *  contributor license agreements.  See the NOTICE file distributed with
 *  this work for additional information regarding copyright ownership.
 *  The ASF licenses this file to You under the Apache License, Version 2.0
 *  (the "License"); you may not use this file except in compliance with
 *  the License.  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package org.apache.mahout.h2obindings

import org.apache.mahout.math.indexeddataset.{BiDictionary, IndexedDataset, Schema, DefaultIndexedDatasetReadSchema}

import scala.reflect._
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical._
import org.apache.mahout.h2obindings.ops._
import org.apache.mahout.h2obindings.drm._
import org.apache.mahout.h2o.common.{Hadoop1HDFSUtil, HDFSUtil}
import org.apache.mahout.logging._

/** H2O specific non-DRM operations */
object H2OEngine extends DistributedEngine {

  private final implicit val log = getLog(H2OEngine.getClass)

  // By default, use Hadoop 1 utils
  var hdfsUtils: HDFSUtil = Hadoop1HDFSUtil

  def colMeans[K](drm: CheckpointedDrm[K]): Vector =
    H2OHelper.colMeans(drm.h2odrm.frame)

  def colSums[K](drm: CheckpointedDrm[K]): Vector =
    H2OHelper.colSums(drm.h2odrm.frame)

  def norm[K](drm: CheckpointedDrm[K]): Double =
    H2OHelper.sumSqr(drm.h2odrm.frame)

  def numNonZeroElementsPerColumn[K](drm: CheckpointedDrm[K]): Vector =
    H2OHelper.nonZeroCnt(drm.h2odrm.frame)

  /** Broadcast support */
  def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix] =
    new H2OBCast(m)

  /** Broadcast support */
  def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector] =
    new H2OBCast(v)

  /**
    * Load DRM from hdfs (as in Mahout DRM format)
    *
    *  @param path Path to DRM file
    *  @param parMin Hint of minimum number of partitions to split while distributing
    *
    *  @return DRM[Any] where Any is automatically translated to value type
    */
  def drmDfsRead(path: String, parMin: Int = 0)(implicit dc: DistributedContext): CheckpointedDrm[_] = {
    val drmMetadata = hdfsUtils.readDrmHeader(path)

    new CheckpointedDrmH2O(H2OHdfs.drmFromFile(path, parMin), dc, CacheHint.NONE)(drmMetadata.keyClassTag.
      asInstanceOf[ClassTag[Any]])
  }

  /** This creates an empty DRM with specified number of partitions and cardinality. */
  def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Int] =
    new CheckpointedDrmH2O[Int](H2OHelper.emptyDrm(nrow, ncol, numPartitions, -1), dc, CacheHint.NONE)

  def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Long] =
    new CheckpointedDrmH2O[Long](H2OHelper.emptyDrm(nrow, ncol, numPartitions, -1), dc, CacheHint.NONE)

  /** Parallelize in-core matrix as H2O distributed matrix, using row ordinal indices as data set keys. */
  def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Int] =
    new CheckpointedDrmH2O[Int](H2OHelper.drmFromMatrix(m, numPartitions, -1), dc, CacheHint.NONE)

  /** Parallelize in-core matrix as H2O distributed matrix, using row labels as a data set keys. */
  def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[String] =
    new CheckpointedDrmH2O[String](H2OHelper.drmFromMatrix(m, numPartitions, -1), dc, CacheHint.NONE)

  def toPhysical[K:ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] =
    new CheckpointedDrmH2O[K](tr2phys(plan), plan.context, ch)

  /** Eagerly evaluate operator graph into an H2O DRM */
  private def tr2phys[K: ClassTag](oper: DrmLike[K]): H2ODrm = {
    oper match {
      case OpAtAnyKey(_) =>
        throw new IllegalArgumentException("\"A\" must be Int-keyed in this A.t expression.")
      // Linear algebra operators
      case op@OpAt(a) => At.exec(tr2phys(a)(a.keyClassTag))
      case op@OpABt(a, b) => ABt.exec(tr2phys(a)(a.keyClassTag), tr2phys(b)(b.keyClassTag))
      case op@OpAtB(a, b) => AtB.exec(tr2phys(a)(a.keyClassTag), tr2phys(b)(b.keyClassTag))
      case op@OpAtA(a) => AtA.exec(tr2phys(a)(a.keyClassTag))
      case op@OpAx(a, v) => Ax.exec(tr2phys(a)(a.keyClassTag), v)
      case op@OpAtx(a, v) => Atx.exec(tr2phys(a)(a.keyClassTag), v)
      case op@OpAewUnaryFunc(a, f, z) => AewUnary.exec(tr2phys(a)(a.keyClassTag), op.f, z)
      case op@OpAewUnaryFuncFusion(a, f) => AewUnary.exec(tr2phys(a)(a.keyClassTag), op.f, op.evalZeros)
      case op@OpAewB(a, b, opId) => AewB.exec(tr2phys(a)(a.keyClassTag), tr2phys(b)(b.keyClassTag), opId)
      case op@OpAewScalar(a, s, opId) => AewScalar.exec(tr2phys(a)(a.keyClassTag), s, opId)
      case op@OpTimesRightMatrix(a, m) => TimesRightMatrix.exec(tr2phys(a)(a.keyClassTag), m)
      // Non arithmetic
      case op@OpCbind(a, b) => Cbind.exec(tr2phys(a)(a.keyClassTag), tr2phys(b)(b.keyClassTag))
      case op@OpCbindScalar(a, d, left) => CbindScalar.exec(tr2phys(a)(a.keyClassTag), d, left)
      case op@OpRbind(a, b) => Rbind.exec(tr2phys(a)(a.keyClassTag), tr2phys(b)(b.keyClassTag))
      case op@OpRowRange(a, r) => RowRange.exec(tr2phys(a)(a.keyClassTag), r)
      // Custom operators
      case blockOp: OpMapBlock[K, _] => MapBlock.exec(tr2phys(blockOp.A)(blockOp.A.keyClassTag), blockOp.ncol, blockOp.bmf,
        (blockOp.keyClassTag == classTag[String]), blockOp.A.keyClassTag, blockOp.keyClassTag)
      case op@OpPar(a, m, e) => Par.exec(tr2phys(a)(a.keyClassTag), m, e)
      case cp: CheckpointedDrm[K] => cp.h2odrm
      case _ => throw new IllegalArgumentException("Internal:Optimizer has no exec policy for operator %s."
          .format(oper))
    }
  }

  implicit def cp2cph2o[K](drm: CheckpointedDrm[K]): CheckpointedDrmH2O[K] = drm.asInstanceOf[CheckpointedDrmH2O[K]]

  /** stub class not implemented in H2O */
  abstract class IndexedDatasetH2O(val matrix: CheckpointedDrm[Int], val rowIDs: BiDictionary, val columnIDs: BiDictionary)
    extends IndexedDataset {}

  /**
   * Reads an IndexedDatasetH2O from default text delimited files
   * @todo unimplemented
   * @param src a comma separated list of URIs to read from
   * @param schema how the text file is formatted
   * @return
   */
  def indexedDatasetDFSRead(src: String,
                            schema: Schema = DefaultIndexedDatasetReadSchema,
                            existingRowIDs: Option[BiDictionary] = None)
                           (implicit sc: DistributedContext):
  IndexedDatasetH2O = {

    error("Unimplemented indexedDatasetDFSReadElements.")

    ???
  }

  /**
   * Reads an IndexedDatasetH2O from default text delimited files
   * @todo unimplemented
   * @param src a comma separated list of URIs to read from
   * @param schema how the text file is formatted
   * @return
   */
  def indexedDatasetDFSReadElements(src: String,
                                    schema: Schema = DefaultIndexedDatasetReadSchema,
                                    existingRowIDs: Option[BiDictionary] = None)
                                   (implicit sc: DistributedContext): IndexedDatasetH2O = {

    error("Unimplemented indexedDatasetDFSReadElements.")

    ???
  }

  /**
   * Optional engine-specific all reduce tensor operation.
   *
   * TODO: implement this please.
   *
   */
  override def allreduceBlock[K](drm: CheckpointedDrm[K], bmf: BlockMapFunc2[K], rf: BlockReduceFunc)
  : Matrix = H2OHelper.allreduceBlock(drm.h2odrm, bmf, rf)

  /**
   * TODO: implement this please.
   */
  override def drmSampleKRows[K](drmX: DrmLike[K], numSamples: Int, replacement: Boolean): Matrix = ???

  /**
   * (Optional) Sampling operation. Consistent with Spark semantics of the same.
   * TODO: implement this please.
   */
  override def drmSampleRows[K](drmX: DrmLike[K], fraction: Double, replacement: Boolean): DrmLike[K] = ???

  /**
   * TODO: implement this please.
   */
  override def drm2IntKeyed[K](drmX: DrmLike[K], computeMap: Boolean)
  : (DrmLike[Int], Option[DrmLike[K]]) = ???
}
