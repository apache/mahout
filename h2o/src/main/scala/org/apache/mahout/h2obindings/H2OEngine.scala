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

/** H2O specific non-DRM operations */
object H2OEngine extends DistributedEngine {
  // By default, use Hadoop 1 utils
  var hdfsUtils: HDFSUtil = Hadoop1HDFSUtil

  def colMeans[K:ClassTag](drm: CheckpointedDrm[K]): Vector =
    H2OHelper.colMeans(drm.h2odrm.frame)

  def colSums[K:ClassTag](drm: CheckpointedDrm[K]): Vector =
    H2OHelper.colSums(drm.h2odrm.frame)

  def norm[K: ClassTag](drm: CheckpointedDrm[K]): Double =
    H2OHelper.sumSqr(drm.h2odrm.frame)

  def numNonZeroElementsPerColumn[K: ClassTag](drm: CheckpointedDrm[K]): Vector =
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

    new CheckpointedDrmH2O(H2OHdfs.drmFromFile(path, parMin), dc)(drmMetadata.keyClassTag.asInstanceOf[ClassTag[Any]])
  }

  /** This creates an empty DRM with specified number of partitions and cardinality. */
  def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Int] =
    new CheckpointedDrmH2O[Int](H2OHelper.emptyDrm(nrow, ncol, numPartitions, -1), dc)

  def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Long] =
    new CheckpointedDrmH2O[Long](H2OHelper.emptyDrm(nrow, ncol, numPartitions, -1), dc)

  /** Parallelize in-core matrix as H2O distributed matrix, using row ordinal indices as data set keys. */
  def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Int] =
    new CheckpointedDrmH2O[Int](H2OHelper.drmFromMatrix(m, numPartitions, -1), dc)

  /** Parallelize in-core matrix as H2O distributed matrix, using row labels as a data set keys. */
  def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[String] =
    new CheckpointedDrmH2O[String](H2OHelper.drmFromMatrix(m, numPartitions, -1), dc)

  def toPhysical[K:ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] =
    new CheckpointedDrmH2O[K](tr2phys(plan), plan.context)

  /** Eagerly evaluate operator graph into an H2O DRM */
  private def tr2phys[K: ClassTag](oper: DrmLike[K]): H2ODrm = {
    oper match {
      case OpAtAnyKey(_) =>
        throw new IllegalArgumentException("\"A\" must be Int-keyed in this A.t expression.")
      // Linear algebra operators
      case op@OpAt(a) => At.exec(tr2phys(a)(op.classTagA))
      case op@OpABt(a, b) => ABt.exec(tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpAtB(a, b) => AtB.exec(tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpAtA(a) => AtA.exec(tr2phys(a)(op.classTagA))
      case op@OpAx(a, v) => Ax.exec(tr2phys(a)(op.classTagA), v)
      case op@OpAtx(a, v) => Atx.exec(tr2phys(a)(op.classTagA), v)
      case op@OpAewB(a, b, opId) => AewB.exec(tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB), opId)
      case op@OpAewScalar(a, s, opId) => AewScalar.exec(tr2phys(a)(op.classTagA), s, opId)
      case op@OpTimesRightMatrix(a, m) => TimesRightMatrix.exec(tr2phys(a)(op.classTagA), m)
      // Non arithmetic
      case op@OpCbind(a, b) => Cbind.exec(tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpRbind(a, b) => Rbind.exec(tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpRowRange(a, r) => RowRange.exec(tr2phys(a)(op.classTagA), r)
      // Custom operators
      case blockOp: OpMapBlock[K, _] => MapBlock.exec(tr2phys(blockOp.A)(blockOp.classTagA), blockOp.ncol, blockOp.bmf,
        (blockOp.classTagK == implicitly[ClassTag[String]]), blockOp.classTagA, blockOp.classTagK)
      case op@OpPar(a, m, e) => Par.exec(tr2phys(a)(op.classTagA), m, e)
      case cp: CheckpointedDrm[K] => cp.h2odrm
      case _ => throw new IllegalArgumentException("Internal:Optimizer has no exec policy for operator %s."
          .format(oper))
    }
  }

  implicit def cp2cph2o[K:ClassTag](drm: CheckpointedDrm[K]): CheckpointedDrmH2O[K] = drm.asInstanceOf[CheckpointedDrmH2O[K]]

  /** stub class not implemented in H2O */
  abstract class IndexedDatasetH2O(val matrix: CheckpointedDrm[Int], val rowIDs: BiDictionary, val columnIDs: BiDictionary)
    extends IndexedDataset {}

    /**
   * reads an IndexedDatasetH2O from default text delimited files
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
    // should log a warning when this is built but no logger here, can an H2O contributor help with this
    println("Warning: unimplemented indexedDatasetDFSReadElements." )
    throw new UnsupportedOperationException("IndexedDatasetH2O is not implemented so can't be read.")
    null.asInstanceOf[IndexedDatasetH2O]
  }

  /**
   * reads an IndexedDatasetH2O from default text delimited files
   * @todo unimplemented
   * @param src a comma separated list of URIs to read from
   * @param schema how the text file is formatted
   * @return
   */
  def indexedDatasetDFSReadElements(src: String,
      schema: Schema = DefaultIndexedDatasetReadSchema,
      existingRowIDs: Option[BiDictionary] = None)
      (implicit sc: DistributedContext):
    IndexedDatasetH2O = {
    // should log a warning when this is built but no logger here, can an H2O contributor help with this
    println("Warning: unimplemented indexedDatasetDFSReadElements." )
    throw new UnsupportedOperationException("IndexedDatasetH2O is not implemented so can't be read by elements.")
    null.asInstanceOf[IndexedDatasetH2O]
  }

}
