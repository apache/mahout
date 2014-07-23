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

import scala.reflect._
import org.apache.mahout.math._
import org.apache.mahout.math.drm._
import org.apache.mahout.math.drm.logical._
import org.apache.mahout.h2obindings.ops._
import org.apache.mahout.h2obindings.drm._

object H2OEngine extends DistributedEngine {
  def colMeans[K:ClassTag](drm: CheckpointedDrm[K]): Vector =
    H2OHelper.colMeans(drm.h2odrm.frame)

  def colSums[K:ClassTag](drm: CheckpointedDrm[K]): Vector =
    H2OHelper.colSums(drm.h2odrm.frame)

  def norm[K: ClassTag](drm: CheckpointedDrm[K]): Double =
    H2OHelper.sumSqr(drm.h2odrm.frame)

  def numNonZeroElementsPerColumn[K: ClassTag](drm: CheckpointedDrm[K]): Vector =
    H2OHelper.nonZeroCnt(drm.h2odrm.frame)

  def drmBroadcast(m: Matrix)(implicit dc: DistributedContext): BCast[Matrix] =
    new H2OBCast(m)

  def drmBroadcast(v: Vector)(implicit dc: DistributedContext): BCast[Vector] =
    new H2OBCast(v)

  def drmFromHDFS(path: String, parMin: Int = 0)(implicit dc: DistributedContext): CheckpointedDrm[_] =
    new CheckpointedDrmH2O(H2OHdfs.drm_from_file(path, parMin), dc)

  def drmParallelizeEmpty(nrow: Int, ncol: Int, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Int] =
    new CheckpointedDrmH2O[Int](H2OHelper.empty_drm(nrow, ncol, numPartitions, -1), dc)

  def drmParallelizeEmptyLong(nrow: Long, ncol: Int, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Long] =
    new CheckpointedDrmH2O[Long](H2OHelper.empty_drm(nrow, ncol, numPartitions, -1), dc)

  def drmParallelizeWithRowIndices(m: Matrix, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[Int] =
    new CheckpointedDrmH2O[Int](H2OHelper.drm_from_matrix(m, numPartitions, -1), dc)

  def drmParallelizeWithRowLabels(m: Matrix, numPartitions: Int)(implicit dc: DistributedContext): CheckpointedDrm[String] =
    new CheckpointedDrmH2O[String](H2OHelper.drm_from_matrix(m, numPartitions, -1), dc)

  def toPhysical[K:ClassTag](plan: DrmLike[K], ch: CacheHint.CacheHint): CheckpointedDrm[K] =
    new CheckpointedDrmH2O[K](tr2phys(plan), plan.context)

  // H2O specific

  private def tr2phys[K: ClassTag](oper: DrmLike[K]): H2ODrm = {
    oper match {
      case OpAtAnyKey(_) =>
        throw new IllegalArgumentException("\"A\" must be Int-keyed in this A.t expression.")
      case op@OpAt(a) => At.At(tr2phys(a)(op.classTagA))
      case op@OpABt(a, b) => ABt.ABt(tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpAtB(a, b) => AtB.AtB(tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpAtA(a) => AtA.AtA(tr2phys(a)(op.classTagA))
      case op@OpAx(a, v) => Ax.Ax(tr2phys(a)(op.classTagA), v)
      case op@OpAtx(a, v) => Atx.Atx(tr2phys(a)(op.classTagA), v)
      case op@OpAewB(a, b, opId) => AewB.AewB(tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB), opId)
      // Non arithmetic
      case op@OpCbind(a, b) => Cbind.Cbind(tr2phys(a)(op.classTagA), tr2phys(b)(op.classTagB))
      case op@OpAewScalar(a, s, opId) => AewScalar.AewScalar(tr2phys(a)(op.classTagA), s, opId)
      case op@OpRowRange(a, r) => RowRange.RowRange(tr2phys(a)(op.classTagA), r)
      case op@OpTimesRightMatrix(a, m) => TimesRightMatrix.TimesRightMatrix(tr2phys(a)(op.classTagA), m)
      // Custom operators, we just execute them
      case blockOp: OpMapBlock[K, _] => MapBlock.exec(tr2phys(blockOp.A)(blockOp.classTagA), blockOp.ncol, blockOp.bmf,
        (blockOp.classTagK == implicitly[ClassTag[String]]), blockOp.classTagA, blockOp.classTagK)
      case op@OpPar(a, m, e) => Par.exec(tr2phys(a)(op.classTagA), m, e)
      case cp: CheckpointedDrm[K] => cp.h2odrm
      case _ => throw new IllegalArgumentException("Internal:Optimizer has no exec policy for operator %s."
          .format(oper))
    }
  }

  implicit def cp2cph2o[K:ClassTag](drm: CheckpointedDrm[K]): CheckpointedDrmH2O[K] = drm.asInstanceOf[CheckpointedDrmH2O[K]]
}
