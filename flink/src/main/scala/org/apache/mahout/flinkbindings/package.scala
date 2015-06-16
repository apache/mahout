package org.apache.mahout

import org.apache.flink.api.java.DataSet
import org.apache.flink.api.java.ExecutionEnvironment
import org.apache.mahout.flinkbindings.FlinkDistributedContext
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.mahout.math.drm._
import org.slf4j.LoggerFactory
import scala.reflect.ClassTag
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings.drm.CheckpointedFlinkDrm
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm

package object flinkbindings {

  private[flinkbindings] val log = LoggerFactory.getLogger("apache.org.mahout.flinkbingings")

  /** Row-wise organized DRM dataset type */
  type DrmDataSet[K] = DataSet[DrmTuple[K]]

  /**
   * Blockifed DRM dataset (keys of original DRM are grouped into array corresponding to rows of Matrix
   * object value
   */
  type BlockifiedDrmDataSet[K] = DataSet[BlockifiedDrmTuple[K]]

  
  implicit def wrapMahoutContext(context: DistributedContext): FlinkDistributedContext = {
    assert(context.isInstanceOf[FlinkDistributedContext], "it must be FlinkDistributedContext")
    context.asInstanceOf[FlinkDistributedContext]
  }

  implicit def wrapContext(env: ExecutionEnvironment): FlinkDistributedContext =
    new FlinkDistributedContext(env)
  implicit def unwrapContext(ctx: FlinkDistributedContext): ExecutionEnvironment = ctx.env

  private[flinkbindings] implicit def castCheckpointedDrm[K: ClassTag](drm: CheckpointedDrm[K]): CheckpointedFlinkDrm[K] = {
    assert(drm.isInstanceOf[CheckpointedFlinkDrm[K]], "it must be a Flink-backed matrix")
    drm.asInstanceOf[CheckpointedFlinkDrm[K]]
  }

  implicit def checkpointeDrmToFlinkDrm[K: ClassTag](cp: CheckpointedDrm[K]): FlinkDrm[K] = {
    val flinkDrm = castCheckpointedDrm(cp)
    new RowsFlinkDrm[K](flinkDrm.ds, flinkDrm.ncol)
  }

}