package org.apache.mahout.flinkbindings.blas

import scala.reflect.ClassTag
import org.apache.mahout.math.drm.logical.OpRbind
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm
import org.apache.flink.api.common.functions.MapFunction
import org.apache.flink.api.java.DataSet
import org.apache.mahout.math.Vector

object FlinkOpRBind {

  def rbind[K: ClassTag](op: OpRbind[K], A: FlinkDrm[K], B: FlinkDrm[K]): FlinkDrm[K] = {
    val res = A.deblockify.ds.union(B.deblockify.ds)
    new RowsFlinkDrm(res.asInstanceOf[DataSet[(K, Vector)]], ncol = op.ncol)
  }

}