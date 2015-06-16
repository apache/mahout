package org.apache.mahout.flinkbindings.blas

import scala.reflect.ClassTag
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings._
import org.apache.mahout.math.drm.drmBroadcast
import org.apache.mahout.math.drm.logical.OpAx
import org.apache.mahout.math.Matrix
import org.apache.flink.api.common.functions.MapFunction
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import java.util.List

object FlinkOpAx {

  def blockifiedBroadcastAx[K: ClassTag](op: OpAx[K], A: FlinkDrm[K]): FlinkDrm[K] = {
    implicit val ctx = A.context
    //    val x = drmBroadcast(op.x)

    val singletonDataSetX = ctx.env.fromElements(op.x)

    val out = A.blockify.ds.map(new RichMapFunction[(Array[K], Matrix), (Array[K], Matrix)] {
      var x: Vector = null

      override def open(params: Configuration): Unit = {
        val runtime = this.getRuntimeContext()
        val dsX: List[Vector] = runtime.getBroadcastVariable("vector")
        x = dsX.get(0)
      }

      override def map(tuple: (Array[K], Matrix)): (Array[K], Matrix) = tuple match {
        case (keys, mat) => (keys, (mat %*% x).toColMatrix)
      }
    }).withBroadcastSet(singletonDataSetX, "vector")

    new BlockifiedFlinkDrm(out, op.nrow.toInt)
  }
}