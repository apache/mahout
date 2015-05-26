package org.apache.mahout.flinkbindings.blas

import scala.reflect.ClassTag
import org.apache.mahout.math.drm.logical.OpTimesRightMatrix
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.DiagonalMatrix
import org.apache.flink.api.common.functions.RichMapFunction
import org.apache.flink.configuration.Configuration
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._

object FlinkOpTimesRightMatrix {

  def drmTimesInCore[K: ClassTag](op: OpTimesRightMatrix[K], A: FlinkDrm[K], inCoreB: Matrix): FlinkDrm[K] = {
    implicit val ctx = A.context

    val singletonDataSetB = ctx.env.fromElements(inCoreB)

    val res = A.blockify.ds.map(new RichMapFunction[(Array[K], Matrix), (Array[K], Matrix)] {
      var inCoreB: Matrix = null

      override def open(params: Configuration): Unit = {
        val runtime = this.getRuntimeContext()
        val dsB: java.util.List[Matrix] = runtime.getBroadcastVariable("matrix")
        inCoreB = dsB.get(0)
      }

      override def map(tuple: (Array[K], Matrix)): (Array[K], Matrix) = tuple match {
        case (keys, block_A) => (keys, block_A %*% inCoreB)
      }

    }).withBroadcastSet(singletonDataSetB, "matrix")

    new BlockifiedFlinkDrm(res, op.ncol)
  }

}