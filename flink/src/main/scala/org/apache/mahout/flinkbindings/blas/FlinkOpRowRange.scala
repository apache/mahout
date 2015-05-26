package org.apache.mahout.flinkbindings.blas

import org.apache.mahout.math.drm.logical.OpRowRange
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.flink.api.common.functions.FilterFunction
import org.apache.mahout.math.Vector
import org.apache.flink.api.common.functions.MapFunction
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm

object FlinkOpRowRange {

  def slice(op: OpRowRange, A: FlinkDrm[Int]): FlinkDrm[Int] = {
    val rowRange = op.rowRange
    val firstIdx = rowRange.head

    val filtered = A.deblockify.ds.filter(new FilterFunction[(Int, Vector)] {
      def filter(tuple: (Int, Vector)): Boolean = tuple match {
        case (idx, vec) => rowRange.contains(idx)
      }
    })

    val res = filtered.map(new MapFunction[(Int, Vector), (Int, Vector)] {
      def map(tuple: (Int, Vector)): (Int, Vector) = tuple match {
        case (idx, vec) => (idx - firstIdx, vec)
      }
    })

    new RowsFlinkDrm(res, op.ncol)
  }

}