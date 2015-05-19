package org.apache.mahout.flinkbindings.blas

import scala.reflect.ClassTag
import org.apache.mahout.math.drm.BlockMapFunc
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.flink.api.common.functions.MapFunction
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm

import org.apache.mahout.math.Matrix
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._

object FlinkOpMapBlock {

  def apply[S, R: ClassTag](src: FlinkDrm[S], ncol: Int, function: BlockMapFunc[S, R]): FlinkDrm[R] = {
    val res = src.blockify.ds.map(new MapFunction[(Array[S], Matrix), (Array[R], Matrix)] {
      def map(block: (Array[S], Matrix)): (Array[R], Matrix) =  {
        val out = function(block)
        assert(out._2.nrow == block._2.nrow, "block mapping must return same number of rows.")
        assert(out._2.ncol == ncol, s"block map must return $ncol number of columns.")
        out
      }
    })

    new BlockifiedFlinkDrm(res, ncol)
  }
}