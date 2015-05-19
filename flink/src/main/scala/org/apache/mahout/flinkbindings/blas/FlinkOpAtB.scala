package org.apache.mahout.flinkbindings.blas

import scala.reflect.ClassTag
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.drm.logical.OpAtB
import org.apache.flink.api.common.functions.MapFunction
import org.apache.flink.api.java.tuple.Tuple2
import org.apache.mahout.math.Vector
import org.apache.mahout.math.Matrix
import org.apache.flink.api.common.functions.FlatMapFunction
import org.apache.flink.util.Collector
import org.apache.mahout.math.drm._
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.flink.api.common.functions.GroupReduceFunction
import java.lang.Iterable
import scala.collection.JavaConverters._
import com.google.common.collect.Lists
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm
import org.apache.mahout.flinkbindings.BlockifiedDrmDataSet
import org.apache.flink.api.scala._
import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.mahout.flinkbindings.DrmDataSet


object FlinkOpAtB {

  def notZippable[K: ClassTag](op: OpAtB[K], At: FlinkDrm[K], B: FlinkDrm[K]): FlinkDrm[Int] = {
    // TODO: to help Flink's type inference
    // only Int is supported now 
    val rowsAt = At.deblockify.ds.asInstanceOf[DrmDataSet[Int]]
    val rowsB = B.deblockify.ds.asInstanceOf[DrmDataSet[Int]]
    val joined = rowsAt.join(rowsB).where(tuple_1[Vector]).equalTo(tuple_1[Vector])

    val ncol = op.ncol
    val nrow = op.nrow
    val blockHeight = 10
    val blockCount = safeToNonNegInt((ncol - 1) / blockHeight + 1)

    val preProduct = joined.flatMap(new FlatMapFunction[Tuple2[(Int, Vector), (Int, Vector)], 
                                                        (Int, Matrix)] {
      def flatMap(in: Tuple2[(Int, Vector), (Int, Vector)],
                  out: Collector[(Int, Matrix)]): Unit = {
        val avec = in.f0._2
        val bvec = in.f1._2

        0.until(blockCount) map { blockKey =>
          val blockStart = blockKey * blockHeight
          val blockEnd = Math.min(ncol, blockStart + blockHeight)

          // Create block by cross product of proper slice of aRow and qRow
          val outer = avec(blockStart until blockEnd) cross bvec
          out.collect((blockKey, outer))
        }
      }
    })

    val res: BlockifiedDrmDataSet[Int] = preProduct.groupBy(tuple_1[Matrix]).reduceGroup(
            new GroupReduceFunction[(Int, Matrix), BlockifiedDrmTuple[Int]] {
      def reduce(values: Iterable[(Int, Matrix)], out: Collector[BlockifiedDrmTuple[Int]]): Unit = {
        val it = Lists.newArrayList(values).asScala
        val (idx, _) = it.head

        val block = it.map(t => t._2).reduce((m1, m2) => m1 + m2)

        val keys = idx.until(block.nrow).toArray[Int]
        out.collect((keys, block))
      }
    })

    new BlockifiedFlinkDrm(res, ncol)
  }

}

class DrmTupleToFlinkTupleMapper[K: ClassTag] extends MapFunction[(K, Vector), Tuple2[Int, Vector]] {
  def map(tuple: (K, Vector)): Tuple2[Int, Vector] = tuple match {
    case (key, vec) => new Tuple2[Int, Vector](key.asInstanceOf[Int], vec)
  }
}