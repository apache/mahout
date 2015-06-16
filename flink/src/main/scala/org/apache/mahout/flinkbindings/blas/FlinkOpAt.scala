package org.apache.mahout.flinkbindings.blas

import org.apache.mahout.math.drm.logical.OpAt
import org.apache.mahout.flinkbindings.DrmDataSet
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.flink.api.common.functions.FlatMapFunction
import org.apache.mahout.math.Matrix
import scala.reflect.ClassTag
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings._
import org.apache.mahout.math._
import scalabindings._
import RLikeOps._
import org.apache.flink.api.common.functions.GroupReduceFunction
import org.apache.mahout.math.drm.DrmTuple
import java.lang.Iterable
import scala.collection.JavaConverters._
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.flink.api.java.functions.KeySelector
import java.util.ArrayList
import org.apache.flink.shaded.com.google.common.collect.Lists

/**
 * Taken from
 */
object FlinkOpAt {

  /**
   * The idea here is simple: compile vertical column vectors of every partition block as sparse
   * vectors of the <code>A.nrow</code> length; then group them by their column index and sum the
   * groups into final rows of the transposed matrix.
   */
  def sparseTrick(op: OpAt, A: FlinkDrm[Int]): FlinkDrm[Int] = {
    val ncol = op.ncol // # of rows of A, i.e. # of columns of A^T

    val sparseParts = A.blockify.ds.flatMap(new FlatMapFunction[(Array[Int], Matrix), DrmTuple[Int]] {
      def flatMap(typle: (Array[Int], Matrix), out: Collector[DrmTuple[Int]]): Unit = typle match {
        case (keys, block) => {
          (0 until block.ncol).map(columnIdx => {
            val columnVector: Vector = new SequentialAccessSparseVector(ncol)

            keys.zipWithIndex.foreach { case (key, idx) =>
                columnVector(key) = block(idx, columnIdx)
            }

            out.collect(new Tuple2(columnIdx, columnVector))
          })
        }
      }
    })

    val regrouped = sparseParts.groupBy(new KeySelector[Tuple2[Int, Vector], Integer] {
      def getKey(tuple: Tuple2[Int, Vector]): Integer = tuple._1
    })

    val sparseTotal = regrouped.reduceGroup(new GroupReduceFunction[Tuple2[Int, Vector], DrmTuple[Int]] {
      def reduce(values: Iterable[DrmTuple[Int]], out: Collector[DrmTuple[Int]]): Unit = {
        val it = Lists.newArrayList(values).asScala
        val (idx, _) = it.head
        val vector = it map { case (idx, vec) => vec } reduce (_ + _)
        out.collect(idx -> vector)
      }
    })

    // TODO: densify or not?
    new RowsFlinkDrm(sparseTotal, ncol)
  }

}