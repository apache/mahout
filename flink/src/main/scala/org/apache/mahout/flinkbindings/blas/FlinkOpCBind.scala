package org.apache.mahout.flinkbindings.blas

import java.lang.Iterable
import scala.reflect.ClassTag
import scala.collection.JavaConverters._
import org.apache.mahout.math.drm.logical.OpCbind
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.Vector
import org.apache.flink.api.java.DataSet
import org.apache.flink.api.common.functions.CoGroupFunction
import org.apache.flink.util.Collector
import com.google.common.collect.Lists
import org.apache.mahout.math.DenseVector
import org.apache.mahout.math.SequentialAccessSparseVector
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm

object FlinkOpCBind {

  def cbind[K: ClassTag](op: OpCbind[K], A: FlinkDrm[K], B: FlinkDrm[K]): FlinkDrm[K] = {
    val n = op.ncol
    val n1 = op.A.ncol
    val n2 = op.B.ncol

    // TODO: cast!
    val rowsA = A.deblockify.ds.asInstanceOf[DataSet[(Int, Vector)]]
    val rowsB = B.deblockify.ds.asInstanceOf[DataSet[(Int, Vector)]]

    val res: DataSet[(Int, Vector)] = 
      rowsA.coGroup(rowsB).where(tuple_1[Vector]).equalTo(tuple_1[Vector])
        .`with`(new CoGroupFunction[(Int, Vector), (Int, Vector), (Int, Vector)] {
      def coGroup(it1java: Iterable[(Int, Vector)], it2java: Iterable[(Int, Vector)], 
                  out: Collector[(Int, Vector)]): Unit = {
        val it1 = Lists.newArrayList(it1java).asScala
        val it2 = Lists.newArrayList(it2java).asScala

        if (!it1.isEmpty && !it2.isEmpty) {
          val (idx, a) = it1.head
          val (_, b) = it2.head

          val result: Vector = if (a.isDense && b.isDense) { 
            new DenseVector(n) 
          } else {
            new SequentialAccessSparseVector(n)
          }

          result(0 until n1) := a
          result(n1 until n) := b

          out.collect((idx, result))
        } else if (it1.isEmpty && !it2.isEmpty) {
          val (idx, b) = it2.head
          val result: Vector = if (b.isDense) { 
            new DenseVector(n)
          } else {
            new SequentialAccessSparseVector(n)
          }
          result(n1 until n) := b
          out.collect((idx, result))
        } else if (!it1.isEmpty && it2.isEmpty) {
          val (idx, a) = it1.head
          val result: Vector = if (a.isDense) {
            new DenseVector(n)
          } else {
            new SequentialAccessSparseVector(n)
          }
          result(n1 until n) := a
          out.collect((idx, result))
        }
      }
    })

    new RowsFlinkDrm(res.asInstanceOf[DataSet[(K, Vector)]], ncol=op.ncol)
  }

}