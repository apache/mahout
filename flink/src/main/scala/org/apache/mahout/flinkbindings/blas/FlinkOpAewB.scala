package org.apache.mahout.flinkbindings.blas

import org.apache.mahout.math.drm.logical.OpAewB
import scala.reflect.ClassTag
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.Vector
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.flink.api.java.DataSet
import org.apache.flink.api.common.functions.CoGroupFunction
import java.lang.Iterable
import org.apache.flink.util.Collector
import com.google.common.collect.Lists
import scala.collection.JavaConverters._
import scala.collection.immutable.Nil
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm

object FlinkOpAewB {

  def rowWiseJoinNoSideEffect[K: ClassTag](op: OpAewB[K], A: FlinkDrm[K], B: FlinkDrm[K]): FlinkDrm[K] = {
    val function = AewBOpsCloning.strToFunction(op.op)

    // TODO: get rid of casts!
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
          out.collect(idx -> function(a, b))
        } else if (it1.isEmpty && !it2.isEmpty) {
          out.collect(it2.head)
        } else if (!it1.isEmpty && it2.isEmpty) {
          out.collect(it1.head)
        }
      }
    })

    new RowsFlinkDrm(res.asInstanceOf[DataSet[(K, Vector)]], ncol=op.ncol)
  }
}


object AewBOpsCloning {
  type VectorVectorFunc = (Vector, Vector) => Vector

  def strToFunction(op: String): VectorVectorFunc = op match {
    case "+" => plus
    case "-" => minus
    case "*" => times
    case "/" => div
    case _ => throw new IllegalArgumentException(s"Unsupported elementwise operator: $op")
  }

  val plus: VectorVectorFunc = (a, b) => a + b
  val minus: VectorVectorFunc = (a, b) => a - b
  val times: VectorVectorFunc = (a, b) => a * b
  val div: VectorVectorFunc = (a, b) => a / b
}
