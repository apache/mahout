package org.apache.mahout.flinkbindings.blas

import java.lang.Iterable

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

import org.apache.flink.api.common.functions.CoGroupFunction
import org.apache.flink.api.java.DataSet
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings._
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.flinkbindings.drm.RowsFlinkDrm
import org.apache.mahout.math.Vector
import org.apache.mahout.math.drm.logical.OpAewB
import org.apache.mahout.math.scalabindings.RLikeOps._

import com.google.common.collect.Lists

/**
 * Implementation is inspired by Spark-binding's OpAewB
 * (see https://github.com/apache/mahout/blob/master/spark/src/main/scala/org/apache/mahout/sparkbindings/blas/AewB.scala) 
 */
object FlinkOpAewB {

  def rowWiseJoinNoSideEffect[K: ClassTag](op: OpAewB[K], A: FlinkDrm[K], B: FlinkDrm[K]): FlinkDrm[K] = {
    val function = AewBOpsCloning.strToFunction(op.op)

    val classTag = extractRealClassTag(op.A)
    val joiner = selector[Vector, Any](classTag.asInstanceOf[ClassTag[Any]]) 

    val rowsA = A.deblockify.ds.asInstanceOf[DrmDataSet[Any]]
    val rowsB = B.deblockify.ds.asInstanceOf[DrmDataSet[Any]]

    val res: DataSet[(Any, Vector)] = 
      rowsA.coGroup(rowsB).where(joiner).equalTo(joiner)
        .`with`(new CoGroupFunction[(_, Vector), (_, Vector), (_, Vector)] {
      def coGroup(it1java: Iterable[(_, Vector)], it2java: Iterable[(_, Vector)], 
                  out: Collector[(_, Vector)]): Unit = {
        val it1 = Lists.newArrayList(it1java).asScala
        val it2 = Lists.newArrayList(it2java).asScala

        if (!it1.isEmpty && !it2.isEmpty) {
          val (idx, a) = it1.head
          val (_, b) = it2.head
          out.collect((idx, function(a, b)))
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
