package org.apache.mahout.flinkbindings.blas


import org.apache.flink.api.common.typeinfo.TypeInformation
import org.apache.flink.api.scala._
import org.apache.flink.util.Collector
import org.apache.mahout.flinkbindings.drm.{FlinkDrm, RowsFlinkDrm}
import org.apache.mahout.math.Vector
import org.apache.mahout.math.drm.logical.OpAewB
import org.apache.mahout.math.scalabindings.RLikeOps._

/**
 * Implementation of Flink OpAewB
 */
object FlinkOpAewB {

  def rowWiseJoinNoSideEffect[K: TypeInformation](op: OpAewB[K], A: FlinkDrm[K], B: FlinkDrm[K]): FlinkDrm[K] = {
    val function = AewBOpsCloning.strToFunction(op.op)

    val rowsA = A.asRowWise.ds
    val rowsB = B.asRowWise.ds
    implicit val kTag = op.keyClassTag

    val res: DataSet[(K, Vector)] =
      rowsA
        .coGroup(rowsB)
        .where(0)
        .equalTo(0) {
        (left, right, out: Collector[(K, Vector)]) =>
          (left.toIterable.headOption, right.toIterable.headOption) match {
            case (Some((idx, a)), Some((_, b))) => out.collect((idx, function(a, b)))
            case (None, Some(b)) => out.collect(b)
            case (Some(a), None) => out.collect(a)
            case (None, None) => throw new RuntimeException("At least one side of the co group " +
              "must be non-empty.")
          }
      }


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
