package org.apache.mahout.flinkbindings.blas

import scala.reflect.ClassTag
import org.apache.mahout.math.drm.logical.OpAewScalar
import org.apache.mahout.flinkbindings.drm.FlinkDrm
import org.apache.mahout.math.Matrix
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.flink.api.common.functions.MapFunction
import org.apache.mahout.flinkbindings.drm.BlockifiedFlinkDrm

object FlinkOpAewScalar {

  def opScalarNoSideEffect[K: ClassTag](op: OpAewScalar[K], A: FlinkDrm[K], scalar: Double): FlinkDrm[K] = {
    val function = EWOpsCloning.strToFunction(op.op)

    val res = A.blockify.ds.map(new MapFunction[(Array[K], Matrix), (Array[K], Matrix)] {
      def map(tuple: (Array[K], Matrix)): (Array[K], Matrix) = tuple match {
        case (keys, mat) => (keys, function(mat, scalar))
      }
    })

    new BlockifiedFlinkDrm(res, op.ncol)
  }

}

object EWOpsCloning {

  type MatrixScalarFunc = (Matrix, Double) => Matrix

  def strToFunction(op: String): MatrixScalarFunc = op match {
    case "+" => plusScalar
    case "-" => minusScalar
    case "*" => timesScalar
    case "/" => divScalar
    case "-:" => scalarMinus
    case "/:" => scalarDiv
    case _ => throw new IllegalArgumentException(s"Unsupported elementwise operator: $op")
  }

  val plusScalar: MatrixScalarFunc = (A, s) => A + s
  val minusScalar: MatrixScalarFunc = (A, s) => A - s
  val scalarMinus: MatrixScalarFunc = (A, s) => s -: A
  val timesScalar: MatrixScalarFunc = (A, s) => A * s
  val divScalar: MatrixScalarFunc = (A, s) => A / s
  val scalarDiv: MatrixScalarFunc = (A, s) => s /: A
}

