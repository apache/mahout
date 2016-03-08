package org.apache.mahout.sparkbindings.blas

import org.apache.mahout.math._
import drm._
import scalabindings._
import RLikeOps._
import org.apache.mahout.sparkbindings._
import org.apache.mahout.sparkbindings.drm._
import org.apache.mahout.logging._
import scala.reflect.ClassTag
import org.apache.mahout.math.DiagonalMatrix
import org.apache.mahout.math.drm.logical.OpTimesRightMatrix


/** Matrix product with one of operands an in-core matrix */
object AinCoreB {

  private final implicit val log = getLog(AinCoreB.getClass)

  def rightMultiply[K: ClassTag](op: OpTimesRightMatrix[K], srcA: DrmRddInput[K]): DrmRddInput[K] = {
    if ( op.right.isInstanceOf[DiagonalMatrix])
      rightMultiply_diag(op, srcA)
    else
      rightMultiply_common(op,srcA)
  }

  private def rightMultiply_diag[K: ClassTag](op: OpTimesRightMatrix[K], srcA: DrmRddInput[K]): DrmRddInput[K] = {
    val rddA = srcA.toBlockifiedDrmRdd(op.A.ncol)
    implicit val ctx:DistributedContext = rddA.context
    val dg = drmBroadcast(op.right.viewDiagonal())

    debug(s"operator A %*% inCoreB-diagonal. #parts=${rddA.partitions.size}.")

    val rdd = rddA
        // Just multiply the blocks
        .map {
      case (keys, blockA) => keys -> (blockA %*%: diagv(dg))
    }
    rdd
  }

  private def rightMultiply_common[K: ClassTag](op: OpTimesRightMatrix[K], srcA: DrmRddInput[K]): DrmRddInput[K] = {

    val rddA = srcA.toBlockifiedDrmRdd(op.A.ncol)
    implicit val sc:DistributedContext = rddA.sparkContext

    debug(s"operator A %*% inCoreB. #parts=${rddA.partitions.size}.")

    val bcastB = drmBroadcast(m = op.right)

    val rdd = rddA
        // Just multiply the blocks
        .map {
      case (keys, blockA) => keys -> (blockA %*% bcastB)
    }

    rdd
  }

}
