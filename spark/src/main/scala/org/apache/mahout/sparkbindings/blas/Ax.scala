package org.apache.mahout.sparkbindings.blas

import org.apache.mahout.math.scalabindings._
import RLikeOps._

import org.apache.mahout.sparkbindings.drm._
import org.apache.mahout.sparkbindings.drm.plan.{OpAtx, OpAx, OpTimesRightMatrix}
import org.apache.mahout.sparkbindings.drm.DrmRddInput
import scala.reflect.ClassTag


/** Matrix product with one of operands an in-core matrix */
object Ax {

  def ax_with_broadcast[K: ClassTag](op: OpAx[K], srcA: DrmRddInput[K]): DrmRddInput[K] = {

    val rddA = srcA.toBlockifiedDrmRdd()
    implicit val sc = rddA.sparkContext

    val bcastX = drmBroadcast(x = op.x)

    val rdd = rddA
        // Just multiply the blocks
        .map({
      case (keys, blockA) => keys -> (blockA %*% bcastX).toColMatrix
    })

    new DrmRddInput(blockifiedSrc = Some(rdd))
  }

  def atx_with_broadcast(op: OpAtx, srcA: DrmRddInput[Int]): DrmRddInput[Int] = {
    val rddA = srcA.toBlockifiedDrmRdd()
    implicit val sc = rddA.sparkContext

    val bcastX = drmBroadcast(x = op.x)

    val inCoreM = rddA
        // Just multiply the blocks
        .map {
      case (keys, blockA) =>
        keys.zipWithIndex.map {
          case (key, idx) => blockA(idx, ::) * bcastX.value(key)
        }
            .reduce(_ += _)
    }
        // All-reduce
        .reduce(_ += _)
        // Convert back to mtx
        .toColMatrix

    // It is ridiculous, but in this scheme we will have to re-parallelize it again in order to plug
    // it back as drm blockified rdd

    val rdd = sc.parallelize(Seq(inCoreM), numSlices = 1)
        .map(block => Array.tabulate(block.nrow)(i => i) -> block)

    new DrmRddInput(blockifiedSrc = Some(rdd))

  }

}
