package org.apache.mahout.sparkbindings.blas

import org.apache.mahout.math.scalabindings._
import RLikeOps._

import org.apache.mahout.sparkbindings.drm.plan.OpTimesRightMatrix
import org.apache.mahout.sparkbindings.drm.DrmRddInput
import org.apache.hadoop.io.Writable
import scala.reflect.ClassTag

/** Matrix product with one of operands an in-core matrix */
object AinCoreB {

  def rightMultiply[K: ClassTag](op: OpTimesRightMatrix[K], srcA: DrmRddInput[K]): DrmRddInput[K] = {

    val rddA = srcA.toBlockifiedDrmRdd()
    val sc = rddA.sparkContext

    // This will be correctly supported for kryo, right?
    val bcastB = sc.broadcast(op.right)

    val rdd = rddA
        // Just multiply the blocks
        .map({
      case (keys, blockA) => keys -> (blockA %*% bcastB.value)
    })

    new DrmRddInput(blockifiedSrc = Some(rdd))
  }

}
