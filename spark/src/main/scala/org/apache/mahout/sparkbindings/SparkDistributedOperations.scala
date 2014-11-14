package org.apache.mahout.sparkbindings

import org.apache.mahout.math.drm.{DrmLike, BlockifiedDrmTuple, CheckpointedDrm, DistributedOperations}
import org.apache.mahout.sparkbindings.drm.DrmRddInput
import org.apache.mahout.math.Vector

import scala.reflect.ClassTag

/**
 * @author gokhan
 */
object SparkDistributedOperations extends DistributedOperations{
  private def toDrmRddInput[K: ClassTag] (drm: DrmLike[K]): DrmRddInput[K] = {
    val cp = drm match {
      case a: CheckpointedDrm[K] => a
      case _ => drm.checkpoint()
    }
    new DrmRddInput[K](rowWiseSrc = Some((cp.ncol, cp.rdd)))
  }



  override def aggregateBlocks[U: ClassTag, K: ClassTag](drm: DrmLike[K])
                                                        (zeroValue: U, seqOp: (U, BlockifiedDrmTuple[K]) => U,
                                                         combOp: (U, U) => U): U = {
    val out = toDrmRddInput(drm).toBlockifiedDrmRdd().aggregate(zeroValue)(seqOp, combOp)
    out

  }

  override def aggregateRows[U: ClassTag, K: ClassTag](drm: DrmLike[K])
                                                      (zeroValue: U, seqOp: (U, (K, Vector)) => U,
                                                       combOp: (U, U) => U): U = {
    val out = toDrmRddInput(drm).toDrmRdd().aggregate(zeroValue)(seqOp, combOp)
    out
  }

}
