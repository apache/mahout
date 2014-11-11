package org.apache.mahout.sparkbindings.blas

import org.apache.mahout.math.Matrix
import org.apache.mahout.math.drm.CheckpointedDrm
import org.apache.mahout.sparkbindings.drm.{DrmRddInput, CheckpointedDrmSpark}

import scala.reflect.ClassTag

/**
 * @author gokhan
 */
object AggregateBlocks {
  def exec[U:ClassTag, K:ClassTag](src: DrmRddInput[K], zeroValue: U, seqOp: (U, (Array[K], _<:Matrix )) => U,
                          combOp:(U, U) => U):U = {
    val rdd = src.toBlockifiedDrmRdd()
    val out = rdd.aggregate(zeroValue)(seqOp, combOp)

    out
  }
}
