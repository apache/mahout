package org.apache.mahout.math.drm

import org.apache.mahout.math.Vector

import scala.reflect.ClassTag

/**
 * @author gokhan
 */
class DistributedOps[K: ClassTag] (protected[drm] val drm: DrmLike[K]){
  def accumulateBlocks[U: ClassTag](zeroValue: U, seqOp: (U, BlockifiedDrmTuple[K]) => U, combOp: (U, U) => U ): U =
    drm.context.operations.aggregateBlocks(drm)(zeroValue, seqOp, combOp)
  def accumulateRows[U: ClassTag](zeroValue: U, seqOp: (U, (K, Vector)) => U, combOp: (U, U) => U ): U =
    drm.context.operations.aggregateRows(drm)(zeroValue, seqOp, combOp)
}
