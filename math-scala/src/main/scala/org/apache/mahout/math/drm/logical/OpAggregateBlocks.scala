package org.apache.mahout.math.drm.logical

import org.apache.mahout.math.Matrix
import org.apache.mahout.math.drm.{CheckpointedDrm, DrmLike}

import scala.reflect.ClassTag

/**
 * @author gokhan
 */
class OpAggregateBlocks[U: ClassTag, K: ClassTag](
    override var A: DrmLike[K],
    val zeroValue: U,
    val seqOp: (U, (Array[K], _<: Matrix )) => U,
    val combOp: (U, U) => U
    ) extends AggregateAction[U, K]
