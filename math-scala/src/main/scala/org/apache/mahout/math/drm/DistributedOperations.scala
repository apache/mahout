package org.apache.mahout.math.drm

import scala.reflect.ClassTag
import org.apache.mahout.math.Vector

/**
 * @author gokhan
 * Engine specific additional distributed operations like aggregating blocks
 */
trait DistributedOperations {
  def aggregateBlocks[U: ClassTag, K: ClassTag] (drm: DrmLike[K])
                                                (zeroValue: U, seqOp: (U, BlockifiedDrmTuple[K]) => U,
                                                 combOp: (U, U) => U ): U
  def aggregateRows[U: ClassTag, K: ClassTag] (drm: DrmLike[K])
                                              (zeroValue: U, seqOp: (U, (K, Vector)) => U,
                                               combOp: (U, U) => U ): U


}
