package org.apache.mahout.sparkbindings.blas

import scala.reflect.ClassTag
import org.apache.mahout.sparkbindings.drm.DrmRddInput
import org.apache.mahout.math.drm.logical.OpPar
import org.apache.spark.rdd.RDD

/** Physical adjustment of parallelism */
object Par {

  def exec[K:ClassTag](op:OpPar[K], src: DrmRddInput[K] ): DrmRddInput[K] = {

    def adjust[T](rdd: RDD[T]):RDD[T] =
      if (op.minSplits > 0) {
        if (rdd.partitions.size < op.minSplits)
          rdd.coalesce(op.minSplits, shuffle = true)
        else rdd.coalesce(rdd.partitions.size)
      } else if (op.exactSplits > 0) {
        if (op.exactSplits < rdd.partitions.size)
          rdd.coalesce(numPartitions = op.exactSplits, shuffle = false)
        else if (op.exactSplits > rdd.partitions.size)
          rdd.coalesce(numPartitions = op.exactSplits, shuffle = true)
        else
          rdd.coalesce(rdd.partitions.size)
      }
    else rdd.coalesce(rdd.partitions.size)

    if (src.isBlockified) {
      val rdd = src.toBlockifiedDrmRdd()
      new DrmRddInput[K](blockifiedSrc = Some(adjust(rdd)))
    } else {
      val rdd = src.toDrmRdd()
      new DrmRddInput[K](rowWiseSrc = Some(op.ncol -> adjust(rdd)))
    }
  }

}
