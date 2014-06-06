package org.apache.mahout.sparkbindings.blas

import scala.reflect.ClassTag
import org.apache.mahout.sparkbindings.drm.DrmRddInput
import org.apache.mahout.math.drm.logical.OpPar
import org.apache.spark.rdd.RDD

/** Physical adjustment of parallelism */
object Par {

  def exec[K: ClassTag](op: OpPar[K], src: DrmRddInput[K]): DrmRddInput[K] = {

    def adjust[T](rdd: RDD[T]): RDD[T] =
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
      } else if (op.exactSplits == -1 && op.minSplits == -1) {

        // auto adjustment, try to scale up to either x1Size or x2Size.
        val clusterSize = rdd.context.getConf.get("spark.default.parallelism", "1").toInt

        val x1Size = (clusterSize * .95).ceil.toInt
        val x2Size = (clusterSize * 1.9).ceil.toInt

        if (rdd.partitions.size <= x1Size)
          rdd.coalesce(numPartitions = x1Size, shuffle = true)
        else if (rdd.partitions.size <= x2Size)
          rdd.coalesce(numPartitions = x2Size, shuffle = true)
        else
          rdd.coalesce(numPartitions = rdd.partitions.size)
      } else rdd.coalesce(rdd.partitions.size)

    if (src.isBlockified) {
      val rdd = src.toBlockifiedDrmRdd()
      new DrmRddInput[K](blockifiedSrc = Some(adjust(rdd)))
    } else {
      val rdd = src.toDrmRdd()
      new DrmRddInput[K](rowWiseSrc = Some(op.ncol -> adjust(rdd)))
    }
  }

}
