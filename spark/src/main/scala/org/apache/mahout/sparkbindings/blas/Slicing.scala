package org.apache.mahout.sparkbindings.blas

import org.apache.mahout.sparkbindings.drm.DrmRddInput
import org.apache.mahout.math.drm.logical.OpRowRange

object Slicing {

  def rowRange(op: OpRowRange, srcA: DrmRddInput[Int]): DrmRddInput[Int] = {
    val rowRange = op.rowRange
    val ncol = op.ncol
    val rdd = srcA.asRowWise()

        // Filter the rows in the range only
        .filter({
      case (key, vector) => rowRange.contains(key)
    })

        // Now we need to adjust the row index
        .map({
      case (key, vector) => (key - rowRange.head) -> vector
    })

    // TODO: we probably need to re-shuffle result or at least cut down the partitions of 0 size

    rdd
  }
}
