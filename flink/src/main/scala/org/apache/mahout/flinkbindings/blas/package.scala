package org.apache.mahout.flinkbindings

/**
  * Created by andy on 3/27/16.
  */
package object blas {




  /**
    * Estimate number of partitions for the product of A %*% B.
    *
    * We take average per-partition element count of product as higher of the same of A and B. (prefer
    * larger partitions of operands).
    *
    * @param anrow A.nrow
    * @param ancol A.ncol
    * @param bncol B.ncol
    * @param aparts partitions in A
    * @param bparts partitions in B
    * @return recommended partitions
    */
  private[blas] def estimateProductPartitions(anrow:Long, ancol:Long, bncol:Long, aparts:Int, bparts:Int):Int = {

    // Compute per-partition element density in A
    val eppA = anrow.toDouble * ancol/ aparts

    // Compute per-partition element density in B
    val eppB = ancol.toDouble * bncol / bparts

    // Take the maximum element density into account. Is it a good enough?
    val epp = eppA max eppB

    // product partitions
    val prodParts = anrow * bncol / epp

    val nparts = math.round(prodParts).toInt max 1

    // Constrain nparts to maximum of anrow to prevent guaranteed empty partitions.
    if (nparts > anrow) anrow.toInt else nparts
  }
}
