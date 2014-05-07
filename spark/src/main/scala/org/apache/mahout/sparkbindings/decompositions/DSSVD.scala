package org.apache.mahout.sparkbindings.drm.decompositions

import scala.reflect.ClassTag
import org.apache.mahout.math.{Matrices, Vector}
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.sparkbindings.drm._
import RLikeDrmOps._
import org.apache.mahout.common.RandomUtils

object DSSVD {

  /**
   * Distributed Stochastic Singular Value decomposition algorithm.
   *
   * @param A input matrix A
   * @param k request SSVD rank
   * @param p oversampling parameter
   * @param q number of power iterations
   * @return (U,V,s). Note that U, V are non-checkpointed matrices (i.e. one needs to actually use them
   *         e.g. save them to hdfs in order to trigger their computation.
   */
  def dssvd[K: ClassTag](A: DrmLike[K], k: Int, p: Int = 15, q: Int = 0):
  (DrmLike[K], DrmLike[Int], Vector) = {

    val drmA = A.checkpoint()

    val m = drmA.nrow
    val n = drmA.ncol
    assert(k <= (m min n), "k cannot be greater than smaller of m, n.")
    val pfxed = safeToNonNegInt((m min n) - k min p)

    // Actual decomposition rank
    val r = k + pfxed

    // We represent Omega by its seed.
    val omegaSeed = RandomUtils.getRandom().nextInt()

    // Compute Y = A*Omega. Instead of redistributing view, we redistribute the Omega seed only and
    // instantiate the Omega random matrix view in the backend instead. That way serialized closure
    // is much more compact.
    var drmY = drmA.mapBlock(ncol = r) {
      case (keys, blockA) =>
        val blockY = blockA %*% Matrices.symmetricUniformView(n, r, omegaSeed)
        keys -> blockY
    }

    var drmQ = dqrThin(drmY.checkpoint())._1
    // Checkpoint Q if last iteration
    if (q == 0) drmQ = drmQ.checkpoint()

    // This actually should be optimized as identically partitioned map-side A'B since A and Q should
    // still be identically partitioned.
    var drmBt = drmA.t %*% drmQ
    // Checkpoint B' if last iteration
    if (q == 0) drmBt = drmBt.checkpoint()

    for (i <- 0  until q) {
      drmY = drmA %*% drmBt
      drmQ = dqrThin(drmY.checkpoint())._1
      // Checkpoint Q if last iteration
      if (i == q - 1) drmQ = drmQ.checkpoint()

      // This on the other hand should be inner-join-and-map A'B optimization since A and Q_i are not
      // identically partitioned anymore.
      drmBt = drmA.t %*% drmQ
      // Checkpoint B' if last iteration
      if (i == q - 1) drmBt = drmBt.checkpoint()
    }

    val inCoreBBt = (drmBt.t %*% drmBt).checkpoint(CacheHint.NONE).collect
    val (inCoreUHat, d) = eigen(inCoreBBt)
    val s = d.sqrt

    // Since neither drmU nor drmV are actually computed until actually used, we don't need the flags
    // instructing compute (or not compute) either of the U,V outputs anymore. Neat, isn't it?
    val drmU = drmQ %*% inCoreUHat
    val drmV = drmBt %*% (inCoreUHat %*%: diagv(1 /: s))

    (drmU(::, 0 until k), drmV(::, 0 until k), s(0 until k))
  }

}
