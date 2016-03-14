package org.apache.mahout.math.decompositions

import org.apache.mahout.math.{Matrix, Matrices, Vector}
import org.apache.mahout.math.scalabindings._
import RLikeOps._
import org.apache.mahout.math.drm._
import RLikeDrmOps._
import org.apache.mahout.common.RandomUtils
import org.apache.mahout.logging._

object DSSVD {

  private final implicit val log = getLog(DSSVD.getClass)

  /**
   * Distributed Stochastic Singular Value decomposition algorithm.
   *
   * @param drmA input matrix A
   * @param k request SSVD rank
   * @param p oversampling parameter
   * @param q number of power iterations
   * @return (U,V,s). Note that U, V are non-checkpointed matrices (i.e. one needs to actually use them
   *         e.g. save them to hdfs in order to trigger their computation.
   */
  def dssvd[K](drmA: DrmLike[K], k: Int, p: Int = 15, q: Int = 0):
  (DrmLike[K], DrmLike[Int], Vector) = {

    // Some mapBlock() calls need it
    implicit val ktag =  drmA.keyClassTag

    val drmAcp = drmA.checkpoint()

    val m = drmAcp.nrow
    val n = drmAcp.ncol
    assert(k <= (m min n), "k cannot be greater than smaller of m, n.")
    val pfxed = safeToNonNegInt((m min n) - k min p)

    // Actual decomposition rank
    val r = k + pfxed

    // We represent Omega by its seed.
    val omegaSeed = RandomUtils.getRandom().nextInt()

    // Compute Y = A*Omega. Instead of redistributing view, we redistribute the Omega seed only and
    // instantiate the Omega random matrix view in the backend instead. That way serialized closure
    // is much more compact.
    var drmY = drmAcp.mapBlock(ncol = r) {
      case (keys, blockA) ⇒
        val blockY = blockA %*% Matrices.symmetricUniformView(n, r, omegaSeed)
        keys → blockY
    }.checkpoint()

    var drmQ = dqrThin(drmY)._1
    // Checkpoint Q if last iteration
    if (q == 0) drmQ = drmQ.checkpoint()

    trace(s"dssvd:drmQ=${drmQ.collect}.")

    // This actually should be optimized as identically partitioned map-side A'B since A and Q should
    // still be identically partitioned.
    var drmBt = drmAcp.t %*% drmQ
    // Checkpoint B' if last iteration
    if (q == 0) drmBt = drmBt.checkpoint()

    trace(s"dssvd:drmB'=${drmBt.collect}.")

    for (i ← 0  until q) {
      drmY = drmAcp %*% drmBt
      drmQ = dqrThin(drmY.checkpoint())._1
      // Checkpoint Q if last iteration
      if (i == q - 1) drmQ = drmQ.checkpoint()

      // This on the other hand should be inner-join-and-map A'B optimization since A and Q_i are not
      // identically partitioned anymore.`
      drmBt = drmAcp.t %*% drmQ
      // Checkpoint B' if last iteration
      if (i == q - 1) drmBt = drmBt.checkpoint()
    }

    val mxBBt:Matrix = drmBt.t %*% drmBt

    trace(s"dssvd: BB'=$mxBBt.")

    val (inCoreUHat, d) = eigen(mxBBt)
    val s = d.sqrt

    // Since neither drmU nor drmV are actually computed until actually used, we don't need the flags
    // instructing compute (or not compute) either of the U,V outputs anymore. Neat, isn't it?
    val drmU = drmQ %*% inCoreUHat
    val drmV = drmBt %*% (inCoreUHat %*%: diagv(1 /: s))

    (drmU(::, 0 until k), drmV(::, 0 until k), s(0 until k))
  }

}
